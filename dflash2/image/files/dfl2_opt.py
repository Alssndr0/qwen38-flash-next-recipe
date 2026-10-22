"""dfl2_opt: draft-side speedups for the DFlash2 drafter on Qwen3.8-Flash-Next (vLLM 0.32, myllmbox v5.3).

Hooked from DFlash2Speculator.load_draft_model (image/patches/dflash2-speculator.patch), after vLLM shares the target's
embedding and LM head into the drafter. Exact for the served text: only the draft proposals change, the target
verifies every one of them (rejection sampling keeps the target's distribution).

Switches (default off):
  DFL2_DRAFT_FP8=1     the drafter's bf16 linears (5 layers, fc, conv kernel projections, selector projection) -> FP8
                       e4m3 per output channel on Marlin W8A16 (mbx_opt_quant.convert, the same conversion as v5.3's
                       MBX_DRAFT_FP8 for the MTP drafter; a layer whose output error on a random input is >= 5 % stays bf16).
  DFL2_HEAD2=<basis>   candidate top-k from a shortlist instead of the full 248,320 x 2,560 bf16 head (1.27 GB read per
                       draft pass): z = h P, approximate scores U z with U = W P on Marlin FP4 (mbx_draft_head2.DraftHead2,
                       MBX_DRAFT_HEAD2_SCORER=fp4), the top DFL2_HEAD2_K (2048) kept, their exact logits from the head's
                       own rows, then the selector's top-k (16) of those. A token outside the shortlist cannot be a
                       candidate; the unary logits of the candidates are exact.
"""
import os

import torch

from vllm.logger import init_logger

log = init_logger("vllm.dfl2_opt")


class ShortlistCandidates:
    """compute_candidates(hidden [n, H]) -> (ids int64 [n, top_k], logits fp32 [n, top_k]) via a two-stage head."""

    def __init__(self, draft_model, vllm_config, basis: str, k_short: int):
        import mbx_draft_head2 as dh2

        lp = draft_model.candidate_logits_processor
        if getattr(lp, "soft_cap", None) is not None:
            raise ValueError("DFL2_HEAD2: logit soft cap not supported")
        vocab = getattr(lp, "org_vocab_size", None) or lp.vocab_size
        self.scale = float(getattr(lp, "scale", 1.0))
        self.dh = dh2.DraftHead2(draft_model.lm_head, vocab, basis, k_short, str(vllm_config.model_config.model))
        assert self.dh.fmt == "dense", "DFL2_HEAD2 expects the dense bf16 head of FP4-Local"
        self.top_k = int(draft_model.model.candidate_selector.top_k)
        self._kernel = dh2._cand_logits_dense_kernel
        self._triton = dh2.triton
        self._dump_init()

    def _dump_init(self):
        """DFL2_HEAD2_DUMP=<path>: keep the drafter's candidate hidden states (the head's input) in a GPU ring buffer
        written inside the CUDA graph; a watcher thread saves the filled rows to <path> (for fit_draft_basis.py)."""
        self.dump_path = os.environ.get("DFL2_HEAD2_DUMP", "")
        if not self.dump_path:
            return
        import threading
        import time
        dev = self.dh.device
        self.dump_cap = int(os.environ.get("DFL2_HEAD2_DUMP_ROWS", "32768"))
        self.dump_buf = torch.zeros(self.dump_cap, self.dh.H, dtype=torch.bfloat16, device=dev)
        self.dump_count = torch.zeros(1, dtype=torch.int64, device=dev)
        self.dump_ar = torch.arange(4096, dtype=torch.int64, device=dev)

        def _save():
            torch.cuda.set_device(dev)
            side = torch.cuda.Stream(dev)
            saved = 0
            while True:
                time.sleep(20)
                with torch.cuda.stream(side):
                    n = int(self.dump_count.item())
                if n > saved + 1024 and n > 4096:   # skip start-up / warm-up rows in the first 4096
                    with torch.cuda.stream(side):
                        rows = self.dump_buf[: min(n, self.dump_cap)].to("cpu")
                    side.synchronize()
                    keep = rows[rows.abs().sum(dim=1) > 0]
                    torch.save(keep, self.dump_path)
                    saved = n
                    log.info("dfl2_opt: dumped %d candidate hidden states to %s", keep.shape[0], self.dump_path)

        threading.Thread(target=_save, daemon=True, name="dfl2-dump").start()

    def __call__(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dh = self.dh
        n = hidden_states.shape[0]
        hc = hidden_states.to(torch.bfloat16).contiguous()
        if self.dump_path and n <= self.dump_ar.shape[0]:
            idx = (self.dump_count + self.dump_ar[:n]) % self.dump_cap
            self.dump_buf.index_copy_(0, idx, hc)
            self.dump_count.add_(n)
        z = hc @ dh.P
        sc = dh.scorer
        if dh.fp4:
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import apply_fp4_marlin_linear
            scores = apply_fp4_marlin_linear(z, sc.weight, sc.weight_scale, sc.weight_global_scale, None, dh.V, dh.r)
        else:
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_fp8_marlin_linear
            scores = apply_fp8_marlin_linear(z, sc.weight, sc.weight_scale, None, dh.V, dh.r, None)
        cand = scores.topk(dh.k, dim=-1, sorted=False).indices.to(torch.int32).contiguous()      # [n, K]
        cl = torch.empty(n, dh.k, dtype=torch.float32, device=hidden_states.device)
        if n > 0:
            BLOCK_C = 32
            self._kernel[(n, self._triton.cdiv(dh.k, BLOCK_C))](
                hc, cand, dh.W, cl, dh.k, hc.stride(0), cand.stride(0), dh.W.stride(0), cl.stride(0),
                H=dh.H, BLOCK_C=BLOCK_C, BLOCK_H=128)
        vals, idx = cl.topk(self.top_k, dim=-1)                                                     # sorted, desc
        ids = cand.gather(1, idx).to(torch.int64)
        if self.scale != 1.0:
            vals = vals * self.scale
        return ids, vals


class _Fp4MarlinMethod:
    """Stand-in quant method: weight already packed for Marlin NVFP4 (W4A16); no-op after loading."""

    def __init__(self, n, k):
        self.n, self.k = n, k

    def process_weights_after_loading(self, layer):
        return

    def apply(self, layer, x, bias=None):
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import apply_fp4_marlin_linear
        return apply_fp4_marlin_linear(x, layer.weight, layer.weight_scale, layer.weight_global_scale, None,
                                       self.n, self.k, bias)


def convert_fp4(model, tag=" (dflash2 drafter)", max_err=0.25) -> None:
    """DFL2_DRAFT_FP4=1: the drafter's bf16 linears to NVFP4 (e2m1, one e4m3 scale per 16 inputs, a global scale) on
    Marlin W4A16; the same layer selection as mbx_opt_quant's FP8 path. A layer whose output error on a random input
    is >= max_err stays bf16."""
    import mbx_draft_head2 as dh2
    import mbx_opt_quant as oq
    from torch import nn
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_fp4_layer_for_marlin
    done, saved, errs = 0, 0, []
    for name, mod in oq._selected(model, None, oq._EXCLUDE, oq._EXCLUDE_SUFFIX, 1 << 20):
        w = mod.weight.data
        N, K = w.shape
        if K % 16:
            continue
        pk, sc, g = dh2._quant_nvfp4(w.float())
        holder = nn.Module()
        holder.weight = nn.Parameter(pk, requires_grad=False)
        holder.weight_scale = nn.Parameter(sc, requires_grad=False)
        holder.weight_global_scale = nn.Parameter(torch.tensor(g, dtype=torch.float32, device=w.device),
                                                  requires_grad=False)
        holder.params_dtype = torch.bfloat16
        holder.output_size_per_partition, holder.input_size_per_partition = N, K
        meth = _Fp4MarlinMethod(N, K)
        try:
            prepare_fp4_layer_for_marlin(holder)
            x = torch.randn(4, K, device=w.device, dtype=torch.bfloat16)
            ref = torch.nn.functional.linear(x, w).float()
            err = ((meth.apply(holder, x).float() - ref).norm() / ref.norm().clamp(min=1e-12)).item()
        except Exception as e:  # noqa: BLE001
            log.warning("dfl2 fp4%s: %s %s not converted (%r)", tag, name, (N, K), e)
            continue
        errs.append(err)
        if not err < max_err:
            log.warning("dfl2 fp4%s: %s %s rel.err %.4f, kept bf16", tag, name, (N, K), err)
            continue
        mod.weight = holder.weight
        mod.weight_scale = holder.weight_scale
        mod.weight_global_scale = holder.weight_global_scale
        mod.quant_method = meth
        done += 1
        saved += N * K * 2 - pk.numel() - sc.numel()
    torch.cuda.empty_cache()
    log.info("dfl2 fp4%s: %d linears converted, %.1f MB saved, output rel.err max %.4f mean %.4f", tag, done,
             saved / 1e6, max(errs) if errs else 0.0, sum(errs) / len(errs) if errs else 0.0)


def target_head_fp4(lm_head) -> None:
    """DFL2_TARGET_HEAD_FP4=1 (CLASS C, changes the served model's outputs): the target's bf16 vocabulary head (shared
    into the drafter) to NVFP4 on Marlin W4A16, 1.27 GB -> 0.36 GB read per verify step. The shortlist candidates keep
    reading the bf16 rows they captured at install time."""
    from torch import nn
    import mbx_draft_head2 as dh2
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_fp4_layer_for_marlin
    w = lm_head.weight.data
    N, K = w.shape
    pk, sc, g = dh2._quant_nvfp4(w.float())
    holder = nn.Module()
    holder.weight = nn.Parameter(pk, requires_grad=False)
    holder.weight_scale = nn.Parameter(sc, requires_grad=False)
    holder.weight_global_scale = nn.Parameter(torch.tensor(g, dtype=torch.float32, device=w.device), requires_grad=False)
    holder.params_dtype = torch.bfloat16
    holder.output_size_per_partition, holder.input_size_per_partition = N, K
    prepare_fp4_layer_for_marlin(holder)
    meth = _Fp4MarlinMethod(N, K)
    x = torch.randn(4, K, device=w.device, dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x, w).float()
    err = ((meth.apply(holder, x).float() - ref).norm() / ref.norm().clamp(min=1e-12)).item()
    lm_head.weight = holder.weight
    lm_head.weight_scale = holder.weight_scale
    lm_head.weight_global_scale = holder.weight_global_scale
    lm_head.quant_method = meth
    torch.cuda.empty_cache()
    log.info("dfl2 fp4 TARGET head (class C): %s, output rel.err %.4f", (N, K), err)


def draft_loaded(draft_model, vllm_config) -> None:
    if os.environ.get("DFL2_DRAFT_FP4", "0") == "1":
        convert_fp4(draft_model)
    elif os.environ.get("DFL2_DRAFT_FP8", "0") == "1":
        import mbx_opt_quant
        mbx_opt_quant.convert(draft_model, tag=" (dflash2 drafter)")
    basis = os.environ.get("DFL2_HEAD2")
    if basis:
        k = int(os.environ.get("DFL2_HEAD2_K", "2048"))
        draft_model.compute_candidates = ShortlistCandidates(draft_model, vllm_config, basis, k)
        log.info("dfl2_opt: shortlist candidates (basis %s, %d per row, top-%d)", basis, k,
                 draft_model.compute_candidates.top_k)
    if os.environ.get("DFL2_TARGET_HEAD_FP4", "0") == "1":
        target_head_fp4(draft_model.lm_head)
