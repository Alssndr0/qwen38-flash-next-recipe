# Qwen3.8-Flash-Next with the DFlash2 drafter (FP4-Local, myllmbox v5.3)

The reference serve of this repo since 2026-10-22: NVIDIA's
[Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) on myllmbox's v5.3 kit
(vLLM 0.32), drafting with [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
(a block-diffusion drafter: one pass proposes 7 tokens) instead of the model's own MTP head. Outputs are the
target's: every drafted token is verified.

## Run it

On a DGX Spark with Docker and the NVIDIA container runtime (run the parent folder's `./tune-host.sh` once per box):

```bash
git clone https://github.com/Alssndr0/qwen38-flash-next-recipe.git
cd qwen38-flash-next-recipe/dflash2
./run.sh
```

The first run builds the image `qwen38-flash-next-vllm:v5.3-dflash2` from [image/](image/) (it pulls
`myllmbox/qwen38-flash-next-vllm:v5.3`, about 24 GB, then takes about a minute), downloads FP4-Local at the measured
revision (about 102 GB) and the drafter (1.2 GB, sha256-checked), prepares the n-gram table map (about 15 minutes,
once), and serves an OpenAI-compatible API on `http://127.0.0.1:8000/v1` as `Qwen/Qwen3.8-Flash-Next`. Later boots
take about 4 minutes. `./stop.sh` stops it, `./view.sh` shows live stats, `DRY_RUN=1 ./run.sh` prints the
`docker run` command and changes nothing. Everything is configured in [recipe.yaml](recipe.yaml) (changed lines are
marked `[dflash2]`).

## Results (one DGX Spark, one stream)

177 recorded OpenCode / Pi agentic requests, replayed with their own sampling (temperature 1.0, top-k 20, top-p
0.95) and paired by request with the same checkpoint on the same engine drafting with its MTP head (our tuned v5.3:
`mtp_depth.min` 3, draft truncation, NVFP4 shortlist scorer, GEMM plan fill):

| | MTP (tuned) | DFlash2 (this kit) | change |
|---|---|---|---|
| decode tok/s | 56.12 | **59.48** | **+5.98 %** (95 % CI +3.34 to +8.68) |
| tokens per verification step | 3.09 | 3.75 | +21.3 % |
| ms per step | 54.9 | 62.9 | |
| OpenCode / Pi requests | | | +6.7 % / +4.7 % |
| tool-call arguments valid | 179 / 179 | 188 / 188 | |

For context: myllmbox's v5.3 as shipped does 51.16 tok/s on the same replay (this kit: +16 %).

Acceptance by draft position: MTP 0.81 / 0.62 / 0.48 / 0.14 / 0.04, DFlash2 0.78 / 0.59 / 0.44 / 0.33 / 0.26 / 0.20 /
0.16. The MTP head is slightly better on the first three tokens; DFlash2 wins because all seven drafts come from one
2.2 ms pass, while each MTP draft is another sequential pass.

Quality, measured to check that nothing changed: sampled outputs have the same distribution as the MTP serve's
(chi-square per position, a positive control detected); teacher-forced logits sit at the serve's own run-to-run
noise (KL 0.033, top-1 agreement 94 %); 400 exact-answer problems, 4 runs: 351.25 vs 349.5 for FP4-Local on its own;
no CJK characters in 320 English replies (2 in 40 Italian ones vs 1 for FP4-Local).

## What is in the image

| file | what it does |
|---|---|
| `patches/qwen4exp-dflash2-capture.patch` | DFlash2 reads the completed outputs of target layers 5 / 15 / 25 / 35 / 45, averaged over the four hyper-connection streams. Flash-Next's vLLM model defers each residual combine into the next layer's fused kernel, so these states are computed as a side output; the target's own computation is unchanged. |
| `patches/qwen4exp-allow-dflash.patch` | Qwen4Exp's config check accepted only MTP and n-gram drafters. |
| `patches/dflash2-speculator.patch` | hooks for `dfl2_opt.py`; `DFL2_TOPK` (candidates per draft position); an optional live file for draft top-p / temperature (off: both hurt acceptance). |
| `patches/gb10-gemm-plan-fill.patch` | GB10 skinny-GEMM plans for every verification width (`MBX_GEMM_FILL=1`). |
| `files/dfl2_opt.py` | the drafter's linears in NVFP4 (Marlin W4A16), and the candidates taken from a 2,048-token shortlist (rank-256 scores in NVFP4, exact logits for the shortlist) instead of the full 1.27 GB bf16 vocabulary head. The draft pass goes from ~9 ms to 2.2 ms per step. |
| `files/dfl2_basis_r256.pt` | the shortlist basis: the top 256 principal directions of the drafter's own hidden states (48 benchmark prompts, disjoint from the replay). The MTP head's basis missed the drafter's top choice on up to 10 % of draft steps. |

Two settings matter outside the image: the drafter runs on Triton attention (with FlashInfer its sliding-window KV
takes 64-token blocks, and this KV pool charges every block its largest page, ~23 GB per request), and all 7 drafts
are verified (`mtp_depth.min: 7`).

Tried and not kept: draft top-p truncation, draft temperature 0.85 / 1.2, 64 candidates, a bf16 KV cache for the
drafter, adaptive depth policies (vLLM's adaptive verification does not start on this model's GDN backend), and a
4-bit target vocabulary head (3.8 ms per step faster but lower acceptance, net +1.2 %, and it moves the model's
outputs: KL 0.043).

Production on this box remains hibrid48 on the parent folder's v5.2 kit (68.13 tok/s on the same replay, a different
checkpoint); this kit is the reference for FP4-Local.
