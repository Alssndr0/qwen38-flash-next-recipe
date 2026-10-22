# Qwen3.8-Flash-Next on one DGX Spark, with faster decoding

This is [myllmbox/qwen38-flash-next-recipe](https://github.com/myllmbox/qwen38-flash-next-recipe) (v5.2), the
one-box serving kit by [@myllmbox](https://github.com/myllmbox) and [@bilikas](https://github.com/bilikas), with a
set of decode speedups added on top. Clone it and run it: everything it needs is in this repo or downloaded by
`run.sh`.

- **Faster, same outputs:** +9 to +24 % tok/s at 1 stream and +2 to +11 % at 16 streams on hibrid48, output-exact.
- **Optional INT8 weights:** a further +10 % at 1 stream, with outputs that change only slightly (details below).
- **Default checkpoint:** [hibrid48](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48). INT4-AutoRound and
  hibrid48-uncensored work too.

**New (2026-10-22): [dflash2/](dflash2/)**, a self-contained kit for NVIDIA's FP4-Local checkpoint on myllmbox's v5.3
with the [DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) drafter: +6 % decode over the same
checkpoint with its tuned MTP head, outputs unchanged. `cd dflash2 && ./run.sh`.

The kit itself (run.sh, view.sh, the proxy, the optional patches, the checkpoints and their trade-offs) is documented
in myllmbox's README, kept here as [docs/UPSTREAM-README.md](docs/UPSTREAM-README.md). This file covers what is
different. Changes offered back upstream: [#7](https://github.com/myllmbox/qwen38-flash-next-recipe/pull/7),
[#8](https://github.com/myllmbox/qwen38-flash-next-recipe/issues/8).

## Run it

On a DGX Spark with Docker and the NVIDIA container runtime:

```bash
git clone https://github.com/Alssndr0/qwen38-flash-next-recipe.git
cd qwen38-flash-next-recipe
./tune-host.sh     # once per box: host settings the kit relies on (shows what it runs, asks first)
./run.sh
```

The first run:

1. builds the image `qwen38-flash-next-vllm:v5.2-opt7` from [image/](image/): it pulls myllmbox's
   `myllmbox/qwen38-flash-next-vllm:v5.2` (about 22 GB) and adds this repo's changes, which takes about a minute;
2. downloads the model from Hugging Face (about 98 GB for hibrid48);
3. downloads the two files this checkpoint's speedups need (about 1.5 GB, from this repo's
   [release](https://github.com/Alssndr0/qwen38-flash-next-recipe/releases/tag/v5.2-opt6)) and checks their sha256;
4. starts the serve. It is healthy after about 5 minutes, with an OpenAI-compatible API on `http://127.0.0.1:8000/v1`.

Later runs skip what is already there. `./stop.sh` stops the serve, `./view.sh` shows live stats.
`DRY_RUN=1 ./run.sh` prints the exact `docker run` command and changes nothing.

Everything is set in [recipe.yaml](recipe.yaml). To switch checkpoints, comment the active `model:` line and uncomment
another; `run.sh` picks that checkpoint's own files. For a checkpoint without published files (hibrid48-uncensored),
the draft head and INT8 switch themselves off and the rest still applies.

## What is added

All of it lives in [image/](image/) as a Dockerfile on top of the v5.2 image: diffs against the files already in that
image ([image/patches/](image/patches/)) and three new files ([image/files/](image/files/)). Each speedup is a line in
`recipe.yaml`'s `env:`; remove the line to switch it off. With none of them set the image serves like v5.2.

**Output-exact** (outputs change only within the serve's own run-to-run noise):

| recipe.yaml | what it does |
|---|---|
| `MBX_SKINNY_GEMM_SM12X: "1"` | Already in v5.2, off there. GB10 plans for the small bf16 decode GEMMs (cuBLAS picks SM80 WMMA kernels on this chip). Not measured on its own; counted as part of the set. |
| `OPT_PLE_ASYNC: "1"` | Reads the n-gram (PLE) table rows on a background thread, removing a CPU-GPU sync per forward (0.4 to 10 ms GPU gaps). Byte-checked against the synchronous reader on both the NVFP4 and fp8 tables. |
| `OPT_MTP_FP8: "1"` | The drafter's 11 projections in FP8 (e4m3 per channel, Marlin W8A16): about 80 MB less read per draft pass. Exact through rejection sampling. |
| `mtp_depth: mode: opt` | Batch draft depth = argmax of expected accepted tokens over measured step time; at 1 stream it keeps the shipped per-request logic. Also always on: the depth bookkeeping no longer syncs GPU tensors to the host. |
| `OPT_DRAFT_HEAD_BASIS`, `OPT_DRAFT_HEAD_SCORER: "fp4"` | Two-stage draft vocabulary head: a rank-256 projection shortlists 2,048 candidates and only those are scored with the real head (draft pass 1554 to 487 us at 1 stream). Exact: only the draft proposals can change, and verification against the target is unchanged. Needs a basis fitted per checkpoint (downloaded). |
| `OPT_DRAFT_TOPK: "20"`, `OPT_DRAFT_TOPP: "0.95"` | Draft truncation (v5.2-opt7, needs the two-stage draft head). vLLM samples drafts with temperature only, but verifies them against the target after the request's top_k / top_p (this checkpoint's defaults: 20 / 0.95), so draft probability outside that set can never be accepted. The draft is cut to its own top 20, then top-p 0.95, in one Triton kernel that is cheaper than the scatter it replaces (draft pass 412 vs 450 us at 1 stream). Exact: the verifier uses the same truncated distribution the draft was sampled from. Measured on hibrid48 on replayed OpenCode and Pi coding-agent traffic at 1 stream (177 requests, 25 sessions, paired): +2.0 % decode tok/s (95 % CI +0.4 to +3.7 %), +2.9 % tokens per draft step; a sampled-output distribution test found no difference. Not measured on INT4-AutoRound, where it is exact but its speed effect is unknown. |

**Changes outputs slightly** (on by default here, remove the four `OPT_TARGET_*` lines for 16-bit):

| recipe.yaml | what it does |
|---|---|
| `OPT_TARGET_FP8`, `OPT_TARGET_QFMT: "int8"`, `OPT_TARGET_QGROUP: "128"`, `OPT_TARGET_QCODES` | The target's remaining 16-bit projections (attention, hyper-connections, shared experts, PLE kv, indexer) in INT8 with one scale per 128 weights, through Marlin W8A16, with offline GPTQ codes (downloaded). Without `OPT_TARGET_QCODES` it rounds to nearest; without `OPT_TARGET_QFMT` it is FP8 per row, which is the same speed with about 3x the excess divergence. |

**Always on** (bug fixes, output-exact):

- [vllm#58863](https://github.com/vllm-project/vllm/pull/58863) commit 514102a: the RecoverSSM GDN commit replays the
  accepted tokens with the verify step's own floats, bit-identical to per-token decoding.
- [vllm#58720](https://github.com/vllm-project/vllm/pull/58720): faster expert-weight loading at boot (85 to 62 s here).
- [vllm#58784](https://github.com/vllm-project/vllm/pull/58784): draft slots that were never proposed are rejected.
  Upstream measured the bug on split prefill/decode setups only, so on one box this should change nothing.

`run.sh` gained three things: it builds the image when `server.image` is the tag in [image/TAG](image/TAG), it fetches
the per-checkpoint files listed in [opt/assets.tsv](opt/assets.tsv) into `cache/opt/<checkpoint>/`, and `DRY_RUN=1`.
It also caps INT4-AutoRound's KV pool at 24 GB instead of only replacing an exact 27 GB.

## Measured

One GB10 with the GPU clock capped at 2200 MHz, so an uncapped box is a little faster. Aggregate tok/s over 60 s decode
windows; single sampled runs vary by about 5 to 10 %. Quality: KL and top-1 agreement of teacher-forced next-token
distributions against the stock serve (`scripts/klprobe.py`, about 10,000 positions), compared with each checkpoint's
own run-to-run floor, plus a 400-problem exact-answer eval (`scripts/taskeval.py`).

**hibrid48** (measured on earlier builds of this code, `v5.2-opt3` to `v5.2-opt5`; `v5.2-opt6` adds only the INT4
draft-head path and the state-dump hook; `v5.2-opt7` adds draft truncation, measured in its own row above):

- Output-exact set vs stock v5.2: +9 to +24 % at 1 stream, +1 to +14 % at 4 streams, +2 to +11 % at 16 streams.
  Prefill unchanged.
- INT8 + GPTQ on top (greedy, fixed depth 7, steps/s): 15.3 to 16.9 at 1 stream (+10 %), 48.0 to 49.4 at 16 (+3 %).

  | target projections | KL | top-1 agreement | task eval (2 runs, /400) |
  |---|---|---|---|
  | bf16 (two captures of the same serve) | 0.016 (floor) | 95.85 % | 347 / 347 |
  | FP8 per row | 0.043 | 93.4 % | 349 / 350 |
  | INT8 per 128, round to nearest | 0.028 | 94.5 % | |
  | **INT8 per 128, GPTQ (this repo's default)** | **0.0244** | **94.8 %** | **349 / 348** |

  NLL on the probe text is unchanged for all of them. FP8's e4m3 has 3 mantissa bits, so every weight carries about
  2.6 % error whatever the scaling; INT8 per 128 has 0.80 % and runs on Marlin's 8-bit kernel at FP8's speed.
  Quantizing only attention already gives KL 0.022: most of the divergence comes with any 8-bit nudge (it flips a few
  MoE routing decisions), so quantizing less buys little quality for a lot of speed.

**INT4-AutoRound** (`v5.2-opt6`):

| prompt | streams | stock v5.2 | output-exact set | + draft head | + INT8/GPTQ for its 16-bit layers |
|---|---|---|---|---|---|
| mixed | 1 | 78.3 | 84.5 | 89.0 | 93.3 |
| mixed | 16 | 332.7 | 359.1 | 374.5 | 377.0 |
| code | 1 | 83.5 | 79.1 | 82.3 | 93.6 |
| code | 16 | 369.6 | 383.9 | 362.9 | 396.2 |
| long prose | 1 | 48.4 | 50.6 | 55.5 | 55.8 |
| thinking on | 1 | 50.4 | 51.4 | 54.0 | 57.4 |

- Single runs per cell. The output-exact set measured +2 to +8 % across the validation suite (one code run at 1 stream
  came in at -5 %).
- Output-exact set: KL 0.058 to 0.061 vs stock, at INT4-AutoRound's own floor (0.059 to 0.065); task eval 328 vs stock
  329 / 333.
- INT8/GPTQ here covers only the layers INT4-AutoRound keeps in 16 bits (hyper-connection mixers, indexer, n-gram
  projection): KL 0.069 against that floor.
- In real use this full INT4 configuration sometimes answered in Chinese. Not investigated, and not seen with hibrid48;
  run a language check before relying on it.

## Making the files for another checkpoint

`run.sh` downloads the files for hibrid48 and INT4-AutoRound. For another checkpoint (or to redo them), put them in
`cache/opt/<checkpoint folder name>/` and `run.sh` uses them.

**Draft-head basis** (`draft_basis.pt`):

1. Add `OPT_DUMP_DRAFT_H: "/cache/draft_h.pt"` and `OPT_DUMP_ROWS: "8192"` to `env:` and run. The serve skips the
   first 2,000 draft calls, then saves up to 4 drafter states per call.
2. Send varied traffic: several languages, code, thinking on. The generation step of `scripts/calib_capture.py` does
   this (set `CALIB_OUT` to any writable directory).
3. `python3 scripts/fit_draft_basis.py cache/draft_h.pt cache/opt/<checkpoint>/draft_basis.pt 256`, then remove the two
   dump lines.

**GPTQ codes** (`gptq_int8_g128.pt`):

1. Copy the vLLM source out of the image for the code part of the calibration text:
   `cid=$(docker create qwen38-flash-next-vllm:v5.2-opt7); docker cp $cid:/usr/local/lib/python3.12/dist-packages/vllm vllm-src; docker rm $cid`
2. Boot a calibration serve: remove the four `OPT_TARGET_*` lines, add `OPT_CALIB_DIR: "/cache/calib"`,
   `OPT_CALIB_ROWS: "12288"` and `OPT_CALIB_PER_CALL: "300"`, and use a smaller KV pool
   (`kv-cache-memory: "16000000000"`).
3. `CALIB_MORE=1 CALIB_OUT=cache/calib CALIB_SRC=vllm-src python3 scripts/calib_capture.py`. It generates text, then
   arms the recorders and prefills it: about 34 GB of recorded rows for hibrid48.
4. `./stop.sh`, then run GPTQ inside the image (about 3 minutes):
   `docker run --rm --gpus all -v $PWD/cache:/cache -v $PWD/scripts:/scripts --entrypoint python3 qwen38-flash-next-vllm:v5.2-opt7 /scripts/gptq_int8.py /cache/calib /cache/opt/<checkpoint>/gptq_int8_g128.pt`
5. Restore `recipe.yaml`; `cache/calib` can be deleted.

## Checking a change

- `scripts/klprobe.py capture ref.json` on one serve, `capture new.json` on the other, then
  `scripts/klprobe.py compare ref.json new.json`. The first capture builds the corpus from `KL_SOURCES`
  (comma-separated text files; keep them out of any calibration text). Capture the same serve twice for your floor.
- `scripts/taskeval.py LABEL`: 400 generated problems with exact answers, greedy, thinking off.
- `scripts/fixedbench.py LABEL 1,16 768 3`: greedy, `ignore_eos`, fixed prompts, so variants decode nearly the same
  text; prints tok/s and accepted tokens per step. Better than sampled runs for A/B tests.

The scripts talk to `http://127.0.0.1:8000` (`OPT_URL` changes it) and write to `results/` (`OPT_RESULTS`).

## Credits and license

The kit, its image and the hibrid48 checkpoint are by myllmbox and bilikas; INT4-AutoRound is by
[@azampatti](https://github.com/azampatti). The always-on fixes are backports from the vLLM pull requests linked above.
See [NOTICE](NOTICE) for what is licensed how: this repo's additions are Apache-2.0 ([LICENSE](LICENSE)), the files that
come from myllmbox's repo keep their authors' terms, and the downloaded model-derived files are under the Qwen licenses
included in the release.
