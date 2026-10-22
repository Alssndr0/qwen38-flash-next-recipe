# DFlash2 drafter for myllmbox/qwen38-flash-next-vllm:v5.3 (see ../README.md). Adds:
#   patches/qwen4exp-dflash2-capture.patch  the target's layer 5/15/25/35/45 outputs (HC-stream mean) for the drafter,
#                                           as a side output; the target's own computation is unchanged
#   patches/qwen4exp-allow-dflash.patch     Qwen4Exp's config check accepts method "dflash"
#   patches/dflash2-speculator.patch        drafter speedup hooks (files/dfl2_opt.py) and opt-in knobs (DFL2_TOPK, a live
#                                           draft top-p / temperature file DFL2_LIVE_FILE, default off)
#   patches/gb10-gemm-plan-fill.patch       GB10 skinny-GEMM plans for every verify width (MBX_GEMM_FILL=1)
#   files/dfl2_opt.py                       NVFP4 / FP8 drafter linears, shortlist candidate head (DFL2_* in recipe.yaml)
#   files/dfl2_basis_r256.pt                the shortlist basis fitted on DFlash2's own hidden states
# With no DFL2_* variable set and method "mtp" in recipe.yaml this image serves like v5.3.
FROM myllmbox/qwen38-flash-next-vllm:v5.3
ARG SITE=/usr/local/lib/python3.12/dist-packages
COPY patches/ /opt/dfl2/patches/
RUN set -e; cd ${SITE}; for p in /opt/dfl2/patches/*.patch; do patch -s -p1 --no-backup-if-mismatch < "$p"; done
COPY files/dfl2_opt.py ${SITE}/
COPY files/dfl2_basis_r256.pt /opt/dfl2/basis.pt
RUN python3 -m py_compile ${SITE}/dfl2_opt.py ${SITE}/vllm/models/qwen4_exp/nvidia/model.py \
        ${SITE}/vllm/model_executor/models/config.py ${SITE}/vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py \
        ${SITE}/vllm/models/qwen4_exp/nvidia/low_latency_gemm.py
