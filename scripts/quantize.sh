#!/usr/bin/env bash
# Stage 3: edited BF16 -> F16 GGUF -> fresh imatrix -> Q4_K_M (poolside's mixed layout).
#
#   scripts/quantize.sh <edited-bf16-dir> <out-dir>
#
# Q4_K_M on this MoE arch already reproduces poolside's mixed precision (attention F16,
# norms/router F32, routed-expert-down + lm_head Q6_K, embeddings/dense/shared/routed-gate-up
# Q4_K). Matching them = their converter (upstream llama.cpp Laguna support) + a fresh imatrix
# on representative deployment text (NOT the official one, which describes the pristine model).
set -euo pipefail
cd "$(dirname "$0")/.."

EDITED="${1:?usage: quantize.sh <edited-bf16-dir> <out-dir>}"
OUT="${2:?usage: quantize.sh <edited-bf16-dir> <out-dir>}"
IMG="${IMG:-laguna-llamacpp:latest}"
CALIB="${CALIB:-data/calibration.txt}"
NAME="${NAME:-laguna-s-2.1-abliterated}"
mkdir -p "$OUT"

run() {  # mount repo, edited checkpoint (ro), and output dir
  docker run --rm \
    -v "$PWD":/work -v "$EDITED":/edited:ro -v "$OUT":/out \
    "$IMG" "$@"
}

F16="/out/${NAME}-F16.gguf"
IMAT="/out/${NAME}.imatrix"
Q4="/out/${NAME}-Q4_K_M.gguf"

echo "[quantize] 1/3 convert edited BF16 -> F16 GGUF"
run python /llama.cpp/convert_hf_to_gguf.py /edited --outfile "$F16" --outtype f16

echo "[quantize] 2/3 fresh imatrix over ${CALIB}"
run /llama.cpp/build/bin/llama-imatrix -m "$F16" -f "/work/${CALIB}" -o "$IMAT" --chunks 200

echo "[quantize] 3/3 quantize -> Q4_K_M (imatrix-guided, poolside layout)"
run /llama.cpp/build/bin/llama-quantize --imatrix "$IMAT" "$F16" "$Q4" Q4_K_M

echo "[quantize] done:"
run ls -la /out
echo "[quantize] compare tensor-type layout to poolside's Q4_K_M:"
run /llama.cpp/build/bin/llama-gguf "$Q4" 2>/dev/null | grep -iE 'q4_k|q6_k|f16|f32' | head || true
