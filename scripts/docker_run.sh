#!/usr/bin/env bash
# Run a command inside the abliteration image, with the repo and the BF16 checkpoint mounted.
#
#   scripts/docker_run.sh                                    # torch/ROCm smoke test (default CMD)
#   scripts/docker_run.sh python -m unittest discover -s tests -q
#   scripts/docker_run.sh python -m laguna_abliterate.probe --model-dir /model \
#       --candidate-layers 12,16,20,24,28,32 --max-ram 88GiB --gpu-mem 20GiB
#
# The model is mounted read-only at /model. Override the host path with MODEL=..., the
# image with IMG=..., and set MODE=cpu to drop the ROCm device passthrough.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMG="${IMG:-laguna-abliterate:rocm}"
MODEL="${MODEL:-/home/hec/models/hf/Laguna-S-2.1-bf16}"
MODE="${MODE:-rocm}"

[ -d "$MODEL" ] || { echo "model dir not found: $MODEL (set MODEL=...)"; exit 1; }
mkdir -p "$REPO/runs" "$REPO/offload" "$REPO/.hf"

DEV=()
if [ "$MODE" = "rocm" ]; then
  DEV+=(--device /dev/kfd --device /dev/dri)
  RENDER_GID="$(getent group render | cut -d: -f3 || true)"
  VIDEO_GID="$(getent group video | cut -d: -f3 || true)"
  [ -n "${RENDER_GID:-}" ] && DEV+=(--group-add "$RENDER_GID")
  [ -n "${VIDEO_GID:-}" ] && DEV+=(--group-add "$VIDEO_GID")
  DEV+=(--security-opt seccomp=unconfined)
fi

exec docker run --rm -it \
  "${DEV[@]}" \
  --shm-size 16g --ipc host \
  -v "$REPO":/work \
  -v "$MODEL":/model:ro \
  "$IMG" "$@"
