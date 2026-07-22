#!/usr/bin/env bash
# Build the abliteration image. Default = ROCm (gfx1151); pass 'cpu' for the CPU variant.
#   scripts/docker_build.sh          # laguna-abliterate:rocm  (TheRock gfx1151 torch)
#   scripts/docker_build.sh cpu      # laguna-abliterate:cpu   (CPU torch, no ROCm)
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-rocm}"
if [ "$MODE" = "cpu" ]; then
  docker build -f docker/Dockerfile \
    --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu \
    --build-arg TORCH_PRE= \
    -t laguna-abliterate:cpu .
  echo "[build] laguna-abliterate:cpu"
else
  docker build -f docker/Dockerfile -t laguna-abliterate:rocm .
  echo "[build] laguna-abliterate:rocm"
fi
