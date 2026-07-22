#!/usr/bin/env bash
# Create the project venv on Strix Halo (gfx1151).
#
# The system python here is 3.14, which is ahead of the ROCm/TheRock torch wheels, so we
# build the venv on python 3.12. torch comes from AMD's TheRock nightly index (native
# gfx1151 kernels); pass --cpu to use the CPU wheel instead (slower, but needs no ROCm).
#
#   scripts/setup_env.sh          # iGPU path (ROCm/TheRock, gfx1151)
#   scripts/setup_env.sh --cpu    # CPU-only path (Zen5 AVX512-BF16)
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-rocm}"
PY="$(command -v python3.12 || true)"
if [ -z "$PY" ]; then
  echo "python3.12 not found. Install it (e.g. 'sudo apt install python3.12 python3.12-venv')"
  echo "then re-run. The system python3 is $(python3 --version), too new for the torch wheels."
  exit 1
fi

"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel

if [ "$MODE" = "--cpu" ] || [ "$MODE" = "cpu" ]; then
  echo "[setup] installing CPU torch"
  pip install --index-url https://download.pytorch.org/whl/cpu torch
else
  echo "[setup] installing ROCm/TheRock torch for gfx1151 (nightly)"
  # TheRock native gfx1151 wheels (7.13-7.15 line). See .research/strix-halo-rocm-stack.
  pip install --pre torch torchvision torchaudio \
    --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ || {
      echo "TheRock nightly install failed. Fall back with: scripts/setup_env.sh --cpu"
      exit 1
    }
fi

pip install -r requirements.txt

echo "[setup] verifying torch"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda/hip available:", torch.cuda.is_available(),
      "| hip:", getattr(torch.version, "hip", None))
print("devices:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY

echo "[setup] running contract tests"
PYTHONPATH=src python -m unittest discover -s tests -q
echo "[setup] done. Next: scripts/download_model.sh"
