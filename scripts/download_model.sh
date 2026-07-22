#!/usr/bin/env bash
# Download the Laguna-S-2.1 BF16 checkpoint (~219 GiB, 46 shards) into models/.
# OpenMDW licensed, public. Needs ~219 GiB free (this box has ~641 GiB free on NVMe).
#
# The reversible probe reads this checkpoint via accelerate offload; nothing is committed.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="models/Laguna-S-2.1"
mkdir -p "$DEST"

# hf CLI is already on PATH (~/.local/bin). Resume-friendly; re-run to continue a partial pull.
hf download poolside/Laguna-S-2.1 \
  --local-dir "$DEST" \
  --exclude "original/*"

echo "[download] done -> $DEST"
echo "[download] sanity: expect 46 shards + config + modeling code + tokenizer"
ls "$DEST"/model-*.safetensors 2>/dev/null | wc -l
