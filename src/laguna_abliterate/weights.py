"""Sharded safetensors access for Laguna-S-2.1.

Two responsibilities:

  * ``load_weight_map`` / ``group_by_shard`` are pure-dict helpers (no torch), so the
    edit-target plan can be validated by the stdlib contract tests.
  * ``ShardedCheckpoint`` mmaps the 46 BF16 shards read-only and hands out individual
    tensors by name. This is the low-level reader for the custom layer-streaming
    executor (next stage); the reversible probe uses the accelerate-offload path in
    engine.py instead. torch/safetensors are imported lazily so this module imports on
    a bare interpreter for the plan helpers.

Nothing here writes. Permanent shard editing (atomic per-shard rewrite with a manifest)
is a separate, deliberately-gated stage documented in docs/RUNBOOK.md.
"""
from __future__ import annotations

import json
import os


def load_weight_map(index_path: str) -> dict[str, str]:
    """tensor-name -> shard-filename map from model.safetensors.index.json."""
    with open(index_path) as f:
        return json.load(f)["weight_map"]


def group_by_shard(weight_map: dict[str, str], names) -> dict[str, list[str]]:
    """Group target tensor names by the shard that holds them.

    Raises KeyError if any requested name is absent from the checkpoint, which is the
    point: a bad edit target must fail loudly, not silently no-op. Editing is then done
    shard-at-a-time so each 5 GB shard is read once.
    """
    out: dict[str, list[str]] = {}
    for n in names:
        shard = weight_map[n]
        out.setdefault(shard, []).append(n)
    return out


class ShardedCheckpoint:
    """Read-only mmap access to the 46 BF16 shards, one tensor at a time."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        self.weight_map = load_weight_map(index_path)
        self._handles: dict[str, object] = {}

    def _handle(self, shard: str):
        h = self._handles.get(shard)
        if h is None:
            from safetensors import safe_open  # lazy

            h = safe_open(os.path.join(self.model_dir, shard), framework="pt", device="cpu")
            self._handles[shard] = h
        return h

    def get(self, name: str):
        """Return one tensor (mmap-backed, cpu, bf16). Cheap; materialize with .contiguous()/.float() as needed."""
        return self._handle(self.weight_map[name]).get_tensor(name)

    def names(self) -> list[str]:
        return list(self.weight_map)

    def close(self):
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
