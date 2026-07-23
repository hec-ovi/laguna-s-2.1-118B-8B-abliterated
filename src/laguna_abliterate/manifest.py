"""Edit manifest: the record that makes a permanent edit auditable and recoverable.

Pure stdlib (json + hashlib) so the manifest round-trip and hashing are covered by the
contract tests. The heavy artifacts it references (the direction basis U and the per-target
`U^T W` restore coefficients) live in a separate torch file written by edit.py; the manifest
only stores their path plus verification metadata.

Recovery model:
  * keep the pristine source  -> guaranteed bit-exact restore (just discard the edit).
  * this manifest + coeff file -> reconstruct the original to within bf16 rounding via
    W = W' + lambda * U (U^T W), without needing the source.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass
class ShardRecord:
    shard: str
    source_sha256: str
    output_sha256: str | None = None
    targets: list[str] = field(default_factory=list)      # tensors edited in this shard
    removal_ratios: dict[str, float] = field(default_factory=dict)  # ||U^T W'|| / ||U^T W|| per target
    status: str = "pending"                                # pending | written | verified | failed


@dataclass
class EditManifest:
    """One permanent-edit run. Serializes to a single JSON next to the edited shards."""
    source_dir: str
    output_dir: str
    direction_file: str            # torch file with U [d_model, k]
    coeff_file: str                # torch file with per-target U^T W restore coefficients
    lam: float
    ablate_layers: list[int]
    policy: str                    # "ffn_down" | "ffn_down+o_proj"
    target_count: int
    expected_target_count: int
    shards: list[ShardRecord] = field(default_factory=list)
    created: str = ""
    dependency_versions: dict[str, str] = field(default_factory=dict)
    projection: str = "norm-preserving"  # "norm-preserving" | "left"; coeff-restore valid only for "left"

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "EditManifest":
        d = json.loads(text)
        d["shards"] = [ShardRecord(**s) for s in d.get("shards", [])]
        return cls(**d)

    def save(self, path: str) -> None:
        tmp = path + ".partial"
        with open(tmp, "w") as f:
            f.write(self.to_json())
        import os

        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "EditManifest":
        with open(path) as f:
            return cls.from_json(f.read())

    def all_verified(self) -> bool:
        edited = sum(len(s.targets) for s in self.shards)
        return (
            edited == self.target_count == self.expected_target_count
            and bool(self.shards)
            and all(s.status == "verified" for s in self.shards)
        )
