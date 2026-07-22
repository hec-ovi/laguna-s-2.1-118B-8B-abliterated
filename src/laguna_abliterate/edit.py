"""Stage 2: permanent BF16 shard editor with a recoverable manifest.

Applies the FP32 left projection ``W' = W - lambda * U (U^T W)`` to the residual-writing
down-projections, shard-at-a-time, reading each 5 GB source shard once and writing a new
shard to a SEPARATE output dir (the source is never touched). Every edit is recorded in a
manifest with per-target removal ratios and per-shard source hashes; a coefficient file
stores ``U^T W`` per target so the edit can be reconstructed without the source.

Verification per shard before it is promoted from ``*.partial``:
  * exact expected target set present, shapes/dtypes preserved, values finite
  * each target's removal ratio ``||U^T W'|| / ||U^T W||`` is near zero
  * every NON-target tensor is byte-equal to the source (nothing edited by accident)

Runs only after a clean, judged stage-1 verdict. Requires the gfx1151 venv.

CLI:
  python -m laguna_abliterate.edit edit    --source DIR --out DIR --direction runs/dir.pt [--lambda 1.0] [--policy ffn_down]
  python -m laguna_abliterate.edit verify  --edited DIR
  python -m laguna_abliterate.edit restore --edited DIR --out DIR
"""
from __future__ import annotations

import argparse
import os
import shutil
import time

import torch

from . import arch
from . import manifest as M
from . import projection as P
from . import weights as W


def _policy_targets(policy: str) -> list[str]:
    if policy == "ffn_down":
        return arch.all_ffn_down_targets()
    if policy == "ffn_down+o_proj":
        return arch.all_ffn_down_targets() + arch.all_attention_o_proj_targets()
    raise ValueError(f"unknown policy {policy!r}")


def _copy_sidecars(source_dir: str, out_dir: str) -> None:
    # tensor names/shapes/dtypes are unchanged, so the index and all config/tokenizer
    # files carry over verbatim.
    for name in os.listdir(source_dir):
        if name.endswith(".safetensors"):
            continue
        src = os.path.join(source_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, name))


def edit(source_dir: str, out_dir: str, direction_file: str, lam: float, policy: str) -> M.EditManifest:
    from safetensors import safe_open
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    payload = torch.load(direction_file, map_location="cpu")
    U = payload["U"].to(torch.float32)              # [d_model, k]
    assert U.shape[0] == arch.HIDDEN, U.shape

    targets = _policy_targets(policy)
    expected = arch.EXPECTED_FFN_DOWNS + (arch.EXPECTED_O_PROJ if policy.endswith("o_proj") else 0)
    assert len(targets) == expected, (len(targets), expected)

    weight_map = W.load_weight_map(os.path.join(source_dir, "model.safetensors.index.json"))
    by_shard = W.group_by_shard(weight_map, targets)
    target_set = set(targets)

    man = M.EditManifest(
        source_dir=os.path.abspath(source_dir),
        output_dir=os.path.abspath(out_dir),
        direction_file=os.path.abspath(direction_file),
        coeff_file=os.path.join(os.path.abspath(out_dir), "restore_coeffs.pt"),
        lam=lam,
        ablate_layers=list(payload.get("ablate_layers", list(range(arch.N_LAYERS)))),
        policy=policy,
        target_count=len(targets),
        expected_target_count=expected,
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        dependency_versions={"torch": torch.__version__},
    )
    coeffs: dict[str, torch.Tensor] = {}

    # every shard, whether or not it holds a target, so pass-through shards are copied too
    all_shards = sorted(set(weight_map.values()))
    for shard in all_shards:
        src_path = os.path.join(source_dir, shard)
        out_path = os.path.join(out_dir, shard)
        rec = M.ShardRecord(shard=shard, source_sha256=M.sha256_file(src_path))
        shard_targets = [n for n in by_shard.get(shard, [])]

        tensors: dict[str, torch.Tensor] = {}
        with safe_open(src_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if name in target_set:
                    Wf = t.to(torch.float32)
                    coeffs[name] = (U.transpose(0, 1) @ Wf).contiguous()  # [k, in], for restore
                    edited = P.ablate_weight_left(t, U, lam)
                    rec.removal_ratios[name] = P.residual_removal_norm(t, edited, U)
                    rec.targets.append(name)
                    tensors[name] = edited.contiguous()
                else:
                    tensors[name] = t.contiguous()

        partial = out_path + ".partial"
        save_file(tensors, partial)
        with open(partial, "rb") as f:
            os.fsync(f.fileno())

        _verify_shard(partial, src_path, target_set, U, lam)
        os.replace(partial, out_path)
        rec.output_sha256 = M.sha256_file(out_path)
        rec.status = "verified"
        man.shards.append(rec)
        print(f"[edit] {shard}: {len(shard_targets)} edited, "
              f"max removal ratio {max(rec.removal_ratios.values(), default=0.0):.2e}")

    torch.save(coeffs, man.coeff_file)
    _copy_sidecars(source_dir, out_dir)
    man.save(os.path.join(out_dir, "edit_manifest.json"))
    assert man.all_verified(), "manifest not fully verified"
    print(f"[edit] done: {man.target_count} targets across {len(man.shards)} shards -> {out_dir}")
    return man


def _verify_shard(out_partial: str, src_path: str, target_set: set, U: torch.Tensor, lam: float) -> None:
    from safetensors import safe_open

    with safe_open(out_partial, framework="pt", device="cpu") as fo, \
         safe_open(src_path, framework="pt", device="cpu") as fs:
        assert set(fo.keys()) == set(fs.keys()), "tensor set changed"
        for name in fo.keys():
            o = fo.get_tensor(name)
            s = fs.get_tensor(name)
            assert o.shape == s.shape and o.dtype == s.dtype, (name, o.shape, s.shape)
            assert torch.isfinite(o.to(torch.float32)).all(), f"non-finite in {name}"
            if name in target_set:
                ratio = P.residual_removal_norm(s, o, U)
                assert ratio < 1e-3, f"{name}: removal ratio {ratio} too high"
            else:
                assert torch.equal(o, s), f"non-target {name} changed"


def verify(edited_dir: str) -> None:
    man = M.EditManifest.load(os.path.join(edited_dir, "edit_manifest.json"))
    bad = [s.shard for s in man.shards if s.status != "verified"]
    ratios = [r for s in man.shards for r in s.removal_ratios.values()]
    edited = sum(len(s.targets) for s in man.shards)
    print(f"[verify] shards={len(man.shards)} edited_targets={edited}/{man.expected_target_count} "
          f"max_removal_ratio={max(ratios, default=0.0):.2e} unverified={bad}")
    assert man.all_verified(), "manifest reports incomplete/failed edit"
    print("[verify] OK")


def restore(edited_dir: str, out_dir: str) -> None:
    """Near-exact reconstruction: W = W' + lambda * U (U^T W). Bit-exact restore = keep source."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    man = M.EditManifest.load(os.path.join(edited_dir, "edit_manifest.json"))
    U = torch.load(man.direction_file, map_location="cpu")["U"].to(torch.float32)
    coeffs = torch.load(man.coeff_file, map_location="cpu")
    os.makedirs(out_dir, exist_ok=True)

    for shard in sorted({s.shard for s in man.shards}):
        with safe_open(os.path.join(edited_dir, shard), framework="pt", device="cpu") as f:
            tensors = {}
            for name in f.keys():
                t = f.get_tensor(name)
                if name in coeffs:
                    restored = t.to(torch.float32) + man.lam * (U @ coeffs[name].to(torch.float32))
                    tensors[name] = restored.to(t.dtype).contiguous()
                else:
                    tensors[name] = t.contiguous()
            save_file(tensors, os.path.join(out_dir, shard))
    _copy_sidecars(edited_dir, out_dir)
    print(f"[restore] near-exact reconstruction written to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Permanent BF16 abliteration edit (stage 2)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("edit")
    e.add_argument("--source", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--direction", required=True)
    e.add_argument("--lambda", dest="lam", type=float, default=1.0)
    e.add_argument("--policy", default="ffn_down", choices=["ffn_down", "ffn_down+o_proj"])
    v = sub.add_parser("verify")
    v.add_argument("--edited", required=True)
    r = sub.add_parser("restore")
    r.add_argument("--edited", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.cmd == "edit":
        edit(a.source, a.out, a.direction, a.lam, a.policy)
    elif a.cmd == "verify":
        verify(a.edited)
    elif a.cmd == "restore":
        restore(a.edited, a.out)


if __name__ == "__main__":
    main()
