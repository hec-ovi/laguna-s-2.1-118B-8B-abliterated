"""bf16 correctness self-check on gfx1151 before trusting iGPU activations.

TheRock gfx1151 nightlies have a documented history of bf16 NaN / wrong-result bugs
(ROCm issue #6034, reported on the Jan 2026 `rocm7.11.0a20260106` build). We run on the
newest Linux wheel (`rocm7.13.0a20260411`, ~3 months newer), but "newer, probably fixed"
is not good enough for capturing refusal directions: a silently-corrupt bf16 forward would
poison every direction and every downstream edit.

So we test the rig directly: run the ops the pipeline depends on (bf16 matmul + SDPA
attention, at the model's real hidden/head sizes) on the GPU and compare to a CPU float32
reference. If bf16 is broken on this build, this fails loudly and the probe falls back to
the CPU device_map instead of trusting the iGPU.

Run standalone:  python -m laguna_abliterate.preflight
"""
from __future__ import annotations

from . import arch


def check(device: str = "cuda", seed: int = 0, tol: float = 2e-2, verbose: bool = True) -> bool:
    import torch
    import torch.nn.functional as F

    if device == "cuda" and not torch.cuda.is_available():
        if verbose:
            print("[preflight] no ROCm GPU visible; CPU-only path, bf16 iGPU check skipped")
        return True

    g = torch.Generator().manual_seed(seed)

    # 1) bf16 matmul at d_model = 3072 (a residual-writer-sized GEMM)
    x = torch.randn(4, 512, arch.HIDDEN, generator=g)
    w = torch.randn(arch.HIDDEN, arch.HIDDEN, generator=g) / 32.0
    ref = x.float() @ w.float()
    got = (x.to(device, torch.bfloat16) @ w.to(device, torch.bfloat16)).float().cpu()
    mm_finite = bool(torch.isfinite(got).all())
    mm_err = float((got - ref).abs().mean() / (ref.abs().mean() + 1e-6))
    mm_ok = mm_finite and mm_err < tol

    # 2) bf16 scaled-dot-product attention at head_dim = 128 (SWA/global head size)
    q = torch.randn(2, arch.N_KV_HEADS, 256, arch.HEAD_DIM, generator=g)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    ref_a = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    got_a = F.scaled_dot_product_attention(
        q.to(device, torch.bfloat16), k.to(device, torch.bfloat16), v.to(device, torch.bfloat16)
    ).float().cpu()
    at_finite = bool(torch.isfinite(got_a).all())
    at_err = float((got_a - ref_a).abs().mean() / (ref_a.abs().mean() + 1e-6))
    at_ok = at_finite and at_err < tol

    ok = mm_ok and at_ok
    if verbose:
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        print(f"[preflight] device={device} ({name}) torch={torch.__version__} hip={getattr(torch.version, 'hip', None)}")
        print(f"[preflight] bf16 matmul   : finite={mm_finite} rel_err={mm_err:.2e} -> {'OK' if mm_ok else 'FAIL'}")
        print(f"[preflight] bf16 attention: finite={at_finite} rel_err={at_err:.2e} -> {'OK' if at_ok else 'FAIL'}")
        print(f"[preflight] verdict: {'bf16 OK on this rig' if ok else 'bf16 BROKEN -> use CPU path'}")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if check() else 1)
