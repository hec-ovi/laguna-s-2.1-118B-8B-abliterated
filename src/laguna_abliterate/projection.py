"""Refusal-direction math: measurement, orthonormalization, and projection.

All the load-bearing linear algebra for both the reversible probe and the eventual
permanent edit lives here. The mathematical contract these functions satisfy is
pinned by tests/test_projection_contract.py (a pure-python reimplementation of the
identities), so this torch code and the stdlib reference cannot drift silently.

Conventions
-----------
* Residual activations ``h`` are ``[..., d_model]`` (row vectors).
* A direction basis ``U`` is ``[d_model, k]`` with orthonormal columns.
* Hugging Face linear weights are ``[out_features, in_features]``. A residual writer
  has ``out_features == d_model``; the left projection removes its ability to write
  into span(U):  ``W' = W - U (U^T W) = (I - U U^T) W``.
* The reversible residual intervention removes the component of ``h`` in span(U):
  ``h' = h - lambda * (h U) U^T``.

Everything is computed in float32 and cast back to the caller's dtype at the edge;
never compound fractional edits on already-edited weights.
"""
from __future__ import annotations

import torch


def unit(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize a 1-D direction."""
    v = v.to(torch.float32)
    return v / (v.norm() + eps)


def diff_of_means(harmful: torch.Tensor, harmless: torch.Tensor) -> torch.Tensor:
    """Raw difference-of-means refusal direction (unit-normalized).

    harmful/harmless: [n_examples, d_model] activations captured at one layer/position.
    Returns a unit vector [d_model] in float32. This is the baseline direction; a
    retention-aware or multidirectional variant is layered on top of it elsewhere.
    """
    r = harmful.to(torch.float32).mean(0) - harmless.to(torch.float32).mean(0)
    return unit(r)


def orthonormalize(columns: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis for the column span of ``columns`` ([d_model, k_raw]).

    Uses reduced QR and drops numerically-zero columns, returning [d_model, k].
    A single direction passes through as a unit column.
    """
    columns = columns.to(torch.float32)
    if columns.ndim == 1:
        columns = columns[:, None]
    q, r = torch.linalg.qr(columns, mode="reduced")
    keep = r.diagonal().abs() > 1e-8
    q = q[:, keep]
    # fix sign so the basis is deterministic (largest-abs entry positive per column)
    for j in range(q.shape[1]):
        col = q[:, j]
        idx = int(col.abs().argmax())
        if col[idx] < 0:
            q[:, j] = -col
    return q


def project_out_residual(h: torch.Tensor, U: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Remove span(U) from residual activations: h' = h - lam * (h U) U^T.

    h: [..., d_model]; U: [d_model, k] orthonormal. Returns same shape/dtype as h.
    This is the reversible intervention used by the probe (no weights change).
    """
    dtype = h.dtype
    hf = h.to(torch.float32)
    Uf = U.to(torch.float32)
    coeff = hf @ Uf            # [..., k]
    hf = hf - lam * (coeff @ Uf.transpose(-1, -2))
    return hf.to(dtype)


def ablate_weight_left(W: torch.Tensor, U: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Permanent edit for a residual writer: W' = W - lam * U (U^T W).

    W: [d_model, in_features] (out_features == d_model). U: [d_model, k] orthonormal.
    Removes W's ability to write into span(U). Computed in float32, returned in W.dtype.
    """
    dtype = W.dtype
    Wf = W.to(torch.float32)
    Uf = U.to(torch.float32)
    Wf = Wf - lam * (Uf @ (Uf.transpose(-1, -2) @ Wf))
    return Wf.to(dtype)


def ablate_weight_left_norm_preserving(W: torch.Tensor, U: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Norm-preserving (biprojected) edit: remove span(U) from the output, keep column magnitudes.

    Plain ``ablate_weight_left`` projects each column (a d_model output vector) onto the
    orthogonal complement of U, which shrinks its norm. That shrink is what the model's
    RMSNorm layers were NOT trained to expect and is a source of capability damage. Here we
    project, then rescale each column back to its ORIGINAL norm:

        W' = (I - lam U U^T) W ; then  W'[:, j] *= ||W[:, j]|| / ||W'[:, j]||

    Rescaling a vector already orthogonal to U keeps it orthogonal (the direction stays
    removed: U^T W' == 0), while restoring the per-input-neuron output magnitude the
    normalization layers expect. This is the grimjim/Jim Lai norm-preserving biprojected
    variant used on gpt-oss-120b. Preferred for the permanent edit to minimize capability loss.
    """
    dtype = W.dtype
    Wf = W.to(torch.float32)
    Uf = U.to(torch.float32)
    Wp = Wf - lam * (Uf @ (Uf.transpose(-1, -2) @ Wf))
    orig = Wf.norm(dim=0, keepdim=True)                  # [1, in] original column norms
    new = Wp.norm(dim=0, keepdim=True).clamp_min(1e-12)  # projected column norms
    Wp = Wp * (orig / new)
    return Wp.to(dtype)


def ablate_weight_input(W: torch.Tensor, U: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Input-insensitivity edit (STRONGER, NOT used in the conservative pass): W' = W - lam (W U) U^T.

    Makes W ignore span(U) on its INPUT side. Valid only for matrices whose input is the
    residual (q/k/v, attention gate, expert gate/up, router, lm_head). Provided for
    completeness and clearly distinct from ``ablate_weight_left``; the two are not
    interchangeable. The conservative attention-preserving edit does not call this.
    """
    dtype = W.dtype
    Wf = W.to(torch.float32)
    Uf = U.to(torch.float32)
    Wf = Wf - lam * ((Wf @ Uf) @ Uf.transpose(-1, -2))
    return Wf.to(dtype)


def residual_removal_norm(W_before: torch.Tensor, W_after: torch.Tensor, U: torch.Tensor) -> float:
    """Diagnostic: ||U^T W_after|| relative to ||U^T W_before||.

    ~0 after a correct left projection (the writer no longer emits span(U)).
    """
    Uf = U.to(torch.float32)
    before = (Uf.transpose(-1, -2) @ W_before.to(torch.float32)).norm().item()
    after = (Uf.transpose(-1, -2) @ W_after.to(torch.float32)).norm().item()
    return after / (before + 1e-12)


def cosine_stability(dirs: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine matrix for a stack of bootstrap directions [n, d_model].

    Off-diagonal mean near 1.0 means the measured direction is stable across resamples;
    a low value means the contrast is confounded (topic/length/format, not refusal).
    """
    d = torch.nn.functional.normalize(dirs.to(torch.float32), dim=-1)
    return d @ d.transpose(0, 1)
