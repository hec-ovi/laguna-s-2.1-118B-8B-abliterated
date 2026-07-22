"""Pins the projection math contract with a pure-python reference (no torch needed).

The identities asserted here are exactly the formulas src/laguna_abliterate/projection.py
implements in torch, so the two cannot drift. When torch is present the final test also
checks the real torch functions against this reference on the same inputs.
"""
import math
import unittest


# --- tiny pure-python linear algebra (lists of lists, row-major) ---
def _lcg(seed):
    x = seed & 0xFFFFFFFF
    while True:
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        yield (x / 0x7FFFFFFF) * 2.0 - 1.0  # in [-1, 1]


def mat(rows, cols, gen):
    return [[next(gen) for _ in range(cols)] for _ in range(rows)]


def matmul(A, B):
    n, k, m = len(A), len(B), len(B[0])
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        for t in range(k):
            a = Ai[t]
            Bt = B[t]
            oi = out[i]
            for j in range(m):
                oi[j] += a * Bt[j]
    return out


def transpose(A):
    return [list(col) for col in zip(*A)]


def norm(vec):
    return math.sqrt(sum(v * v for v in vec))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def col(A, j):
    return [row[j] for row in A]


def unit_vec(v):
    n = norm(v) + 1e-12
    return [x / n for x in v]


# reference implementations of the projection.py formulas
def ablate_weight_left(W, U):
    # W [d_model, in], U [d_model, k] orthonormal -> W - U (U^T W)
    Ut = transpose(U)
    return _sub(W, matmul(U, matmul(Ut, W)))


def ablate_weight_input(W, U):
    # W [d_model, in], U [in, k] orthonormal -> W - (W U) U^T
    Ut = transpose(U)
    return _sub(W, matmul(matmul(W, U), Ut))


def project_out_residual(h_rows, U):
    # h_rows [n, d_model], U [d_model, k] -> h - (h U) U^T
    Ut = transpose(U)
    return _sub(h_rows, matmul(matmul(h_rows, U), Ut))


def _sub(A, B):
    return [[a - b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


DM, IN = 12, 5  # small d_model, in_features


class ProjectionContract(unittest.TestCase):
    def setUp(self):
        g = _lcg(20260722)
        self.W = mat(DM, IN, g)                     # residual writer [d_model, in]
        self.d = unit_vec([next(g) for _ in range(DM)])  # unit direction in d_model
        self.U = [[x] for x in self.d]               # [d_model, 1] orthonormal basis
        self.H = mat(7, DM, g)                        # residual activations [n, d_model]

    def test_left_projection_kills_output_direction(self):
        Wp = ablate_weight_left(self.W, self.U)
        # every column of W' must be orthogonal to d (W' no longer writes span(U))
        for j in range(IN):
            self.assertAlmostEqual(dot(self.d, col(Wp, j)), 0.0, places=9)

    def test_left_projection_idempotent(self):
        Wp = ablate_weight_left(self.W, self.U)
        Wpp = ablate_weight_left(Wp, self.U)
        for r1, r2 in zip(Wp, Wpp):
            for a, b in zip(r1, r2):
                self.assertAlmostEqual(a, b, places=9)

    def test_residual_removal_orthogonal(self):
        Hp = project_out_residual(self.H, self.U)
        for row in Hp:
            self.assertAlmostEqual(dot(self.d, row), 0.0, places=9)

    def test_left_and_input_edits_differ(self):
        # input edit needs U in the in-features space
        g = _lcg(11)
        Uin = [[x] for x in unit_vec([next(g) for _ in range(IN)])]
        left = ablate_weight_left(self.W, self.U)
        inp = ablate_weight_input(self.W, Uin)
        diff = sum(abs(a - b) for ra, rb in zip(left, inp) for a, b in zip(ra, rb))
        self.assertGreater(diff, 1e-6)

    def test_removal_norm_ratio_near_zero(self):
        Wp = ablate_weight_left(self.W, self.U)
        Ut = transpose(self.U)
        before = norm(matmul(Ut, self.W)[0])
        after = norm(matmul(Ut, Wp)[0])
        self.assertLess(after / (before + 1e-12), 1e-9)

    def test_matches_torch_impl(self):
        try:
            import torch
            from laguna_abliterate import projection as P
        except Exception as e:  # torch not installed in this interpreter
            self.skipTest(f"torch/projection unavailable: {e}")
        W = torch.tensor(self.W, dtype=torch.float32)
        U = torch.tensor(self.U, dtype=torch.float32)
        H = torch.tensor(self.H, dtype=torch.float32)
        ref_W = torch.tensor(ablate_weight_left(self.W, self.U), dtype=torch.float32)
        ref_H = torch.tensor(project_out_residual(self.H, self.U), dtype=torch.float32)
        self.assertTrue(torch.allclose(P.ablate_weight_left(W, U), ref_W, atol=1e-5))
        self.assertTrue(torch.allclose(P.project_out_residual(H, U), ref_H, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
