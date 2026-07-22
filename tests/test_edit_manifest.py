"""Contract tests for the stage-2 manifest and the restore identity (stdlib only).

manifest.py is stdlib; edit.py (torch) is not imported here. The restore-formula test
reuses the verified pure-python linear algebra from test_projection_contract so the
"W' + lambda U (U^T W) == W" reconstruction is checked without torch.
"""
import hashlib
import os
import tempfile
import unittest

from laguna_abliterate import manifest as M

from test_projection_contract import (
    _lcg, mat, matmul, transpose, unit_vec, ablate_weight_left,
)


class Manifest(unittest.TestCase):
    def _sample(self):
        rec = M.ShardRecord(
            shard="model-00002-of-00046.safetensors",
            source_sha256="a" * 64,
            output_sha256="b" * 64,
            targets=["model.layers.1.mlp.experts.0.down_proj.weight"],
            removal_ratios={"model.layers.1.mlp.experts.0.down_proj.weight": 1e-7},
            status="verified",
        )
        return M.EditManifest(
            source_dir="/src", output_dir="/out",
            direction_file="/out/dir.pt", coeff_file="/out/restore_coeffs.pt",
            lam=1.0, ablate_layers=list(range(48)), policy="ffn_down",
            target_count=1, expected_target_count=1, shards=[rec],
            created="2026-07-22T00:00:00",
        )

    def test_json_roundtrip(self):
        m = self._sample()
        m2 = M.EditManifest.from_json(m.to_json())
        self.assertEqual(m2.policy, "ffn_down")
        self.assertEqual(m2.shards[0].shard, m.shards[0].shard)
        self.assertEqual(m2.shards[0].removal_ratios, m.shards[0].removal_ratios)
        self.assertEqual(m2.to_json(), m.to_json())  # stable

    def test_save_load_atomic(self):
        m = self._sample()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "edit_manifest.json")
            m.save(p)
            self.assertTrue(os.path.exists(p))
            self.assertFalse(os.path.exists(p + ".partial"))
            self.assertEqual(M.EditManifest.load(p).target_count, 1)

    def test_all_verified_requires_counts_and_status(self):
        m = self._sample()
        self.assertTrue(m.all_verified())
        m.shards[0].status = "written"
        self.assertFalse(m.all_verified())
        m.shards[0].status = "verified"
        m.expected_target_count = 2  # count mismatch
        self.assertFalse(m.all_verified())

    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.bin")
            with open(p, "wb") as f:
                f.write(b"laguna" * 1000)
            self.assertEqual(M.sha256_file(p), hashlib.sha256(b"laguna" * 1000).hexdigest())


class RestoreIdentity(unittest.TestCase):
    def test_restore_reconstructs_original(self):
        # W' = W - lam U (U^T W); restore = W' + lam U (U^T W) == W (exact in float)
        g = _lcg(424242)
        DM, IN = 10, 4
        W = mat(DM, IN, g)
        U = [[x] for x in unit_vec([next(g) for _ in range(DM)])]  # [d_model, 1]
        lam = 1.0
        Wp = ablate_weight_left(W, U)
        coeff = matmul(transpose(U), W)          # [k, in] = U^T W (from ORIGINAL W)
        delta = matmul(U, coeff)                 # [d_model, in]
        restored = [[wp + lam * dv for wp, dv in zip(rp, rd)] for rp, rd in zip(Wp, delta)]
        for r_orig, r_rest in zip(W, restored):
            for a, b in zip(r_orig, r_rest):
                self.assertAlmostEqual(a, b, places=9)


if __name__ == "__main__":
    unittest.main()
