"""Validates the edit-target enumeration against the real safetensors index.

Every residual-writer name arch.py enumerates must exist in the checkpoint, the counts
must match the architecture (12,032 routed + 47 shared + 1 dense FFN downs, 48 o_proj),
and no forbidden tensor (router, q/k/v, attention gate, norms, embeddings, lm_head) may
appear in the conservative target set. Skips if the local index is absent.
"""
import os
import unittest

from laguna_abliterate import arch
from laguna_abliterate import weights

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(REPO, "vendor", "laguna", "model.safetensors.index.json")


@unittest.skipUnless(os.path.exists(INDEX), "vendor index.json not present")
class ArchPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wm = weights.load_weight_map(INDEX)

    def test_counts_match_architecture(self):
        routed = sum(len(arch.routed_down_projs(l)) for l in range(arch.N_LAYERS))
        shared = sum(1 for l in range(arch.N_LAYERS) if arch.shared_down_proj(l))
        dense = sum(1 for l in range(arch.N_LAYERS) if arch.dense_down_proj(l))
        self.assertEqual(routed, arch.EXPECTED_ROUTED_DOWNS)   # 12032
        self.assertEqual(shared, arch.EXPECTED_SHARED_DOWNS)   # 47
        self.assertEqual(dense, arch.EXPECTED_DENSE_DOWNS)     # 1
        self.assertEqual(len(arch.all_ffn_down_targets()), arch.EXPECTED_FFN_DOWNS)  # 12080
        self.assertEqual(len(arch.all_attention_o_proj_targets()), arch.EXPECTED_O_PROJ)  # 48

    def test_all_ffn_targets_exist_in_checkpoint(self):
        missing = [n for n in arch.all_ffn_down_targets() if n not in self.wm]
        self.assertEqual(missing, [], f"{len(missing)} FFN targets absent, e.g. {missing[:3]}")

    def test_all_o_proj_targets_exist(self):
        missing = [n for n in arch.all_attention_o_proj_targets() if n not in self.wm]
        self.assertEqual(missing, [])

    def test_group_by_shard_covers_every_target(self):
        targets = arch.all_ffn_down_targets()
        groups = weights.group_by_shard(self.wm, targets)
        covered = [n for names in groups.values() for n in names]
        self.assertEqual(sorted(covered), sorted(targets))
        # each 5 GB shard is visited once; sanity: FFN downs span multiple shards
        self.assertGreater(len(groups), 1)

    def test_forbidden_tensors_not_targeted(self):
        conservative = set(arch.all_ffn_down_targets())
        forbidden = [
            "model.layers.1.mlp.gate.weight",                 # router
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.g_proj.weight",         # attention softplus gate
            "model.layers.1.self_attn.q_norm.weight",
            "model.layers.1.mlp.experts.0.gate_proj.weight",  # expert gate (input-side)
            "model.layers.1.mlp.experts.0.up_proj.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
        ]
        for name in forbidden:
            self.assertIn(name, self.wm, f"{name} should exist in checkpoint")
            self.assertNotIn(name, conservative, f"{name} must NOT be a conservative target")

    def test_dense_layer_has_no_routed_experts(self):
        self.assertEqual(arch.routed_down_projs(0), [])
        self.assertIsNone(arch.shared_down_proj(0))
        self.assertEqual(arch.dense_down_proj(0), "model.layers.0.mlp.down_proj.weight")


if __name__ == "__main__":
    unittest.main()
