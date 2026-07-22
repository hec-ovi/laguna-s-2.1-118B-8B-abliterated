"""Laguna-S-2.1 architecture constants and residual-writer target enumeration.

Ground truth: vendor/laguna/config.json + vendor/laguna/modeling_laguna.py.
Pure python, no torch, so the contract tests import it on a bare interpreter.

Only three matrix families write into the d_model residual stream and are therefore
valid targets for a left/output projection ``W' = W - U (U^T W)``:

  * attention  ``self_attn.o_proj``            [d_model, n_heads*head_dim]
  * dense MLP  ``mlp.down_proj``   (layer 0)   [d_model, dense_intermediate]
  * routed exp ``mlp.experts.{e}.down_proj``   [d_model, moe_intermediate]  (sparse layers)
  * shared exp ``mlp.shared_expert.down_proj`` (sparse layers)              [d_model, moe_intermediate]

The router (``mlp.gate.weight``), q/k/v, attention gate (``g_proj``), expert gate/up,
q/k norms, layernorms, embeddings and lm_head are NOT d_model-output residual writers
and must not receive the left projection (see modeling_laguna.py). The conservative
first edit touches only the FFN down-projections; ``o_proj`` is escalation-only.
"""

# --- core dimensions (config.json) ---
HIDDEN = 3072
N_LAYERS = 48
HEAD_DIM = 128
N_KV_HEADS = 8
VOCAB = 100352
N_EXPERTS = 256
TOP_K = 10
MOE_INTERMEDIATE = 1024
SHARED_INTERMEDIATE = 1024
DENSE_INTERMEDIATE = 12288
ROUTED_SCALING_FACTOR = 2.5
SLIDING_WINDOW = 512
RMS_NORM_EPS = 1e-6
NORM_TOPK_PROB = True
ROUTER_LOGIT_SOFTCAP = 0.0  # disabled on this checkpoint

# --- per-layer schedule ---
# mlp_only_layers = [0]; every other layer is sparse MoE.
DENSE_LAYERS = frozenset({0})
# layer_types: full_attention at indices 0,4,8,...,44 ; sliding elsewhere.
FULL_ATTENTION_LAYERS = frozenset(range(0, N_LAYERS, 4))
Q_HEADS_FULL = 48
Q_HEADS_SLIDING = 72


def is_sparse(layer: int) -> bool:
    return layer not in DENSE_LAYERS


def is_full_attention(layer: int) -> bool:
    return layer in FULL_ATTENTION_LAYERS


def layer_type(layer: int) -> str:
    return "full_attention" if is_full_attention(layer) else "sliding_attention"


def num_q_heads(layer: int) -> int:
    return Q_HEADS_FULL if is_full_attention(layer) else Q_HEADS_SLIDING


# --- on-disk tensor names (see vendor/laguna/model.safetensors.index.json) ---
def attn_o_proj(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.o_proj.weight"


def routed_down_projs(layer: int) -> list[str]:
    """256 per-expert down-projections for a sparse layer (empty for dense)."""
    if not is_sparse(layer):
        return []
    return [f"model.layers.{layer}.mlp.experts.{e}.down_proj.weight" for e in range(N_EXPERTS)]


def shared_down_proj(layer: int) -> str | None:
    return None if not is_sparse(layer) else f"model.layers.{layer}.mlp.shared_expert.down_proj.weight"


def dense_down_proj(layer: int) -> str | None:
    return f"model.layers.{layer}.mlp.down_proj.weight" if not is_sparse(layer) else None


def ffn_down_targets(layer: int) -> list[str]:
    """All FFN residual-writing down-projections for one layer."""
    if is_sparse(layer):
        return routed_down_projs(layer) + [shared_down_proj(layer)]
    return [dense_down_proj(layer)]


def all_ffn_down_targets() -> list[str]:
    """Conservative attention-preserving edit set across the whole model."""
    out: list[str] = []
    for layer in range(N_LAYERS):
        out.extend(ffn_down_targets(layer))
    return out


def all_attention_o_proj_targets() -> list[str]:
    """Escalation-only edit set (add to FFN downs only if the probe proves it necessary)."""
    return [attn_o_proj(layer) for layer in range(N_LAYERS)]


# expected counts, asserted by tests/test_arch_plan.py against the real index
EXPECTED_ROUTED_DOWNS = (N_LAYERS - len(DENSE_LAYERS)) * N_EXPERTS  # 47 * 256 = 12032
EXPECTED_SHARED_DOWNS = N_LAYERS - len(DENSE_LAYERS)               # 47
EXPECTED_DENSE_DOWNS = len(DENSE_LAYERS)                            # 1
EXPECTED_FFN_DOWNS = EXPECTED_ROUTED_DOWNS + EXPECTED_SHARED_DOWNS + EXPECTED_DENSE_DOWNS  # 12080
EXPECTED_O_PROJ = N_LAYERS  # 48
