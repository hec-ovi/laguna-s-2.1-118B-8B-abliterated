# ruff: noqa
# Copyright 2025 Poolside and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.integrations import use_experts_implementation, use_kernelized_func
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, can_return_tuple, is_grouped_mm_available
from transformers.utils.generic import TransformersKwargs, merge_with_config_defaults
from transformers.utils.output_capturing import OutputRecorder, capture_outputs
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.generic import maybe_autocast
from .configuration_laguna import LagunaConfig

from transformers import initialization as init
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available


@use_kernel_forward_from_hub("RMSNorm")
class LagunaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        LagunaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LagunaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: LagunaConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(config, device=None, seq_len=None) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        partial = config.rope_parameters.get("partial_rotary_factor", 1.0)
        dim = int(head_dim * partial)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LagunaMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LagunaTopKRouter(nn.Module):
    """Laguna MoE router using sigmoid scoring (not softmax).

    Supports optional router-logit soft-capping and auxiliary-loss-free load
    balancing (arXiv:2408.15664): the per-expert bias ``e_score_correction_bias``
    is added to selection scores but the returned routing weights remain unbiased.
    The bias lives on the router so accelerate's per-module hooks can co-locate it
    with the gate — moving it to the experts module would cross a hook boundary
    and leave the bias on meta under ``device_map="auto"`` / CPU-offload.
    """

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))
        # Zero-initialised so inference on checkpoints that don't ship the bias
        # is a no-op. ``_checkpoint_conversion_mapping`` below remaps the
        # ``mlp.experts.e_score_correction_bias`` key from vLLM-trained
        # checkpoints onto this attribute.
        self.e_score_correction_bias = nn.Parameter(torch.zeros(config.num_experts), requires_grad=False)
        self.router_logit_softcapping = float(getattr(config, "moe_router_logit_softcapping", 0.0) or 0.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight).float()
        if self.router_logit_softcapping > 0.0:
            router_logits = torch.tanh(router_logits / self.router_logit_softcapping) * self.router_logit_softcapping
        routing_scores = torch.sigmoid(router_logits)
        scores_for_selection = routing_scores + self.e_score_correction_bias.to(routing_scores.dtype)
        _, selected_experts = torch.topk(scores_for_selection, self.top_k, dim=-1)
        routing_weights = routing_scores.gather(-1, selected_experts)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        return router_logits, routing_weights, selected_experts


@use_experts_implementation
class LagunaExperts(nn.Module):
    """Fused expert weights as 3D tensors for batched execution."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class LagunaSparseMoeBlock(nn.Module):
    """Laguna MoE block using sigmoid router, fused expert tensors, and a shared expert."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.routed_scaling_factor = float(getattr(config, "moe_routed_scaling_factor", 1.0))
        # ``moe_apply_router_weight_on_input=True`` would require scaling each expert's
        # input (rather than its output) by the routing weight. Supporting it cleanly
        # alongside the fused experts kernels (``grouped_mm`` / ``batched_mm``) is future
        # work; for now we fail loudly so a checkpoint that needs it can't silently
        # diverge from its numerical form.
        if getattr(config, "moe_apply_router_weight_on_input", False):
            raise NotImplementedError(
                "moe_apply_router_weight_on_input=True is not yet supported in the "
                "transformers implementation of Laguna."
            )
        self.gate = LagunaTopKRouter(config)
        self.experts = LagunaExperts(config)
        self.shared_expert = LagunaMLP(config, intermediate_size=config.shared_expert_intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        shared_expert_output = self.shared_expert(hidden_states)
        _, routing_weights, selected_experts = self.gate(hidden_states)
        expert_output = self.experts(hidden_states, selected_experts, routing_weights)
        if self.routed_scaling_factor != 1.0:
            expert_output = expert_output * self.routed_scaling_factor

        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Adapted from transformers.models.glm.modular_glm.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Removes the interleaving of cos and sin from GLM

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# Laguna attention is identical to Qwen2MoE attention except:
# - No QKV bias
# - Explicit head_dim from config
# - Output gating: attn_output = attn_output * softplus(g_proj(hidden_states)) (optional)
# - Per-layer sliding window attention with optional attention sinks
@use_kernelized_func(apply_rotary_pos_emb)
class LagunaAttention(nn.Module):
    def __init__(self, config: LagunaConfig, layer_idx: int, num_heads: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        # Allow the caller (decoder layer) to supply a per-layer head count; fall back
        # to config.num_attention_heads when not provided.
        self.num_heads = num_heads if num_heads is not None else config.num_attention_heads
        self.num_key_value_groups = self.num_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # Per-layer sliding window (follows Gemma2/Cohere2 convention)
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None:
            self.is_sliding = layer_types[layer_idx] == "sliding_attention"
            self.sliding_window = config.sliding_window if self.is_sliding else None
        else:
            self.is_sliding = False
            self.sliding_window = None

        # Laguna: no QKV bias, explicit head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * config.head_dim, config.hidden_size, bias=False)

        # Laguna-specific: optional gating projection.
        # ``gating`` may be:
        #   - True / "per-element": one gate per (head, head_dim) channel
        #   - "per-head":           one gate per head, broadcast across head_dim
        #   - False:                no gating
        gating = getattr(config, "gating", True)
        self.gating = bool(gating)
        self.gate_per_head = gating == "per-head"
        if self.gating:
            g_out = self.num_heads if self.gate_per_head else self.num_heads * config.head_dim
            self.g_proj = nn.Linear(config.hidden_size, g_out, bias=False)

        # Attention sinks (learnable per-head bias for SWA layers)
        if self.is_sliding and getattr(config, "swa_attention_sink_enabled", False):
            self.sink = nn.Parameter(torch.zeros(self.num_heads))

        # QK normalization (RMSNorm applied per-head after reshape, before RoPE)
        self.q_norm = LagunaRMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = LagunaRMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        # QK normalization (applied per-head before RoPE)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # ``attention_mask`` here is already the correct mask for this layer type —
        # ``LagunaModel.forward`` builds separate full-attention and sliding-attention
        # masks (using ``create_causal_mask`` / ``create_sliding_window_causal_mask``)
        # and the decoder layer passes the right one in.
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        # Laguna-specific: apply gating BEFORE o_proj (optional)
        if self.gating:
            gate = F.softplus(self.g_proj(hidden_states).float()).to(attn_output.dtype)
            if self.gate_per_head:
                # gate: [..., num_heads]; broadcast across head_dim
                attn_shape = attn_output.shape
                attn_output = (
                    attn_output.view(*attn_shape[:-1], self.num_heads, self.head_dim) * gate.unsqueeze(-1)
                ).view(attn_shape)
            else:
                attn_output = attn_output * gate

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class LagunaDecoderLayer(GradientCheckpointingLayer):
    """Laguna decoder layer with gated attention and sigmoid-routed MoE."""

    def __init__(self, config: LagunaConfig, layer_idx: int):
        super().__init__()
        per_layer_heads = getattr(config, "num_attention_heads_per_layer", None)
        layer_num_heads = per_layer_heads[layer_idx] if per_layer_heads is not None else config.num_attention_heads
        # Layer type drives mask and position-embedding dispatch in ``LagunaModel.forward``.
        layer_types = getattr(config, "layer_types", None)
        self.attention_type = layer_types[layer_idx] if layer_types is not None else "full_attention"
        self.self_attn = LagunaAttention(config, layer_idx, num_heads=layer_num_heads)
        # Use MoE or dense MLP based on layer configuration
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = LagunaSparseMoeBlock(config)
        else:
            self.mlp = LagunaMLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = LagunaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LagunaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_size = config.hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class LagunaPreTrainedModel(PreTrainedModel):
    config: LagunaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LagunaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = (
        is_grouped_mm_available()
    )  # https://huggingface.co/docs/transformers/experts_interface#torchcompile
    _supports_attention_backend = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(LagunaTopKRouter, index=0),
        "hidden_states": LagunaDecoderLayer,
        "attentions": LagunaAttention,
    }
    # vLLM-trained Laguna checkpoints store the aux-loss-free routing bias on the
    # experts module (``mlp.experts.e_score_correction_bias``). In this impl the
    # bias lives on the router to stay co-located with its consumer across
    # accelerate's per-module hooks, so remap the legacy key on load.
    _checkpoint_conversion_mapping = {
        r"^(.*)\.mlp\.experts\.e_score_correction_bias$": r"\1.mlp.gate.e_score_correction_bias",
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, LagunaExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, LagunaTopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
        # Bare ``nn.Parameter``s that are not covered by the parent's generic
        # Linear/Embedding/norm handling need their own rules so that the
        # __init__ and from_pretrained(state_dict={}) paths produce identical
        # weights under a fixed seed.
        if isinstance(module, LagunaTopKRouter):
            torch.nn.init.zeros_(module.e_score_correction_bias)
        if isinstance(module, LagunaAttention) and hasattr(module, "sink"):
            torch.nn.init.zeros_(module.sink)


class LagunaModel(LagunaPreTrainedModel):
    def __init__(self, config: LagunaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LagunaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LagunaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # ``LagunaRotaryEmbedding`` inherits ``Qwen2MoeRotaryEmbedding``'s flat-shape
        # contract — it reads ``config.rope_parameters["rope_type"]`` at the outer
        # level. Laguna stores rope nested by layer type (``{"full_attention": {...},
        # ...}``), so pass a config clone with the full-attention sub-dict flattened.
        rp = getattr(config, "rope_parameters", None)
        if isinstance(rp, dict) and isinstance(rp.get("full_attention"), dict):
            import copy

            full_config = copy.deepcopy(config)
            full_config.rope_parameters = dict(rp["full_attention"])
            self.rotary_emb = LagunaRotaryEmbedding(config=full_config)
        else:
            self.rotary_emb = LagunaRotaryEmbedding(config=config)

        # Separate RoPE for sliding-window attention layers (when configured).
        # Be careful with ``partial_rotary_factor`` — ``PreTrainedConfig.standardize_rope_params``
        # unconditionally overwrites ``rope_parameters["partial_rotary_factor"]`` with
        # ``self.partial_rotary_factor``, so we must align the top-level field on the
        # cloned config to the SWA value, otherwise the global partial factor silently
        # clobbers the SWA one.
        if getattr(config, "swa_rope_parameters", None) is not None:
            import copy

            swa_config = copy.deepcopy(config)
            swa_config.rope_parameters = dict(config.swa_rope_parameters)
            swa_partial = swa_config.rope_parameters.get("partial_rotary_factor")
            swa_config.partial_rotary_factor = swa_partial
            self.swa_rotary_emb = LagunaRotaryEmbedding(config=swa_config)
        else:
            self.swa_rotary_emb = None

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            ).unsqueeze(0)

        # Build one mask per layer-type so each layer can be dispatched with the right
        # attention pattern (follows the afmoe / cohere2 v5 convention).
        layer_types = getattr(self.config, "layer_types", None)
        has_swa = layer_types is not None and "sliding_attention" in layer_types
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if has_swa:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        global_pe = self.rotary_emb(hidden_states, position_ids)
        # Per-layer-type position embeddings: Laguna optionally uses a different rope for
        # sliding layers (``swa_rope_parameters``). When absent, SWA layers share the
        # global rope.
        if has_swa:
            swa_pe = self.swa_rotary_emb(hidden_states, position_ids) if self.swa_rotary_emb is not None else global_pe
            position_embeddings_mapping = {"full_attention": global_pe, "sliding_attention": swa_pe}
        else:
            position_embeddings_mapping = None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_attn_mask = causal_mask_mapping[decoder_layer.attention_type]
            layer_pos_emb = (
                position_embeddings_mapping[decoder_layer.attention_type]
                if position_embeddings_mapping is not None
                else global_pe
            )
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_attn_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=layer_pos_emb,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = None,
    top_k=2,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | int:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


@auto_docstring
class LagunaForCausalLM(LagunaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LagunaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = ["LagunaForCausalLM", "LagunaModel", "LagunaPreTrainedModel"]


# --- Added: register the native Laguna checkpoint-conversion for trust_remote_code loads.
# transformers >=5.12 skips checkpoint-conversion mappings for custom (remote) code
# unless explicitly registered, which broke loading the shipped per-expert MoE weights.
try:
    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping as _lg_get,
        register_checkpoint_conversion_mapping as _lg_reg,
        USER_REGISTERED_MAPPINGS as _lg_user,
    )

    if "laguna" not in _lg_user:
        _lg_m = _lg_get("laguna")
        if _lg_m is not None:
            _lg_reg("laguna", _lg_m, overwrite=True)
except Exception:
    pass
