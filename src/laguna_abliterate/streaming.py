"""Layer-major streaming executor: run Laguna one decoder layer at a time.

The 219 GiB model does not fit, and the iGPU can only allocate ~5-8 GiB of GTT without
host tuning. So instead of loading the whole model, we build the reference module skeleton
on `meta` (no memory), then for each decoder layer stream that layer's weights from the
shards onto the iGPU (~5 GiB), run the whole batch through it, capture the residual, free
it, and move on. Each shard is read once per corpus pass.

The math is the reference model's own `LagunaDecoderLayer.forward` (loaded via
trust_remote_code), not a reimplementation, so attention gating, sigmoid top-10 routing,
per-head QK-norm, YaRN, and the sliding/global split are exactly correct. The only new code
is the plumbing: fused-expert construction from the per-expert shards, rope/mask wiring, and
per-layer materialize/free.

This is the capture engine for stage 1 (prefill activation harvesting). Validate it with:
    python -m laguna_abliterate.streaming --model-dir /model --prompt "The capital of France is"
which runs one full prefill pass and prints the top next tokens (expect "Paris").
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from . import arch
from . import weights as W


def _rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = arch.RMS_NORM_EPS) -> torch.Tensor:
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(-1, keepdim=True)
    x32 = x32 * torch.rsqrt(var + eps)
    return (w * x32.to(x.dtype))


class StreamingLaguna:
    def __init__(self, model_dir: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

        self.device = device
        self.dtype = dtype
        self.ckpt = W.ShardedCheckpoint(model_dir)
        self.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

        with torch.device("meta"):
            self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)
        self.model.eval()
        self.lm = self.model.model
        assert len(self.lm.layers) == arch.N_LAYERS, len(self.lm.layers)

        # Real rotary embeddings (meta buffers are unusable). Mirror LagunaModel.__init__:
        # the global rotary is built from the full_attention (YaRN) rope params; the sliding
        # rotary only exists when config.swa_rope_parameters is set, else sliding shares global.
        RotClass = type(self.lm.rotary_emb)
        full_cfg = copy.deepcopy(self.config)
        full_cfg.rope_parameters = dict(self.config.rope_parameters["full_attention"])
        self.rotary = RotClass(config=full_cfg).to(device)
        self.swa_rotary = None
        if getattr(self.config, "swa_rope_parameters", None) is not None:
            swa_cfg = copy.deepcopy(self.config)
            swa_cfg.rope_parameters = dict(self.config.swa_rope_parameters)
            self.swa_rotary = RotClass(config=swa_cfg).to(device)

        self._make_full_mask = create_causal_mask
        self._make_swa_mask = create_sliding_window_causal_mask

        # small always-resident tensors
        self.embed = self.ckpt.get("model.embed_tokens.weight").to(device, dtype)
        self.norm_w = self.ckpt.get("model.norm.weight").to(device, dtype)
        self.lm_head = self.ckpt.get("lm_head.weight").to(device, dtype)

    # --- build one layer's real state_dict from the per-expert shards ---
    def _layer_state_dict(self, i: int) -> dict[str, torch.Tensor]:
        g = self.ckpt.get
        sd: dict[str, torch.Tensor] = {}
        for n in ("q_proj", "k_proj", "v_proj", "o_proj", "g_proj"):
            sd[f"self_attn.{n}.weight"] = g(f"model.layers.{i}.self_attn.{n}.weight")
        sd["self_attn.q_norm.weight"] = g(f"model.layers.{i}.self_attn.q_norm.weight")
        sd["self_attn.k_norm.weight"] = g(f"model.layers.{i}.self_attn.k_norm.weight")
        sd["input_layernorm.weight"] = g(f"model.layers.{i}.input_layernorm.weight")
        sd["post_attention_layernorm.weight"] = g(f"model.layers.{i}.post_attention_layernorm.weight")

        if arch.is_sparse(i):
            gate = [g(f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight") for e in range(arch.N_EXPERTS)]
            up = [g(f"model.layers.{i}.mlp.experts.{e}.up_proj.weight") for e in range(arch.N_EXPERTS)]
            down = [g(f"model.layers.{i}.mlp.experts.{e}.down_proj.weight") for e in range(arch.N_EXPERTS)]
            # fused: gate_up_proj[e] = cat(gate, up) -> [2*inter, hidden]; stack over experts
            sd["mlp.experts.gate_up_proj"] = torch.stack(
                [torch.cat([gate[e], up[e]], dim=0) for e in range(arch.N_EXPERTS)], dim=0
            )
            sd["mlp.experts.down_proj"] = torch.stack(down, dim=0)  # [E, hidden, inter]
            sd["mlp.gate.weight"] = g(f"model.layers.{i}.mlp.gate.weight")
            sd["mlp.gate.e_score_correction_bias"] = g(f"model.layers.{i}.mlp.experts.e_score_correction_bias")
            for n in ("gate_proj", "up_proj", "down_proj"):
                sd[f"mlp.shared_expert.{n}.weight"] = g(f"model.layers.{i}.mlp.shared_expert.{n}.weight")
        else:
            for n in ("gate_proj", "up_proj", "down_proj"):
                sd[f"mlp.{n}.weight"] = g(f"model.layers.{i}.mlp.{n}.weight")

        # correction bias stays fp32; everything else to compute dtype
        out = {}
        for k, v in sd.items():
            out[k] = v.to(self.device) if k.endswith("e_score_correction_bias") else v.to(self.device, self.dtype)
        return out

    def _free_layer(self, layer) -> None:
        layer.to_empty(device="meta")
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _prep(self, input_ids: torch.Tensor):
        seq = input_ids.shape[1]
        pos_ids = torch.arange(seq, device=self.device).unsqueeze(0)
        inputs_embeds = F.embedding(input_ids, self.embed)
        mk = dict(config=self.config, inputs_embeds=inputs_embeds, attention_mask=None,
                  past_key_values=None, position_ids=pos_ids)
        masks = {"full_attention": self._make_full_mask(**mk),
                 "sliding_attention": self._make_swa_mask(**mk)}
        global_pe = self.rotary(inputs_embeds, pos_ids)
        swa_pe = self.swa_rotary(inputs_embeds, pos_ids) if self.swa_rotary is not None else global_pe
        pes = {"full_attention": global_pe, "sliding_attention": swa_pe}
        return inputs_embeds, pos_ids, masks, pes

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, capture_layers=None, verbose: bool = True):
        """One prefill pass. Returns (last-token logits, {layer: last-token residual [b, d_model]})."""
        import time

        input_ids = input_ids.to(self.device)
        hidden, pos_ids, masks, pes = self._prep(input_ids)
        want = set(capture_layers or [])
        caps: dict[int, torch.Tensor] = {}
        t0 = time.time()
        for i in range(arch.N_LAYERS):
            layer = self.lm.layers[i]
            layer.load_state_dict(self._layer_state_dict(i), strict=True, assign=True)
            typ = layer.attention_type
            hidden = layer(hidden, attention_mask=masks[typ], position_ids=pos_ids,
                           position_embeddings=pes[typ], use_cache=False)
            if i in want:
                caps[i] = hidden[:, -1, :].float().cpu()
            self._free_layer(layer)
            if verbose and (i % 4 == 0 or i == arch.N_LAYERS - 1):
                dt = time.time() - t0
                print(f"[streaming] layer {i:2d}/{arch.N_LAYERS - 1}  {dt:6.1f}s  "
                      f"{dt / (i + 1):.2f}s/layer", flush=True)
        hidden = _rmsnorm(hidden, self.norm_w)
        logits = F.linear(hidden[:, -1, :], self.lm_head).float()
        return logits, caps


    @torch.no_grad()
    def capture_corpus(self, input_ids_list, capture_layers, verbose: bool = True):
        """Layer-major over the WHOLE corpus: load each layer once, run every prompt through it.

        Total weight reads = one pass for the entire calibration set, not one per prompt. This
        is the design that makes activation capture practical on a box that cannot hold the
        model resident. Per-prompt hidden states stay on-device between layers (tiny: a few
        hundred MB for hundreds of short prompts). Returns {layer: [n_prompts, d_model]} at the
        last token.
        """
        import time

        want = set(capture_layers)
        states = []
        for ids in input_ids_list:
            h, pos, masks, pes = self._prep(ids.to(self.device))
            states.append((h, pos, masks, pes))
        caps: dict[int, list] = {l: [] for l in capture_layers}
        t0 = time.time()
        for i in range(arch.N_LAYERS):
            layer = self.lm.layers[i]
            layer.load_state_dict(self._layer_state_dict(i), strict=True, assign=True)
            typ = layer.attention_type
            for j, (h, pos, masks, pes) in enumerate(states):
                h = layer(h, attention_mask=masks[typ], position_ids=pos,
                          position_embeddings=pes[typ], use_cache=False)
                states[j] = (h, pos, masks, pes)
                if i in want:
                    caps[i].append(h[:, -1, :].float().cpu())
            self._free_layer(layer)
            if verbose:
                print(f"[streaming] layer {i:2d}/{arch.N_LAYERS - 1}  {time.time() - t0:6.1f}s", flush=True)
        return {l: torch.cat(caps[l], 0) for l in capture_layers}


def _main():
    import argparse

    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description="Validate the streaming executor with one prefill pass")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt", default="The capital of France is")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    ids = tok(a.prompt, return_tensors="pt").input_ids
    print(f"[streaming] prompt={a.prompt!r} tokens={ids.shape[1]} device={a.device}")
    m = StreamingLaguna(a.model_dir, device=a.device)
    logits, _ = m.forward(ids)
    top = torch.topk(logits[0], 8)
    print("[streaming] top-8 next tokens:")
    for v, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"    {tok.decode([idx])!r:20s} logit={v:.2f}")


if __name__ == "__main__":
    _main()
