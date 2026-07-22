"""Low-VRAM forward runner for Laguna-S-2.1 with residual-stream hooks.

This is the correctness-oracle path for the reversible go/no-go: load the reference
model with accelerate device_map offload (weights split across iGPU / 96 GiB RAM / NVMe,
since 219 GiB does not fit), then use forward hooks on each decoder layer to

  * capture the residual stream (for measuring the refusal direction), and
  * project a direction out of the residual at inference (the reversible intervention),

without ever editing a weight. A faster custom layer-streaming executor is a later stage;
for the go/no-go, reference correctness matters more than speed (see docs/RUNBOOK.md).

Requires the gfx1151 venv (torch + transformers >= 5.14 + safetensors). Imports are at
module top on purpose so a bare interpreter surfaces the missing-deps error immediately.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch

from . import arch
from . import projection as P


@dataclass
class Ablation:
    """A reversible residual-stream intervention: remove span(U) at the given layers."""
    U: torch.Tensor              # [d_model, k] orthonormal basis
    layers: list[int]            # decoder layers whose OUTPUT residual gets projected
    lam: float = 1.0


@dataclass
class RunnerConfig:
    model_dir: str
    dtype: str = "bfloat16"
    device_map: str = "auto"
    max_memory: dict | None = None       # e.g. {"cpu": "88GiB", 0: "20GiB"}
    offload_folder: str = "offload"
    enable_thinking: bool = False        # Laguna template supports interleaved reasoning
    attn_implementation: str = "sdpa"


class LagunaRunner:
    def __init__(self, cfg: RunnerConfig):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_dir,
            trust_remote_code=True,
            dtype=getattr(torch, cfg.dtype),
            device_map=cfg.device_map,
            max_memory=cfg.max_memory,
            offload_folder=cfg.offload_folder,
            attn_implementation=cfg.attn_implementation,
        )
        self.model.eval()
        self.layers = self.model.model.layers
        assert len(self.layers) == arch.N_LAYERS, (len(self.layers), arch.N_LAYERS)

    # --- tokenization respecting the real chat template + thinking mode ---
    def render(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.cfg.enable_thinking,
        )

    def _encode(self, prompts: list[str]):
        texts = [self.render(p) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        dev = next(self.model.parameters()).device
        return {k: v.to(dev) for k, v in enc.items()}

    def _last_token_index(self, attention_mask: torch.Tensor) -> torch.Tensor:
        # index of the final real (non-pad) token per sequence, for right padding
        return attention_mask.sum(dim=1) - 1

    # --- residual capture: layer OUTPUT at the last prompt token ---
    @torch.no_grad()
    def capture_residuals(self, prompts: list[str], layers: list[int], batch_size: int = 1):
        """Return {layer: tensor[n_prompts, d_model]} captured at the last prompt token.

        batch_size=1 is the numerical reference; raise it once you have confirmed the
        batched activations match bs=1 (see docs/RUNBOOK.md validation ladder).
        """
        want = set(layers)
        out: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
        captured: dict[int, torch.Tensor] = {}

        def mk_hook(idx):
            def hook(_m, _inp, output):
                if idx in want:
                    captured[idx] = output.detach()
            return hook

        handles = [self.layers[i].register_forward_hook(mk_hook(i)) for i in layers]
        try:
            for s in range(0, len(prompts), batch_size):
                batch = prompts[s : s + batch_size]
                enc = self._encode(batch)
                captured.clear()
                self.model(**enc, use_cache=False)
                last = self._last_token_index(enc["attention_mask"])
                for i in layers:
                    h = captured[i]  # [b, seq, d_model]
                    for b in range(h.shape[0]):
                        out[i].append(h[b, last[b]].float().cpu())
        finally:
            for hd in handles:
                hd.remove()
        return {i: torch.stack(out[i], 0) for i in layers}

    # --- reversible ablation hooks (modify layer output in place-of-return) ---
    @contextlib.contextmanager
    def ablated(self, ab: Ablation | None):
        if ab is None:
            yield
            return
        U = ab.U
        layerset = set(ab.layers)

        def mk_hook():
            def hook(_m, _inp, output):
                return P.project_out_residual(output, U.to(output.device), ab.lam)
            return hook

        handles = [self.layers[i].register_forward_hook(mk_hook()) for i in ab.layers if i in layerset]
        try:
            yield
        finally:
            for hd in handles:
                hd.remove()

    @torch.no_grad()
    def generate(self, prompts: list[str], ablation: Ablation | None = None,
                 max_new_tokens: int = 96, batch_size: int = 1) -> list[str]:
        results: list[str] = []
        with self.ablated(ablation):
            for s in range(0, len(prompts), batch_size):
                batch = prompts[s : s + batch_size]
                enc = self._encode(batch)
                gen = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tok.pad_token_id,
                )
                new = gen[:, enc["input_ids"].shape[1]:]
                results.extend(self.tok.batch_decode(new, skip_special_tokens=True))
        return results

    @torch.no_grad()
    def logits(self, prompts: list[str], ablation: Ablation | None = None, batch_size: int = 1):
        """Teacher-forced next-token logits for benign KL. Returns (logits, mask) lists per batch."""
        outs = []
        with self.ablated(ablation):
            for s in range(0, len(prompts), batch_size):
                batch = prompts[s : s + batch_size]
                enc = self._encode(batch)
                lg = self.model(**enc, use_cache=False).logits.float().cpu()
                outs.append((lg, enc["attention_mask"].cpu()))
        return outs


def pick_layer(caps_harmful: dict[int, torch.Tensor], caps_harmless: dict[int, torch.Tensor]):
    """Choose the layer with the strongest, most stable refusal contrast.

    Score = normalized separation of the two class means along the diff-of-means
    direction. Returns (best_layer, {layer: score}). A real run should also weigh
    bootstrap cosine stability (projection.cosine_stability) before committing.
    """
    scores: dict[int, float] = {}
    for l in caps_harmful:
        h, b = caps_harmful[l], caps_harmless[l]
        d = P.diff_of_means(h, b)
        proj_h = (h.float() @ d)
        proj_b = (b.float() @ d)
        sep = (proj_h.mean() - proj_b.mean()).abs()
        spread = (proj_h.std() + proj_b.std() + 1e-6)
        scores[l] = float(sep / spread)
    best = max(scores, key=scores.get)
    return best, scores
