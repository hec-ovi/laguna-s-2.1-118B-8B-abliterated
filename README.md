# laguna-s-2.1 abliteration

Refusal-direction weight surgery for poolside **Laguna-S-2.1** (118B-total / ~8B-active
custom `laguna` MoE, OpenMDW), targeted at running on an AMD Strix Halo box
(Ryzen AI MAX+ 395, Radeon 8060S / gfx1151, 128 GB unified memory).

The plan is deliberately staged so the cheap diagnostic runs before the expensive,
artifact-producing edit. Nothing in the pipeline is unrecoverable: the pristine 219 GiB
checkpoint stays on disk (and is re-downloadable from HF), so any edit can be discarded
and the exact original restored. The edit also writes a manifest (a saved rank-k delta):
auditable, and enough to reconstruct the original to within bf16 rounding without the
source. "Cheap first" is about compute and disk, not about losing the model.

1. **Reversible probe** (`probe`): measure the refusal direction from matched
   harmful/benign activations, project it out of the residual stream at inference with a
   forward hook (no weight edits, no conversion, no quantize), and read three axes back:
   refusal removed, benign KL, over-refusal. This is the go/no-go.
2. **Permanent BF16 edit** (next stage): FP32 rank-one projection of the FFN
   down-projections (12,032 routed + 47 shared + 1 dense), shard-at-a-time, attention
   `o_proj` held back initially. Only run if the probe is clean.
3. **GGUF pipeline** (next stage): edited BF16 to F16 GGUF, fresh imatrix, quantize to
   Q4_K_M. Never requantize from Q8.

## Why it is not the copy-paste abliteration notebook

Laguna is a custom architecture (`trust_remote_code`), not a Llama/Qwen. The generic
Heretic/abliterate flow edits the 48 attention `o_proj` and silently misses all 12,079
sparse residual writers (256 routed + 1 shared down-projection per sparse layer). The
correct edit set is the FFN down-projections, and the direction has to be measured against
the real chat template and both thinking modes. See `.research/` (local) for the full
architecture and method notes.

## Layout

```
src/laguna_abliterate/
  arch.py         architecture constants + residual-writer target enumeration (pure python)
  projection.py   refusal-direction math: diff-of-means, orthonormalize, ablate (torch)
  weights.py      safetensors shard index + mmap streaming loader + edit plan
  data.py         matched harmful / benign / benign-lookalike prompt sets
  scoring.py      refusal detection + teacher-forced benign KL
  engine.py       low-VRAM forward runner (accelerate offload) + residual hooks
  probe.py        the reversible go/no-go CLI
tests/            stdlib-only contract tests (run without the model or a GPU)
vendor/laguna/    upstream config + modeling code (ground truth for the port)
scripts/          env setup (gfx1151) and model download
docs/RUNBOOK.md   step-by-step
```

## Setup

```
scripts/setup_env.sh        # python 3.12 venv + torch (TheRock gfx1151) + deps
scripts/download_model.sh   # ~219 GiB BF16 checkpoint to ./models
.venv/bin/pytest -q         # contract tests (no model needed)
```

## Status

- Architecture, method, hardware, and timing research: done (`.research/`, local).
- Core math + weight loader + prompt sets + scoring + tests: in this repo.
- Reversible probe engine: in this repo (accelerate-offload oracle path).
- Custom layer-streaming capture executor, permanent edit, GGUF pipeline: next stages.

License note: this repo is tooling. Laguna-S-2.1 weights are OpenMDW and downloaded
separately; nothing in `models/` is committed.
