# laguna-s-2.1 abliteration

Refusal-direction weight surgery for poolside **Laguna-S-2.1** (118B-total / ~8B-active
custom `laguna` MoE, OpenMDW), targeted at running on an AMD Strix Halo box
(Ryzen AI MAX+ 395, Radeon 8060S / gfx1151, 128 GB unified memory).

The plan is deliberately staged so the cheap diagnostic runs before the expensive,
artifact-producing edit. Nothing in the pipeline is unrecoverable: the pristine 219 GiB
checkpoint stays on disk (and is re-downloadable from HF), so any edit can be discarded and
the exact original restored. The edit also writes a manifest (a saved rank-k delta):
auditable, and enough to reconstruct the original to within bf16 rounding without the
source. "Cheap first" is about compute and disk, not about losing the model.

1. **Reversible probe** (`probe`): measure the refusal direction from matched
   harmful/benign activations, project it out of the residual stream at inference with a
   forward hook (no weight edits, no conversion, no quantize), and read three axes back:
   refusal removed, benign KL, over-refusal. This is the go/no-go.
2. **Permanent BF16 edit** (`edit`): FP32 rank-one projection of the FFN down-projections
   (12,032 routed + 47 shared + 1 dense), shard-at-a-time to a separate output dir,
   attention `o_proj` held back initially, with a verified recoverable manifest. Only run
   if the probe is clean.
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
  probe.py        the reversible go/no-go CLI (stage 1)
  manifest.py     edit manifest: recoverable, auditable record of a permanent edit
  edit.py         permanent BF16 shard editor + verify + restore (stage 2)
tests/            stdlib-only contract tests (run without the model or a GPU)
vendor/laguna/    upstream config + modeling code (ground truth for the port)
docker/Dockerfile runtime image (TheRock gfx1151 torch; nothing installed on the host)
scripts/          docker build + run (mounts the model + repo, ROCm device passthrough)
docs/RUNBOOK.md   step-by-step
```

## Setup (Docker, no host python env)

The BF16 checkpoint is already local at `/home/hec/models/hf/Laguna-S-2.1-bf16`
(219 GiB, 46 shards). Everything runs inside a container; nothing is installed on the host.

```
scripts/docker_build.sh                                        # laguna-abliterate:rocm (TheRock gfx1151 torch)
scripts/docker_build.sh cpu                                    # CPU variant, no ROCm
scripts/docker_run.sh                                          # torch/ROCm smoke test
scripts/docker_run.sh python -m unittest discover -s tests -q  # contract tests in-image
```

The go/no-go, once the image is built:

```
scripts/docker_run.sh python -m laguna_abliterate.probe --model-dir /model \
  --candidate-layers 12,16,20,24,28,32 --lambda 1.0 --max-ram 88GiB --gpu-mem 20GiB
```

## Status

- Architecture, method, hardware, and timing research: done (`.research/`, local).
- Core math + weight loader + prompt sets + scoring + tests: in this repo (25 tests pass).
- Reversible probe engine + go/no-go CLI (stage 1): in this repo (accelerate-offload oracle).
- Permanent BF16 shard editor + recoverable manifest (stage 2): in this repo.
- Docker runtime (gfx1151, no host env): built; torch 2.12.0a0+rocm7.13, bf16 verified
  on the iGPU (Radeon 8060S) by the preflight self-check.
- GGUF convert / imatrix / quantize (stage 3): next.

License note: this repo is tooling. Laguna-S-2.1 weights are OpenMDW and already local at
`/home/hec/models/hf/Laguna-S-2.1-bf16`; no weights are committed.
