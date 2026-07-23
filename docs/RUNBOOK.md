# Runbook

End-to-end plan for abliterating Laguna-S-2.1 on Strix Halo. Stage 1 (the reversible
go/no-go) is built and runnable once the venv and weights are in place. Stages 2 and 3
are specified here and built next, gated on a clean stage-1 verdict.

## 0. Environment (once) - Docker, no host python

The BF16 checkpoint is already local at `/home/hec/models/hf/Laguna-S-2.1-bf16` (do not
re-download). Everything runs in a container; nothing is installed on the host.

```
scripts/docker_build.sh                    # laguna-abliterate:rocm (TheRock gfx1151 torch)
scripts/docker_build.sh cpu                # CPU fallback image, no ROCm
scripts/docker_run.sh                      # torch/ROCm smoke test
scripts/docker_run.sh python -m unittest discover -s tests -q   # contract tests in-image
```

The image bundles python 3.12 + TheRock native gfx1151 torch (the host python is 3.14,
ahead of the wheels, which is why this is containerized). `docker_run.sh` passes /dev/kfd
and /dev/dri, mounts the checkpoint read-only at `/model` and the repo at `/work`. 219 GiB
does not fit 123 GiB RAM, so the model loads with accelerate offload (iGPU + RAM + NVMe).
Vulkan is the stable llama.cpp backend used later for imatrix and serving.

ROCm version: the image bundles TheRock torch `2.12.0a0+rocm7.13.0a20260411`, the newest
gfx1151 Linux wheel. 7.14 is Windows-only, 7.15 does not exist, and Python 3.13 would
regress to the January `7.11` build that carries the bf16 NaN bugs (ROCm issue #6034). bf16
is not assumed correct, it is verified on this rig by a self-check (run once):

```
scripts/docker_run.sh python -m laguna_abliterate.preflight
```

Confirmed on the Radeon 8060S: bf16 matmul rel-err 2.8e-3, bf16 attention rel-err 3.5e-3,
no NaN. The probe runs the same check automatically and drops to a CPU device_map if a
future build breaks bf16.

### iGPU memory (GTT) and the load path

Measured: torch reports the iGPU as 116 GiB total, but only ~8 GiB is actually allocatable
at model-load time (the amdgpu GTT/TTM cap on unified memory). accelerate's full-model load
does a single ~10 GiB contiguous warmup allocation, which OOMs against that cap
(`expandable_segments` does not help a hard limit). Two ways to run:

1. Raise GTT once (host, one reboot). Append to the kernel cmdline and reboot:
   `amd_iommu=off amdgpu.gttsize=131072 ttm.pages_limit=33554432`
   Then the iGPU can hold ~100 GiB and the accelerate-offload reference engine loads
   directly. Fast, reuses the proven reference forward, but needs a host boot-time change.

2. No host change: the layer-streaming executor (`streaming.py`) never loads the whole
   model. It keeps one decoder layer (~5 GiB) on the iGPU at a time, which fits the current
   ~8 GiB cap, and reads each shard once. This is the default path for the probe on an
   untuned box.

## 1. Reversible go/no-go (built)

The one question this answers: is the refusal direction removable cleanly, or does removing
it drag capability with it. No weight is edited, nothing is converted or quantized. If the
verdict is bad you have spent about an hour of forward passes, not a shard.

```
scripts/docker_run.sh python -m laguna_abliterate.probe \
  --model-dir /model \
  --candidate-layers 12,16,20,24,28,32 \
  --lambda 1.0 --max-ram 88GiB --gpu-mem 20GiB \
  --harmful-file data/harmful.jsonl        # plug a real eval set (AdvBench/StrongREJECT)
```

What it does: captures residual activations for matched harmful/benign prompts at the
candidate layers, picks the layer with the strongest, most stable refusal contrast,
projects that direction out of the residual stream at inference with a hook, and reports
three axes on held-out prompts:

1. refusal removed   compliance rate on held-out harmful, base vs ablated
2. capability kept   teacher-forced benign KL, base vs ablated (near-zero = localized)
3. collateral        over-refusal on legitimate security/coding lookalikes

The built-in verdict is a lexical proxy. Before trusting a GO, add a semantic judge on the
harmful set and a coding eval (run base-Q4 vs a later abliterated-Q4 through the existing
llama-vulkan-strix server). Also run the probe with `--thinking` on and off; the template
default and reasoning mode change the activation contrast.

Stop here if: the direction is unstable across candidate layers, benign KL is not near zero,
gains show up only in refusal substrings, or over-refusal on lookalikes rises.

## 2. Permanent BF16 edit (next stage)

Only after a clean, judged stage-1 result. FP32 rank-one projection of the FFN
down-projections, shard-at-a-time:

- Targets: `arch.all_ffn_down_targets()` (12,032 routed + 47 shared + 1 dense down-projections).
  Attention `o_proj` is held back initially; add `arch.all_attention_o_proj_targets()` only if
  stage-1 component tests prove FFN-only cannot reach the frontier.
- Edit: `projection.ablate_weight_left(W, U, lam)` in FP32, cast once to BF16, from pristine
  weights. Never compound edits.
- Mechanics: `weights.group_by_shard` to read each 5 GB shard once; write
  `*.safetensors.partial`, fsync, verify (keys, shapes, dtypes, finiteness, per-target removal
  ratio via `projection.residual_removal_norm`, unchanged hashes on every non-target tensor),
  then atomic rename and update a manifest. Keep source + edited BF16 until validation
  (fits: 438 GiB of 641 GiB free).
- Reversibility: this edit is recoverable, not destructive. The projection itself is a
  lossy matrix operation (you cannot invert `I - U U^T` from the result alone), but the
  workflow is recoverable two ways: (a) the pristine source stays on disk and is
  re-downloadable from HF (this is the guaranteed bit-exact restore), and (b) the manifest
  saves U, lambda, and the per-target `U^T W` coefficients (tiny, k x in_features), so
  `W = W' + lambda * U (U^T W)` reconstructs the original to within bf16 rounding of the
  edited tensors (near-exact, and cheap/auditable). Never edit shards in place without the
  manifest.
- Gate: the reloaded edited model must reproduce the reversible hook's behavior within BF16
  tolerance. If it disagrees, do not ship.

## 3. GGUF pipeline (next stage)

- Convert edited BF16 to F16 GGUF (Poolside llama.cpp fork).
- Generate a fresh imatrix over representative coding/tool/prose/chat text (not refusal
  prompts) via llama.cpp Vulkan. The official imatrix describes the pristine model; do not
  reuse it.
- Quantize to Q4_K_M with the fresh imatrix. Never requantize from Q8; Q4 comes from the
  edited F16/BF16 master.
- Disk: source + edited F16 + Q4 is about 508 GiB, fits. Serve through llama-vulkan-strix.

## Validation ladder (every stage compares against the one above it)

1. pristine BF16
2. reversible hook (stage 1)
3. in-memory permanent edit
4. cleanly reloaded edited BF16
5. edited F16 GGUF
6. Q4_K_M from the edited master

Structural checks: 46 shards, 36,769 tensors, exact target counts, unchanged hashes on every
untargeted tensor, finite values, reloaded logits match the pre-save edit within tolerance.
Behavioral checks: semantic harmful compliance, over-refusal, coding/agent/tool performance,
KL/NLL, long-context, router top-k overlap and entropy. Disable the DFlash drafter during
initial target validation (it was trained on the original target distribution).

## Timing on this box (from .research/abliteration-pipeline-timing)

Full abliterate-to-BF16 then Q4_K_M is about 2.5-3.5h realistic (1-6h range). The two
prefill-bound stages (activation harvest, imatrix) dominate; edit, convert, and quantize are
each under an hour, gated by NVMe write. Measured Laguna Q4 on this box: prefill 293->196
tok/s, decode 22.7->19.5 tok/s (Vulkan).

## Acceptance gates (from research; all must pass before shipping the edit)

Goal is reduced over-refusal + kept/gained capability (not jailbreak, not dumber, not reckless).
The first-token probe only SELECTS a direction; acceptance requires long-generation eval on the
edited model (base-Q4 vs abliterated-Q4, via the llama-vulkan-strix server), thinking-on AND off.

1. Over-refusal removed (the target): XSTest-safe, OR-Bench-Hard, CyberSecEval false-refusal,
   plus a coding-defensive set. Want a large drop in false refusals on legitimate requests.
2. Coding + agent capability KEPT (the "not dumber / smarter" gate, per the user, load-bearing):
   - unit-test pass rate (HumanEval / MBPP style), patch correctness (SWE-style repo repair),
   - tool-call schema validity, long-horizon / Terminal-Bench-style completion,
   - measured thinking-on AND thinking-off (Laguna's agentic mode is thinking-on).
   Red flag: >2-3pp adjusted drop on held-out coding tasks.
3. Off-target disposition (the reckless-agent risk, arXiv 2607.17427): on non-refusal prompts,
   check confidence/uncertainty calibration, destructive-action rate, tool calls without
   confirmation, "assumed tests passed" claims, overthinking. Abliteration can make an agent
   overconfident; that is the opposite of smarter and must be caught.
4. Fidelity: benign first-token/full KL < 0.01 (Heretic-scale); MMLU/GSM8K (empty-adjusted) within
   2-3pp; no removal of legitimate "missing info / cannot verify" epistemic refusals.
5. MoE stability: router top-10 overlap, router-logit KL, expert-load entropy, and shared-vs-routed
   contribution before/after; a low benign KL can hide categorical routing changes.

Judge protocol (anti-cheat): >=100-token greedy generations + LLM judge, not keyword/first-token
(the "3/100 re-tested as 60/100" failure). Attention o_proj held back; norm-preserving projection
on the expert down_projs; measure per-component (expert edits are more damaging than attention).
