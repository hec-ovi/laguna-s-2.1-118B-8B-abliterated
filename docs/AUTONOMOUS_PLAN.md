# Autonomous pipeline playbook (read this FIRST on every wake-up)

Goal: produce an **abliterated Q4_K_M** of Laguna-S-2.1 that has **reduced over-refusal on
legitimate coding/security requests** while keeping coding capability (not dumber, not
reckless). Quantize with the same mixed layout poolside used. Then run an **HONEST eval** and
report the REAL result.

**Hard rule (the "97/100 was nothing" lesson):** never report a refusal/quality number from
first-token or keyword matching. Acceptance requires >=128-token greedy generations judged
semantically, base-Q4 vs abliterated-Q4, on held-out prompts. A drop in a first-token proxy is
NOT success. If the real eval shows it did nothing or damaged capability, SAY SO.

## How to resume (idempotent, reality-driven — do this every wake-up)

1. Run the status check: `docker ps -a --filter name=laguna_ --format '{{.Names}} {{.Status}}'`
   and `ls -la runs/ models/ artifacts/ 2>/dev/null`.
2. Derive the current stage from REALITY (artifacts on disk), not memory:
   - `runs/dir.pt` missing + `laguna_probe` running  -> STAGE 1 (probe) in progress. Wait.
   - `runs/dir.pt` present, verdict GO/REVIEW         -> go to STAGE 2 (edit).
   - `models/laguna-abliterated-bf16/edit_manifest.json` present -> go to STAGE 3 (GGUF).
   - `models/gguf-abliterated/*-Q4_K_M.gguf` present  -> go to STAGE 4 (honest eval).
   - `runs/eval_report.md` present                    -> DONE. Report honestly, stop the loop.
3. If the current stage's container exited 0, validate its output, then LAUNCH the next stage
   (detached container) and hook a tracked `docker wait` on it (Bash run_in_background:true).
4. If a container exited non-zero, read its logs, fix, relaunch. Do not skip a failed stage.
5. Re-schedule the 30-min heartbeat (ScheduleWakeup) unless DONE/failed-unrecoverable.
6. Append one line to `runs/pipeline.log` describing what you did this wake-up.

## Stages, commands, decision rules

Common docker run prefix (rocm):
```
RG=$(getent group render|cut -d: -f3); VG=$(getent group video|cut -d: -f3)
docker run -d --name <NAME> --device /dev/kfd --device /dev/dri --group-add $RG --group-add $VG \
  --security-opt seccomp=unconfined --shm-size 16g --ipc host -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$PWD":/work -v /home/hec/models/hf/Laguna-S-2.1-bf16:/model:ro <IMG> <cmd>
```

### STAGE 1 - probe (running): `laguna_probe`, image `laguna-abliterate:rocm`
Model-filtered over-refusal direction on synthetic + XSTest + OR-Bench. On completion read
`runs/dir.json`:
- verdict `STOP` (too few refused candidates) -> the candidate set does not trigger Laguna's
  over-refusal enough. Regenerate a harder/more on-model synthetic set (more action-framed,
  add tool-call framing), re-run. Do NOT proceed to edit on a noise direction.
- verdict `GO`/`REVIEW` with a sane chosen lambda + over-refusal drop + benign_kl < 0.1 ->
  proceed to STAGE 2 with `runs/dir.pt` (it stores U + chosen lambda).

### STAGE 2 - permanent BF16 edit
```
docker run ... -v /home/hec/models:/hostmodels laguna-abliterate:rocm \
  python -m laguna_abliterate.edit edit --source /model \
  --out /hostmodels/hf/Laguna-S-2.1-abliterated-bf16 --direction /work/runs/dir.pt \
  --policy ffn_down --projection norm-preserving --lambda <chosen from dir.json>
```
(Norm-preserving, attention o_proj held back.) Verify manifest `all_verified`. ~10-40 min.
NOTE: the edit writes ~219 GiB to /home/hec/models/hf/Laguna-S-2.1-abliterated-bf16.

### STAGE 3 - GGUF (build llama.cpp image first if missing)
`scripts/docker_build.sh` already made `laguna-abliterate:rocm`. Build llama.cpp image:
`docker build -f docker/Dockerfile.llamacpp -t laguna-llamacpp:latest .` (needs the base image).
Then: `IMG=laguna-llamacpp:latest scripts/quantize.sh /home/hec/models/hf/Laguna-S-2.1-abliterated-bf16 /home/hec/models/gguf-abliterated`
Produces F16 -> fresh imatrix (needs data/calibration.txt; if missing, build a small
coding/prose/chat calib) -> Q4_K_M. ~1.5-3 h (imatrix dominates).

### STAGE 4 - HONEST eval (the deliverable)
Serve base-Q4 (`/home/hec/models/gguf/laguna-s-2.1/laguna-s-2.1-Q4_K_M.gguf`) and
abliterated-Q4 via llama.cpp, generate >=128 tokens greedy on HELD-OUT prompts:
- over-refusal: held-out XSTest-safe + OR-Bench + synthetic-action -> semantic compliance rate.
- coding capability: HumanEval/MBPP-style -> unit-test pass rate (RUN the code).
- disposition: benign decision prompts -> check it did not become recklessly overconfident.
Judge semantically (comply = produced the requested legit content, not just absence of "I can't").
Write `runs/eval_report.md` with base vs abliterated numbers, thinking-on AND off, and an
honest verdict. If over-refusal did not drop, or coding pass-rate dropped >2-3pp, or
disposition got reckless: report FAILURE and what to change. Commit + push everything.

## State (update this section each stage)

- 2026-07-22: STAGE 1 probe running on model-filtered synthetic+public over-refusal set,
  candidate-layers 16/20/24/28/32, lambdas 0.8/1.2, norm-preserving edit staged for STAGE 2.
