# Training Speedup Validation Runbook

**Date:** 2026-04-08
**Goal:** Run a short fixed-seed A/B to compare `main` against `training-speedup-pass1` without changing normal training configs.

For the current Track A `R13_A_trackA_teams_enhanced` canaries, use
`DeepfakeBench/training/experiments/phase2_round13/R13_TRACKA_SPEEDCHECK_RUNBOOK.md`
instead of this generic branch-vs-branch procedure.

## What This Validates

- Wall-clock throughput improvement for the first training window.
- Early-train behavior sanity: loss, confidence, and gradient-health traces should stay aligned.
- The speedup branch should not show unexpected startup stalls, loader failures, or ordering regressions.

## Scope

- This runbook is for an **exact branch-vs-branch** comparison.
- Use separate worktrees so the baseline run uses baseline code, not just baseline YAML values.
- The helper script generates a **short-run derived config** that:
  - preserves the source config's seed
  - stops after a fixed number of train steps
  - disables validation and OOD monitoring noise
  - logs progress every step
  - suppresses frequent checkpoint saves

These derived configs are for validation only. Do not use them for real experiment reporting.

## Recommended Procedure

1. Use the current `training-speedup-pass1` worktree as the candidate and add one detached baseline worktree:

```bash
git worktree add --detach ../DtectVision-main main
```

If you prefer two disposable worktrees, commit or stash your candidate changes first and then add the second one with `git worktree add --detach ../DtectVision-speedup HEAD`.

2. Pick one source experiment to compare.

Recommended starting point:

```text
DeepfakeBench/training/experiments/phase2_round13/R13_B_k64_capacity.yaml
```

3. In the baseline worktree and in the current candidate worktree, generate a short validation config with the same label style and step count:

```bash
python DeepfakeBench/training/scripts/run/prepare_training_speedup_validation.py \
  --input-config DeepfakeBench/training/experiments/phase2_round13/R13_B_k64_capacity.yaml \
  --output-config /tmp/R13_B_k64_capacity.main.speedval.yaml \
  --label main \
  --max-train-steps 100 \
  --force
```

```bash
python DeepfakeBench/training/scripts/run/prepare_training_speedup_validation.py \
  --input-config DeepfakeBench/training/experiments/phase2_round13/R13_B_k64_capacity.yaml \
  --output-config /tmp/R13_B_k64_capacity.speedup.speedval.yaml \
  --label speedup \
  --max-train-steps 100 \
  --force
```

4. Launch each run in the same environment shape you normally use.

Vertex example:

```bash
cd DeepfakeBench/training
./launch_experiment.sh -y phase2-speedup-validation asia-southeast1 /tmp/R13_B_k64_capacity.main.speedval.yaml
```

```bash
cd DeepfakeBench/training
./launch_experiment.sh -y phase2-speedup-validation asia-southeast1 /tmp/R13_B_k64_capacity.speedup.speedval.yaml
```

Local example:

```bash
cd DeepfakeBench/training
python train_sweep.py --param-config /tmp/R13_B_k64_capacity.main.speedval.yaml
```

```bash
cd DeepfakeBench/training
python train_sweep.py --param-config /tmp/R13_B_k64_capacity.speedup.speedval.yaml
```

## Comparison Checklist

- Compare wall-clock from process start to step 100.
- Compare `train/steps_per_sec` after warmup. Ignore the first few startup steps.
- Compare `train/loss/overall` step-by-step. Look for close overlap, not a sustained drift.
- Compare `train/confidence/mean`, `train/confidence/std`, `train/grad_norm`, and `train/params_with_grad`.
- Check logs for loader warnings, missing-frame spikes, worker restarts, or OOD/validation unexpectedly firing.

## Expected Outcome

- The speedup branch should finish the 100-step run materially faster.
- Early-train metrics should remain close enough that there is no sign of a behavioral regression.
- If loss diverges early or loader warnings appear only on the speedup branch, stop and inspect before launching long runs.

## Notes

- Keep the seed from the source YAML unchanged.
- Keep the hardware shape identical between the two runs.
- Run only one branch at a time on a given machine if you want clean wall-clock numbers.
- For broader coverage, repeat the same procedure on one FT config and one scratch config.
