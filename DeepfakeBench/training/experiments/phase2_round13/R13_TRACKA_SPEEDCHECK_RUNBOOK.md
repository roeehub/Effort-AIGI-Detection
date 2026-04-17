# R13 Track A Speedcheck Runbook

**Date:** 2026-04-09  
**Target config:** `R13_A_trackA_teams_enhanced`

## Why This Runbook Exists

The active loader modules under `DeepfakeBench/training/data/sources/` are
workspace files rather than a clean git baseline, so a detached `main`
branch-vs-branch A/B is not the right next step here.

For Track A, the fastest reliable validation is:

1. prime the supported discovery caches once
2. run a 600-step **parity canary** with conservative loader knobs
3. run a 600-step **speed canary** with the current speedup knobs
4. compare both against the already-running long `R13_A_trackA_teams_enhanced`
   run for early-path behavior

These canaries are exact for the train/validation path through step 600,
including the real validation at step 500. They intentionally defer OOD loader
construction at startup because OOD first matters at step 5000 and otherwise
startup discovery would dominate the measurement.

## Configs

- Warmup: `DeepfakeBench/training/experiments/phase2_round13/R13_A_trackA_teams_enhanced_SPEEDCHECK_WARMUP1.yaml`
- Parity canary: `DeepfakeBench/training/experiments/phase2_round13/R13_A_trackA_teams_enhanced_SPEEDCHECK_PARITY600.yaml`
- Speed canary: `DeepfakeBench/training/experiments/phase2_round13/R13_A_trackA_teams_enhanced_SPEEDCHECK_SPEED600.yaml`

## What Each One Tests

- `WARMUP1`
  - Writes the DeepLive / VisoMaster / external-train-real discovery caches.
  - Stops after 1 train step.

- `PARITY600`
  - Uses `num_workers: 4`, `prefetch_factor: 2`,
    `visomaster_parallel_download_workers: 1`,
    `teams_parallel_download_workers: 1`.
  - Goal: catch logic/data-path regressions with minimal loader-scheduling drift.

- `SPEED600`
  - Uses the current speedup knobs:
    `num_workers: 8`, `prefetch_factor: 4`,
    `visomaster_parallel_download_workers: 4`,
    `teams_parallel_download_workers: 4`.
  - Goal: measure the actual throughput win on the real Track A recipe.

## Launch Order

Run all three from the current workspace.

```bash
cd DeepfakeBench/training
./launch_experiment.sh -y phase2-tracka-speedcheck asia-southeast1 \
  experiments/phase2_round13/R13_A_trackA_teams_enhanced_SPEEDCHECK_WARMUP1.yaml
```

After that finishes, launch the parity canary:

```bash
cd DeepfakeBench/training
./launch_experiment.sh -y phase2-tracka-speedcheck asia-southeast1 \
  experiments/phase2_round13/R13_A_trackA_teams_enhanced_SPEEDCHECK_PARITY600.yaml
```

Then launch the speed canary:

```bash
cd DeepfakeBench/training
./launch_experiment.sh -y phase2-tracka-speedcheck asia-southeast1 \
  experiments/phase2_round13/R13_A_trackA_teams_enhanced_SPEEDCHECK_SPEED600.yaml
```

Local equivalents:

```bash
cd DeepfakeBench/training
python train_sweep.py --param-config experiments/phase2_round13/R13_A_trackA_teams_enhanced_SPEEDCHECK_WARMUP1.yaml
```

```bash
cd DeepfakeBench/training
python train_sweep.py --param-config experiments/phase2_round13/R13_A_trackA_teams_enhanced_SPEEDCHECK_PARITY600.yaml
```

```bash
cd DeepfakeBench/training
python train_sweep.py --param-config experiments/phase2_round13/R13_A_trackA_teams_enhanced_SPEEDCHECK_SPEED600.yaml
```

## Comparison Checklist

- Compare source-discovery counts and train/val sample counts in the logs.
- Compare `train/loss/overall` for the parity canary against the long Track A run.
- Compare the step-500 validation metrics from parity versus the long Track A run.
- Compare startup-to-first-step wall-clock between parity and speed canaries.
- Compare steady-state throughput between roughly steps 50 and 450.
- Compare `train/confidence/mean`, `train/confidence/std`, `train/grad_norm`, and loader warnings.

## How To Read The Result

- If `PARITY600` tracks the long run closely through the step-500 validation:
  the training/data path is behaving correctly.
- If `SPEED600` is materially faster while keeping the same early-train shape
  and similar step-500 validation:
  launch the full `R13_A_trackA_teams_enhanced` rerun.
- If `PARITY600` is already off:
  stop and inspect before trusting any speed result.
- If `PARITY600` is fine but `SPEED600` drifts:
  the likely cause is loader scheduling rather than a broken code path; judge it
  primarily by validation and gross trend, not exact per-step overlap.
