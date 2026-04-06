# B16 Lambda Sweep - Phase 1 (January 18, 2026)

## Purpose

Validate that the λ (lambda_reg) fix from DF40-only diagnostic experiments transfers to the full `combined_paired` data setup (DF40 + DeepLive).

**Goal:** Achieve ≥0.95 AUC on B16_LAION with `combined_paired` data using CE head (no ArcFace).

## Background

The B16 diagnostic sweep (B1/B2) showed that `lambda_reg=1.0` was **pinning the solution**, preventing capacity from being utilized. When λ was relaxed to 0.01 or 0, performance improved significantly on DF40-only data.

This sweep validates whether the λ fix transfers to the full training setup.

## Focused Experiment Matrix (3 runs)

| ID | Config | λ | Purpose |
|----|--------|---|---------|
| **P1_lambda001** | `P1_lambda001.yaml` | 0.01 | **PRIMARY** - Direct transfer of B1 diagnostic |
| **P1_lambda_anneal** | `P1_lambda_anneal.yaml` | 1.0 → 0.01 | Gradual relaxation (may preserve more pretrained structure) |
| **P1_lambda0** | `P1_lambda0.yaml` | 0.0 | Upper bound (no constraint, may be unstable) |

**All experiments use:**
- `data_source: combined_paired` (DF40 + DeepLive)
- `use_arcface_head: false` (CE loss only - removes instability variable)
- `learning_rate: 2.0e-4`
- `rank: 760` (k=8 trainable directions per layer)
- `train_split: 0.9, val_split: 0.1, test_split: 0.0`

## Key Math

For B16_LAION: `k = embed_dim - rank = 768 - rank`
- k=8: rank=760

## Lambda Annealing (NEW FEATURE)

The `P1_lambda_anneal` experiment uses newly implemented λ annealing:
```yaml
lambda_reg_start: 1.0      # Start with full constraint
lambda_reg_end: 0.01       # End with relaxed constraint
lambda_reg_anneal_steps: 6000  # Anneal over ~6k steps (finish mid-run)
```

This allows the model to initially preserve pretrained structure, then gradually relax to allow adaptation.

## Decision Tree

```
After results:

Any run ≥ 0.95 AUC?
├─ YES → Phase 2: Add ArcFace (conservative: m=0.15, s_end=12)
│        Use best λ setting from this sweep
│
└─ NO → Investigate:
        - Is combined_paired data harder than DF40-only?
        - Check per-dataset breakdown (DF40 vs DeepLive)
        - May need different approach or more data
```

## Launch Commands

```bash
cd DeepfakeBench/training

# Launch all 3 simultaneously
./launch_experiment.sh B16-lambda-sweep asia-southeast1 experiments/B16_lambda_sweep/P1_lambda001.yaml
./launch_experiment.sh B16-lambda-sweep asia-southeast1 experiments/B16_lambda_sweep/P1_lambda_anneal.yaml
./launch_experiment.sh B16-lambda-sweep asia-southeast1 experiments/B16_lambda_sweep/P1_lambda0.yaml
```

## Success Criteria

- **Minimum Success:** Any experiment ≥0.95 AUC on val
- **Target:** Best performer stable and reproducible

## Phase 2 (Conditional - only if Phase 1 succeeds)

Reintroduce ArcFace with conservative settings:
- Use best λ from Phase 1
- Start with `m=0.15, s_end=12` (not aggressive L14 settings)
- Add gradient clipping

## Notes on Previous Experiments

The files `P1_1_*.yaml` through `P1_6_*.yaml` in this folder were from a broader sweep design. 
The focused 3-experiment design (`P1_lambda*.yaml`) follows the refined strategy from the Jan 18 planning discussion.
