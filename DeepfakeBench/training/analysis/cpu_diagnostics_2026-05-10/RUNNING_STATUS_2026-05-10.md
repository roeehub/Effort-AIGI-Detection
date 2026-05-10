# Auto-mode running status — T4 Stage A launches (2026-05-10)

> Updated by Claude during autonomous execution. Most recent state at the top.
> User can intervene at any point: cancel jobs via `gcloud ai custom-jobs cancel
> <JOB_NAME> --region=<REGION> --project=train-cvit2`. Cancellation requires
> user authorization per memory `feedback_no_cancelling_vertex_jobs.md` —
> user must initiate; Claude will not cancel without explicit OK.

---

## What is running and why

**Slot 1 (long, ~5h Vertex training)** — `R13_T4_MULTI_AXIS_GRL_2026-05-10.yaml`
- Region: us-east1 (primary)
- λ_max: 1.0 (mainline, conservative)
- Mechanism: 4-axis adversarial GRL on encoder pooled features (post-ln_post / post-proj L11 CLS for B16). Axes: chronic_flag, is_dor, sharpness_laplacian_high, color_a_approx_dev_high. k=128 bottleneck. λ ramps 0→1.0 over first 500 steps, flat after.
- Init: P8A_REFERENCE_STEP5000 (per pre-test 1: P8A frozen features lift inv_mean 4× under projected multi-axis GRL; T3 features did NOT respond).
- Data: T3 SLOT1 keep-list (drop top-25% lap_var Teams REALS) + visomaster_enhanced + visomaster_teams_enhanced + family_weights. Inherits T3 data discovery.
- Periodic ckpts: {100, 500, 1000, 1500, 2500, 3500, 5000}.
- Cost: ~$60-80.

**Slot 2 (long, ~5h Vertex training)** — `R13_T4_MULTI_AXIS_GRL_LAMBDA2_2026-05-10.yaml`
- Region: us-west4 (parallel; different region per CLAUDE.md to avoid quota contention)
- λ_max: 2.0 (sister-run; pre-test 1's actual peak on frozen features)
- Everything else identical to Slot 1
- Cost: ~$60-80
- Purpose: λ-sensitivity hedge. If λ=1.0 trains stably but doesn't bite, λ=2.0 may extract more invariance. If λ=2.0 collapses forgery signal, we learn the GPU-scale upper bound and ship Slot 1.

**Slot 3 (short, GPU, optional)** — held as backup.
- Will deploy IF: (a) T4 needs a quick re-run with adjusted parameters, or (b) HDTF promotion-contract bug-fix verification (closes `project_phase_c_hdtf_promotion_contract_failure_2026-05-08` open thread).
- If unused after Slot 1+2 deliver clean results, skipped.

---

## Why this combination, briefly

The forgery_signal_atlas (analysis/cpu_diagnostics_2026-05-10/outputs/) shows L11 inv_mean ≈ 0.018-0.031 across **all 7 trained ckpts** (P8A, E2B, T3 step1500, T3 step2500, MCLIOEXB, P18T, P18C) — within probe noise. The encoder co-mingles forgery and shortcut signal regardless of training recipe (data lever, optimizer, head). Pre-test 1 showed an explicit invariance objective on frozen P8A features lifts inv_mean 4× — directional positive evidence. T4 is the GPU-scale realization of that mechanism, with three structural changes that distinguish it from prior failed GRL/penalty packets:

1. **Encoder-level attachment** (vs P15/P18 output-level)
2. **Multi-axis decomposition** (vs single-axis prior packets)
3. **Compressed bottleneck k=128** (matches pre-test 1 winning condition)

---

## Decision gates after T4 lands

For each of the 7 periodic ckpts per run, evaluate:
- forgery_AUC ≥ 0.97 (preserved forgery signal)
- F4_viso_dev recall ≥ 70% (vs P8A 27%)
- may6 false-flag ≤ 6/92 (vs E2B 53/92)
- atlas inv_mean at L11 ≥ 0.05 (vs ceiling 0.031)

| Outcome | Decision |
|---|---|
| All 4 gates clear at any ckpt | Promote that ckpt |
| forgery_AUC ≥ 0.95 + atlas inv_mean ≥ 0.05 but scorecard misses some gates | Mechanism works; iterate on data/τ |
| Mechanism partial bite (some axes drop, others don't) | Stage A2: redesign axis set |
| forgery_AUC < 0.95 OR shortcut accs never drop | λ_max=2.0 (Slot 2) is the answer if its forgery survived |
| Both Slot 1 + Slot 2 miss | Pivot to L6 or L9 attachment OR add localization aux |

---

## Status timeline

(Most recent at top. Updated as events occur.)

| Time | Event |
|---|---|
| TBD | Slot 1 + Slot 2 launches |
| TBD | Cloud Build completion (image bumped from 1.3.277) |
| 2026-05-10 19:12 | Cloud Build started (./dev.sh build-prod -y) |
| 2026-05-10 19:00 | Smoke test passed: encoder grad norm 56.4, GRL head grad norm 3.7, per-axis CE finite, λ=0 stable, GRL reverses by -λ verified |
| 2026-05-10 18:30 | Code committed: 8 files, 1435 insertions; commit 38a7d1f |
| 2026-05-10 17:00-18:30 | Built MultiAxisGRLBlock + wiring + yamls + smoke test |

---

## Where to find things

- Yamls: `experiments/phase2_round13/R13_T4_MULTI_AXIS_GRL_*.yaml`
- New module: `detectors/effort_detector.py::MultiAxisGRLBlock` (line ~265-340)
- Trainer schedule: `trainer/trainer.py::_update_multi_axis_grl_lambda` (line ~795)
- Smoke test: `analysis/cpu_diagnostics_2026-05-10/scripts/smoke_test_multi_axis_grl.py`
- W&B (once running): https://wandb.ai/dtect-vision/phase2-round13
- Vertex jobs: `gcloud ai custom-jobs list --region=<REGION> --project=train-cvit2 --filter="displayName:exp-R13_T4*"`
