# S1 + S2 — P22-redux experiments with built-in probes

**Date drafted**: 2026-05-02 evening
**Author**: this agent, after the P22 CPU follow-up packet
**Foundation evidence**:
- `analysis/p22_eval_2026-05-02/cpu_followups/CPU_FOLLOWUPS_FINDINGS.md`
- `docs/relaunch_handoffs/PSERIES_FACTS_2026-05-02.md` Section 6.5
- W&B run `dot1buye` `train/collapse/class_separation` trajectory:
  peaked step 650, halved by step 800, collapsed step 7000.

## What problem these experiments are solving

P22 (`dot1buye`) showed that aug-curriculum FT can move the contract metric
in the right direction (+25.7pp dev_fake_macro_recall at FPR=2%) but
collapsed the model's score distribution by step 7000-8000 (logit_std went
from 3.0 → 0.10, a 30× collapse). Two structural causes are plausible:

1. **Training too long**. The aug curriculum hits the model with conflicting
   signals (classify accurately AND stay invariant to perturbations). 8000
   steps is enough for the second objective to dominate. Class_separation
   peaked at step 650 — the actual best model is in the first ~1000 steps.

2. **Base checkpoint over-saturated**. P8A_step5000 has saturated on
   shortcut features (Pearson |r| with viso laplacian = 0.51). A short
   aug curriculum may not have enough optimizer steps to undo P8A's
   shortcut commitment AND find a non-shortcut alternative.

S1 tests cause #1 in isolation. S2 tests cause #2 (assuming #1 is also
fixed by the shorter training cap).

## Decision matrix after both experiments complete

|  | S1 better than P22 step1k | S1 not better than P22 step1k |
|--|--|--|
| **S2 better than S1** | base matters more than length; promote S2 base + curriculum | base matters; length doesn't; promote S2 base |
| **S2 not better than S1** | length matters; base is fine; promote S1 | structural problem; aug curriculum + P8A is wrong path |

## S1 — P22 REDUX (short training cap, capped s annealing)

**Hypothesis**: cap training at 1000 steps and ArcFace s at 8.0 to
preserve class separation; the aug curriculum will then deliver its
shortcut-weakening WITHOUT the over-aug collapse.

**Yaml**: `experiments/phase2_round13/R13_S1_P22_REDUX_SHORT.yaml`
**Base ckpt**: P8A_step5000 (same as P22, isolates the cap)
**Curriculum**: identical to P22 (blur σ ∈ [0.5, 4.0] @ p=0.7, JPEG q ∈ [40, 95] @ p=0.6, brightness β ∈ [-40, 40] @ p=0.5)
**Total steps**: 1000
**ArcFace s**: 6.0 → 8.0 over 1000 steps (vs P22's 6.0 → 12.0 over 8000)
**Periodic saves**: every 100 steps for fine-grained ckpt selection
**Cost**: ~$5-8 (8× cheaper than P22 by step count)

**Expected outcome (success)**: a step-700-to-1000 checkpoint where:
- class_separation ≥ 4.0 (vs P22 peak 5.5; step1k 2.7)
- viso recall at FPR=10% (joint) > 24.2% (better than P22 step1k)
- dor lockbox FPs ≤ 5/1170 (not the 36/1170 regression of P22 step8k)

## S2 — EARLIER BASE (FT from P8A step 2500)

**Hypothesis**: a less-saturated base has more representational room to
learn shortcut-resistant features.

**Yaml**: `experiments/phase2_round13/R13_S2_P22_EARLIER_BASE.yaml`
**Base ckpt**: P8A_step2500 (`gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step2500_auc0.9936_eer0.0270.pth`)
**Why step 2500**: peak val AUC of P8A's run was at step 4000 (0.9941).
   Step 2500 (0.9936) is mid-training, before P8A "fully saturated" but
   already discriminative. Step 1000 (0.9928, EER 0.0541) is too early
   (high EER suggests distribution issues).
**Curriculum + training cap**: identical to S1 (1000 steps, s_end=8)
**Cost**: ~$5-8

**Expected outcome (success)**: a step-700-to-1000 checkpoint where:
- viso recall at FPR=10% (joint) > 30% (breaks P8A's 27.1% dev-only ceiling)
- class_separation ≥ S1's best (the upside hypothesis)

## Probes — three layers

### Layer 1: Pre-launch (CPU, pre-Vertex)

**Wiring tests** (`tests/test_s1_redux_wiring.py`, `tests/test_s2_earlier_base_wiring.py`):
- yaml parses; values match expected (1000 steps, s_end=8, base ckpt path)
- single-lever discipline: anchor_aware/face_scale_jitter/GRL all DISABLED
- pipeline_randomization params match P22's curriculum exactly
- periodic_saves at 100-step granularity

**Smoke gate** (already passed for P22; reuse):
- The aug curriculum's post-aug Laplacian distribution overlaps eval suites
  per `analysis/cpu_decision_2026-05-02_pm_late/figures/p22_smoke_realtrainer_pipeline.png`.

### Layer 2: Mid-training (W&B live monitoring)

**Primary metrics to track in real-time** (already logged by trainer):
- `train/collapse/class_separation` — TARGET ≥ 2.0 throughout. ALERT if drops
  below 1.5 before step 800.
- `train/collapse/logit_std` — TARGET ≥ 1.5 throughout. ALERT if drops below 1.0.
- `train/collapse/prob_entropy` — TARGET ≤ 0.55. ALERT if rises above 0.60.
- `train/loss/overall` — TARGET stays in [0.4, 1.2]. ALERT if exceeds 1.5
  (suggests divergence).
- `val_holdout/overall/auc` — informational; NOT used for ckpt selection.

**Per-bucket recall** (informational; tells us when curriculum is working):
- `final_eval/by_value_composite/val_holdout/per_bucket/visomaster*/recall_fake`
- `final_eval/by_value_composite/val_holdout/per_bucket/deeplive*/recall_fake`

**Stop conditions**:
- Hard stop at step 1000 (yaml-enforced via `total_training_steps`)
- If class_separation drops below 1.0 at any step, the run is dead — pull
  the periodic save just BEFORE the drop and skip remaining training (save
  cost). Implementation: monitor wakeup script (CPU-side, no trainer change).

### Layer 3: Post-training (CPU + 1 GPU scorecard each)

**Post-training probes** (`analysis/s1_s2_2026-05-02_planning/probes/`):

#### Probe 1 — `01_select_best_ckpt_by_class_sep.py` (CPU, $0)
Pull W&B history for the run. Identify checkpoint with PEAK
`train/collapse/class_separation`. This is the canonical "best ckpt"
under the new selection policy (vs val_AUC which mislead us on P22).

#### Probe 2 — Run promotion contract scorecard (GPU, ~$3-5 each)
For S1 and S2 separately, launch promotion contract scorecard with
checkpoint map containing:
- The class-sep-best ckpt (per Probe 1)
- The val-AUC-best ckpt (per W&B, for comparison)
- The final-step ckpt (step 1000)
- P8A_REFERENCE_STEP5000 (control)
- P22_AUG_STEP1000 (informative comparison)

Use `arena/launch_teams_promotion_contract.sh` with the
`teams_promotion_contract_2026-04-23_with_dor.yaml` suite manifest (same
as P22 eval).

#### Probe 3 — `02_compute_falsifiers_s1_s2.py` (CPU, $0)
After scorecard threshold_grid.csv lands, compute:
1. F1: Pearson r(score, lap) on full viso (n=550) — re-uses
   `outputs/07_viso_laplacian_fetched.csv` from P22 follow-ups
2. F2: R²(score | attrs) on reals-only AND on pooled (Job C methodology)
3. F3: viso recall at FPR=2/5/10% (joint dev+lockbox)
4. Per-identity FPR at calibrated τ (Job E methodology) — confirm dor
   invariance preserved

Output: `verdict_S1.csv`, `verdict_S2.csv` with PASS/FAIL on each F-criterion.

#### Probe 4 — `03_p22_redux_diff.py` (CPU, $0)
Direct comparison vs P22's existing scorecard:
- For S1's best ckpt and P22 step1k: compare the recall-vs-FPR curves on viso
- Identify whether S1 strictly dominates step1k or wins on some FPR floors only

## Pre-registered falsifiers

### F-S1 (P22 redux: short training cap)

The morning's P22 falsifiers (F1/F2/F3 from
`cpu_decision_2026-05-02_pm_late/FINDINGS_AND_DECISION.md`) carry over
because the aug curriculum is the same. ADDITIONALLY:

- **F-S1-A**: Best ckpt's `train/collapse/class_separation` ≥ 4.0
  (P22's was 5.5 at peak, 2.7 at step1k)
- **F-S1-B**: Best ckpt's viso recall at FPR=10% (joint dev+lockbox) > **24.2%**
  (must beat P22 step1k under the same metric)
- **F-S1-C**: Best ckpt's lockbox dor FPR ≤ **5/1170 = 0.43%** at calibrated
  τ (must preserve P8A's signature defense better than step8k's 36/1170)

**Verdict mapping**:
- 3/3 → S1 succeeded; commit S1's best ckpt as the new FT-base
- 2/3 → S1 partially worked; investigate which lever failed
- 1/3 → S1 didn't bite; structural issue
- 0/3 → S1 failed; cap-on-training is not the dominant factor

### F-S2 (Earlier-base re-FT)

- **F-S2-A**: Best ckpt's class_separation ≥ S1's best ckpt class_separation
- **F-S2-B**: Best ckpt's viso recall at FPR=10% (joint) > **30.0%**
  (must break P8A's dev-only 27.1% ceiling — the structural improvement)
- **F-S2-C**: Best ckpt's lockbox dor FPR ≤ 5/1170 (preserve P8A's defense)

**Verdict mapping**:
- 3/3 → S2 succeeded; the base matters; this is the new FT-base
- 2/3 → S2 partially worked; investigate
- 0-1/3 → P8A_step5000 was right; revert to FT'ing from it for future packets

### F-bundle (S1 + S2 collectively)

If neither S1 nor S2 hits F-x-B (viso > 24% / 30% at FPR=10%):
- The viso problem is not solvable by aug curriculum alone
- Next packet must attack viso structurally (e.g. paired-substrate loss,
  viso-specific GRL, or change in training data composition)
- This rules out two of the prior plausible explanations
  ("trained too long" and "wrong base")

## Order of operations

**Day 1 (pre-launch, CPU)**:
1. Write yamls + tests, run pytest to confirm wiring (this conversation)
2. Smoke gate: re-confirm aug curriculum's CPU smoke is still valid
   (already done for P22 — same curriculum, no need to repeat)
3. Image rebuild (./dev.sh build-prod -y) to bake the new yamls

**Day 1 (launch, GPU)**:
4. Launch S1 first (lower-risk, validates "training cap" hypothesis cheaply)
5. While S1 trains, launch S2 in parallel (different region if possible
   for capacity, per CLAUDE.md region preference)
6. Monitor both runs via W&B + Cloud Logging; check class_separation
   every 30-60 min; abort if collapse signature appears early

**Day 2 (post-launch, CPU + 1 GPU each)**:
7. Pull periodic saves from GCS for both runs
8. Run Probe 1 (CPU) to identify best ckpt by class_separation per run
9. Launch promotion contract scorecard for top-3 ckpts of each run
   (one Vertex job each, ~$3-5)
10. Run Probes 2-4 to compute falsifiers + verdicts
11. Write findings doc; update FACTS + OPINIONS + HANDOFF

**Total cost estimate**: ~$15-25 (vs P22's ~$30-45). Probes are CPU.

## What we'll learn (regardless of outcome)

- Whether the aug curriculum has a "useful window" that's much shorter than 8000 steps (S1 result)
- Whether P8A_step5000 is the right base for shortcut-removal training (S2 result)
- Whether the viso glass ceiling is breakable by aug + training-cap alone, or needs a structural lever (joint S1+S2 result)
- Class_separation as a checkpoint selection metric — this is the first packet where we apply it deliberately
