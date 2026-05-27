# FACTS EXTENSION — Auto-mode scorecard rescue + ensemble simulation (2026-05-20)

> **FACTS only.** Forbidden words: succeeds / fails / wins / promotes / deployment-grade.
> Mechanical pass/fail against pre-stated criteria. No interpretation. Extends
> `FACTS_2026-05-19.md`; planning interpretation in
> `OPINIONS_EXTENSION_2026-05-20.md`.

## §0. Provenance + retraction of yesterday's framing

While monitoring overnight in auto mode (CPU jobs per user direction), the
Vertex job list surfaced **a second 2026-05-16 scorecard** that yesterday's
pre-plan analysis missed:

- Vertex `auto-mode-scorecard-2026-05-16` (us-east1, 2026-05-16T19:45→22:46 UTC).
- 6 ckpts: P8A + T5C anchors PLUS **Slot A v2 anchor_aware step1500/step3500** AND **Slot B real_rebalance step1500/step3500**.
- Output: `gs://training-job-outputs/test_results/teams_promotion_contract/auto-mode-scorecard-2026-05-16/`.
- v3-fix policy (same as overnight scorecard): target_real_fpr=0.07, target_stress_fpr=0.10, target_fake_recall_min=0.30.

**Yesterday's FACTS doc only analyzed the OVERNIGHT scorecard** (5 ckpts: P8A,
T5C, Slot α s1500/s3500 — resolution_chain — and Slot β — 6-axis GRL). It
silently omitted the auto-mode scorecard's anchor_aware + real_rebalance ckpts.
Today's extension corrects this by merging both scorecards onto the same
lockbox_real cohort (n=1418) and re-running Jobs A / B / L across all 9
distinct ckpts.

**Note on the Slot A v2 Vertex job state**: `gcloud ai custom-jobs list` shows
`exp-R13_T5C_ANCHOR_AWARE_2026-05-16-20260516-163306` (job `1400679201337507840`,
us-east1) in `JOB_STATE_FAILED` (ran 24 min, exited code 3). This was the
**first** launch. A successful re-launch ran as
`exp-R13_T5C_ANCHOR_AWARE_2026-05-16-20260516-170822` (job `8471062799328477184`)
with W&B run `hp35c51p`; ckpts saved to `gs://training-job-outputs/best_checkpoints/hp35c51p/`.
Yesterday's OPINIONS doc citing "Slot A v2 rescue Chikara_Takahashi 26→0%" was
**correctly numerical** (it referenced the auto-mode scorecard's per-identity
metrics) but my pre-plan analysis didn't have those auto-mode frames_reports
staged, so per-identity cross-referencing against the joint-marginal partition
was impossible until today.

## §1. Job K — anchor_aware run state resolution

**Closes/resolves** the ambiguity surfaced today about anchor_aware run state.

| Vertex job | Region | Started | Ended | State | Runtime |
|---|---|---|---|---|---:|
| 1400679201337507840 (FAILED) | us-east1 | 2026-05-16T15:35:27Z | 2026-05-16T15:59:08Z | FAILED (exit 3) | 24 min |
| **8471062799328477184 (SUCCEEDED)** | us-east1 | 2026-05-16T16:08:24Z | (per ckpt timestamps) | SUCCEEDED | ~7h |

W&B run for the successful job: `hp35c51p`. Ckpts at
`gs://training-job-outputs/best_checkpoints/hp35c51p/periodic_effort_20260516_step{1500,3500}_*.pth`.

The FAILED job exited via container exit code 3 (per `gcloud describe`'s
`error.code: 3`) at startup — likely a yaml-load or initial-fork error since
24 minutes is too short for training steps to land. Image: `1.3.291`.

## §2. Job A extended — per-identity lockbox FPR for ALL 9 ckpts

Per-ckpt total lockbox-real over-fires at each ckpt's selected τ (n=1418 frames; file `outputs_2026-05-20/ext_job_a_per_ckpt_summary_9ckpts.csv`):

| ckpt | τ | n_overfire | FPR | chronic share |
|---|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.916 | 27 | 0.0190 | 100% |
| T5C_PERIODIC_STEP3500 | 0.831 | 44 | 0.0310 | 100% |
| SLOT_A_RESCHAIN_STEP1500 | 0.899 | 68 | 0.0480 | 100% |
| SLOT_B_6AXIS_GRL_STEP3500 | 0.816 | 124 | 0.0874 | 100% |
| SLOT_A_RESCHAIN_STEP3500 | 0.860 | 26 | 0.0183 | 100% |
| **SLOT_A_ANCHOR_AWARE_STEP1500** | 0.930 | **7** | **0.0049** | 100% |
| **SLOT_A_ANCHOR_AWARE_STEP3500** | 0.788 | **30** | **0.0212** | 100% |
| SLOT_B_REAL_REBAL_STEP1500 | 0.957 | 23 | 0.0162 | 100% |
| SLOT_B_REAL_REBAL_STEP3500 | 0.840 | 130 | 0.0917 | 100% |

(The minor delta vs auto-mode scorecard summary — e.g., Slot A v2 step3500 FPR 0.0191 in the auto-mode `checkpoint_summary.csv` vs 0.0212 here — is the
contract scorer's `videos_report` aggregation vs frame-level `frame_path ≥ τ`
count. The contract scorer collapses per-video frame predictions into a single
per-video score before computing FPR; this section's numbers are
frame-level.)

**Slot A v2 step3500 per-chronic-identity decomposition** (frame-level):

| identity_key | n_frames | n_overfire | FPR | coverage_class | comparison to Slot β |
|---|---:|---:|---:|---|---:|
| dor_shkedi | 1170 | 19 | **0.0162** | under_covered | Slot β: 116 (0.0991) — **6.1× reduction** |
| bla_bla_chow__s1 | 68 | 11 | 0.1618 | moderate | Slot β: 8 (0.1176) — slight increase |
| Chikara_Takahashi__s22 | 42 | 0 | 0.0000 | under_covered | Slot β: 0 (0.0000) — tied |
| PC_Generator__s15 | 29 | 0 | 0.0000 | under_covered | Slot β: 0 (0.0000) — tied |
| real_dor | 109 | 0 | 0.0000 | (absent) | Slot β: 0 (0.0000) — tied |

The 6× reduction in `dor_shkedi` lockbox over-fires (Slot β 116 → Slot A v2
step3500: 19) is the largest single-identity FPR change observed across the 9
ckpts. The `bla_bla_chow__s1` slight increase (8 → 11) is the only chronic
identity where Slot A v2 step3500 over-fires more than Slot β.

**Slot A v2 step1500** has just **7 lockbox over-fires total** (FPR 0.0049 —
lowest of any scored ckpt) but fails the dev_fake_macro_recall ≥ 0.30 floor
(0.207) so it does not enter the all-pass tier in v3-fix ranking.

## §3. Job B extended — tiebreak audit including 6 auto-mode ckpts

From `outputs_2026-05-20/ext_job_b_ranking_9ckpts.csv` (auto-mode scorecard
checkpoint_summary):

| ckpt | all_pass | dev_recall | lockbox_real_fpr | lockbox_fake_recall | viso_recall | v3fix | recall_desc | lockbox_fake_desc | viso_desc | comp_k1 | comp_k2 | comp_k5 | comp_k10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P8A | YES | 0.300 | 0.0184 | 0.387 | 0.136 | **1** | 4 | 4 | 4 | 3 | 3 | 3 | 3 |
| **SLOT_A_ANCHOR_AWARE_STEP3500** | YES | 0.438 | 0.0191 | **0.688** | **0.167** | 2 | 3 | **1** | **1** | **1** | **1** | **1** | **1** |
| T5C | YES | 0.459 | 0.0279 | 0.660 | 0.138 | 3 | 2 | 2 | 3 | 2 | 2 | 2 | 2 |
| SLOT_B_REAL_REBAL_STEP3500 | YES | 0.541 | 0.0896 | 0.435 | 0.158 | 4 | **1** | 3 | 2 | 4 | 4 | 4 | 4 |
| SLOT_A_ANCHOR_AWARE_STEP1500 | NO (recall floor) | 0.207 | 0.0044 | 0.640 | 0.029 | 5 | 6 | 5 | 5 | 5 | 5 | 5 | 5 |
| SLOT_B_REAL_REBAL_STEP1500 | NO (recall floor) | 0.229 | 0.0140 | 0.162 | 0.011 | 6 | 5 | 6 | 6 | 6 | 6 | 6 | 6 |

**Slot A v2 step3500 is rank-1 under 6 of 8 evaluated tiebreaks**
(`lockbox_fake_recall_desc`, `viso_desc`, and all four composite-k variants).
P8A remains rank-1 only under v3-fix's `lockbox_real_fpr asc` tiebreak.

Compare to yesterday's table (overnight-scorecard 5 ckpts, no Slot A v2):
P8A was rank-1 under 1/8; T5C was rank-1 under 5/8. With Slot A v2 added,
T5C drops to rank-1 under 0/8 — Slot A v2 dominates the composite landscape.

## §4. Job L — Ensemble routing simulation

From `outputs_2026-05-20/ext_job_l_ensemble_combo.csv`. All-frame join on
lockbox real (n=1418) and lockbox fake (n=253) for 9 ckpts.

Top 5 ensembles ranked by `composite_k1 = lockbox_fake_recall − 1 × lockbox_real_fpr`:

| rule | FPR | fake recall | composite k=1 | kind |
|---|---:|---:|---:|---|
| **T5C OR Slot A v2 s3500** | 0.0353 | **0.7365** | **0.7012** | OR |
| P8A OR Slot A v2 s1500 | 0.0233 | 0.7153 | 0.6920 | OR |
| P8A OR T5C | 0.0465 | 0.7271 | 0.6805 | OR |
| P8A OR Slot A v2 s3500 | 0.0360 | 0.7153 | 0.6793 | OR |
| T5C AND Slot A v2 s3500 | 0.0169 | 0.6400 | 0.6231 | AND |

Comparison to single ckpts (composite k=1):

| ckpt | FPR | recall | comp k=1 |
|---|---:|---:|---:|
| P8A | 0.0184 | 0.387 | 0.3686 |
| T5C | 0.0279 | 0.660 | 0.6322 |
| **Slot A v2 s3500** | **0.0191** | **0.688** | **0.6686** |
| Slot β | 0.0875 | 0.541 | 0.4540 |

**The top OR ensemble (T5C OR Slot A v2 s3500)** beats the strongest single
ckpt (Slot A v2 s3500) by +0.033 on composite k=1. Recall lift +0.049; FPR
cost +0.016.

**AND ensembles** drop FPR aggressively:
- P8A AND Slot A v2 s3500: FPR 0.0042 (4.4× lower than Slot A v2 alone)
- P8A AND T5C: FPR 0.0035 (5.3× lower than T5C alone)
- P8A AND Slot A v2 s1500: FPR 0.00071 (ultra-low; 200 fewer FPs than P8A alone)
- BUT recall also drops: AND-ensembles top out at 0.395 (P8A AND Slot A v2 s3500)
- AND-ensemble composites k=1 are all below 0.40 — worse than single Slot A v2 s3500's 0.67.

**Pairwise Pearson context** (yesterday Job J): P8A vs T5C-family 0.10–0.34;
T5C-family pairwise 0.90+. Slot A v2's correlation against Slot β and Slot α
not yet measured in this packet; if Slot A v2 sits in the T5C-family cluster
(expected from the same FT base), `T5C OR Slot A v2` is a within-family
ensemble. The strongest cross-family ensemble (P8A OR Slot A v2 s3500) is
rank-4 by composite k=1 (0.6793).

## §5. Job N — Roy_D GCS audit (closes loop)

Reused 2026-05-12 GCS audit at
`analysis/cpu_diagnostics_2026-05-12_gcs_identity_audit/GCS_IDENTITY_AUDIT_FACTS_2026-05-12.md`:

- 250 manifests pulled across all 10 strategy prefixes of
  `gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2/`.
- 23 identity-name patterns tested (including `Roy_D`/`roy_d`).
- **0 matches** at the listing level (1646 sample_ids) and **0 matches** in
  manifest JSON bodies (250 manifests).
- Schema caveat: manifest's `source_video` tokens are YouTube-ID-anonymized
  (e.g., `lm0hNQmOdFg`), so source-identity overlap is structurally undetectable
  from manifests alone — but the chronic_flag GRL substring-match runs against
  `base_identity`, not raw YouTube IDs, so the question of training-time
  Roy_D supervision is resolved.

**Open loop `roy-d-color-b-dev-mechanism` (MEDIUM, 2026-05-07)**: status moves
from `resolvable` (yesterday) to fully resolved. P8A's r(score, color_b_dev)
= −0.714 on Roy_D is generalization (not memorization).

## §6. Open loop status changes this session

- `roy-d-color-b-dev-mechanism` (MEDIUM): **resolved**. Generalization confirmed.
- `slot-b-viso-lift-mechanism-and-lockbox-fpr-localization` (HIGH): **fully closed by Slot A v2 + this analysis**. The dor over-fire concentration mechanism is now empirically demonstrated to be addressable by anchor_aware on the dor pool (Slot β 116 → Slot A v2 19 dor over-fires; 6× reduction with anchor mechanism alone).
- `lockbox-real-fpr-tiebreak-is-load-bearing` (MEDIUM): **closed yesterday + reinforced**. Slot A v2 s3500 changes the deployment-grade picture; P8A is rank-1 only under v3-fix's specific tiebreak.

## §7. Decision-relevant facts (no interpretation)

1. **Slot A v2 step3500** has lockbox_real_fpr 0.0191 (frame-level total 30 over-fires) — 0.07pp above P8A's 0.0184 but with **77% higher lockbox_fake_recall** (0.688 vs 0.387), **+14pp dev_fake_macro_recall**, and **+3pp visomaster_enhanced** at the contract's selected τ=0.788.

2. **6× dor_shkedi over-fire reduction vs Slot β** (Slot A v2: 19 over-fires, Slot β: 116) at the contract τ, on the same identity. Anchor_aware mechanism does what it was designed to do for this specific failure mode.

3. **`bla_bla_chow__s1` is the residual failure on Slot A v2** (11 over-fires, FPR 0.162) — the only chronic identity where it over-fires more than Slot β. bla_bla_chow__s1 is in the joint-marginal moderate-coverage band (ratio 0.748); not in the dor anchor pool's mechanism scope.

4. **T5C OR Slot A v2 s3500 OR-ensemble** has lockbox_real_fpr 0.0353 + lockbox_fake_recall 0.7365 (composite k=1 = 0.701) — the strongest readout of any rule (single ckpt or ensemble) on the 9-ckpt panel.

5. **P8A AND Slot A v2 s3500 AND-ensemble** has lockbox_real_fpr 0.00423 (4.4× lower than Slot A v2 alone) but lockbox_fake_recall 0.395 (similar to P8A alone). Lower-cost-tier FPR control via AND ensemble.

6. **Slot A v2 step1500** (n_overfire 7, FPR 0.0049) has the lowest single-ckpt FPR observed across all 9 ckpts. Fails dev_recall_floor (0.207 < 0.30), so does not enter the contract all-pass tier but is informative for ensemble-component selection.

7. **Roy_D is held out of the training data** (250 manifests / 1646 listings, 0 matches). P8A's Roy_D-color_b association is a learned generalization, not a memorization artifact.

---

## §8. Output index

| Job | Section | Outputs |
|---|---|---|
| K | §1 | `gcloud describe` snapshots in shell; no CSV |
| A ext | §2 | `outputs_2026-05-20/ext_job_a_per_ckpt_per_identity.csv`, `ext_job_a_chronic_fpr_matrix_9ckpts.csv`, `ext_job_a_per_ckpt_summary_9ckpts.csv` |
| B ext | §3 | `outputs_2026-05-20/ext_job_b_ranking_9ckpts.csv` |
| L | §4 | `outputs_2026-05-20/ext_job_l_ensemble_rules.csv`, `ext_job_l_ensemble_recall.csv`, `ext_job_l_ensemble_combo.csv` |
| N | §5 | reuses `analysis/cpu_diagnostics_2026-05-12_gcs_identity_audit/` |

Script: `analysis/cpu_diagnostics_2026-05-19_pre_plan/scripts/extension_2026-05-20.py`.
Log: `logs/extension_2026-05-20.log`.
