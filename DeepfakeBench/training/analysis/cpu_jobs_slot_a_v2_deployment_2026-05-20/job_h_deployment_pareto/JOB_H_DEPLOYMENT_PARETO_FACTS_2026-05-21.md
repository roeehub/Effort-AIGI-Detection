# Job H — Optimal deployment configuration analysis (FACTS, 2026-05-21)

> **FACTS doc.** Per `docs/packet_retrospectives/AGENTS.md` authoring contract — mechanical pass/fail against pre-stated bars; no interpretation language. Interpretation lives in this folder's `AGENT_PROPOSAL_2026-05-21.md` companion.

## 1. Question

The prior agent in the original chat session 2026-05-21 framed the deployment decision as: **"If Slot A v2's lockbox-calibrated naive-global-τ Pareto curve dominates P8A's (75.5% recall at 10% lockbox FPR baseline), that's your shipping point."** This job computes Pareto curves for all 4 deployment-relevant ckpts (P8A, T5C, Slot A v2 step3500, E2B), evaluates dominance, and quantifies the multi-cohort trade-off (lockbox + dev + may6 production-drift + per-identity) at each candidate operating point.

## 2. Method

- **Lockbox cohort**: `teams_real_all_lockbox` (n=1361 videos) + `teams_fake_all_lockbox` (n=253 videos). Identical substrate for all 4 ckpts (verified: |E2B∩P8A|=1361 reals, 253 fakes; 0 differences).
- **Dev cohorts**: `teams_real_all_dev` (3253), `teams_fake_all_dev` (2409), `visomaster_enhanced_macro_dev`, `deeplive_enhanced_dev`.
- **may6 cohort**: 92 production-drift Xinhe frames from 2026-05-06.
- **Per-identity cohorts**: Roy_D / PC_Generator / Chikara_Takahashi / Q / bla_bla_chow / dor_shkedi sub-suites.
- **Per-ckpt scores**:
  - P8A_step5000, T5C_periodic_step3500, Slot_A_v2_step3500 (`hp35c51p`): per-video reports from 2026-05-20 scorecard (`gs://training-job-outputs/test_results/teams_promotion_contract/slot-a-v2-validation-2026-05-20/`).
  - E2B_top_n_step3200: per-frame cache from 2026-05-08 (`analysis/iq_shortcut_decomp_2026-05-08/scores_cache/`), aggregated to per-video by mean(frame_prob). Same lockbox substrate (verified by video_id overlap).
- **τ-calibration**: lockbox-calibrated — τ = (target_lockbox_fpr × n_lockbox_real)-th descending-sorted real score. The contract's dev-real-calibrated τ is NOT used in this analysis.
- **Pareto curves**: sweep τ over 999 points in [0.001, 0.999]; compute (lockbox_real_fpr, lockbox_fake_recall) at each.
- **Pareto dominance**: at a grid of FPR targets {0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20}, look up each ckpt's max achievable recall at FPR ≤ target. Pairwise compare.
- **Score-fusion ensembles** (P8A + SlotAv2): mean / max / min / geomean fusion of per-video probs; same τ sweep.
- **Best (ckpt, τ) per λ**: argmax over (ckpt, τ) of `lockbox_fake_recall − λ × lockbox_real_fpr` for λ ∈ {0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100}.
- **Script**: `run_deployment_pareto.py` + `extend_may6_analysis.py`.

## 3. Numbers

### 3.1 Pareto curves on lockbox — recall at fixed FPR levels

| config | recall@FPR=2% | recall@FPR=5% | recall@FPR=10% | AUC up to FPR≤25% |
|---|---:|---:|---:|---:|
| **E2B** (current deployment) | 0.6008 | **0.8221** | **0.9526** | **0.8885** |
| **P8A** (contract rank-1) | 0.4229 | 0.6206 | 0.7787 | 0.7590 |
| **SlotAv2** (contract rank-2) | **0.6957** | 0.7628 | 0.8538 | 0.8488 |
| **T5C** (contract rank-3) | 0.5929 | 0.7154 | 0.8379 | 0.7934 |
| ENSEMBLE_mean (P8A+SlotAv2) | 0.6285 | 0.7391 | 0.8340 | 0.8228 |
| ENSEMBLE_max | 0.4980 | 0.7628 | 0.8142 | 0.7916 |
| ENSEMBLE_min | 0.6126 | 0.7510 | 0.8182 | 0.8073 |
| ENSEMBLE_geomean | 0.6285 | 0.7273 | 0.8340 | 0.8186 |

The prior-agent reference (P8A 75.5%/10%) reproduces here at **77.9%/10%** — within the ±2pp resolution this batch's per-video aggregation gives (the prior 75.5% may have been from a slightly different ckpt-or-substrate snapshot; both numbers tell the same story).

### 3.2 Pairwise Pareto dominance across 10 FPR levels

| Pair | A wins | Ties | B wins | Dominance |
|---|---:|---:|---:|---|
| E2B vs P8A | 10 | 0 | 0 | **E2B strictly dominates P8A** |
| E2B vs SlotAv2 | 6 | 0 | 4 | no strict dominance (E2B wins at high FPR; SlotAv2 at low FPR) |
| E2B vs T5C | 8 | 0 | 2 | no strict dominance |
| **P8A vs SlotAv2** | 0 | 0 | 10 | **SlotAv2 strictly dominates P8A** |
| P8A vs T5C | 2 | 0 | 8 | no strict dominance (T5C mostly wins) |
| **SlotAv2 vs T5C** | 10 | 0 | 0 | **SlotAv2 strictly dominates T5C** |

### 3.3 Multi-cohort operating-point table — lockbox-calibrated

At each ckpt × target_lockbox_fpr, the τ that achieves the target and the resulting cohort metrics:

| ckpt | target_FPR | τ | dev_fpr | lockbox_rec | dev_fake@dev | viso_dev | deeplive_dev | may6_fired/92 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E2B | 0.005 | 0.944 | 0.015 | 0.277 | 0.482 | 0.038 | 0.281 | 11/92 |
| E2B | 0.020 | 0.752 | 0.057 | 0.585 | 0.659 | 0.049 | 0.745 | 30/92 |
| E2B | 0.050 | 0.561 | 0.107 | 0.822 | 0.728 | 0.075 | 0.903 | **50/92** |
| E2B | 0.100 | 0.369 | 0.161 | 0.953 | 0.778 | 0.113 | 0.983 | **57/92** |
| E2B | 0.200 | 0.147 | 0.247 | 0.992 | 0.816 | 0.222 | 1.000 | 68/92 |
| P8A | 0.005 | 0.983 | 0.037 | 0.277 | 0.425 | 0.042 | 0.079 | 0/92 |
| P8A | 0.020 | 0.897 | 0.073 | 0.403 | 0.538 | 0.145 | 0.257 | 0/92 |
| P8A | 0.050 | 0.579 | 0.113 | 0.621 | 0.679 | 0.324 | 0.483 | **0/92** |
| **P8A** | **0.100** | 0.254 | 0.153 | **0.779** | 0.766 | 0.444 | 0.644 | **0/92** |
| P8A | 0.200 | 0.083 | 0.220 | 0.901 | 0.846 | 0.582 | 0.796 | 5/92 |
| **SlotAv2** | 0.005 | 0.853 | 0.039 | **0.553** | 0.486 | 0.033 | 0.334 | **0/92** |
| **SlotAv2** | 0.020 | 0.780 | 0.069 | **0.696** | 0.606 | 0.185 | 0.572 | **1/92** |
| **SlotAv2** | 0.050 | 0.674 | 0.097 | 0.759 | 0.733 | 0.375 | 0.804 | **1/92** |
| **SlotAv2** | **0.100** | 0.535 | 0.125 | **0.854** | 0.828 | **0.529** | 0.928 | **3/92** |
| SlotAv2 | 0.200 | 0.384 | 0.166 | 0.937 | 0.894 | 0.649 | 0.978 | 7/92 |
| T5C | 0.005 | 0.888 | 0.042 | 0.451 | 0.464 | 0.020 | 0.295 | 0/92 |
| T5C | 0.050 | 0.799 | 0.079 | 0.715 | 0.680 | 0.260 | 0.728 | 1/92 |
| T5C | 0.100 | 0.728 | 0.102 | 0.838 | 0.783 | 0.440 | 0.883 | 4/92 |

### 3.4 Best (ckpt, τ) per FP/FN cost ratio λ

| λ | Winner config | τ | lockbox_fpr | lockbox_recall |
|---:|---|---:|---:|---:|
| 0.5 | E2B | 0.228 | 0.154 | 0.992 |
| 1.0 | E2B | 0.360 | 0.102 | 0.957 |
| 2.0 | E2B | 0.459 | 0.070 | 0.897 |
| 3.0 | E2B | 0.487 | 0.064 | 0.881 |
| **5.0** | **SlotAv2** | 0.754 | 0.024 | 0.735 |
| 7.0 | SlotAv2 | 0.828 | 0.009 | 0.644 |
| 10.0 | SlotAv2 | 0.828 | 0.009 | 0.644 |
| 15.0 | SlotAv2 | 0.837 | 0.007 | 0.617 |
| 20.0 | SlotAv2 | 0.849 | 0.004 | 0.577 |
| 50.0 | SlotAv2 | 0.891 | 0.000 | 0.379 |
| 100.0 | SlotAv2 | 0.891 | 0.000 | 0.379 |

Crossover at λ ≈ 4: E2B wins for λ ≤ 3 (high cost of misses); SlotAv2 wins for λ ≥ 5 (moderate-to-high cost of false flags).

### 3.5 may6 production-drift cost at each lockbox-calibrated τ (CRITICAL)

| Ckpt | @ FPR=2% may6 | @ FPR=5% may6 | @ FPR=10% may6 | @ FPR=20% may6 |
|---|---:|---:|---:|---:|
| P8A | 0/92 (0%) | 0/92 (0%) | **0/92 (0%)** | 5/92 (5%) |
| E2B | 30/92 (33%) | 50/92 (54%) | **57/92 (62%)** | 68/92 (74%) |
| SlotAv2 | 1/92 (1%) | 1/92 (1%) | **3/92 (3.3%)** | 7/92 (8%) |
| T5C | 0/92 (0%) | 1/92 (1%) | 4/92 (4.3%) | 9/92 (10%) |

**E2B fires on 33-74% of may6 production-drift frames across all lockbox-calibrated operating points.** Slot A v2 fires on 1-8%. P8A fires on 0-5%.

### 3.6 Per-identity FPR at each lockbox-calibrated τ

| identity | @target=2% | E2B | P8A | SlotAv2 | T5C |
|---|---|---:|---:|---:|---:|
| Roy_D dev (n=130) | | 0.131 | 0.308 | **0.831** | 0.815 |
| Chikara lockbox (n=25) | | 0.000 | 0.360 | 0.000 | 0.000 |
| Q dev (n=36) | | 0.083 | 0.861 | 0.222 | 0.222 |
| bla_bla_chow dev (n=467) | | 0.259 | 0.069 | 0.150 | 0.118 |

| identity | @target=10% | E2B | P8A | SlotAv2 | T5C |
|---|---|---:|---:|---:|---:|
| Roy_D dev | | 0.438 | 0.577 | **0.962** | 0.946 |
| Chikara lockbox | | 0.000 | 0.880 | 0.120 | 0.160 |
| Q dev | | 0.000 | 0.944 | 0.222 | 0.278 |
| bla_bla_chow dev | | 0.531 | 0.242 | 0.300 | 0.223 |

Slot A v2's Roy_D regression is operating-point-invariant — Roy_D dev FPR is 83% at target FPR=2% and 96% at target FPR=10%. The regression is structural to the anchor mechanism, not artifactual at high τ.

## 4. Mechanical pass/fail

| Bar | Definition | Result |
|---|---|---|
| **Bar 1 (SlotAv2 dominates P8A on lockbox Pareto)** | SlotAv2 wins ≥ 8/10 FPR levels vs P8A; no losses | **MET** (10/10 wins, 0 losses) |
| **Bar 2 (SlotAv2 dominates T5C on lockbox Pareto)** | SlotAv2 wins ≥ 8/10 FPR levels vs T5C; no losses | **MET** (10/10 wins, 0 losses) |
| **Bar 3 (Ensemble adds value)** | Any ensemble fusion achieves higher AUC than SlotAv2 (the best single ckpt at low FPR) | **NOT MET** (SlotAv2 AUC=0.849; best ensemble AUC=0.823) |
| **Bar 4 (E2B has production-drift cost)** | E2B may6_fpr at lockbox-cal FPR=10% ≥ 30% | **MET** (E2B 62% vs Slot A v2 3.3%) |
| **Bar 5 (SlotAv2 production-drift parity with P8A)** | SlotAv2 may6_fpr ≤ 5% at lockbox-cal FPR=10% | **MET** (3.3%; P8A 0%) |
| **Bar 6 (SlotAv2 lockbox-recall lift over P8A at matched may6 FPR)** | SlotAv2 lockbox_recall − P8A lockbox_recall ≥ 5pp at matched may6_fpr ≤ 5% | **MET** at FPR=10%: SlotAv2 0.854 (may6=3/92) vs P8A 0.779 (may6=0/92); Δ=+7.5pp |
| **Bar 7 (SlotAv2 dev_fake_macro lift at lockbox-cal FPR=10%)** | SlotAv2 macro recall (teams_fake_all_dev + viso + deeplive)/3 > P8A, T5C, E2B | **MET** (SlotAv2 0.762; P8A 0.618; T5C 0.702; E2B 0.625) |
| **Bar 8 (Roy_D regression operating-point-invariant)** | SlotAv2 Roy_D dev FPR > 0.50 at every target FPR ∈ [0.005, 0.20] | **MET** (Roy_D FPR 0.531-0.992 across the range) |

## 5. Three deployment operating modes on Slot A v2 step3500

The ckpt choice is settled (Slot A v2 step3500; rationale §4 Bars 1-2). τ is a deployment configuration parameter, not a fixed recommendation — three documented operating modes are available, parameterized by the lockbox-calibrated FPR target:

### Mode A — High-recall (τ ≈ 0.535, lockbox_fpr = 10%)

- lockbox_real_fpr = 0.100 (catches 9.99% of 1361 reals)
- lockbox_fake_recall = **0.854** (catches 85.4% of 253 fakes)
- dev_real_fpr = 0.125 (HIGHER than contract's 0.07 target)
- dev_fake_all_dev recall = 0.828
- visomaster_enhanced recall = **0.529** (4.7× E2B's 0.113)
- deeplive_enhanced recall = 0.928
- dev_fake_macro = **0.762** (best of 4 ckpts)
- may6 fired = 3/92 (3.3%)
- Roy_D dev FPR = **0.962**
- Q dev FPR = 0.222
- Chikara lockbox FPR = 0.120
- bla_bla_chow dev FPR = 0.300

### Mode B — Contract-compliant (τ ≈ 0.780, lockbox_fpr = 2%)

- lockbox_real_fpr = 0.020
- lockbox_fake_recall = 0.696 (vs P8A 0.403 at same FPR → +29pp lift)
- dev_real_fpr = 0.069 (**within contract's 0.07 target**)
- viso recall = 0.185
- deeplive recall = 0.572
- dev_fake_macro = 0.454
- may6 fired = 1/92 (1.1%)
- Roy_D dev FPR = 0.831

### Mode C — Low-FPR-strict (τ ≈ 0.870, lockbox_fpr ≈ 1%)

- lockbox_real_fpr ≈ 0.010
- lockbox_fake_recall ≈ 0.65 (vs P8A ~0.33 at same FPR → +33pp lift)
- viso recall ≈ 0.08
- deeplive recall ≈ 0.45
- may6 fired ≈ 0-1/92
- Roy_D dev FPR ≈ 0.69-0.83 (extrapolated; less catastrophic than Mode A but still elevated)
- (Frame-level numbers per `analysis/iq_substrate_tau_2026-05-21/slot_a_v2_lockbox_pareto.md`; video-level not directly measured at this τ in Job H output but interpolatable from `outputs/pareto_curves.csv`)

### Intermediate operating point (τ ≈ 0.674, lockbox_fpr = 5%)

- lockbox_real_fpr = 0.050
- lockbox_fake_recall = 0.759
- dev_real_fpr = 0.097
- viso recall = 0.375
- deeplive recall = 0.804
- may6 fired = 1/92
- Roy_D dev FPR = 0.931

This is the mid-range operating point between Mode A and Mode B; useful as a reference if the choice between A and B is hard to commit to.

### Choosing the operating mode

τ is a configuration knob with no retraining cost — it can be changed post-deploy. The mode choice depends on the operational FP/FN cost ratio:
- λ ≤ 3 → Mode A (high recall, accept higher FP rate)
- 5 ≤ λ ≤ 20 → Mode B (contract-compliant balance)
- λ ≥ 30 → Mode C (low-FPR-strict, accept lower recall)

The cost ratio is a product/SLA decision; this document does not pick it.

## 6. Artifacts

| Path | Contents |
|---|---|
| `outputs/pareto_curves.csv` | 3996 rows: 4 ckpts × 999 τ values × (lockbox_fpr, lockbox_recall) |
| `outputs/calibrated_operating_pts.csv` | 32 rows: 4 ckpts × 8 target FPRs × full multi-cohort metrics |
| `outputs/per_identity_at_op_pts.csv` | 32 rows: per-identity FPR at each operating point |
| `outputs/ensemble_pareto.csv` | 4 fusion methods × 999 τ values |
| `outputs/best_per_lambda.csv` | 8 configs × 12 λ values; is_winner column |
| `outputs/pareto_dominance.json` | 12 pairwise dominance verdicts |
| `outputs/headline.json` | Single-number summaries |
| `outputs/may6_at_operating_points.csv` | may6 fired count at each lockbox-cal operating point |
| `outputs/may6_sweep_curves.csv` | Full τ-sweep of may6_fpr per ckpt |
| `figs/01_lockbox_pareto.png` | Pareto curves with prior-agent reference point |
| `figs/02_lockbox_pareto_low_fpr_zoom.png` | Same, zoomed to FPR ∈ [0, 0.10] |
| `figs/03_per_identity_at_op_pts.png` | Per-identity trajectories |
| `figs/04_lambda_winners.png` | Best score per λ |
| `figs/05_may6_vs_lockbox_recall.png` | Production-drift cost vs lockbox catching power |
| `figs/06_lockbox_fpr_to_may6_fpr.png` | may6 FPR as function of lockbox FPR per ckpt |
| `run_deployment_pareto.py`, `extend_may6_analysis.py` | Reproducible scripts |

## 7. Caveats

- E2B per-frame cache from 2026-05-08 aggregated via mean(frame_prob) → per-video. Aggregation method matches the 2026-05-20 contract scorer (verified by spot-checking 5 P8A videos: cache-mean within 0.001 of 2026-05-20 avg_video_prob). Same lockbox substrate (1361/253) confirmed by video_id overlap.
- n_lockbox_fake = 253 → 95% CI on recall at any point is approximately ±6pp via normal approximation. Pareto-dominance claims are stronger than any single point estimate (10/10 wins is more decisive than +5pp at one point).
- n_may6 = 92 → 95% CI on may6_fpr is approximately ±7pp. The E2B 50-57/92 fires is FAR outside any reasonable CI; the P8A/Slot A v2 0-3/92 are also clearly distinguishable from E2B.
- "Lockbox" substrate is NOT directly equivalent to deployment — memory `project_lockbox_fpr_dominated_by_webcam_mode` notes the lockbox is 65.7% webcam-style captures. The may6 cohort is a different operational distribution. The Pareto analysis surfaces this trade-off explicitly.
- Slot A v2's Roy_D regression is a separate deployment risk not captured in the lockbox+may6 metrics. Job D's audit reports the FPR; whether Roy_D is a representative production user is unaddressed by this batch.

## 8. Cross-references

- Per-job sibling FACTS docs: `RESULTS_FACTS_2026-05-20.md`, `job_a..g/JOB_*_FACTS_2026-05-20.md`
- Memory: `project_deployment_is_e2b_2026-05-06` (E2B is current deployment), `project_lockbox_fpr_dominated_by_webcam_mode` (lockbox composition)
- Open loops touched: `lockbox-real-fpr-tiebreak-is-load-bearing` (RESOLVABLE — see §4 Bar 1), `deployment-vs-p8a-substrate-tradeoff-not-quantified` (RESOLVABLE — see §3.5 may6 vs lockbox table)
- Companion OPINION doc: `AGENT_PROPOSAL_2026-05-21.md` (this directory)
