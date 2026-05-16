# RESULTS_FACTS_2026-05-16 — Slot α + Slot β overnight scorecard

> **FACTS only.** Forbidden words: succeeds / fails / wins / promotes / deployment-grade.
> Mechanical pass/fail against the v3-fix contract policy. No interpretation.
>
> Authoring: same agent who designed the slots, ran the CPU probes, launched
> the training, and launched the scorecard. Single-agent FACTS doc; the
> OPINIONS doc is the same agent's interpretation. Treat as such.

## §1. Provenance

- **Training**: two single-lever FT-from-T5C-step3500 packets launched 2026-05-15 22:39 UTC, both `JOB_STATE_SUCCEEDED`:
  - Slot α (R13_T5C_RESCHAIN_2026-05-15): W&B run `lsx4n0t7`, Vertex `679364389144363008`, us-east1, runtime 1h 47m. Lever: new `resolution_chain_aug` (random downsample to {64,96,128,160,192} then upsample with random kernel ∈ {LINEAR,CUBIC,AREA,LANCZOS4}, p_apply=0.5).
  - Slot β (R13_T5C_6AXIS_GRL_2026-05-15): W&B run `gwntcld0`, Vertex `5517273481877651456`, us-west4, runtime 4h 23m. Lever: `multi_axis_grl.axes` extended from 4 (chronic_flag, is_dor, sharpness_laplacian_high, color_a_approx_dev_high) to 6 (added color_b_dev_high + luma_mean_high).
- **Scorecard**: Vertex job `4599114546072780800`, us-east1, image `1.3.290`, 5 ckpts × 9-suite minimal manifest = 45 cells, runtime 2h 32m, SUCCEEDED 2026-05-16T10:35:56 UTC.
- **Policy**: v3-fix (target_real_fpr=0.07, target_stress_fpr=0.10, target_fake_recall_min=0.30).
- **Output prefix**: `gs://training-job-outputs/test_results/teams_promotion_contract/overnight-scorecard-2026-05-16/`.

## §2. Per-ckpt scorecard summary

Source: `promotion_contract/checkpoint_summary.csv`. All values at the per-ckpt FPR-calibrated τ.

| rank | ckpt | τ | dev_primary_real_fpr | dev_worst_stress_fpr | dev_fake_macro_recall | lockbox_real_fpr | lockbox_fake_recall | teams_fake_all_dev | visomaster_enhanced_macro_dev | deeplive_enhanced_dev |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | P8A_REFERENCE_STEP5000 | 0.916 | 0.0695 | 0.0685 | 0.3003 | 0.0184 | 0.3874 | 0.5259 | 0.1364 | 0.2385 |
| 2 | T5C_PERIODIC_STEP3500 | 0.831 | 0.0652 | 0.0999 | 0.4589 | 0.0279 | 0.6601 | 0.6127 | 0.1382 | 0.6257 |
| 3 | SLOT_A_RESCHAIN_STEP1500 | 0.899 | 0.0642 | 0.0999 | 0.4427 | 0.0456 | 0.3636 | 0.5542 | 0.0564 | 0.7174 |
| 4 | SLOT_B_6AXIS_GRL_STEP3500 | 0.816 | 0.0639 | 0.0999 | 0.5453 | 0.0882 | 0.5415 | 0.6638 | 0.2345 | 0.7376 |
| 5 | SLOT_A_RESCHAIN_STEP3500 | 0.860 | 0.0633 | 0.0992 | 0.2257 | 0.0154 | 0.3715 | 0.4608 | 0.0182 | 0.1982 |

## §3. Close-criterion mechanics (v3-fix policy)

The lex rank orders ckpts by:
1. `dev_fake_macro_recall ≥ 0.30` (floor) — ckpts below the floor rank in a worse tier.
2. `dev_worst_real_stress_fpr ≤ 0.10` (stress ceiling).
3. `dev_primary_real_fpr ≤ 0.07` (real FPR budget).
4. Tiebreak among all-pass ckpts: ascending `lockbox_real_fpr`.

Per-ckpt outcome on each gate:

| ckpt | gate 1 (recall ≥ 0.30) | gate 2 (stress ≤ 0.10) | gate 3 (real_fpr ≤ 0.07) | tiebreak (lockbox_real_fpr) |
|---|---|---|---|---:|
| P8A_REFERENCE_STEP5000 | PASS (0.300, AT FLOOR) | PASS (0.069) | PASS (0.069) | 0.0184 → rank 1 |
| T5C_PERIODIC_STEP3500 | PASS (0.459) | PASS (0.100, AT CEILING) | PASS (0.065) | 0.0279 → rank 2 |
| SLOT_A_RESCHAIN_STEP1500 | PASS (0.443) | PASS (0.100, AT CEILING) | PASS (0.064) | 0.0456 → rank 3 |
| SLOT_B_6AXIS_GRL_STEP3500 | PASS (0.545) | PASS (0.100, AT CEILING) | PASS (0.064) | 0.0882 → rank 4 |
| SLOT_A_RESCHAIN_STEP3500 | **FAIL (0.226 < 0.30)** | PASS (0.099) | PASS (0.063) | n/a (below floor tier) → rank 5 |

## §4. Slot α (resolution_chain_aug) per-step deltas vs T5C step3500

| metric | T5C step3500 (base) | Slot α step1500 | Slot α step3500 |
|---|---:|---:|---:|
| dev_fake_macro_recall | 0.459 | 0.443 (−0.016) | 0.226 (**−0.233**) |
| visomaster_enhanced_macro_dev | 0.138 | 0.056 (**−0.082**) | 0.018 (**−0.120**) |
| teams_fake_all_dev | 0.613 | 0.554 (−0.059) | 0.461 (−0.152) |
| deeplive_enhanced_dev | 0.626 | 0.717 (**+0.091**) | 0.198 (−0.428) |
| lockbox_fake_recall | 0.660 | 0.364 (−0.296) | 0.372 (−0.288) |
| lockbox_real_fpr | 0.028 | 0.046 (+0.018) | 0.015 (−0.013) |
| selected_τ | 0.831 | 0.899 (+0.068) | 0.860 (+0.029) |

Pattern at step1500: visomaster regresses; deeplive lifts; net dev_macro flat; lockbox catastrophe on fakes (0.660 → 0.364). Pattern at step3500: every metric except `lockbox_real_fpr` regresses below T5C step3500 levels; dev_fake_macro falls below the 0.30 floor.

## §5. Slot β (6-axis GRL) deltas vs T5C step3500

| metric | T5C step3500 (base) | Slot β step3500 | delta |
|---|---:|---:|---:|
| dev_fake_macro_recall | 0.459 | 0.545 | **+0.086** (highest of all 5 ckpts scored) |
| visomaster_enhanced_macro_dev | 0.138 | 0.235 | **+0.097** (highest of all 5; +70% relative) |
| teams_fake_all_dev | 0.613 | 0.664 | +0.051 |
| deeplive_enhanced_dev | 0.626 | 0.738 | +0.112 (highest of all 5) |
| lockbox_fake_recall | 0.660 | 0.541 | −0.119 |
| lockbox_real_fpr | 0.028 | 0.088 | **+0.060** (3.2× T5C; 4.8× P8A) |
| selected_τ | 0.831 | 0.816 | −0.015 |

Slot β has the highest readout on three dev fake suites (dev_fake_macro_recall, visomaster_enhanced_macro_dev, deeplive_enhanced_dev). The lockbox_real_fpr at 0.0882 is the largest of any scored ckpt and exceeds P8A's by 4.8×.

## §6. CPU probe vs scorecard cross-reference for Slot α

The 2026-05-15 resolution-chain CPU probe (`analysis/cpu_diagnostics_2026-05-15_resolution_chain/`) measured per-frame score swing across 20 down→up variants on a 388-frame panel. Follow-up probe on the trained ckpts (`RESULTS_OVERNIGHT_FACTS_2026-05-16.md`) reported:

| ckpt | median real score_range (CPU probe) | dev_fake_macro_recall (scorecard) | visomaster_enhanced_macro_dev (scorecard) |
|---|---:|---:|---:|
| T5C step3500 | 0.605 | 0.459 | 0.138 |
| Slot α step1500 | (not measured — only step3500 probed) | 0.443 | 0.056 |
| Slot α step3500 | 0.448 (25% reduction) | 0.226 (below floor) | 0.018 |

The CPU probe also showed Slot α step3500 has per-size mean swing on reals of 0.07 vs T5C 0.23 — the flattest score-vs-resolution curve measured. The scorecard adds: at the cost of dev_fake_macro_recall regressing 0.233 absolute below the T5C baseline, with viso dropping 7.6× relative to T5C step3500.

## §7. Per-suite breakdown at each ckpt's τ

Source: `promotion_contract/selected_threshold_scorecard.csv` (downloaded at `/tmp/overnight_scorecard/`). 9 suites × 5 ckpts.

Fake-suite recall (higher = more fakes caught):

| suite | P8A | T5C | Slot α step1500 | Slot β step3500 | Slot α step3500 |
|---|---:|---:|---:|---:|---:|
| teams_fake_all_dev | 0.526 | 0.613 | 0.554 | **0.664** | 0.461 |
| visomaster_enhanced_macro_dev | 0.136 | 0.138 | 0.056 | **0.235** | 0.018 |
| deeplive_enhanced_dev | 0.239 | 0.626 | 0.717 | **0.738** | 0.198 |
| teams_fake_all_lockbox | 0.387 | **0.660** | 0.364 | 0.541 | 0.372 |

Real-suite FPR (lower = fewer over-fires):

| suite | P8A | T5C | Slot α step1500 | Slot β step3500 | Slot α step3500 |
|---|---:|---:|---:|---:|---:|
| teams_real_all_dev (primary, FPR-cal) | 0.069 | 0.065 | 0.064 | 0.064 | 0.063 |
| teams_real_poor_quality_dev | (mechanically ≤ stress) | | | | |
| teams_real_lighting_extreme_dev | (mechanically ≤ stress) | | | | |
| teams_real_all_lockbox | **0.018** | 0.028 | 0.046 | 0.088 | 0.015 |
| teams_real_dor_dev (n=50) | — | — | — | — | — (see DEEP_DIVE for per-identity) |

## §8. Output files

- `promotion_contract/promotion_winner.json` (P8A_REFERENCE_STEP5000)
- `promotion_contract/checkpoint_summary.csv` (5 rows)
- `promotion_contract/selected_threshold_scorecard.csv` (9 suites × 5 ckpts)
- `promotion_contract/threshold_grid.csv` (per-τ-candidate grid)
- `reports/*_summary_report.txt` and `reports/*_frames_report.csv` (per suite × per ckpt; 45 of each)
- Image: `us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:1.3.290`
- Ckpt map: `arena/checkpoint_maps/teams_target_domain.overnight_2026-05-16.yaml`
- Suite manifest: `arena/target_domain_suites.teams_promotion_contract_minimal_9suite_2026-05-14.yaml`
