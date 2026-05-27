# RESULTS_FACTS_2026-05-16 — auto-mode anchor_aware + rebalance scorecard

> **FACTS only.** Scorecard verdict for the auto-mode 2026-05-16 packets.

## §1. Provenance

- Scorecard Vertex job `6605151520717537280` (us-east1, image `1.3.293`),
  submitted 2026-05-16 19:45 UTC, SUCCEEDED 2026-05-16 22:46 UTC (~3h runtime).
- 6 ckpts × 9-suite minimal manifest = 54 cells.
- v3-fix policy (target_real_fpr=0.07, target_stress_fpr=0.10, target_fake_recall_min=0.30).
- Output: `gs://training-job-outputs/test_results/teams_promotion_contract/auto-mode-scorecard-2026-05-16/`.

## §2. Promotion contract ranking

| rank | ckpt | τ | dev_macro | lockbox_FPR | viso_enh | deeplive_enh | teams_fake_dev | teams_fake_lockbox |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | P8A_REFERENCE_STEP5000 | 0.916 | 0.300 | **0.0184** | 0.136 | 0.239 | 0.526 | 0.387 |
| **2** | **SLOT_A_ANCHOR_AWARE_STEP3500** | **0.788** | **0.438** | **0.0191** | **0.167** | **0.552** | **0.595** | **0.688** |
| 3 | T5C_PERIODIC_STEP3500 | 0.831 | 0.459 | 0.0279 | 0.138 | 0.626 | 0.613 | 0.660 |
| 4 | SLOT_B_REAL_REBAL_STEP3500 | 0.840 | 0.541 | 0.0896 | 0.158 | 0.795 | 0.670 | 0.435 |
| 5 (gate1 fail) | SLOT_A_ANCHOR_AWARE_STEP1500 | 0.930 | 0.207 | 0.0044 | 0.029 | 0.134 | 0.459 | 0.640 |
| 6 (gate1 fail) | SLOT_B_REAL_REBAL_STEP1500 | 0.957 | 0.229 | 0.0140 | 0.011 | 0.218 | 0.459 | 0.162 |

P8A holds rank 1 by `lockbox_real_fpr` tiebreak (0.0184 vs Slot A 0.0191 — 0.07pp). All four rank-1-to-4 ckpts pass all gates.

## §3. Slot A (anchor_aware) deltas vs P8A

| metric | P8A | Slot A v2 step3500 | Δ absolute | Δ relative |
|---|---:|---:|---:|---:|
| dev_fake_macro_recall | 0.300 | **0.438** | **+0.138** | **+46.0%** |
| visomaster_enhanced_dev | 0.136 | **0.167** | +0.031 | **+22.8%** |
| deeplive_enhanced_dev | 0.239 | **0.552** | **+0.313** | **+131%** |
| teams_fake_all_dev | 0.526 | 0.595 | +0.069 | +13.1% |
| teams_fake_all_lockbox | 0.387 | **0.688** | **+0.301** | **+77.7%** |
| lockbox_real_fpr | 0.0184 | 0.0191 | +0.0007 | +3.8% |
| dev_primary_real_fpr | 0.069 | 0.065 | −0.004 | −5.8% |

Slot A v2 outperforms P8A on every fake-recall metric (massively on deeplive
and teams_fake_lockbox) at +0.07pp cost on lockbox FPR. Same suite gates
passed.

## §4. Per-identity lockbox over-fire (where the FPR comes from)

| identity | n | P8A | Slot A v2 | T5C | Slot B |
|---|---:|---:|---:|---:|---:|
| Chikara_Takahashi | 42 | **26.2%** | **0%** | 0% | 11.9% |
| PC_Generator | 29 | **27.6%** | **0%** | 0% | 3.4% |
| bla_bla_chow | 68 | 0% | **16.2%** | 17.6% | 0% |
| dor_shkedi | 1170 | 0.7% | 1.6% | 2.7% | **10.6%** |
| real_dor | 109 | 0% | 0% | 0% | 0% |

Slot A v2 ELIMINATED two chronic-FP identities (Chikara_Takahashi 26→0%, PC_Generator 28→0%) but INTRODUCED a new chronic-FP on bla_bla_chow (0→16%). Net lockbox FPR is +0.07pp on P8A despite the tradeoff because the recovered identities are smaller cohorts than dor_shkedi.

Slot B traded the same direction differently: improved Chikara_Takahashi + PC_Generator partially but blew up dor_shkedi (1→11%).

## §5. Per-identity dev_PNG over-fire (Roy_D)

| identity | n | P8A | Slot A v2 | T5C | Slot B |
|---|---:|---:|---:|---:|---:|
| **Roy_D** | 130 | **29.2%** | **81.5%** | 86.2% | 76.9% |
| PC_Generator | 835 | 24.1% | 6.8% | 6.1% | 6.9% |
| Q | 54 | 88.9% | 16.7% | 16.7% | 16.7% |
| bla_bla_chow | 491 | 6.3% | 15.1% | 13.0% | 12.8% |
| xiang | 159 | 0% | 4.4% | 4.4% | 4.4% |

Roy_D is the structural ceiling for Slot A and Slot B — both regress Roy_D
dramatically vs P8A (29→81% and 29→77%). This was predicted by the encoder
probe: Roy_D's encoder representation is essentially unchanged by either
intervention. The improvement on PC_Generator (24→7%) and Q (89→17%)
generalizes from the dor anchor pool supervision but does NOT extend to Roy_D.

## §6. Encoder probe cross-reference

| cohort | T5C base | Slot β baseline | Slot A v2 | Slot B |
|---|---:|---:|---:|---:|
| anchor_pool (Slot A target) | 53.3% | 13.3% | **0.0%** | 46.7% |
| dor_shkedi.png | 59.7% | 67.8% | **8.6%** | 73.8% |
| VCD (Slot B target) | 27.0% | 18.0% | 16.0% | 17.0% |
| Roy_D | 100% | 97.7% | 98.5% | 99.2% |

Slot A's anchor_pool collapse to 0% Roy_D-adjacency directly translated to
the per-identity wins (Chikara_Takahashi, PC_Generator). Slot B's lack of
encoder movement (VCD 27→17%) → no real-side improvement.

## §7. Trade-off summary

Slot A v2 vs P8A: PRO-FAKE-RECALL trade.
- Wins: viso +0.031, deeplive +0.313, teams_fake_lockbox +0.301, dev_macro +0.138
- Loss: 0.07pp lockbox FPR (tiebreak)
- Mechanism: anchor_aware pulled dor false-flag pool to clean cluster,
  partly generalized to Chikara_Takahashi + PC_Generator + Q

Slot B vs P8A: NET-LOSS at floor + tiebreak.
- Wins: dev_macro +0.241, deeplive +0.556 (huge fake recall)
- Loss: lockbox FPR +0.071 (4.8× worse), bla_bla_chow + dor_shkedi regressed
- Mechanism: family weight rebalance moved VCD reals toward clean side,
  not toward Roy_D — encoder probe predicted this exact outcome

## §8. Output files

- `scorecard_outputs/promotion_contract/checkpoint_summary.csv`
- `scorecard_outputs/promotion_contract/promotion_winner.json` (P8A)
- `scorecard_outputs/promotion_contract/selected_threshold_scorecard.csv`
- `scorecard_outputs/teams_real_*_frames_report.csv` (6 ckpts × 2 suites)
- Encoder probe: `analysis/slot_b_property_shortcut_2026-05-16/outputs/encoder_probe_4ckpts_2026-05-16.log`
