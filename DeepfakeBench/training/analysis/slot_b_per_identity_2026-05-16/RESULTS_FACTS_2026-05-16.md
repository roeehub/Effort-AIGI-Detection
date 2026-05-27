# RESULTS_FACTS_2026-05-16 — Slot β per-identity lockbox decomposition + tiebreak re-eval

> **FACTS only.** Numerical results from the three CPU diagnostics run after
> the RESCHAIN_GRL6 scorecard verdict. No interpretation.
>
> Origin: post-scorecard CPU follow-up authorized 2026-05-16 to test three
> open loops opened in the RESCHAIN_GRL6 packet retro
> (`docs/packet_retrospectives/packets/RESCHAIN_GRL6.md`).

## §1. Provenance

- Per-frame inputs pulled from `gs://training-job-outputs/test_results/teams_promotion_contract/overnight-scorecard-2026-05-16/reports/teams_real_all_lockbox_*_frames_report.csv` (3 ckpts: P8A, T5C, Slot β).
- Suite: `teams_real_all_lockbox` — 5 identities, 1418 frames total (`Chikara_Takahashi`, `PC_Generator`, `bla_bla_chow`, `dor_shkedi`, `real_dor`).
- τ values from `checkpoint_summary.csv` (selected_threshold column).
- Scripts: `analysis/slot_b_per_identity_2026-05-16/scripts/{decompose_per_identity,tiebreak_reranking,rule_sensitivity_probe}.py`.

## §2. Per-identity decomposition of lockbox over-fires (Step 1)

Each row = identity. `n_overfires` = frames with `frame_prob ≥ τ_cal`.

### P8A (τ=0.916, 27 over-fires total, FPR=0.0190)

| identity | n_frames | n_overfires | frac_over | p95_score | is_chronic6 |
|---|---:|---:|---:|---:|---|
| Chikara_Takahashi | 42 | 11 | 0.262 | 0.989 | no |
| PC_Generator | 29 | 8 | 0.276 | 0.991 | yes |
| dor_shkedi | 1170 | 8 | 0.007 | 0.354 | yes |
| bla_bla_chow | 68 | 0 | 0 | 0.109 | yes |
| real_dor | 109 | 0 | 0 | 0.084 | no |

Chronic-6 share: 16/27 = **59.3%**. Non-chronic share: 11/27 (all `Chikara_Takahashi`).

### T5C step3500 (τ=0.831, 44 over-fires total, FPR=0.0310)

| identity | n_frames | n_overfires | frac_over | p95_score | is_chronic6 |
|---|---:|---:|---:|---:|---|
| dor_shkedi | 1170 | 32 | 0.027 | 0.797 | yes |
| bla_bla_chow | 68 | 12 | 0.176 | 0.861 | yes |
| Chikara_Takahashi | 42 | 0 | 0 | 0.814 | no |
| PC_Generator | 29 | 0 | 0 | 0.566 | yes |
| real_dor | 109 | 0 | 0 | 0.411 | no |

Chronic-6 share: 44/44 = **100%**.

### SLOT_B step3500 (τ=0.816, 124 over-fires total, FPR=0.0874)

| identity | n_frames | n_overfires | frac_over | p95_score | is_chronic6 |
|---|---:|---:|---:|---:|---|
| dor_shkedi | 1170 | 116 | 0.099 | 0.846 | yes |
| bla_bla_chow | 68 | 8 | 0.118 | 0.825 | yes |
| Chikara_Takahashi | 42 | 0 | 0 | 0.579 | no |
| PC_Generator | 29 | 0 | 0 | 0.591 | yes |
| real_dor | 109 | 0 | 0 | 0.346 | no |

Chronic-6 share: 124/124 = **100%**. Two identities account for all over-fires: `dor_shkedi` (116) and `bla_bla_chow` (8).

## §3. Tiebreak re-eval (Step 2)

v3-fix policy lex gates: (1) dev_fake_macro_recall ≥ 0.30, (2) dev_worst_stress_fpr ≤ 0.10, (3) dev_primary_real_fpr ≤ 0.07, (4) tiebreak. All 4 passing ckpts clear gates 1-3. The tiebreak determines rank.

| rank by lockbox_real_fpr (orig) | ckpt | dev_fake_macro_recall | three_suite_sum | lockbox_real_fpr |
|---:|---|---:|---:|---:|
| 1 | P8A | 0.300 (at floor) | 0.901 | 0.018 |
| 2 | T5C | 0.459 | 1.377 | 0.028 |
| 3 | Slot α step1500 | 0.443 | 1.328 | 0.046 |
| 4 | Slot β step3500 | 0.545 | 1.636 | 0.088 |
| 5 (gate1 fail) | Slot α step3500 | 0.226 | 0.677 | 0.015 |

| rank by dev_fake_macro_recall desc | ckpt | dev_fake_macro_recall | lockbox_real_fpr |
|---:|---|---:|---:|
| 1 | Slot β step3500 | 0.545 | 0.088 |
| 2 | T5C | 0.459 | 0.028 |
| 3 | Slot α step1500 | 0.443 | 0.046 |
| 4 | P8A | 0.300 | 0.018 |

| rank by three_suite_sum desc | ckpt | three_suite_sum | dev_fake_macro_recall | lockbox_real_fpr |
|---:|---|---:|---:|---:|
| 1 | Slot β step3500 | 1.636 | 0.545 | 0.088 |
| 2 | T5C | 1.377 | 0.459 | 0.028 |
| 3 | Slot α step1500 | 1.328 | 0.443 | 0.046 |
| 4 | P8A | 0.901 | 0.300 | 0.018 |

Under both alternative tiebreaks, ranks (1, 2, 3, 4) flip to (Slot β, T5C, Slot α step1500, P8A). P8A sits at the dev_fake_macro_recall floor (0.3003 vs 0.30 required).

## §4. Rule sensitivity probe (Step 3)

Per-identity Option-3 rule (memory `project_blend_unsharp_lever_2026-05-14`):
- Default: `frac_above_0.6 ≥ 0.4 AND count_above_0.9 ≥ 1 ⇒ identity FAKE`
- Applied as a deployment suppression layer: a frame fires FAKE iff (frame_prob ≥ τ) AND (rule says identity FAKE).

Identity-level rule verdicts on the 5 lockbox real identities:

| identity | P8A rule_says_fake | T5C rule_says_fake | Slot β rule_says_fake |
|---|---|---|---|
| Chikara_Takahashi | yes (frac=0.74, count=12) | no (frac=0.17, count=0) | no (frac=0.05, count=0) |
| PC_Generator | yes (frac=0.72, count=10) | no (frac=0.03, count=0) | no (frac=0.07, count=0) |
| dor_shkedi | no (frac=0.02, count=8) | no (frac=0.28, count=4) | **yes (frac=0.505, count=1)** |
| bla_bla_chow | no (frac=0, count=0) | no (frac=0.63, count=0) | no (frac=0.56, count=0) |
| real_dor | no (frac=0, count=0) | no (frac=0, count=0) | no (frac=0, count=0) |

Slot β dor_shkedi is the only mis-flagged real identity. It sits right at the decision boundary: frac_above_0.6 = 0.505 vs threshold 0.4; max_score = 0.909 with exactly 1 frame strictly above 0.9.

Suppression-mode post-rule FPR on the 1418-frame lockbox real cohort, under 4 rule-threshold variants:

| variant | P8A FPR | T5C FPR | Slot β FPR | Slot β rule-says-REAL count |
|---|---:|---:|---:|---:|
| Default (`frac≥0.4 + count_0.9≥1`) | 1.34% | 0.00% | **8.18%** | 4/5 |
| A: `count_0.92≥1` (tighter extreme) | 1.34% | 0.00% | **0.00%** | 5/5 |
| B: `frac≥0.55` (tighter bulk) | 1.34% | 0.00% | **0.00%** | 5/5 |
| C: `count_0.95≥1` (loosest extreme) | 1.34% | 0.00% | **0.00%** | 5/5 |

All three threshold-tightening variants tested rescue Slot β to 0% lockbox real FPR. None changes the verdict for P8A or T5C on this 5-identity sample.

Note: post-rule P8A FPR (1.34%) is reduced from baseline 1.90% by suppressing 8 over-fires on `dor_shkedi` (rule correctly says REAL). The remaining 19 over-fires are on `Chikara_Takahashi` (11) + `PC_Generator` (8) — the rule says FAKE on both (correctly characterizing the score distribution), so the rule does NOT suppress them.

## §5. Outputs

- `outputs/per_identity_{p8a,t5c,slot_b}.csv` — full per-identity breakdowns
- `outputs/summary_chronic6_share.csv` — chronic-6 concentration summary
- `outputs/per_identity_rule_verdicts.csv` — default-rule classification per identity per ckpt
- `outputs/rule_sensitivity_variants.csv` — 4 rule-threshold variants × 3 ckpts
- `outputs/rerank_orig_lockbox_fpr.csv` — original v3-fix ranking
- `outputs/rerank_alt_dev_recall.csv` — alt tiebreak (dev_fake_macro_recall)
- `outputs/rerank_alt_three_suite_sum.csv` — alt tiebreak (3-fake-suite sum)
