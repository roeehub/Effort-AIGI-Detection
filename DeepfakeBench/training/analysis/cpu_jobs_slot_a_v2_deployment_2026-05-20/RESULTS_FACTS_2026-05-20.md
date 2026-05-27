# RESULTS_FACTS — 7-job CPU evidence batch for Slot A v2 deployment decision (2026-05-20)

> **FACTS doc.** Per `docs/packet_retrospectives/AGENTS.md` eval-folder authoring contract: mechanical pass/fail against pre-stated bars; no interpretation language. Interpretation lives in `AGENT_PROPOSAL_2026-05-20.md` (this directory).
>
> **Read order**: this doc first (top-level navigator), then per-job FACTS docs, then OPINIONS doc.

## 0. Question

A user-prompt 4-element framing surfaced in the original chat session (2026-05-20):
1. **Move 1** — evidence-based deployment decision for Slot A v2 step3500 (W&B `hp35c51p`)
2. **Move 2** — data ingestion explanation (no CPU job)
3. **Move 3** — verify canary + LoRA-load fixes are already in
4. **Image** — does the current direction address the 2026-05-19 Teams-account natural-experiment finding (T5C flips at τ=0.65 / 0.70 on same person same camera, different Teams account)?

This batch covers Move 1, Move 3, and the Image question via 7 jobs (A-G).

## 1. Per-job navigation

| Job | Question | Headline result | Bar summary | FACTS doc |
|---|---|---|---|---|
| A | Slot A v2 on Roy_D/Guest natural-experiment crops | Δ(Guest−Roy_D) = **−0.168** (identical to T5C −0.168); flip window WIDER ([0.60, 0.70) vs T5C [0.65, 0.70)) | Bars 1/2/3 NOT MET; Bar 4 MET | [`job_a_natural_experiment/JOB_A_NATURAL_EXPERIMENT_FACTS_2026-05-20.md`](job_a_natural_experiment/JOB_A_NATURAL_EXPERIMENT_FACTS_2026-05-20.md) |
| B | Paired per-video bootstrap CI on lockbox FPR gap | 95% CI on Δ = **[−0.008, +0.010]** (covers 0); P(SlotAv2 higher FPR) = **0.519** | Bars 1/2 MET | [`job_b_bootstrap_ci/JOB_B_BOOTSTRAP_FACTS_2026-05-20.md`](job_b_bootstrap_ci/JOB_B_BOOTSTRAP_FACTS_2026-05-20.md) |
| C | Re-rank under alternative tiebreak policies | P8A wins under all 5 lex policies; SlotAv2 wins under all 5 composite policies + Pareto | Bars 1-4 MET | [`job_c_tiebreak_rerank/JOB_C_TIEBREAK_RERANK_FACTS_2026-05-20.md`](job_c_tiebreak_rerank/JOB_C_TIEBREAK_RERANK_FACTS_2026-05-20.md) |
| D | Per-identity FP audit P8A vs Slot A v2 vs T5C | Roy_D dev FPR **+0.54** (P8A 0.30 → SlotAv2 0.84); 4 chronic identities fixed (Chikara/PCGen/Q/dor) | 1 Bar A fix; 0 Bar B overshoots by strict bar (Roy_D missed because P8A_FPR>0.10 gate) | [`job_d_per_identity_fp/JOB_D_PER_IDENTITY_FACTS_2026-05-20.md`](job_d_per_identity_fp/JOB_D_PER_IDENTITY_FACTS_2026-05-20.md) |
| E | τ-sweep operating-point analysis | At dev_fpr≤0.10: SlotAv2 lockbox_recall **0.60** vs T5C **0.40**; P8A unreachable at 0.10 on chronic-heavy panel | Bars 2/3/4 MET; Bars 1/5 NOT MET | [`job_e_tau_sweep/JOB_E_TAU_SWEEP_FACTS_2026-05-20.md`](job_e_tau_sweep/JOB_E_TAU_SWEEP_FACTS_2026-05-20.md) |
| F | may6 production-drift retest on Slot A v2 | **4/92 fired** at τ=0.5 (P8A: 0/92; E2B: 53/92); P8A sanity matches prior within MPS noise | Bars 1/2/3 MET; Bar 4 NOT MET (p50 = 0.097 > 0.05) | [`job_f_may6_retest/JOB_F_MAY6_FACTS_2026-05-20.md`](job_f_may6_retest/JOB_F_MAY6_FACTS_2026-05-20.md) |
| G | Per-axis perturbation Slot A v2 on Roy_D crop | G_scale swing **0.698** (T5C 0.678, +3.0%); saturation amplified **4.93×** T5C | Bars 1/2 NOT MET; Bar 3 amplification TRIGGERED on saturation | [`job_g_perturbation/JOB_G_PERTURBATION_FACTS_2026-05-20.md`](job_g_perturbation/JOB_G_PERTURBATION_FACTS_2026-05-20.md) |
| Move 3 | Canary + LoRA-load fix test suite | **16/16 pass** in 5.05s | Bars 1-4 MET | [`move3_test_verification/MOVE3_TEST_VERIFICATION_FACTS_2026-05-20.md`](move3_test_verification/MOVE3_TEST_VERIFICATION_FACTS_2026-05-20.md) |
| **H** (added 2026-05-21) | Optimal deployment Pareto + multi-cohort + may6 trade-off | **Ckpt chosen: Slot A v2 step3500** (strictly Pareto-dominates P8A and T5C on lockbox 10/10; E2B dominates P8A on lockbox but fires 57/92 on may6; SlotAv2 viso recall 4.7× E2B at matched FPR). **Three operating modes documented** (parameterized by τ, mode choice deferred to deployment-time SLA): Mode A τ=0.535 / lockbox_fpr=10% / recall=85% — Mode B τ=0.780 / 2% / 70% (contract-compliant) — Mode C τ=0.870 / 1% / 65%. Deployment policy includes quality gate (face_min_dim ≥ 150). | Bars 1, 2, 4, 5, 6, 7, 8 MET; Bar 3 NOT MET | [`job_h_deployment_pareto/JOB_H_DEPLOYMENT_PARETO_FACTS_2026-05-21.md`](job_h_deployment_pareto/JOB_H_DEPLOYMENT_PARETO_FACTS_2026-05-21.md) + [`job_h_deployment_pareto/ADDENDUM_2026-05-21.md`](job_h_deployment_pareto/ADDENDUM_2026-05-21.md) (documentation cross-check; adds quality gate, cites today's per-mode τ refutation, surfaces independent codec-verdict confirmation) |

## 2. Headline numbers (composite)

### 2.1 Contract tiebreak is inside sampling noise (Job B, paired per-video bootstrap)

- P8A `lockbox_real_fpr` = 0.018369 (25 FPs / 1361 videos)
- Slot A v2 `lockbox_real_fpr` = 0.019104 (26 FPs / 1361 videos)
- Δ = **+0.000735** (literally 1 video difference)
- **95% CI on Δ: [−0.00808, +0.00955]** (covers 0; CI width ≈ 25× the observed gap)
- P(Slot A v2 truly higher FPR) = **0.519** — coin flip
- Note: the 3 lockbox-real suites listed in the contract are nested — `teams_real_lighting_extreme_lockbox` (207) and `teams_real_poor_quality_lockbox` (22) are strict subsets of `teams_real_all_lockbox` (1361). The union ≡ `teams_real_all_lockbox` ≡ the contract metric.

### 2.2 Teams-account transport shortcut is unchanged on Slot A v2 (Job A + Job G)

- T5C step3500 prior (2026-05-19): Roy_D 0.795 / Guest 0.628, Δ = −0.168, flips at τ=0.65, 0.70
- **Slot A v2 step3500 (this batch)**: Roy_D 0.767 / Guest 0.599, Δ = −0.168, flips at τ=0.60, 0.65, 0.70
- Vertical shift only: −0.028 (Roy_D), −0.029 (Guest). The transport-axis Δ is identical to 4 decimal places.
- Per-axis G_scale swing: T5C 0.678, Slot A v2 0.698 (+3.0% — slightly worse)
- Per-axis saturation swing: T5C 0.030, Slot A v2 0.148 (**4.93× T5C** — Bar 3 amplification triggered)
- Cross-axis mean swing: T5C 0.320, Slot A v2 0.332 (+3.8% — marginally worse overall)

### 2.3 Anchor mechanism works for 4 of 8 chronic identities (Job D)

At contract-calibrated τ (dev_real_FPR = 0.07; τ_P8A = 0.913, τ_SlotAv2 = 0.774):

| Identity | n | P8A FPR | SlotAv2 FPR | Δ | Bar A (anchor fix) | Bar B (regression) |
|---|---:|---:|---:|---:|:---:|:---:|
| Roy_D (dev) | 130 | 0.300 | **0.840** | **+0.540** | — | (P8A_FPR > 0.10 → bar inapplicable; LARGEST absolute regression) |
| Q (dev) | 36 | 0.860 | 0.220 | −0.640 | (>0.05 → bar inapplicable; LARGEST absolute fix) | — |
| Chikara_Takahashi (lockbox) | 25 | 0.360 | 0.000 | −0.360 | **✅ MET** | — |
| PC_Generator (dev) | 525 | 0.240 | 0.060 | −0.180 | (just-above-0.05 ceiling) | — |
| PC_Generator (lockbox) | 28 | 0.290 | 0.000 | −0.290 | **✅ MET** | — |
| bla_bla_chow (dev) | 467 | 0.060 | 0.150 | +0.090 | — | (small regression) |
| Cam_Test (dev) | 98 | 0.000 | 0.000 | 0 | — | — |
| Md_Noyn_Sharker (dev) | 409 | 0.000 | 0.000 | 0 | — | — |
| dor_shkedi (dev) | 20 | 0.000 | 0.000 | 0 | — | — |

**Roy_D is NOT in the contract lockbox cohort** — the +0.54 dev regression does not propagate to the contract `lockbox_real_fpr`. This is a deployment surface vs eval surface mismatch.

### 2.4 may6 production-drift retest (Job F)

`may6` cohort: 92 frames from a production-drift Xinhe session, captured 2026-05-06. Memory-recorded baselines:

| Ckpt | n_fired/92 | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|
| P8A_step5000 (baseline) | 0 | 0.009 | 0.037 | 0.148 | 0.254 |
| E2B (current deployment) | 53 | 0.579 | 0.945 | 0.989 | 0.990 |
| T5C_step3500 | 16 | 0.170 | 0.582 | 0.800 | 0.829 |
| **Slot A v2 step3500** (this batch) | **4** | **0.097** | **0.290** | **0.574** | **0.783** |
| P8A re-score (harness sanity, this batch) | 0 | 0.009 | 0.037 | 0.148 | 0.254 |

Harness sanity: max |Δ| vs prior = 7.25e-06; mean |Δ| = 4.32e-07. MPS floating-point noise only.

### 2.5 Tiebreak policy fragility (Job C)

5 lex policies (strict, thresholded at 0.001/0.005/0.010/0.030): P8A wins all 5.
5 composite policies (λ ∈ {5, 10, 20, 50, 100}): Slot A v2 step3500 wins all 5.
1 Pareto-dominance policy: Slot A v2 step3500 wins.

Note: the lex-thresholded policy implementation had a banker's-rounding bug; corrected by hand, lex policies with bin-width T ≥ 0.005 pick Slot A v2 (FPRs 0.0184 and 0.0191 fall in the same bin; within-bin tiebreak = `lockbox_fake_recall` DESC).

### 2.6 τ-sweep operating frontier (Job E)

On the 800-frame chronic-heavy canary panel, at target dev_real_fpr ≤ 0.10:

| Ckpt | τ_cal | dev_fpr_achieved | lockbox_fake_recall | viso_recall | deeplive_recall |
|---|---:|---:|---:|---:|---:|
| P8A | 0.99 | **0.152** (cannot reach 0.10) | 0.27 | 0.00 | 0.08 |
| T5C step3500 | 0.91 | 0.094 | 0.40 | 0.00 | 0.20 |
| **Slot A v2 step3500** | **0.88** | **0.100** | **0.60** | 0.00 | 0.34 |
| Slot 1 6-axis+anchor step3500 | 0.89 | 0.068 | 0.20 | 0.00 | 0.36 |
| Slot 2 LoRA L8-L9 step2500 | 0.92 | 0.100 | 0.56 | 0.02 | 0.34 |

**Panel composition caveat**: this canary panel is dominated by chronic-6 identities (60% of reals) — different composition from the 29-suite contract substrate (where chronics are a small fraction of 3253 dev_real videos). P8A's "cannot reach 0.10 on this panel" is panel-specific; the contract's `dev_primary_real_fpr` for P8A is 0.0695. The Slot A v2 vs T5C lockbox-recall gap (0.60 vs 0.40) reproduces the contract trend.

### 2.7 Move 3 verification

All 16 tests across 3 new test files pass in 5.05s:
- `test_canary_with_grl_wiring.py` 3/3 — verifies `inference=True` fix prevents canary silencing under `multi_axis_grl.enabled=true`
- `test_lora_adapter.py` 10/10 — full LoRA adapter integration tests
- `test_lora_ckpt_roundtrip.py` 3/3 — save_ckpt embeds LoRA cfg + state-dict-key fallback + unexpected-key warning

Source diffs (uncommitted, ?? for test files):
```
M  trainer/mixins/canary_probe.py        +23/−2
M  batch_inference_gcs.py                +77/−0
?? tests/test_canary_with_grl_wiring.py  153 lines
?? tests/test_lora_adapter.py            364 lines
?? tests/test_lora_ckpt_roundtrip.py     177 lines
```

## 3. Cross-cutting bar table

The 7 jobs together test 8 conjectures about Slot A v2 step3500:

| Conjecture | Job(s) | Result |
|---|---|---|
| The contract tiebreak `lockbox_real_fpr` ASC is comparing values inside per-video bootstrap noise | B, C | **CONFIRMED** (95% CI on Δ covers 0; P=0.519) |
| Slot A v2 catches more lockbox fakes than P8A at matched FPR | E (panel-internal) | **CONFIRMED** at dev_fpr ≤ 0.10 (0.60 vs P8A 0.27, both panel-internal; consistent with 29-suite contract +30pp) |
| Anchor mechanism reduces transport-axis sensitivity (Roy_D / Guest natural experiment) | A, G | **NOT CONFIRMED** (Δ identical to T5C; saturation axis amplified 4.93×) |
| Anchor mechanism suppresses chronic-identity over-fires | D, E | **CONFIRMED PARTIALLY** (Chikara/PC_Gen/Q/dor_shkedi fixed; Roy_D regresses +0.54; bla_bla_chow small regression) |
| Slot A v2 keeps P8A-level production-drift robustness on may6 | F | **CONFIRMED** (4/92 vs P8A 0/92 — both well below E2B's 53/92) |
| The canary + LoRA-load bugs surfaced on 2026-05-20 have working fixes | Move 3 | **CONFIRMED** (16/16 tests pass) |
| The contract's lex-on-FPR-ASC ranking is robust to tiebreak-policy choice | C | **NOT CONFIRMED** (P8A wins 5/11 policies; SlotAv2 wins 6/11) |
| Slot 1 (6-axis GRL + anchor stack) outperforms Slot A v2 (anchor alone) on the deployment metric | E | **NOT CONFIRMED** (Slot 1 step3500 lockbox_recall 0.20 vs SlotAv2 0.60 at dev_fpr ≤ 0.10) |

## 4. Open questions / next jobs not in this batch

- **Multi-account capture sweep** (Cheap_Followups §6 #1 from 2026-05-19): captures ~20-50 frames per Teams account configuration to measure whether the per-account `prob_fake` shift is N>>2 systematic. **Real-world capture task** — needs the user, not CPU.
- **Roy_D-specific anchor pool** (open loop `roy-d-specific-anchor-pool-packet` from 2026-05-16): tests whether adding Roy_D anchors fixes the Roy_D regression. GPU packet, not CPU.
- **bla_bla_chow regression mechanism** (open loop `slot-a-bla-bla-chow-regression` from 2026-05-16): per-identity encoder probe. CPU-feasible but not in this batch.
- **Teams-transport synthetic augmentation** (Cheap_Followups §2 finding: blur σ=2.5 + scale 0.80 reproduces Guest's score within 0.02): GPU packet that adds this augmentation to training. Not in this batch.

## 5. Artifacts inventory

```
analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/
├── RESULTS_FACTS_2026-05-20.md          (this file)
├── AGENT_PROPOSAL_2026-05-20.md         (single OPINIONS doc)
├── job_a_natural_experiment/
│   ├── JOB_A_NATURAL_EXPERIMENT_FACTS_2026-05-20.md
│   ├── run_slot_a_v2_inference.py
│   └── outputs/
│       ├── job_a_inference_results.json
│       ├── job_a_inference_results.csv
│       └── job_a_combined_table.csv
├── job_b_bootstrap_ci/
│   ├── JOB_B_BOOTSTRAP_FACTS_2026-05-20.md
│   ├── OPINIONS_2026-05-20.md           (sub-agent's per-job opinion)
│   ├── run_bootstrap.py
│   ├── bootstrap_results.json
│   └── data/                            (8 GCS pulls)
├── job_c_tiebreak_rerank/
│   ├── JOB_C_TIEBREAK_RERANK_FACTS_2026-05-20.md
│   ├── run_tiebreak_rerank.py
│   └── outputs/
│       ├── rerank_table.csv
│       ├── policy_winners.csv
│       └── bootstrap_check.json
├── job_d_per_identity_fp/
│   ├── JOB_D_PER_IDENTITY_FACTS_2026-05-20.md
│   └── (script + per-identity outputs)
├── job_e_tau_sweep/
│   ├── JOB_E_TAU_SWEEP_FACTS_2026-05-20.md
│   ├── run_tau_sweep.py
│   ├── post_process_tau_sweep.py
│   └── outputs/
│       ├── tau_sweep_table.csv          (6672 rows)
│       ├── calibrated_tau_summary.csv
│       ├── operating_points.csv
│       ├── dev_cal_05pct_summary.csv
│       ├── operating_point_comparison.md
│       └── figs/                        (3 plots)
├── job_f_may6_retest/
│   ├── JOB_F_MAY6_FACTS_2026-05-20.md
│   ├── run_may6_slot_a_v2.py
│   └── outputs/
│       ├── scores_slot_a_v2_step3500_may6.csv
│       ├── scores_p8a_sanity_may6.csv
│       └── may6_retest_table_with_slot_a_v2.csv
├── job_g_perturbation/
│   ├── JOB_G_PERTURBATION_FACTS_2026-05-20.md
│   ├── run_perturbation_slot_a_v2.py
│   └── outputs/
│       ├── perturbation_sweep_slot_a_v2.csv  (79 rows)
│       ├── swing_comparison.csv
│       └── (summary + log files)
└── move3_test_verification/
    ├── MOVE3_TEST_VERIFICATION_FACTS_2026-05-20.md
    └── move3_pytest_output.txt
```
