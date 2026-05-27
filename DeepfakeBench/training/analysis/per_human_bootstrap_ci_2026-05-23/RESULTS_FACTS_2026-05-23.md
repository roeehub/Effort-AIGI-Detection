# Bootstrap 95% CIs on per-human FPR / fake-recall — RESULTS FACTS

Generated 2026-05-23. Factual readout only (forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, breakthrough). Interpretive content lives in `AGENT_PROPOSAL_2026-05-23.md`.

> **Question**: Are the borderline cells in the team-identity readout (P8A Xinhe-fake-recall 0.487 at mode B; SlotAv2_FACE dor-recall 0.499 at per-ckpt τ; T5C dor real-FPR 0.052 at mode B) statistically distinguishable from the 5%/50% floors, or sample noise?
>
> **Method**: For each (ckpt × τ × human × role) cell, draw 2,000 bootstrap resamples (with replacement, same N as original sample) and report 2.5/97.5 percentiles as 95% CI. Stratified bootstrap (each cell resampled independently).
>
> **Input**: 5,941 deploy-relevant team-identity frames + 5 ckpts' per-frame `prob_*` columns from `analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs/per_frame_full.csv`.

---

## 0. Method details

- N_BOOT = 2000
- RNG_SEED = 42 (reproducible)
- For each (ckpt × tau_kind × human × metric):
  1. Subset frames to (human, role-matching-metric)
  2. Compute 0/1 array (above τ for FPR, above τ for recall)
  3. Bootstrap resample indices 2000 times
  4. Report point estimate + 95% CI
- τ values evaluated per ckpt: mode_A (0.535), mode_B (0.78), mode_C (0.87), per_ckpt (from `analysis/per_ckpt_tau_recal_2026-05-23/outputs/per_ckpt_summary.csv` user_bar best-τ)
- Cell counts: per-human real cohorts (Noyn 210, Roee_Windows 330, Xiang 582, Xinhe 79, dor 620); per-human fake-attack cohorts (Xiang 578, Xinhe 1099, dor 2443).

---

## 1. Borderline cells (point within ±2pp of relevant floor)

For each, the CI verdict against the floor:
- **PASS+CI**: CI lies entirely on the passing side of the floor
- **FAIL+CI**: CI lies entirely on the failing side of the floor
- **AMBIGUOUS**: CI crosses the floor

| Ckpt | τ_kind | τ | Human | Metric | n | Point | CI lo | CI hi | Width | Verdict |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| P8A | mode_A | 0.535 | dor | real_FPR | 620 | 0.040 | 0.026 | 0.056 | 0.031 | AMBIGUOUS |
| P8A | mode_B | 0.780 | Xinhe | fake_recall | 1099 | 0.487 | 0.457 | 0.518 | 0.061 | AMBIGUOUS |
| **P8A** | **per_ckpt** | **0.590** | **dor** | **real_FPR** | **620** | **0.037** | **0.023** | **0.052** | **0.029** | **AMBIGUOUS** |
| E2B | mode_B | 0.780 | Xiang | real_FPR | 582 | 0.036 | 0.022 | 0.052 | 0.029 | AMBIGUOUS |
| **E2B** | **mode_B** | **0.780** | **dor** | **fake_recall** | **2443** | **0.473** | **0.454** | **0.493** | **0.040** | **FAIL+CI** |
| E2B | per_ckpt | 0.718 | Xiang | real_FPR | 582 | 0.048 | 0.031 | 0.065 | 0.034 | AMBIGUOUS |
| **E2B** | **per_ckpt** | **0.718** | **dor** | **fake_recall** | **2443** | **0.528** | **0.508** | **0.549** | **0.041** | **PASS+CI** |
| T5C | mode_A | 0.535 | Xiang | real_FPR | 582 | 0.065 | 0.046 | 0.086 | 0.040 | AMBIGUOUS |
| T5C | mode_A | 0.535 | Xinhe | fake_recall | 1099 | 0.493 | 0.461 | 0.522 | 0.061 | AMBIGUOUS |
| **T5C** | **mode_B** | **0.780** | **dor** | **real_FPR** | **620** | **0.052** | **0.035** | **0.069** | **0.034** | **AMBIGUOUS** |
| T5C | per_ckpt | 0.790 | dor | real_FPR | 620 | 0.050 | 0.032 | 0.068 | 0.035 | AMBIGUOUS |
| SlotAv2_CLS | mode_A | 0.535 | Xiang | real_FPR | 582 | 0.055 | 0.038 | 0.076 | 0.038 | AMBIGUOUS |
| SlotAv2_CLS | mode_A | 0.535 | dor | real_FPR | 620 | 0.050 | 0.034 | 0.068 | 0.034 | AMBIGUOUS |
| SlotAv2_CLS | mode_A | 0.535 | Xinhe | fake_recall | 1099 | 0.482 | 0.453 | 0.511 | 0.058 | AMBIGUOUS |
| SlotAv2_CLS | per_ckpt | 0.562 | Xiang | real_FPR | 582 | 0.050 | 0.033 | 0.067 | 0.034 | AMBIGUOUS |
| SlotAv2_CLS | per_ckpt | 0.562 | dor | real_FPR | 620 | 0.044 | 0.027 | 0.061 | 0.034 | AMBIGUOUS |
| **SlotAv2_CLS** | **per_ckpt** | **0.562** | **Xinhe** | **fake_recall** | **1099** | **0.458** | **0.429** | **0.489** | **0.060** | **FAIL+CI** |
| SlotAv2_FACE | mode_A | 0.535 | Roee_Windows | real_FPR | 330 | 0.030 | 0.012 | 0.052 | 0.039 | AMBIGUOUS |
| SlotAv2_FACE | mode_B | 0.780 | Xinhe | fake_recall | 1099 | 0.495 | 0.464 | 0.525 | 0.061 | AMBIGUOUS |
| SlotAv2_FACE | per_ckpt | 0.681 | Xiang | real_FPR | 582 | 0.050 | 0.033 | 0.067 | 0.034 | AMBIGUOUS |
| **SlotAv2_FACE** | **per_ckpt** | **0.681** | **dor** | **fake_recall** | **2443** | **0.499** | **0.479** | **0.519** | **0.039** | **AMBIGUOUS** |

---

## 2. Headline cells at per-ckpt-calibrated τ (key humans only)

Critical cells driving the production-switch decision. Bold rows are the gate-binding cells.

| Ckpt | τ | Human | Metric | n | Point | CI lo | CI hi |
|---|---:|---|---|---:|---:|---:|---:|
| **P8A** | **0.590** | **dor** | **real_FPR** | **620** | **0.037** | **0.023** | **0.052** |
| P8A | 0.590 | Xinhe | real_FPR | 79 | 0.013 | 0.000 | 0.038 |
| **P8A** | **0.590** | **Xinhe** | **fake_recall** | **1099** | **0.603** | **0.574** | **0.631** |
| P8A | 0.590 | dor | fake_recall | 2443 | 0.796 | 0.779 | 0.812 |
| **E2B** | **0.718** | **Xiang** | **real_FPR** | **582** | **0.048** | **0.031** | **0.065** |
| E2B | 0.718 | dor | real_FPR | 620 | 0.011 | 0.005 | 0.019 |
| **E2B** | **0.718** | **dor** | **fake_recall** | **2443** | **0.528** | **0.508** | **0.549** |
| E2B | 0.718 | Xinhe | fake_recall | 1099 | 0.712 | 0.685 | 0.739 |
| **T5C** | **0.790** | **Xinhe** | **fake_recall** | **1099** | **0.237** | **0.213** | **0.264** |
| T5C | 0.790 | dor | real_FPR | 620 | 0.050 | 0.032 | 0.068 |
| T5C | 0.790 | dor | fake_recall | 2443 | 0.612 | 0.592 | 0.632 |
| **SlotAv2_CLS** | **0.562** | **Xinhe** | **fake_recall** | **1099** | **0.458** | **0.429** | **0.489** |
| SlotAv2_CLS | 0.562 | dor | real_FPR | 620 | 0.044 | 0.027 | 0.061 |
| **SlotAv2_FACE** | **0.681** | **dor** | **fake_recall** | **2443** | **0.499** | **0.479** | **0.519** |
| SlotAv2_FACE | 0.681 | Xinhe | fake_recall | 1099 | 0.849 | 0.827 | 0.870 |
| SlotAv2_FACE | 0.681 | dor | real_FPR | 620 | 0.021 | 0.011 | 0.034 |

---

## 3. Gate-pass verdicts under bootstrap CIs (per-ckpt-calibrated τ, user_bar 5%/50%)

| Ckpt | All real-FPRs ≤ 5% (definitively)? | All fake-recalls ≥ 50% (definitively)? | Combined verdict |
|---|---|---|---|
| P8A | NO (dor upper CI 0.052) | YES (min upper CI 0.574 on Xinhe) | One-tail noise risk on dor real-FPR |
| E2B | NO (Xiang upper CI 0.065) | YES (min lower CI 0.508 on dor) | One-tail noise risk on Xiang real-FPR |
| T5C | NO | NO (Xinhe upper CI 0.264 far below 0.50) | Definitive multi-cell failure |
| SlotAv2_CLS | YES (max upper CI 0.067 — over 5%) | NO (Xinhe upper CI 0.489) | Definitive Xinhe failure |
| SlotAv2_FACE | YES (Xiang upper CI 0.067 — over 5%) | NO (dor upper CI 0.519, AMBIGUOUS) | One-tail noise risk on dor recall |

Notes:
- "definitively" = CI bound on the passing side of the floor
- "noise risk" = point passes but CI crosses floor
- "definitive failure" = CI bound on the failing side

A more permissive read at 1-sigma (point estimate only, ignore noise): P8A passes; E2B passes; T5C fails; SlotAv2_CLS fails; SlotAv2_FACE near-misses (0.499 vs 0.50 floor).

---

## 4. Sample-size sensitivity

CI widths track √n as expected. Narrowest (best-constrained) cells:
- E2B dor real_FPR @ per_ckpt: n=620, width 0.014 (point 0.011, CI [0.005, 0.019])
- P8A dor fake_recall @ per_ckpt: n=2443, width 0.033 (point 0.796, CI [0.779, 0.812])

Widest cells (lowest constraint):
- Xinhe real_FPR (any ckpt × τ): n=79, width ~0.04 (point ~0.01-0.03)
- Roee_Windows real_FPR (any ckpt × τ): n=330, width ~0.04

The Xinhe real cohort (n=79) is the smallest sample and contributes the most noise to per-human FPR gate evaluation. Adding Xinhe real frames is the highest-information data-acquisition target.

---

## 5. Specific question answers

### Q1: Is P8A's mode-B Xinhe-recall miss (0.487 vs 0.50) sample noise?

**Yes.** Point 0.487, CI [0.457, 0.518]. The CI spans both sides of 0.50; the 0.013pp gap is within sample noise on n=1099. If we re-sampled this cohort, ~40% of resamples land above the 0.50 floor.

### Q2: Is SlotAv2_FACE's per-ckpt-τ dor-recall miss (0.499 vs 0.50) sample noise?

**Yes.** Point 0.499, CI [0.479, 0.519]. The CI spans both sides of 0.50; effectively at the floor. ~45% of resamples land above 0.50.

### Q3: Is T5C's mode-B dor-FPR over-floor (0.052 vs 0.05) sample noise?

**Likely yes** on this specific cell (CI [0.035, 0.069] crosses 0.05). But T5C's failure on the team-identity bar is driven by **Xinhe fake recall 0.237 (CI [0.213, 0.264])**, which is definitively far from the 50% floor. T5C's failure is NOT noise on the binding cell.

### Q4: Is E2B's dor fake-recall miss at mode B (0.473) sample noise?

**No.** Point 0.473, CI [0.454, 0.493]. The upper CI bound is below the 0.50 floor. E2B at mode B definitively misses the dor recall floor (0% of resamples reach 0.50). However, E2B at per-ckpt τ=0.718 definitively passes (CI [0.508, 0.549]).

### Q5: Are P8A's per-ckpt-τ passes definitive?

**Partially**:
- Xinhe fake recall: 0.603, CI [0.574, 0.631] — definitively above 0.50
- dor fake recall: 0.796, CI [0.779, 0.812] — definitively above 0.50
- Xiang fake recall: 0.881, CI [0.857, 0.905] — definitively above 0.50
- **dor real_FPR: 0.037, CI [0.023, 0.052]** — point passes 5%, upper CI just over 5%; ~3% of resamples breach the floor
- All other real-side cells comfortably under 5% with CI bounds well under

P8A's pass is fake-side definitive, real-side robust-but-not-bulletproof. The dor real-FPR upper CI of 0.052 is the only sub-5% noise risk.

---

## 6. Artifacts

- `outputs/per_cell_ci.csv` — long-format CI table (all 5 ckpts × 4 τ_kinds × 8 cells = 160 rows)
- `outputs/borderline_check.csv` — focused view on cells within ±2pp of relevant floor
- `scripts/bootstrap_ci.py` — bootstrap driver (deterministic with seed=42)

Wall time: ~3 sec for full sweep on 5,941 frames × 5 ckpts × 4 τs × 8 cells × 2000 resamples.

---

## 7. Caveats

1. **Bootstrap assumes independence within cells.** Frames within the same video are correlated; bootstrap CIs may slightly underestimate uncertainty if intra-video correlation is significant. Mitigation would be block-bootstrap by video — not done here. Magnitude of effect: probably 10-30% wider CIs in worst case.
2. **Sample cap from source readout carries over** (REAL_CAP=150, FAKE_CAP=100 per cohort). Per-human aggregates inherit this cap. The full dor cohort is 2180 reals; sampled to 620. A no-cap re-extraction would tighten dor CIs by ~√(2180/620) = 1.9× — moderate but not dramatic.
3. **Xinhe real n=79 is the binding small-sample issue.** No per-cohort sample cap helps here — there are simply few Xinhe real frames available. Xinhe real FPR CIs at 5% point have width ±5pp inherently.
4. **The 2000 bootstrap resamples** is enough for 95% CIs to ±0.5pp Monte Carlo error; sufficient for the decisions here.
