# RESULTS FACTS — Track B Contract Reframe (Composite Tiebreak with λ)

**Date:** 2026-05-22
**Owner:** working agent
**Plan reference:** `/Users/roeedar/.claude/plans/ok-so-we-don-t-valiant-quasar.md` (Track B)

## 1. Scope

This document captures the closed-form rerank of the published 2026-05-20 4-checkpoint panel under a composite cross-checkpoint ranking policy that replaces the lockbox-FPR lexicographic tiebreak with an explicit FP/FN cost ratio λ. No new model evaluations were run; the contract's underlying suite probabilities are unchanged. Only the ranking policy was edited and verified via unit tests + closed-form arithmetic on the published per-suite numbers.

## 2. Inputs

### 2.1 Published 4-checkpoint panel (source: `docs/packet_retrospectives/STATE.md` line 100–103)

29-suite scorecard, Vertex job `146629015254335488`, GCS prefix `slot-a-v2-validation-2026-05-20/`.

| ckpt | dev_real_fpr | dev_stress_fpr | dev_fake_macro | lockbox_real_fpr | lockbox_fake_recall |
|---|---:|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.070 | 0.069 | 0.300 | **0.0184** | 0.387 |
| SLOT_A_ANCHOR_AWARE_STEP3500 | 0.066 | 0.099 | 0.438 | 0.0191 | **0.688** |
| T5C_PERIODIC_STEP3500 | 0.065 | 0.099 | 0.459 | 0.0279 | 0.660 |
| SLOT_A_ANCHOR_AWARE_STEP1500 | 0.065 | 0.099 | **0.207** | 0.0044 | 0.640 |

All four checkpoints pass the dev FPR budgets (≤0.07 primary / ≤0.10 stress). Three of the four meet the `dev_fake_macro_recall ≥ 0.30` floor; SLOT_A_ANCHOR_AWARE_STEP1500 does not (0.207 < 0.30) and is therefore demoted to tier-1 by the recall-floor gate independent of any tiebreak policy.

### 2.2 Lex-policy ranking (current scorer behavior)

| rank | ckpt | tier | sort key (lockbox_real_fpr, −lockbox_fake_recall, …) |
|---:|---|:---:|---|
| 1 | P8A_REFERENCE_STEP5000 | 0 | (0.0184, −0.387, …) |
| 2 | SLOT_A_ANCHOR_AWARE_STEP3500 | 0 | (0.0191, −0.688, …) |
| 3 | T5C_PERIODIC_STEP3500 | 0 | (0.0279, −0.660, …) |
| 4 | SLOT_A_ANCHOR_AWARE_STEP1500 | 1 | (recall floor violated) |

## 3. Methodology

### 3.1 Composite tiebreak formula

For any row with finite `lockbox_real_fpr` and `lockbox_fake_recall`:

```
composite(λ) = lockbox_real_fpr + λ × (1 − lockbox_fake_recall)
```

Lower composite ranks higher. The formula treats one percentage point of `lockbox_real_fpr` as equivalent to (1/λ) percentage points of `lockbox_fake_recall` loss. The recall-floor tier gate is preserved unchanged; the composite only replaces the post-tier ordering.

### 3.2 Closed-form pairwise crossover

For two rows A and B sharing the same tier, the policy switches the order between them at

```
λ* = (FPR_B − FPR_A) / (recall_B − recall_A)
```

(with A defined as the lower-FPR row, i.e. lex-rank-better; positive λ* iff B has higher recall).

### 3.3 Code change

`arena/score_teams_promotion_contract.py`: added `tiebreak_policy: str = "lex"` and `tiebreak_lambda: float = 1.0` to `ContractConfig`; new `_composite_tiebreak_score` helper; new branch in `_promotion_summary_sort_key` activated when `tiebreak_policy == "composite"`; `composite_tiebreak_score` and `composite_tiebreak_lambda` columns stamped into each summary row under the composite branch; payload `contract` dict now reports both new fields. CLI flags `--tiebreak_policy {lex,composite}` and `--tiebreak_lambda` added with defaults `lex` and `1.0`. The wrapper `arena/run_target_domain_validation_sequential.py` gained the parallel pass-throughs `--promotion_tiebreak_policy` and `--promotion_tiebreak_lambda`.

### 3.4 Tests

`tests/test_score_teams_promotion_contract.py`: six new test functions covering (a) lex remains the default, (b) the `_composite_tiebreak_score` formula on the published panel numbers, (c) missing-metric rows collapse to `+inf`, (d) the per-row sort key flips between policies, (e) an end-to-end run with synthetic suite reports observes the rerank, and (f) CLI defaults parity between dataclass, scorer, and wrapper. Full test suite (12 functions including the 6 pre-existing back-compat tests) passes in 0.08s under pytest.

## 4. Closed-form pairwise crossovers on the panel

| pair | (A − B) FPR Δ | (B − A) recall Δ | λ* | interpretation |
|---|---:|---:|---:|---|
| P8A vs SLOT_A_v2_STEP3500 | +0.0007 | +0.301 | **0.00233** | composite flips at any λ > ~0.0023 |
| P8A vs T5C_STEP3500 | +0.0095 | +0.273 | **0.0348** | composite flips at any λ > ~0.035 |
| SLOT_A_v2_STEP3500 vs T5C_STEP3500 | −0.0088 | −0.028 | n/a (negative) | Slot A v2 is Pareto-dominant on both metrics — no positive crossover |

Slot A v2 step3500 dominates T5C step3500 on both axes (lower lockbox_real_fpr by 0.88pp AND higher lockbox_fake_recall by 2.8pp). Under any non-negative λ, Slot A v2 step3500 ranks above T5C step3500.

## 5. Composite score table (tier-0 rows only)

| λ | P8A | Slot A v2 step3500 | T5C step3500 | ordering |
|---:|---:|---:|---:|---|
| 0.000 | 0.01840 | 0.01910 | 0.02790 | P8A < Slot A v2 < T5C |
| 0.001 | 0.01901 | 0.01941 | 0.02824 | P8A < Slot A v2 < T5C |
| 0.00233 | 0.01983 | 0.01983 | 0.02867 | P8A = Slot A v2 (tie) < T5C |
| 0.005 | 0.02147 | 0.02066 | 0.02960 | Slot A v2 < P8A < T5C |
| 0.01 | 0.02453 | 0.02222 | 0.03130 | Slot A v2 < P8A < T5C |
| 0.0348 | 0.03973 | 0.02995 | 0.03973 | Slot A v2 < P8A = T5C (tie) |
| 0.05 | 0.04905 | 0.03470 | 0.04490 | Slot A v2 < T5C < P8A |
| 0.10 | 0.07970 | 0.05030 | 0.06190 | Slot A v2 < T5C < P8A |
| 1.0 | 0.63140 | 0.33110 | 0.36790 | Slot A v2 < T5C < P8A |
| 5.0 | 3.08340 | 1.57910 | 1.72790 | Slot A v2 < T5C < P8A |
| 20.0 | 12.27840 | 6.25910 | 6.74790 | Slot A v2 < T5C < P8A |

All values computed by applying `composite(λ) = FPR + λ × (1 − recall)` to the published per-suite numbers in section 2.1; SLOT_A_ANCHOR_AWARE_STEP1500 omitted because its tier-1 demotion is independent of λ.

## 6. Rank shifts vs. lex baseline

| λ regime | ordering of tier-0 rows | shift from lex |
|---|---|---|
| λ ∈ [0, 0.00233) | P8A, Slot A v2 step3500, T5C step3500 | identical to lex |
| λ ∈ (0.00233, 0.0348) | Slot A v2 step3500, P8A, T5C step3500 | rank-1 ↔ rank-2 swap |
| λ ∈ (0.0348, ∞) | Slot A v2 step3500, T5C step3500, P8A | rank-1 ↔ rank-2 swap AND P8A drops to rank-3 |

Tier-1 row (SLOT_A_ANCHOR_AWARE_STEP1500) remains at rank-4 across all λ.

## 7. Open data

- The 12-checkpoint U_SLOTS panel rerank under composite is not in this document. The numerical inputs for that panel were not re-pulled from GCS; the scorer code change supports reranking it offline by re-running the scorer against the existing `*_videos_report.csv` artifacts under `slot-a-v2-validation-2026-05-20/` and any equivalent U_SLOTS report root. Operator can replay either panel by adding `--tiebreak_policy composite --tiebreak_lambda <chosen>` to an existing scorer invocation.
- The 95% paired-bootstrap CI on the P8A vs. Slot A v2 step3500 lockbox_real_fpr gap was published 2026-05-20 as [−0.008, +0.010] (covers zero). The composite crossover λ* ≈ 0.00233 is computed against the point estimates; under the bootstrap CI bounds the crossover can range from approximately λ* ≈ −0.033 (gap = −0.010) to λ* ≈ +0.033 (gap = +0.010). The sign of (FPR_SlotAv2 − FPR_P8A) itself is uncertain at the 95% level — under that uncertainty the lex policy's choice of P8A as rank-1 is decided by a quantity whose sign is not statistically resolved.
- λ is a USER policy decision and is not chosen by this document. The scorer's CLI default remains `tiebreak_policy=lex` for back-compat; the composite path is opt-in.

## 8. Reproducibility

Rerank either panel by:

```
python arena/score_teams_promotion_contract.py \
  --report_root <existing report root> \
  --checkpoint_map <existing checkpoint map> \
  --checkpoints ALL \
  --output_dir <new output dir> \
  --tiebreak_policy composite \
  --tiebreak_lambda <λ>
```

Or via the sequential validation wrapper with `--promotion_tiebreak_policy composite --promotion_tiebreak_lambda <λ>`.

Unit-test verification:

```
python -m pytest tests/test_score_teams_promotion_contract.py -v
```

All 12 tests pass on the working tree as of 2026-05-22.
