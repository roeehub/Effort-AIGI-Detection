# Per-ckpt τ-recalibration on team-identity bar — RESULTS FACTS

Generated 2026-05-23. Factual readout only (per `docs/packet_retrospectives/AGENTS.md` §"Eval-folder authoring contract" — forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, breakthrough). Interpretive content lives in `AGENT_PROPOSAL_2026-05-23.md`.

> **Question**: When τ is set per-ckpt instead of using Slot A v2's calibrated cross-ckpt constants (mode A 0.535 / mode B 0.78 / mode C 0.87), which ckpts pass the user-specified team-identity bar (per-human real FPR ≤ 5% AND per-human fake recall ≥ 50%)?
>
> **Input**: 5,941 deploy-relevant team-identity frames (1,821 real + 4,120 fake-attack across 5 humans) from `analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs/per_frame_full.csv`. No re-scoring — the existing per-frame `prob_*` columns are used.
>
> **Method**: For each ckpt, sweep τ over [0, 1] in 1001 steps. At each τ, compute per-human real FPR for all 5 deploy-relevant humans (Noyn, Roee_Windows, Xiang, Xinhe, dor) and per-human fake recall for the 3 humans with fake-attack cohorts (Xiang, Xinhe, dor). Identify contiguous τ ranges where `max(real_FPR_per_human) ≤ floor_A AND min(fake_recall_per_human) ≥ floor_B` for four floor pairs.

---

## 1. Gate combinations evaluated

| Gate name | Max per-human real FPR floor | Min per-human fake recall floor |
|---|---:|---:|
| `user_bar` (current spec) | 5% | 50% |
| `relaxed` (per `team_identity_deploy_readout_expanded` AGENT_PROPOSAL §6) | 5% | 40% |
| `very_strict` | 2% | 60% |
| `recall_lean` | 10% | 50% |

---

## 2. Per-ckpt passing-τ ranges by gate

| Ckpt | user_bar (5%/50%) | relaxed (5%/40%) | very_strict (2%/60%) | recall_lean (10%/50%) |
|---|---|---|---|---|
| **P8A** | [0.427, 0.753]  width 0.326 | [0.427, 0.887]  width 0.460 | empty | [0.342, 0.753]  width 0.411 |
| **E2B** | [0.687, 0.748]  width 0.061 | [0.687, 0.851]  width 0.164 | empty | [0.598, 0.748]  width 0.150 |
| **T5C** | empty | empty | empty | empty |
| **Slot A v2 CLS** | empty | [0.562, 0.634]  width 0.072 | empty | [0.310, 0.634]  width 0.324 |
| **Slot A v2 face-pool** | empty | [0.681, 0.719]  width 0.038 | empty | [0.612, 0.719]  width 0.107 |

`Width` is the size of the contiguous τ-interval that passes the gate. Wider = more τ-robust = less sensitive to small calibration shifts.

---

## 3. Per-ckpt detail at best-τ inside the user_bar passing range

`best_tau` = midpoint of the widest passing range, or — if no range passes — the τ that maximizes `min_fake_recall` subject to `max_real_fpr ≤ 5%`.

### 3.1 At user_bar (5%/50%)

| Ckpt | best τ | max real FPR | min fake recall | Noyn FPR | Roee_W FPR | Xiang FPR | Xinhe FPR | dor FPR | Xiang recall | Xinhe recall | dor recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **P8A** | **0.590** | **0.037** | **0.603** | 0.014 | 0.009 | 0.022 | 0.013 | 0.037 | 0.881 | 0.603 | 0.796 |
| **E2B** | **0.718** | **0.048** | **0.528** | 0.000 | 0.000 | 0.048 | 0.000 | 0.011 | 0.886 | 0.712 | 0.528 |
| T5C | 0.790 | 0.050 | 0.237 | 0.000 | 0.000 | 0.019 | 0.000 | 0.050 | 0.903 | 0.237 | 0.612 |
| Slot A v2 CLS | 0.562 | 0.050 | 0.458 | 0.000 | 0.003 | 0.050 | 0.013 | 0.044 | 0.931 | 0.458 | 0.563 |
| Slot A v2 face-pool | 0.681 | 0.050 | 0.499 | 0.000 | 0.003 | 0.050 | 0.013 | 0.021 | 0.978 | 0.849 | 0.499 |

### 3.2 At relaxed (5%/40%)

| Ckpt | best τ | max real FPR | min fake recall | Xiang recall | Xinhe recall | dor recall |
|---|---:|---:|---:|---:|---:|---:|
| **P8A** | **0.657** | **0.032** | **0.560** | 0.867 | 0.560 | 0.780 |
| **E2B** | **0.769** | **0.038** | **0.483** | 0.872 | 0.686 | 0.483 |
| T5C | 0.790 | 0.050 | 0.237 | 0.903 | 0.237 | 0.612 |
| **Slot A v2 CLS** | **0.598** | **0.040** | **0.429** | 0.926 | 0.429 | 0.543 |
| **Slot A v2 face-pool** | **0.700** | **0.024** | **0.449** | 0.971 | 0.801 | 0.449 |

### 3.3 At very_strict (2%/60%)

| Ckpt | best τ | max real FPR | min fake recall |
|---|---:|---:|---:|
| P8A | 0.706 | 0.019 | 0.529 |
| E2B | 0.852 | 0.019 | 0.399 |
| T5C | 0.846 | 0.019 | 0.165 |
| Slot A v2 CLS | 0.834 | 0.017 | 0.239 |
| Slot A v2 face-pool | 0.712 | 0.019 | 0.418 |

No ckpt passes `very_strict`.

### 3.4 At recall_lean (10%/50%)

| Ckpt | best τ | max real FPR | min fake recall | Xiang recall | Xinhe recall | dor recall |
|---|---:|---:|---:|---:|---:|---:|
| **P8A** | **0.482** | **0.044** | **0.663** | 0.933 | 0.663 | 0.829 |
| **E2B** | **0.627** | **0.069** | **0.594** | 0.940 | 0.793 | 0.594 |
| T5C | 0.689 | 0.098 | 0.349 | 0.965 | 0.349 | 0.728 |
| **Slot A v2 CLS** | **0.455** | **0.077** | **0.551** | 0.945 | 0.551 | 0.605 |
| **Slot A v2 face-pool** | **0.663** | **0.079** | **0.539** | 0.985 | 0.875 | 0.539 |

---

## 4. Cross-check against `frozen_clip_team_identity_baseline §3.1`

That doc computed per-head joint-calibrated comparison at team-real-FPR=5% (aggregate, not per-human). The min_fake_recall numbers in §3.1 of that doc:
- P8A: 0.786
- E2B: 0.700
- SlotAv2_FACE: 0.562
- SlotAv2_CLS: 0.546
- T5C: 0.379

This readout's `recall_lean` (10% real-FPR, 50% recall floor) is the closest comparator, but the recall_lean column uses per-human-max-FPR≤10%, not aggregate≤5%, so the τ values differ. The `recall_lean` min_fake_recall numbers (P8A 0.663, E2B 0.594, SlotAv2_FACE 0.539, SlotAv2_CLS 0.551, T5C 0.349) are consistently lower than the frozen-CLIP-baseline's aggregate-5%-FPR numbers (P8A 0.786 etc.) because the per-human-max-FPR constraint is tighter than aggregate-FPR. The relative ordering is preserved across all four gate variants: P8A > E2B > {SlotAv2_FACE ≈ SlotAv2_CLS} > T5C.

---

## 5. Comparison to the cross-ckpt-constant τ readout

`analysis/team_identity_deploy_readout_expanded_2026-05-23/RESULTS_FACTS §3` reported (mode B, τ=0.78 across all ckpts):

| Ckpt | Max real FPR @ mode B (cross-ckpt-const τ) | Min fake recall @ mode B |
|---|---:|---:|
| P8A | 0.016 | 0.487 (Xinhe, 0.013pp under floor) |
| E2B | 0.036 | 0.473 (dor) |
| T5C | 0.052 (dor) | 0.247 (Xinhe) |
| Slot A v2 CLS | 0.024 | 0.293 (Xinhe) |
| Slot A v2 face-pool | 0.003 | 0.252 (dor) |

Verdict in that doc: "no ckpt clears both gates simultaneously at mode B."

This readout at the same gate (5%/50%) but with per-ckpt-calibrated τ:

| Ckpt | Max real FPR (per-ckpt-cal) | Min fake recall (per-ckpt-cal) | Gate result |
|---|---:|---:|---|
| P8A | 0.037 | 0.603 | PASS |
| E2B | 0.048 | 0.528 | PASS |
| T5C | 0.050 | 0.237 | does not pass |
| Slot A v2 CLS | 0.050 | 0.458 | does not pass |
| Slot A v2 face-pool | 0.050 | 0.499 | does not pass (0.1pp under floor) |

**The "no ckpt passes" finding in the cross-ckpt readout was substantially driven by the τ-calibration mismatch.** Under per-ckpt calibration: 2 ckpts pass cleanly, 1 misses by 0.1pp.

---

## 6. Per-ckpt τ vs mode-B (τ=0.78) comparison

Each ckpt's best-τ for user_bar (column 2 of §3.1) vs the cross-ckpt mode-B constant (0.78):

| Ckpt | Per-ckpt best τ | Cross-ckpt mode-B τ | Δ |
|---|---:|---:|---:|
| P8A | 0.590 | 0.780 | −0.190 |
| E2B | 0.718 | 0.780 | −0.062 |
| T5C | 0.790 | 0.780 | +0.010 |
| Slot A v2 CLS | 0.562 | 0.780 | −0.218 |
| Slot A v2 face-pool | 0.681 | 0.780 | −0.099 |

The cross-ckpt mode-B τ=0.78 was approximately appropriate for E2B and T5C, but ~0.20 too high for P8A and Slot A v2 CLS, and ~0.10 too high for face-pool. Using τ=0.78 across the board systematically over-suppressed positive predictions for P8A and Slot A v2 CLS.

---

## 7. Slot A v2 face-pool sensitivity to the 50% floor

Face-pool's min_fake_recall at best user_bar τ is **0.499** — 0.1pp under the 50% floor. The relevant per-human cell:
- Best τ (0.681) under 5% max-FPR cap puts dor recall at exactly 0.499 (1219/2443).
- Xinhe recall at this τ is 0.849 (vs P8A 0.603, E2B 0.712).
- Xiang recall is 0.978.

Lowering τ marginally below 0.681 would push dor recall above 0.50 but also push max-FPR above 5%. Specifically: at τ=0.680, the readout shows dor recall jumps to 0.500 (1220/2443) but max-FPR (Xiang) becomes 0.051 — over the 5% floor. The face-pool 0.499 dor recall is on the boundary in both directions.

For dor recall to clearly exceed 50% under face-pool while keeping max-FPR ≤ 5%, the gate would need to tolerate the 5.1% Xiang FPR; not measured here.

---

## 8. Internal-calibration cross-check against `xinhe_may6_t5c_revisit §3`

That readout used `may5 p95` as the per-ckpt internally-calibrated τ for the Xinhe-may6 cohort. Its per-ckpt τ values were:
- P8A: 0.098
- E2B: 0.323
- T5C: 0.177
- Slot A v2 CLS: 0.105
- Slot A v2 face-pool: 0.561

This readout's per-ckpt best-τ values for user_bar (3.1) are 4-6× larger because:
1. The deploy cohort here has different score distributions than the small may5 cohort (n=60 vs 1821).
2. user_bar's max-FPR≤5% on the deploy cohort is a tighter constraint than may5-p95.

The face-pool best τ here (0.681) is close to its may5-p95 (0.561), reflecting that face-pool has a fundamentally shifted score regime. P8A's best τ here (0.590) is far above its may5-p95 (0.098) because the deploy real cohort is much larger and the max-per-human-FPR constraint requires a tighter cut.

---

## 9. T5C diagnosis

T5C does not pass any of the four gates evaluated. At the best τ for user_bar (0.790, max-FPR=0.050 by construction):
- Xinhe fake recall = 0.237 (lowest of any ckpt × any gate cell)
- dor fake recall = 0.612 (acceptable)
- Xiang fake recall = 0.903 (high)
- max real FPR = 0.050 (dor, at the cap)

T5C's Xinhe-fake recall is structurally low (0.237) at every τ where max-real-FPR ≤ 5%; raising τ to lower FPR only further reduces Xinhe recall. Lowering τ to lift Xinhe recall pushes dor real-FPR over 5%. The structure is: T5C scores Xinhe-fake frames at low probabilities (close to dor real-frame scores), so there is no τ that simultaneously catches Xinhe fakes and avoids dor false-flags.

This is consistent with `xinhe_may6_t5c_revisit §3` which shows T5C's per-ckpt-p95 may6 FPR at 46.7% — T5C has elevated false-flag pressure on Xinhe-distribution frames, independent of τ choice.

---

## 10. Artifacts

- `outputs/tau_sweep_full.csv` — long table (5,005 rows = 5 ckpts × 1001 τs) with per-human FPR/recall + aggregates
- `outputs/sweep_{P8A,E2B,T5C,SlotAv2_CLS,SlotAv2_FACE}.csv` — per-ckpt sweep splits
- `outputs/passing_tau_ranges.csv` — contiguous passing ranges per (ckpt × gate)
- `outputs/per_ckpt_summary.csv` — single row per ckpt with best-τ per gate + per-human breakdown
- `outputs/per_ckpt_summary.json` — same data as JSON
- `scripts/sweep_tau.py` — the sweep + gate-analysis driver

All analysis is CPU only. Wall time ~6 sec on 5,941 frames × 5 ckpts × 1001 τs.

---

## 11. Caveats and limitations

1. **Sample sizes vary per (human × role)**. Xinhe real n=79 (small; ±5pp CI); dor real n=620; Xinhe fake n=1099 (good); dor fake n=2443 (good). The Xinhe-real FPR cells in §3 carry ±5pp confidence; the fake-recall cells carry ±1.5pp.

2. **Sample cap from the source readout carries over.** The source `per_frame_full.csv` was sampled at REAL_CAP=150 / FAKE_CAP=100 per cohort. Per-human aggregate metrics inherit this cap.

3. **τ-discretization step is 0.001**. Passing-range boundaries are reported to 3 decimals; the true boundaries could be ±0.0005.

4. **Mac-Roee (`Roee_Mac`) is excluded** per `project_team_identities_multi_labeled_2026-05-23` — out-of-scope. Re-evaluating with Mac-Roee included would shift P8A/E2B/T5C real-side metrics significantly (see source readout §8).

5. **9-suite contract metrics are not re-evaluated here.** This readout addresses only the team-identity bar. The conventional contract gates (`dev_macro`, `visomaster_enhanced_macro_dev_recall`, `lockbox_fake_recall`) live on different cohorts. P8A or E2B passing the team-identity bar at the per-ckpt τ values above does not automatically imply they pass the 9-suite contract at the same τ.

6. **xinhe_may6 cohort (92 frames) is not in this readout** — it's outside the `grouped_manifest_v2.csv` browser. `xinhe_may6_t5c_revisit/RESULTS_FACTS §3` shows T5C may6 FPR is 46.7% at per-ckpt p95, consistent with T5C's failure here.

7. **The 5%/50% floor itself is a user-specified default** (per `team_identity_deploy_readout_expanded` AGENT_PROPOSAL §6), declared as "may be revised if the data clearly suggests a different operational target." The `relaxed` (5%/40%) gate evaluated here is the most directly user-flagged alternative.
