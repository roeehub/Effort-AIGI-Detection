# IQ-axis fake-frame cross-check — RESULTS FACTS

Generated 2026-05-24. Factual readout. Companion to `RESULTS_FACTS_2026-05-23.md` (real-frame analysis). Interpretive content in §6 below ("Verdict implications").

> **Question**: The IQ detector trained on Roee-Mac vs Roee-Windows (REAL frames) classifies Mac-like frames as P8A-risky. **If we apply an IQ-margin abstain rule, does it also suppress P8A's correct fake detections?**
>
> **Method**: Extract IQ features on all 4,120 fake frames (3 fake-target roles × 52 base_identity cohorts). Project them onto the same IQ-margin axis used for reals (re-fit on Roee-Mac vs Roee-Windows REAL frames; fakes held out). Quantify: per-cohort margin distribution, per-decile fake-recall, abstain-threshold cost curve.

---

## 0. Extraction stats

| Metric | Value |
|---|---:|
| Fake frames | 4,120 |
| By role | dor 2,443 / Xinhe 1,099 / Xiang 578 |
| By deploy_relevant | True 4,120, False 0 (all in-distribution targets) |
| Local cache hits | 778 (19%) |
| GCS downloads | 3,342 (81%) |
| Failures | 0 |
| Wall time | ~13 min |

---

## 1. Headline finding — fakes are NOT uniformly Windows-like

The hopeful outcome was: fakes are Windows-like → abstain rule selectively drops bad reals. **Reality is split.**

| Fake-role | n | margin_mean | frac_mac_like (>0) | frac_extreme_mac (>+3) | P8A recall @τ=0.10 |
|---|---:|---:|---:|---:|---:|
| **fake_target_dor** | 2,443 | **+1.52** | **61.0%** | **18.6%** | 91.2% |
| fake_target_Xiang | 578 | -2.72 | 5.5% | 0.0% | 96.5% |
| fake_target_Xinhe | 1,099 | -5.87 | 0.0% | 0.0% | 86.5% |
| **Reals (real_clean)** | 2,319 | (mixed) | 42.2% | **31.9%** | 17.3% (FPR) |

**Key observations:**
- **Xinhe fakes** are strongly Windows-like (margin -4 to -8). An IQ-margin abstain rule does NOT touch them. 
- **Xiang fakes** are mostly Windows-like (5.5% Mac-like). Mostly safe from abstain.
- **dor fakes are bimodal**: 61% Mac-like, with 18.6% in the "extreme Mac" tail (margin > +3). Abstain would hit these hard.

The **selectivity ratio is 2.89×** (31.9% extreme-Mac reals vs 11.0% extreme-Mac fakes). The detector tilts toward selectively removing reals, but not cleanly.

---

## 2. Per-fake-cohort margin distribution — where the recall cost concentrates

There are 7 dor fake-cohorts where ALL frames are extreme-Mac (margin > +6), and 100% of those frames are detected by P8A at τ=0.10. An abstain rule at margin>+3 would discard all of them — these cohorts become completely undetectable.

| Cohort | n | margin_mean | margin_p50 | frac_above_p3 | P8A recall @τ=0.10 | P8A recall @τ=0.59 |
|---|---:|---:|---:|---:|---:|---:|
| dor_shkedi__s16 | 78 | **9.56** | 9.50 | **100%** | 100% | 97.4% |
| dor_fake_trump_regular | 44 | **8.83** | 8.83 | **100%** | 100% | 100% |
| dor_fake_bill_gates_regular | 24 | **8.53** | 8.48 | **100%** | 100% | 95.8% |
| dor_fake_chinese_regular | 33 | **8.26** | 8.18 | **100%** | 100% | 100% |
| dor_fake_face_1_enhanced | 64 | **7.81** | 7.93 | **100%** | 100% | 100% |
| dor_fake_face_1_regular | 53 | **7.72** | 8.14 | **100%** | 100% | 100% |
| dor_fake_chinese_enhanced | 34 | **6.87** | 6.68 | **100%** | 100% | 100% |
| dor_fake_vladi_enhanced | 82 | 3.07 | 0.14 | 48.8% | 97.6% | 81.7% |
| dor_fake_elone_enhanced | 47 | 3.06 | 1.95 | 25.5% | 100% | 100% |
| dor_fake_vladi_regular | 100 | 2.91 | 1.69 | 40.0% | 86.0% | 64.0% |
| dor_fake_mark_zukerberk_enhanced | 70 | 2.77 | 1.76 | 31.4% | 100% | 100% |
| dor_fake_elone_regular | 24 | 2.14 | 2.14 | 41.7% | 100% | 100% |
| dor_fake_mark_zukerberk_regular | 30 | 1.06 | 1.08 | 0.0% | 100% | 100% |
| ... (lower-margin dor fakes — mostly Windows-side) | | | | 0% | 60-98% | 35-94% |
| (all 16 Xiang/Xinhe fake-cohorts) | varied | -1.6 to -7.8 | | 0% | 56-100% | 9-100% |

**The 7 extreme-Mac dor cohorts represent 330 fake frames (8% of all fakes). At margin>+3 abstain, P8A loses 100% of its detection capacity on these — going from 95-100% recall to 0% recall.**

These are mostly "trump/bill-gates/chinese/face_1" swaps. The "live_prod__xinhe-*", "dor_fake_inswapper_*", "dor_fake_ghostface_*", and Xiang fakes are all safe.

---

## 3. Per-margin-decile fake-recall vs real-FPR

Decile boundaries computed from the COMBINED reals+fakes distribution (10 equal-population buckets):

| Decile | margin range | n_real | real_FPR @τ=0.10 | real_FPR @τ=0.59 | n_fake | fake_recall @τ=0.10 | fake_recall @τ=0.59 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 (most Windows) | [-31.8, -6.8] | 311 | 4.5% | 1.3% | 333 | 91.3% | 63.4% |
| 1 | [-6.8, -4.8] | 155 | 5.2% | 1.3% | 489 | 82.8% | 53.0% |
| 2 | [-4.8, -3.1] | 234 | 6.0% | 1.3% | 410 | 90.7% | 73.4% |
| 3 | [-3.1, -1.7] | 305 | 7.9% | 1.6% | 339 | 90.6% | 85.0% |
| 4 | [-1.7, -0.6] | 230 | 5.7% | 0.9% | 414 | 76.8% | 56.8% |
| 5 | [-0.6, -0.05] | 96 | 9.4% | 1.0% | 547 | 92.1% | 76.6% |
| 6 | [-0.05, +0.45] | 107 | 8.4% | 0.0% | 537 | 92.4% | 78.8% |
| 7 | [+0.45, +1.86] | 87 | 13.8% | 4.6% | 557 | 96.8% | 89.2% |
| 8 (Mac-like) | [+1.86, +7.35] | 515 | **40.0%** | **17.9%** | 129 | **100%** | **100%** |
| 9 (most Mac) | [+7.35, +15.9] | 279 | **32.6%** | **14.3%** | 365 | **99.5%** | **97.3%** |

**Decile 8-9 (margin > +1.86) is where the crisis is**: real FPR balloons to 33-40% AND fake recall is at 100%. Abstaining on these deciles drops both signals together.

---

## 4. Abstain-threshold cost curve (deploy_relevant only — production-realistic)

If we abstain when iq_margin > T, what fraction of reals vs fakes get dropped? **Ratio > 1 = good (selective)**, **ratio < 1 = bad (suppresses fakes more)**.

| T | frac_real_abstain | frac_fake_abstain | ratio | retained real_FPR @τ=0.59 | retained fake_recall @τ=0.59 |
|---:|---:|---:|---:|---:|---:|
| -2 | 48.7% | 63.7% | 0.76 | 1.4% | 67.8% |
| -1 | 36.4% | 56.3% | 0.65 | 1.3% | 66.8% |
| 0 | 26.4% | 36.9% | 0.72 | 1.3% | 67.8% |
| +1 | 18.9% | 17.2% | 1.10 | 1.2% | 71.3% |
| **+2** | **16.0%** | **11.7%** | **1.37** | **1.4%** | **72.7%** |
| **+3** | **13.7%** | **11.0%** | **1.24** | **1.5%** | **72.9%** |
| +4 | 8.5% | 10.8% | 0.79 | 1.6% | 73.0% |
| +5 | 5.0% | 10.8% | **0.46** | 2.1% | 73.0% |
| +6 | 4.2% | 10.1% | **0.41** | 2.1% | 73.2% |

**Baseline (no abstain), deploy_relevant only:** real_FPR@τ=0.59 = **2.36%**, fake_recall@τ=0.59 = **75.66%**.

The peak selectivity is at T=+2 (ratio 1.37×). At T=+2:
- Real FPR drops from 2.36% → 1.4% (**40% relative reduction**)
- Fake recall drops from 75.66% → 72.7% (**3.9% relative reduction**)
- 11.7% of fakes are silenced

At T=+5 or higher, the rule **inverts** — it suppresses more fakes than reals (ratio < 1).

---

## 5. Per-frame correlation: iq_margin → P8A score (within fakes)

| Subset | n | Spearman ρ | p-value |
|---|---:|---:|---|
| All fakes | 4,120 | **+0.40** | 7.6e-157 |
| Deploy fakes (= all fakes) | 4,120 | +0.40 | 7.6e-157 |

The IQ-margin axis is **positively correlated with P8A score within fakes** (ρ=0.40), the same direction as within reals (ρ=0.35-0.52). This means: **Mac-like fakes are systematically P8A's MORE-confident fake detections**, not the borderline-difficulty ones. Abstaining on them loses P8A's strongest fake-detection signals.

This is the structural reason the abstain rule is not clean: IQ-margin is not orthogonal to "true fakeness." Both signals point the same way, so dropping Mac-like frames drops good detections along with bad ones.

---

## 6. Verdict implications (interpretive)

### What the abstain rule actually does

At the best operating point (T=+2 on deploy-relevant subset):
- **Aggregate trade looks favorable**: 40% FPR reduction, 4% recall reduction (10:1 ratio)
- **Cohort-level damage is concentrated**: 7 specific dor fake-cohorts (trump, bill_gates, chinese, face_1, chinese_enhanced) are 100% extreme-Mac and would be 100% silenced. These are not edge cases — they're complete blind spots
- **The Mac-Roee abstain "wins" are out-of-scope**: Roee deploys on Windows; abstaining on the 4 Mac-Roee cohorts removes frames that wouldn't be in production anyway

### Why aggregate ≠ deployment-safe

The 4% recall loss is averaged across 4,120 fake frames; the loss is NOT uniformly distributed. It's concentrated on specific fake methodologies. From a security standpoint, total blindness to "trump/bill_gates/chinese deepfake swaps of dor" is worse than 4% recall loss elsewhere.

### Recommendation

**Do NOT ship the IQ-margin abstain rule as a hard abstain.** The aggregate trade looks acceptable, but the per-cohort variance creates complete blind spots for ~7 fake-cohort types.

**Acceptable uses of the IQ-margin signal:**

1. **Operator alert / telemetry** — flag when a deployed user's frames score high IQ-margin so the operator knows the model is operating OOD. Don't take automatic action.
2. **Soft down-weighting in majority vote** — instead of dropping Mac-like frames, weight them by `1 / (1 + exp(iq_margin - 2))` in the MV aggregation. Marginal frames count less but aren't silenced.
3. **τ shift instead of abstain** — for Mac-like frames, apply a higher τ (e.g., +0.05) so they need more confidence to flag. Preserves recall on extreme-Mac fakes while reducing FPR on extreme-Mac reals.
4. **Per-user calibration aid** — collect IQ-margin per user; if user X is persistently Mac-like, recalibrate τ for that user. Requires longitudinal telemetry.

### What this means for the SHIPMENT_OPTIONS doc

**Confirms the Option 1 recommendation (τ=0.59 simple-majority).** The IQ-abstain trigger is NOT a safe Option 2 enhancement as currently designed. Option 2 should either drop the IQ-abstain idea or replace it with one of the soft-signal uses above.

---

## 7. Caveats

1. **The 7 extreme-Mac dor fake-cohorts** may share an upstream property (source video resolution, codec, swap-engine output) that the IQ detector keys on but isn't actually device-related. Specifically `dor_fake_trump_regular`, `dor_fake_bill_gates_regular`, `dor_fake_chinese_regular`, `dor_fake_face_1_regular`, etc. are different swap subjects but produce similar IQ profiles — could be that they share a source video processing pipeline.
2. **dor_shkedi__s16 (margin 9.56, 100% recall)** is structurally unusual: it has high margin (= Mac-like IQ) AND 100% P8A recall AND 0% real-cohort FPR (from §2 of `RESULTS_FACTS_2026-05-23.md`). The real-side dor_shkedi__s16 cohort is the only "true false-positive" of the IQ detector on reals; the fake-side dor_shkedi__s16 cohort is also Mac-like. Suggests these are recaptures sharing the same physical capture pipeline.
3. **No analysis of dor_fake_vladi_enhanced** despite its bimodal margin distribution (mean 3.07, median 0.14, p95 7.68). Worth a per-frame look to understand the split.
4. **The "deploy-relevant" qualifier doesn't fully filter the right thing**: all 4,120 fakes are flagged deploy_relevant=True, but the Mac-Roee reals (498 frames, ~21% of real_clean) are deploy_relevant=False. So §4's "deploy_relevant only" curve has the right real-side filter but the fake side is unchanged.
5. **The axis was fit on a single person.** A multi-person device axis (Mac-N vs Windows-N for N users) would give a different (likely cleaner) decision boundary.

---

## 8. Artifacts

- `outputs/per_frame_iq_fakes_v2.parquet` — 4,120 fake frames × 30 columns
- `outputs/per_fake_cohort_iq_margin.csv` — 52 fake-cohort margin statistics
- `outputs/per_decile_fake_vs_real.csv` — combined-decile fake-recall vs real-FPR
- `outputs/abstain_threshold_cost_curve.csv` — full-dataset abstain curve
- `outputs/abstain_threshold_cost_curve_deploy.csv` — deploy-only abstain curve
- `scripts/extract_iq_fakes.py` — fake-frame IQ extraction
- `scripts/analyze_fakes_on_iq_axis.py` — projection + analysis
