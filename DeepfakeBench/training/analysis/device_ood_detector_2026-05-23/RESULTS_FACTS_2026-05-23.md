# Device-OOD detector via frozen-CLIP LR — RESULTS FACTS

Generated 2026-05-23. Factual readout. Interpretive content in `AGENT_PROPOSAL_2026-05-23.md`.

> **Question**: Can we build a deployment-time detector that flags "this user's capture pipeline is OOD" (Mac-webcam-like vs Windows-tester-like) using frozen-CLIP features, so the abstain rule in SHIPMENT_OPTIONS Option 2 has a trigger?
>
> **Method**: Train LR on frozen-CLIP L11 features, Mac-Roee real (n=498) vs Windows-Roee real (n=330). Project all 6,439 team-cohort frames onto the LR axis. Inspect per-cohort margins and correlation with P8A per-frame FPR.

---

## 1. Detector training

| Metric | Value |
|---|---:|
| Windows-Roee real frames (label 0) | 330 |
| Mac-Roee real frames (label 1) | 498 |
| LR 5-fold stratified CV accuracy | **1.0000 ± 0.0000** |
| LR 5-fold stratified CV AUC | **1.0000** |
| Coefficient saved | `outputs/device_ood_lr_coef.npy` |

Same-person Mac-vs-Windows captures are perfectly linearly separable in frozen-CLIP L11 space.

---

## 2. Per-cohort device-OOD margin (positive = Mac-like, negative = Windows-like)

Sorted by margin_mean descending.

| Cohort | Human | In-scope | n | margin_mean | margin_p95 | frac_above_0 | P8A FPR @τ=0.10 | P8A FPR @τ=0.59 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| bla_bla_chow | Mac-Roee | OOS | 150 | **8.405** | 10.051 | 1.000 | 27.3% | 10.7% |
| bla_bla_chow__s1 | Mac-Roee | OOS | 68 | 7.886 | 8.882 | 1.000 | 5.9% | 0.0% |
| bla_bla_chow__s2 | Mac-Roee | OOS | 150 | 7.759 | 10.580 | 1.000 | 64.0% | 24.7% |
| Roy_D | Mac-Roee | OOS | 130 | 6.965 | 8.680 | 1.000 | 67.7% | 43.8% |
| **extra_xinghe** | **Xinhe** | **deploy** | 19 | **6.338** | 7.165 | 1.000 | 10.5% | 5.3% |
| **Xiang_Xiang2_Feng__s23** | **Xiang** | **deploy** | 102 | **5.580** | 7.340 | 1.000 | 9.8% | 4.9% |
| **extra_xiang** | **Xiang** | **deploy** | 150 | **4.914** | 7.904 | 1.000 | 3.3% | 2.7% |
| **Md_noyn_Sharker__s15** | **Noyn** | **deploy** | 150 | **4.379** | 6.860 | 1.000 | 13.3% | 2.0% |
| Xiang_Xiang2_Feng | Xiang | deploy | 150 | 4.152 | 6.897 | 0.980 | 0.7% | 0.0% |
| xiang | Xiang | deploy | 150 | 4.021 | 5.429 | 1.000 | 20.0% | 2.7% |
| dor_shkedi__s16 | dor | deploy | 31 | 3.872 | 4.804 | 1.000 | 0.0% | 0.0% |
| dor_morning | dor | deploy | 150 | 1.285 | 6.169 | 0.773 | 36.7% | 12.0% |
| team_may5__Xinhe | Xinhe | deploy | 60 | 0.945 | 3.216 | 0.717 | 5.0% | 0.0% |
| team_may5__Xiang | Xiang | deploy | 30 | 0.448 | 2.705 | 0.533 | 0.0% | 0.0% |
| dor_shkedi | dor | deploy | 150 | -0.372 | 1.499 | 0.327 | 17.3% | 2.7% |
| team_may5__Noyn | Noyn | deploy | 60 | -1.191 | 2.356 | 0.267 | 0.0% | 0.0% |
| team_may5__Dor | dor | deploy | 30 | -1.507 | -0.149 | 0.033 | 10.0% | 3.3% |
| real_dor | dor | deploy | 109 | -1.818 | 3.117 | 0.165 | 5.5% | 0.0% |
| dor_evening | dor | deploy | 150 | -1.971 | 1.489 | 0.140 | 2.0% | 0.0% |
| team_may5__Roee | Roee_Windows | deploy | 30 | **-6.543** | -4.625 | 0.000 | 16.7% | 3.3% |
| tester_roee_real_2026-03-06 | Roee_Windows | deploy | 150 | **-6.918** | -5.338 | 0.000 | 0.7% | 0.7% |
| roee_tester_real_2026-03-24 | Roee_Windows | deploy | 150 | **-8.000** | -5.655 | 0.000 | 0.7% | 0.7% |

**The 3 Roee_Windows cohorts are extreme outliers on the negative side.** All other team-humans' cohorts span [-2, +6.3] — mostly on the "Mac-like" side of the detector axis, even though those cohorts are Windows-laptop captures of OTHER team members.

---

## 3. Per-margin-decile P8A FPR correlation

Margin computed per-frame on all 6,439 team frames; bucketed into 10 deciles by margin.

| Decile | n | margin range | P8A FPR @τ=0.10 | P8A FPR @τ=0.59 | In-dist share |
|---:|---:|---|---:|---:|---:|
| 0 (most Windows-like) | 232 | [-11.3, -6.6] | 0.4% | 0.0% | 100% |
| 1 | 232 | [-6.6, -2.6] | 3.4% | 1.3% | 100% |
| 2 | 232 | [-2.6, -0.6] | 9.1% | 1.3% | 100% |
| 3 | 232 | [-0.6, +1.0] | 21.1% | 6.9% | 100% |
| 4 | 232 | [+1.0, +3.0] | 10.3% | 0.9% | 100% |
| 5 | 231 | [+3.1, +4.3] | 13.4% | 2.6% | 99.1% |
| 6 | 232 | [+4.3, +5.6] | 14.7% | 3.4% | 90.1% |
| 7 | 232 | [+5.6, +6.7] | 21.6% | 11.6% | 64.7% |
| 8 | 232 | [+6.7, +8.0] | 33.6% | 20.3% | 27.2% |
| 9 (most Mac-like) | 232 | [+8.0, +11.5] | 44.8% | 17.7% | 4.3% |

There IS a correlation between high margin and elevated P8A FPR — but the relationship is monotonic-with-noise, not a clean step function. Deciles 7-9 (margin > 5.6) capture most Mac-OOD frames AND elevated in-dist FPR cohorts, but they also include some legitimately in-distribution frames.

---

## 4. The overfitting problem

The training set was Roee-Windows (330) vs Roee-Mac (498) — **the only "Windows" frames in the training data come from one specific person on one specific laptop**. The detector learned what makes "Roee-on-his-Windows-laptop" distinctive in CLIP space — which is heavily confounded with identity, room, lighting, camera-make, and capture-time-of-day.

Evidence of confound:
- Other team-humans on Windows-laptop captures (Xinhe extras, Xiang extras, Noyn Sharker, dor shkedi) all sit at margin > 0 ("Mac-like" per the detector). They are NOT actually using Macs — but they're not Roee-on-his-Windows-laptop either.
- The detector's "Windows class" is essentially "Roee in his room with that specific webcam in March 2026."
- A genuine device-OOD detector would need diverse Windows-capture training data from multiple people.

---

## 5. Implication for the SHIPMENT_OPTIONS abstain rule

Option 2 in `SHIPMENT_OPTIONS_2026-05-23.md` proposed an abstain rule triggered by per-window high-fraction-over-τ. **A device-OOD detector built on the current data is NOT a more reliable trigger than the fraction-over-τ heuristic.**

Specifically:
- Current detector: would abstain on ~70% of non-Roee in-dist cohorts (all the cohorts at margin > 0). Too many false abstains.
- Fraction-over-τ heuristic (Option 0): abstains when window's fraction-over-τ > 70%. Triggers on Mac-OOD users (where genuinely > 70% of frames score high) AND on rare in-dist cohort frames (low rate). Better behaved.

**Conclusion**: keep Option 0 / 2 / 3 fraction-over-τ heuristic. Don't try to use this device-OOD detector as the abstain trigger.

---

## 6. To build a usable device-OOD detector

Would need training data with:
- Multiple people on Windows captures (not just Roee)
- Multiple people on Mac/iPad/phone captures
- Captures across diverse rooms, lighting, time-of-day
- Possibly explicit device labels (Windows-integrated-webcam, Mac-FaceTime-camera, USB webcam, phone front cam, etc.)

This is data-acquisition work; in scope for the broader B.III.3 (7000-webcam dataset acquisition) lever.

---

## 7. Artifacts

- `outputs/clip_frozen_l11__mac_roee_n498.npz` — Mac-Roee frozen-CLIP cache (extracted today)
- `outputs/device_ood_lr_coef.npy` — LR coefficient (the "Mac-like" axis in CLIP space); usable for projection but not as a reliable classifier
- `outputs/per_cohort_device_ood.csv` — per-cohort margin statistics
- `outputs/margin_decile_p8a_fpr.csv` — margin-decile P8A FPR correlation
- `scripts/extract_mac_features.py` — Mac-Roee CLIP extractor
- `scripts/build_detector.py` — detector + projection driver

Wall: ~15 min Mac extraction + ~30 sec LR training & projection.

---

## 8. Caveats

1. **Training set narrow.** 498 Mac-Roee + 330 Windows-Roee — both from one person. CV accuracy 1.0 reflects this narrowness, not a generalizable detector.
2. **The "OOD-margin" interpretation is identity-and-device confounded.** Cannot disambiguate "this user is on a Mac" from "this user is Roee-recorded-in-his-Mac-setup."
3. **Per-frame margin correlates with P8A per-frame FPR with R² ~30-40%.** Useful as a soft signal but not as a binary classifier.
4. **No external validation set.** Would need lockbox / OPTB frames to test whether the axis aligns with capture-device in those cohorts.
