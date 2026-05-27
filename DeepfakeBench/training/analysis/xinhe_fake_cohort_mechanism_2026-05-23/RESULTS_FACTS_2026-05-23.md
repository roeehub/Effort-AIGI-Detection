# Xinhe-fake cohort mechanism analysis — RESULTS FACTS

Generated 2026-05-23. Factual readout only (forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, breakthrough). Interpretive content lives in `AGENT_PROPOSAL_2026-05-23.md`.

> **Question**: What distinguishes the Xinhe-fake cohorts where all ckpts catch <40% at τ=0.5 (xinhe-fake-1, -2, -3) from the cohorts where all ckpts catch ≥90% (xinhe-fake-7)? Is it identifiable in frozen-CLIP feature space? Does any ckpt's gain on these specifically concentrate on the hard cluster?
>
> **Inputs**: 1,099 Xinhe-fake frames across 13 cohorts from `analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs/per_frame_full.csv`; frozen-CLIP L11 features (768-d) from `analysis/frozen_clip_team_identity_baseline_2026-05-23/outputs/clip_frozen_l11__team_identity_n5941.npz`. 100% feature-cache coverage (0/1099 missing).
>
> **Method**: (1) per-cohort score statistics across 5 ckpts; (2) per-cohort centroid in CLIP feature space; (3) pairwise centroid cosine distance + nearest-neighbor structure; (4) logistic-regression probe for hard-vs-easy class separability; (5) PCA on cohort centroids.

---

## 0. Difficulty classes (a priori from `team_identity_deploy_readout_expanded §6.2`)

- `hard`: xinhe-fake-1, -2, -3 (all ckpts catch <40% at mode B; 286 frames total)
- `easy`: xinhe-fake-7 (all ckpts catch ≥95% at mode B; 66 frames)
- `medium`: xinhe-fake-{4, 5, 6, 8, 8-glasses, 9-glasses, 10-glasses, 11-glasses} (647 frames)
- `other`: extra_xinghe (older non-Teams cohort, recall ~94%; 100 frames)

All `live_prod` cohorts come from `gs://live-fakes-teams-prod/fake/session_20260414_112354/<cohort>/`. Within-session.

---

## 1. Per-cohort score summary (sorted by mean recall @ τ=0.5)

| Cohort | difficulty | n | mean_recall @0.5 | P8A_mean | E2B_mean | T5C_mean | SlotAv2_CLS_mean | SlotAv2_FACE_mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| live_prod__xinhe-fake-2 | hard | 86 | 0.379 | 0.233 | 0.510 | 0.309 | 0.217 | 0.693 |
| live_prod__xinhe-fake-1 | hard | 100 | 0.400 | 0.325 | 0.479 | 0.284 | 0.277 | 0.725 |
| live_prod__xinhe-fake-3 | hard | 100 | 0.472 | 0.449 | 0.469 | 0.374 | 0.325 | 0.707 |
| live_prod__xinhe-fake-5 | medium | 88 | 0.677 | 0.562 | 0.852 | 0.447 | 0.541 | 0.795 |
| live_prod__xinhe-fake-6 | medium | 82 | 0.688 | 0.756 | 0.639 | 0.466 | 0.553 | 0.786 |
| live_prod__xinhe-fake-8-glasses | medium | 100 | 0.716 | 0.721 | 0.844 | 0.551 | 0.355 | 0.693 |
| live_prod__xinhe-fake-4 | medium | 100 | 0.772 | 0.644 | 0.862 | 0.565 | 0.662 | 0.809 |
| live_prod__xinhe-fake-9-glasses | medium | 60 | 0.777 | 0.605 | 0.933 | 0.583 | 0.618 | 0.805 |
| live_prod__xinhe-fake-11-glasses | medium | 44 | 0.805 | 0.641 | 0.926 | 0.610 | 0.659 | 0.794 |
| live_prod__xinhe-fake-8 | medium | 100 | 0.824 | 0.817 | 0.914 | 0.645 | 0.569 | 0.772 |
| live_prod__xinhe-fake-10-glasses | medium | 73 | 0.836 | 0.736 | 0.913 | 0.644 | 0.669 | 0.794 |
| live_prod__xinhe-fake-7 | easy | 66 | 0.918 | 0.956 | 0.940 | 0.687 | 0.646 | 0.844 |
| extra_xinghe | other | 100 | 0.936 | 0.867 | 0.878 | 0.788 | 0.805 | 0.821 |

---

## 2. Aggregate per-difficulty means (each ckpt × difficulty class)

| Ckpt | hard mean ± std (n=286) | easy mean ± std (n=66) | medium mean ± std (n=647) | hard → easy Δ |
|---|:---:|:---:|:---:|---:|
| P8A | 0.341 ± 0.324 | 0.956 ± 0.068 | 0.692 ± 0.308 | +0.615 |
| E2B | 0.485 ± 0.336 | 0.940 ± 0.072 | 0.854 ± 0.223 | +0.455 |
| T5C | 0.323 ± 0.211 | 0.687 ± 0.168 | 0.560 ± 0.245 | +0.364 |
| SlotAv2_CLS | 0.276 ± 0.227 | 0.646 ± 0.216 | 0.566 ± 0.280 | +0.370 |
| **SlotAv2_FACE** | **0.709 ± 0.065** | **0.844 ± 0.027** | **0.777 ± 0.074** | **+0.135** |

**Face-pool's score variance across difficulty classes is dramatically smaller than the other ckpts** (Δ hard→easy = +0.135 vs P8A's +0.615). Face-pool gives ~0.70+ to most Xinhe-fake frames regardless of cohort; the other ckpts compress scores on hard cohorts.

---

## 3. Hard vs easy LR probe in frozen-CLIP space

Logistic regression on per-frame CLIP L11 features (768-d), labels hard (1, n=286) vs easy (0, n=66).

| Probe | Score |
|---|---:|
| In-sample AUC | **1.0000** |
| 5-fold CV AUC | **1.0000 ± 0.0000** |

**Hard vs easy is perfectly linearly separable in frozen-CLIP feature space**, even with 5-fold cross-validation. There is a single CLIP-direction that distinguishes the cohorts at 100% accuracy.

Projection of all 13 cohorts onto this LR-defined hard-axis (signed margin; positive = looks like hard cohorts):

| Cohort | mean hard-axis margin | std | difficulty |
|---|---:|---:|---|
| xinhe-fake-8-glasses | 7.766 | 0.838 | medium |
| xinhe-fake-5 | 7.526 | 0.773 | medium |
| **xinhe-fake-2** | **7.268** | **0.854** | **hard** |
| **xinhe-fake-1** | **6.980** | **0.861** | **hard** |
| xinhe-fake-9-glasses | 6.872 | 0.920 | medium |
| xinhe-fake-10-glasses | 6.812 | 0.805 | medium |
| **xinhe-fake-3** | **6.511** | **1.080** | **hard** |
| extra_xinghe | 6.435 | 1.799 | other |
| xinhe-fake-4 | 6.375 | 1.293 | medium |
| xinhe-fake-11-glasses | 6.039 | 1.054 | medium |
| xinhe-fake-8 | −0.930 | 0.732 | medium |
| xinhe-fake-6 | −1.680 | 1.432 | medium |
| **xinhe-fake-7** | **−5.139** | **0.578** | **easy** |

Two clusters separated by the hard-axis at margin ~0:
- **High cluster** (margin > 6): xinhe-fake-{1, 2, 3, 4, 5, 8-glasses, 9-glasses, 10-glasses, 11-glasses}, extra_xinghe — 10 of 13 cohorts
- **Low cluster** (margin < 0): xinhe-fake-{6, 7, 8} — 3 of 13 cohorts

The hard/medium/easy labels from §0 partially track this binary geometric split, but not perfectly: extra_xinghe and several medium cohorts also sit on the "hard" side geometrically, even though their measured recall is medium-to-high.

---

## 4. Cohort centroid distances (CLIP L11 cosine distance)

Nearest-cohort neighbors per centroid:

| Cohort | NN1 | dist | NN2 | dist | NN3 | dist |
|---|---|---:|---|---:|---|---:|
| extra_xinghe | xinhe-fake-3 | 0.028 | xinhe-fake-4 | 0.034 | xinhe-fake-1 | 0.035 |
| xinhe-fake-1 | xinhe-fake-11-glasses | 0.011 | xinhe-fake-4 | 0.016 | xinhe-fake-3 | 0.017 |
| xinhe-fake-10-glasses | xinhe-fake-4 | 0.023 | xinhe-fake-5 | 0.024 | xinhe-fake-9-glasses | 0.024 |
| xinhe-fake-11-glasses | xinhe-fake-1 | 0.011 | xinhe-fake-4 | 0.019 | xinhe-fake-9-glasses | 0.019 |
| xinhe-fake-2 | xinhe-fake-1 | 0.032 | xinhe-fake-3 | 0.044 | xinhe-fake-11-glasses | 0.048 |
| xinhe-fake-3 | xinhe-fake-1 | 0.017 | xinhe-fake-4 | 0.020 | xinhe-fake-11-glasses | 0.024 |
| xinhe-fake-4 | xinhe-fake-1 | 0.016 | xinhe-fake-11-glasses | 0.019 | xinhe-fake-3 | 0.020 |
| xinhe-fake-5 | xinhe-fake-1 | 0.018 | xinhe-fake-9-glasses | 0.019 | xinhe-fake-4 | 0.024 |
| **xinhe-fake-6** | **xinhe-fake-7** | **0.030** | xinhe-fake-8 | 0.063 | xinhe-fake-3 | 0.066 |
| **xinhe-fake-7** | **xinhe-fake-6** | **0.030** | xinhe-fake-8 | 0.089 | xinhe-fake-11-glasses | 0.090 |
| xinhe-fake-8 | xinhe-fake-6 | 0.063 | xinhe-fake-8-glasses | 0.073 | xinhe-fake-7 | 0.089 |
| xinhe-fake-8-glasses | xinhe-fake-2 | 0.054 | xinhe-fake-9-glasses | 0.055 | xinhe-fake-1 | 0.056 |
| xinhe-fake-9-glasses | xinhe-fake-5 | 0.019 | xinhe-fake-11-glasses | 0.020 | xinhe-fake-10-glasses | 0.024 |

Centroid geometry shows two distinct clusters:
- **Cluster A** (the "hard" geometric cluster): xinhe-fake-{1, 2, 3, 4, 5, 8-glasses, 9-glasses, 10-glasses, 11-glasses} + extra_xinghe — all near each other (cosine distance 0.01-0.06)
- **Cluster B** (the "easy" geometric cluster): xinhe-fake-{6, 7, 8} — mutually closest (xinhe-fake-6 ↔ xinhe-fake-7 at 0.030, an order of magnitude tighter than their distances to cluster A)

xinhe-fake-7 (the easy cohort) is closest to xinhe-fake-6 (medium), not to any other "easy" cohort — there's just one tight cluster of 3.

---

## 5. PCA on cohort centroids

5-PC PCA on the 13 centroid vectors (centered):

| Component | Explained variance |
|---|---:|
| PC1 | 40.8% |
| PC2 | 23.5% |
| PC3 | 10.2% |
| PC4 | 7.5% |
| PC5 | 4.3% |

PC1 (40.8% of cohort variance) separation:

| Cohort | PC1 | difficulty |
|---|---:|---|
| **xinhe-fake-7** | **+4.216** | **easy** |
| xinhe-fake-6 | +3.316 | medium |
| xinhe-fake-8 | +3.149 | medium |
| xinhe-fake-2 | −0.130 | hard |
| extra_xinghe | −0.778 | other |
| xinhe-fake-3 | −0.810 | hard |
| xinhe-fake-11-glasses | −0.863 | medium |
| xinhe-fake-1 | −0.944 | hard |
| xinhe-fake-4 | −0.964 | medium |
| xinhe-fake-8-glasses | −1.067 | medium |
| xinhe-fake-10-glasses | −1.575 | medium |
| xinhe-fake-9-glasses | −1.645 | medium |
| xinhe-fake-5 | −1.906 | medium |

PC1 cleanly separates the geometric easy cluster (3 cohorts with PC1 > +3) from the hard cluster (10 cohorts with PC1 < 0). This is the same axis the LR probe identified.

---

## 6. The face-pool / hard-cluster specific gain

§2 shows face-pool's mean score on hard cohorts is 0.709 vs P8A's 0.341 — a +37pp Xinhe-fake gain concentrated on the hard cluster. Decomposing this gain by cohort:

| Cohort | difficulty | P8A mean | SlotAv2_FACE mean | Face_pool − P8A | Face_pool gain direction |
|---|---|---:|---:|---:|---|
| xinhe-fake-2 | hard | 0.233 | 0.693 | +0.460 | gain |
| xinhe-fake-1 | hard | 0.325 | 0.725 | +0.400 | gain |
| xinhe-fake-3 | hard | 0.449 | 0.707 | +0.258 | gain |
| xinhe-fake-5 | medium | 0.562 | 0.795 | +0.233 | gain |
| xinhe-fake-4 | medium | 0.644 | 0.809 | +0.165 | gain |
| xinhe-fake-11-glasses | medium | 0.641 | 0.794 | +0.153 | gain |
| xinhe-fake-9-glasses | medium | 0.605 | 0.805 | +0.200 | gain |
| xinhe-fake-10-glasses | medium | 0.736 | 0.794 | +0.058 | gain |
| xinhe-fake-8 | medium | 0.817 | 0.772 | −0.045 | neutral |
| xinhe-fake-8-glasses | medium | 0.721 | 0.693 | −0.028 | neutral |
| xinhe-fake-6 | medium | 0.756 | 0.786 | +0.030 | neutral |
| **xinhe-fake-7** | **easy** | **0.956** | **0.844** | **−0.112** | **loss** |
| extra_xinghe | other | 0.867 | 0.821 | −0.046 | neutral |

Face-pool's largest gains are on the hardest cohorts (xinhe-fake-1 +40pp, xinhe-fake-2 +46pp). On easy cohorts (xinhe-fake-7) it slightly under-performs P8A.

---

## 7. Artifacts

- `outputs/per_cohort_score_summary.csv` — per-cohort score stats × ckpt
- `outputs/centroid_cosine_distance.csv` — 13×13 pairwise centroid cosine distance matrix
- `outputs/centroid_nn.csv` — nearest-neighbor cohorts per centroid
- `outputs/hard_vs_easy_lr_coef.npy` — 768-d LR coefficient vector (the "hard axis" in CLIP space)
- `outputs/hard_axis_per_cohort.csv` — per-cohort LR-margin projection
- `outputs/cohort_pca_projection.csv` — PCA PC1-PC5 per cohort

Wall time: ~5 sec on 1099 frames × 13 cohorts.

---

## 8. Caveats

1. **Same session**: all `live_prod` cohorts come from `session_20260414_112354`. Within-session cohort separation in CLIP space measures something below "different swap method" — could be source-identity, background, lighting moment, frame-pose, etc. The 1.0 AUC reflects a separable signal in CLIP space but doesn't identify what the signal IS at the image level.

2. **Sample size**: easy cluster (geometric: xinhe-fake-{6, 7, 8}) n=248; hard cluster n=851. LR probe trained on the strict-label subset (n=352).

3. **Per-cohort sample cap from source readout** (FAKE_CAP=100) means cohorts at full size have 80-100 frames; fewer for some (-7 has 66; -11-glasses 44).

4. **PC1 of cohort centroids does NOT necessarily equal the LR hard-vs-easy axis** — PC1 is unsupervised; LR is supervised. They agree directionally (both put xinhe-fake-7 furthest from the "hard" cluster), but the magnitudes are not directly comparable.

5. **Image-level inspection not done.** What specifically distinguishes the hard cluster geometrically in CLIP space (face identity? background? lighting?) is not measured here. Would require pulling frame thumbnails and visual inspection.

6. **No cross-cohort score axis comparison.** Whether the "hard axis" in CLIP space aligns with any known shortcut axis (sharpness, identity, capture-mode) is not measured here.
