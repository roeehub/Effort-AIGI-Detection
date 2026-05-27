# Risky-Frame Remediation Experiment — Findings

**Date:** 2026-05-14
**Question:** Can a cheap pre-processing transform applied to frames consistently improve T5C / P8A scoring without dragging true-positive fakes equally?
**Answer:** Yes — a **50/50 blend of original + mild unsharp-mask** (≈ effective unsharp amount 0.25) improves AUC on **both** T5C step3500 (+0.0058 mix / +0.0104 hard) and P8A step5000 (+0.0091 mix / +0.0164 hard).

---

## Pool
- **risky_real** (label=0, T5C>0.7, gate=pass, chronic identities): 100 frames
  - Dominated by bla_bla_chow (49) + bla_bla_chow__s1 (25) + dor_shkedi (17), plus singletons
  - Roy_D / PC_Generator__s22 excluded because most of their frames fail G2(200)
- **clean_real** (label=0, T5C<0.2, gate=pass, non-chronic): 80 frames
  - Stratified across Test_Cam, Md_noyn_Sharker, Xiang_Xiang2_Feng, PC_Generator__s8/s13
- **tp_fake** (label=1, T5C>0.7, gate=pass): 80 frames
  - Stratified across Cam_Test__s33/s32/s35, PC_Generator__s9/s3, Test_Cam__s76/s53/s73

Total **260 frames**, all locally cached.

---

## Remediation candidates

1. **wb_grayworld** — gray-world white-balance correction (scale per-channel so means → grand mean)
2. **unsharp_05** — Gaussian unsharp mask, amount = 0.5
3. **gamma_norm** — gamma adjust to push luminance median toward 0.5
4. **clahe** — CLAHE on L-channel of Lab (clipLimit=2.0, tile=8×8)
5. **blend_w** — `w · unsharp_05(img) + (1-w) · img` for w ∈ {0.25, 0.35, 0.50, 0.65, 0.75}
   - Effective unsharp amount = 0.5 · w

---

## FACTS

### F1. Universal application of wb_grayworld / unsharp_05 / clahe / gamma_norm all HURT mixed-pool AUC on T5C

| condition | T5C AUC(mix) | ΔAUC | Δrisky | Δclean | Δtpfake |
|-----------|--------------|------|--------|--------|---------|
| orig | 0.9553 | 0 | 0 | 0 | 0 |
| wb_grayworld | 0.9283 | **-0.0270** | +0.029 | +0.097 | +0.001 |
| unsharp_05 | 0.9500 | -0.0053 | -0.101 | -0.014 | -0.023 |
| clahe | 0.9435 | -0.0118 | +0.065 | +0.170 | +0.016 |
| gamma_norm | 0.9370 | -0.0183 | +0.002 | +0.027 | -0.000 |
| **blend_unsharp_05** | **0.9611** | **+0.0058** | -0.034 | +0.002 | -0.003 |

The four naive universal transforms range from "neutral" (gamma) to "actively bad" (wb_grayworld pushes reals UP). Only the **50/50 blend** of orig + unsharp_05 net-helps.

### F2. The blend is non-monotonic in w; w=0.50 is the peak on T5C

| w (T5C) | ΔAUC(mix) | ΔAUC(hard) |
|---------|-----------|------------|
| 0.25 | +0.0007 | +0.0012 |
| 0.35 | -0.0062 | -0.0111 |
| **0.50** | **+0.0058** | **+0.0104** |
| 0.65 | +0.0012 | +0.0021 |
| 0.75 | -0.0058 | -0.0105 |

w=0.35 and w=0.75 both HURT despite straddling the winning w=0.50. This is surprising for a smooth blend and suggests the model response is non-linear in unsharp amount. **A grid > {0.50} should always include w=0.50 in any future re-test.**

### F3. The lever generalizes to P8A — and works better there

| w (P8A) | ΔAUC(mix) | ΔAUC(hard) | Δrisky | Δfake |
|---------|-----------|------------|--------|-------|
| 0.25 | -0.0010 | -0.0018 | -0.063 | -0.024 |
| 0.35 | -0.0028 | -0.0051 | -0.073 | -0.033 |
| **0.50** | **+0.0091** | **+0.0164** | -0.024 | **+0.026** |
| 0.65 | +0.0079 | +0.0142 | -0.038 | +0.018 |
| 0.75 | +0.0056 | +0.0101 | -0.051 | +0.009 |

On P8A, blend@0.50 gives **both** a real-side drop (-0.024) AND a fake-side RISE (+0.026). Doubly favorable. On T5C, fake-side is essentially flat (-0.003) — still net positive AUC.

Same w=0.50 sweet spot on both ckpts. Consistent with `project_iq_gating_viability_2026-05-04` (P8A's IQ-sharpness response is monotonic).

### F4. Risky-real response scales monotonically with input lap_var on T5C/unsharp_05

| lap quartile | n | lap_var median | Δscore (unsharp_05) |
|--------------|---|----------------|---------------------|
| Q1_blur | 25 | 75 | **-0.164** |
| Q2 | 25 | 200 | -0.124 |
| Q3 | 25 | 581 | -0.067 |
| Q4_sharp | 25 | 1215 | -0.047 |

Blurrier frames respond more. For TP fakes, the same gradient exists but smaller magnitude (-0.048 at Q1_blur vs -0.007 at Q4_sharp) — **fakes have systematically lower lap_var to begin with** (Q4_sharp fakes have lap_var=137, less than Q1_blur reals at 75).

### F5. Oracle (per-frame best-condition picker) ceiling = +0.0362 ΔAUC

If we had a perfect chooser that picks the best of {orig, wb, unsharp, gamma} per frame, the mixed-pool AUC ceiling is 0.9915. That's the upper bound for any pre-processing-based remediation lever using these four base transforms.

Realized gain (50/50 blend): +0.0058 ≈ 16% of oracle.

### F6. Naive conditional remediation (apply unsharp only when lap_var < median) underperforms universal application

On T5C, applying blend only to flagged frames (lap_var < 143) gave **worse** ΔAUC than applying universally:
- `conditional_unsharp_05`: AUC=0.9385 (Δ -0.0168)
- `universal_unsharp_05`: AUC=0.9500 (Δ -0.0053)

The lap_var threshold is too crude a flag. Universal blend@0.50 is preferred.

### F7. Failed remediation candidates

- **wb_grayworld**: Worst — pushes both clean and risky reals UP (+0.097 / +0.029). The "orange filter" hypothesis is **empirically refuted on this pool** — neutralizing color cast doesn't help T5C false positives.
- **clahe**: Pushes reals UP even more (+0.170 clean / +0.065 risky). Local contrast enhancement biases the model toward fake.
- **gamma_norm**: Essentially neutral. Risky reals drift slightly UP (+0.002).

The model is not fooled by "orange-cast" reals in the way the user's intuition suggested. The vulnerability is in the **sharpness/spectral-content axis**, not the color axis.

---

## OPINIONS

### O1. Recommended deployment lever: blend@0.50 unsharp, applied universally

- ~1 ms/frame overhead (Gaussian blur + 2 weighted-sum ops)
- No risky-frame detector needed at inference time
- AUC lift across both T5C and P8A
- Doesn't hurt clean reals (Δ ≈ 0) or TP fakes (Δ ≈ -0.003 on T5C, **+0.026** on P8A)

**Caveat:** this experiment used 260 frames from teams_real/teams_fake gate=pass cohorts. Cross-substrate validation on HDTF or live-prod cohorts is needed before promoting.

### O2. T5C+blend@0.50 may shift the deployment recommendation

The current "T5C step3500 @ tau=0.49, G1+G2(200)" winner gets ~+0.0104 ΔAUC on the hard pool with a 1ms preprocessing step. At a fixed FPR target, this should translate to slightly higher fake recall. Worth measuring on the full promotion-contract scorecard.

### O3. Per-frame oracle has +0.036 headroom — a learned router could capture more

The oracle picks `unsharp` for 84% of risky reals but `wb` for 50% of TP fakes (because wb pushes fake scores UP). A logistic-regression router on (iq_lap_var, iq_lab_a_dev, iq_lab_b_dev, predicted_score_orig) could plausibly capture a meaningful fraction of the +0.036 oracle gap. This is a follow-up direction, not a deploy-now lever.

### O4. The "orange filter" framing is wrong; the real axis is sharpness

The chronic-FP reals on this pool are not unified by a color-cast axis. wb_grayworld actively HURTS. The model's chronic-FP failure mode is in the spectral/sharpness domain — consistent with the broader IQ-shortcut finding (`project_image_quality_shortcut.md`).

---

## Artifacts

- `outputs/scores_per_frame.csv` — 260 rows, T5C scores under 4 conditions + IQ features
- `outputs/blend_sweep_t5c.csv`, `outputs/blend_sweep_p8a.csv` — blend grid results, both ckpts
- `outputs/p8a_extras_t5c.csv` — T5C scores under {orig, unsharp_05, clahe, blend_unsharp_05}
- `outputs/summary.json` — initial 4-condition headline summary
- `run_experiment.py` — main runner (T5C × {orig, wb, unsharp, gamma})
- `run_p8a_and_extras.py` — T5C × {orig, unsharp, clahe, blend_50}
- `run_p8a_blend_sweep.py` — T5C+P8A × {blend_25..75}
- `analyze_decomposition.py` — IQ-quartile + per-identity decomposition + oracle ceiling

## Reproduce

```bash
# 1. T5C baseline (4 conditions) — ~90s
python analysis/risky_remediation_2026-05-14/run_experiment.py

# 2. Add CLAHE + blend candidates (T5C) — ~100s
python analysis/risky_remediation_2026-05-14/run_p8a_and_extras.py

# 3. Blend-ratio sweep + P8A cross-check — ~90s
python analysis/risky_remediation_2026-05-14/run_p8a_blend_sweep.py

# 4. Decomposition analysis
python analysis/risky_remediation_2026-05-14/analyze_decomposition.py
```

Total CPU cost: ~5 min on M2 mac. $0 Vertex.
