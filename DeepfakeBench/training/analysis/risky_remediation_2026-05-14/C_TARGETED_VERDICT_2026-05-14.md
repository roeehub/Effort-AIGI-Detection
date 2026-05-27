# Targeted Remediation — Final Verdict (Run C, 2026-05-14)

## TL;DR

**No deployable targeted-remediation lever exists for T5C step3500 with cheap frame features.** All three pre-registered falsifiers failed (or were rendered moot by an unexpected calibration effect). The oracle (per-frame perfect chooser) ceiling is +0.037 AUC, but the realizable gain with any feature-based routing is ≤ 0.

**Action:** Continue to ship T5C step3500 + τ=0.49 + G1+G2(200) with no inference-side remediation. The original `A_SHIP_SPEC_T5C_2026-05-14.md` stands.

---

## What we tested

T5C step3500 scored under 8 new remediation variants (in addition to the 2 already tested) on the 7,331-frame G2-pass pool. New variants:

| name | description |
|------|-------------|
| `blend_035` | weight=0.35 (effective unsharp 0.175) |
| `blend_065` | weight=0.65 (effective unsharp 0.325) |
| `blur_5` | 5×5 Gaussian σ=1.0 — OPPOSITE direction (low-pass) |
| `blur_7` | 7×7 Gaussian σ=1.5 — stronger low-pass |
| `desat_50` | 50% HSV-S desaturation (color reduction) |
| `downup_168` | resize to 168×168 then back to 224 (frequency-band low-pass) |
| `clahe_mild` | CLAHE on L-channel, clipLimit=1.0 (local contrast) |
| `tta_3way` | mean(orig, blend_050, blur_5) — TTA aggregation |

10 total remediations including orig and blend@0.50.

---

## F1 — does ANY remediation rescue ≥30% of T5C errors?

Definition: a frame is "rescued" by remediation R if it was an ERROR under orig (`real with score>0.49` or `fake with score≤0.49`) and NOT an error under R.

| remediation | errors_orig | rescued | broken | net | rescue rate |
|-------------|-------------|---------|--------|-----|-------------|
| orig | 1317 | 0 | 0 | 0 | 0% |
| blend_035 | 1317 | 283 | 79 | **+204** | 21.5% |
| downup_168 | 1317 | 205 | 470 | −265 | 15.6% |
| blur_7 | 1317 | 135 | 1175 | **−1040** | 10.2% |
| blend_065 | 1317 | 105 | 149 | −44 | 8.0% |
| clahe_mild | 1317 | 93 | 827 | −734 | 7.1% |
| blur_5 | 1317 | 79 | 990 | −911 | 6.0% |
| blend_050 | 1317 | 69 | 170 | −101 | 5.2% |
| desat_50 | 1317 | 69 | 453 | −384 | 5.2% |
| tta_3way | 1317 | 24 | 341 | −317 | 1.8% |

**Best: `blend_035` at 21.5% rescue rate.** Falls short of the 30% threshold.

**F1: FAIL.**

Also: every single remediation EXCEPT `blend_035` breaks more frames than it rescues. Only `blend_035` has positive net (+204).

---

## Oracle ceiling — what's the upper bound?

If we had a perfect per-frame chooser that picks the best remediation per frame:
- Errors drop from 1317 → 786 (a 40.3% reduction)
- AUC goes from 0.9532 → 0.9902 (Δ = **+0.0369**)
- Oracle's remediation distribution: `blend_035` 25%, `downup_168` 23%, `clahe_mild` 21%, `desat_50` 15%, others <10% each

**The errors are heterogeneous.** No single remediation dominates. To realize even a small fraction of the +0.037 ceiling, we'd need a router that picks the right transform per frame from cheap features.

---

## F2 — can cheap features predict which frames each remediation rescues?

Trained a gradient-boosted classifier on (`lap_var`, `luma`, `lab_a_dev`, `lab_b_dev`, `edge_density`) features. Train on `teams_dev`, test on `lockbox + dor_cross`. Target: "was this frame rescued by remediation R."

| remediation | rescued_train | rescued_test | predictor held-out AUC |
|-------------|---------------|--------------|------------------------|
| `downup_168` | 80 | 125 | **0.7108** ✓ |
| `blend_035` | 59 | 224 | 0.5298 |
| `blur_7` | 32 | 103 | 0.5238 |
| others | <30 | various | (insufficient data — skip) |

Only `downup_168` passes the F2 ≥ 0.65 threshold. But `downup_168` has F1-net **= −265** (rescues 205 but breaks 470). So even with a good rescue-predictor, the underlying remediation creates more errors than it fixes.

**F2: PASS only for downup_168. F3 cannot proceed since no remediation passes BOTH F1-net>0 AND F2-AUC≥0.65.**

---

## F3 — surprise finding & why it still failed

Going beyond the original 3 falsifiers, I noticed `clahe_mild` had a striking substrate-specific effect on per-pool AUC:

| pool | n | T5C orig AUC | clahe_mild AUC | Δ | bootstrap 95% CI |
|------|---|--------------|----------------|---|------------------|
| teams_dev | 3630 | 0.9802 | 0.9779 | −0.0023 | [−0.0047, +0.0002] |
| **teams_lockbox** | **1686** | **0.9085** | **0.9517** | **+0.0432** | **[+0.0298, +0.0575]** ✓ |
| dor_cross | 1624 | 0.9417 | 0.9324 | −0.0093 | [−0.0145, −0.0042] |
| all combined | 6940 | 0.9532 | 0.9458 | −0.0074 | [−0.0098, −0.0049] |

CLAHE@clipLimit=1.0 SIGNIFICANTLY HELPS the lockbox pool (+0.043 AUC) and SIGNIFICANTLY HURTS dor_cross. Universal application is net negative.

**The obvious follow-up: apply CLAHE only to lockbox-like frames.** I tested this directly:

1. Trained a "is-this-frame-lockbox-like" classifier on (lap_var, luma, lab_a/b_dev, edge_density). **Held-out AUC: 0.9874** — near-perfect detection of lockbox cohort from cheap features.
2. Used the lockbox-classifier as a router: apply `clahe_mild` if `p_lockbox ≥ threshold`, else `orig`.
3. Swept threshold from 0.10 to 0.90.

| router threshold | frac routed | combined AUC | Δ vs orig | CI95 |
|------------------|-------------|--------------|-----------|------|
| 0.10 | 28.2% | 0.9422 | −0.0079 | [−0.0106, −0.0054] ✗ |
| 0.30 | 23.7% | 0.9426 | −0.0075 | [−0.0101, −0.0050] ✗ |
| 0.50 | 22.2% | 0.9422 | −0.0078 | [−0.0105, −0.0054] ✗ |
| 0.70 | 20.6% | 0.9418 | −0.0083 | [−0.0108, −0.0058] ✗ |
| 0.90 | 18.3% | 0.9414 | −0.0087 | [−0.0110, −0.0063] ✗ |

**Every threshold gives a STATISTICALLY SIGNIFICANT NEGATIVE Δ on combined pool.**

The reason this surprised me but on reflection shouldn't:

**Score-calibration mismatch.** Even though CLAHE improves real-vs-fake *ranking* WITHIN the lockbox subset, it shifts the absolute score range. When you combine CLAHE'd lockbox scores with non-CLAHE'd dev/dor_cross scores into a single AUC, frames at the boundary between cohorts get reordered incorrectly. The boundary effect dominates the within-pool gain.

In retrospect this is the same bimodal-distribution problem that killed my earlier "lap_var < median → blend" conditional design. Conditional remediation creates two score populations that the AUC metric can't combine cleanly.

**F3: FAIL.** No deployable recipe exists even when the routing condition is near-perfectly detectable.

---

## Final falsifier table

| falsifier | result | note |
|-----------|--------|------|
| F1 — any remediation rescues ≥30% errors | **FAIL** | best is blend_035 at 21.5% |
| F2 — rescuer cluster held-out AUC ≥ 0.65 | partial PASS | only downup_168 (but F1-net negative) |
| F3 — held-out recipe Δ ≥ +0.005 with CI > 0 | **FAIL** | even with 0.987 AUC router, ΔAUC is significantly NEGATIVE |

---

## Why targeted remediation fails on T5C

1. **T5C errors are heterogeneous.** Oracle picks 5 different remediations as "best" for >10% of errors each. No single fix.
2. **Most remediations break more than they rescue.** 8 of 9 candidates had negative F1-net. Only `blend_035` had positive net (+204) but failed F2.
3. **GRL training removed the easy levers.** T5C was trained on `sharpness_laplacian_high` and `color_a_approx_dev_high` axes. Remediations along those axes have nothing to exploit. Blur and downup test the *opposite* direction (low-pass, frequency reduction) — even worse.
4. **Score-calibration mismatch kills conditional routing.** Even a near-perfect routing classifier (AUC 0.987 for lockbox detection) can't deploy CLAHE without creating bimodal score distributions that hurt combined AUC.
5. **The cheap features predict POOL not ERROR.** The lockbox-classifier was 0.987 AUC; the rescuer-classifier was 0.53–0.71 AUC. Features can identify which substrate a frame came from, but not which T5C errors are "fixable" by which transform.

---

## What would change the picture

The targeted lever could exist if any of these were true (none are, based on this experiment):

1. **A new feature** that predicts T5C errors better than the IQ stack. Frequency-spectrum features, learned embeddings from a small auxiliary network, motion signatures. None tested here.
2. **A remediation that doesn't shift score calibration.** All tested transforms move the score distribution. A transform that preserves the score-rank-vs-score-magnitude relationship would avoid the bimodal-distribution problem. None obvious.
3. **A model that's already invariant to the routing decision.** If T5C had been trained with the remediations as augmentations, applying them at test time wouldn't shift scores — they'd be no-ops. This would be a TRAINING-side intervention, not inference.

---

## Recommendation

1. **Stop pursuing inference-side levers for T5C.** Three runs of progressively more aggressive experiments (universal blend → conditional blend → 10-remediation sweep with statistical falsifiers) have converged on the same answer: T5C does not have a free inference-side lever with the features we have.

2. **Ship T5C step3500 as already specced** — `A_SHIP_SPEC_T5C_2026-05-14.md` is authoritative.

3. **For future T5C improvement, focus on training-side**: continue LoRA work (`R13_LORA_L10_L11`), add new GRL axes that the current model doesn't cover (frequency-band, edge density), or pursue identity-fresh data for chronic FPs.

4. **One lateral idea worth filing**: the CLAHE-on-lockbox-only finding (+0.043 AUC within lockbox) hints at a substrate-specific lever. The blocker is score calibration, not detection. If we ever rebuild the pipeline with per-substrate τ calibration (which would be a major plumbing change in production), this lever could re-emerge. Not worth doing for +0.04 AUC on a 25%-of-traffic slice today.

---

## Artifacts

In `analysis/risky_remediation_2026-05-14/`:

| file | content |
|------|---------|
| `A_SHIP_SPEC_T5C_2026-05-14.md` | (still) authoritative ship spec |
| `B_CROSS_SUBSTRATE_FINAL_VERDICT.md` | universal blend verdict |
| `C_TARGETED_VERDICT_2026-05-14.md` | this document |
| `run_targeted_remediation.py` | inference runner for 8 new remediations |
| `analyze_targeted.py` | F1/F2/F3 falsifier engine |
| `outputs/targeted_remediation_scored.csv` | 7331 frames × 10 conditions |
| `outputs/targeted_f1_rescue_counts.csv` | per-remediation rescue/broken table |
| `outputs/targeted_f2_predictability.csv` | rescue-predictor held-out AUC |

Total CPU spent: ~50 min on M2 mac, ~$0 Vertex.

## Statistical methodology summary

- All ΔAUC values report 95% bootstrap CIs from 1000 paired resamples (2000 in B verdict).
- F1 rescue/broken counts are at fixed τ=0.49 (the deployment threshold).
- F2 train pool = teams_dev (in-sample); test pool = teams_lockbox + dor_cross (held-out).
- F3 was extended beyond the original spec to evaluate the CLAHE-lockbox finding once it appeared in per-pool AUC analysis. Recipe tested with stratified 60/40 train/test split, gradient-boosted classifier, 1000-iter bootstrap on the final ΔAUC.
- Wilcoxon paired signed-rank used for per-frame Δscore tests in the B verdict.
