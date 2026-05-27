# skin_frac viso eval-vs-train gap — VERDICT: NO GAP

_skin_frac KS p=6.74e-54 (rejects identity), but |delta_mean|=0.041 < 0.05 floor.
Eval-viso-fake mean=0.629 vs train-viso-fake mean=0.588.
By the pre-registered rule (NO_GAP if KS p>0.05 OR |delta|<0.05), the third
image-quality leg of the known shortcut does NOT have a meaningful train-eval
mean mismatch on the visomaster_enhanced viso suite._

Run: 2026-05-03 10:15:26

## Group distribution table (skin_frac, laplacian_var, luma_mean)

| group | n | skin_frac mean±std | skin_frac p10/p90 | laplacian_var mean±std | luma_mean mean±std |
|---|---:|---|---|---|---|
| eval_viso_fake | 550 | 0.629±0.060 | 0.595/0.669 | 91.0±42.4 | 159.9±19.2 |
| train_viso_fake | 793 | 0.588±0.210 | 0.355/0.860 | 482.2±294.5 | 108.3±29.2 |
| eval_real | 550 | 0.566±0.188 | 0.294/0.814 | 164.4±139.0 | 130.3±25.7 |
| train_viso_real | 391 | 0.627±0.199 | 0.402/0.901 | 146.9±152.1 | 109.0±29.1 |

## KS-test summary (two-sample, two-sided)

| attr | pair | statistic | p_value | mean_a | mean_b | delta(a-b) |
|---|---|---:|---:|---:|---:|---:|
| skin_frac | eval_viso_fake_vs_train_viso_fake | 0.428 | 6.74e-54 | 0.629 | 0.588 | +0.041 |
| skin_frac | eval_viso_fake_vs_eval_real | 0.395 | 1.33e-38 | 0.629 | 0.566 | +0.063 |
| skin_frac | train_viso_fake_vs_train_viso_real | 0.100 | 0.00955 | 0.588 | 0.627 | -0.039 |
| laplacian_var | eval_viso_fake_vs_train_viso_fake | 0.893 | 1.29e-274 | 91.002 | 482.223 | -391.221 |
| laplacian_var | eval_viso_fake_vs_eval_real | 0.380 | 9.14e-36 | 91.002 | 164.418 | -73.416 |
| laplacian_var | train_viso_fake_vs_train_viso_real | 0.679 | 7.47e-116 | 482.223 | 146.910 | +335.313 |
| luma_mean | eval_viso_fake_vs_train_viso_fake | 0.741 | 2.86e-174 | 159.924 | 108.330 | +51.593 |
| luma_mean | eval_viso_fake_vs_eval_real | 0.498 | 2.22e-62 | 159.924 | 130.302 | +29.622 |
| luma_mean | train_viso_fake_vs_train_viso_real | 0.035 | 0.894 | 108.330 | 109.045 | -0.715 |

## Implications for next packet

**Headline:** The pre-registered hypothesis is REFUTED. A skin-aware augmentation is
NOT the cheap lever to break the 27% viso recall ceiling — the eval and training
viso-enhanced distributions agree on skin_frac mean (Δ=0.041). Don't propose a
skin-aug-only sister to P22.

**Three larger findings emerged from the same run, however, and they outrank
this hypothesis on the priority list:**

1. **laplacian_var gap is enormous (5.3× train > eval, KS=0.89, p≈10⁻²⁷⁴).**
   Training viso fakes are vastly sharper than eval viso fakes. This confirms
   the cross-suite finding from `project_image_quality_shortcut.md` *intra-suite*
   on visomaster_enhanced specifically. P22's blur-jitter is the right shape
   but apparently not strong enough — eval p10 (46) sits below training p10 (188).
   A heavier blur curriculum, skewed toward eval-suite sharpness, is the
   single-lever try with the most direct evidence behind it.

2. **luma_mean gap of +51 (eval brighter than train, KS=0.74, p≈10⁻¹⁷⁴).**
   Eval viso-enhanced frames are systematically brighter than what the model
   has been trained on. P22's brightness jitter was the right idea; the
   curriculum should be re-tuned with viso-eval as the explicit target.

3. **eval_viso_fake variance is collapsed across all three attributes**
   (skin_frac std 0.06 vs train 0.21; laplacian std 42 vs 295; luma std 19 vs 29).
   Consistent with `project_eval_production_crop_tightness_gap.md`: the eval
   suite is a thin ridge in the joint image-quality feature space. Even when no
   individual axis crosses the verdict floor, the joint mismatch may dominate.

**Recommendation for next packet:** P-* sister to P22 that shifts the
augmentation distribution toward the eval viso joint (laplacian_var ~50–150,
luma_mean ~135–185), keep skin_frac alone. Validate via the promotion-contract
scorecard. Skin-aware aug is dropped from the candidate list.