# CPU-3 Per-Ckpt Axis vs Anchor Decomposition — FACTS (2026-05-23)

> **Status: factual-only.** Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, ship, kill, best, worst, unfortunately, remarkably, lucky, confirmed, refuted, shortcut-aligned, gap-is-wide, gap-is-narrow.

## 1. Method

Inputs:
- 3 per-ckpt substrate axes (LR-classifier weight vectors fit on the 1880 paired clean-teams identity frames per 2026-05-22 Probe 1 KLIEP re-fit): `analysis/substrate_pair_geometry_2026-05-22/_trained_encoder_substrate_axis_{P8A_step5000,SlotAv2_step3500,T5C_step3500}.npy` (each (768,), L2-normalized).
- Cached L11 features (`feats/{ckpt}_L11_{clean,teams}.npy`, each (5475 or 5478, 768)) and cached head probs (`scores/{ckpt}_{clean,teams}.npy`).
- D8 frozen-CLIP-L11 features: `analysis/cpu_diagnostics_2026-05-12_d8_head_retrain_substrate_balanced/outputs/clip_frozen_l11__n4839.npz` ((4839, 768) = 4000 dev + 839 lockbox).
- D8 metadata: `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` (identity_key, split, clip_capture_mode, label).

Computations:
- Pairwise cosines between the 3 per-ckpt substrate axes.
- For each ckpt: `cos(per_ckpt_axis, prob_fake_gradient_direction)` where the gradient direction = `normalize(mean(feat | head_prob > 0.5) − mean(feat | head_prob ≤ 0.5))` on the 5475 + 5478 paired frames using cached L11 features + cached head probs.
- False-flag normal: `normalize(mean(feat | dor real lockbox webcam) − mean(feat | dor real dev normal_photo))` on the D8 frozen-CLIP-L11 cache.
- Dor real dev normal_photo cohort: n = 15.
- Dor real lockbox webcam cohort: n = 38.
- Wall time: 0.4s.

Caveat: per-ckpt axes are trained-encoder L11 (768-dim). The false-flag normal is frozen-CLIP-L11 (also 768-dim). The cosine across these two spaces is a CROSS-ENCODER cosine and is a LOWER BOUND on the trained-encoder-internal cosine between the substrate axis and the false-flag direction. The plan accepts this approximation as the cheap probe.

## 2. Pairwise cosines (substrate axes only)

| Pair | cosine |
|---|---:|
| P8A vs SlotAv2 | +0.6922 |
| P8A vs T5C | +0.7264 |
| SlotAv2 vs T5C | +0.9236 |

## 3. Per-ckpt cosine table

| Ckpt | cos(axis, prob_fake_gradient) | cos(axis, false_flag_normal_frozen) |
|---|---:|---:|
| P8A_step5000 | +0.0704 | +0.0808 |
| SlotAv2_step3500 | -0.0497 | +0.0628 |
| T5C_step3500 | -0.0410 | +0.0393 |

## 4. Close criterion verdict

**Verdict: gamma**

Summary: in-between: |cos(axis, false-flag-normal)| = P8A=0.081, SlotAv2=0.063, T5C=0.039; cos(P8A,SlotAv2)=0.692; both BACKBONE runs proceed

Decision rule:
- alpha: ALL 3 |cos(axis, false-flag-normal)| < 0.3 AND cos(P8A, SlotAv2) > 0.7 → anchor + substrate decoupled → BACKBONE-SlotAv2 stacks substrate-pair on anchor without bundle penalty
- beta:  ANY |cos(axis, false-flag-normal)| > 0.6 → anchor reduces substrate-pair variance → BACKBONE-T5C becomes primary signal
- gamma: in-between → both BACKBONE runs proceed

## 5. Output artifacts

- `axis_cosines.csv` — all cosine pairs
- `RESULTS_FACTS_2026-05-23.md` — this file
