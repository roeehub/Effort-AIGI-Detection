# P8A trainable parameter audit — FINDINGS

**Date**: 2026-05-03
**Hypothesis tested**: Did P8A's `in_proj-SVD` residuals actually learn, or did they
train as no-op due to the gradient bug fixed in commit `2feea58` (2026-04-26)?
**P8A trained**: 2026-04-24 — i.e., **before** the fix landed.

## Verdict: **PARTIAL** (depth-stratified)

| Layer range | in_proj-SVD status | Evidence |
|---|---|---|
| **L0–L4 (early)** | Effectively no-op | sign-flip frac 0.34–0.66, &#124;S&#124; 2–4 orders below same-layer out_proj |
| **L5–L11 (deep)** | Genuinely trained | sign-flip ≤ 0.22, &#124;S&#124; growing monotonically with depth |

**13 of 36 in_proj-SVD slots (36%) are noise-dominated** — concentrated in L0–L4 q/k/v.

## Headline numbers (`outputs/audit_summary.json`)

| Metric | Value |
|---|---:|
| Total SVD layers found | 72 (12 blocks × 6 slots) |
| Total non-zero residual layers | 72 (all technically non-zero) |
| in_proj-SVD layers effectively unlearned | **13 / 36 (36%)** |
| Trainable params (non-zero contribution) | 5.31M |
| - of which in_proj | 1.77M |
| - of which other (out_proj + MLP) | 3.54M |
| Median contribution L2: in_proj vs out_proj | **0.060 vs 1.36 (23× smaller)** |
| Median &#124;S&#124; mean: in_proj vs out_proj | **0.008 vs 0.11 (14× smaller)** |

For reference, L14 (per `docs/JAN_13_L14_vs_B16_LAION_Investigation.md`) trains
~199K backbone params across 96 SVD layers (24 × 4). P8A on B16 has 5.31M
non-zero residual params nominally, but **the in_proj contribution is dwarfed
by the out_proj + MLP residuals across the board, and is effectively zero
for the early layers where camera-signature, image-quality, and crop-tightness
shortcuts live**.

## What this implies for the viso ceiling

The "B16 capacity ceiling" framing has been load-bearing in this project's
recent architectural-pivot discussions. This audit refines it:

- **It is not 4× self-imposed across all layers** (the naive read of the
  Jan-13 in_proj-SVD coverage gap).
- It is roughly **2× self-imposed and concentrated in early-layer attention
  QKV (L0–L4)** — exactly where the substrate-cue shortcuts live according
  to memory entries (`project_image_quality_shortcut`, `project_face_size_label_leak`,
  `project_p8a_breakthrough`).
- Deep layers (L5–L11) trained fine on the existing recipe and explain why
  the model has any cross-domain signal at all (frame-level AUC 0.75 viso
  per FACTS §1.1).

## Cheaper test before full from-scratch retry

The user has a $60 L14 budget option but wants to keep B16. Before either
direction, this audit suggests a **cheap intermediate test**:

1. Warm-start from P8A.
2. Re-initialise ONLY the L0–L4 `svd_{q,k,v}` residuals (the 13 noise-dominated
   slots) to non-zero values.
3. Train 1k–2k steps with the existing recipe.
4. Score against contract.

If early-layer in_proj awakening alone shifts viso recall above the 27%
ceiling, full from-scratch is justified and the L14 budget can wait.
If it doesn't shift, the binding constraint is somewhere else (shortcuts,
data, calibration) and architectural change becomes the rational next move.

## Caveats

- The bug is described as `forward_pre_hook + .data.copy_` skipping autograd.
  Per memory, "the orthogonal-loss path still flowed" — so the early-layer
  residuals received SOME gradient (from reg_loss orthogonality maintenance),
  just not from the classification loss. The "non-zero magnitude" we observe
  is consistent with reg-loss-only training: small magnitudes, frequent
  sign-flips (no consistent gradient direction).
- Deep layers may have trained well not because the bug was fixed but
  because their forward signal magnitudes are larger and the orthogonal
  loss has more leverage there. Verifying this would require running
  the same audit on a post-2026-04-26 (post-fix) FT-from-P8A checkpoint
  (e.g., P22 step1k) — left as follow-up.

## Artefacts

- `outputs/svd_layer_audit.csv` — per-layer × per-slot (72 rows)
- `outputs/audit_summary.json` — aggregate stats + verdict
- `scripts/01_audit_p8a_svd.py` — reproducible script
- `run.log` — execution log
