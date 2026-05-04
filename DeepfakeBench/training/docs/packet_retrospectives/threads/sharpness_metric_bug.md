# Thread: `sharpness_laplacian` is computed on the full image, not the face

> **Post-build maintenance addition (2026-04-29 afternoon)**: a visual audit of the very-sharp-FP slice (sharpness_laplacian > 402, real, prob_fake > 0.97; n=421) and the is_no_face slice (n=219) discovered that `sharpness_laplacian` is computed on the **full grayscale frame**, not on a face crop. Source: 2026-04-29 morning visual audit (640 symlinked frames, local HTML viewer staged at `~/audit_slices_2026-04-29/index.html`); audit lineage references `wiki/ml-training/investigations/2026-04-27-lockbox-dev-property-investigation.md` (external to this canonical wiki). The bug lives at `analysis/lockbox_tagging/layers/quality.py:82` where `cv2.Laplacian(gray, cv2.CV_64F).var()` runs on the full grayscale frame rather than a face-crop region. **The 2026-04-27 investigation's per-quartile FPR table for `sharpness_laplacian` is partially confounded** because the metric it indexes mixes face-sharpness and non-face content.

## The question

Does the `sharpness_laplacian` tag in `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` measure **face sharpness** (the property the slice naming and the per-quartile tables imply it measures) or **whole-frame sharpness** (the property the implementation actually computes)? If the latter, every slice indexed on `sharpness_laplacian` is partially confounded — including the "very-sharp-FP" 45% FPR finding from the 2026-04-27 investigation that motivated the post-build visual audit in the first place.

## Initial belief

Through Slices 1-7, the `sharpness_laplacian` metric in `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` was treated as a **face-sharpness signal**. The 2026-04-27 investigation's per-quartile FPR table (in `~/Documents/repos/vault/wiki/ml-training/investigations/2026-04-27-lockbox-dev-property-investigation.md`) — *"sharpness_laplacian: very blurry (<45) 44% FPR; mid (45-402) 12-13% FPR; very sharp (>402) 45% FPR"* — was read as a U-shape on **face sharpness**. The very-sharp-FP slice (>402, real, prob_fake>0.97; n=421) was hypothesized as *"likely upsampled or post-processed reals that look 'too sharp' for the model's prior"* (per the 2026-04-27 doc). That hypothesis assumed face sharpness was the variable being thresholded.

## What changed our mind

- **2026-04-29 morning — Audit Finding 2: `sharpness_laplacian` is computed on the full grayscale frame, not the face crop** (`analysis/lockbox_tagging/layers/quality.py:80-82`). The implementation is:
  ```python
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  out["sharpness_laplacian"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
  ```
  `gray` is the full-frame grayscale (`img` is read by `cv2.imread(str(path))` at `quality.py:76`), not a face crop. The metric conflates face sharpness with non-face content sharpness.

- **2026-04-29 morning — Audit Finding 3: two visually-dissimilar frames have nearly identical full-image sharpness because the metric is dominated by non-face content.** Reproduction case (both lockbox, real, prob_fake>0.97):
  - `dor_shkedi__frame_000059_seq906__fec59bc1.jpg` (494×393 source, face 148×166): full-image laplacian **760.7**, face-crop laplacian **484.1**, **bg-only laplacian 844.9**, face-area-fraction 12.67%.
  - `PC_Generator__s15_1754.4_frame_058496_crop_002__d83b853d.jpg` (99×110 source, face 55×65): full-image laplacian **555.1**, face-crop laplacian **443.6**, **bg-only laplacian 1018.1**, face-area-fraction 33.50%.

  Re-derived face-crop laplacians are 484 vs 444 (essentially identical face sharpness). Visually one looks sharp and the other doesn't, but the metric they share isn't actually about the face. What drives the high laplacian on the small frame: virtual-background mountains, body silhouette, JPEG ringing. **The very-sharp-FP slice mixes (a) genuinely sharp faces in sharp environments and (b) soft faces in tiny / busy / compressed crops where non-face content drives the metric.** Two failure modes, not one.

- **2026-04-29 morning — Audit's stated implication: existing per-quartile FPR analyses on `sharpness_laplacian` are partially confounded.** The 2026-04-27 *"very sharp (>402): 45% FPR"* table entry mixes both populations the audit identified. **Not flagged in the 2026-04-27 investigation's caveats section.** Any downstream FPR-by-quartile reading that depends on this column inherits the confound.

## Current stance (2026-04-29)

`sharpness_laplacian` as computed in `analysis/lockbox_tagging/layers/quality.py:82` measures **whole-frame sharpness** (face + background + JPEG-ringing artifacts on the source crop), not face sharpness. The metric is **not wrong as a property** but is **wrong as an indicator of face sharpness**. The very-sharp-FP slice that the 2026-04-27 investigation flagged at 45% FPR is now known to mix two populations:

1. Genuinely sharp faces in sharp environments — a candidate "the model is mistrustful of unusually sharp face content" hypothesis.
2. Soft faces in tiny / busy / compressed crops where non-face content (virtual backgrounds, JPEG ringing on small source crops, body silhouettes) drives the laplacian — a candidate "the eval substrate has small / busy crops where the model uses the surrounding context, not the face" hypothesis.

**These are different failure modes and need to be split before any downstream conclusion.** The fix is mechanical: re-derive the metric on a face-crop region (using the existing `face_pixel_area` and bbox tags from the face-geometry layer) and re-tag the parquet. The fix unlocks two things:

1. The very-sharp-FP slice can be re-partitioned by face-crop laplacian, enabling per-population analysis.
2. Every existing per-quartile FPR analysis indexed on `sharpness_laplacian` (the 2026-04-27 investigation's table being the principal anchor) can be rerun with the corrected metric, producing the un-confounded reading.

The audit's lineage doc (`~/Documents/repos/vault/wiki/ml-training/investigations/2026-04-27-lockbox-dev-property-investigation.md`) is **outside the canonical wiki**. Do not litigate the 2026-04-27 investigation here; it is the audit's source material, not a thread under this protocol. The action this thread tracks is on the metric and on downstream FPR-by-quartile reports that depend on the metric, **not** on the 2026-04-27 doc.

The bug is also a small instance of a more general pattern that recurs in this codebase: a metric labeled as one thing (face-sharpness) implemented as another thing (full-image-sharpness). The pattern overlaps with the `silent-feature-failures-pattern` open loop in [`wandb_flattening`](wandb_flattening.md), but is not the same — that loop is about yaml-declared trainer features being silently disabled at runtime; this is about a property tag being correctly computed but mislabeled / misinterpreted. **No new "silent failure" loop is opened**; the existing patterns cover the broader axis. This thread holds the specific bug.

## Packet timeline

- *(no packet authored against this finding yet — surfaced 2026-04-29 afternoon as post-Slice-7 maintenance.)*
- The closest packet-side surface is the lockbox-tagging infrastructure ([`analysis/lockbox_tagging/`](../../../analysis/lockbox_tagging/)), which produced the parquet that motivates this thread.

## Evidence locations

- `analysis/lockbox_tagging/layers/quality.py:80-82` — the buggy implementation:
  ```
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  out["sharpness_laplacian"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
  ```
- `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` — n=7,334 dev+lockbox tagged frames; the column `sharpness_laplacian` has the buggy values that downstream slicing has been using.
- Reproduction frames (audit Finding 3, both lockbox real, prob_fake>0.97):
  - `dor_shkedi__frame_000059_seq906__fec59bc1.jpg` (494×393, face 148×166, full-image laplacian 760.7, face-crop laplacian 484.1, bg-only laplacian 844.9, face-area-fraction 12.67%).
  - `PC_Generator__s15_1754.4_frame_058496_crop_002__d83b853d.jpg` (99×110, face 55×65, full-image laplacian 555.1, face-crop laplacian 443.6, bg-only laplacian 1018.1, face-area-fraction 33.50%).
- Audit lineage doc (external to canonical wiki): `~/Documents/repos/vault/wiki/ml-training/investigations/2026-04-27-lockbox-dev-property-investigation.md`. The per-quartile FPR table that depends on the buggy `sharpness_laplacian` lives in this doc's "Property findings" section.
- Cross-thread anchors:
  - [`eval_production_crop_tightness_gap`](eval_production_crop_tightness_gap.md) — Finding 5 (the structurally most consequential audit finding) shares the same 2026-04-29 audit. The full-image-laplacian bug is the mechanism by which Finding 5 (eval frames carry more background context) gets injected into measurements that nominally index on face properties.
  - [`processing_signature_shortcut`](processing_signature_shortcut.md) — any prior investigation that read "FPR is high in the very-sharp slice" as evidence about face sharpness should now re-read it as evidence about whole-frame sharpness, which overlaps with the camera/ISP-signature shortcut surface.

## Open loops

### Open loop: sharpness-metric-computed-on-full-image-not-face
status: open
severity: high
first_seen: 2026-04-29
last_verified: 2026-04-29
close_criterion: `analysis/lockbox_tagging/layers/quality.py:82` is updated to compute the laplacian on a face-crop region (using the existing face-geometry layer's bbox tags), the parquet `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` (or its successor) is re-tagged with the corrected metric, AND any downstream FPR-by-quartile report that depends on `sharpness_laplacian` is either rerun under the corrected metric OR explicitly annotated as "indexed on full-image laplacian, not face laplacian — read as a confounded U-shape". The 2026-04-27 investigation's *"very sharp (>402): 45% FPR"* table entry is the canonical downstream report; flagging it covers the most-cited claim.

**High severity** because (a) the metric is wrong as labeled, (b) downstream per-quartile FPR analyses depend on it, and (c) the very-sharp-FP slice was a flagged "open follow-up" from the 2026-04-27 investigation that motivated the post-build audit in the first place. The fix is mechanical and small (~15 lines in `quality.py` + one re-tag pass on the parquet); the cost is in auditing the downstream consumers of the metric. No prior loop covers this — `silent-feature-failures-pattern` ([`wandb_flattening`](wandb_flattening.md)) is about trainer features silently disabled at runtime, which is a different mechanism.

### Open loop: very-sharp-fp-slice-mixes-two-populations
status: open
severity: medium
first_seen: 2026-04-29
last_verified: 2026-04-29
close_criterion: the very-sharp-FP slice (n=421 in the 2026-04-27 substrate) is re-tagged with face-crop laplacian and re-partitioned into the two populations the audit identified — (a) genuinely sharp faces in sharp environments, (b) soft faces in tiny / busy / compressed crops where non-face content drives the laplacian; per-population FPR is reported separately; the wiki narrative around the very-sharp-FP slice is updated to reflect the split.

**Medium severity** because this is a corollary of the metric bug above, not an independent question. Closes naturally once `sharpness-metric-computed-on-full-image-not-face` closes, but listed separately because the per-population FPR split is the actual analysis output that downstream readers care about (the metric fix alone is necessary-but-not-sufficient).

### Cross-thread refs

- [`eval_production_crop_tightness_gap`](eval_production_crop_tightness_gap.md) — Finding 5 of the same audit; the full-image-laplacian bug is one of the mechanisms by which the eval-substrate's larger background surface gets injected into per-property measurements.
- [`face_size_label_leak`](face_size_label_leak.md) — the per-method face-size signatures the leak documents are computed on the same parquet; `sharpness_laplacian` is a different column from `face_pixel_area` and is **not** confounded by the face-pixel-area cluster. The two threads share the parquet but document independent properties.
- [`webcam_fpr_dominance`](webcam_fpr_dominance.md) — modern_lockbox_v2 filter does not currently slice on `sharpness_laplacian`; the v2 filter is unaffected by this bug.
