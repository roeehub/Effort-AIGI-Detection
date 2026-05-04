# Thread: Eval-substrate data hygiene (is_no_face degeneracy + source-resolution floor)

> **Post-build maintenance addition (2026-04-29 afternoon)**: a visual audit of the very-sharp-FP and is_no_face slices surfaced two adjacent eval-substrate hygiene issues that are smaller in scope than the structurally-upstream `eval_production_crop_tightness_gap` finding (Finding 5 of the audit) but bear on the same axis. (1) The is_no_face slice (n=219 across no_face_real and no_face_fake) is **data degeneracy**, not model weakness — frames don't contain faces or contain faces too degraded to use, so MediaPipe's "0 faces" verdict is correct and the model has no useful signal to score. (2) The eval substrate contains **99×110-pixel source crops** (e.g. `PC_Generator__s15_1754.4_frame_058496_crop_002__d83b853d.jpg`) — thumbnails that no production deployment would feed the model. The 2026-04-27 investigation's `face_pixel_area > 57k` cut catches the small-face problem indexed by face area but **misses the small-source-image problem**. Source: 2026-04-29 morning visual audit (640 symlinked frames, local HTML viewer staged at `~/audit_slices_2026-04-29/index.html`); audit lineage references `wiki/ml-training/investigations/2026-04-27-lockbox-dev-property-investigation.md` (external to this canonical wiki).

## The question

The 2026-04-27 lockbox-dev property investigation flagged two slices as "open follow-ups" requiring visual confirmation: the very-sharp-FP slice and the is_no_face slice. The 2026-04-29 visual audit answered both. **The is_no_face slice is data-degeneracy** (no model signal available), not a high-lift model-fault hypothesis. **The eval substrate also contains thumbnail-resolution source crops** that pass the existing `face_pixel_area > 57k` filter but that no production deployment would score. Together these are eval-substrate hygiene issues — not the headline finding from the audit (that's [`eval_production_crop_tightness_gap`](eval_production_crop_tightness_gap.md)) but live on the same axis: **the eval substrate has known data-quality issues that bias scoring.**

## Initial belief

Through Slices 1-7, the lockbox + dev eval substrate was treated as an honest deployment proxy modulo the slice-specific caveats already tracked: webcam-mode dominance ([`webcam_fpr_dominance`](webcam_fpr_dominance.md)), face-pixel-area label leak ([`face_size_label_leak`](face_size_label_leak.md)), camera-signature shortcut ([`processing_signature_shortcut`](processing_signature_shortcut.md)), bucket gap on the headline metric ([`viso_bucket_gap`](viso_bucket_gap.md)). The 2026-04-27 investigation's `is_no_face` slice readout (*"3.0% flag share, FN lift 2.09×, FP lift 0.44×"*) was filed as a *"small but high-lift … concrete data-cleanup target"* — the 2× missed-as-fake reading was provisionally interpreted as model weakness on degraded faces.

The 2026-04-27 investigation's `face_pixel_area > 57k` quick-win bound was framed as the eval-time scope-of-applicability cut. **What was missing**: a parallel cut on **source-image resolution** that catches frames whose absolute pixel size makes them inappropriate as production inputs even when the face inside them clears the face-area threshold.

## What changed our mind

- **2026-04-29 morning — Audit Finding 1: is_no_face slice is data degeneracy, not model weakness.** Both no_face_real (n=174) and no_face_fake (n=45) sub-slices contain frames where MediaPipe's failure to detect a face is **correct**: frames don't contain faces or contain faces too degraded to use. Visual confirmation: the audit viewer browsed both folders. The 2026-04-27 caveat about *"small but high-lift"* is not wrong as a pattern (the 2× FN lift is real) — but the lift comes from data degeneracy, not a model failure mode where the model misses fakes that have viable face content. **Action**: target for removal or relocate from eval substrate. The 219 frames should not contribute to FN-lift conclusions about model behavior.

- **2026-04-29 morning — Audit Finding 4: 99×110-pixel source crops sit in the eval substrate.** Reproduction frame `PC_Generator__s15_1754.4_frame_058496_crop_002__d83b853d.jpg` is 99×110 at source — a thumbnail. The HTML viewer rendered it scaled-up, masking the resolution. The 2026-04-27 investigation's `face_pixel_area > 57k` cut catches the small-face problem **but indexed by face area, not source-image area**. A 99×110 source frame with face_pixel_area ~3.6k (the entire image is ~10.9k px²) passes the implicit `face_pixel_area > 1k` floor but fails any reasonable source-resolution bound like `min(width,height) >= 200`. **The model is being scored on inputs no production deployment would feed it**; dev/lockbox FPR numbers may be partly driven by these degenerate inputs. The fix is a small additive filter (source-image-size minimum) that would compose cleanly with the existing modern_lockbox_v2 definition.

- **2026-04-29 morning — Audit Finding 6 (re-confirmation, no new content): identity_key collision.** The audit identified the person in the right-hand frame as "Noyn" while `identity_key=PC_Generator__s15`. Re-validates the existing 2026-04-27 caveat about identity_key collisions; no new information. Logged for completeness; not a new loop.

## Current stance (2026-04-29)

Both findings are **eval-substrate hygiene issues**, not model failures. They live on the same axis as the structurally-upstream finding (eval-vs-production crop-tightness gap, see [`eval_production_crop_tightness_gap`](eval_production_crop_tightness_gap.md)) but are smaller in scope. The cleanest fix path:

1. **is_no_face**: remove or relocate the 219 frames from any eval substrate that informs FPR/recall/FN-lift conclusions. Since the slice is small (n=219 of ~7,334), the headline numbers will not move materially — but the *interpretation* of "the model misses fakes 2× more often when MediaPipe finds no face" will change from "model weakness" to "no usable face content; correctly-degenerate input." Document the frames-removed delta on at least one canonical checkpoint (P8A_step5000) so the audit trail is explicit.

2. **source-image-resolution floor**: add a `min(width, height) >= 200` (or similar) cut as part of the modern_lockbox_v2 filter set, run the headline FPR/recall under the bound, and decide whether the bound becomes a permanent piece of the v2 definition or stays as a side-by-side reporting column. The cut is independent of the existing face-area cut (face_pixel_area > 57k is a different axis).

3. The two findings together do not require a packet-level intervention; they are reporting-side hygiene that closes via re-tagging or re-filtering the existing parquet. The cost is small enough that this thread does not need its own packet retro.

The audit-of-the-audit framing (the 2026-04-27 investigation flagged these slices as "open follow-ups"; the 2026-04-29 visual audit closed them) means the close criteria for both loops are concrete enough to land within one short working session — **but they have not landed yet**, so the loops stay open until the artifacts are produced.

The audit's structurally most consequential finding (Finding 5 — eval-vs-production crop-tightness gap) is split into its own thread: [`eval_production_crop_tightness_gap`](eval_production_crop_tightness_gap.md). Findings 2 and 3 (sharpness-metric bug + slice-mixing implications) live in [`sharpness_metric_bug`](sharpness_metric_bug.md). This thread holds Findings 1 and 4. Finding 6 is re-confirmation of an existing caveat and is logged here for completeness without a new loop.

## Packet timeline

- *(no packet authored against these findings yet — surfaced 2026-04-29 afternoon as post-Slice-7 maintenance.)*
- The closest packet-side surface is the modern_lockbox_v2 build ([`webcam_fpr_dominance`](webcam_fpr_dominance.md)), which addressed the capture-mode tail and added `face_area_ratio >= 0.10` + `not is_pose_extreme` + `not is_no_face` as part of the v2 definition. **`not is_no_face` is already in v2_recommended** — so the headline 0.71% v2 FPR already excludes is_no_face. Headline-FPR-wise, this thread's is_no_face finding is mostly informational for v2 interpretation: the v2 number was already correct in excluding the data-degenerate slice. What this thread changes is the *baseline* lockbox FPR (4.6%) interpretation and the FN-lift narrative around is_no_face.

## Evidence locations

- `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` — n=7,334 dev+lockbox tagged frames; the substrate that supports the per-slice claims in this thread.
- `analysis/modern_lockbox_v2_2026-04-27/build_modern_subset.py` — the existing modern_lockbox_v2 filter; `not is_no_face` is already in the v2_recommended definition. The source-image-resolution floor is the proposed addition.
- Reproduction frame for Finding 4: `PC_Generator__s15_1754.4_frame_058496_crop_002__d83b853d.jpg` — 99×110 source crop; passes `face_pixel_area > 1k` but fails any reasonable `min(width, height) >= 200` floor.
- Audit lineage doc (external to canonical wiki): `~/Documents/repos/vault/wiki/ml-training/investigations/2026-04-27-lockbox-dev-property-investigation.md`. The is_no_face *"flag_share 3.0%, FN lift 2.09×, FP lift 0.44×"* row is in the "Smaller-but-real signals" table; the `face_pixel_area > 57k` cut is in the "Face size dominates FPR" section.
- Cross-thread anchors:
  - [`eval_production_crop_tightness_gap`](eval_production_crop_tightness_gap.md) — the structurally-upstream finding from the same audit; this thread holds the smaller-scope hygiene findings.
  - [`sharpness_metric_bug`](sharpness_metric_bug.md) — the third finding cluster from the same audit; metric bug + slice-mixing implications. Independent of this thread's findings but discovered in the same session.
  - [`webcam_fpr_dominance`](webcam_fpr_dominance.md) — modern_v2 already excludes is_no_face; the v2 number is unaffected by this thread's is_no_face finding. The source-image-resolution floor would compose with the v2 definition cleanly.
  - [`face_size_label_leak`](face_size_label_leak.md) — the source-image-resolution finding is a **parallel cut** to the face-size leak, indexed on a different variable. The face-size leak is about per-method face-pixel-area distribution; the resolution floor is about source-image dimensions. Both bear on the eval substrate but address different mechanisms.

## Open loops

### Open loop: is-no-face-slice-is-data-degeneracy
status: open
severity: low
first_seen: 2026-04-29
last_verified: 2026-04-29
close_criterion: the 219 is_no_face frames (174 real + 45 fake) are removed or relocated from any eval substrate that informs FPR/recall/FN-lift conclusions, AND the 2026-04-27 investigation's *"small but high-lift, FN lift 2.09×"* reading on this slice is documented as data-degeneracy (correct MediaPipe verdict — the frames don't contain face content the model could reasonably score) rather than model weakness. Headline FPR numbers will not move materially (n=219 of ~7,334) but the interpretation of "the model misses fakes when no face is present" changes. Documenting the frames-removed delta on at least one canonical checkpoint (P8A_step5000) makes the close criterion auditable.

**Low severity** because (a) modern_lockbox_v2 already excludes is_no_face from the v2 number, (b) the slice size is small (~3% of substrate), (c) closure is mechanical (filter + re-tag). Documented as its own loop because the **interpretation correction** matters for any future agent who reads the 2026-04-27 *"small but high-lift … data-cleanup target"* phrasing and assumes there's a model-side intervention to find here.

### Open loop: source-image-resolution-floor-not-applied-to-eval
status: open
severity: medium
first_seen: 2026-04-29
last_verified: 2026-04-29
close_criterion: a source-image-resolution minimum (e.g., `min(width, height) >= 200`) is applied as an eval-scope bound, the headline FPR / recall numbers on at least one canonical checkpoint (P8A_step5000) are rerun under the bound, AND the bound is either (a) added to the modern_lockbox_v2 filter set as a permanent piece of the v2 definition, OR (b) reported as a side-by-side column alongside v2 (so a reader can see both the v2 number and the v2-plus-resolution-floor number).

**Medium severity** because (a) the eval substrate is genuinely scoring on thumbnail-resolution inputs that no production deployment would feed the model, (b) the cut is independent of the existing `face_pixel_area > 57k` filter (a 99×110 source frame can pass the face-area floor and still be a thumbnail by source dimensions), and (c) closure is small (filter + rerun). The 2026-04-27 investigation's quick-win face-area cut caught the small-face problem but did not catch this; adding the resolution floor closes the parallel cut.

### 2026-05-04 evening update — F4 substrate-cleaning pipeline reproduces Job 14 within 0.5pp; reusable tool available

A reusable F4 substrate-cleaning eval pipeline landed at `analysis/substrate_cleaning_eval_2026-05-05/` (script `run_clean_eval.py`, frozen filter manifest `cleaned_substrate_manifest.json`, reference runs for P8A and E2B, `USAGE.md` for new ckpts). The F4 contract is:

- Drop chronic-6 identity list: `bla_bla_chow`, `bla_bla_chow__s2`, `pc_generator__s22`, `pc_generator__s45`, `roy_d`, `q__s6`
- Drop frames with `min(W, H) < 200`
- Drop frames where `is_no_face == True`

This pipeline operationalizes the existing closure pattern (drop is_no_face, source-image-resolution floor) PLUS the chronic-6 identity drop from the 2026-05-04 morning Job 14 follow-up (memory `project_job14_substrate_clean_2026-05-04.md`). The reference run reproduces Job 14 numbers within 0.5pp on every checkpoint tested:

| ckpt | suite | F0 (full) | F4 (cleaned) | Job 14 expected |
|---|---|---|---|---|
| P8A | viso | 26.9% | 67.1% | 27→67% ✓ |
| P8A | deeplive | 42.4% | 92.5% | 42→92% ✓ |
| P8A | teams_fake | 69.9% | 92.2% | (new) |
| P8A | real FPR | 10.0% | 0.86% | (new) |
| E2B_3200 | viso | 8.4% | 30.9% | 8→31% ✓ |
| E2B_3200 | deeplive | 93.9% | 100% | (new) |
| E3_6600 | viso | 13.8% | 77.6% | 14→78% ✓ |

Implication for this thread's open loops: both `is-no-face-slice-is-data-degeneracy` and `source-image-resolution-floor-not-applied-to-eval` are now operationally satisfied by F4 (which applies both filters PLUS the chronic-6 drop). The loops stay open until the contract scorecard reports F4-cleaned numbers as a first-class metric (currently only side-by-side in analysis dirs, not in the production scorer at `arena/score_teams_promotion_contract.py`). The pipeline is ready to score Packet A / Packet C-codec checkpoints (running in us-east1, ETA ~24h) the moment they finish.

### Cross-thread refs

- [`eval_production_crop_tightness_gap`](eval_production_crop_tightness_gap.md) — Finding 5 of the same audit, structurally upstream of the findings here. The crop-tightness gap is the bigger and more consequential issue; the resolution floor is a smaller axis on the same substrate.
- [`sharpness_metric_bug`](sharpness_metric_bug.md) — Findings 2 and 3 of the same audit; the metric bug overlaps with the question of how to interpret existing per-quartile FPR tables. Independent of this thread's loops; cross-cited for audit-trail completeness.
- [`webcam_fpr_dominance`](webcam_fpr_dominance.md) — modern_lockbox_v2's `not is_no_face` clause already excludes the is_no_face slice from the v2 headline. The source-image-resolution floor would compose with v2 cleanly; deciding whether to fold it into v2 or report side-by-side is part of the close criterion above.
- [`face_size_label_leak`](face_size_label_leak.md) — face-pixel-area is a different axis from source-image dimensions; the two filters are parallel cuts on the same substrate.
