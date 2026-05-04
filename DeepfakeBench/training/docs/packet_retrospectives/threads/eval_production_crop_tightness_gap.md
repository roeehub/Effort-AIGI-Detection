# Thread: Eval-vs-production crop-tightness gap

> **Post-build maintenance addition (2026-04-29 afternoon)**: a visual audit of the very-sharp-FP and is_no_face slices surfaced a structural mismatch between eval-substrate face crops and production-deployment face crops. Faces in eval frames carry more background context around the face than production crops do. **This is upstream of multiple shortcut/FPR investigations the wiki already tracks.** Source: 2026-04-29 morning visual audit (640 symlinked frames, local HTML viewer staged at `~/audit_slices_2026-04-29/index.html`); audit lineage references `wiki/ml-training/investigations/2026-04-27-lockbox-dev-property-investigation.md` (in `~/Documents/repos/vault/wiki/ml-training/investigations/`, external to this canonical wiki). The 2026-04-27 investigation flagged the very-sharp-FP and is_no_face slices as open follow-ups; the 2026-04-29 audit visualized them and discovered the crop-tightness mismatch as a corollary.

## The question

When the eval substrate scores a checkpoint at some FPR (e.g. P8A baseline lockbox FPR 4.6% at τ=0.9741, modern_v2 FPR 0.71% at the same τ), how much of that number is a **deployment-distribution-faithful measurement** vs how much is **artifact of the eval substrate's frame composition** — specifically, the fact that eval frames carry more background context around the face than production face crops do? Equivalently: does any FPR (or recall) number measured on the eval substrate translate one-to-one to production deployment, or does the eval-vs-production crop-tightness delta inflate / deflate / scramble the translation?

## Initial belief

Through Slices 1-7, eval-substrate FPR and recall numbers were treated as **deployment proxies** modulo the slice-specific caveats already tracked: webcam-mode dominance ([`webcam_fpr_dominance`](webcam_fpr_dominance.md), Slice 6), face-pixel-area label leak ([`face_size_label_leak`](face_size_label_leak.md), Slice 6), camera-signature shortcut ([`processing_signature_shortcut`](processing_signature_shortcut.md), Slice 4 origin), bucket gap on the headline metric ([`viso_bucket_gap`](viso_bucket_gap.md), Slice 7). The implicit assumption was that **once those slice-specific caveats are accounted for, the residual eval number maps cleanly to production**. The 2026-04-29 audit refutes the implicit assumption: the eval frames themselves are not cropped at production tightness, so any failure mode that depends on background content is amplified in eval relative to production.

## What changed our mind

- **2026-04-29 morning — Visual audit of very-sharp-FP slice and is_no_face slice surfaces crop-tightness mismatch** (audit notes; source local viewer + per-folder `_manifest.csv`, 640 symlinked frames at `~/audit_slices_2026-04-29/index.html`). The audit's headline observation was Finding 5: *"faces in eval are not cropped tightly as in production; eval frames carry more background context around the face."* Quoted-direct observation, not a derived statistic.

- **The reproduction case demonstrates the mechanism** (audit Finding 3). Two visually-dissimilar lockbox real frames at prob_fake>0.97:
  - `dor_shkedi__frame_000059_seq906__fec59bc1.jpg` (494×393 source, face 148×166, full-image laplacian 760.7, face-crop laplacian 484.1, **bg-only laplacian 844.9**, face-area-fraction 12.67%).
  - `PC_Generator__s15_1754.4_frame_058496_crop_002__d83b853d.jpg` (99×110 source, face 55×65, full-image laplacian 555.1, face-crop laplacian 443.6, **bg-only laplacian 1018.1**, face-area-fraction 33.50%).
  - Faces have near-identical sharpness (484 vs 444). What the model sees as the surrounding signal — virtual-background mountains, body silhouette, JPEG ringing on the small frame — is what diverges. **The eval frames' larger background-to-face ratio means the model's score is partly informed by non-face content that production crops would not include.**

- **The implication compounds existing shortcut findings.** The camera-signature shortcut ([`processing_signature_shortcut`](processing_signature_shortcut.md)) is "the model uses pipeline/processing signature on top of (or in place of) face artifact content." If the **eval substrate's background-to-face ratio is higher than production's**, then any pipeline-signal cue (codec, color, ISP noise, bg objects) the model uses gets a larger surface to read in eval than in production. That means: (a) eval FPR overstates production FPR for shortcut-driven false flags, (b) eval recall might overstate or understate production recall depending on whether the manipulation method's pipeline signal is read mostly from the face region or from the surrounding context, and (c) the close criterion of `shortcut-deployment-block` (in-progress, critical) is measured against an eval substrate whose mismatch with production is now an open variable.

- **The audit also exposes two adjacent eval-substrate hygiene issues** that compound the same gap (Findings 1 + 4):
  - **`is_no_face` slice is data degeneracy, not model weakness** (Finding 1): both no_face_real (n=174) and no_face_fake (n=45) slices contain frames where MediaPipe's "0 faces" verdict is correct — frames don't contain faces, or contain faces too degraded for any face-aware analysis. The 2026-04-27 investigation's "small but high-lift, 2× missed-as-fake" reading was data degeneracy, not a model signal.
  - **99×110 source crops sit in the eval substrate** (Finding 4): the PC_Generator__s15 reproduction frame has a 99×110 source-image — a thumbnail no production deployment would feed the model. The 2026-04-27 investigation's `face_pixel_area > 57k` cut catches the small-face problem indexed by face area but **misses the small-source-image problem** (99×110 frames pass `face_pixel_area > 1k` but fail any reasonable source-resolution bound like `min(width,height) >= 200`).

  These are not the headline finding (Finding 5 is), but they share the same axis: **the eval substrate has known data-quality issues that bias scoring.**

- **The 2026-04-27 investigation's slice-level FPR claims are partly confounded by the sharpness-metric bug** (Findings 2, 3 — see [`sharpness_metric_bug`](sharpness_metric_bug.md)). The "very sharp (>402): 45% FPR" entry in the 2026-04-27 investigation's per-quartile table is mixing two populations (genuinely sharp faces in sharp environments vs soft faces in tiny/busy/compressed crops where non-face content drives the laplacian). That's a separate thread; cited here because it's part of the same audit and also bears on how to read existing eval-substrate FPR breakdowns.

## Current stance (2026-04-29)

The eval substrate has a **structural mismatch with production deployment crops** that is independent of (and upstream of) the shortcut, FPR-dominance, and face-size-leak investigations the wiki already tracks. The mismatch is best described as:

1. Eval frames carry more background context around the face than production crops do.
2. Multiple existing eval-substrate FPR / recall numbers are partially driven by non-face content (camera signature, codec ringing, virtual-background objects, body silhouette, etc.) that a tighter production crop would suppress.
3. **All FPR numbers in the wiki — baseline lockbox 4.6%, modern_v2 0.71%, P13 modern_v2 30.2%, etc. — should be read with this caveat. They are not strict over- or under-estimates of production FPR; they are measurements on a substrate that gives shortcut-style cues a larger surface than production does.**
4. The translation from eval FPR to production FPR is **suspect for any failure mode that depends on background content**. The two failure modes the wiki flags as background-driven (camera/ISP signature; webcam-mode tail) inherit this caveat directly. The face-size leak inherits it indirectly: if the leak's mechanism is per-method crop-tightness signature, the gap between eval and production crops adds a confound to every "the leak is closing" measurement.
5. The deployment-block close criterion in `shortcut-deployment-block` (parent: [`processing_signature_shortcut`](processing_signature_shortcut.md)) reads `dor-real-webcam-false-flag-no-virtual-bg ≤ 0.30` on the eval substrate; whether hitting that target on eval would translate to a similar number in production is now a separately-open question. **Do not unilaterally rewrite the close criterion.** This thread tracks the disposition decision; the criterion stays as-is unless explicitly revised by the user.

The cleanest closure path is: **quantify the crop-tightness delta between eval and production, decide whether to retag eval at production tightness or apply a correction at scoring time.** The cheapest closure path is: re-crop a sample of the lockbox at production tightness and re-score; if the score distribution shifts materially, the gap is real and load-bearing. **Until that is done, every existing eval readout should carry the caveat.**

### 2026-04-30 evening update - quality-gate recrop-rescore landed

Source: `analysis/quality_gate_2026-04-30/phase2_recrop_rescore.py` and outputs `analysis/quality_gate_2026-04-30/outputs/{phase2_tightness_fpr_table.csv,phase2_normalize_result.json}`. The cheapest closure-path probe has now been run for P8A step5000 on the 839-frame lockbox and the larger dev split.

Important convention correction: in the local crop scripts, `t > 1.0` means tighter center crop, `t = 1.0` means native crop, and `t < 1.0` means looser crop via shrink plus edge padding.

P8A lockbox at tau=0.9741:

| Tightness | Direction | FPR | Recall | Reading |
|---:|---|---:|---:|---|
| 0.70 | looser | 7.5% | 15.8% | worse FPR, much worse recall |
| 0.85 | looser | 3.6% | 16.5% | slightly lower FPR, much worse recall |
| 1.00 | native | 4.6% | 32.9% | baseline |
| 1.20 | tighter | 3.1% | 23.1% | lower FPR, worse recall |
| 1.50 | tighter | 3.6% | 30.1% | slightly lower FPR, slightly worse recall |

The normalize-to-target experiment, which tightened only frames below `face_area_ratio=0.45`, changed 39.6% of lockbox frames and moved lockbox FPR from 4.6% to 8.0% with recall unchanged at 32.9%. On dev it moved FPR from 6.3% to 7.6% and recall from 66.3% to 80.3%.

This materially sharpens the thread's stance:

1. The crop-tightness gap is real and load-bearing: P8A score distributions and operating metrics move under crop intervention.
2. Current P8A should not be deployed with naive inference-time tightening as a shortcut fix. The lockbox normalize-to-target result is net-negative.
3. The right next question is training/evaluation substrate choice: select and document a stable crop policy, train or recalibrate under it, then measure FPR/recall. Do not infer from this P8A probe that recropping alone fixes the current checkpoint.
4. The open loop remains open because the probe does not yet quantify eval-vs-production crop distributions and does not establish a replacement canonical reporting surface. It does satisfy the "if scores shift materially, the gap is load-bearing" part of the cheap first move.

## Packet timeline

- *(no packet authored against this finding yet — surfaced 2026-04-29 afternoon as post-Slice-7 maintenance.)*
- The closest packet-side surface is the modern_lockbox_v2 build ([`webcam_fpr_dominance`](webcam_fpr_dominance.md)), which addressed one axis of eval-substrate hygiene (capture-mode tail) without re-cropping. Re-cropping at production tightness would be a different intervention class than the v2 filter.

## Evidence locations

- Audit notes (verbal, condensed; not in the repo): 2026-04-29 morning visual audit of 640 symlinked frames at `~/audit_slices_2026-04-29/index.html`. Audit's lineage doc lives at `~/Documents/repos/vault/wiki/ml-training/investigations/2026-04-27-lockbox-dev-property-investigation.md` (external to this canonical wiki).
- `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` — n=7,334 dev+lockbox tagged frames; the substrate parquet that supports the per-slice tables flagged by the audit.
- `analysis/quality_gate_2026-04-30/outputs/{phase2_tightness_fpr_table.csv,phase2_normalize_result.json}` — P8A recrop-rescore evidence showing material metric shifts and rejecting naive deployment-time tightening on current P8A.
- The reproduction frames (audit Finding 3): `dor_shkedi__frame_000059_seq906__fec59bc1.jpg` (494×393), `PC_Generator__s15_1754.4_frame_058496_crop_002__d83b853d.jpg` (99×110). Both are eval lockbox real, prob_fake > 0.97.
- Cross-thread anchors:
  - [`processing_signature_shortcut`](processing_signature_shortcut.md) — the camera-signature shortcut may be partly an eval-substrate artifact; the close criterion of `shortcut-deployment-block` is measured against an eval substrate whose crop tightness does not match production.
  - [`webcam_fpr_dominance`](webcam_fpr_dominance.md) — modern_v2 FPR 0.71% is the deployment-relevant number under the slice-mode caveat; under the crop-tightness gap, even modern_v2 may not be a strict production-FPR floor.
  - [`face_size_label_leak`](face_size_label_leak.md) — the face-size leak's mechanism is per-method crop-tightness signature; the eval-substrate's looser crop adds a background confound on top of the per-method signature.
  - [`sharpness_metric_bug`](sharpness_metric_bug.md) — the sharpness-metric bug is the mechanism by which the very-sharp-FP slice mixes two populations; the audit Finding 5 (this thread) and Findings 2-3 (sharpness thread) were discovered in the same session.

## Open loops

### Open loop: eval-production-crop-tightness-mismatch
status: open
severity: high
first_seen: 2026-04-29
last_verified: 2026-04-30
close_criterion: a written disposition is recorded — either (a) the crop-tightness delta between the eval substrate and production deployment crops is quantified (e.g., distributions of face_area_fraction or background_to_face ratio measured on a representative sample of eval frames vs production captures, with median/p90 of the delta) AND a rule is documented for translating eval FPR to production FPR (or for re-cropping eval at production tightness), OR (b) a packet retrains/re-evaluates with eval re-cropped at production tightness and the resulting headline FPR / recall numbers are recorded as the deployment-relevant readout, replacing the looser-crop substrate as the canonical reporting surface

This loop is structurally upstream of the shortcut-deployment-block (critical, in-progress) loop and the webcam-mode-fpr-dominance-headline-misleading (medium, open) loop. **High severity** because it potentially reframes how every existing eval result should be read — not just one slice or one method. The thread's current stance does not unilaterally rewrite any existing close criterion; it tracks the disposition decision so a future agent does not have to re-derive the question. The 2026-04-30 P8A recrop-rescore satisfies the cheap first move's "scores shift materially" test and rejects naive P8A inference-time tightening, but it does not yet close the loop because the eval-vs-production crop distributions and canonical reporting rule are still not written.

### Cross-thread refs

- [`processing_signature_shortcut`](processing_signature_shortcut.md) — the camera-signature shortcut may be partly an eval-substrate artifact. The `shortcut-deployment-block` (critical, in-progress) loop's close criterion reads against the eval substrate, so under this thread's open question the close criterion's translation to production is itself an open variable. **The two close criteria are not paired** (the shortcut close criterion does not need to wait on this one) — but a future agent reading `shortcut-deployment-block`'s close criterion cold should immediately see this thread's caveat.
- [`webcam_fpr_dominance`](webcam_fpr_dominance.md) — modern_v2 FPR is the deployment-relevant number under the capture-mode caveat; under this thread's caveat, even modern_v2 may not be a strict production-FPR floor. The two caveats compound.
- [`face_size_label_leak`](face_size_label_leak.md) — the face-size leak's mechanism is per-method crop-tightness signature; the eval-substrate's looser crop adds a background confound on top of the per-method signature. Closing the face-size leak at the eval-substrate level might leave a residual gap when the model deploys to tighter production crops.
- [`sharpness_metric_bug`](sharpness_metric_bug.md) — same audit, paired finding; the sharpness-metric bug is one of the mechanisms by which the eval-substrate's larger background surface drives a non-face signal into measurements that were nominally about face properties.
- [`eval_substrate_data_hygiene`](eval_substrate_data_hygiene.md) — the smaller-scope companion thread holding Findings 1 (`is_no_face` data degeneracy) and 4 (99×110 source-resolution floor not applied to eval). Those two issues do not require crop-tightness work to close, but they live on the same axis: the eval substrate has known data-quality issues that bias measurements.
