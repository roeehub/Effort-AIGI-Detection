# Phase 0 A0.2 — Agent Proposal (OPINION-ONLY) — 2026-05-22

> Opinion-only doc. Numbers and methodology live in `RESULTS_FACTS_2026-05-22.md`.

## Bottom line

The Phase-0 gate verdict on the SlotAv2_step3500 / L11 cell is **AMBIGUOUS** (`Δ = −0.0774`, `|kliep_proj_mean| = 0.0135`). The gate was designed so that the two numbers tell the same story; here they tell different stories.

- The Δ-side argues "there IS a fulcrum": the cross-substrate cluster contraction is real (`cos_pair − cos_within = −0.077`, ~5× the L8 gap and well past the −0.05 STRONG threshold).
- The KLIEP-side argues "there ISN'T a fulcrum on the axis we care about": the per-pair `(teams − clean)` vectors project onto the dev_real → lockbox_real discriminator axis with mean ~0.014 and ~10 % of pairs exceeding |proj| > 0.05 — i.e., the substrate translation is geometrically present but largely orthogonal to the in-deployment substrate axis the FPR pathology lives on.

My read is: **the substrate-pair lever has a fulcrum in CLIP-L11 geometry, but that fulcrum is approximately perpendicular to the FPR-causing substrate axis estimated by KLIEP on dev_real vs lockbox_real**. Pulling on the pair-loss fulcrum at full strength would shrink the (clean, teams) pair distance but would do almost nothing to the dev-vs-lockbox real-side mass shift. That is, the lever has the right local geometry but the wrong global direction.

This is a **soft refutation** of Track A's central premise as originally formulated. It is not as clean as "NO FULCRUM" because the cluster contraction is unambiguous, but the lever points away from the production problem.

## What I think the numbers mean

1. **The 3 ckpts have nearly indistinguishable L0/L4/L8 geometry, and even L11 `cos_pair` is identical to 0.001.** This is consistent with the shared OpenCLIP backbone + shared SVD residuals: P8A, T5C, and SlotAv2 all train shallow-layer SVD residuals (`k=32`) on the same OpenCLIP-B/16 ImageNet weights, and the only "real" divergence is in L11 + head. The pair-geometry at L11 differs between P8A and the other two on `cos_cross_id` (0.868 vs 0.720/0.714) — P8A's L11 has tighter inter-identity clustering, which is consistent with the A3.2-era finding that P8A's invariance is partly an artifact of broader identity collapse rather than substrate-invariance per se. SlotAv2 and T5C have spread identities apart (`cos_cross_id` dropped ~0.15) but did not commensurately tighten cross-substrate pairs (`cos_pair` stayed at 0.87). That asymmetry is the geometric residue of anchor_aware + T5C-base.

2. **The fact that `cos_pair` at L11 is the same (~0.87) across all 3 ckpts is the actual interesting datum.** It is not the kind of thing you would expect if SlotAv2's `anchor_aware` lever (or T5C's classifier-hidden-dim 1024 lift) had moved substrate-invariance in either direction. The lever moved `cos_cross_id`, not `cos_pair`. The "fulcrum" the pair-loss would attack — `cos_pair`'s gap to `cos_within_same` — is approximately the same on all three ckpts. That is bad news for the "SlotAv2 already filled the substrate-pair room and there's no headroom" reading; it is also bad news for the "SlotAv2 already broke substrate-invariance and pair-loss would un-break it" reading. The reading consistent with both these projections plus the geometry is: **SlotAv2/T5C/P8A all have roughly the same substrate-pair geometric structure at L11; the lever class is OPEN in geometric terms but the deployment-axis (KLIEP) shows little signal**.

3. **The KLIEP-axis orthogonality is the load-bearing finding here.** If the per-pair `(teams − clean)` difference had projected onto the KLIEP axis at mean ~0.3–0.5 (i.e., similar magnitude to the dev_real / lockbox_real centroid separation in the same units), we would have strong evidence that the substrate-pair lever attacks the same direction the deployment FPR sits on. Instead, the per-pair difference is ~30× smaller than that scale. The pair vector is almost a null vector w.r.t. the KLIEP axis.

4. **Sign-consistency across all 3 ckpts on the KLIEP projection (mean +0.005 to +0.015) is mildly informative.** The teams-side projects in the same direction as lockbox-real (which is also the direction of higher fake probability under the deployment head). The teams transport pushes pairs toward the FPR-prone region, but by an amount (~0.01 std-units) so small that pulling pairs back via a contrastive loss would close almost none of the FPR pathology. This is consistent with the 2026-05-04 finding that the viso→teams subtype is IQ-compounded rather than transport-specific.

5. **`score_corr` ordering (P8A 0.28 < SlotAv2 0.37 < T5C 0.43) is the opposite of what the "anchor_aware causes substrate stability" story would predict.** SlotAv2 has anchor_aware; T5C does not. T5C nonetheless has the highest pair-wise probability correlation. Possible explanations: (a) anchor_aware operates in feature space, not score space, and the head averages it away; (b) T5C's wider classifier hidden_dim is decorrelating less than SlotAv2's narrower one on cross-substrate pairs; (c) the 1,825-pair sample is on a different distribution (mostly HDTF + quickclips reals) than the dev/lockbox pools that determined the deployment FPR. I don't know which.

## Recommended next phase

**Fallback 1** (face-region attention pool CPU probe) per the master plan — but with a **two-track caveat**.

### Track A — Substrate-Pair Contrastive: I recommend DEFER, NOT PROCEED

Phase 0 was supposed to gate Phase 1 (loss-class infra + smoke). The KLIEP orthogonality finding tells me that even a perfectly-binding contrastive loss on the substrate-pair fulcrum we just measured would move the encoder along an axis that is ~89° away from the FPR-causing axis. The hypothesis that motivates Phase 1 — "pulling pairs together at L11 will move features off the substrate-shifted region" — is the part the numbers don't support. Proceeding to Phase 1 would burn a ~$30 GPU smoke on a lever whose direction we now have evidence is wrong.

This is not "NO FULCRUM" → "abort Track A forever." The lever exists; it just points elsewhere. If we later identify a different per-pair difference (e.g., on the IQ-PC1 axis from D10, or on a face-region attention-pool axis) where the projection magnitude is large, the substrate-pair contrastive infrastructure becomes useful. But not on the cross-substrate axis at L11 alone.

A0.3 (the conditional "anchor_aware ablation cheap probe") is **not needed** because the numbers above show SlotAv2 and T5C and P8A all have approximately the same `cos_pair` at L11 (within 0.001). The "anchor_aware filled the room" hypothesis it would test is already implausible given that P8A (no anchor_aware) shows the same `cos_pair`.

### Track B — Contract reframe: PROCEED as planned

Independent of A. Contract reframe + per-substrate τ-calibration is unconditionally valuable and was always queued in parallel.

### What I'd actually run next

1. **Fallback 1 (face-region attention pool CPU probe)** — ~1 day, $0. Same inventory, same 3 ckpts, same `clean ↔ teams` pair structure; measure whether a face-only-region attention pool of L11 (rather than the [CLS] token) gives a `cos_pair` that's noticeably different from the [CLS]-token number we just got (0.87). If face-pool `cos_pair` is markedly tighter than CLS `cos_pair` AND projects more strongly onto the KLIEP axis, the substrate-pair lever has a different fulcrum to attack and Phase 1 becomes viable. This is the same inventory + same forward features + same KLIEP axis → can re-use the cached features in `feats/`, no GCS calls, no GPU.
2. **Re-run KLIEP fit on Phase 0 cache features** as a sanity probe. We refit the KLIEP discriminator on `clip_frozen_l11__n4839.npz` (the OpenCLIP-frozen feature cache from D8). Now that we have L11 features from each of the 3 *trained* ckpts on a held-out 1,825-pair dataset, we can also refit a KLIEP-style LR on `(clean[0], teams[0])` pairs of the trained-encoder features. If the trained-encoder KLIEP axis aligns with the frozen-CLIP KLIEP axis at cos > 0.5, the orthogonality finding is robust to encoder change. If the axes are themselves orthogonal, the dev/lockbox shift the KLIEP axis represents may already be partially squashed by training. Either way the answer informs Track A direction.
3. **Track B contract reframe** — proceed in parallel.

### What I'd NOT do

- **Do NOT launch Phase 1 GPU smoke on the substrate-pair contrastive loss as currently scoped.** The geometric fulcrum exists; the deployment-axis projection does not. Building the loss-class infra without evidence the lever direction matches the FPR axis is a 6-week-class mistake we've already learned.
- **Do NOT compute A0.3 (anchor_aware ablation probe)** unless someone reads this and disagrees with my reading of why all 3 ckpts have the same `cos_pair`. The numbers don't seem to need it.

## What I am most uncertain about

- Whether the 1,825-pair (mostly HDTF + quickclips) sample is representative of the substrate axis the deployment FPR actually sits on. Lockbox is dominated by webcam captures (per memory `project_lockbox_fpr_dominated_by_webcam_mode`); HDTF + quickclips are mostly news / political video. The KLIEP axis was fit on dev_real vs lockbox_real (so it carries the webcam-mode signal), but the per-pair difference vectors were computed on HDTF + quickclips visomaster_teams pairs. There is a domain mismatch between the axis and the projected vectors. A direct check would be to run the same Phase-0 probe on `lockbox`-substrate `(clean, teams)` pairs if such pairs exist on disk. Per `D9_FACTS_2026-05-12.md` they do not exist cross-bucket; only within-bucket pairs.
- Whether `cos_pair = 0.87` at L11 is "the contraction we have to break" or "a structural feature of the OpenCLIP-B/16 encoder under cross-substrate transport that no detector-head training will move." The numbers above can't tell those apart; you would need a frozen-CLIP baseline measurement (run Phase 0 on the un-finetuned OpenCLIP backbone with no trained head). That's a ~1-hour additional probe; might be worth adding to Fallback 1's scope.

## Decision arrow

→ **Fallback 1** (face-region attention pool CPU probe) + Track B contract reframe in parallel.

→ Do **NOT** proceed to Phase 1 (loss class + smoke) until Fallback 1 surfaces a fulcrum whose KLIEP projection is at least 10× larger than what we just measured.
