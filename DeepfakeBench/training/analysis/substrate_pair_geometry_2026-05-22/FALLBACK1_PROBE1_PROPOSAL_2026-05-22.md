# Fallback 1 (Face-Region Pool) + Probe 1 (KLIEP Re-Fit) — Agent Proposal (OPINION-ONLY) — 2026-05-22

> Opinion-only doc. Numbers + methodology in `FALLBACK1_PROBE1_FACTS_2026-05-22.md`.
> Companion to A0.2's `AGENT_PROPOSAL_2026-05-22.md` (which read the A0.2 result as "soft refutation of Track A").

## Bottom line

Probe 1's result **substantially weakens the A0.2 "orthogonality" reading** without overturning it. Probe 2 results are pending (sentinel marker `_probe_complete.json` not yet written; ~3–4 h ETA from launch at 00:19 local).

The shape of Probe 1's evidence:

- Each of the 3 trained encoders carries a **highly discriminative substrate axis at L11**: a logistic-regression substrate classifier on the trained-encoder L11 features achieves 0.980–0.984 held-out accuracy. The substrate signal at L11 IS present in the trained-encoder features.
- The per-pair `(teams − clean)` direction projects ~0.13 onto the per-ckpt substrate axis, with σ ~ 0.04 — i.e., the pair-direction is heavily one-sided (uniform sign across pairs) and well-separated from zero.
- The per-pair direction projects only ~0.005 to ~0.015 onto the frozen-CLIP-L11 KLIEP axis (the A0.2 finding, reproduced exactly).
- The two axes are ~84°–88° apart by cosine: trained-encoder substrate axis is nearly but not perfectly orthogonal to the frozen-CLIP KLIEP axis.

**The A0.2 reading was: "the substrate-pair fulcrum is orthogonal to the FPR axis."** Probe 1 reframes this: the substrate-pair lever has a strong fulcrum in the trained-encoder space (10–23× larger projection than on the frozen axis), but that fulcrum is not on the same axis the dev_real → lockbox_real KLIEP discriminator was fit on. The trained encoders have rotated the substrate axis relative to the CLIP-frozen substrate axis.

This is **not** the same as "the substrate-pair lever is dead." It IS the same as "the substrate-pair lever points at a substrate axis the encoder constructed during training, NOT at the dev/lockbox substrate axis that drives deployment FPR."

## What changed vs A0.2's reading

A0.2 measured the per-pair direction's projection on the **frozen-CLIP KLIEP axis** (one fixed axis, fit on a different population — dev_real vs lockbox_real CLIP-frozen features) and found it near-zero. The natural inference was: the substrate-pair fulcrum is geometrically present but doesn't move features along the FPR axis. A0.2's bottom line was "the lever has the right local geometry but the wrong global direction."

Probe 1 measures the per-pair direction's projection on the **trained-encoder substrate axis** (one axis per ckpt, fit on the trained-encoder L11 features themselves) and finds it large (0.12–0.14). The pair direction is NOT a null vector in feature space; it just lies on a different axis than the one A0.2 measured.

The reframing matters for Track A because:

1. The "no fulcrum" half of the A0.2 verdict was based on the frozen KLIEP axis being the right axis to measure. Probe 1 shows the trained-encoder substrate axis is a strictly stronger discriminator on the very data we're looking at (clean vs teams sides of the 1,825-pair inventory). Of the two axes, the per-ckpt axis is the **operative substrate axis** for the trained encoder under cross-substrate transport.
2. Cosine ~0.04–0.10 between the two axes means they are not redundant. A pair-loss that pulls pair vectors toward zero on the per-ckpt axis would NOT directly move them on the frozen-CLIP KLIEP axis. The trained-encoder substrate axis is the right target for an *encoder-side* contrastive lever; the frozen-CLIP KLIEP axis describes a *deployment population* shift that the encoder has partially rotated away from.

## What I think the numbers mean

1. **Training rotated the substrate signal off the frozen-CLIP axis without removing it.** All three ckpts share the same OpenCLIP backbone and the same shallow-layer SVD; the L0–L8 geometry was already identical across them per A0.2 §3. Probe 1's per-ckpt axes are ~84°–88° from the frozen-CLIP axis, and acc 0.98 — so the trained encoders construct substrate-discriminating features at L11 that are nearly orthogonal in direction to the frozen-CLIP substrate axis. This is consistent with the L11 layer doing the substrate discrimination work and doing it on a learned representation, NOT on a passively inherited CLIP feature.

2. **The KLIEP-orthogonality finding is encoder-dependent, NOT axis-invariant.** A0.2's pair_proj on the frozen axis is small because the frozen axis points one way and the trained-encoder's substrate signal points another way. If we re-ran A0.2 measuring projection on **the trained-encoder's own substrate axis**, we would see μ ≈ 0.13 with σ ≈ 0.04 — i.e., a fulcrum that meets the master plan's "STRONG FULCRUM" threshold of |proj| ≥ 0.4 in *relative* terms (ratio 9–23× the frozen baseline) but does NOT meet the 0.4 absolute threshold the gate was written with.

3. **Per-ckpt axes are not interchangeable.** P8A's axis lies closer to the frozen axis (cos 0.10) than SlotAv2's (0.04) or T5C's (0.06). Of the three ckpts, P8A has the smallest pair-projection magnitude on its own axis (0.119) and the largest ratio (22.9× the frozen-axis projection). SlotAv2 and T5C have larger trained-axis projections (~0.134) and lower trained-vs-frozen ratios. This is consistent with the A0.2 observation that P8A's L11 has tighter inter-identity clustering (`cos_cross_id = 0.868` vs 0.72) — P8A's L11 is more compressed overall, so its substrate axis has less amplitude on the pair direction even though it's discriminative.

4. **Sign-consistency on both axes is suggestive but not load-bearing.** All three ckpts have positive pair_proj_mean on both the per-ckpt axis and the frozen axis. The teams side projects in the +1 (teams-class) direction of the per-ckpt classifier (~tautology, since the axis is fit on this very data) AND in the +1 (lockbox-real-class) direction of the frozen-CLIP KLIEP classifier (~0.01 magnitude — small but not zero). The frozen-axis projection sign agreement is a weak repeat of the A0.2 observation that the teams transport pushes pairs *toward* the FPR-prone region by a small amount.

5. **`cos(w_trained, w_frozen) = 0.04–0.10` is the load-bearing finding for Track A direction.** Two interpretations are consistent with this:
   - **(a) Substrate signal genuinely lives on multiple, partially-overlapping axes**, and the L11 trained encoder picks up one (the cross-substrate clean-vs-teams direction); the dev/lockbox shift lives on another (the frozen-CLIP KLIEP direction). Pulling on the per-ckpt axis fulcrum would tighten cross-substrate pair distances on the data the encoder sees during training, but would NOT close the dev/lockbox distribution gap.
   - **(b) The two axes are different projections of one underlying substrate signal**, and rotating the encoder along the per-ckpt axis (via pair-loss) would induce a partial rotation along the frozen axis too. Without an intervention experiment we can't tell (a) from (b).
   - My read: (a) is more likely given that the frozen-axis projection sign is positive but tiny (~0.01) AND the trained-axis projection is large and one-signed. If (b) held, we'd expect either the frozen-axis projection to grow when the trained-axis projection grows (which we can't test without a packet) or the two axes to be closer in cosine than 0.04. The orthogonality is suggestive but is exactly the kind of question the next experiment would answer.

## Probe 2: what I expect and what would change my mind

Probe 2 is still running at write time. Expected directional outcomes:

| Probe 2 finding | Implication |
|---|---|
| Face-region pool `delta_pair_vs_within` markedly more negative than CLS pool (−0.077) at SlotAv2 → e.g., −0.15 or lower | Face-region pool tightens cross-substrate pairs more than CLS — substrate signal lives in BG/chrome patches and face-pool removes part of it |
| Face-region pool `delta_pair_vs_within` ~ −0.077 (same as CLS pool) | Substrate signal is uniformly distributed across the face crop; pooling change does not help; lever class no different on face-pool than on CLS |
| Face-region pool `delta_pair_vs_within` near zero or positive | Face-pool removes the substrate cluster contraction; substrate signal lives outside the face region; face-pool is a viable inference-side lever (different question than the contrastive-loss question) |
| Face-region pool kliep_projection_mean ≥ 0.4 on at least SlotAv2 | Phase 1 face-region contrastive becomes viable — fulcrum AND axis-alignment both present on face-pool features |
| Face-region pool kliep_projection_mean ~ A0.2's value (~0.013) | Frozen-CLIP KLIEP axis is still orthogonal; face-pool does NOT solve the KLIEP-orthogonality problem; the substrate axis the encoder constructed is the dominant axis regardless of pool |
| Face-region pool kliep_projection_mean magnitude 5–10× A0.2's | Partial improvement; face-pool is on the right rotation but doesn't fully align with frozen axis |

The Probe 2 outcome I'd bet on (~60 %, low confidence): face-region pool gives a `delta` slightly more negative than CLS (maybe −0.10 instead of −0.077) and a `kliep_projection_mean` ~ 1.5–2× A0.2's. The substrate signal is partly in the face region (because the teams pipeline modifies the face) and partly outside (because the teams pipeline also adds chrome/codec borders that occupy outer patches). The face-pool would tighten the face-only substrate fulcrum a little but wouldn't dramatically realign it with the frozen-CLIP KLIEP axis.

## Recommended next phase

### Decision matrix (assuming Probe 2 completes successfully)

| Scenario | Recommendation |
|---|---|
| Probe 2 face-pool kliep_proj_mean ≥ 0.4 on SlotAv2 | Phase 1 face-region contrastive smoke; the fulcrum + axis-alignment finally both line up. ~$30 GPU. |
| Probe 2 face-pool kliep_proj_mean in [0.05, 0.4] OR delta ≤ −0.15 | Phase 1 face-region contrastive smoke at REDUCED scope: cosine_mse only (skip InfoNCE and triplet), single yaml, single seed. ~$15 GPU. If this binds, escalate. |
| Probe 2 face-pool kliep_proj_mean ~ 0.013–0.05 AND delta ~ −0.07 to −0.10 | **Defer Track A; pivot to encoder-side intervention scoped to the TRAINED-ENCODER substrate axis.** This is the case where Probe 1's reframing matters most: the substrate axis exists, the pair-direction lies on it, but it's the *encoder's own* axis. A pair-loss at L11 would attack that axis — but whether closing pair distance on that axis also moves the dev/lockbox shift is unknown without an intervention. The cheap probe to inform this: train a 200-step smoke with `cosine_mse` λ_pair=0.3 on Slot A v2 base and measure (i) does the per-ckpt substrate axis classifier accuracy drop below 0.90 in 200 steps? (ii) does the frozen-CLIP KLIEP axis projection shift? If (i) yes and (ii) no, the two axes are genuinely independent and Track A's deployment value is limited. If (i) yes and (ii) yes, they couple. |
| Probe 2 face-pool delta near zero or positive | Track A as substrate-pair contrastive deferred; instead, face-region pool as an **inference-side** lever — replace the CLS pool with the face-region pool at evaluation (no retraining). This is a different intervention class. Worth a one-evening scorecard. |

### What I'd actually queue next

1. **(Pending Probe 2)** Wait for `_probe_complete.json` to appear at `analysis/substrate_pair_geometry_2026-05-22/_probe_complete.json`. ETA ~3–4 h from launch (00:19 local 2026-05-22). On completion, read `per_ckpt_face_region_cosines.csv` + `face_region_kliep_projections.csv` + the sentinel's `verdict_summary` field. Apply the decision matrix above.
2. **In parallel — independent of Probe 2** — a 1-hour CPU follow-up using the per-ckpt substrate axes Probe 1 just produced: refit a per-ckpt classifier accuracy on the **dev_real / lockbox_real CLIP-frozen** features projected onto each per-ckpt axis (uses the existing `clip_frozen_l11__n4839.npz` cache from D8). The question: does the trained-encoder substrate axis (fit on clean-vs-teams) also discriminate dev_real-vs-lockbox_real? If yes — i.e., one axis lives in two distributions — the substrate signal is a single underlying axis, not two independent ones. This sharpens the Probe 1 finding's deployment implication.
3. **Track B remains independent and proceeds as planned.** The composite-tiebreak landed on 2026-05-22; user λ pick and live rerun pending. None of the Probe 1 / Probe 2 outcomes touch Track B.

### What I'd NOT do

- **Do NOT proceed to Phase 1 GPU smoke on substrate-pair contrastive using the CLS-pool features as the lever target.** A0.2 + Probe 1 establish that the CLS-pool pair direction projects strongly onto the per-ckpt axis but weakly onto the frozen-CLIP KLIEP axis. Pulling on the CLS-pool pair lever would move features along the per-ckpt axis (and likely improve cross-substrate `cos_pair` on the training population), but the deployment FPR pathology sits on the dev/lockbox axis which is ~84°–88° away. The 6-week-class mistake risk the A0.2 proposal flagged is unchanged.
- **Do NOT report A0.2's verdict as "ORTHOGONALITY CONFIRMED" without the Probe 1 caveat.** The two axes are not the same axis; the A0.2 reading was correct only for the frozen-CLIP axis, not for the operative substrate axis the trained encoders carry.

## What I am most uncertain about

- **Whether the per-ckpt substrate axis is itself a deployment-relevant axis.** Probe 1 only shows the axis is highly discriminative on the 1,825 pair inventory (HDTF + quickclips + viso_teams). It does NOT show whether projecting dev_real and lockbox_real features onto this same axis would discriminate them. The follow-up in §"What I'd actually queue next" §2 addresses this. Until it runs, the per-ckpt axis could be a "training-pop" axis that has no deployment correspondence.
- **Whether `cos(w_trained, w_frozen) = 0.04` is genuine orthogonality or an artifact of training-pop vs deployment-pop axis difference.** If both axes were fit on the same population, we'd expect them to align more closely; the 0.04 cosine could be telling us about population mismatch rather than about substrate-signal direction.
- **Whether Probe 2's face-pool will distinguish "substrate signal lives outside the face crop" from "substrate signal is uniform across patches but is encoded in inter-token attention pathways at L11."** The face-pool measurement does NOT separate these.

## Decision arrow

→ Wait for `_probe_complete.json` sentinel (~3–4 h ETA from 2026-05-22 00:19 local).
→ Apply the decision matrix above on the sentinel's `verdict_summary` + the two CSVs.
→ In parallel: run the 1-hour per-ckpt-axis dev/lockbox discrimination follow-up (CPU, $0). This is a strict superset of "should we proceed to Phase 1" diagnostics.
→ Track B unaffected; continues per its own timeline.
