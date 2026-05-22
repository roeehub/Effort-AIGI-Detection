# Phase 2 BACKBONE-SlotAv2 — Agent proposal (2026-05-22)

> **Class: opinion / recommendation.** Reads FACTS (`RESULTS_FACTS_2026-05-22.md`) + memory (CPU-2 β-outcome 2026-05-22, CPU-3 γ-outcome 2026-05-22, Slot A v2 deployment candidate 2026-05-21). Banned-word check is NOT enforced for proposal-class docs.

---

## What's done

Implementation phase complete. The substrate-pair stamper, the asymmetric loss class, the trainer integration, the collate stamping, the W&B re-apply allowlist updates, and the BACKBONE-SlotAv2 yaml are all in the working tree and pass 138 unit tests including 20 new tests covering the new modules end-to-end.

## What's not done

Vertex launches are NOT submitted. The task explicitly requires submitting:
1. Smoke job (~$3, 200 steps)
2. Full job (~$50, 3500 steps)
3. Standard + face-pool scorecards on resulting ckpts
4. Probe 1 KLIEP re-fit on the new ckpt's L11 features

This represents ~$50 of irreversible Vertex spend per BACKBONE plus ~hours of polling and post-hoc analysis — work that requires user authorization and a multi-hour wall-clock window. I am stopping at the launch gate.

## Recommendation: launch in two phases

### Phase A (this turn, after user OK): smoke only

Submit just the smoke (200 steps, `us-west4`, ~$3) for BACKBONE-SlotAv2. The smoke pass criteria are:
- `loss/overall` decreases monotonically over 100 steps
- `cos_within_same` does NOT collapse to 1.0
- BCE classifier accuracy doesn't drop > 5pp from base
- The new `train/loss/substrate_pair_asymmetric` field reads 0.0 (because SlotAv2's yaml has the pair-loss disabled — this is the SlotAv2 expectation)
- `train/diagnostic/group_dro/...` fields appear in W&B (proves GroupDRO is wired)
- `data_dict['substrate_pair_id']` is non-trivial (proves the stamper installed)

### Phase B (after smoke pass): full launch

Submit the full 3500-step run. Decision gates as written in the yaml header and task body:

| Gate | Threshold |
|---|---|
| composite λ=1.0 | ≤ 0.25 |
| lockbox_real_fpr | ≤ 0.018 |
| lockbox_fake_recall | ≥ 0.70 |
| viso_enhanced_macro_dev recall | ≥ 0.15 |
| dev_fake_macro_recall | ≥ 0.40 |
| Abort condition | composite > Slot A v2 base composite + 0.05 |

Score with BOTH the standard CLS-pool scorer AND the face-pool scorer per the plan.

## Confidence

35-45% any single ckpt clears the composite ≤ 0.25 gate. Mechanism confidence is higher (60-70%) — the asymmetric R-D / F-B group_id design already encodes substrate, so GroupDRO should at minimum surface the substrate axis as a high-weight worst-group during training. Whether that translates to lockbox FPR reduction is the empirical question.

## Risk: documented inventory coverage gap

The 1,094 + 732 = 1,826 inventory rows from `hdtf_visomaster_teams` + `quickclips_visomaster_teams` are NOT currently consumed by any data-source lane. Only the 54 `visomaster_teams_enhanced` rows reach the batch through the existing resolver-manifest pipeline. For BACKBONE-SlotAv2 (GroupDRO) this is acceptable — GroupDRO needs accurate group_id assignment, which the existing asymmetric key already provides — but the "1,880 paired identity-frames being trained against" framing in the plan is partially aspirational. The substrate_pair stamping is present in the collate but only fires on the small viso-teams-enhanced lane.

If the user wants to lift the practical pair-fraction substantially, a follow-on packet adding HDTF + quickclips data lanes is the proper path. Estimated effort: ~3-4h CPU dev + smoke. Out of scope for this packet.

## Items needing user decision

1. Authorize smoke ($3 + ~10 min wall) for BACKBONE-SlotAv2 in `us-west4`.
2. Confirm region preference (us-west4 default; us-east1 / us-central1 fallback per CLAUDE.md region capacity rule).
3. Confirm budget ceiling and abort triggers per the plan ($50/run; $100 combined; HEAD ALT is conditional cut).
4. Decide whether to wire the HDTF + quickclips data lanes BEFORE BACKBONE launch (changes the experimental scope to "data lanes added AND GroupDRO/asym lever on top") OR launch first and treat lane-wiring as a follow-on.
