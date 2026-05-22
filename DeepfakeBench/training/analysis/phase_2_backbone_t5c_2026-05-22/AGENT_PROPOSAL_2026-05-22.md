# Phase 2 BACKBONE-T5C — Agent proposal (2026-05-22)

> **Class: opinion / recommendation.** Reads FACTS (`RESULTS_FACTS_2026-05-22.md`) + memory (CPU-2 β-outcome with secondary cohort asymmetry 62:1, T5C step3500 base candidates 2026-05-12, CPU-3 γ-outcome).

---

## What's done

Implementation phase complete. The substrate_pair_asymmetric loss is implemented with asymmetric gradient (teams detached) and registered under LOSSFUNC. The BACKBONE-T5C yaml is in the working tree. 20 new unit tests cover the loss + collate + stamper end-to-end. 138 tests pass in the broader sweep with no regressions.

## What's not done

Vertex launches are NOT submitted. Pending user authorization for ~$50 of GPU spend and a ~3.5h wall-clock window for the full run, plus scorecard time post-completion.

## Recommendation: parallel launch with BACKBONE-SlotAv2

Per the plan, both BACKBONE runs are independent — different loss families on different base ckpts in different US regions. Launching them in parallel uses both regions' A100 capacity simultaneously and yields the head-to-head comparison the plan calls for: SlotAv2 has anchor_aware (controls for invariance lever) while T5C does not (single-lever delta on the new asymmetric pair-loss).

### Phase A (this turn, after user OK): both smokes

- BACKBONE-SlotAv2 smoke: `us-west4`, 200 steps, ~$3, ~10 min
- BACKBONE-T5C smoke: `us-central1`, 200 steps, ~$3, ~10 min

Smoke pass criteria for BACKBONE-T5C specifically:
- `loss/overall` decreases monotonically
- `train/loss/substrate_pair_asymmetric` is non-zero at some step (proves the loss is firing on actual matched pairs from the viso_teams_enhanced lane)
- `prob_fake(clean)` catches up to `prob_fake(teams)` on the cohort partition (CPU follow-up — not measurable in-run)
- BCE classifier accuracy doesn't drop > 5pp from base
- Cohort asymmetry ratio (target:wrong_way at τ=0.20 on a held-out paired set) drops from 62:1 toward 5:1 or below

If the substrate_pair_asymmetric loss reads 0.0 across the whole smoke (no matched pairs in any batch), the loss is structurally dormant and the smoke is INVALID for this packet — the only fixable cause is wiring the HDTF + quickclips lanes first (the small viso_teams_enhanced lane may not surface matched pairs at the 32-frame batch granularity).

### Phase B (after smoke passes): both full launches

| Gate (BACKBONE-T5C) | Threshold |
|---|---|
| composite λ=1.0 | ≤ 0.30 |
| lockbox_real_fpr | ≤ 0.025 |
| lockbox_fake_recall | ≥ 0.70 |
| dev_fake_macro_recall | ≥ 0.40 |
| Abort condition | composite > T5C base composite + 0.05 |

## Confidence

30-40% any single ckpt clears the composite ≤ 0.30 gate. The mechanism confidence is conditional on matched pairs surfacing in the batch:
- If the practical pair-fraction is ≥ 5% of the batch, the loss has a fighting chance to bite (60% mechanism-works confidence).
- If it's < 1% (the realistic case today given the 54-row viso_teams_enhanced lane), the loss is mostly a no-op — the smoke will reveal this within 100 steps.

## The inventory-coverage gap matters MORE here than for SlotAv2

GroupDRO doesn't need matched pairs; the asymmetric pair-loss does. If we want the pair-loss lever to actually exercise the 62:1 asymmetry CPU-2 found, the loss must see clean+teams pairs in the same batch. Today only 54 inventory rows reach the batch. The other 1,826 sit in unread GCS buckets.

**Pragmatic recommendation.** Treat the first BACKBONE-T5C smoke as a diagnostic. If `train/loss/substrate_pair_asymmetric` reads ~0 across the smoke, abort the full launch and instead pivot to writing the HDTF + quickclips data lane code (next-agent task, ~3-4h CPU). Then re-launch BACKBONE-T5C with the lanes wired.

## Alternative: bypass the inventory gap with a manual sampling override

A cheaper short-term fix to surface matched pairs without writing new data lanes: enable `combined_paired.visomaster_teams_enhanced.p_original=1.0` (force the teams_v2_companion branch on every sample of that lane) AND raise its `family_weights` entry to 8.0-10.0. This still only exercises 54 identities but maximizes their batch frequency. Documented as an option but not encoded in the yaml — the FACTS yaml uses the standard p_original=0.5 family weight 4.0 setup matching T5C base.

## Items needing user decision

1. Authorize smokes ($6 + ~20 min wall) for BOTH BACKBONE-SlotAv2 (us-west4) AND BACKBONE-T5C (us-central1).
2. If BACKBONE-T5C smoke shows the pair-loss is structurally dormant (read: ~0 across the smoke), decide between:
   - (a) Skip full T5C launch and write HDTF+quickclips data lanes first.
   - (b) Launch full T5C anyway as a no-op control (the multi_axis_grl will still train; the new lever is just disabled in practice).
3. Confirm parallel launch (both regions) vs sequential (SlotAv2 first, T5C after results known).
