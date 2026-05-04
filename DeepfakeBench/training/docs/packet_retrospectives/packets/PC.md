# Packet PC-codec  ·  Stacked data-availability + Teams codec aug on E2B

> **In-flight stub authored 2026-05-04 evening.** This packet is currently training on Vertex (us-east1, JOB_STATE_RUNNING). This file holds the pre-launch design + reasoning. First launch attempt failed in 30s due to a wandb-entity gotcha (see Configuration); relaunched as `1202657157175050240`. Results sections will be filled in when training + scorecard land (~24h ETA from relaunch).

## Status card

| Field | Value |
|---|---|
| Dates | 2026-05-04 → in-flight |
| Slots | 1 (Packet C-codec only) |
| Headline lever | Same as Packet A (enable `visomaster_enhanced` + `visomaster_teams_enhanced` data) PLUS enable `teams_codec_simulation` aug with `policy: adaptive_mixture` |
| Leader slot | Packet C-codec (single slot) |
| Leader metric | TBD (training in flight) |
| Verdict | 🟡 in-flight |
| Next-packet decision | TBD; depends on whether the data + codec aug stack lifts viso recall above E2B and above Packet A under deployment-honest single-τ |
| Themes touched | [`processing_signature_shortcut`](../threads/processing_signature_shortcut.md), [`viso_bucket_gap`](../threads/viso_bucket_gap.md), [`anti_shortcut_bundle_decomposition`](../threads/anti_shortcut_bundle_decomposition.md), [`eval_substrate_data_hygiene`](../threads/eval_substrate_data_hygiene.md) |

## Configuration

- **Base recipe**: same as Packet A — E2B (B16 scratch + CE + heavy aug).
- **Lever 1 (data, same as Packet A)**: enable `visomaster_enhanced` + `visomaster_teams_enhanced` at fw=4.0.
- **Lever 2 (codec aug, NEW for Packet C)**: enable `teams_codec_simulation` block in `combined_paired.augmentation:`:
  - `enabled: true`
  - `probability: 0.5` (matches P8A historical value, Slice-5 codec_hedge used 0.65)
  - `policy: "adaptive_mixture"` (routes enhanced families to heavier mode)
  - `enhanced_families`: `visomaster_enhanced_fake`, `deeplive_enhanced_fake`, `proper_visomaster_enhanced_clean_fake`, `proper_visomaster_enhanced_teams_fake`
  - `exclude_families`: all `*_teams_*` buckets — don't apply synthetic codec aug to data that already carries real Teams codec fingerprint (would double-codec it). Specifically: `deeplive_teams_real`, `deeplive_teams_fake`, `visomaster_hints_teams_real`, `visomaster_hints_teams_fake`, `proper_visomaster_teams_fake`, `proper_visomaster_enhanced_teams_fake`, `proper_real_teams`.
- **Bundle status**: NO anchor_aware, NO pipeline_random beyond E2B baseline. The two enabled levers (data + codec aug) are intentionally STACKED not isolated; this packet is a multi-lever test in violation of the strict bundle-decomposition discipline. Justification: the two levers are complementary not orthogonal (data adds the missing distribution; codec aug shapes existing distribution to look like the missing one) and the 72h ship goal called for parallel exploration. Packet A is the data-only single-lever counterpart that ISOLATES the data lever; comparing PA vs PC isolates the codec aug lever's contribution.
- **Seed**: 3024 (PA uses 3023; deliberate offset).
- **Yaml**: `experiments/phase2_round13/R13_PC_CODEC_PLUS_DATA.yaml` (uncommitted as of 2026-05-04 evening).
- **First launch attempt** (FAILED): Vertex job `318825730303590400`, us-east1, image `1.3.257`, JOB_STATE_FAILED in 30s. Root cause: I exported `WANDB_ENTITY=roeehub` (a wandb username, not the entity that exists in this project's wandb workspace `dtect-vision`); container's `wandb.init` crashed with HTTP 404. Workaround: `unset WANDB_ENTITY` and let the launcher default flow through. See [`wandb_yaml_propagation_bugs`](../threads/wandb_yaml_propagation_bugs.md) "2026-05-04 evening update" for the bug-class entry.
- **Relaunch**: Vertex job `1202657157175050240`, us-east1, image `1.3.257`. State: `JOB_STATE_RUNNING`.

## Why this packet (pre-launch reasoning)

Two complementary motivations:

1. **Data lever** (same as Packet A): the missing distribution `visomaster_enhanced` + `visomaster_teams_enhanced` has not been cleanly tested at fw=4.0 on a non-FT base. Per memory `project_data_axis_lever_pulled_twice_no_lift.md`, the prior tests were confounded.

2. **Codec aug lever**: `data/augmentations/teams_simulation.py:104` `TeamsCodecSimulation` is calibrated against measured Teams transport deltas (sharpness −50.9%, brightness +19.0%, HF energy −77.4%, etc.). On 2026-05-04 evening (`analysis/codec_aug_verification_2026-05-05/`), this calibration was independently re-verified on 30 paired (raw, teams) viso frames: cosine 0.87-0.99 across 8 IQ axes, magnitude ratio 97-106%. The aug is a faithful simulation of actual transport. The Slice-6 finding that codec_hedge did NOT promote (`april-26-training-master-plan-v2.LOG.md:390-422`) was about the lockbox-readout effect of higher codec aug intensity on the prior recipe family (P11/RLP6), NOT about whether the aug class faithfully models transport. The two findings are independent.

The combination tests whether: (a) adding the missing data + (b) augmenting clean viso to look like teams-transported viso jointly closes the gap on the user-flagged primary deployment threat (teams-transported viso, per memory note from this session).

User-context constraint: the 72h ship goal called for parallel exploration. Packet A (data-only) and Packet C-codec (data + codec aug) launched in parallel from same FT base; comparing the two will isolate the codec-aug contribution.

**Pair loss was considered as a third packet** (per session in-flight 2026-05-04 evening). Sub-agent verification at `analysis/pair_loss_effect_verification_2026-05-05/FINDINGS.md` returned LOW verdict — premise empirically refuted on E2B for viso (sign of effect REVERSED; opposite-direction cohort 29/275 vs target 3/275 at τ=0.5). Pair loss was not drafted as a packet. See [`clean_teams_identity_pairing`](../threads/clean_teams_identity_pairing.md) DEBATE block (2026-05-04).

## Results at the time

*(Pending training completion + scorecard run; ETA ~24h from relaunch.)*

To be reported when results land:
- Trainer-side `value_composite` trajectory + final.
- Promotion-contract scorecard verdict.
- F0 vs F4 substrate-cleaning re-eval.
- Per-substrate τ-calibration.
- Comparison vs E2B_3200 baseline AND vs Packet A on viso (especially teams-transported subtype), deeplive, teams_fake at deployment-honest single-τ.
- Per-pair score correlation between Packet A and Packet C-codec on the 275 paired viso fakes (does codec aug move scores in a meaningful direction?).

## Conclusions drawn in-session

*(Pending results.)*

- **Pre-launch session ID**: documented in `docs/relaunch_handoffs/SESSION_LOG_2026-05-04.md` (the in-flight handoff for the 2026-05-04 session).

## Retrospective

*(To fill in when results land.)*

## Source files

- **Yaml**: `experiments/phase2_round13/R13_PC_CODEC_PLUS_DATA.yaml` (uncommitted)
- **Pre-launch handoff**: `docs/relaunch_handoffs/SESSION_LOG_2026-05-04.md`
- **Vertex jobs**: `318825730303590400` (FAILED — wandb entity bug), `1202657157175050240` (RUNNING)
- **W&B**: `https://wandb.ai/dtect-vision/phase2r13-experiments`
- **Memory pointers**: `project_data_axis_lever_pulled_twice_no_lift.md`, `project_e2b_breaks_deeplive_ceiling.md`, `project_viso_ceiling_unbroken_10_packets.md`, `project_clean_teams_same_identity.md`, `feedback_promotion_contract_launch.md` (wandb-entity gotcha)
- **Calibration evidence (load-bearing for codec aug lever)**: `data/augmentations/teams_simulation.py:6-22` (original 18-video × 132-frame calibration), `analysis/codec_aug_verification_2026-05-05/FINDINGS.md` (2026-05-04 independent re-verification on 30 paired viso frames)
- **Companion analysis (parallel work)**: `analysis/pair_loss_effect_verification_2026-05-05/FINDINGS.md` (informed the decision to NOT add a pair-loss packet to this slate), `analysis/substrate_cleaning_eval_2026-05-05/`, `analysis/per_substrate_tau_calibration_2026-05-05/`
