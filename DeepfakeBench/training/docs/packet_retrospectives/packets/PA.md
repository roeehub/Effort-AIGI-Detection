# Packet PA  ·  Single-lever data-availability test (visomaster_enhanced + visomaster_teams_enhanced on E2B)

> **In-flight stub authored 2026-05-04 evening.** This packet is currently training on Vertex (us-east1, JOB_STATE_RUNNING). This file holds the pre-launch design + reasoning. Results sections will be filled in when training + scorecard land (~24h ETA from launch). Memory entries and threads referenced here may evolve in parallel.

## Status card

| Field | Value |
|---|---|
| Dates | 2026-05-04 → in-flight |
| Slots | 1 (Packet A only) |
| Headline lever | Enable `visomaster_enhanced` + `visomaster_teams_enhanced` data sources at `family_weight=4.0` on E2B baseline (B16 scratch + CE + heavy aug) |
| Leader slot | Packet A (single slot) |
| Leader metric | TBD (training in flight) |
| Verdict | 🟡 in-flight |
| Next-packet decision | TBD; depends on whether viso recall lifts above E2B baseline (8.4% F0 / 30.9% F4) under deployment-honest single-τ at 5% FPR |
| Themes touched | [`viso_bucket_gap`](../threads/viso_bucket_gap.md), [`clean_teams_identity_pairing`](../threads/clean_teams_identity_pairing.md), [`eval_substrate_data_hygiene`](../threads/eval_substrate_data_hygiene.md) |

## Configuration

- **Base recipe**: E2B (B16 scratch + CrossEntropy + heavy aug) — the new anchor candidate per `project_e2b_breaks_deeplive_ceiling_2026-05-04.md`. NOT FT-from-P8A; this is a single-lever delta from E2B.
- **Single lever** (vs E2B): enable two new data sources in `combined_paired:`:
  - `visomaster_enhanced.enabled: true` (gcs_bucket: `visomaster-enhanced-face-cropped`, exclude_tiers: `[ARTIFACT]`)
  - `visomaster_teams_enhanced.enabled: true` (resolver_manifest_uri, enhanced_bucket: `enhanced-visomaster-cropped`, companion_domains: `[teams_v2]`, p_original: 0.5)
- **Family weight**: `family_weights.visomaster_enhanced_fake = 4.0` (matches the base `visomaster_fake` weight in E2B; intentionally NOT fw=8.0 which collapsed in P14_DATA_FIX with the bundle confound).
- **Bundle status**: NO anchor_aware, NO pipeline_random, NO additional aug intervention beyond the E2B baseline. This is the bundle-decomposition discipline applied to the data axis.
- **Seed**: 3023.
- **Yaml**: `experiments/phase2_round13/R13_PA_VISOMASTER_ENHANCED_DATA.yaml` (uncommitted as of 2026-05-04 evening).
- **Vertex job**: `3140330896851206144`, us-east1, image `1.3.256`. State: `JOB_STATE_RUNNING`.

## Why this packet (pre-launch reasoning)

Per memory `project_data_axis_lever_pulled_twice_no_lift.md`, the data axis was framed as exhausted on the basis of two prior tests:
- `xan4dfto` (P14_DATA_FIX): bundle + visomaster_teams_enhanced fw=8.0 → collapsed (other_fakes_tpr=0.047).
- `rmic6wrc` (P16_DATA_AXIS): fw=2.0 → didn't lift recall above 1.1% calibrated.

Both tests had confounds: P14_DATA_FIX stacked the data lever with the anti-shortcut bundle (per `anti_shortcut_bundle_decomposition`); P16 used fw=2.0 which is below the documented family-weight threshold for visomaster_fake. Neither was a clean single-lever test of "what does enabling the missing data distribution do at fw=4.0 on a clean base?"

The 2026-05-04 morning CPU diagnostic pass (Jobs 1-7 + Job 14) further established that:
- The viso ceiling on the F0 (full) substrate is 27% on P8A (the strongest available ckpt), unbroken across 13+ R13 packets (memory `project_viso_ceiling_unbroken_10_packets.md`).
- F4 substrate cleaning lifts P8A viso to 67%, E3 to 77.6%, E2B to 30.9% (memory `project_job14_substrate_clean_2026-05-04.md`).
- E2B's frozen-feature representation can separate caught-vs-uncaught viso fakes at AUC 0.97 (memory `project_viso_head_boundary_finding_2026-05-04.md`).

Combined: the data-axis lever was structurally undertested; the viso ceiling has a representation-side floor (E2B head-boundary finding) but the impact of enabling the missing data has not been measured cleanly. Packet A is the clean retest.

User-context constraint: the 72h ship goal (deeplive + viso recall across the board, low FPR, robust) requires ANY lever that could lift viso recall. Packet A is one of two GPU packets launched in parallel to maximize parallel exploration; Packet C-codec is the other (see [PC.md](PC.md)).

## Results at the time

*(Pending training completion + scorecard run; ETA ~24h from launch.)*

To be reported when results land:
- Trainer-side `value_composite` trajectory + final.
- Promotion-contract scorecard verdict (default policy + v3 policy with `--promotion_target_fake_recall_min 0.30` if launcher passes the flag).
- F0 vs F4 substrate-cleaning re-eval via `analysis/substrate_cleaning_eval_2026-05-05/run_clean_eval.py`.
- Per-substrate τ-calibration via `analysis/per_substrate_tau_calibration_2026-05-05/run_calibration.py`.
- Comparison vs E2B_3200 baseline on viso, deeplive, teams_fake at deployment-honest single-τ at 5% / 10% / 20% FPR ceilings.

## Conclusions drawn in-session

*(Pending results.)*

- **Pre-launch session ID**: documented in `docs/relaunch_handoffs/SESSION_LOG_2026-05-04.md` (the in-flight handoff for the 2026-05-04 session that drafted + launched both Packet A and Packet C-codec).

## Retrospective

*(To fill in when results land.)*

## Source files

- **Yaml**: `experiments/phase2_round13/R13_PA_VISOMASTER_ENHANCED_DATA.yaml` (uncommitted)
- **Pre-launch handoff**: `docs/relaunch_handoffs/SESSION_LOG_2026-05-04.md`
- **Vertex job**: `3140330896851206144` (us-east1, image 1.3.256)
- **W&B**: `https://wandb.ai/dtect-vision/phase2r13-experiments` (run id assigned after job starts logging)
- **Memory pointers** (load-bearing for design): `project_data_axis_lever_pulled_twice_no_lift.md`, `project_viso_head_boundary_finding_2026-05-04.md`, `project_job14_substrate_clean_2026-05-04.md`, `project_e2b_breaks_deeplive_ceiling.md`, `project_viso_ceiling_unbroken_10_packets.md`, `project_clean_teams_same_identity.md`
- **Companion analysis (parallel work that informed launch)**: `analysis/codec_aug_verification_2026-05-05/`, `analysis/pair_loss_effect_verification_2026-05-05/`, `analysis/substrate_cleaning_eval_2026-05-05/`, `analysis/per_substrate_tau_calibration_2026-05-05/`
