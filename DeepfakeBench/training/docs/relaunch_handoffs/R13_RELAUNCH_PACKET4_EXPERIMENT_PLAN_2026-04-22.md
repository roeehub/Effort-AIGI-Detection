# R13 Packet-4 Experiment Plan — 8 overnight slots, half-half control/spatial backbone

**Generated**: 2026-04-22 ~21:10 UTC
**Branch**: `teams-relaunch-root-2026-04-17`
**Status**: **Ready for launch pending user approval.** 8 yamls drafted; 4 asia-southeast1 slots freed via cancel.

## TL;DR

Packet 4 runs **8 experiments overnight** against image `1.3.195` (no code changes required; yaml-only). Core structure is a half-half split:
- **4 control-backbone slots**: arcface-margin scan (m=0.125, 0.175) + seed-B replication of the current leader + Teams-gap intervention (C2).
- **4 spatial-backbone slots**: mirrors the control-backbone structure on the P3-leader spatial augmentation (`context_variation_shift: 0.08`, `context_variation_individual_p: 0.40`).

All slots inherit the packet-3.5 NEW `value_composite` gate `(target_mean_fpr=0.03, max_pool_fpr=0.05, stability_jitter_stat=p95)`. C1 (proper_visomaster_teams as OOD lane) was dropped from tonight's plan — it requires code work in the OOD assembler (~2-4h) and is deferred to a post-training eval task on the best packet-4 checkpoint.

## Context: where P3.5 left us

- **Current P3.5 leader**: slot 02 `arcface_m=0.15`, control backbone, `value_composite = 0.7442`. Only lever with `Δ ≥ +0.03` vs. retro-scored P3 baseline. Two arcface variants (m=0.10, m=0.15) tie at the top.
- **P3 leader under new gates** (retro-scored): slot 05 `low_arcface + spatial` at `0.7232`. Spatial aug is the P3 winning lever.
- **m=0.20 is blocked**: P3.5 slot 03 sits at max_fpr ≈ 0.055 and never clears the 0.05 gate. Margin ceiling is somewhere between 0.15 and 0.20.
- **22pp Teams gap** at deployed tau: teams_tpr 0.72 vs other_tpr 0.93 on the leader checkpoint. Packet-3.5 §5 slot-07 decision was DON'T FIRE stack_top3; only arcface passed the +0.03 rule.
- **Stress-lane stability** (p95 jitter 0.60-0.97) was not validated by user visual inspection. Stability_lambda dropped from tonight's plan pending that review.

## The 8 experiments

| # | Name | Backbone | Key delta from `R13_RLP35_02_arcface_m015` | Region | Hypothesis |
|---|------|----------|----------------------------------------------|--------|------------|
| 1 | `R13_RLP4_01_arcface_m0125_control` | control | `arcface_m: 0.125` | `us-west4` | Localize the margin optimum between known-tied m=0.10 (0.7435) and m=0.15 (0.7442). |
| 2 | `R13_RLP4_02_arcface_m0175_control` | control | `arcface_m: 0.175` | `us-east1` | Probe the upper margin ceiling. If this blocks like m=0.20, the ceiling is tight near 0.15. |
| 3 | `R13_RLP4_03_arcface_m015_seedB` | control | `seed: 737 → 742` (top-level only; identity split_seed stays 737) | `asia-southeast1` | Confirm the 0.7442 leader peak is not a seed fluke. |
| 4 | `R13_RLP4_04_arcface_m015_teams_reweight` | control | `family_weights.proper_visomaster_teams_fake: 1.0 → 3.0` | `asia-southeast1` | Up-weighting Teams-family fakes should narrow the 22pp teams_tpr gap. |
| 5 | `R13_RLP4_05_arcface_m0125_spatial` | spatial | `arcface_m: 0.125` + `context_variation_shift: 0.04 → 0.08` + `context_variation_individual_p: 0.15 → 0.40` | `us-west4` | Margin gap-fill on the spatial backbone. |
| 6 | `R13_RLP4_06_arcface_m015_spatial` | spatial | spatial aug block (see slot 5) | `us-east1` | **Strongest new-leader candidate** — stacks the P3.5 margin win on the P3 spatial win. |
| 7 | `R13_RLP4_07_arcface_m0175_spatial` | spatial | `arcface_m: 0.175` + spatial aug block | `asia-southeast1` | Upper-margin probe on spatial backbone; completes the m ∈ {0.125, 0.15, 0.175} scan on spatial. |
| 8 | `R13_RLP4_08_arcface_m015_spatial_teams_reweight` | spatial | spatial aug block + `proper_visomaster_teams_fake: 1.0 → 3.0` | `asia-southeast1` | Teams reweighting on spatial backbone; mirrors slot 4 to test additivity with the spatial win. |

### Coverage matrix

| Lever | Control backbone | Spatial backbone |
|-------|------------------|------------------|
| arcface m=0.10 | P3.5 slot 01 (done, 0.7435) | — |
| arcface m=0.125 | **slot 1** | **slot 5** |
| arcface m=0.15 | P3.5 slot 02 (done, 0.7442 ★) + **slot 3** (seedB) | **slot 6** |
| arcface m=0.175 | **slot 2** | **slot 7** |
| arcface m=0.20 | P3.5 slot 03 (blocked, done) | — (would also likely block) |
| Teams reweight | **slot 4** | **slot 8** |

Full arcface margin scan on both backbones; Teams intervention covered twice (one per backbone); seed replication on the current leader.

## Slot allocation (respects user region constraints)

```
us-west4         slot 1  arcface_m=0.125 control
us-west4         slot 5  arcface_m=0.125 spatial
us-east1         slot 2  arcface_m=0.175 control
us-east1         slot 6  arcface_m=0.15  spatial  ★ (new-leader candidate)
asia-southeast1  slot 3  arcface_m=0.15  seedB
asia-southeast1  slot 4  arcface_m=0.15  + teams reweight
asia-southeast1  slot 7  arcface_m=0.175 spatial
asia-southeast1  slot 8  arcface_m=0.15  spatial + teams reweight
```

- `us-west4` (2 jobs): 1st-choice region, known-good today.
- `us-east1` (2 jobs): includes the most-likely-winner candidate (slot 6). Has a GPU-availability risk but the launcher will fail fast if quota is exhausted.
- `asia-southeast1` (4 jobs): after cancelling 4 jobs to free slots. All 7 P3 slots + 1 P3.5 slot had been occupying this region.
- `europe-west4`: P3.5 slot 05 (label_smoothing) still running, **not touched** per user constraint.
- `us-central1`: **reserved by user, not used**.

## Cancels executed (pre-approved)

4 jobs cancelled in `asia-southeast1` at 21:07 UTC to free slots for packet 4:

| Job ID | Name | Reason | Peak persisted? |
|--------|------|--------|-----------------|
| `3243173098479943680` | `RLP3_02_FT_proper_main` | Retro-scored, peak captured (vc=0.7027) | ✓ |
| `7438839101328982016` | `RLP3_04_FT_proper_spatial` | Retro-scored, peak captured (vc=0.7215) | ✓ |
| `5533253508997840896` | `RLP3_05_FT_proper_low_arcface_spatial` | Retro-scored, peak captured (vc=0.7232, P3 new-gate leader) | ✓ |
| `7562969566058381312` | `RLP35_03_arcface_m020` | Blocked on worst_pool_fpr 22+ hours; margin ceiling signal saturated | N/A (never unblocked) |

Remaining P3 jobs in `asia-southeast1` (not touched): slots 01, 03, 06, 07 — will complete naturally; peaks can be verified before any future cancel.

## Launch plan

Once user approves:

```bash
# One job per line. Uses the canonical launcher and image 1.3.195.
# All jobs use W&B project enhanced-aug-test. Names match yaml `name:`.

./scripts/launch/launch_experiment.sh -y enhanced-aug-test us-west4 \
    experiments/phase2_round13/R13_RLP4_01_arcface_m0125_control.yaml
./scripts/launch/launch_experiment.sh -y enhanced-aug-test us-west4 \
    experiments/phase2_round13/R13_RLP4_05_arcface_m0125_spatial.yaml

./scripts/launch/launch_experiment.sh -y enhanced-aug-test us-east1 \
    experiments/phase2_round13/R13_RLP4_02_arcface_m0175_control.yaml
./scripts/launch/launch_experiment.sh -y enhanced-aug-test us-east1 \
    experiments/phase2_round13/R13_RLP4_06_arcface_m015_spatial.yaml

./scripts/launch/launch_experiment.sh -y enhanced-aug-test asia-southeast1 \
    experiments/phase2_round13/R13_RLP4_03_arcface_m015_seedB.yaml
./scripts/launch/launch_experiment.sh -y enhanced-aug-test asia-southeast1 \
    experiments/phase2_round13/R13_RLP4_04_arcface_m015_teams_reweight.yaml
./scripts/launch/launch_experiment.sh -y enhanced-aug-test asia-southeast1 \
    experiments/phase2_round13/R13_RLP4_07_arcface_m0175_spatial.yaml
./scripts/launch/launch_experiment.sh -y enhanced-aug-test asia-southeast1 \
    experiments/phase2_round13/R13_RLP4_08_arcface_m015_spatial_teams_reweight.yaml
```

Recommended launch order: 1/5 (us-west4 first — fastest to hit RUNNING), 2/6 (us-east1 second — monitor for quota push-back), 3/4/7/8 (asia-southeast1 last — slots just freed).

## Gotchas / warnings

- **Cancelled jobs may take ~60-120s to fully release their slots** even after transitioning to CANCELLED. If asia-southeast1 launches fail with quota, wait a minute and retry.
- **All 8 yamls share the `anneal_steps: 8000` convention** (matches P3.5 base, not P3). ArcFace anneal completes inside the 10k training steps.
- **Header comment block** at the top of each packet-4 yaml still references packet-3.5 (copy artifact). The `name:` and parameters are correct; the lineage comment is stale. Cosmetic only.
- **Slot 3 (seedB)**: only the top-level `seed: 742` changed. `combined_paired.split_seed` and `combined_paired.seed` stay at 737, and `external_training_reals[0].identity_split_seed` stays at 737. This preserves the train/held-out identity partition so the Δ vs. seed-737 runs measures training-order RNG variance, not data-split variance.
- **Slots 4 and 8 (Teams reweight)**: only `proper_visomaster_teams_fake` family weight changes (1.0 → 3.0). `deeplive_teams_fake` stays at 5.0. This is per-deployment-category reweighting (Teams context), not per-method (the facedancer case the user declined).
- **Don't auto-commit** `VERSION`, this plan doc, the 8 yamls, or `HANDOFF.md` unless the user explicitly asks.

## C1 deferred: post-training eval

Adding `proper_visomaster_teams_fake` as an OOD fake lane requires a manifest-aware OOD source in `data/sources/combined_paired.py` (the existing `external_fake_sources` schema does a blind bucket walk and only supports VCD-style identity filtering). Rough scope:
- Add a new `proper_data_fake_sources` block under `ood_monitoring`.
- In `_collect_fake_sources`: branch for proper-data manifest entries, load the lockbox split via the `lockbox_ratio` field.
- Ensure identity exclusion uses the proper-data identity hash, not the VCD regex.

Scope estimate: ~2-4 hours including tests. Deferred to after packet 4 lands — can be run as a retro-style eval on the best packet-4 checkpoint via a small extension of `retro_score_value_composite.py`.

## Setup required (unchanged from packet-3.5)

- GCP project `train-cvit2`, account `roee@dtectvision.ai` (authed).
- W&B entity `dtect-vision`, project `enhanced-aug-test`. Key hardcoded in `scripts/launch/launch_experiment.sh:89`.
- Image `1.3.195`. No code changes needed for packet 4 → no rebuild.
- Local bash 3.2 (macOS) for any monitoring scripts.

## When the runs finish — summary/results process

After overnight, expected completion around 07:00-10:00 UTC 2026-04-23. Pull:

1. `best_value_composite/metric` per slot from W&B (already logged by the trainer).
2. Rank the 8 slots against the current leader 0.7442.
3. Evaluate whether slot 6 (arcface_m=0.15 × spatial) is the new leader, and whether Teams reweighting (slots 4 + 8) moves the teams_tpr gap.
4. Decide packet-5 direction: (a) further margin-scan refinement, (b) Teams intervention scale-up, (c) new lever exploration.

Draft results doc at `docs/relaunch_handoffs/R13_RELAUNCH_PACKET4_RESULTS_2026-04-23.md` using the packet-3 retro-score results doc as a template.
