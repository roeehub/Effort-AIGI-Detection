# R13 Packet-4 Planning Handoff — pick 6 overnight experiments, fill the 6 free slots

**Generated**: 2026-04-22 ~18:50 UTC
**Branch**: `teams-relaunch-root-2026-04-17` (HEAD `c7ac78c`, not pushed)
**Status**: **Planning in progress** — experiments not yet selected; slot plan drafted but needs user sign-off before any job launches.

## ⚠ What you are being asked to do

Plan **6 experiments** (exactly — the user specified six slots) to run overnight.
Slot placement constraint: **launch into `asia-southeast1`, `us-east1`, and/or `us-west4` only.**
Do **NOT** place any job into `us-central1`. User explicitly reserved that region.

Before launching: produce a concrete `packet-4` experiment plan (yaml set + launch matrix) for the user to approve. **Do not launch jobs without user sign-off.** Prior-session feedback explicitly flagged auto-launching jobs without prior-session approval as unwanted.

## Goal

Pick the 6 experiments that have the highest expected information value given what we now know about P3.5, **respecting the user's explicit preferences** (see "User preferences" below) and the region/slot constraint. Produce:

1. A `packet-4` plan doc at `docs/relaunch_handoffs/R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md` describing the 6 chosen experiments, the rationale, and the slot matrix (which job goes in which region).
2. 6 yaml files at `experiments/phase2_round13/R13_RLP4_*.yaml` (one per experiment), ready to fire via `scripts/launch/launch_experiment.sh` once the user approves.
3. A "jobs to cancel first" section listing which current jobs in the target regions must be terminated to free the 6 slots.

## Where we are (what the user knows about and what P3.5 showed)

### Current P3.5 leader: **slot 02 arcface_m=0.15, `value_composite` = 0.7442**

Full profile in `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_RETRO_SCORE_RESULTS_2026-04-22.md`. TL;DR:

- **In-distribution**: val_holdout AUC 0.9894, EER 0.00386, F1 0.9701. 11 of 19 fake methods at 100%. Very strong.
- **Teams gap (the important finding)**: at the deployed tau=0.9767:
  - `teams_tpr = 0.7154` on the OLDER `teams_ood_fake` pool (bucket `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2`, 270 monitored videos).
  - `other_tpr = 0.9335` on `wma_failure_fake` (1083 monitored videos).
  - **22pp gap** = at the exact same operating point, the model catches 93% of non-Teams fakes but only 72% of Teams-context fakes. Teams-call pipeline degradations (compression, low bitrate, small faces) push Teams fakes closer to the decision boundary.
- **Newer proper Teams data** (`proper_visomaster_wave_2026_04_19_provisional`) is scoring ≥83% on val/test but (i) only 4 of 9 swap methods have val samples, (ii) model saw 297/342 total proper_teams samples in training (identity-split held-out, but not method-held-out), (iii) eval at EER threshold, not deployed tau. So the 95%-ish proper-Teams accuracy is an in-distribution-style number. It is NOT yet in the OOD monitoring set.
- **Stress-lane instability**: p95 jitter 0.60–0.97 on `ood_spatial_stress_{crop_shift,rotation,scale}` and `ood_lighting_stress_{warm_harsh,backlight_dim,general}`. This is where the 0.35 stability number comes from. Real-data jitter (teams_ood_fake p95 = 0.031) is very stable.
- **Checkpoint fragility**: peak at step 6500 is a ~1500-step window; model re-blocks by step ~9000 with max_fpr=0.097. The current leader is a moment, not a plateau.

### P3.5 leaderboard under NEW gates `(0.03, 0.05, p95)` as of now:

| Slot | Hypothesis          | State      | best_vc |
|------|---------------------|------------|:-------:|
| 01   | arcface_m=0.10      | SUCCEEDED  | 0.7435  |
| 02 ★ | arcface_m=0.15      | SUCCEEDED  | **0.7442** |
| 03   | arcface_m=0.20      | RUNNING    | (blocked worst_pool_fpr — max_fpr≈0.055, continues to fail 0.05 gate) |
| 04   | stability_lambda=0.03 | SUCCEEDED | 0.7094 |
| 05   | label_smoothing=0.05 | RUNNING    | 0.7080 (latest 18:03 UTC) |
| 06   | family_rebalance    | SUCCEEDED  | 0.7113  |

### Retro-scored P3 top-3 under the same NEW gates:

| Slot | Hypothesis                   | retro VC |
|------|------------------------------|:--------:|
| 02   | main (baseline)              | 0.7027   |
| 04   | spatial                      | 0.7215   |
| 05 ★ | low_arcface + spatial (stack)| **0.7232** (P3 leader under new gates) |

### Slot-07 decision (packet-3.5 §5 rule): **DON'T FIRE**
Only arcface_m=0.15 clears the +0.03 threshold vs retro-P3_02 (+0.0415). Stability_lambda (−0.008 vs slot 02 baseline; with baseline drift it's now ~+0.007 at best), label_smoothing (−0.005 with latest run still improving). Rule requires ≥2 of 3; only 1 passes. Stacking heterogeneous hypotheses would dilute the single real win.

## User preferences (respect these when designing packet 4)

1. **AVOID: targeted facedancer augmentation or per-family data weighting.** User explicitly declined this lever even though facedancer is the weakest family at 75% val_holdout accuracy.
2. **PENDING: stress-augmentation visual review.** User wants to visually inspect sample frames from `ood_spatial_stress_*` and `ood_lighting_stress_*` presets before deciding if the 0.35 stability number is a model weakness or a test-harness harshness. If the user does this check before packet-4 launches, it will inform whether stability_lambda is worth burning a slot on.
3. **INTERESTING: late-phase stability_lambda injection** (last ~20% of steps) — proposed way to preserve the arcface win while damping stress-lane jitter. Candidate experiment.
4. **IMPORTANT: the 22pp Teams gap** — user reacted strongly ("wow, this might be important"). Packet 4 should include at least one experiment that directly addresses this: either by adding proper_visomaster_teams to the OOD monitoring set (measurement), by re-weighting Teams-family fakes in training (intervention), or both. User specifically asked "how are we doing over there [on the newer proper Teams data]?" — so an experiment that answers this quantitatively is high-value.
5. **DON'T auto-commit** `HANDOFF.md`, `VERSION`, yamls, or planning docs unless the user explicitly asks. The user has flagged this preference multiple times across prior sessions.
6. **DON'T auto-launch jobs.** Produce the plan; get approval; then launch.

## Slot plan — 6 slots needed in `asia-southeast1` / `us-east1` / `us-west4`

Current occupancy (as of ~18:30 UTC, confirm before planning):

| Region | Jobs to clear | Notes |
|--------|---------------|-------|
| `asia-southeast1` | 7 P3 training jobs + 1 P3.5 (slot 03 arcface_m=0.20, still blocked after ~22 hours) | All P3 training is effectively DONE — their best checkpoints are already captured and were retro-scored today. P3.5 slot 03 has been blocked the entire run; unlikely to break through. Safe to cancel all 8 IF their best_value_composite/step and best_value_composite/gcs_path are confirmed checkpoint-stable in W&B. Validate in W&B before cancel. |
| `us-east1` | 0 | Already empty. Open capacity but today had GPU-availability issues (slot 05 retro was stuck 85 min PENDING with no contention, had to be relocated). Use with caution — consider as 2nd-choice region. |
| `us-west4` | 0 | Already empty. 1st-choice region — slots 02 and slot-05-relaunch went through cleanly here today. |
| `europe-west4` | P3.5 slot 05 (label_smoothing) | Still running, still below the +0.03 threshold. NOT a target region per user; leave alone. |
| `us-central1` | (not a target region per user) | Do not touch. |

**Proposed slot allocation for 6 experiments** (planner should adjust based on chosen experiments):

```
us-west4         slot 1   (1st choice, known-good today)
us-west4         slot 2
asia-southeast1  slot 3   (after cancelling freed P3 slot)
asia-southeast1  slot 4   (after cancelling freed P3 slot)
asia-southeast1  slot 5   (after cancelling freed P3 slot)
us-east1         slot 6   (accept GPU-availability risk)
```

Rationale: us-west4 is the lowest-risk region today. Spread across regions to hedge quota contention. us-east1 is riskier but the plan needs 6 and Singapore+us-west4 together might not cleanly give 6 independent slots without stacking.

### Safety check before cancelling asia-southeast1 jobs

Confirm each job's peak is persisted:

```bash
for JOB in 3243173098479943680 7438839101328982016 4506432793957367808 \
           5533253508997840896 1977661603188834304 7854859116907331584 \
           8730809244430893056 7562969566058381312; do
  echo "=== $JOB ==="
  gcloud logging read "resource.labels.job_id=\"$JOB\" AND textPayload:\"best_value_composite/gcs_path\"" \
    --project=train-cvit2 --limit=1 --format='value(textPayload)' --order=desc
done
```

If any job has no `best_value_composite/gcs_path` logged (e.g. slot 03 P3.5 which has never unblocked), DO NOT cancel without user approval — user may want to let it keep trying. Flag to user for a per-job decision.

## Candidate experiment menu

Pick 6 from this list (or propose alternatives, with rationale). All should build on the arcface_m=0.15 win since that's the only lever with +0.03 at the +0.03 threshold vs baseline.

### Tier A — high signal, low risk

**A1. arcface-margin scan: m=0.125.**
- Rationale: arcface_m=0.10 (0.7435) and arcface_m=0.15 (0.7442) are essentially tied. m=0.125 fills the gap and helps localize the optimum.
- Base: `R13_RLP35_02_arcface_m015.yaml`. Change: `arcface_head.margin: 0.125`.
- Expected VC: 0.73-0.75.

**A2. arcface-margin scan: m=0.175.**
- Rationale: m=0.20 is blocked, m=0.15 peaks. m=0.175 tells us if the sweet spot is 0.15 or just-above.
- Base: `R13_RLP35_02_arcface_m015.yaml`. Change: `arcface_head.margin: 0.175`.
- Expected VC: 0.70-0.74 (may start to hit the m=0.20 blocker).

**A3. arcface_m=0.15 × spatial backbone (stacked lever).**
- Rationale: P3 leader is slot 05 (low_arcface+spatial) at 0.7232. Layering the arcface win onto a spatial-trained backbone is the cleanest "stack the winners" experiment.
- Base: `R13_RLP35_02_arcface_m015.yaml`. Add spatial_aug block from `R13_RLP3_04_FT_proper_spatial.yaml`. Keep all else identical.
- Expected VC: could be the new leader if effects add, flat/down if they conflict.

**A4. arcface_m=0.15 seed B (replication / seed variance).**
- Rationale: before investing in more derivative experiments, confirm the 0.7442 peak is not a seed fluke. The P3 runs used seedB variants for this purpose (slot 06 main_seedB).
- Base: `R13_RLP35_02_arcface_m015.yaml`. Change: `seed: 742` (or any non-737 canonical).
- Expected VC: 0.72-0.75 if the win is real. If much lower, the peak is seed-brittle.

### Tier B — high signal if user validates stress augmentations first

**B1. arcface_m=0.15 × late stability_lambda=0.03 (last 20% of steps).**
- Rationale: preserve arcface win, damp stress-lane jitter. Currently-running stability_lambda=0.03 (P3.5 slot 04) applied from step 0 got 0.7094 — notably below the arcface peak. Applying stability_lambda only in the last 20% of steps (~step 7200 onwards for a 9000-step run) may give us the best of both worlds.
- Base: `R13_RLP35_02_arcface_m015.yaml`. Add stability_lambda schedule: off until step 7200, then 0.03. **Requires trainer support** — verify in `trainer/trainer.py` whether stability_lambda can be schedule-gated before launching. If not supported, either add the schedule (small code change) or drop this experiment.
- **Gating on user visual-inspection check**: if the user decides stress-lane jitter is a test-harness artifact rather than a real weakness, drop this experiment.

**B2. arcface_m=0.15 × stability_lambda=0.01 (lighter, full-run).**
- Rationale: P3.5 slot 04 used 0.03 which dragged the main VC down. A lighter 0.01 through the whole run might give partial stability damping without killing the arcface win.
- Same gating as B1.

### Tier C — addresses the Teams gap (the user cares about this)

**C1. Measurement-only: arcface_m=0.15 + `proper_visomaster_teams` added to OOD monitoring set.**
- Rationale: directly answers the user's question "how are we doing on the newer proper Teams data at the deployed tau?" The answer tells us whether the 22pp gap is a property of the OLDER teams_ood_v2 pipeline specifically, or a real model blind spot.
- Base: `R13_RLP35_02_arcface_m015.yaml`. Add a new entry under `external_fake_sources` for `proper_visomaster_teams_fake` (hold out the method-level identity-split to avoid evaluating on train data — reuse the lockbox split in the existing manifest). **Requires yaml + manifest + code coordination.** Most likely needs a small change in the OOD assembler to accept proper-data lanes as an OOD fake source — verify the existing path_contains/grouping schema supports it before promising this as a yaml-only change.
- Expected: if `teams_tpr` on proper_visomaster_teams >> 0.72, the 22pp gap is mainly about the old teams_ood_v2 pipeline and we should de-prioritize it. If it's still ~0.72 or worse, the model genuinely has a Teams-pipeline blind spot.

**C2. Intervention: arcface_m=0.15 + increased Teams-fake family weight.**
- Rationale: if the Teams gap is a real weakness (and C1 confirms it), up-weighting `deeplive_teams_fake` or `proper_visomaster_teams_fake` in the identity-family weights may narrow it. Current weights: `deeplive_teams_fake: 5.0`, `proper_visomaster_teams_fake: 1.0`. Try raising proper_visomaster_teams_fake to 2.0 or 3.0.
- Base: `R13_RLP35_02_arcface_m015.yaml`. Change `identity_family_weights.proper_visomaster_teams_fake: 3.0`.
- ⚠ This is per-family reweighting and the user said they want to avoid facedancer-specific weighting — confirm that Teams reweighting is OK (it's an architectural preference question: Teams is a deployment category, facedancer is a fake-generation method; I'd guess the user meant per-method not per-family).

### Tier D — nice-to-have, lower priority

**D1. Checkpoint stability sweep** — rerun arcface_m=0.15 with more frequent `ood_monitoring` cadence (every 250 steps instead of every 500) to catch the peak more precisely. The current run's peak was a narrow 1500-step window; finer cadence might find a higher peak within that window.

**D2. arcface_m=0.15 × larger ID-consistency batch** — if the arcface head is bandwidth-limited on identity contrast, raising the batch from current to 2× may give a cleaner gradient. Trainer support required — verify before planning.

### User's original suggestion from this session

> "Packet-4 suggestion: arcface-margin scan (m ∈ {0.125, 0.15, 0.175}) on the spatial-enabled baseline, not heterogeneous stacking."

That reads as a ~3-experiment core (A1 + A3 + A2 variants on the spatial backbone). The remaining 3 slots should cover replication (A4), the Teams-gap question (C1 or C2), and possibly one stability idea (B1 if the user validates stress augs first).

## Suggested 6-experiment plan (starting point — planner to refine)

| # | Experiment | Tier | Region |
|---|------------|------|--------|
| 1 | arcface_m=0.125 (control backbone) | A1 | us-west4 |
| 2 | arcface_m=0.175 (control backbone) | A2 | us-west4 |
| 3 | arcface_m=0.15 × spatial backbone  | A3 | asia-southeast1 |
| 4 | arcface_m=0.15 seed B              | A4 | asia-southeast1 |
| 5 | arcface_m=0.15 + proper_visomaster_teams in OOD | C1 | asia-southeast1 |
| 6 | arcface_m=0.15 × late stability_lambda=0.03 (last 20% steps) | B1 | us-east1 |

Covers: arcface-margin sweep (3 points), backbone stacking, seed variance, Teams-gap measurement, stability lever. Drops: C2 intervention (conservative — wait for C1 result), D* tier.

**Planner: validate this against what the user actually wants before finalizing.** In particular, confirm:
- Does the user want the arcface scan on control backbone OR on spatial backbone? (They said "on the spatial-enabled baseline" but the existing P3.5 02 leader uses the control backbone. Ask.)
- Has the user completed the stress-aug visual check? If yes, adjust B1.
- Is Teams-family reweighting (C2) OK given the "no facedancer-specific weighting" rule?

## Files to Know

| File | Why it matters for planning |
|------|-----------------------------|
| `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_RETRO_SCORE_RESULTS_2026-04-22.md` | Full retro-score results + slot-07 decision + the apples-to-apples P3 vs P3.5 comparison. Start here. |
| `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_5_EXPERIMENT_PLAN_2026-04-22.md` | Packet-3.5 plan — shows the yaml pattern you will mirror for packet-4. §5 contains the slot-07 rule that we just evaluated. |
| `experiments/phase2_round13/R13_RLP35_02_arcface_m015.yaml` | THE base yaml for packet-4. It's the current leader's config. All packet-4 yamls should be diffs off this one. |
| `experiments/phase2_round13/R13_RLP35_04_stability_lambda_003.yaml` | Reference for the stability_lambda block. |
| `experiments/phase2_round13/R13_RLP3_04_FT_proper_spatial.yaml` | Reference for the spatial augmentation block (if any A3/spatial-backbone experiment is chosen). |
| `arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml` | Inventory of the newer proper Teams data. For C1 you'll need to identify a held-out split. |
| `trainer/trainer.py:411-430` | Where trainer reads the value_composite block. Verify any schedule-gated lambda work against this. |
| `trainer/trainer.py:2815-3170` | OOD monitoring — where teams_tpr / other_tpr / stability are computed. |
| `scripts/launch/launch_experiment.sh` | Launcher. CLI pattern: `./launch_experiment.sh -y <WANDB_PROJECT> <REGION> <CONFIG_YAML>` (confirm exact flags; prior session used a slightly different signature for the retro launcher). |
| `VERSION` | Currently 1.3.195. If any packet-4 experiment needs code changes, it will auto-bump to 1.3.196+. |

## What's NOT in scope for packet 4

- **Committing this handoff or any packet-4 artifacts before user says so.** Prior-session user feedback flagged unsolicited commits as unwanted.
- **Launching jobs before user approves the plan.**
- **Slot-07 stack_top3** — decision was DON'T FIRE. If the planner revisits this, they must engage with the packet-3.5 §5 rule and explain why the rule threshold should change, not just pick top-3 by eyeballing.
- **Facedancer-specific data work.** User declined.
- **us-central1 placements.** User reserved.

## Resume Instructions (for the packet-4 planner)

1. Read this document fully.
2. Read `R13_RELAUNCH_PACKET3_RETRO_SCORE_RESULTS_2026-04-22.md` fully — especially the P3.5 slot 02 profile at the bottom and the Teams-gap discussion.
3. Confirm current slot occupancy with `gcloud ai custom-jobs list --region=<R> --project=train-cvit2 --filter="state=JOB_STATE_RUNNING" --format="table(name.segment(-1):label=JOB_ID,displayName,state)"` for each of asia-southeast1, us-east1, us-west4, and europe-west4. Flag any drift from what this handoff recorded.
4. Ask the user the 3 refinement questions in the "Suggested 6-experiment plan" section above before drafting yamls.
5. Draft the packet-4 plan doc at `docs/relaunch_handoffs/R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md`.
6. Draft the 6 yamls at `experiments/phase2_round13/R13_RLP4_{01..06}_<shortname>.yaml`.
7. Draft a "jobs to cancel" list — specific job IDs + regions + the W&B run URL for each, so the user can sanity-check that the peak is persisted before cancel.
8. Present the plan, the cancel list, and the yamls to the user for approval. **STOP THERE.**
9. After user approval: execute the cancels, launch the 6 jobs, then start a live-monitoring handoff for the overnight run.

## Setup Required

- GCP: project `train-cvit2`, account `roee@dtectvision.ai` (already authed in this session).
- W&B: entity `dtect-vision`, project `enhanced-aug-test`. Key hardcoded in `scripts/launch/launch_experiment.sh:89`.
- Image: `1.3.195` if no code changes; otherwise build a new tag via `./dev.sh build-prod -y`.
- Local: bash 3.2 (macOS) — any monitoring scripts must avoid `declare -A`.
- Known-good regions for A100: `us-east1`, `asia-southeast1`, `us-west4`, `europe-west4`, `us-central1`. User forbade `us-central1` for this packet.
- Known-bad regions (as of today): `us-east4`, `us-west1`, `asia-east1`, `asia-northeast3`, `asia-northeast1`.

## Warnings

- **Do not commit `HANDOFF.md`, `VERSION`, this handoff, the draft plan, or the draft yamls unless the user explicitly asks.** User has flagged auto-commits as unwanted multiple times.
- **Do not launch jobs before user approval.** Surface the 6-experiment plan; wait.
- **Do not cancel any asia-southeast1 job without first verifying its best checkpoint is persisted to GCS.** Some P3 jobs are 22 hours in and losing a peak would be painful.
- **If B1 (late stability_lambda) is in the final plan**: verify trainer supports a step-gated lambda schedule before committing. If not supported and the user still wants the experiment, scope the code change explicitly.
- **If C1 (proper_visomaster_teams in OOD) is in the final plan**: identify a HELD-OUT slice of the inventory. The model trained on 297 of 342 total proper_teams samples — evaluating on those would be useless. The inventory has a `lockbox_ratio: 0.2` field; use the lockbox split.
- **The current leader's peak is a narrow window (step 5800-7000, ~1500 steps wide out of 9000).** If your packet-4 yamls change training length or cadence, the peak may move and the comparison to 0.7442 will not be clean.
- **The P3.5 slot 03 (arcface_m=0.20) is blocked on worst_pool_fpr with max_fpr≈0.055.** This data point is telling us something about the arcface-margin ceiling. Factor that into your A2 (m=0.175) expected outcome.

## When you're done — hand back to the user

Post a message like:

```
Packet-4 plan ready for review.

6 experiments chosen (rationale in the plan doc):
  1. <name>  | <region>  | <one-line hypothesis>
  2. <name>  | <region>  | <hypothesis>
  3. <name>  | <region>  | <hypothesis>
  4. <name>  | <region>  | <hypothesis>
  5. <name>  | <region>  | <hypothesis>
  6. <name>  | <region>  | <hypothesis>

To free the slots, we propose cancelling the following jobs (peaks verified persisted):
  <job id> <region>  <display name>  <best_vc was X at step Y>
  ... (up to 6 cancels in asia-southeast1)

Plan doc:  docs/relaunch_handoffs/R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md
Yamls:     experiments/phase2_round13/R13_RLP4_{01..06}_*.yaml

Shall I proceed with the cancels and launches?
```
