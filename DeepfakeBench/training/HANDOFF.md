# Handoff: R13 Packet 3 + Packet 3.5 — 13 runs live, W&B anomalies flagged by user

**Generated**: 2026-04-22 14:18 UTC
**Branch**: `teams-relaunch-root-2026-04-17` (commits `477b00b` → `bcf4c61`, not pushed)
**Status**: In Progress — all 13 runs RUNNING, packet 3.5 config-bug fixed this session, user has flagged "strange stuff on W&B" for the next agent to investigate.

## ⚠️ First thing next session

The user ended this session saying: *"I'm seeing some strange stuff on W&B"*. They'll want to talk to you about what they're seeing before anything else. **Ask them what they're looking at before you recommend anything.** W&B URL: `https://wandb.ai/dtect-vision/enhanced-aug-test` (project for packet 3.5). Packet 3 is also under `dtect-vision` entity.

Once their W&B question is resolved, fall back to the watchlist below.

## Goal

Push `value_composite` from packet-3's ~0.60 plateau toward the **0.90–0.95** deployment target. Packet 3.5 is a 6-slot single-lever FT wave that turns on 4 disabled training knobs (`arcface_m`, `stability_lambda`, `label_smoothing`, fixed `anneal_steps`) plus a user-approved metric-definition change (`target_mean_fpr` 2%→3%, `max_pool_fpr` 4%→5%, jitter stat `max`→`p95`).

## Completed this session (2026-04-22 11:55 → 14:18 UTC)

- [x] **Diagnosed a hidden config-propagation bug in `train_sweep.py`**. The prior session's slot 01/02 init logs printed LEGACY metric values (`0.0200 / 0.0400 / max`) despite yamls declaring new values — `train_sweep.py` explicitly copies nested config blocks from `single_cfg` into the effective `config` because W&B flattens nested dicts, and the `value_composite` block was never added to that copy list.
- [x] **Patched `train_sweep.py:276-282`** with the missing copy-over. Legacy-safe; trainer defaults unchanged. Commit `872502c`.
- [x] **Built image `1.3.193`** (Cloud Build `513d4eb1-de6f-4852-b822-dfea171e223d`, SUCCESS in 2m32s — most layers cached). VERSION bumped `1.3.192 → 1.3.193`, committed in `872502c`.
- [x] **Cancelled 6 original RLP35 slots** in us-east1 (image 1.3.192). User explicitly confirmed before the cancels fired.
- [x] **Relaunched 6 RLP35 slots on image 1.3.193** distributed across 4 regions, plus a 5th when slot 06 was stuck PENDING in europe-west4 and user requested relocation to us-central1.
- [x] **Verified all 6 new slots log the NEW metric**: `value_composite config: target_mean_fpr=0.0300 max_pool_fpr=0.0500 stability_jitter_stat=p95`. Confirmed via `gcloud logging read` against each job ID.
- [x] **Committed HANDOFF.md refreshes** (`2cdb804`, `bcf4c61`) to keep session-to-session state in sync. ← User noted this was more than asked; next agent should default to updating only when explicitly requested or when the plan contains an update task.

## Not Yet Done

- [ ] **Investigate user's W&B observations** — opened at end of session, no details captured yet. Ask them for specifics (slot, metric, what looks wrong) before acting.
- [ ] **Slot 03 collapse watch** — `arcface_m=0.20`, asia-southeast1. Transitioned to RUNNING at 13:34 UTC. Abort if `val_holdout/auc < 0.95` in first 1k steps. First validation eval is ~500 steps in, so the first meaningful read is ~30-45 min after RUNNING. Confirm with user before cancelling.
- [ ] **Early signals read ~step 3–4k (~16:45–17:30 UTC for slot 01, later for others)**. Rank 01–06 by `Δvalue_composite` vs RLP3_02 baseline. Because packet 3.5 now runs under the NEW metric directly, no retro-score is needed for this ranking (unlike packet 3 below).
- [ ] **Slot 07 decision** — `R13_RLP35_07_stack_top3.yaml` authored but not launched. Fire only if ≥2 of {highest-healthy arcface slot, slot 04, slot 05} show `Δvalue_composite ≥ +0.03`. **Before firing**: edit `arcface_m` in the yaml to match the best-healthy margin from 01/02/03 (default is 0.15).
- [ ] **Packet 3 completion** — 7/7 still RUNNING past the original 07:00–08:30 UTC ETA. Not stuck (no errors; likely just longer than estimated).
- [ ] **Retro-score packet 3 top-3 under new metric `(0.03, 0.05, p95)`** via `rerun_validation.py` with image 1.3.193. `rerun_validation.py` uses plain `yaml.safe_load` + `config.update` and does NOT have the train_sweep.py bug — passing a yaml with the `value_composite` block works directly.
- [ ] **Produce packet 3.5 results doc** after all 7 slots (01–06 + 07 if fired) complete: `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_5_RESULTS_<date>.md`.

## Failed Approaches (Don't Repeat)

- **Trusting that yaml keys would propagate**. The nested `value_composite` block was present in every RLP35 yaml but silently dropped by `train_sweep.py` because W&B flattens nested dicts. Resolution: every nested config block needs an explicit copy-over in `train_sweep.py:171-282`. Lesson: when adding a new nested block, grep `train_sweep.py` for the existing copy pattern (`dataset_methods`, `combined_paired`, `backbone`, `checkpointing`, `group_dro_params`, etc.) and add a sibling entry.
- **Launching RLP35 in `us-east4`, `us-west1`, `asia-east1`, `asia-northeast3`, `asia-northeast1`**. All rejected.
  - `us-east4`, `asia-east1`: `ERROR: (gcloud.ai.custom-jobs.create) INVALID_ARGUMENT: Machine type "a2-highgpu-1g" is not supported.`
  - `us-west1`, `asia-northeast3`, `asia-northeast1`: `RESOURCE_EXHAUSTED: The following quota metrics exceed quota limits: aiplatform.googleapis.com/custom_model_training_nvidia_a100_gpus`
  - **Working regions for A100 40GB**: `us-east1`, `asia-southeast1`, `us-west4`, `europe-west4`, `us-central1` (8-slot quota, shared with other team jobs).
- **Proposing `arcface_m = 0.35` as a slot**. User warned the LAION-DataComp backbone collapses under aggressive margins. Resolved by cautious 0.10 / 0.15 / 0.20 progression. **Do not exceed 0.20** unless slot 03 is healthy.
- **Running trainer unit tests locally**. `trainer/__init__.py` imports `Trainer` which transitively imports `torchdata` (not installed on laptop). Tests only run inside image via `./dev.sh test`.
- **Long `sleep N` blocking waits**. Harness blocks `sleep 60` and similar patterns when used alone. Use `until <condition>; do sleep 15; done` pattern for polling, or `run_in_background: true` for one-shot waits.
- **Parallel `gcloud ai custom-jobs cancel &`**. Harness denies destructive parallel ops on shared infra. Always enumerate, confirm with user, cancel sequentially.
- **Pre-build VERSION commit**. `dev.sh build-prod -y` auto-bumps VERSION mid-build. Commit AFTER Cloud Build returns SUCCESS (on failure, `dev.sh` reverts VERSION).
- **Using `$status` as a shell variable name for polling**. zsh has `status` as read-only. Use `build_state` or similar.

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Cancel + relaunch on 1.3.193 (not retro-score-only) | User explicitly wanted in-training composite to be correct rather than relying on post-hoc retro-score; relaunch was fast given the fix. |
| Distribute 6 slots across 4 regions | us-east1-only wave was quota-throttled (4/6 PENDING >30 min). Spread avoids single-region contention. |
| Slot 06 relocated to us-central1 late session | europe-west4 slot 06 sat PENDING ~80 min; user said us-central1 had 5/8 A100 slots free → relocate, cancel europe-west4 job. |
| ArcFace margin progression 0.10 / 0.15 / 0.20 | Backbone collapse risk; keep cautious. |
| Jitter stability aggregator `max` → `p95` | `_aggregate_jitter_across_videos` already computes p95; `max` is fragile (one frame kills term). |
| `anneal_steps: 15000 → 8000` | Let ArcFace anneal finish by step 8k of 10k training. |
| Single-lever slots + contingent stacked slot 07 | Attribution over speed. |

## Current State

**Working** (2026-04-22 14:18 UTC, all 13 jobs RUNNING):

Packet 3 — `asia-southeast1`, image `1.3.191`:
```
RLP3_01_control         1977661603188834304  RUNNING
RLP3_02_main            3243173098479943680  RUNNING
RLP3_03_low_arcface     4506432793957367808  RUNNING
RLP3_04_spatial         7438839101328982016  RUNNING
RLP3_05_lowarc_spatial  5533253508997840896  RUNNING
RLP3_06_seedB           7854859116907331584  RUNNING
RLP3_07_lighting        8730809244430893056  RUNNING
```

Packet 3.5 — image `1.3.193`, distributed across 5 regions, **all verified on NEW metric**:
```
slot 01 arcface_m=0.10       us-east1         6776286107534360576   RUNNING
slot 02 arcface_m=0.15       us-east1         3065882964534493184   RUNNING
slot 03 arcface_m=0.20       asia-southeast1  7562969566058381312   RUNNING  (collapse watch)
slot 04 stability_lambda=0.03 us-west4        154374731074633728    RUNNING
slot 05 label_smoothing=0.05 europe-west4     5892697191696302080   RUNNING
slot 06 family_rebalance     us-central1      9185612453914869760   RUNNING
(slot 07 stack_top3 — yaml authored, not launched)
```

Cancelled this session (reference only; all image 1.3.192 or stale):
```
CANCELLED  775802554016595968  us-east1       RLP35_01  (wrong metric — train_sweep bug)
CANCELLED  7062827633825808384 us-east1       RLP35_02  (wrong metric)
CANCELLED  7297014814449074176 us-east1       RLP35_03  (wrong metric)
CANCELLED  6183394253465452544 us-east1       RLP35_04  (wrong metric)
CANCELLED  3877551244251758592 us-east1       RLP35_05  (wrong metric)
CANCELLED  5681348448129908736 us-east1       RLP35_06  (wrong metric)
CANCELLED  3309882805399322624 europe-west4   RLP35_06  (stale PENDING → relocated to us-central1)
```

**Broken**: Nothing known. User has open W&B observations — treat as unknown, not confirmed-broken.

**Uncommitted Changes**: Only pre-existing files (arena yamls/reports, older handoff docs, packet 1/2/3 yamls). **Nothing from this session is uncommitted.** Session commits `872502c`, `2cdb804`, `bcf4c61` all landed on `teams-relaunch-root-2026-04-17`; not pushed.

## Files to Know

| File | Why It Matters |
|------|----------------|
| `HANDOFF.md` | This file. |
| `train_sweep.py:171-282` | The explicit nested-block copy-over list. New nested yaml blocks MUST be added here or they'll be silently dropped (W&B flattens). `value_composite` was just added at L276-282. |
| `trainer/trainer.py:414-429` | Trainer.__init__ reads `value_composite` config block with legacy defaults. Emits the `value_composite config:` INFO log used for the Step 1 verification. |
| `trainer/trainer.py:103` | `_aggregate_jitter_across_videos` emits `p95` (and `mean`, `max`, `spike_rate_0p3`). |
| `trainer/trainer.py:3085-3104` | `value_composite` call site — picks stat key, calls `_compute_value_composite(...)`, logs `value_composite_target_mean_fpr`/`value_composite_max_pool_fpr` to W&B. |
| `trainer/mixins/arcface.py` | Anneal-mismatch warning (commit `477b00b`). |
| `rerun_validation.py:210-215` | Plain `yaml.safe_load` + `config.update` — does NOT have the train_sweep.py bug. Use for retro-score. |
| `experiments/phase2_round13/R13_RLP35_0{1..7}_*.yaml` | 7 packet-3.5 slot yamls. 01-06 launched; 07 contingent. |
| `experiments/phase2_round13/R13_RLP3_02_FT_proper_main.yaml` | Baseline every RLP35 slot copies from. |
| `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_5_EXPERIMENT_PLAN_2026-04-22.md` | Plan of record for packet 3.5 (§5 decision rules, §6 retro-score path). |
| `VERSION` | `1.3.193` — must match image tag for any new launches. |
| `scripts/launch/launch_experiment.sh` | Signature: `./launch_experiment.sh [-y] <WANDB_PROJECT> [<REGION>] <PARAM_CONFIG>`. Reads VERSION directly; no env override needed for image tag. |
| `dev.sh` | `build-prod -y` auto-bumps VERSION patch+1 and submits Cloud Build synchronously; on fail, VERSION reverts. |

## Code Context

### The `train_sweep.py` patch (commit `872502c`)

```python
# train_sweep.py:276-282 (immediately after group_dro_params block)
# Apply value_composite config directly (nested dict — W&B flattens; must copy).
# Keys: target_mean_fpr, max_pool_fpr, stability_jitter_stat. Trainer falls
# back to legacy (0.02 / 0.04 / "max") when absent, so this is legacy-safe.
if 'value_composite' in single_cfg:
    config['value_composite'] = single_cfg['value_composite']
    print(f"  ✅ Applied value_composite: {single_cfg['value_composite']}")
    logger.info(f"  Applied value_composite: {single_cfg['value_composite']}")
```

### Every RLP35 yaml includes this block (required — defines the evaluation metric)

```yaml
value_composite:
  target_mean_fpr: 0.03         # was 0.02 hardcoded
  max_pool_fpr: 0.05            # was 0.04 hardcoded
  stability_jitter_stat: "p95"  # was "max" hardcoded
anneal_steps: 8000              # was 15000 (fixes ArcFace anneal finishing inside 10k training)
```

### Trainer-side read (unchanged from prior session, `trainer/trainer.py:415-418`)

```python
vc_cfg = (self.config.get("value_composite") or {})
self._vc_target_mean_fpr = float(vc_cfg.get("target_mean_fpr", 0.02))
self._vc_max_pool_fpr = float(vc_cfg.get("max_pool_fpr", 0.04))
self._vc_stability_jitter_stat = str(vc_cfg.get("stability_jitter_stat", "max"))
```

### value_composite arithmetic for the 0.90–0.95 target

```
composite = 0.6·teams_tpr + 0.3·other_tpr + 0.1·stability
With stability = 0.5 ceiling:
  to reach 0.90: 0.6·teams_tpr + 0.3·other_tpr = 0.85
    if TPRs equal: TPR ≈ 0.944 at (3%-mean, 5%-max) FPR gate
  to reach 0.95:  TPR ≈ 1.00
```

### Per-slot single-variable delta (from RLP3_02 baseline)

| Slot | Delta | Baseline |
|------|-------|----------|
| 01 | `arcface_m: 0.10` | 0.0 |
| 02 | `arcface_m: 0.15` | 0.0 |
| 03 | `arcface_m: 0.20` | 0.0 (collapse risk) |
| 04 | `stability_lambda: 0.03` (+ `noise_std=0.02`, `crop_jitter=0.03`) | 0.0 |
| 05 | `label_smoothing: 0.05` | 0.0 |
| 06 | `family_weights`: proper_clean/teams_fake 1.0→1.5, realpool/external_real 2.5→3.5, df40_fake 0.15→0.10 | as baseline |
| 07 (contingent) | stacks winners from 01–06; edit `arcface_m` before firing | — |

## Resume Instructions

1. **Read the warning at the top.** User will bring up W&B anomalies first. Don't guess — ask what they're seeing. Have the run URLs ready:
   - Packet 3.5: `https://wandb.ai/dtect-vision/enhanced-aug-test` (filter: `exp-R13_RLP35_*`)
   - Packet 3: same entity, run name filter `exp-R13_RLP3_*`

2. **Baseline status check** (you'll need these numbers when the user asks):
   ```bash
   cd /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training

   # Packet 3 (all should be RUNNING)
   for id in 1977661603188834304 3243173098479943680 4506432793957367808 7438839101328982016 5533253508997840896 7854859116907331584 8730809244430893056; do
     gcloud ai custom-jobs describe "$id" --region=asia-southeast1 --project=train-cvit2 --format="value(state,displayName)"
   done

   # Packet 3.5 (spread across 5 regions)
   gcloud ai custom-jobs describe 6776286107534360576 --region=us-east1        --project=train-cvit2 --format="value(state,displayName)"
   gcloud ai custom-jobs describe 3065882964534493184 --region=us-east1        --project=train-cvit2 --format="value(state,displayName)"
   gcloud ai custom-jobs describe 7562969566058381312 --region=asia-southeast1 --project=train-cvit2 --format="value(state,displayName)"
   gcloud ai custom-jobs describe 154374731074633728  --region=us-west4        --project=train-cvit2 --format="value(state,displayName)"
   gcloud ai custom-jobs describe 5892697191696302080 --region=europe-west4    --project=train-cvit2 --format="value(state,displayName)"
   gcloud ai custom-jobs describe 9185612453914869760 --region=us-central1     --project=train-cvit2 --format="value(state,displayName)"
   ```
   Expected: all 13 in `JOB_STATE_RUNNING`.

3. **Config-wiring verification** (do once per slot if you haven't seen the log yet, or to double-check a slot the user is worried about):
   ```bash
   gcloud logging read "resource.type=ml_job AND resource.labels.job_id=<JOB_ID> AND textPayload:\"value_composite config\"" \
     --project=train-cvit2 --limit=1 --format="value(textPayload)"
   ```
   Expected (all 6 RLP35 slots): `value_composite config: target_mean_fpr=0.0300 max_pool_fpr=0.0500 stability_jitter_stat=p95`.
   If any slot shows legacy `0.0200 / 0.0400 / max`, something broke — check image tag on the job (`gcloud ai custom-jobs describe ... --format="value(jobSpec.workerPoolSpecs[0].containerSpec.imageUri)"`) and `git log train_sweep.py`.

4. **Slot 03 collapse watch** (m=0.20, asia-southeast1):
   ```bash
   gcloud logging read "resource.type=ml_job AND resource.labels.job_id=7562969566058381312 AND textPayload:\"val_holdout/auc\"" \
     --project=train-cvit2 --limit=10 --format="value(textPayload)"
   ```
   Abort rule: `val_holdout/auc < 0.95` in first 1k steps. **Confirm with user before cancelling.** Cancel command: `gcloud ai custom-jobs cancel 7562969566058381312 --region=asia-southeast1 --project=train-cvit2`.

5. **Early signals read (~step 3–4k)**. Pull W&B summary for each slot; rank by `summary/value_composite`. Numbers are directly comparable across RLP35 slots (all on NEW metric) but NOT directly comparable to packet 3 (which is under LEGACY metric — retro-score needed, see step 7).

6. **Slot 07 decision**. If ≥2 of {highest-healthy arcface slot, slot 04, slot 05} show `Δvalue_composite ≥ +0.03` vs RLP3_02: edit `arcface_m` in `experiments/phase2_round13/R13_RLP35_07_stack_top3.yaml` to the best healthy margin, then:
   ```bash
   ./launch_experiment.sh -y enhanced-aug-test <REGION> experiments/phase2_round13/R13_RLP35_07_stack_top3.yaml
   ```
   Pick whichever of us-east1/asia-southeast1/us-west4/europe-west4/us-central1 has capacity.

7. **Retro-score packet 3** once those runs finish. Use `rerun_validation.py` with image 1.3.193 + a yaml containing the new `value_composite` block. This produces fair packet-3-vs-packet-3.5 comparison.

## Setup Required

- **GCP auth**: project `train-cvit2`, account `roee@dtectvision.ai`.
- **W&B**: entity `dtect-vision`, project `enhanced-aug-test`. Key hardcoded at `scripts/launch/launch_experiment.sh:89`.
- **Image for any new launch**: `1.3.193`. Do NOT relaunch anything on 1.3.192 — it has the `train_sweep.py` config-propagation bug.
- **Running trainer tests**: only inside image via `./dev.sh test` (laptop lacks `torchdata`).

## Edge Cases & Error Handling

- **Slot 03 collapses (m=0.20)** → cancel it; safe margin cap is in [0.15, 0.20). Packet 4 must not exceed 0.15 without further evidence.
- **A slot PENDING >1h** → check regional quota. Known-good regions: us-east1, asia-southeast1, us-west4, europe-west4, us-central1 (shared 8-slot quota). Known-bad as of today: us-east4, us-west1, asia-east1, asia-northeast3, asia-northeast1.
- **A slot logs legacy metric** → the image pinned to that job isn't 1.3.193. Check `gcloud ai custom-jobs describe ... --format="value(jobSpec.workerPoolSpecs[0].containerSpec.imageUri)"`.
- **Packet 3 runs past initial ETA** (already happening — 7/7 still RUNNING >6h past the 07:00–08:30 UTC estimate). Possibilities: early-stopping patience hasn't triggered; real run time underestimated. Not a crisis.
- **`score_jitter_p95 ≈ 1.0` too** (not just `max`) → stability term stays pinned despite stat swap. Only slot 04 (stability_lambda) can unpin stability via training. Retro-score will reveal.

## Warnings

- **User flagged W&B anomalies at end of session — ask them first.** Don't recommend action until you know what they're looking at.
- **Handoff-file updates**: user pushed back mid-session on me auto-updating `HANDOFF.md` across sub-steps. Default to updating only when the user explicitly asks or the approved plan contains an update task.
- **Cancels**: always enumerate + confirm with user + cancel sequentially. Harness denies parallel destructive ops, and the user wants explicit confirmation per cancel batch.
- **ArcFace margin ceiling**: do NOT launch `arcface_m > 0.20` in this wave. Backbone is LAION-DataComp and collapses under aggressive margins (user has prior evidence).
- **Metric comparability**: packet 3 is LEGACY-metric W&B; packet 3.5 is NEW-metric W&B. Direct composite comparison is invalid until packet 3 is retro-scored.
- **`dev.sh build-prod -y` auto-bumps VERSION.** Commit the bump AFTER Cloud Build SUCCESS (on failure, VERSION reverts and an un-reverted commit would be wrong).
- **Nested config blocks in yamls are silently dropped by `train_sweep.py`** unless explicitly added to the copy list at L171-282. Every time a new nested block appears, add a sibling entry.
- **Data recipe is LOCKED to RLP3_02**: unenhanced proper-data, no hints, no enhanced proper. Packets 1–2 closed these decisions. Do not reopen in packet 3.5.
- **Slot 07 is NOT launched.** Do not fire until early signals from 01–06 are in.
