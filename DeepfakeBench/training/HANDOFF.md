# Handoff: R13 Packet 3 + Packet 3.5 — 6 RLP35 slots relaunched on image 1.3.193 under correct new-metric config

**Generated**: 2026-04-22 12:43 UTC (superseding the 11:55 UTC snapshot)
**Branch**: `teams-relaunch-root-2026-04-17`
**Status**: In Progress — packet 3 unchanged (7 RUNNING in asia-southeast1). Packet 3.5 fully relaunched on image 1.3.193 after a critical `train_sweep.py` bug was found and fixed; 6 new RLP35 slots now PENDING, distributed across 4 regions.
**Next chat's first ask**: user will want a **status update** — run the commands in "Step 0" below to produce it.

## Goal

Push `value_composite` from packet-3's ~0.60 plateau toward the deployment target of **0.90–0.95** (with jitter=0.5 accepted as ceiling on the stability term). Packet 3.5 is a parallel wave of 6 single-lever FT experiments turning on disabled training knobs (`arcface_m=0`, `stability_lambda=0`, `label_smoothing=0`, `anneal_steps=15000 > 10000`) plus two user-approved metric-definition relaxations (`target_mean_fpr` 2%→3%, `max_pool_fpr` 4%→5%, stability jitter stat `max`→`p95`).

## What happened this session (2026-04-22 11:55 → 12:43 UTC)

1. **Found a critical config-propagation bug**. The prior session's image 1.3.192 had the correct trainer-side code (commit `477b00b`), and the packet-3.5 yamls had the correct `value_composite` block. But the slot-01 / slot-02 trainer init logs both printed LEGACY defaults: `value_composite config: target_mean_fpr=0.0200 max_pool_fpr=0.0400 stability_jitter_stat=max`. Root cause: `train_sweep.py` (around lines 171–274) explicitly copies nested config blocks from `single_cfg` into the effective `config` (because W&B flattens nested dicts), but never got an entry for `value_composite`. All 6 slots would have trained under LEGACY metric — defeating the primary purpose of packet 3.5.

2. **Patched `train_sweep.py`** with the missing copy-over (commit `872502c`). Same pattern as the existing explicit blocks for `dataset_methods`, `combined_paired`, `backbone`, `checkpointing`, flat stability keys, etc. Legacy-safe: runs without the block behave exactly as before.

3. **Built image `1.3.193`** via `./dev.sh build-prod -y` (Cloud Build `513d4eb1-de6f-4852-b822-dfea171e223d`, SUCCESS in 2m32s at 12:30 UTC). VERSION bumped `1.3.192 → 1.3.193`. Patch + VERSION committed together in `872502c`.

4. **Cancelled all 6 original RLP35 slots** in us-east1 (image 1.3.192). One was ~1h into training (slot 01); the other five were PENDING.

5. **Relaunched 6 RLP35 slots distributed across 4 regions** on image 1.3.193. `us-east4`, `us-west1`, `asia-east1`, `asia-northeast3`, `asia-northeast1` all rejected (either no `a2-highgpu-1g` machine type or A100 quota exhausted). Final landing: `us-east1` ×2, `asia-southeast1` ×1, `us-west4` ×1, `europe-west4` ×2.

6. **Audited for other nested-drop bugs.** None found. `family_weights` (slot 06) rides `dataset_methods` which is already copied. Flat stability keys + `label_smoothing` are in the explicit flat list. `arcface_m` / `anneal_steps` propagate via the top-level scalar path (confirmed in old slot 01 logs). `_aggregate_jitter_across_videos` emits `p95` (`trainer/trainer.py:103`). `apply_all_wandb_overrides` has no `value_composite` backup path — confirming the yaml block is the only wiring path.

## Completed

- [x] **Diagnosed the value_composite ceiling** — legacy session identified `arcface_m=0.0` hardcoded, `stability_lambda=0.0`, `label_smoothing=0.0`, `anneal_steps=15000 > 10000`, `stab_max=max(...)` hardcoded, FPR gate hardcoded `0.02 / 0.04`.
- [x] **Landed trainer-side code plumbing** (commit `477b00b` — prior session): `trainer/trainer.py` reads `value_composite` config block with legacy defaults; jitter-stat switch at call site; `trainer/mixins/arcface.py` anneal-mismatch warning.
- [x] **Wrote 7 packet-3.5 yamls** under `experiments/phase2_round13/R13_RLP35_0{1..7}_*.yaml` (prior session).
- [x] **Built image 1.3.192** (prior session, Cloud Build `f03237d4`, SUCCESS 11:25 UTC).
- [x] **Launched original slots 01–06 in us-east1** (prior session, 13:26 UTC). 2 of 6 transitioned to RUNNING before we caught the bug.
- [x] **Ran Step 1 verification and caught the config-propagation bug** (this session).
- [x] **Patched `train_sweep.py` with `value_composite` copy-over** (this session, commit `872502c`).
- [x] **Built image 1.3.193** (this session, Cloud Build `513d4eb1-de6f-4852-b822-dfea171e223d`, SUCCESS 12:30 UTC).
- [x] **Committed patch + VERSION bump** (this session, commit `872502c` on `teams-relaunch-root-2026-04-17`, not pushed).
- [x] **Cancelled 6 original RLP35 slots in us-east1** (this session).
- [x] **Relaunched 6 RLP35 slots on image 1.3.193 distributed across 4 regions** (this session, ~12:38–12:43 UTC).

## Not Yet Done

- [ ] **Wait for the 6 new RLP35 slots to transition PENDING → RUNNING.** All 6 are PENDING as of 12:43 UTC. Quota-distribution should be cleaner than the prior us-east1-only wave, but individual slots may still wait tens of minutes.
- [ ] **Verify new-metric wiring on each slot once RUNNING.** Trainer init log (~15 min post-RUNNING, after data-loader startup) must say `value_composite config: target_mean_fpr=0.0300 max_pool_fpr=0.0500 stability_jitter_stat=p95`. If ANY slot still logs legacy values, halt and re-audit before proceeding.
- [ ] **Watch slot 03 (m=0.20, asia-southeast1) for collapse** on transition to RUNNING. Abort if `val_holdout/auc < 0.95` in first 1k steps — user explicitly warned the LAION-DataComp backbone collapses under aggressive margins.
- [ ] **Read early signals (~step 3–4k, ~2h post-RUNNING)** and decide whether to fire slot 07 stacked best-of. New-metric composite is now reported live in W&B, so ranking is direct — no retro-score needed for packet 3.5.
- [ ] **Launch slot 07** — `R13_RLP35_07_stack_top3.yaml` authored but not launched. Before firing, edit `arcface_m` in the yaml to match the best-performing healthy margin from slots 01–03.
- [ ] **Packet 3 completion** — still 7 RUNNING past the 07:00–08:30 UTC ETA. Monitor.
- [ ] **Retro-score packet 3 under new metric definition** `(0.03, 0.05, p95)` via `rerun_validation.py` with image 1.3.193. Produces fair baseline for comparing packet 3 vs packet 3.5. `rerun_validation.py` uses plain `yaml.safe_load` + `config.update` and does NOT have the train_sweep.py bug — it will correctly pick up the `value_composite` block from any yaml passed in.
- [ ] **Apply packet-3 §4.4 decision rules** once packet 3 finishes.
- [ ] **Produce packet 3.5 results doc** after all 7 slots complete. Filename: `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_5_RESULTS_*.md`.

## Failed Approaches (Don't Repeat)

- **Proposing arcface_m = 0.35 as primary slot.** User pushed back; LAION-DataComp backbone collapses under aggressive margins. Resolved by cautious 0.10 / 0.15 / 0.20 progression. **The backbone is not replaceable** — do not exceed 0.20 without evidence of slot-03 health.
- **Testing `from trainer.mixins.arcface import ArcFaceMixin` locally** — fails because `trainer/__init__.py` triggers `trainer.trainer` which imports `dataset.dataloaders` → `torchdata` (not installed locally). Accept that trainer-facing tests only run inside the image via `./dev.sh test`.
- **Long `sleep N` blocking waits** — harness blocks `sleep 60` and similar patterns. Resolved with `until <condition>; do sleep 15; done` pattern for polling, or `run_in_background: true` for one-shot waits.
- **Parallel `gcloud ai custom-jobs cancel <id> &`** — denied as destructive parallel operation on shared infra. Always enumerate, confirm with user, cancel one-by-one.
- **Pre-build VERSION commit** — `dev.sh build-prod` auto-bumps VERSION mid-build. Always commit the bump AFTER the build returns SUCCESS.
- **Trusting that setting `value_composite` in the yaml was enough** — W&B flattens nested config dicts, so `train_sweep.py` must explicitly copy each nested block. The copy-over list is **mandatory** reading for anyone adding a new nested config block. Location: `train_sweep.py:171–282`.
- **Launching RLP35 slots in us-east4, us-west1, asia-east1, asia-northeast3, asia-northeast1** — all rejected. us-east4 and asia-east1 do not support `a2-highgpu-1g` (A100 40GB). us-west1, asia-northeast3, asia-northeast1 have zero A100 quota on `train-cvit2`. **Working regions** (as of 2026-04-22): us-east1, asia-southeast1, us-west4, europe-west4.

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| ArcFace margin progression 0.10 / 0.15 / 0.20 (not 0.35) | User warned backbone collapses with aggressive margins. |
| Relaxed FPR gate 2%/4% → 3%/5% | User-approved as deployment-defensible operating point. |
| Jitter stability aggregator `max` → `p95` | `_aggregate_jitter_across_videos` already computes p95; `max` is fragile. |
| `anneal_steps: 15000` → `8000` (yaml-only fix) | Lets ArcFace scale anneal complete by step 8k. |
| Config-gated code changes with legacy defaults | Zero backcompat break. |
| Single-lever slots 01–06, stacked slot 07 contingent | Attribution over speed. |
| **Cancel + relaunch on image 1.3.193 instead of retro-score-only** | The train_sweep.py bug meant in-training composite was wrong for all 6 slots; user explicitly authorized relaunch to get clean in-training signal rather than rely on post-hoc retro-score. |
| **Distribute across 4 regions** | us-east1 quota was already throttling the prior wave (4 of 6 pending 30+ min). Spreading across us-east1/asia-southeast1/us-west4/europe-west4 reduces single-region contention; user confirmed us-central1 is off-limits. |
| Slot 07 still contingent | Attribution window first (~step 3-4k); tune `arcface_m` to best healthy margin before firing. |

## Current State

**Working**:
- Packet 3 (asia-southeast1): 7 of 7 `JOB_STATE_RUNNING` on image 1.3.191 (past original ETA).
- Packet 3.5 relaunch (4 regions): 6 of 6 `JOB_STATE_PENDING` on image 1.3.193 as of 12:43 UTC.
- Image `1.3.193` live with the `train_sweep.py` `value_composite` fix.
- Commits on `teams-relaunch-root-2026-04-17`: `477b00b` (trainer-side code), `e1cd23e` (VERSION 1.3.191), `179c9e6` (prior HANDOFF), `5262e8a` (prior HANDOFF refresh), `872502c` (**train_sweep.py fix + VERSION 1.3.193, this session**). Not pushed.

**Blocked**:
- All 6 RLP35 slots are PENDING awaiting GPU allocation across 4 regions. Distribution reduces the risk of the "single-region quota throttle" seen in the prior us-east1-only wave.

**Broken**: Nothing known.

**Uncommitted Changes** (all pre-existing, not touched this session):
- `arena/*` auto-generated artifacts, `docs/relaunch_handoffs/NEW_DATA_LOADER*`, `RELAUNCH_UPGRADE_REVIEW*`, `WT_B_AND_NEW_DATA_READINESS*`. Untracked packet-1/2 yamls and various older handoff docs. Safe to ignore.

## Live Job IDs

### Packet 3 (asia-southeast1, started 2026-04-22 ~00:15 UTC, image 1.3.191) — unchanged
```
RLP3_01_control         1977661603188834304   RUNNING
RLP3_02_main            3243173098479943680   RUNNING
RLP3_03_low_arcface     4506432793957367808   RUNNING
RLP3_04_spatial         7438839101328982016   RUNNING
RLP3_05_lowarc_spatial  5533253508997840896   RUNNING
RLP3_06_seedB           7854859116907331584   RUNNING
RLP3_07_lighting        8730809244430893056   RUNNING
```

### Packet 3.5 (**RELAUNCHED** 2026-04-22 ~12:38–12:43 UTC, image 1.3.193, distributed across 4 regions)
```
SLOT                                 REGION            JOB ID                             STATE
RLP35_01_arcface_m010                us-east1          6776286107534360576                PENDING
RLP35_02_arcface_m015                us-east1          3065882964534493184                PENDING
RLP35_03_arcface_m020                asia-southeast1   7562969566058381312                PENDING   (watch for collapse on transition)
RLP35_04_stability_lambda_003        us-west4          154374731074633728                 PENDING
RLP35_05_label_smoothing_005         europe-west4      5892697191696302080                PENDING
RLP35_06_family_rebalance_proper_up  europe-west4      3309882805399322624                PENDING
(slot 07 stack_top3 — yaml authored, not launched; edit arcface_m before firing)
```

### Packet 3.5 CANCELLED slots (reference only; image 1.3.192, us-east1, wrong-metric)
```
CANCELLED 775802554016595968   RLP35_01_arcface_m010
CANCELLED 7062827633825808384  RLP35_02_arcface_m015
CANCELLED 7297014814449074176  RLP35_03_arcface_m020
CANCELLED 6183394253465452544  RLP35_04_stability_lambda_003
CANCELLED 3877551244251758592  RLP35_05_label_smoothing_005
CANCELLED 5681348448129908736  RLP35_06_family_rebalance_proper_up
```

## Files to Know

| File | Why It Matters |
|------|----------------|
| `HANDOFF.md` | This file. Next session's status-update context. |
| `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_5_EXPERIMENT_PLAN_2026-04-22.md` | Plan of record for packet 3.5 (§3 global changes, §4 slate, §5 decision rules, §6 retro-score path). Still authoritative; the train_sweep.py bug does not change any packet-3.5 experiment design. |
| `experiments/phase2_round13/R13_RLP35_0{1..7}_*.yaml` | Seven packet-3.5 yamls. 01–06 launched on image 1.3.193. 07 contingent. |
| `train_sweep.py:276–282` | The new `value_composite` explicit copy-over. Next person adding a nested config block should add a sibling here. |
| `trainer/trainer.py:414–429, 3085–3104` | Trainer-side value_composite read + wandb logging. Unchanged from `477b00b`. |
| `rerun_validation.py:210–215` | Plain `yaml.safe_load` + `config.update`. Does NOT have the train_sweep.py bug. Used for retro-scoring packet 3 checkpoints. |
| `VERSION` | `1.3.193` — must match the image tag for any new launches. |
| `scripts/launch/launch_experiment.sh` | Reads VERSION, submits Vertex job to a region passed as arg. |

## Code Context

### The train_sweep.py patch (commit `872502c`)

```python
# Apply value_composite config directly (nested dict — W&B flattens; must copy).
# Keys: target_mean_fpr, max_pool_fpr, stability_jitter_stat. Trainer falls
# back to legacy (0.02 / 0.04 / "max") when absent, so this is legacy-safe.
if 'value_composite' in single_cfg:
    config['value_composite'] = single_cfg['value_composite']
    print(f"  ✅ Applied value_composite: {single_cfg['value_composite']}")
    logger.info(f"  Applied value_composite: {single_cfg['value_composite']}")
```

### Packet 3.5 global config block (every yaml)
```yaml
value_composite:
  target_mean_fpr: 0.03         # was 0.02 hardcoded
  max_pool_fpr: 0.05            # was 0.04 hardcoded
  stability_jitter_stat: "p95"  # was "max" hardcoded
anneal_steps: 8000              # was 15000 (fix: anneal now completes inside 10k training)
```

### value_composite arithmetic for the 0.90 target (unchanged)
```
With stability = 0.5 (user-accepted ceiling):
  composite = 0.6·teams_tpr + 0.3·other_tpr + 0.1·0.5   (active_weight=1.0)
To hit 0.90: 0.6·teams_tpr + 0.3·other_tpr = 0.85
  If both TPRs equal:   TPR ≈ 0.944 at (3%-mean, 5%-max) FPR gate
To hit 0.95:            TPR ≈ 1.00
```

### Per-slot single-variable delta (unchanged from packet 3.5 plan)
| Slot | Delta | Baseline value |
|------|-------|----------------|
| 01 | `arcface_m: 0.10` | 0.0 |
| 02 | `arcface_m: 0.15` | 0.0 |
| 03 | `arcface_m: 0.20` | 0.0 (collapse risk — watch closely) |
| 04 | `stability_lambda: 0.03` (+ `noise_std=0.02, crop_jitter=0.03`) | 0.0 |
| 05 | `label_smoothing: 0.05` | 0.0 |
| 06 | `family_weights`: proper_clean/teams_fake 1.0→1.5, realpool/external_real 2.5→3.5, df40_fake 0.15→0.10 | as-baseline |
| 07 (contingent) | stacks winners from 01–06; edit `arcface_m` before firing | — |

## Resume Instructions

### Step 0 — Status update (run these first; user will ask)

```bash
cd /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training

# Packet 3 state (asia-southeast1)
echo "=== packet 3 (asia-southeast1) ==="
for id in 1977661603188834304 3243173098479943680 4506432793957367808 7438839101328982016 5533253508997840896 7854859116907331584 8730809244430893056; do
  gcloud ai custom-jobs describe "$id" --region=asia-southeast1 --project=train-cvit2 --format="value(state,displayName)"
done

# Packet 3.5 state (across 4 regions)
echo "=== packet 3.5 us-east1 ==="
for id in 6776286107534360576 3065882964534493184; do
  gcloud ai custom-jobs describe "$id" --region=us-east1 --project=train-cvit2 --format="value(state,displayName)"
done
echo "=== packet 3.5 asia-southeast1 ==="
gcloud ai custom-jobs describe 7562969566058381312 --region=asia-southeast1 --project=train-cvit2 --format="value(state,displayName)"
echo "=== packet 3.5 us-west4 ==="
gcloud ai custom-jobs describe 154374731074633728 --region=us-west4 --project=train-cvit2 --format="value(state,displayName)"
echo "=== packet 3.5 europe-west4 ==="
for id in 5892697191696302080 3309882805399322624; do
  gcloud ai custom-jobs describe "$id" --region=europe-west4 --project=train-cvit2 --format="value(state,displayName)"
done
```
Compare against the "Live Job IDs" snapshot at 12:43 UTC: packet 3 was 7 RUNNING, packet 3.5 was 6 PENDING (all freshly relaunched).

### Step 1 — Verify packet 3.5 config wiring (once any slot transitions to RUNNING, ~15 min in)

```bash
# Per-slot (swap job ID + region):
gcloud logging read "resource.type=ml_job AND resource.labels.job_id=<JOB_ID> AND textPayload:\"value_composite config\"" \
  --project=train-cvit2 --limit=1 --format="value(textPayload)"
```
**Expected (must match)**: `value_composite config: target_mean_fpr=0.0300 max_pool_fpr=0.0500 stability_jitter_stat=p95`.
If ANY slot logs legacy values (`0.0200 / 0.0400 / max`): halt, open the trainer __init__ block around `trainer/trainer.py:414`, and re-audit. The image is 1.3.193 and the train_sweep.py patch is at `train_sweep.py:276–282`; the most likely cause of a regression would be an incorrectly-loaded yaml or a stale image tag pinned somewhere.

### Step 2 — ArcFace collapse check for slot 03 (critical, m=0.20, asia-southeast1)

```bash
# Check val_holdout/auc on W&B: https://wandb.ai/dtect-vision/enhanced-aug-test (filter to exp-R13_RLP35_03_arcface_m020*)
gcloud logging read "resource.type=ml_job AND resource.labels.job_id=7562969566058381312 AND textPayload:\"val_holdout/auc\"" \
  --project=train-cvit2 --limit=10 --format="value(textPayload)"
```
Abort rule: `val_holdout/auc < 0.95` in first 1k steps → confirm with user, then cancel slot 03:
```bash
gcloud ai custom-jobs cancel 7562969566058381312 --region=asia-southeast1 --project=train-cvit2
```

### Step 3 — Early signals read (~step 3-4k, ~2h post-RUNNING transition)

On W&B, for each of slots 01–06 read `summary/value_composite` directly — numbers are now under the NEW metric `(0.03, 0.05, p95)` since the train_sweep.py fix. Rank slots by `Δvalue_composite` vs packet-3 RLP3_02 retro-scored under the same new metric (see Step 5).

### Step 4 — Decide + fire slot 07

If ≥ 2 of {highest-healthy arcface slot, slot 04, slot 05} show `Δvalue_composite ≥ +0.03`:
```bash
# Edit arcface_m to best-healthy margin from 01-03 (default in yaml is 0.15):
# $EDITOR experiments/phase2_round13/R13_RLP35_07_stack_top3.yaml
./launch_experiment.sh -y enhanced-aug-test <REGION> experiments/phase2_round13/R13_RLP35_07_stack_top3.yaml
```
Region: pick whichever of us-east1/asia-southeast1/us-west4/europe-west4 has capacity.

### Step 5 — Retro-score packet 3 under new metric (once packet 3 finishes)

Use `rerun_validation.py` with image 1.3.193, injecting the new `value_composite` config block to score packet-3 best checkpoints under `(0.03, 0.05, p95)`. `rerun_validation.py` uses plain `yaml.safe_load` + `config.update` (not the train_sweep.py path) and does NOT have the copy-over bug — passing a yaml with the `value_composite` block works directly.

### Step 6 — Apply §4.4 decision rules + write results doc

Write `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_5_RESULTS_<date>.md`. Rank slots 01–06 by `Δvalue_composite` (now directly comparable since packet 3.5 runs under the new metric). Feed into packet 4 design (third seed, enhanced-clean dose-matched, margin > 0.20 if safe).

## Setup Required

- **GCP auth**: `train-cvit2` project, `roee@dtectvision.ai` account.
- **W&B**: project `enhanced-aug-test`, entity `dtect-vision`. Key hardcoded in `scripts/launch/launch_experiment.sh:89`.
- **Image for ANY new launch**: `1.3.193`. If you bump VERSION via `./dev.sh build-prod -y`, commit the bump after SUCCESS.
- **Laptop has `torch 2.11.0` but not `torchdata`** — trainer tests fail locally, succeed inside image.

## Edge Cases & Error Handling

- **Slot 03 collapses (m=0.20)** → cancel it; safe margin upper bound is in [0.15, 0.20). Cap packet-4 margin at 0.15.
- **Any slot still PENDING >1h after relaunch** → check regional quota in GCP console; ask user before relocating. Regions `us-east4`, `us-west1`, `asia-east1`, `asia-northeast3`, `asia-northeast1` are known-bad as of this session (no a2-highgpu-1g or no A100 quota). Known-good: `us-east1`, `asia-southeast1`, `us-west4`, `europe-west4`.
- **Slot log doesn't show `value_composite config: target_mean_fpr=0.0300 ...` line** → the new code wasn't picked up. Check:
  - `cat VERSION` is `1.3.193`.
  - The image pinned by the launch script matches (grep `IMAGE_URI` in slot logs for `1.3.193`).
  - `git log --oneline train_sweep.py | head -3` shows `872502c` at top.
  - The trainer did actually reach the `value_composite config:` print statement (not an earlier crash).
- **Packet 3 runs past 07:00–08:30 UTC ETA** (happened). Not a crisis — wait. Possibilities: early-stopping patience hasn't triggered; real run time underestimated.
- **score_jitter_p95 also ≈ 1.0** (not just `max`) → stability term stays pinned despite stat swap. Retro-score will reveal; if so, only slot 04 (stability_lambda) can unpin stability via training.

## Warnings

- **The ArcFace backbone collapses at aggressive margins.** Do NOT raise `arcface_m > 0.20` in this wave. Packet 5 may push higher only with slot-03 health confirmed.
- **Packet 3 absolute numbers were logged under LEGACY gate** (0.02 / 0.04) and LEGACY stat (max). DO NOT compare packet-3 W&B `value_composite` directly to packet-3.5 W&B `value_composite` — run the retro-score first.
- **Packet 3.5 RELAUNCHED slots now log under NEW gate** (`0.03 / 0.05 / p95`) because image 1.3.193 has the train_sweep.py fix. The old cancelled slots (image 1.3.192) logged under LEGACY by accident of the bug; their numbers are irrelevant.
- **Every packet-3.5 yaml explicitly sets the `value_composite` block + `anneal_steps: 8000`.** Do not remove these on a duplicate-slot or new-slot yaml unless you deliberately want legacy behavior.
- **Slot 07 is NOT launched.** Do not fire until early signals from slots 01–06 are in. Edit `arcface_m` in its yaml first (default is 0.15).
- **Data recipe is LOCKED to RLP3_02** — unenhanced proper-data only, no hints, no enhanced proper. Packets 1 and 2 closed these decisions.
- **Packet 3.5 launches are on image 1.3.193** but `trainer/__init__.py` still imports `Trainer` transitively → local `./dev.sh test` (inside Docker) is the only way to run the new unit tests.
- **`dev.sh build-prod -y` auto-bumps VERSION.** Running it twice from a clean state would produce 1.3.194. Commit the bump after each build.
- **Nested config blocks must be added to the explicit copy-over list in `train_sweep.py:171–282`.** W&B flattens nested dicts, so the copy-over is the only propagation path. Next time you add a new nested block to a yaml, add a sibling entry to the copy list.
