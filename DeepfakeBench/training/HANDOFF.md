# Handoff: R13 Packet 3 + Packet 3.5 — 13 runs in flight, awaiting GPU quota + signals

**Generated**: 2026-04-22 11:55 UTC
**Branch**: `teams-relaunch-root-2026-04-17`
**Status**: In Progress — 7 RLP3 still RUNNING in `asia-southeast1` (past initial ETA); 2 of 6 RLP35 RUNNING in `us-east1`, 4 still PENDING on us-east1 GPU quota.
**Next chat's first ask**: user will want a **status update** — run the commands in "Step 0" below to produce it.

## Goal

Push `value_composite` from packet-3's ~0.60 plateau toward the user's deployment target of **0.90–0.95** (with jitter=0.5 accepted as ceiling on the stability term). Packet 3.5 is a parallel wave of 6 single-lever FT experiments designed to turn on training knobs that were disabled (`arcface_m=0`, `stability_lambda=0`, `label_smoothing=0`, `anneal_steps` overrunning training horizon) plus two user-approved metric-definition relaxations (`target_mean_fpr` 2%→3%, `max_pool_fpr` 4%→5%, stability jitter stat `max`→`p95`).

## Completed

- [x] **Diagnosed the value_composite ceiling** — `arcface_m=0.0` hardcoded in every RLP yaml (never varied), `stability_lambda=0.0` (mixin disabled), `label_smoothing=0.0`, `anneal_steps=15000 > total_training_steps=10000` so ArcFace scale never reaches `s=12`, `stab_max = max(...)` hardcoded at `trainer/trainer.py:3068` pinning stability to 0 when `score_jitter_max/teams_ood_fake ≈ 1.0`.
- [x] **Landed code plumbing** (commit `477b00b`):
  - `trainer/trainer.py` Trainer.__init__ reads new `value_composite` config block (`target_mean_fpr`, `max_pool_fpr`, `stability_jitter_stat`). Defaults match legacy packet-3 values (0.02 / 0.04 / "max") so runs without the block are bit-identical.
  - `trainer/trainer.py` jitter-stat switch at the value_composite call site (`stab_candidates = [... get(stat_key, 0.0) ...]`).
  - `trainer/mixins/arcface.py` warns at init when `anneal_steps > 1.1 * total_training_steps`.
- [x] **Added unit tests**: `tests/test_value_composite_config.py`, `tests/test_arcface_anneal_warning.py`. Run inside image via `./dev.sh test` (local run fails on `torchdata` — same as every existing trainer test).
- [x] **Wrote 7 packet-3.5 yamls** under `experiments/phase2_round13/R13_RLP35_0{1..7}_*.yaml`. All inherit global `value_composite` block + `anneal_steps: 8000`. Each single-lever slot applies ONE delta to RLP3_02 baseline. YAML deltas verified via Python yaml load.
- [x] **Built image 1.3.192** via `./dev.sh build-prod -y` (Cloud Build `f03237d4-d6ba-4ce5-8f89-e72803c8e9a3`, SUCCESS 11:25 UTC). VERSION bumped 1.3.191 → 1.3.192, committed (`e1cd23e`).
- [x] **Launched slots 01–06** in `us-east1` at ~13:26 UTC. All 6 submitted successfully. Slots 01 and 02 have transitioned to RUNNING (confirmed via gcloud logging: `arcface_m: 0.1` and `anneal_steps: 8000` applied correctly on slot 01).
- [x] **Wrote packet-3.5 plan doc** at `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_5_EXPERIMENT_PLAN_2026-04-22.md`.
- [x] **Committed updated HANDOFF.md** (`179c9e6`).

## Not Yet Done

- [ ] **Wait for slots 03–06 to leave PENDING** in `us-east1`. At 11:54 UTC, 4 of 6 still PENDING ~30 min after submission. us-east1 GPU quota is throttling the queue.
- [ ] **Confirm slot 01 config wiring at trainer __init__** — look for log line `value_composite config: target_mean_fpr=0.0300 max_pool_fpr=0.0500 stability_jitter_stat=p95`. Did not appear in first 15 min of logs (trainer init happens after data-loader setup, which is long with cloud data).
- [ ] **Watch slot 03 (m=0.20) for collapse** — abort if `val_holdout/auc < 0.95` in first 1k steps. User explicitly warned the LAION-DataComp backbone collapses under aggressive margins.
- [ ] **Read early signals (~step 3-4k)** and decide whether to fire slot 07 stacked best-of. ETA ~2h after slot-specific RUNNING transition.
- [ ] **Launch slot 07** — `R13_RLP35_07_stack_top3.yaml` is authored but not launched. Before firing, edit `arcface_m` in the yaml to match the best-performing healthy margin from slots 01–03.
- [ ] **Packet 3 completion** — still 7 RUNNING past the 07:00–08:30 UTC ETA. Monitor, wait.
- [ ] **Retro-score packet 3 under new metric definition** `(0.03, 0.05, p95)` via `rerun_validation.py` with image 1.3.192. Produces fair baseline for comparing packet 3 vs packet 3.5 single-lever deltas.
- [ ] **Apply packet-3 §4.4 decision rules** + A2b retroactive per plan §4.3 once packet 3 finishes.
- [ ] **Produce packet 3.5 results doc** after all 7 slots complete. Filename: `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_5_RESULTS_*.md`.

## Failed Approaches (Don't Repeat)

- **Proposing arcface_m = 0.35 as primary slot.** User pushed back: past experiments collapsed this LAION-DataComp backbone at aggressive margins. Resolved by cautious 0.10 / 0.15 / 0.20 progression. **The backbone is not replaceable** — do not exceed 0.20 without evidence of slot-03 health.
- **Launching `./launch_experiment.sh` for each slot sequentially inline** — works but hits the tool timeout. Resolved by batching slots 02–06 in a single for-loop after slot 01 confirmed the command syntax.
- **Testing `from trainer.mixins import ArcFaceMixin` locally** — fails because `trainer/__init__.py` triggers `trainer.trainer` which imports `dataset.dataloaders` → `torchdata` (not installed locally). Even `from trainer.mixins.arcface import ArcFaceMixin` fails because Python loads parent packages. **Resolution**: accept that trainer-facing tests only run inside the image via `./dev.sh test`. Matches the pattern of every existing `tests/test_trainer_*.py`.
- **Long `sleep N` blocking waits** — harness blocks `sleep 60` and similar patterns. Resolved with `until <condition>; do sleep 15; done` pattern for polling, or `run_in_background: true` for one-shot waits.
- **Parallel `gcloud ai custom-jobs cancel <id> &`** (carried forward from previous session) — denied as destructive parallel operation on shared infra. Always enumerate, confirm, cancel one-by-one.
- **Pre-build VERSION commit** (carried forward) — `dev.sh build-prod` auto-bumps VERSION mid-build. Always commit the bump AFTER the build returns SUCCESS.

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| ArcFace margin progression 0.10 / 0.15 / 0.20 (not 0.35) | User warned backbone collapses with aggressive margins; cautious exploration preserves options. |
| Relaxed FPR gate 2%/4% → 3%/5% | User-approved as deployment-defensible operating point; retro-applies cleanly via config. |
| Jitter stability aggregator `max` → `p95` | `_aggregate_jitter_across_videos` already computes p95; `max` is fragile (one frame kills the term). User agreed. |
| `anneal_steps: 15000` → `8000` (yaml-only fix) | Lets ArcFace scale anneal complete by step 8k, leaving 2k at full `s=12` plateau. |
| Config-gated code changes with legacy defaults | Zero backcompat break; packet-3 runs without the new block are bit-identical to pre-3.5 trainers. |
| Single-lever slots 01–06, stacked slot 07 contingent | Attribution over speed; can't stack confidently until we know which single levers work. |
| Launch on 1.3.192 directly (built before launch, not after) | W&B numbers during training already reflect new metric definition; no retro-score needed for packet 3.5 runs themselves (packet 3 still needs one). |
| us-east1 for packet 3.5, in parallel with packet 3 (asia-southeast1) | User confirmed us-east1 quota (8 slots). No regional interference. Saves 6–8h of waiting. |
| Slot 07 authored but NOT launched | Attribution window first (~step 3-4k); tune `arcface_m` to best healthy margin before firing. |

## Current State

**Working**:
- Packet 3 (asia-southeast1): 7 of 7 `JOB_STATE_RUNNING` (past initial ETA of 07:00–08:30 UTC; may be running longer than expected).
- Packet 3.5 (us-east1): 2 of 6 `JOB_STATE_RUNNING` (slots 01 `m=0.10`, 02 `m=0.15`). Slot 01 logs confirm `arcface_m: 0.1` + `anneal_steps: 8000` applied. Image 1.3.192 in use.
- Smokes + scratch from prior session: `SUCCEEDED`.
- Image `1.3.192` live.
- Commits `477b00b`, `e1cd23e`, `179c9e6` on `teams-relaunch-root-2026-04-17`. Not pushed.

**Blocked**:
- Packet 3.5 slots 03–06 stuck `JOB_STATE_PENDING` (us-east1 GPU allocation). Not a bug; quota will release as other workloads free GPUs. If stuck >30 min past slot-02 transition, consider checking Vertex AI quotas page.

**Broken**: Nothing known.

**Uncommitted Changes** (all pre-existing, untouched this session):
- `arena/*` auto-generated artifacts, `docs/relaunch_handoffs/NEW_DATA_LOADER*`, `RELAUNCH_UPGRADE_REVIEW*`, `WT_B_AND_NEW_DATA_READINESS*` modified pre-session. Untracked packet-1/2 yamls, various older handoff docs. Safe to ignore.

## Live Job IDs

### Packet 3 (asia-southeast1, started 2026-04-22 ~00:15 UTC, image 1.3.191)
```
RLP3_01_control         1977661603188834304   RUNNING
RLP3_02_main            3243173098479943680   RUNNING
RLP3_03_low_arcface     4506432793957367808   RUNNING
RLP3_04_spatial         7438839101328982016   RUNNING
RLP3_05_lowarc_spatial  5533253508997840896   RUNNING
RLP3_06_seedB           7854859116907331584   RUNNING
RLP3_07_lighting        8730809244430893056   RUNNING
SCRATCH_RLP1_08         6868236546978349056   SUCCEEDED  (last session)
```

### Packet 3.5 (us-east1, started 2026-04-22 ~13:26 UTC, image 1.3.192)
```
RLP35_01_arcface_m010               775802554016595968    RUNNING
RLP35_02_arcface_m015               7062827633825808384   RUNNING
RLP35_03_arcface_m020               7297014814449074176   PENDING (watch for collapse on transition)
RLP35_04_stability_lambda_003       6183394253465452544   PENDING
RLP35_05_label_smoothing_005        3877551244251758592   PENDING
RLP35_06_family_rebalance_proper_up 5681348448129908736   PENDING
(slot 07 stack_top3 — yaml authored, not launched)
```

## Files to Know

| File | Why It Matters |
|------|----------------|
| `HANDOFF.md` | This file. Next session's status-update context. |
| `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_5_EXPERIMENT_PLAN_2026-04-22.md` | Plan of record for packet 3.5 (§3 global changes, §4 slate, §5 decision rules, §6 retro-score path). |
| `experiments/phase2_round13/R13_RLP35_0{1..7}_*.yaml` | Seven packet-3.5 yamls. 01–06 launched. 07 contingent (edit `arcface_m` before firing). |
| `experiments/phase2_round13/R13_RLP3_02_FT_proper_main.yaml` | Baseline every packet-3.5 slot copies from. |
| `trainer/trainer.py` | C1–C3 edits: value_composite config block at ~line 410; jitter stat switch at ~line 3087; gate passthrough to `_compute_value_composite`. Line numbers shifted ~30 from pre-packet-3.5. |
| `trainer/mixins/arcface.py` | C4 edit: anneal-mismatch warning at end of `init_arcface()`. |
| `tests/test_value_composite_config.py` | Unit test for gate + p95 stat. Runs inside image. |
| `tests/test_arcface_anneal_warning.py` | Unit test for the new warning. Runs inside image. |
| `VERSION` | `1.3.192` — must match the image tag for any new launches. |
| `scripts/launch/launch_experiment.sh` | Reads VERSION, submits Vertex job. `./launch_experiment.sh` at repo root is a thin wrapper. |
| `dev.sh` | `build-prod -y` auto-bumps VERSION patch+1 and triggers Cloud Build. Commit AFTER. |

## Code Context

### Packet 3.5 global config block in every yaml
```yaml
value_composite:
  target_mean_fpr: 0.03         # was 0.02 hardcoded
  max_pool_fpr: 0.05            # was 0.04 hardcoded
  stability_jitter_stat: "p95"  # was "max" hardcoded at trainer.py:3068
anneal_steps: 8000              # was 15000 (fix: anneal now completes inside 10k training)
```

### trainer.py config reads (the edit)
```python
# In Trainer.__init__, after OOD monitoring cadence block (~line 410):
vc_cfg = (self.config.get("value_composite") or {})
self._vc_target_mean_fpr = float(vc_cfg.get("target_mean_fpr", 0.02))
self._vc_max_pool_fpr = float(vc_cfg.get("max_pool_fpr", 0.04))
self._vc_stability_jitter_stat = str(vc_cfg.get("stability_jitter_stat", "max"))

# At value_composite call site (~trainer.py:3087):
stat_key = self._vc_stability_jitter_stat
stab_candidates = [
    ood_jitter_summary_cache.get(k, {}).get(stat_key, 0.0)
    for k in _VALUE_COMPOSITE_STABILITY_JITTER_METHODS
]
stab_max = max(stab_candidates) if stab_candidates else 0.0
vc = _compute_value_composite(
    ..., stability_jitter_max=float(stab_max),
    target_mean_fpr=self._vc_target_mean_fpr,
    max_pool_fpr=self._vc_max_pool_fpr,
)
```

### value_composite arithmetic for the 0.90 target
```
With stability = 0.5 (user-accepted ceiling):
  composite = 0.6·teams_tpr + 0.3·other_tpr + 0.1·0.5   (active_weight=1.0)
To hit 0.90: 0.6·teams_tpr + 0.3·other_tpr = 0.85
  If both TPRs equal:   TPR ≈ 0.944 at (3%-mean, 5%-max) FPR gate
To hit 0.95:            TPR ≈ 1.00
```

### Per-slot single-variable delta (from RLP3_02 baseline)
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

# Packet 3 state
echo "=== packet 3 (asia-southeast1) ==="
for id in 1977661603188834304 3243173098479943680 4506432793957367808 7438839101328982016 5533253508997840896 7854859116907331584 8730809244430893056; do
  gcloud ai custom-jobs describe "$id" --region=asia-southeast1 --project=train-cvit2 --format="value(state,displayName)"
done

# Packet 3.5 state
echo "=== packet 3.5 (us-east1) ==="
for id in 775802554016595968 7062827633825808384 7297014814449074176 6183394253465452544 3877551244251758592 5681348448129908736; do
  gcloud ai custom-jobs describe "$id" --region=us-east1 --project=train-cvit2 --format="value(state,displayName)"
done
```
Compare against the "Live Job IDs" section snapshot (11:54 UTC): at that time, packet 3 = 7 RUNNING, packet 3.5 = 2 RUNNING + 4 PENDING, slot 07 not launched.

### Step 1 — Verify packet 3.5 config wiring (once all 6 RUNNING)

```bash
# Confirm the new value_composite config block is being read (log appears after trainer __init__, which is ~15 min post-RUNNING)
gcloud logging read "resource.type=ml_job AND resource.labels.job_id=775802554016595968 AND textPayload:\"value_composite config\"" --project=train-cvit2 --limit=3 --format="value(textPayload)"
```
Expected: `value_composite config: target_mean_fpr=0.0300 max_pool_fpr=0.0500 stability_jitter_stat=p95`.
If absent after >30 min of RUNNING: the config block isn't being read — check trainer.py line ~410 edit survived the image build, potentially re-check commit `477b00b`.

### Step 2 — ArcFace collapse check for slot 03 (critical, m=0.20)

```bash
# Check val_holdout/auc on W&B: https://wandb.ai/dtect-vision/enhanced-aug-test (filter to exp-R13_RLP35_03_arcface_m020*)
# OR via logs:
gcloud logging read "resource.type=ml_job AND resource.labels.job_id=7297014814449074176 AND textPayload:\"val_holdout/auc\"" --project=train-cvit2 --limit=10 --format="value(textPayload)"
```
Abort rule: `val_holdout/auc < 0.95` in first 1k steps → cancel slot 03:
```bash
gcloud ai custom-jobs cancel 7297014814449074176 --region=us-east1 --project=train-cvit2
```
(Confirm with user first. Harness denies parallel cancels.)

### Step 3 — Early signals read (~step 3-4k, ~2h post-RUNNING transition)

On W&B, for each of slots 01–06 read `summary/value_composite` and `value_composite_stability` (under the new `p95` stat). Rank slots by `Δvalue_composite` vs packet-3 RLP3_02 baseline retro-scored under the same metric definition (see Step 5).

### Step 4 — Decide + fire slot 07

If ≥ 2 of {highest-healthy arcface slot, slot 04, slot 05} show `Δvalue_composite ≥ +0.03`:
```bash
# Edit arcface_m to best-healthy margin from 01-03 (default in yaml is 0.15):
# $EDITOR experiments/phase2_round13/R13_RLP35_07_stack_top3.yaml  # set arcface_m
./launch_experiment.sh -y enhanced-aug-test us-east1 experiments/phase2_round13/R13_RLP35_07_stack_top3.yaml
```

### Step 5 — Retro-score packet 3 under new metric (once packet 3 finishes)

Use `rerun_validation.py` with image 1.3.192, injecting the new `value_composite` config block to score packet-3 best checkpoints under `(0.03, 0.05, p95)`. Baseline numbers enable honest packet-3 vs packet-3.5 comparison. This separates "training-knob moved the composite" from "metric-definition change moved the composite."

### Step 6 — Apply §4.4 decision rules + write results doc

Write `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_5_RESULTS_<date>.md`. Rank slots 01–06 by `Δvalue_composite`. Feed into packet 4 design (third seed, enhanced-clean dose-matched, margin > 0.20 if safe).

## Setup Required

- **GCP auth**: `train-cvit2` project, `roee@dtectvision.ai` account.
- **W&B**: project `enhanced-aug-test`, entity `dtect-vision`. Key hardcoded in `scripts/launch/launch_experiment.sh:89`.
- **Image for ANY new launch**: `1.3.192`. If you bump VERSION via `./dev.sh build-prod -y`, commit the bump after.
- **Laptop has `torch 2.11.0` but not `torchdata`** — trainer tests fail locally, succeed inside image.

## Edge Cases & Error Handling

- **Slot 03 collapses (m=0.20)** → cancel it; safe margin upper bound is in [0.15, 0.20). Cap packet-4 margin at 0.15.
- **All 4 PENDING slots stuck >1h** → us-east1 quota maxed. Options: (a) wait, (b) ask user to cancel an older run, (c) relaunch some slots in us-central1 (one smoke succeeded there).
- **Slot 01 log doesn't show `value_composite config:` line** → new code not picked up by image. Check commit `477b00b` landed before `dev.sh build-prod` was invoked (git log should show it before `e1cd23e`).
- **Packet 3 runs past 07:00–08:30 UTC ETA** (already happened by this handoff). Two possibilities: (a) early-stopping patience (10 evals × 500 steps = 5000 steps of no improvement) hasn't triggered, runs hitting 15 epochs; (b) real run time was underestimated. Not a crisis — just wait.
- **score_jitter_p95 also ≈ 1.0** (not just `max`) → stability term stays pinned despite stat swap. Retro-score will reveal this; if so, only slot 04 (stability_lambda) can unpin stability via training.

## Warnings

- **The ArcFace backbone collapses at aggressive margins.** User has prior evidence; do NOT raise `arcface_m > 0.20` in this wave. Packet 5 may push higher only with slot-03 health confirmed.
- **Packet 3 absolute numbers were logged under LEGACY gate** (0.02 / 0.04) and LEGACY stat (max). DO NOT compare packet-3 W&B `value_composite` directly to packet-3.5 W&B `value_composite` — run the retro-score first.
- **Every packet-3.5 yaml explicitly sets the `value_composite` block + `anneal_steps: 8000`.** Do not remove these on a duplicate-slot or new-slot yaml unless you deliberately want legacy behavior.
- **Slot 07 is NOT launched.** Do not fire it until early signals from slots 01–06 are in. Edit `arcface_m` in its yaml first (default is 0.15).
- **Data recipe is LOCKED to RLP3_02** — unenhanced proper-data only, no hints, no enhanced proper. Packets 1 and 2 closed these decisions; do not reopen in packet 3.5.
- **Packet 3.5 launches are on image 1.3.192** but `trainer/__init__.py` still imports `Trainer` transitively → local `./dev.sh test` (inside Docker) is the only way to run the new unit tests.
- **`dev.sh build-prod -y` auto-bumps VERSION.** Running it twice from a clean state would produce 1.3.193. Commit the bump after each build.
