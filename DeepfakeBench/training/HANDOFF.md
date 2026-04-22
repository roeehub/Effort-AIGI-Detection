# Handoff: R13 Packet 3 + Packet 3.5 — 13 runs in flight, 2 regions

**Generated**: 2026-04-22 ~13:30 UTC
**Branch**: `teams-relaunch-root-2026-04-17`
**Status**: In Progress.
- **Packet 3**: 7 RLP3 FT runs RUNNING in `asia-southeast1` (started ~00:15 UTC, ETA ~07:00-08:30 UTC).
- **Packet 3.5**: 6 RLP35 FT runs PENDING/RUNNING in `us-east1` (started ~13:25 UTC).
- **Image**: `us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:1.3.192` (Cloud Build `f03237d4`, SUCCEEDED 11:25 UTC).

## What changed this session

Launched a parallel "packet 3.5" wave in `us-east1` to attack the `value_composite` ceiling that packet 3 trajectory was telegraphing. Four disabled training knobs + two metric-definition issues were turned on/relaxed in a single new wave of 6 single-lever experiments.

## Completed

- [x] **Diagnosed the value_composite ceiling** via a deep read of `trainer/trainer.py:112-294`, `trainer/mixins/arcface.py`, `trainer/mixins/stability.py`, and every RLP yaml. Found four disabled knobs (`arcface_m=0.0`, `stability_lambda=0.0`, `label_smoothing=0.0`, `anneal_steps=15000 vs total=10000`) and two hardcoded metric-definition caps (`max` jitter aggregator, `0.02/0.04` FPR gate).
- [x] **User-approved two metric-definition relaxations**: FPR gate 2%/4% → 3%/5%, and jitter stability aggregator `max` → `p95`. Applied globally to packet 3.5 and to packet-3 retro-score.
- [x] **Landed code plumbing** (commit `477b00b`): trainer.py reads new `value_composite` config block (`target_mean_fpr`, `max_pool_fpr`, `stability_jitter_stat`) with defaults matching legacy packet-3 values. arcface.py warns when `anneal_steps > 1.1 * total_training_steps`.
- [x] **Added unit tests** (`tests/test_value_composite_config.py`, `tests/test_arcface_anneal_warning.py`) — run inside image via `./dev.sh test`.
- [x] **Wrote 7 packet-3.5 yamls** (`experiments/phase2_round13/R13_RLP35_0{1..7}_*.yaml`) from the RLP3_02 baseline. Every slot inherits the global `value_composite` block + `anneal_steps: 8000`. Each single-lever slot adds ONE delta.
- [x] **Bumped VERSION 1.3.191 → 1.3.192** (commit `e1cd23e`). Image built and pushed by Cloud Build `f03237d4-d6ba-4ce5-8f89-e72803c8e9a3`.
- [x] **Launched slots 01–06** in `us-east1`. All jobs submitted successfully; state transitioning from PENDING to RUNNING.
- [x] **Authored packet 3.5 plan doc** at `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_5_EXPERIMENT_PLAN_2026-04-22.md`.

## Not Yet Done

- [ ] **Wait for packet 3.5 slots 01–06 to enter RUNNING state** (currently PENDING) then hit step 3-4k ~2h later.
- [ ] **Review early signals** from slots 01–06 around step 3-4k (~15:30-16:00 UTC). Watch for ArcFace-collapse on slot 03 (m=0.20): abort if `val_holdout/auc < 0.95` in first 1k steps.
- [ ] **Decide slot 07 launch**: if at least two of {healthy arcface slot, slot 04, slot 05} show ≥ +0.03 composite lift over baseline, tune slot 07's `arcface_m` to the best healthy margin and fire it. Yaml is committed but not launched.
- [ ] **Wait for packet 3 RLP3 runs to finish** in `asia-southeast1` (~07:00-08:30 UTC).
- [ ] **Retro-score packet 3 under packet-3.5 metric definition** via `rerun_validation.py` with image 1.3.192. Produces baseline `value_composite` numbers under `(0.03, 0.05, p95)` for fair comparison to packet 3.5.
- [ ] **Apply packet-3 §4.4 decision rules** once packet 3 finishes.
- [ ] **A2b retroactive eval** on packet-1/2 runs per packet-3 plan §4.3.
- [ ] **Produce packet 3.5 results doc** after all 7 slots complete.

## Failed Approaches (Don't Repeat)

*(carried forward from previous session)*

- **Parallel `gcloud ai custom-jobs cancel <id> &`** — harness denies shared-infra destructive operations in parallel. Enumerate IDs, get sign-off, cancel one at a time.
- **Pre-build VERSION commit** — `dev.sh build-prod -y` auto-bumps VERSION mid-build. Always commit the bump AFTER the build returns.
- **`sleep 900 && ...` scheduled follow-ups** — blocked by harness. Use `Monitor` pattern or `run_in_background` + notification.
- **Tests importing `from trainer.mixins import X`** — triggers `trainer/__init__.py` which imports `Trainer` → `dataset.dataloaders` → `torchdata` (may be missing locally). Tests run inside the image via `./dev.sh test`.

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| ArcFace progression = 0.10/0.15/0.20 (NOT 0.35) | User explicitly flagged that the LAION-DataComp backbone collapses with aggressive margins. Cautious range trades slower climb for safety; abort-on-collapse gate at `val_holdout/auc < 0.95`. |
| Single-lever deltas per slot 01–06 | Attribution over speed. Stacking left for slot 07 (contingent). |
| FPR gate relaxed from 2%/4% → 3%/5%, stability stat `max`→`p95` | User-approved. `p95` is already computed by `_aggregate_jitter_across_videos`; `max` was a fragile summary (one frame pinned stability to 0). |
| Config-gated, backcompat-safe code changes | Runs without the new config block default to legacy `(0.02, 0.04, max)`. No existing run's reproducibility broken. |
| Launched on 1.3.192, not 1.3.191 | Cleaner: W&B numbers during training already reflect new metric definition. Avoids a retro-score step for the packet 3.5 runs themselves. (Packet 3 still needs retro-score.) |
| us-east1, parallel with packet 3 in asia-southeast1 | User has quota. No regional interference. Cuts 6-8h of waiting. |
| Slot 07 contingent, not fired yet | Attribution needs slot 01–06 early signals first (~2h post-launch). |

## Current State

**Working**:
- All 7 RLP3 jobs `JOB_STATE_RUNNING` in `asia-southeast1` (packet 3).
- 6 RLP35 jobs `JOB_STATE_PENDING` in `us-east1` (packet 3.5, transitioning).
- 2 smokes + scratch baseline `JOB_STATE_SUCCEEDED` (last session).
- Image `1.3.192` live on GCS.
- Commits `477b00b` (packet 3.5 code + yamls + plan) and `e1cd23e` (VERSION bump) on `teams-relaunch-root-2026-04-17`. Not pushed.

**Broken**: Nothing known.

**Uncommitted Changes** (all pre-existing, untouched this session):
- `arena/*` auto-generated artifacts, older handoff docs, packet-1/2 yamls, and similar pre-existing noise. Safe to ignore for packet 3.5.

## Live Job IDs

### Packet 3 (asia-southeast1, started 2026-04-22 ~00:15 UTC)
```
SMOKE_us-east1          6522149387936727040   us-east1           SUCCEEDED
SMOKE_us-central1       2713966177662533632   us-central1        SUCCEEDED
RLP3_01_control         1977661603188834304   asia-southeast1    RUNNING
RLP3_02_main            3243173098479943680   asia-southeast1    RUNNING
RLP3_03_low_arcface     4506432793957367808   asia-southeast1    RUNNING
RLP3_04_spatial         7438839101328982016   asia-southeast1    RUNNING
RLP3_05_lowarc_spatial  5533253508997840896   asia-southeast1    RUNNING
RLP3_06_seedB           7854859116907331584   asia-southeast1    RUNNING
RLP3_07_lighting        8730809244430893056   asia-southeast1    RUNNING
SCRATCH_RLP1_08         6868236546978349056   asia-southeast1    SUCCEEDED
```

### Packet 3.5 (us-east1, started 2026-04-22 ~13:25 UTC, image 1.3.192)
```
RLP35_01_arcface_m010              775802554016595968    us-east1   PENDING
RLP35_02_arcface_m015              7062827633825808384   us-east1   PENDING
RLP35_03_arcface_m020              7297014814449074176   us-east1   PENDING
RLP35_04_stability_lambda_003      6183394253465452544   us-east1   PENDING
RLP35_05_label_smoothing_005       3877551244251758592   us-east1   PENDING
RLP35_06_family_rebalance_proper_up 5681348448129908736  us-east1   PENDING
(slot 07 — contingent, not launched)
```

## Files to Know (new since last session)

| File | Why |
|------|-----|
| `experiments/phase2_round13/R13_RLP35_0{1..7}_*.yaml` | Packet 3.5 slots. 01–06 launched. 07 contingent — tune `arcface_m` before firing. |
| `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_5_EXPERIMENT_PLAN_2026-04-22.md` | Plan of record for packet 3.5 (§3 global changes, §4 slate, §5 decision rules, §6 retro-score path). |
| `trainer/trainer.py` | New `value_composite` config block at ~line 410; jitter stat switch at ~line 3087. Lines in the codebase have shifted by ~30 from packet 3. |
| `trainer/mixins/arcface.py` | Anneal-mismatch warning added at `init_arcface()` after the existing info log. |
| `tests/test_value_composite_config.py` | Pytest covering gate + p95 stat behavior. Runs inside image. |
| `tests/test_arcface_anneal_warning.py` | Pytest confirming warning fires at mismatch. Runs inside image. |

## Resume Instructions

### Step 1 — Verify packet 3.5 made it to RUNNING

```bash
for id in 775802554016595968 7062827633825808384 7297014814449074176 6183394253465452544 3877551244251758592 5681348448129908736; do
  gcloud ai custom-jobs describe "$id" --region=us-east1 --project=train-cvit2 --format="value(state,displayName)"
done
```
All should be `JOB_STATE_RUNNING` within 5–10 min of launch. If any stays `PENDING` for >30 min, us-east1 quota may be maxed; check the Vertex AI Quotas page.

### Step 2 — First-eval health check (~step 500, ~30 min after job transitions to RUNNING)

Go to https://wandb.ai/dtect-vision/enhanced-aug-test, filter runs to `exp-R13_RLP35_*`. For each, confirm:
- `train/loss` is finite and decreasing
- `train/arcface_s` is present and rising (confirms `anneal_steps: 8000` is wired)
- `value_composite_stability_stat` is `"p95"` (confirms config block was read)
- `val_holdout/auc > 0.95` (abort threshold for arcface slots)

If slot 03 (m=0.20) hits `val_holdout/auc < 0.95` in first 1k steps:
```bash
gcloud ai custom-jobs cancel 7297014814449074176 --region=us-east1 --project=train-cvit2
```
(User-confirm first — harness denies parallel cancels.)

### Step 3 — Early-signal read for slot 07 decision (~step 3-4k, ~2h post-RUNNING)

From W&B, for slots 01–06 look at `summary/value_composite` trend. Rank by `Δvalue_composite` vs packet-3 baseline. The slot with the highest healthy arcface_m becomes the margin choice for slot 07.

### Step 4 — Fire slot 07

```bash
# Edit arcface_m in the yaml to the best-performing margin from slots 01-03
./launch_experiment.sh -y enhanced-aug-test us-east1 experiments/phase2_round13/R13_RLP35_07_stack_top3.yaml
```

### Step 5 — When packet 3 finishes (~07:00-08:30 UTC), retro-score under new metric

Use `rerun_validation.py` with image 1.3.192, overriding the yaml's `value_composite` block to `(0.03, 0.05, p95)`. Packet 3 runs currently log value_composite under legacy `(0.02, 0.04, max)` — the retro-score produces comparable numbers for the single-lever comparison.

### Step 6 — Apply §4.4 + produce results doc

Merge packet 3 retro-scored + packet 3.5 fresh into `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_5_RESULTS_*.md`. Rank single-lever slots 01–06 by Δ. Decide packet-4 priorities based on which levers moved the number.

## Setup Required

- **GCP auth**: `train-cvit2` project, `roee@dtectvision.ai` account (confirmed this session).
- **W&B**: project `enhanced-aug-test`, entity `dtect-vision`. Key hardcoded in `scripts/launch/launch_experiment.sh:89`.
- **Image for ANY new launch**: `1.3.192`. If you bump VERSION, commit the bump per the dev.sh auto-bump pattern.

## Edge Cases & Warnings

- **Slot 03 is the collapse probe.** `m=0.20` is the cautious upper bound; the user has prior evidence the backbone collapses above some margin threshold. Monitor closely. If it collapses and slot 02 (m=0.15) is healthy, packet 4 caps margin at 0.15.
- **Slot 07 is contingent.** Do NOT launch before reading slot 01–06 early signals. Tune `arcface_m` in the yaml first to match the best healthy margin from slots 01–03.
- **Retro-score is a separate step after packet 3 finishes.** The packet-3 W&B `value_composite` numbers are under LEGACY gate (0.02/0.04) and LEGACY stat (max) — don't compare them directly to packet-3.5 numbers without running retro-score.
- **Metric defaults in code are legacy-preserving.** `_vc_target_mean_fpr` defaults to `0.02`, `_vc_max_pool_fpr` to `0.04`, `_vc_stability_jitter_stat` to `"max"` when config block is absent. Every packet-3.5 yaml overrides via an explicit block.
- **Packet 3 `score_jitter_max/teams_ood_fake ≈ 1.0` observation** holds regardless of stat swap if `score_jitter_p95` is also near 1.0. Retro-score will reveal whether stability-term unpinning is real.
