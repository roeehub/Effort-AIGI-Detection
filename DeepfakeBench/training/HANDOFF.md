# Handoff: Day 3 evening — P13_FROM_SCRATCH code committed, image building, ready to launch

**Generated**: 2026-04-28 evening CEST
**Branch**: `teams-relaunch-root-2026-04-17`
**Status**: All anti-shortcut code committed (`cab2909`). Cloud Build running for image **1.3.224** (started ~17:30 CEST, takes ~12-15 min). Local 200-step micro-smoke skipped per user (this Mac can't run trainer end-to-end without GCS+GPU). Substitute is **close monitoring of first 500 steps of the live launch.** User authorization for the ~$150 launch already given as part of plan v6 approval.

## Goal

Push the Effort detector to the **triple-axis Day-4 α gate** by training **R13_P13_FROM_SCRATCH** for 18,000 steps from the OpenCLIP ViT-B/16 + datacomp_xl_s13b_b90k base (no FT history), with the in_proj-SVD lever working (post commit 2feea58) and three new anti-shortcut interventions layered on:

1. **Anchor-aware loss term** — training-time hinge penalty pushing prob_fake on the false-flag anchor pool toward 0.10.
2. **Pipeline-randomization aug** — symmetric-across-labels JPEG/downscale/chroma/YUV/gamma stack.
3. **Face scale-jitter** [0.75, 1.25] — counters face-pixel-area label leak.

**Triple-axis target (Day 4):**
- 90/5: viso ≥ 90 AND deeplive ≥ 90 AND teams_fake_all ≥ 90 AND modern_v2 FPR ≤ 5 AND teams_real_all_dev FPR ≤ 7
- clean_eval_v1: recall ≥ 80 (deployment-honest, ~50 frames)
- shortcut_probe_v1: max-min Δprob ≤ 0.15 across same-face-different-pipeline pairs

Sprint deadline: Day 4 (2026-04-29). One extension day authorized (Day 5, 2026-04-30) for FT-from-P8A fallback if scratch doesn't clear all three axes.

## Completed today (Day 3, 2026-04-28 evening)

- [x] **Verified periodic_saves bug fix** at `trainer/trainer.py:2326-2340`. Already committed in `c7dc828` (the prior session bundled the fix). Wrote `tests/test_periodic_saves_resolution.py` — **12 tests pass** including the canonical p13 step_list `[2000,4000,6000,9000,12000,15000,18000]`.
- [x] **Implemented anchor-aware loss** at `loss/anchor_aware_penalty.py`. Reads cached anchor pool via `analysis.teams_pool_rescore.cache_anchor_pools_locally`; per-step hinge penalty `weight * max(0, mean(prob_fake) - target)^2`. **Wired at `trainer/trainer.py:1485` next to `compute_stability_loss`**; logs as `train/loss/anchor_aware`. **8 tests pass** including gradient flow.
- [x] **Implemented pipeline-randomization aug** at `data/augmentations/pipeline_randomization.py`. Five sub-augs (gamma → chroma blur → YUV roundtrip → downscale-upscale → JPEG roundtrip), each with own probability; gated overall by `p_real` / `p_fake`. Defensive `np.nan_to_num` + `np.clip(0,255)` after each. **Wired into `QualityTargetedFamilyRouter.__call__` after `_maybe_apply_teams_sim`**; reads label from `meta` dict. Yaml block under `augmentation.pipeline_randomization`. **10 tests pass.**
- [x] **Implemented face scale-jitter** at `data/augmentations/face_scale_jitter.py`. Module-level config (set once at trainer init); applied in collate-fn loop **before** the canonical 224×224 resize at both `data/batching/df40_paired.py:362` and `data/sources/combined_paired.py:3455`. **6 tests pass.**
- [x] **Authored** `experiments/phase2_round13/R13_P13_FROM_SCRATCH.yaml`. Cloned from `experiments/phase2_round9/R9_C_teams_scratch.yaml`. Diffs from R9_C: `stability_lambda=0`, `label_smoothing=0`, `total_training_steps=18000`, ArcFace `s=10→18` over 18000, `apply_svd_to_in_proj=true` (now actually working), `early_stopping_patience=50`, `visomaster_fake` family weight 4.0 (was 2.0), all three new interventions enabled, `periodic_saves` with the bug-fixed step_list. **Yaml parses cleanly via `yaml.safe_load`.**
- [x] **Committed as `cab2909`** — 12 files, +1297 lines, all 36 tests passing post-commit. Bundled in two pre-session-uncommitted edits in `pipelines.py` (webcam_harden scaffolding, off by default) and `combined_paired.py` (path_exclude_contains plumbing) so the working tree matches HEAD post-commit.
- [x] **Triggered Cloud Build** via `./dev.sh build-prod -y` (in background). VERSION already auto-bumped to **1.3.224**. Build log at `/tmp/build_prod_2026-04-28.log`. Expected to land at `us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:1.3.224`.

## Not yet done (resume here)

- [ ] **Verify Cloud Build succeeded.** `tail -50 /tmp/build_prod_2026-04-28.log` — look for `✅ Build successful!`. If failed, **VERSION will have been auto-reverted** to 1.3.223; investigate before retrying. The build packages the working tree (uncommitted dirty files and all), but our committed cab2909 is a clean reference for what's intended.
- [ ] **Launch P13_FROM_SCRATCH on Vertex.** Recipe:
  ```bash
  cd /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training
  ./scripts/launch/launch_experiment.sh -y phase2r13-experiments us-east1 \
      experiments/phase2_round13/R13_P13_FROM_SCRATCH.yaml
  ```
  - **W&B project**: `phase2r13-experiments` (matches recent R13 runs).
  - **Region**: us-east1 first (us-multi-region buckets, ~3.4 it/s expected per `project_gcs_region_locality`). Fallback `us-west4` then `us-central1` if PENDING > 30 min — see CLAUDE.md "Region capacity" rule.
  - **Cost**: ~$150 for 18h on A100. **Don't cancel without explicit user authorization** (see `feedback_no_cancelling_vertex_jobs`).
- [ ] **Monitor first 500 steps closely** (substitute for the skipped local micro-smoke). Specific things to watch in W&B (`https://wandb.ai/dtect-vision/phase2r13-experiments/`):
  1. **No NaN/Inf**. The trainer raises `RuntimeError` at step N if `unscaled_loss` is non-finite (`trainer/trainer.py:1502`); job will fail-fast. Webcam_harden produced NaN at step 5683 in P11 (~$30 wasted) — pipeline-random has the same NaN-risk profile (chroma blur + YUV roundtrip can produce out-of-range values). Defensive `np.nan_to_num + np.clip` is in place; verify it holds.
  2. **`train/loss/anchor_aware` logs every step.** Initially the random model gives ~0.5 prob_fake on everything → expect anchor_aware ≈ 5 * (0.5 - 0.10)² = 0.80 in early steps, decaying as the model learns. If it stays flat at 0 throughout, the cache failed to load (check trainer init log: `AnchorAwarePenalty ENABLED` vs `DISABLED`).
  3. **`train/loss/stability` should log 0** (we set `stability_lambda=0` per R95 finding). If it logs non-zero, the override didn't take.
  4. **`train/loss/overall` decreasing** by step ~200-500. Scratch is slow to start; minor wiggles fine, monotonic increase is a fail.
  5. **`anchor/anchor_mean` at first validation (step 500)** — random init should be ~0.5; if it's already ~0.10, the cache is misconfigured.
  6. **First periodic save fires at step 2000** — search W&B logs for `periodic_save triggered at step=2000`. If silent, the bug fix didn't survive the build (rebuild + relaunch).
  7. **Augmentation init log**: `PipelineRandomization ENABLED p_real=0.55 p_fake=0.45` and `FaceScaleJitter ENABLED: scale_limit=0.250` should both appear at trainer startup. If either says DISABLED, the yaml override didn't load.
- [ ] **End-of-Day-3 RESULTS + LOG entry** once launch is RUNNING (not PENDING). Don't write entries before then; PENDING > 30 min means region switch, not progress.
- [ ] **Wednesday daytime parallel work** (no GPU contention with the live training):
  - Track H.1 — `analysis/modern_v2_audit_2026-04-29/` modern_v2 filter audit (~1.5h CPU). Plan §3.3 step 3.5. May change Day-4 verdict retroactively if filter is over-pruning.
  - Track H.2 — ArcFace identity-purity audit (~3h CPU, **n_jobs=1 only** per `feedback_sklearn_njobs`).
  - clean_eval_v1 + shortcut_probe_v1 frame URI lists (~2h). Source: `analysis/lockbox_tagging/full_tags_2026-04-27.parquet`. Plan §3.3 step 3.6.

## Failed Approaches (don't repeat)

This handoff inherits all sections from the prior `HANDOFF.md` snapshot in commit history (specifically items 1-6 from the earlier Day-3 handoff). New additions tonight:

### 7. ❌ Trying to commit only my hunks via `git add -p` non-interactively

The pre-session-uncommitted webcam_harden block in `pipelines.py` and `path_exclude_contains` plumbing in `combined_paired.py` were entangled with my anti-shortcut changes. Surgical staging needs interactive `git add -p`. **What I did instead**: bundled the pre-existing dirty content into commit `cab2909` with a clear note in the commit message about what was bundled. Acceptable because both the webcam_harden scaffolding (off by default) and path_exclude_contains plumbing are anti-shortcut-adjacent. **Lesson for future**: don't let pre-session dirty content sit uncommitted — it forces these awkward bundling decisions.

### 8. ❌ Hoping pipelines.py/combined_paired.py changes would land via build alone

`./dev.sh build-prod` packages the **working tree**, not the **commit**. So Cloud Build will get my changes regardless of commit status. But a future agent cherry-picking commits would have an incomplete cab2909 if I'd skipped staging those two files. **What I did**: included them in the commit anyway (per #7). **Lesson**: always commit the full set of files needed for a feature, even if part of it is messy.

## Key Decisions (tonight)

| Decision | Rationale |
|---|---|
| Pivot from FT-from-P8A (Plan v5) to from-scratch (Plan v6) | R95 final report (March 2026) explicitly recommended scratch; project deferred for FT velocity, now saturated as predicted. 0/85 ensembles passing → 0.647 AUC ceiling on FT lineage. |
| 18K steps over 12K or 24K | Split-the-difference between R9_C's 12K baseline and R95's recommended 20K. ~$150 commitment. |
| Visomaster oversample 4.0 | HEAVY's 3.0 was insufficient; HEAVY_DEEPLIVE's 6.0 broke pair structure. 4.0 is the mid-bet. |
| Symmetric pipeline-random across labels (p_real=0.55, p_fake=0.45) | Asymmetric aug just inverts the shortcut. Tiny p_real/p_fake gap acknowledges fakes are usually slightly cleaner pre-aug. |
| Defer FT-from-P8A to Day-5 fallback (β path) | Saves $60 upfront; if scratch hits triple-axis α, no need to spend at all. |
| Skip local 200-step micro-smoke | This Mac can't run trainer end-to-end (no GPU+GCS pipeline). Substitute: monitor first 500 steps of live launch. Cost of wrong: ~$1 to cancel/relaunch (cheap given $150 total). |
| Single focused commit `cab2909` for tonight's work | 12 files / 1297 lines / 36 tests bundled cleanly. Easier to revert as a unit if needed. |

## Current State

**Working** (verified by tests):
- `loss/anchor_aware_penalty.py` — 8 tests pass; gradient flows through model dummy param.
- `data/augmentations/pipeline_randomization.py` — 10 tests pass; output stays in [0, 255] uint8 even under extreme gamma.
- `data/augmentations/face_scale_jitter.py` — 6 tests pass; min-dim floor handles 16×16 input safely.
- `tests/test_periodic_saves_resolution.py` — 12 tests pass; verifies the c7dc828 bug fix is correct for both dict and non-dict-wrapper inputs.
- `experiments/phase2_round13/R13_P13_FROM_SCRATCH.yaml` — `yaml.safe_load` succeeds; all critical fields verified at write time.
- Trainer integration — module imports clean (no circular import); router accepts new `pipeline_randomization` param; integration smoke shows pipe + scale + anchor all chained correctly.

**In progress** (background process):
- Cloud Build for image 1.3.224. Log at `/tmp/build_prod_2026-04-28.log`. Background bash ID `bv3dkssae`. Expected completion ~17:45 CEST.

**Pending** (Day 3 launch + Day 4 evaluation):
- Build verification.
- Vertex launch (`./scripts/launch/launch_experiment.sh -y phase2r13-experiments us-east1 experiments/phase2_round13/R13_P13_FROM_SCRATCH.yaml`).
- First-500-step monitoring.
- Wednesday daytime parallel audits.
- Day 4 scoring + triple-axis verdict.

**Broken** (not blocking tonight):
- Pre-session dirty files unrelated to tonight's work (HANDOFF.md, VERSION (auto-bumped now), arena/*, docs/*, etc.) — these are leftover from prior sessions, not tonight's. They appear in `git status` but are not part of `cab2909`.

## Critical Files (where to look)

### New (tonight)

| File | Purpose | Lines |
|---|---|---|
| `loss/anchor_aware_penalty.py` | Training-time false-flag pool penalty | 140 |
| `data/augmentations/pipeline_randomization.py` | Symmetric anti-shortcut aug | 147 |
| `data/augmentations/face_scale_jitter.py` | Module-level scale-jitter for collate | 85 |
| `experiments/phase2_round13/R13_P13_FROM_SCRATCH.yaml` | The candidate run | 291 |
| `tests/test_anchor_aware_penalty.py` | 8 unit tests | 154 |
| `tests/test_pipeline_randomization.py` | 10 unit tests | 132 |
| `tests/test_face_scale_jitter.py` | 6 unit tests | 82 |
| `tests/test_periodic_saves_resolution.py` | 12 unit tests | 144 |

### Modified (tonight)

| File | What changed |
|---|---|
| `trainer/trainer.py` | + `from loss.anchor_aware_penalty import AnchorAwarePenalty`. + `self.anchor_aware_penalty = AnchorAwarePenalty(...)` in `__init__` near `init_stability_reg`. + `set_face_scale_jitter_config(...)` in `__init__`. + `losses['overall'] += anchor_loss` in training loop after `compute_stability_loss`. **+29 lines.** |
| `data/augmentations/pipelines.py` | `QualityTargetedFamilyRouter.__init__` accepts `pipeline_randomization` param. `__call__` applies it after `_maybe_apply_teams_sim`. Factory `create_quality_targeted_family_router` forwards param. **(Also bundled: pre-existing webcam_harden defaults block.)** |
| `data/sources/combined_paired.py` | Reads `pipeline_randomization` from `aug_config`; passes to factory. Imports `apply_face_scale_jitter` and calls in collate before resize. **(Also bundled: pre-existing path_exclude_contains plumbing.)** |
| `data/batching/df40_paired.py` | Imports `apply_face_scale_jitter` and calls before the canonical resize at line 362. **+5 lines.** |

### Existing (read-only context)

- `trainer/trainer.py:2326-2340` — periodic_saves resolution with the c7dc828 isinstance(dict) defense. Diagnostic log fires every check.
- `trainer/trainer.py:1485-1517` — main loss aggregation. Anchor-aware compute now lives at line 1517+.
- `trainer/mixins/checkpointing.py:73-77` — `self.anchor_cache_dir` setup (auto-discovered by AnchorAwarePenalty).
- `analysis/teams_pool_rescore.py` — `cache_anchor_pools_locally` + `_LocalAnchorDataset`. Anchor pool is `dor-real-webcam-false-flag-no-virtual-bg`. Cached PNGs go to the dir set by `anchor_cache_dir` (default `~/.cache/anchor_pools/`).
- `experiments/phase2_round9/R9_C_teams_scratch.yaml` — the from-scratch precedent we cloned.
- `experiments/phase2_round9_5/R95_FINAL_REPORT.md` — load-bearing prior analysis recommending scratch.
- `april-26-training-master-plan-v4.md` and prior `HANDOFF.md` — context for the FT path that's now Day-5 fallback only.

## Resume Instructions (next agent)

1. **Read this handoff fully.** Then read `git log -3 --stat` to see commits `cab2909` and `c7dc828`.
2. **Check Cloud Build status.** If still running: `tail -30 /tmp/build_prod_2026-04-28.log` and watch for `✅ Build successful!`. Background process ID was `bv3dkssae`. If completed: verify `cat VERSION` shows `1.3.224` and image is at `us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:1.3.224`.
3. **If build failed**: VERSION will have been auto-reverted to 1.3.223. Inspect log; common causes: file too large (was already 1.6 GiB pre-compression — close to limit; check `.gcloudignore`), Cloud Build quota, transient GCP outage. Don't retry blindly — diagnose first.
4. **Launch.** Once build is good: run the launch command in §"Not yet done" above. Confirm Vertex job state moves to RUNNING within 30 min. If stuck PENDING, switch region per CLAUDE.md.
5. **Monitor.** Open the W&B run page; watch the 7 monitoring items in §"Not yet done" item 3. The first ~10 min after RUNNING is the critical window — anchor_aware logs, pipeline-random init, NaN watch.
6. **Tell user when launched.** Specifically: which region, which W&B run URL, what the first-50-step loss values look like.
7. **Append RESULTS entry** at `april-26-training-master-plan-v2.RESULTS.md` once launched. Append LOG entry at `april-26-training-master-plan-v2.LOG.md`.
8. **Wednesday daytime: launch parallel audits** (Track H.1, H.2, clean_eval_v1, shortcut_probe_v1). These don't need GPU and don't contend with the live training. Use `n_jobs=1` for sklearn (per `feedback_sklearn_njobs`).
9. **Day 4 (Wednesday afternoon ~17:00 CEST): scoring + triple-axis verdict.** Pull all 7 periodic ckpts from `gs://training-job-outputs/phase2r13_experiments/<run_id>/` (the periodic_saves block in the yaml writes there). Score on full validation suite + clean_eval_v1 + shortcut_probe_v1. Build the substrate-comparison table. Verdict α/β/γ per plan v6 §4.

## Memory references the next agent should re-read

- `project_promotion_contract` — Teams Promotion Contract is authoritative; trainer's `value_composite` is not deployment-grade.
- `project_signature_shortcut_finding` — the camera/pipeline shortcut that pipeline-random aug counters.
- `project_face_size_label_leak` — the face-area shortcut that face scale-jitter counters.
- `project_lockbox_fpr_dominated_by_webcam_mode` — modern_v2 filter context for FPR scoring.
- `project_in_proj_svd_gradient_bug` — the in_proj-SVD lever was silently broken pre-2026-04-26; first-time-correctly-active during P13.
- `project_shortcut_is_upstream` — RLP6_04 / FT-only ceiling that motivates the from-scratch pivot.
- `project_p8a_breakthrough` — what FT achieved when it worked (anchor improvement); the bar P13 is trying to clear without inheriting the shortcut substrate.
- `feedback_no_cancelling_vertex_jobs` — explicit user authorization required for cancellation.
- `feedback_sklearn_njobs` — n_jobs=-1 caused 3 reboots on this Mac; use n_jobs=1.
- `project_gcs_region_locality` — US compute on US-multi-region buckets is 10x faster.
- `feedback_decision_points` — present recommendation + tradeoffs; user picks.
- `feedback_small_sample_guidance` — under ~200 frames, summarize aggregate signal; don't bog down in per-subject details.

## Plan reference

Full plan: `~/.claude/plans/ultrathink-read-all-the-generic-flute.md` (Plan v6). Tonight's slice was §10 ("Tonight's Execution Schedule") with §10.1 (bug fix verification finding) noting the c7dc828 commit had already landed before this session.
