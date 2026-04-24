# Handoff: R13 Packet-7 — Camera-Signature Shortcut, Pre-Launch Gate

**Generated**: 2026-04-24
**Branch**: `teams-relaunch-root-2026-04-17`
**Status**: Ready for pre-launch gate. Four commits shipped; launch is blocked on one cheap verification step (post-fix baseline re-score) that the previous agent recommended but did not run.

---

## Goal

Make the Effort detector (leader `RLP6_04`, `value_composite=0.9006`) robust to camera/ISP-signature shortcuts while preserving fake recall. Concretely: stop false-flagging real Teams participants when they use certain webcams (Dor webcam → 0.94, same subject on laptop → 0.02; Roee Mac → 0.90, Roee Windows → 0.01). **Do not launch new experiments** until the pre-launch gate below is resolved — user retains the final decision.

Plan file (approved this session): `/Users/roeedar/.claude/plans/it-s-hard-for-me-proud-journal.md`

---

## TL;DR — Start Here

Before launching any Packet-7 experiment, do the post-fix baseline re-score. Reason: the 0.94 Dor score came from a code path that had silent preprocessing drift (INTER_AREA vs INTER_LINEAR). That drift was fixed this session in `batch_inference_gcs.py` and `arena/model_arena.py`, but **the fix was never re-validated against the original false-flag numbers.** If the fix alone closes most of the gap, the entire training-aug story needs re-scoping.

1. Run post-fix baseline re-score on the 6 Dor/Roee pools (§ Resume Instructions step 1).
2. Interpret per the decision tree (§ Resume Instructions step 2).
3. Launch the appropriate subset from 5 candidate yamls (§ Resume Instructions step 3).

Expected work: step 1 ≈30–60 min, step 2 ≈10 min, step 3 ≈1 hr to kick off.

---

## Completed This Session

- [x] **WS-P0 — Train/inference preprocessing parity fix.** Commit `855871e`. `cv2.INTER_AREA → cv2.INTER_LINEAR` in `batch_inference_gcs.py:407` and `arena/model_arena.py:472` to match training (`combined_paired.py:3455`). Guarded by `tests/test_inference_train_preprocessing_parity.py` (5/5 pass). `arena/model_arena.py` was the load-bearing one: that's the retro-score path that produced the original 0.94 Dor number.
- [x] **WS-P1 — Per-camera calibration probe.** Commit `deac44e`. `analysis/calibration_probe_2026-04-24.py` + `.summary.json`. **Verdict: 0.301 avg gap closure** across target FPRs {5,10,15,25}%. Boundary result between plan decision gates (≥0.60 = calibration, ≤0.30 = training-aug). Read as: training-aug is dominant remaining lever; per-camera calibration is complementary.
- [x] **WS-P2.b — Per-identity reducer.** Commit `deac44e`. `arena/postprocess_per_identity.py` groups `videos_report.csv` by `group_key` and recomputes `real_fpr`/`fake_recall` per identity with `--verify` aggregate-sum check. Covered by `tests/test_postprocess_per_identity.py` (2 tests, synthetic 3-identity fixture).
- [x] **WS-P4.a — Teams spatial aug wiring.** Verified in pipelines.py; no code change needed. The `teams_passthrough_special_*` knobs already live in `_TEAMS_PASSTHROUGH_DEFAULTS` and `_build_teams_passthrough_pipeline` inserts `A.ShiftScaleRotate` when `teams_passthrough_special_aug_enabled=true` + `teams_passthrough_special_shift_p>0`. Smoke-tested end-to-end.
- [x] **Packet-7 yamls drafted (2 new).** Commit `c4affa1`:
  - `experiments/phase2_round13/R13_RLP7_04_teams_spatial_only.yaml` (seed 744, spatial ±3% shift / ±5% scale / ±3° rotate, p=0.5 on Teams passthrough)
  - `experiments/phase2_round13/R13_RLP7_05_teams_spatial_plus_codec.yaml` (seed 745, spatial + `teams_codec_sim_p=0.25`, quality [25,70])
  - Both verified produce expected transforms via `_build_teams_passthrough_pipeline`. Override keys survive the `combined_paired.py:4772` preset-key filter.
- [x] **Handoff doc drafted.** Commit `584118e`. `docs/relaunch_handoffs/R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md`. Longer-form version of this file; candidate subset described there.

---

## Not Yet Done (in priority order)

- [ ] **Post-fix baseline re-score (CRITICAL, pre-launch gate).** Run `RLP6_04` through the now-fixed inference path on the 6 Dor/Roee pools. Compare to the scores in `combined_frame_tags.json` (pre-fix). Outcome determines launch plan. See § Resume Instructions step 1.
- [ ] **Launch Packet-7 subset.** Options ordered by diagnostic alignment:
    - `R13_RLP7_05_teams_spatial_plus_codec` — targets both axes (spatial + compression/bitrate)
    - `R13_RLP7_02_codec_aggressive` — tightest match to fingerprint-diff evidence (`dct_hf_ratio`, `bits_per_pixel`)
    - `R13_RLP7_04_teams_spatial_only` — isolated spatial signal, cheap insurance
    - `R13_RLP7_01_lighting_aggressive` + `R13_RLP7_03_combined` — pre-existing; lighting axis is weaker per controlled data (yellow-vs-white clean control stayed near 0).
- [ ] **WS-P2.a — Stress-variant suites for retro-scoring.** Add `arena/target_domain_suites.*_stress_2026-04-24.yaml` with cells applying `eval_augmentation` = `crop_shift`/`scale`/`rotation`/`jpeg_q30`/`color_warm`/`color_cold` to the `teams_real_all_dev/lockbox` pool. Reuse the presets at `data/augmentations/pipelines.py:1702-1738`. Not started.
- [ ] **WS-P3 — Lockbox-scale fingerprint diagnostics.** Run `analysis/fingerprint_diff.py` with `lockbox_csv_identity` source at scale; rank metrics by mean \|Spearman ρ\| across identities. Expensive (2–3 days GCS fetch). Not started.
- [ ] **Deploy server preprocessing verification.** The original 0.94 Dor score came from `http://34.16.217.28:8999`. We fixed two code paths in this repo, but the deploy server's preprocessing is untouched. If server still uses INTER_AREA, production behavior differs from what retro-score now shows. Out of scope for this repo, but flag to user.

---

## Failed Approaches (Don't Repeat These)

- **Regex `cv2\.resize\([^)]*interpolation\s*=...` in the parity test.** Nested parens from `cv2.resize(img, (self.resolution, self.resolution), ...)` broke the `[^)]*` character class. **Fix:** line-by-line scan — check `"cv2.resize" in line` then match `INTERP_KWARG = re.compile(r"interpolation\s*=\s*cv2\.(INTER_\w+)")`. See `tests/test_inference_train_preprocessing_parity.py:36`.
- **`python -c "from data.augmentations.pipelines import ..."` fails with `ModuleNotFoundError: No module named 'torchdata'`.** The `data/__init__.py` imports trigger the chain. **Fix:** `pip install 'torchdata<0.10'` (0.11 removed the `datapipes` submodule). Noted in commit history.
- **Initial probe design used 5% target FPR with 15 test frames/pool.** Binarized FPR quantizes to multiples of 1/15 = 0.067; noise dominated the signal at tight operating points and produced gap_closure=0.000. **Fix:** probe now sweeps {5, 10, 15, 25}% and reports both per-target and average. The 0.301 headline is the average — not the 5% number. See `analysis/calibration_probe_2026-04-24.py:28`.
- **Probe denominator bug inherited from the original `dor_pool_fingerprint_diff_2026-04-24.py`.** Divided by near-zero `c_std` when clean-pool std was 0 (e.g., `specular_hotspots` all zeros), producing "billion σ" readings. **Fix in `analysis/fingerprint_diff.py`:** `denom = max(c_std, f_std, 1e-6)` with `std_floor_hit` flag. Dropped `specular_hotspots` and `highlight_clip_pct` as "separating metrics" — they were noise.
- **Shared RNG across `.sample()` calls in the original fingerprint diff.** Original script advanced one RNG state through multiple pool draws, so adding a new pool shifted the sample of every subsequent one. **Fix:** fresh per-pool seeds. Reproduces first-called pools exactly; subsequent pools differ slightly from the original script (intended).

---

## Key Decisions

| Decision | Rationale |
|---|---|
| Calibration probe uses average gap closure across target FPRs (not a single target) | Small-sample FPR is heavily quantized at any single operating point (15 test frames → FPR resolution 1/15). Averaging {5,10,15,25}% gives a more robust signal. 0.301 is the average. |
| Per-identity reducer is a separate CLI, not integrated into the scorer | User can run it against any existing `videos_report.csv` without re-running retro-score. Lower commitment; higher reuse. Plan (WS-P2.b) left the option open. |
| RLP7_04 + RLP7_05 fork RLP6_04 (not RLP6_04 + fine-tune from scratch) | Matches RLP7_01/02/03 precedent. Validates deltas as narrow as possible. |
| Yaml-only enablement of `teams_passthrough_special_*` (no code change) | All knobs already exist; all survive the override filter. Any code change would be a nop that adds risk. |
| Recommended launch priority flipped from my own initial ordering | Fingerprint diff pointed at `dct_hf_ratio` + `bits_per_pixel` as cleanest pool separators — both more codec-aligned than spatial-aligned. ShiftScaleRotate doesn't touch either metric directly. |

---

## Current State

**Working**:
- Preprocessing parity — test green, both inference paths use INTER_LINEAR.
- Calibration probe reproducible: `python analysis/calibration_probe_2026-04-24.py` → deterministic output at `analysis/calibration_probe_2026-04-24.summary.json`.
- Per-identity reducer: `python arena/postprocess_per_identity.py --reports <CSV> --threshold <τ> --out <OUT> --verify`.
- Two new yamls parse and produce correct pipelines.

**Broken**: Nothing known broken. `launch_retro_score.sh` and promotion-contract launcher were not touched this session; assume unchanged from main.

**Uncommitted Changes**: Only the long tail of pre-existing modifications on this branch (HANDOFF.md, VERSION, several arena/docs files from prior sessions). None of this session's work is uncommitted — everything shipped across 4 commits.

---

## Files to Know

| File | Why It Matters |
|------|----------------|
| `batch_inference_gcs.py:407` | Fixed INTER_LINEAR. Use for any new bulk re-score. |
| `arena/model_arena.py:472` | Fixed INTER_LINEAR. Retro-score path. |
| `analysis/calibration_probe_2026-04-24.py` | WS-P1 probe. Inputs `/tmp/dor_roee_combined_2026-04-24/combined_frame_tags.json`. |
| `analysis/calibration_probe_2026-04-24.summary.json` | Probe result: 0.301 avg gap closure; per-target breakdown. |
| `arena/postprocess_per_identity.py` | Per-identity reducer for any `videos_report.csv`. |
| `tests/test_inference_train_preprocessing_parity.py` | Guards the INTER_LINEAR fix from future regressions. |
| `tests/test_postprocess_per_identity.py` | Guards the reducer. |
| `experiments/phase2_round13/R13_RLP7_04_teams_spatial_only.yaml` | Isolated spatial-aug variant. Seed 744. |
| `experiments/phase2_round13/R13_RLP7_05_teams_spatial_plus_codec.yaml` | Spatial + `teams_codec_sim_p=0.25`. Seed 745. |
| `experiments/phase2_round13/R13_RLP7_02_codec_aggressive.yaml` | Codec-heavy variant; diagnosis-aligned. Seed 742. |
| `data/augmentations/pipelines.py:787` | `_TEAMS_PASSTHROUGH_DEFAULTS` — single source of truth for override-filter-safe Teams knobs. |
| `data/augmentations/pipelines.py:1086` | `_build_teams_passthrough_special_block` — spatial/cct/shadow/gamma knobs. |
| `data/augmentations/pipelines.py:1386` | `_build_teams_passthrough_pipeline` — where the special block is assembled. |
| `data/sources/combined_paired.py:4772` | Override filter: any yaml key NOT in the first preset's keys is silently dropped. |
| `arena/score_teams_promotion_contract.py` | Contract scorer. `videos_report.csv` has columns `video_id,label,avg_video_prob,method,group_key,family_key,prediction`. |
| `docs/relaunch_handoffs/R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md` | Longer-form companion to this file. |
| `/Users/roeedar/.claude/plans/it-s-hard-for-me-proud-journal.md` | Approved plan — workstream definitions + decision gates. |

---

## Code Context

### Per-identity reducer CLI
```
python arena/postprocess_per_identity.py \
    --reports <suite>_<ckpt>_videos_report.csv [more...] \
    --threshold <tau> \
    --out per_identity_<ckpt>.csv \
    --verify
```
`--verify` prints `-> OK` to stderr when per-group `fp_count`/`tp_count` sums equal the aggregate (exits non-zero otherwise). Handles multiple reports in one invocation — each row carries `report`, `suite`, `checkpoint` inferred from `<suite>_<ckpt>_videos_report.csv`.

### Calibration probe summary structure
```
{
  "summary": {
    "avg_gap_closure_across_target_fprs": 0.301,
    "gap_closure_per_target_fpr": {0.05: 0.000, 0.10: 0.333, 0.15: 0.286, 0.25: 0.583},
    "verdict": "mixed_both_levers_needed"
  },
  "pool_stats": [...],
  "results_by_target_fpr": [...]
}
```

### Teams passthrough override knobs (all survive `combined_paired.py:4772` filter)
```yaml
augmentation:
  version: "quality_targeted_family"
  strength: "vcd_targeted"
  teams_passthrough_special_aug_enabled: true    # MUST be true, else all special-block knobs are no-ops
  teams_passthrough_special_shift_p: 0.5         # probability spatial transform fires
  teams_passthrough_special_shift: 0.03          # +/-3% translation
  teams_passthrough_special_scale: 0.05          # +/-5% scale
  teams_passthrough_special_rotate: 3            # +/-3 degrees
  teams_codec_sim_p: 0.25                        # generic codec re-encode probability
  teams_codec_sim_quality: [25, 70]              # quality range
```

### Reference data
- Pools: `/tmp/dor_roee_combined_2026-04-24/combined_frame_tags.json` — 6 tags × 30 frames, all real subjects, with per-frame `score`/`verdict`.
- GCS source: `gs://real-teams-dor-roee/session_20260424_combined_tags_121458_121007/` — full 180 frames plus metadata.
- RLP6_04 checkpoint: `gs://training-job-outputs/phase2r13_experiments/h2pdu6i5/value_composite_effort_20260424_step23500_auc0.9942_eer0.0169.pth`

---

## Resume Instructions

### Step 1 — Post-fix baseline re-score (pre-launch gate, CRITICAL)

Goal: score `RLP6_04` on the 6 Dor/Roee pools through the now-fixed `INTER_LINEAR` path. Compare to the scores in `combined_frame_tags.json`, which came from the deploy server at `http://34.16.217.28:8999` (preprocessing behavior of that server is **unknown** — see step 1c).

**1a. Pick a path.**
Two viable options:
- **Option A (preferred, cheap):** Use `batch_inference_gcs.py` on the 6 tag folders in `gs://real-teams-dor-roee/session_20260424_combined_tags_121458_121007/<tag>/`. This is the path that was fixed. Takes ~20–40 min incl. checkpoint download.
- **Option B:** Use `arena/run_target_domain_validation_sequential.py` (retro-score path). Requires a checkpoint-map yaml + a target-domain-suites yaml pointing at the 6 pools. More setup, more authoritative. Skip unless Option A is blocked.

**1b. Option A concrete steps.**
```bash
# Reuse the launcher if appropriate, or run batch_inference_gcs.py directly with
#   --checkpoint gs://training-job-outputs/phase2r13_experiments/h2pdu6i5/value_composite_effort_20260424_step23500_auc0.9942_eer0.0169.pth
#   --gcs-prefix  gs://real-teams-dor-roee/session_20260424_combined_tags_121458_121007/
#   (point at each tag folder or process the whole session)
```
Check the launcher usage first: `./scripts/launch/launch_batch_inference.sh --help`. Per memory, `launch_batch_inference.sh` does NOT need `WANDB_*` env vars (unlike the promotion-contract launcher).

**1c. Confirm what the deploy server does.**
The 0.94 Dor score that anchors the whole "camera shortcut" narrative came from `http://34.16.217.28:8999` (see `analysis/check_frame_4people_2026-04-24.py`). Ask the user whether that server's preprocessing is INTER_AREA or INTER_LINEAR. If unknown, the user can either (i) check the server code, or (ii) re-run `check-frame` after we confirm the server has been updated. Until we know, interpret step 1's result with caution.

**1d. Aggregate per-pool mean score and write the result.**
Pair each frame in the re-score output with its tag (from `combined_frame_tags.json`). Compute per-pool mean score post-fix. Compare to the pre-fix means:

| Pool | Pre-fix mean (from `combined_frame_tags.json`) |
|---|---|
| `dor-real-laptop-correct-no-virtual-bg-whiteish` | 0.018 |
| `dor-real-laptop-correct-no-virtual-bg-yellowish` | 0.040 |
| `roee-real-windows-laptop-correct` | 0.007 |
| `dor-real-webcam-false-flag` | 0.878 |
| `dor-real-webcam-false-flag-no-virtual-bg` | 0.940 |
| `roee-mac-laptop-false-flag-virtual-bg` | 0.900 |

Save your result as `analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json` with the same pool keys.

### Step 2 — Interpret and branch

Use this decision tree:

- **If post-fix `dor-real-webcam-false-flag-no-virtual-bg` mean drops by ≥0.30 (to ≤0.64):** preprocessing drift was a LARGE part of the "shortcut." The calibration-probe verdict (0.301 gap closure) overstates the training-aug need because its inputs are pre-fix scores. Before launching anything:
    - Re-run `analysis/calibration_probe_2026-04-24.py` after substituting post-fix scores into `combined_frame_tags.json` (or a derivative file). Preserve the old file for comparison.
    - If the re-run verdict flips to "calibration_is_right_lever" (≥0.60), **descope Packet-7 training and explore production-side per-camera calibration** (separate workstream, not covered by current plan).
- **If post-fix mean drops by 0.10–0.30:** preprocessing drift contributed but not decisively. Launch 1–2 experiments from the codec/spatial side (RLP7_05 first). Hold RLP7_02 + RLP7_04 in reserve.
- **If post-fix mean drops by <0.10:** the shortcut is essentially fully-structural. Launch the full suggested subset (RLP7_05, RLP7_02, RLP7_04).
- **If post-fix mean drops by MORE than expected** (e.g., clean pools now also rise): something else is wrong. Stop, debug.

### Step 3 — Launch (only after step 2)

Recommended order if step 2 doesn't descope:

1. `R13_RLP7_05_teams_spatial_plus_codec.yaml` — hits both axes; highest prior for breaking the shortcut.
2. `R13_RLP7_02_codec_aggressive.yaml` — tightest match to fingerprint-diff evidence.
3. `R13_RLP7_04_teams_spatial_only.yaml` — isolated spatial-only reference (helps interpret _05 vs _02).

Skip `R13_RLP7_01_lighting_aggressive.yaml` and `R13_RLP7_03_combined.yaml` unless budget permits. Lighting axis is weaker per the yellow-vs-white clean control.

Launch path is the existing `./arena/launch_teams_promotion_contract.sh` or equivalent packet-6/7 launcher — per memory, this **requires `WANDB_API_KEY` / `WANDB_ENTITY` / `WANDB_PROJECT` exported first** (unlike batch_inference).

### Step 4 — How to evaluate the results

Once trained checkpoints exist, to decide which Packet-7 variant actually broke the shortcut:

**4a. Retro-score each variant on the promotion contract suite.** Same suite as Packet-6 used; compare `selected_threshold_scorecard.csv`. Make sure `max_pool_fpr` doesn't blow up on any `teams_real_*` pool — that would indicate the aug hurt fake recall or benign-pool calibration.

**4b. Score each variant on the 6 Dor/Roee pools.** Same method as step 1. Decision criteria:
- Target: `dor-real-webcam-false-flag*` + `roee-mac-*` mean scores fall below 0.30 (from ~0.88–0.94 baseline).
- Clean-pool means (`*_clean`) must stay below 0.10 — otherwise the aug generalized poorly.

**4c. Per-identity FPR breakdown using the reducer.** On the lockbox/dev retro-score CSVs:
```
python arena/postprocess_per_identity.py \
    --reports arena/reports/<suite>_<ckpt>_videos_report.csv \
    --threshold <selected_threshold_from_scorecard> \
    --out per_identity_<ckpt>.csv --verify
```
Watch for identities where `real_fpr` is dramatically higher than the suite mean — those are the "new Dor/Roee"s — and for identities where `fake_recall` drops vs RLP6_04 baseline. An ideal Packet-7 variant has narrower `real_fpr` spread across identities with no drop in `fake_recall`.

**4d. Combined scorecard heuristic.** The winner is the variant with:
- Lowest `max(per-identity real_fpr)` on the Teams lockbox
- `fake_recall ≥ RLP6_04 - 0.01` (within noise)
- `value_composite ≥ 0.89` on the contract suite

Any variant that fails condition 2 is rejected regardless of camera-shortcut improvement. Fake-recall regression is the main failure mode to guard against.

---

## Setup Required

- Python deps: `pip install 'torchdata<0.10'` if importing `data.augmentations.pipelines` in a fresh interpreter (the 0.11 release removed the `datapipes` submodule our code imports). Only needed for smoke-testing; training containers pin a working version.
- W&B env vars exported for the contract launcher: `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`. Not needed for `launch_batch_inference.sh`.
- GCP auth: `gcloud auth application-default login` if running locally. On Vertex AI jobs this is handled by the service account.

---

## Edge Cases & Warnings

- **`combined_paired.py:4772` preset-key filter silently drops unknown augmentation overrides.** Always verify new augmentation yaml keys appear in `set(_QUALITY_TARGETED_PRESETS["<any>"].keys())`. The cleanest way: print overrides through `_build_teams_passthrough_pipeline` and inspect the resulting `A.Compose.transforms`. Done for _04 and _05 this session.
- **`teams_passthrough_special_aug_enabled` must be `true` OR every `teams_passthrough_special_*` knob is a no-op.** See `pipelines.py:1088`. RLP7_04 and _05 both set it true; don't forget in any derivative yaml.
- **Calibration probe input is PRE-fix scores.** If you re-run the probe after post-fix scoring, you'll want to either substitute post-fix scores into the JSON, or write a sibling script that reads post-fix scores directly. Don't interpret the 0.301 verdict as applying to post-fix data.
- **RLP7_04/05 seeds (744, 745) were chosen to not collide with RLP7_01/02/03 (740, 742, 743).** Verify no currently-running Packet-7 job is using 744 or 745 before launch.
- **The original 0.94 Dor number came from the deploy server, not this repo.** The INTER fix in this repo doesn't touch production. Flag to user: a full "is the shortcut gone?" answer requires either updating the deploy server or using the repo's inference path as the ground truth for deployment behavior.
- **Don't re-read `arena/model_arena.py` in full** — it's large (>2k lines). Use `Grep` or `Read` with specific offset/limit ranges.

---

## Pointers for the next agent

- Memory system is at `/Users/roeedar/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/`. Respect existing memories; update them if you discover something surprising.
- User (Roee) reserves judgment calls at decision points — present recommendations with tradeoffs and wait for an explicit pick. Do not auto-launch training jobs.
- User prefers: concise summaries, explicit commits, no per-subject interpretive questions when sample size is small. When asked about confidence, give honest calibrated ranges, not "yes this will work."
- When in doubt about a workstream boundary, re-read the approved plan at `/Users/roeedar/.claude/plans/it-s-hard-for-me-proud-journal.md`.
