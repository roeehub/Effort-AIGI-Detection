# Sub-report 05: Measurement Apparatus, GRL, and Metric Gaps

Date: 2026-04-26
Scope: audit of the measurement and probing apparatus that decides whether the Effort detector "passes" or fails. Read with the suspicion that the metrics themselves are wrong: shortcut-learned models can satisfy them while still relying on illegal cues.

Key finding up front: the GRL we ship is structurally incapable of penalizing the camera-signature shortcut, the lockbox is leaky at the identity level, the trainer's `value_composite` is computed from in-distribution-by-source data, and we have no probes that would tell us shortcut learning is happening. The metrics are not just optimistic; they are systematically blind to the failure mode the user already documented (slot-07 dor_shkedi vs real_dor flip).

---

## 1. GRL Implementation in `detectors/effort_detector.py`

### 1.1 The three classes — what each does

`detectors/effort_detector.py:211-274`:

- `GradientReversalFunction` (`:211-221`) — `torch.autograd.Function` whose forward is `x.clone()` (identity) and whose backward multiplies the upstream gradient by `-lambda_val`. Standard DANN GRL (Ganin 2015).
- `GradientReversalLayer` (`:224-233`) — `nn.Module` wrapper holding `lambda_val` and a `set_lambda(val)` setter, calling the function above.
- `QualityDomainHead` (`:236-274`) — small MLP on top of the GRL. Architecture: `Linear(in_features, hidden_dim) → ReLU → Dropout(0.3) → Linear(hidden_dim, num_domains)`. `forward(features)` runs `reversed = self.grl(features); return self.classifier(reversed)`. Default `hidden_dim=128`, `num_domains=4`.

The head is plumbed into `EffortDetector`:
- Constructor (`:361-379`): gated by `config['use_quality_domain_head']`, default off. Reads `quality_domain_loss_weight` (default 0.1), `quality_domain_require_labels` (default True), `quality_domain_count` (default 4), `quality_head_hidden_dim` (default 128).
- Forward (`:1224-1226`): only emits `pred_dict['quality_domain_logits']` when `use_quality_head and not inference`, so quality-head logits are training-only.
- Loss (`:885-958`): computes `F.cross_entropy(quality_domain_logits, domain_labels.long())` and adds `quality_loss_raw * self.quality_domain_loss_weight` to the overall loss. Three diagnostic tensors are exported in the loss dict: `quality_domain_loss_raw`, `quality_domain_has_logits`, `quality_domain_has_labels`, `quality_domain_unique_count`.

### 1.2 Quality-domain label assignment

The label is assigned per-frame inside `data/sources/combined_paired.py`. The mapping table lives at `:65-82`:

```
QUALITY_DOMAIN_MAP = {
    "df40": 0,                             # clean_academic
    "external": 1,                         # webcam_codec (VCD/external)
    "deeplive": 2,                         # studio_capture
    "visomaster": 2,                       # studio_capture
    "visomaster_hints": 2,
    "visomaster_hints_teams": 1,
    "visomaster_enhanced": 2,
    "visomaster_res_variant": 2,
    "visomaster_teams_enhanced": 1,        # mixed; iterator overrides per item
    "deeplive_teams": 1,                   # Teams-passthrough
    "proper_visomaster_clean": 2,
    "proper_visomaster_enhanced_clean": 2,
    "proper_visomaster_teams": 1,
    "proper_visomaster_enhanced_teams": 1,
    "youtube": 3,                          # social_media
}
```

The detector docstring (`:243-247`) advertises four buckets but the actual map (in `combined_paired.py`) has **fifteen source keys collapsed to four IDs**. The `_quality_domain_for_source` resolver (`:85-87`) does a flat dict lookup, defaulting any unknown source to `0` (clean_academic).

Per-source assignments in the iterators:
- `_iterate_df40_paired_sample` (`:2762, :2783`) → `df40` → ID 0.
- `_iterate_deeplive_sample` (`:2841, :2862`) → `deeplive` → ID 2.
- `_iterate_visomaster_sample` (`:2914, :2935`) → uses `unified_sample.source` so could be `visomaster`, `visomaster_hints`, `visomaster_hints_teams`, etc → IDs 1 or 2.
- `_iterate_visomaster_enhanced_sample` (`:2986, :3007`) → hard-coded `visomaster_enhanced` → ID 2.
- `_iterate_visomaster_teams_enhanced_sample` (`:3048-3057, :3085, :3107`) — the only iterator that does per-pair logic: real comes from `deeplive_teams` (ID 1) if `companion_domain == 'teams_v2'` else `visomaster` (ID 2); fake gets the same as real if `branch == "original"` else `visomaster_enhanced` (ID 2). Note the asymmetry — the same physical face can carry different `quality_domain` IDs depending on the random branch draw.
- `_iterate_visomaster_res_variant_sample` (`:3165, :3186`) → `visomaster_res_variant` → ID 2.
- `_iterate_proper_data_sample` (`:3246`) → uses `unified_sample.source` (`proper_visomaster_clean` / `_teams` / `_enhanced_clean` / `_enhanced_teams`).
- `_iterate_teams_sample` (`:3322, :3343`) → uses `unified_sample.source`, typically `deeplive_teams` → ID 1.
- `_iterate_unpaired_real_sample` (`:3399`) → e.g. `external` → ID 1.

The collate function (`:3438, :3467, :3490, :3497`) takes the `quality_domain` of the **first frame** in each video group and emits a per-video tensor under key `'quality_domain'`.

### 1.3 Default lambda

Default `quality_domain_loss_weight = 0.1` (`detectors/effort_detector.py:364`). In `R13_P10_GRL_baseline.yaml:73` it is set to `0.1` (matches default). Note: the GRL `lambda_val` inside `GradientReversalFunction` is a **separate** scalar (the gradient-flip multiplier), defaulted to `1.0` and never overridden in the config (no `set_lambda` call exists in trainer or detector init). So the effective adversarial strength is `1.0 (GRL flip) × 0.1 (loss weight) = 0.1`. There is annealing infrastructure for `lambda_reg` (the SVD orthogonal regularizer, unrelated) but **not** for the GRL strength.

### 1.4 The "R6 brokenness" incident

`git log` does not show a single dedicated fix commit. The defensive code (the `_quality_head_warning_emitted` flag, the `quality_domain_require_labels` strict mode, the `has_quality_logits / domain_labels is None / domain_labels.numel() == 0` branches at `effort_detector.py:888-942`) was all introduced in the Mar 19 monolithic refactor `38558ee` ("refactor: modular trainer + dataset + config system"), and the `quality_domain` field was added to the data path on Apr 19 in `f9303eb` ("Land WT-B runtime and launcher smoke readiness").

The original R6 silent-zero shape:
- `use_quality_head=True` flips on the head and the loss adder.
- Without the data-pipeline plumbing, `data_dict['quality_domain']` is missing and falls through to the `domain_labels is None` branch.
- In the **old** code, `quality_loss` stayed at the initialized `torch.tensor(0.0)`. No exception was raised.
- The only way to detect this was to examine whether `quality_domain_loss != 0`, which we were not logging.

The fix shape (current code, `:885-958`):
- `quality_domain_require_labels: True` is the new default — when on, missing `data_dict['quality_domain']` or empty tensor or missing `quality_domain_logits` raises `RuntimeError` instead of silently zeroing the loss.
- Five new diagnostic tensors get exported in the loss dict: `quality_domain_loss_raw`, `quality_domain_loss`, `quality_domain_has_logits`, `quality_domain_has_labels`, `quality_domain_unique_count`. These let W&B charts show whether the GRL is actually firing.

The fix is real but lives across two commits and was not flagged in the diff message; "GRL" appears in only one commit message in the entire repo (`2c9778b`, the P10 packet).

### 1.5 Is the GRL actually firing in P10_GRL runs?

I cannot inspect live W&B run data from this audit. The expected runtime signature when working:
- `quality_domain_loss_raw > 0` (cross-entropy loss on the head's predictions),
- `quality_domain_has_logits == 1.0` and `quality_domain_has_labels == 1.0`,
- `quality_domain_unique_count > 1` (a batch with only one domain ID gives a perfect-classifier loss of 0 and a uniform gradient — adversarial signal collapses).

Plumbing as wired today *should* be live: P10_GRL_baseline.yaml has `use_quality_domain_head: true`, `quality_domain_loss_weight: 0.1`, `quality_domain_require_labels: true`. With `require_labels=True`, the detector will raise if labels are missing — so a silent-zero R6-style failure is no longer possible, but a **degenerate batch with all samples in one domain** would still produce ~0 loss.

The risk worth checking on the live runs:
- `quality_domain_unique_count` per step. If it's almost always 1 or 2 (out of 4), the head has nothing to discriminate against and GRL is a no-op even when "firing."
- The training mix (`R13_P10_GRL_baseline.yaml:222-236`) heavily upweights `deeplive_teams_real` (4.0), `deeplive_teams_fake` (5.0), and `realpool_real` / `external_real` (2.5 each). Many of those land in **the same** quality_domain bucket (1 or 2). A batch could easily be 80%+ ID 1 + ID 2 with very few df40 (ID 0) or youtube (ID 3) samples. Verify on W&B before claiming GRL is doing work.

### 1.6 CRITICAL: the GRL is orthogonal to the camera-signature shortcut

The user's diagnosis (memory `project_signature_shortcut_finding.md`) is that slot-07 flips its prediction depending on the **processing/capture signature** even within the same physical person (`dor_shkedi` vs `real_dor`). Both of those are ID 2 in our quality-domain map (both Teams-distribution real captures of the same person). They differ in:
- Camera/sensor used to record the original video,
- Compression history (Teams encoder pass vs raw),
- Color/exposure pipeline,
- Cropping pipeline.

None of those distinctions show up as separate `quality_domain` IDs. The GRL is asked to remove information that lets the head distinguish among 4 coarse buckets (`clean_academic / webcam_codec / studio_capture / social_media`), not the much finer per-camera signature that slot-07 actually exploits.

A model that perfectly fools the QualityDomainHead can still trivially encode `(camera_id, encoder_id, color_pipeline_id)` in its features — those are subspaces orthogonal to the 4-way coarse domain label. The features will still be useful for shortcut learning; the GRL just won't punish that.

This is not a bug in the GRL math; it is a **specification bug** in the choice of adversarial labels. To penalize the camera shortcut, we would need:
- domain labels at the camera/session/identity level, OR
- a different debiasing target (per-identity contrastive, per-session adversarial, etc).

Treat any P10_GRL improvement as evidence of "less coarse quality leakage" and **not** as evidence the camera shortcut is suppressed. We have not built the probe that would falsify the GRL→shortcut-killed claim; see §8 for what's missing.

---

## 2. Promotion Contract Scoring (`arena/score_teams_promotion_contract.py`)

### 2.1 The lexicographic scoring logic

`_threshold_sort_key` lives at `:430-465`. Two policies coexist:

**Budget-active path (`:436-454`)** — engaged when both `target_real_fpr < 1.0` and `target_stress_fpr < 1.0`:
- key = `(0 if not violates else 1, -macro_recall, -threshold, primary_fpr)`,
- "violates" means `dev_primary_real_fpr > target_real_fpr` or `dev_worst_real_stress_fpr > target_stress_fpr`,
- Among satisfying rows: maximize `dev_fake_macro_recall`, prefer higher τ (more conservative), tiebreak on lower primary FPR.
- Among violating rows: lex-minimize FPR (just to be deterministic).

**Legacy path (`:457-465`)** — engaged when budgets are ≥1.0 (disabled):
- key = `(primary_fpr, stress_fpr, *(-fake_recall per dev fake suite), threshold)`,
- This is the broken minimize-FPR-with-no-budget policy.

Default budgets in code (`:260-261`): `target_real_fpr: 0.02`, `target_stress_fpr: 0.05`. Default budgets in CLI (`:672-685`): same `0.02 / 0.05`. So **as currently invoked**, the budget path is active by default.

`_promotion_summary_sort_key` (`:468-479`) ranks checkpoints by:
1. `lockbox_real_fpr` (lower better),
2. `lockbox_fake_recall` (higher better),
3. `dev_primary_real_fpr`, `dev_worst_real_stress_fpr`,
4. per-suite dev fake recalls,
5. `selected_threshold` (higher better) — tiebreaker.

### 2.2 Calibration on dev → readout on lockbox

`ContractConfig` (`:249-280`) defines:
- `dev_real_suite` (singleton, default `teams_real_all_dev`),
- `dev_real_stress_suites` (default `teams_real_poor_quality_dev,teams_real_lighting_extreme_dev`),
- `dev_fake_suites` (default `teams_fake_all_dev,visomaster_enhanced_macro_dev,deeplive_enhanced_dev`),
- `lockbox_real_suite` (default `teams_real_all_lockbox`),
- `lockbox_fake_suite` (default `teams_fake_all_lockbox`),
- `readout_only_suites` (default `teams_real_dor_dev`) — appears in scorecard but does NOT influence τ.

τ candidates are the union of every dev sample's predicted probability plus 0.0 and 1.0 (`_candidate_thresholds:373-384`). For each candidate, build a grid row of dev FPR/recall metrics, sort by `_threshold_sort_key`, take the top row. That τ is then used to read out the two lockbox suites. Lockbox suite predictions never influence τ selection.

### 2.3 The "contract policy bug"

Memory file `project_contract_policy_bug.md` describes the bug as: "step 1 has no budget — picks lowest τ where FPR rounds exactly to 0." That refers to the **legacy path** at `:457-465`. On a sharp-prediction model the cumulative FPR distribution has a long flat tail at zero that runs almost to 1.0, so the lex-min picks τ ≈ 0.995 and crushes fake recall.

**Status:** the bug is fixed conditionally. The budget path (`:436-454`) was added 2026-04-23 with default budgets `0.02 / 0.05`. So scorecards run with the current defaults are interpretable. But:
- **Historical scorecards** (anything from before 2026-04-23, including the slot-07 sanity-check numbers in the memory file) were run under the legacy policy. The recorded `selected_threshold` of 0.9953 in that memory is the broken-policy artifact, not deployment-grade.
- **Override risk:** if a caller passes `--target_real_fpr 1.0`, the legacy path comes back. The CLI accepts that, no warning emitted.
- **Budget choice risk:** at budget `target_real_fpr=0.02` the policy picks the τ that maximizes dev macro fake recall while staying ≤ 2% FPR on `teams_real_all_dev` and ≤ 5% on each stress suite. That's a legitimate operating point but it is NOT robust to leakage — see §3.3.

### 2.4 `selected_threshold` sanity-check rule

The "fix" still leaves a hidden hole. Even with the budget active:
- If no τ satisfies both budgets (the model's natural FPR floor on `teams_real_all_dev` is > 2% across all τ), all rows are flagged "violates", and the fallback ranking is again "lex-minimize FPR." That re-introduces the original pathology.
- The scorecard will report a τ but it will be drawn from the violating-row ordering.

So **always** sanity-check `selected_threshold` against `dev_primary_real_fpr` and `dev_worst_real_stress_fpr` in the same row of `threshold_grid.csv`:
- `selected_threshold > 0.99` → policy is operating under "violating" mode (or budgets disabled). Rerun with looser budgets or accept the readings as a lower-bound diagnostic only.
- `dev_fake_macro_recall < 0.20` with `selected_threshold > 0.95` → same story.

### 2.5 Hard caps

The promotion contract scorer itself has **no hard caps** on lockbox metrics — it always returns a winner. The hard caps cited in the user's memory ("7% lockbox FPR hard cap") live in the **checkpoint map description** (`arena/checkpoint_maps/teams_target_domain.p10_partial_2026-04-26.yaml:9-14`), not in code:

```
# Crowning protocol (lockbox-anchored, NOT trainer's value_composite):
# - lockbox_real_fpr ≤ RLP6_04 (≤ 0.441%)
# - lockbox_fake_recall ≥ max(0.80, RLP6_04 − 0.02)  (≤2pt macro slack)
# - Per-method recall: deeplive_enhanced ≥ 3.4%, visomaster_enhanced_macro ≥ 0.3%
# - 7% lockbox FPR hard cap
```

These are operator-side rules, applied manually when reading `promotion_winner.json`. There is no automated enforcement.

---

## 3. Target Domain Suites (`arena/target_domain_suites.teams_promotion_contract_2026-04-17.yaml`)

### 3.1 The 8 suites — composition and sample counts

All eight suites pull from `arena/manifests/teams_target_domain_manifest_2026-04-06_frozen.json`. That manifest holds 7,276 videos pre-split by hashing `identity_key` against `split_seed=737` with `lockbox_ratio=0.20` (`build_teams_target_domain_manifest.py:475-480`). Manifest summary block: 4,614 reals, 2,662 fakes; 5,662 dev / 1,614 lockbox.

| Suite | Split | Slice | Sample type | Count |
|---|---|---|---|---|
| `teams_real_all_dev` | dev | `teams_real_all` | real-only | derived from manifest `teams_real_all` ∩ dev (~4,000 of 4,614) |
| `teams_real_poor_quality_dev` | dev | `teams_real_poor_quality` | real-only | subset of dev real flagged by `quality_thresholds.poor_quality` |
| `teams_real_lighting_extreme_dev` | dev | `teams_real_lighting_extreme` | real-only | subset of dev real flagged by `quality_thresholds.lighting_extreme` |
| `teams_fake_all_dev` | dev | `teams_fake_all` | fake-only | dev portion of `teams_fake_all` |
| `visomaster_enhanced_macro_dev` | dev | `visomaster_enhanced_macro` | fake-only | 550 in manifest (split-applied) |
| `deeplive_enhanced_dev` | dev | `deeplive_enhanced` | fake-only | 545 in manifest (split-applied) |
| `teams_real_all_lockbox` | lockbox | `teams_real_all` | real-only | 1,361 (4,614 × ~0.295 lockbox-fraction observed for reals) |
| `teams_fake_all_lockbox` | lockbox | `teams_fake_all` | fake-only | 253 (2,662 × ~0.095) |

Slice membership is set in `arena/build_teams_target_domain_manifest.py:469-474` and refined per-prefix in `arena/prefix_rules.teams_manifest.template.yaml`. Identity-key counts: 39 unique identity_keys total, 33 dev-only, 6 lockbox-only, 0 in both.

Lockbox-only identity_keys (verified from manifest):
- `Cam_Test__s33` (191 fake videos),
- `Chikara_Takahashi__s22` (25 real),
- `PC_Generator__s15` (62 fake + 28 real, paired),
- `bla_bla_chow__s1` (61 real),
- `dor_shkedi` (1,138 real),
- `real_dor` (109 real).

### 3.2 Dev vs lockbox classification

- Dev (used to calibrate τ): `teams_real_all_dev`, `teams_real_poor_quality_dev`, `teams_real_lighting_extreme_dev`, `teams_fake_all_dev`, `visomaster_enhanced_macro_dev`, `deeplive_enhanced_dev`.
- Lockbox (readout only): `teams_real_all_lockbox`, `teams_fake_all_lockbox`.

### 3.3 CRITICAL: is the lockbox actually held out?

Three distinct holdout claims to test:

**(a) Identity-key disjointness** — VERIFIED. `dor_shkedi` (lockbox) and `dor_shkedi__s16` (dev) are distinct `identity_key` strings; the hash split places them in different splits. `bla_bla_chow` (dev) vs `bla_bla_chow__s1` (lockbox) — same. Hash policy is correct under the literal definition of identity_key.

**(b) Person-level disjointness** — VIOLATED. The same physical person ("Dor") appears in:
- dev as `dor_shkedi__s16` (fake, teams_capture_dor_shkedi_s16, 57 videos), `dor` (real, 269 videos), `deeplive_dor` (fake, deeplive_enhanced, 545 videos),
- lockbox as `dor_shkedi` (real, 1,138 videos), `real_dor` (real, 109 videos).

The lockbox readout for `dor_shkedi` real is therefore tested on a face the model has seen in training context as `dor` real, `dor_shkedi__s16` real, and as the swap target of `deeplive_dor` fake. A real-FPR readout on `dor_shkedi` is **not** an OOD test of generalization to a new person. Same for `bla_bla_chow__s1` (lockbox) vs `bla_bla_chow` and `bla_bla_chow__s2` (dev). Same for `PC_Generator__s15` vs the many other `PC_Generator__s*` sessions in dev.

This is exactly the confound in the slot-07 finding: the model uses the camera/encoder/color-pipeline signature, not the identity, to decide. Within the same person, the lockbox capture has a different processing signature than the dev capture, so a model that has learned the signature shortcut gets a non-trivial test, but only because the SHORTCUT differs across signatures, not because the IDENTITY does.

**(c) Bucket-level disjointness from training** — UNCLEAR / partial. The validation manifest pulls from `gs://teams-faces-data-test-2914-fake-4420-real-feb-28`. Training (`R13_P10_GRL_baseline.yaml:188`) pulls from `gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2`. Different buckets. But:
- Both buckets sample from the same set of recorded sessions (the user's collection of Teams calls / cam_test / pc_generator captures); the test bucket is a **renamed re-export** of the same underlying material with the manifest-builder's video-ID conventions.
- Without auditing whether the same video files exist in both buckets, I can't confirm zero file-level overlap. There is no automated leak check in the codebase. **Gap.**

The `external_training_reals` block in training (`R13_P10_GRL_baseline.yaml:210-220`) uses the VCD bucket, not Teams, so external_real → external_real overlap is moot. But Teams-real frames in training and Teams-real frames in lockbox could overlap and we have no audit script to disprove it.

### 3.4 Per-method fake leakage in lockbox

`teams_fake_all_lockbox` consists almost entirely of `teams_capture_cam_test_s33` (191) plus `teams_capture_pc_generator_s15` (62). `Cam_Test__s33` is one of many `Cam_Test__s*` sessions; Cam_Test sessions s32, s35, s38, s46, s53, s73, s76 are all in dev fake. So:
- The lockbox fake test is almost entirely "did the model overfit the Cam_Test camera-pipeline signature seen in dev." If it did (the slot-07 finding), lockbox fake recall is artifactually high.
- This explains why slot-07 showed strong dev fake recall but the user still distrusts deployment-readiness — the lockbox is leaky **at the camera-signature level** even when identity_keys are formally disjoint.

---

## 4. Validation Runner (`arena/run_target_domain_validation_sequential.py`)

End-to-end:
1. Parses CLI: checkpoint map, suite manifest, output_gcs_folder, optional scorecard CSV/JSON paths, optional `--promotion_contract_dir`.
2. For each `(checkpoint, suite)` pair, shells out to `validate_custom_sources.py` (`:47, :312`) which runs the model in inference mode over the suite's manifested data and writes `<suite>_<ckpt_lower>_videos_report.csv` to `output_gcs_folder`.
3. Each `videos_report.csv` carries columns: `method, label, video_id, avg_video_prob, prediction, is_correct, group_key, family_key` (`trainer/trainer.py:3250-3252`). `prediction` is at fixed threshold 0.5.
4. After all jobs complete, the runner:
   - Aggregates per-suite metrics at fixed-threshold 0.5 into the diagnostic scorecard (CSV / wide CSV / int8-delta / JSON) — `_build_scorecard_row:312-393`.
   - If `--promotion_contract_dir` set, dynamically imports `score_teams_promotion_contract.py` and runs `score_promotion_contract` (`:436-491`), producing `threshold_grid.csv`, `selected_threshold_scorecard.csv`, `checkpoint_summary.csv`, `promotion_contract.json`, `promotion_winner.json`.

Per-suite metrics (at 0.5 threshold, real_only/fake_only auto-detected from `_suite_label_mode`):
- `n_videos`, `mean_prob`, `p50_prob`, `p90_prob`,
- `tn / fp / fn / tp`,
- `accuracy_at_0p5`, `real_fpr_at_0p5`, `real_tnr_at_0p5`, `fake_recall_at_0p5`, `fake_fnr_at_0p5`,
- `score_metric_name` and `score_metric_value` (the suite's headline number — FPR for real-only, recall for fake-only, accuracy for mixed).

These are diagnostic; the calibrated contract is the authoritative scoring layer.

---

## 5. Trainer-time vs Deployment-time Metrics

### 5.1 `value_composite` definition

Lives in `trainer/trainer.py:113-295`. Three pieces:

- `_VALUE_COMPOSITE_REAL_POOLS` (`:115-122`): the six real pools that make up the FPR operating point — `df40_real, external_youtube_avspeech_real, zoom_vcd_real, teams_ood_real, proper_clean_real, proper_teams_real`.
- `_VALUE_COMPOSITE_DF40_TRAINING_FAKES` (`:124-129`): seven df40 methods (simswap, facedancer, blendface, e4s, inswap, mobileswap, uniface) excluded from the "other fakes" TPR because they are training-distribution.
- `_VALUE_COMPOSITE_STABILITY_JITTER_METHODS` (`:131-135`): three methods feeding the stability term — `teams_ood_fake, teams_ood_real, external_youtube_avspeech`.

Computation (`_compute_value_composite:220-295`):
1. Bisect τ on the union of real pools so `mean_FPR_across_pools == target_mean_fpr` AND `max_pool_FPR ≤ max_pool_fpr` (default 0.02 / 0.04, `R13_P10_GRL_baseline.yaml:79-80` overrides to 0.03 / 0.05).
2. If no τ satisfies both gates → `value_composite = NaN`, `blocked_by ∈ {insufficient_real_pools, worst_pool_fpr}`.
3. At the chosen τ, compute mean TPR over `teams_fake_pools` (Teams-flavored fakes), mean TPR over `other_fake_pools` (the rest, with df40-training fakes filtered out), and a stability term `1 − clamp(stability_jitter_max, 0, 1)`.
4. `value_composite = (0.6 × teams_TPR + 0.3 × other_TPR + 0.1 × stability) / active_weight`, where `active_weight = 0.1 + 0.6×[teams present] + 0.3×[other present]`.

Pool routing happens in `ood_monitoring_epoch:3062-3098`. Methods are matched to pools by substring on the method-name string (`"teams_ood_real" in m_norm`, `"deeplive" in m_norm`, etc).

### 5.2 When AUC and EER are logged

`_run_validation:1853-2059` runs at every `evaluate_every_steps` (500 in P10, `R13_P10_GRL_baseline.yaml:349`). The data flow:

1. `val_in_dist_loader` → `test_epoch(log_prefix="val_in_dist", is_primary_metric=False)`. Logs `val_in_dist/overall/{auc, eer, eer_threshold, ...}`.
2. `val_holdout_loader` → `test_epoch(log_prefix="val_holdout", is_primary_metric=True)`. Logs `val_holdout/overall/...`. This is the AUC/EER reported in checkpoint filenames.
3. `_run_ood_monitoring` (gated by `first_ood_step` and `ood_cadence`, defaults 1000 / 500 in P10). Logs `ood/overall/auc`, `ood/method/<m>{_real|_fake}/{acc,auc,eer}`, family aggregates `ood_lighting_stress/overall/auc`, `ood_spatial_stress/overall/auc`. Also computes `value_composite` from the OOD pools at this point — the OOD loader is the **only** source of pool data for value_composite.

What data feeds these:
- `val_in_dist` and `val_holdout` come from `combined_paired.create_combined_paired_loaders` → split by identity hash (`hash_stable` mode in P10) into 85 / 10 / 5 across the same training sources. **In-distribution by source**, identity-disjoint at the hash-key level (proper-data uses `split_group_id`, deeplive/visomaster use `realpool_*`; df40 uses `df40_*`).
- `val_holdout` does not include any Teams-distribution capture — it is whatever 5% of training-source identities the hash put in test. Deeplive/visomaster captures of the same person across realpool can collide because of shared identity prefix `realpool_` (`combined_paired.py:1240-1247`).
- `ood_loader` is built from the `ood_monitoring` block in the YAML (`R13_P10_GRL_baseline.yaml:246-340`) — `teams_ood_real / teams_ood_fake` from a third Teams bucket, plus `lighting_stress_*` / `spatial_stress_*` synthesized via on-the-fly augmentation of `external_youtube_avspeech` reals. There is also a `readout_only_external_real_sources` `teams_real_dor_20260424_session` from a "Roee/Dor" Teams session — added 2026-04-24 for OOD tracking.

### 5.3 Why `value_composite` is NOT deployment-grade

Per memory `project_promotion_contract.md` and the actual code:
1. **Real-pool composition is wrong for Teams deployment.** The FPR operating point is bisected to satisfy `df40_real`, `external_youtube_avspeech_real`, `zoom_vcd_real`, `proper_clean_real`, `proper_teams_real`, AND `teams_ood_real` simultaneously. AVSpeech (YouTube celebrity reads) and zoom_vcd are noisier and more diverse than Teams calls, so to keep their FPR ≤ ceiling τ has to climb. The same τ is then applied to teams_fake recall, which crashes.
2. **df40 inclusion in real pool.** df40 reals are clean academic-style frames; FPR there at any normal τ is ~0. Including df40 in the mean does not constrain τ but does dilute the signal — the bisection target is a flawed proxy for "Teams real distribution."
3. **No lockbox readout.** value_composite is computed from data the model has been hyperparameter-tuned against (since OOD AUC drives `ood_composite` checkpoint selection at `:1962-2054`). It is in-distribution for hyperparameter selection.
4. **Misalignment with deployment target.** The promotion contract reports lockbox FPR/recall on a **fixed Teams-only manifest** with deterministic dev/lockbox split. The value_composite operating point is at a different τ on a different real-pool mix and tells you nothing directly about lockbox numbers.

The user's memory says slot-07 reports `value_composite = 0.7736` (deflated) while at its natural τ teams TPRs are 97-100%. That gap is the value_composite mis-specification.

---

## 6. `value_composite` Checkpoint Selection

`trainer/trainer.py:2060-2135`. When `value_composite_enabled` and a non-NaN `value_composite` is computed, the trainer:
- Tracks `best_value_composite` and `best_value_composite_step`.
- Caches the state dict in CPU memory (`_best_value_composite_state_dict_cpu`).
- Saves to GCS with prefix `value_composite_` if it improves AND a holdout AUC is available (`:2105-2135`). Maintains a top-N list (default size 1).

The checkpoint maps you actually consume (`arena/checkpoint_maps/teams_target_domain.p10_partial_2026-04-26.yaml`) all reference `value_composite_effort_*` paths — i.e., **deployment review currently selects checkpoints by value_composite**.

This is the wrong selection criterion if value_composite is misleading:
- The trainer chooses τ ≈ 0.6-0.8 to satisfy `target_mean_fpr=0.03` across the six diverse real pools. The promotion contract, on a Teams-only real distribution, picks τ ≈ 0.93-0.98 at budget 0.02. Different τ → different model state is "best."
- The improvement signal that lifts `value_composite` could be "model just got better at separating df40 + AVSpeech reals from teams fakes" — that has nothing to do with whether the camera-signature shortcut is suppressed.
- A model that learns the camera-signature shortcut harder will likely also have **higher** value_composite (because it can confidently assign training-distribution real vs Teams-fake), so this selection criterion structurally rewards shortcut intensification.

There is also a parallel `ood_composite` checkpoint (hmean of holdout_auc and ood_auc, `:1962-2054`). It has the same in-distribution-by-source problem. The `final_eval` block at `:2680-2810` reloads both `best_ood_composite` and `best_value_composite` for end-of-run reports. There is no checkpoint selection criterion based on lockbox metrics.

**Recommendation (out of scope for this audit):** add a third checkpoint track ranked by per-step promotion-contract scoring, or at minimum by lockbox readout. This requires running the validation pipeline mid-training, which currently happens only via `arena/launch_teams_promotion_contract.sh` post-hoc.

---

## 7. OOD Monitoring During Training

### 7.1 Sources

`R13_P10_GRL_baseline.yaml:246-340` (representative):

- `external_real_sources`: `teams_ood_real` from `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2/samples` filtered by path `/frames/real/`. 300 videos. **Same bucket and prefix as TRAINING teams data.** With `exclude_training_identities: true` (`:248`), training identities are removed; the remaining 300 are held out at the identity level.
- `readout_only_external_real_sources`: `teams_real_dor_20260424_session` from `real-teams-dor-roee/session_20260424_110139/uniform30`. 10 videos, OOD-monitor-only, does not flow into value_composite.
- `external_fake_sources`: `teams_ood_fake` from same training bucket / `frames/fake/` excluding visomaster. 300 videos. **Same bucket as training fakes.**
- `lighting_stress_sources` (3 presets): `external_youtube_avspeech` reals run through `vcd_targeted_stress`, `backlight_dim_stress`, `warm_harsh_stress` augmentations at eval time. 200 videos each.
- `spatial_stress_sources` (3 presets): `external_youtube_avspeech` reals through `crop_shift`, `scale`, `rotation` augmentations. 200 videos each.

The eval-augmentation feature lives in the augmentation registry; the augmentations are applied deterministically per-video (same seed → same warp).

### 7.2 When OOD eval fires

- `first_ood_step: 1000` (`:250`),
- `ood_cadence: 500` (`:251`),
- triggered inside `_run_validation:1959` whenever `(step_cnt - first_ood_step) % ood_cadence == 0`.
- Re-entrant guard at `trainer.py:2852-2853` prevents firing twice on the same step.

### 7.3 Metrics produced

In `ood_monitoring_epoch:2860-3155`:
- Per-method: `ood/method/<m>{_real|_fake}/{acc, auc, eer}`.
- Overall: `ood/overall/{auc, eer, acc, ap}`.
- Score-jitter (per-video frame-to-frame std of probs): `ood/score_jitter[/_max/_p95/_spike_rate_0p3]/<m>`. Logged to `summary/` namespace for `teams_ood_fake`.
- Stress family aggregates (`ood_lighting_stress`, `ood_spatial_stress`): per-preset AUC + family overall AUC.
- value_composite block: `value_composite/{tau, mean_fpr, max_fpr, max_fpr_at_mean_02, teams_fakes_tpr, other_fakes_tpr, stability, value_composite, blocked_by}` plus per-pool FPR.

### 7.4 Are these checked or just logged?

- **Driving checkpointing:** Yes — `ood_composite_top_n` checkpoints at `:2026-2054`, `value_composite_top_n` checkpoints at `:2105-2135`.
- **Driving early stopping:** No directly. `EarlyStoppingMixin` watches `val_holdout` AUC.
- **Used for go/no-go:** No automated gate. The user manually compares run W&B summaries against the criteria in the checkpoint map preamble.
- **Reviewed at promotion time:** Lightly. The promotion contract uses an entirely separate evaluation pipeline (`arena/`, against the frozen Teams manifest), and the trainer-time OOD numbers are not the deciding metric. They function as a sanity dashboard, nothing more.

---

## 8. What We DO NOT Measure (Gaps)

### 8.1 Confirmed missing probes

**(a) Linear probe on embedding for camera/identity.** The `EffortDetector.features` output (`:777-779`, `pooler_output` from the backbone) is what the head consumes and what shortcut learning would encode. We do not run an offline probe of the form: freeze the backbone, train a small linear classifier on `feature → camera_id` (or `→ identity_key`, or `→ session_id`), and report classifier accuracy. If a linear probe trained on a small subset reaches near-perfect camera_id accuracy on held-out frames from the same camera, the embedding has memorized the camera signature. **We don't have this.** Adding it would give us a direct quantitative shortcut measurement.

**(b) Cross-camera train/test split.** The current dev/lockbox split is on `identity_key`. The promotion contract holds out *some* identities, but every identity in lockbox shares its camera/encoder/color-pipeline with multiple identities in dev (Cam_Test__s33 lockbox vs Cam_Test__s32/s35/s38/s46 dev — same room, same camera). A genuinely camera-stratified split (lockbox = videos from cameras unseen in dev) does not exist. **No such split is built.**

**(c) Same-identity-different-camera spread monitoring during training.** For people who appear in both Teams (`live-deepfake-methods-real-and-fake-frames-cropped-teams-v2`) and the proper-data inventory (clean studio capture), we could log `mean_prob[same_person, camera_A] - mean_prob[same_person, camera_B]` per step. A model learning the identity should give similar probs across cameras; a model learning the camera signature shows large spread. **No such metric exists.**

**(d) Source-of-origin classifier on embedding.** Same idea as (a) but at the source-bucket granularity (df40 / external_vcd / proper_teams / live-cropped / teams-passthrough). A classifier on frozen features to predict the source bucket directly tests whether the GRL is doing its job. Currently the `quality_domain` cross-entropy loss is reported, but only on the **adversarial head** (which is being actively suppressed). We never read out a NON-adversarial probe of the same target. **Nothing in the codebase reads out a frozen-feature source classifier accuracy.**

**(e) Identity-recall / face-recognition test on embedding.** Take features for two frames of the same person under different cameras and measure cosine similarity vs across-person same-camera similarity. If face-ID similarity within-person is *lower* than within-camera, the embedding is camera-dominated. This is the textbook test for "is the representation about the face or the camera?" **Not in the repo.**

**(f) File-path / metadata leak audit.** No script in `arena/` or `analysis/` cross-references training file paths against validation manifest file paths to confirm zero overlap at the file-path level. Given that training and lockbox draw from related but differently-named buckets of the same underlying recordings, the risk of literal file-level leakage is non-zero. **No automated check exists.**

### 8.2 Additional probes a research-grade defense would include

- **Background-only / face-only ablation** at evaluation time. If a model still scores Teams real "real" at high confidence when the face is masked out (just the room background), it is using the room/lighting signature, not the face. We do have eval-time augmentation slots (`lighting_stress_*`, `spatial_stress_*`) but no face-occlusion preset.
- **Color/texture invariance probes.** Recolor / desaturate a video and measure prob delta. Real face features should be largely color-invariant; codec/encoder signatures will shift dramatically. We have lighting stress but not a clean color-invariance probe.
- **JPEG-quality scrub probe.** Re-encode lockbox real videos at different JPEG qualities and measure FPR drift. If FPR jumps when re-encoding shifts the codec signature, the codec signature is the deciding feature.
- **Per-video score-jitter alignment with semantic content.** We log per-video score jitter (`ood/score_jitter/...`) but never correlate jitter with semantic transitions (face turn, occlusion, lighting change). High jitter at semantic transitions = model attending to face; high jitter at random frames = model attending to compression noise.
- **Counterfactual same-person face swap into a different camera.** Take a real frame of person X in camera A and naive face-warp it into camera B's color statistics. Compare the model's prob. If the model predicts based on camera, the warped frame flips. We have nothing like this.
- **Gradient-based attribution maps over a held-out subset.** Saliency / Integrated Gradients on lockbox real and lockbox fake to confirm the model attends to faces, not background. We compute embeddings (analysis dir has many `_fingerprint_*` artifacts) but no automated saliency monitoring.
- **Confidence-calibration plot per-bucket (camera, identity, lighting).** A well-debiased model has flat confidence across buckets; our current scorecards only break out by `method` and `slice_hint`. We do not break out by `camera_id` or by `lighting_extreme/poor_quality` cross-tabulation.

### 8.3 What we DO have (so the gap list is honest)

- `analysis/dor_pool_fingerprint_diff_2026-04-24.py`, `analysis/feature_space_2026-04-23/` — ad-hoc embedding visualizations specifically for the dor case. Manual, not part of the automated dashboard.
- Per-method scorecards in the promotion contract.
- Score-jitter metrics on OOD pools.
- `quality_domain_unique_count` per training step (added with the GRL fix) — diagnostic only.

The gap is the **automation layer** that would turn any of these into a flag during a P10 training run and the absence of probes targeting the camera shortcut specifically.

---

## Summary (1-paragraph version for the receiver)

The GRL currently shipped operates on a 4-bucket coarse domain map (clean_academic / webcam_codec / studio_capture / social_media) — orthogonal to the per-camera signature the slot-07 finding documented; suppressing the GRL head will not suppress the camera shortcut. The promotion-contract scorer's policy bug was conditionally fixed 2026-04-23 by adding default FPR budgets `0.02 / 0.05`, but historical scorecards remain uninterpretable, the budget can fail to be satisfied (silently reverting to the broken policy), and the contract's lockbox is identity-key-disjoint but person-level leaky (e.g. dor_shkedi/lockbox vs dor/dev/dor_shkedi__s16 are the same person; Cam_Test__s33/lockbox shares its camera signature with Cam_Test__s32/s35/s38/s46 in dev). The trainer's `value_composite` is computed against a 6-pool real mix that includes df40, AVSpeech, and zoom_vcd — none of them Teams-passthrough — so its τ is misaligned with deployment, and selecting checkpoints by `value_composite` (which is what `arena/checkpoint_maps/*.yaml` does) structurally rewards models that satisfy the composite, including via shortcut learning on Teams-distribution data. We have no linear probe, no camera-stratified split, no same-identity-different-camera spread monitor, no source-of-origin classifier on embedding, no identity-recall test, no file-path leak audit. The metrics will continue to pass while shortcut learning continues, until at least one of those probes is built and watched.

---

## File:line index

- `detectors/effort_detector.py:211-274` — GRL classes.
- `detectors/effort_detector.py:243-247` — quality-domain comment (4 buckets).
- `detectors/effort_detector.py:361-379` — GRL gating in `EffortDetector.__init__`.
- `detectors/effort_detector.py:885-958` — GRL loss path.
- `detectors/effort_detector.py:1224-1226` — GRL forward path.
- `data/sources/combined_paired.py:65-87` — `QUALITY_DOMAIN_MAP` and resolver.
- `data/sources/combined_paired.py:2762,2783,2841,2862,2914,2935,2986,3007,3048-3057,3085,3107,3165,3186,3246,3322,3343,3399` — per-iterator `quality_domain` assignments.
- `data/sources/combined_paired.py:3438,3467,3490,3497` — collate of `quality_domain` (first-frame-per-video).
- `data/sources/combined_paired.py:1188-1280` — identity-stratified split (`shuffle` and `hash_stable` modes).
- `arena/score_teams_promotion_contract.py:249-280` — `ContractConfig` dataclass and budget defaults.
- `arena/score_teams_promotion_contract.py:430-465` — `_threshold_sort_key` (budget vs legacy paths).
- `arena/score_teams_promotion_contract.py:468-479` — `_promotion_summary_sort_key`.
- `arena/score_teams_promotion_contract.py:505-604` — `score_promotion_contract` orchestration.
- `arena/run_target_domain_validation_sequential.py:312-393` — diagnostic scorecard row builder.
- `arena/run_target_domain_validation_sequential.py:436-491` — promotion contract launch.
- `arena/build_teams_target_domain_manifest.py:212-258` — `_parse_discovery_metadata` (identity_key construction).
- `arena/build_teams_target_domain_manifest.py:475-480` — dev/lockbox split by `_stable_fraction(identity_key, split_seed)`.
- `arena/manifests/teams_target_domain_manifest_2026-04-06_frozen.json` — frozen 7,276-video manifest.
- `arena/target_domain_suites.teams_promotion_contract_2026-04-17.yaml` — 8-suite promotion contract.
- `arena/checkpoint_maps/teams_target_domain.p10_partial_2026-04-26.yaml` — currently-active checkpoint map (selects by `value_composite_*`).
- `arena/launch_teams_promotion_contract.sh` — the canonical promotion-contract launcher.
- `trainer/trainer.py:113-135` — value_composite source-list constants.
- `trainer/trainer.py:138-218` — `_find_threshold_for_mean_fpr` bisection.
- `trainer/trainer.py:220-295` — `_compute_value_composite`.
- `trainer/trainer.py:1853-2059` — `_run_validation` flow.
- `trainer/trainer.py:1959-2054` — ood_composite checkpoint selection.
- `trainer/trainer.py:2056-2135` — value_composite checkpoint selection.
- `trainer/trainer.py:2680-2810` — final_eval reload of both ckpt tracks.
- `trainer/trainer.py:2830-3160` — OOD monitoring epoch.
- `trainer/trainer.py:3236-3258` — `videos_report.csv` writer.
- `experiments/phase2_round13/R13_P10_GRL_baseline.yaml:71-77` — GRL knobs in P10.
- `experiments/phase2_round13/R13_P10_GRL_baseline.yaml:78-81` — value_composite knobs.
- `experiments/phase2_round13/R13_P10_GRL_baseline.yaml:246-340` — OOD monitoring config.
