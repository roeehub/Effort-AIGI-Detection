# Evaluation Contract And Shortlist

## 1. What this contract is trying to decide

The selection problem is not "highest training AUC." It is:

> Which available checkpoint gives the lowest false-positive behavior on real Microsoft Teams participants, while still retaining acceptable recall on direct Teams fakes and enhanced fake stress slices?

That requires freezing three different roles that were blurred together in prior round summaries:

- `training-selection role`: choose checkpoints during training
- `promotion role`: choose the candidate that is safest for live Teams deployment
- `diagnostic regression role`: make sure a candidate did not collapse on broader source families

## 2. Established contract mismatch in current repo behavior

### Training-side checkpointing

Current training does **not** select checkpoints on the live Teams objective.

- Primary best-checkpoint logic is driven by `val_holdout` `metric_scoring` (usually AUC) in `trainer/trainer.py`.
- Secondary OOD checkpointing uses the harmonic mean of holdout AUC and OOD AUC.
- The computed in-dist EER threshold and `ood/at_indist/*` metrics are logged, but they do not drive primary selection.

### Arena-side scorecarding

Current target-domain scorecarding in `arena/run_target_domain_validation_sequential.py` uses a fixed threshold of `0.5` and suite-specific summary metrics:

- real-only suites: `real_fpr_at_0p5`
- fake-only suites: `fake_recall_at_0p5`
- mixed suites: `accuracy_at_0p5`

That means the repo is currently doing this:

- shortlist by holdout/OOD AUC during training
- judge Teams deployment slices at threshold `0.5`

This is the central selection mismatch for Round 2.

## 3. Authoritative suite set for this round

### Promotion-authoritative suites

These are the suites that should decide promotion, in this order of importance.

| Suite | Count | Metric | Role |
| --- | ---: | --- | --- |
| `teams_real_all_lockbox` | 1361 real | FPR | Primary deployment-safety gate |
| `teams_real_all_dev` | 3253 real | FPR | Threshold fitting / real-domain ranking |
| `teams_real_poor_quality_dev` | 923 real | FPR | Stress lane for poor capture quality |
| `teams_real_lighting_extreme_dev` | 1401 real | FPR | Stress lane for lighting sensitivity |
| `teams_fake_all_dev` | 2409 fake | Recall/TPR | Direct Teams fake coverage |
| `visomaster_enhanced_macro_dev` | 550 fake | Recall/TPR | Enhanced clean fake stress lane |
| `deeplive_enhanced_dev` | 545 fake | Recall/TPR | Enhanced DeepLive stress lane |

### Missing but required addition

`teams_fake_all_lockbox` exists in the frozen manifest with `253` fake videos, but it is not present in the current frozen suite YAML. That omission matters.

Round 2 contract recommendation:

- add `teams_fake_all_lockbox` before the next decisive scorecard rerun
- until that rerun exists, treat the current contract as incomplete on the fake-side lockbox axis

### Diagnostic-only suites

These should not decide promotion.

- per-session fake slices such as `teams_capture_cam_test_dev`, `teams_capture_pc_generator_dev`, `teams_capture_test_cam_dev`, `teams_capture_noyn_sharker_dev`, `teams_capture_dor_shkedi_dev`
- mixed-source mega-eval additions: `deeplive_all_sources`, `visomaster_all_sources`, `visomaster_enhanced_v2_all`
- training-side `val_holdout/*`, `ood/*`, and `val_primary/ood_composite`

Use them only to detect obvious regressions or broken generalization.

## 4. Authoritative threshold policy

### Recommended policy

The authoritative threshold is **not** `0.5`.

For each checkpoint:

1. Fit or sweep a checkpoint-specific threshold on the dev portion of the frozen Teams suite.
2. Optimize lexicographically:
   - first minimize real Teams FPR on `teams_real_all_dev`
   - then minimize worst-slice FPR across `teams_real_poor_quality_dev` and `teams_real_lighting_extreme_dev`
   - then maximize fake recall on `teams_fake_all_dev`, `visomaster_enhanced_macro_dev`, and `deeplive_enhanced_dev`
3. Freeze that threshold before looking at lockbox.
4. Promote only from lockbox results at that frozen threshold.

This is the correct deployment-selection contract for this repo stage.

### Backup policy if calibrated reruns are not yet available

If the only available artifacts remain the current threshold-`0.5` scorecards:

1. Rank first by `teams_real_all_lockbox` FPR.
2. Break ties with `teams_real_all_dev`, then `teams_real_poor_quality_dev`, then `teams_real_lighting_extreme_dev`.
3. Use `teams_fake_all_dev`, `visomaster_enhanced_macro_dev`, and `deeplive_enhanced_dev` only as tie-breakers after the real-side ordering.
4. Treat any result from mixed-source mega-eval as diagnostic only.

This backup policy is inferior, but still better than mixing holdout AUC and `0.5`-threshold Teams tables in one narrative.

## 5. Frozen shortlist for this round

### Keep in the shortlist

- `R12_G_FP32`
- `R13_A_STEP15500`
- `R13_E_BESTSOFAR`
- `R13_FT7_FP32`
- `R13_FT9_FP32`

### De-prioritize as likely duplicates under local runtime truth

- `R13_FT8_FP32`
  - differs from `R13_FT7_FP32` mainly by real-side weight boosts
  - if unpaired `external_training_reals` are truly absent at runtime, this is functionally near-duplicate on paired-object exposure
- `R13_FT10_FP32`
  - same logic relative to `R13_FT9_FP32`

### Why `FT15` / `FT16` / `R13_I` are not in the frozen shortlist yet

- they are configs, not currently frozen finalist checkpoints in the local checkpoint maps
- they are still useful as sampler/curriculum evidence
- they should not be promoted into the shortlist without actual scorecard artifacts

## 6. Older comparisons that are invalid or misleading under this contract

- Comparing `val_holdout` AUC across rounds as if it were the deployment objective.
- Treating training `ood_composite` as the final winner metric.
- Treating threshold `0.5` Teams tables as equivalent to calibrated low-FP deployment behavior.
- Treating `FT7` vs `FT8` or `FT9` vs `FT10` as distinct intervention wins without proving that `external_training_reals` actually loaded.
- Treating earlier `R12_G` vs `TRACK_A_CANDIDATE` comparisons as final, because they exclude the later merged-source finalists and still inherit the fixed-threshold issue.

## 7. Status

- Established:
  - the current repo mixes different checkpoint-selection and deployment-selection objectives
  - the frozen Teams suite is the right promotion base, not mixed-source mega-eval
  - the present shortlist can be reduced to five meaningful candidates under local truth
- Plausible:
  - calibrated per-checkpoint thresholds may change the winner
  - fake lockbox may change the final ordering once added
- Still unknown:
  - whether any current R13 finalist beats `R12_G` on the full calibrated promotion contract
