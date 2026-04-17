# Repo System Map: Training and Inference for Teams-Focused Frame Detection

## Executive Take

For the current Microsoft Teams target, the real system is not the clean config architecture described in some refactor docs. The active path is:

`entrypoint.sh` -> `train_sweep.py` -> manual config patching around W&B flattening -> `create_data_pipeline()` -> `combined_paired.py` -> `EffortDetector` -> `Trainer` validation/checkpoint logic.

That path is powerful for fast ablations, but it is fragile in three ways that matter directly for your next experiment:

1. Config intent can silently drift between YAML, W&B, and the effective runtime config.
2. Target-domain selection is split across incompatible objectives: holdout AUC during training, optional OOD-composite for checkpointing, then fixed-threshold scorecards in arena.
3. Backbone and augmentation behavior depend on multiple partially duplicated registries and override paths, which is dangerous when changing to a new OpenCLIP B/16 checkpoint under time pressure.

## Most Important System Risks

- **Config propagation is only partially fail-fast.** `train_sweep.py` contains a large manual bypass because W&B flattens nested YAML, but explicit mismatch checks only exist for the quality-domain-head fields. Other nested keys can still land wrong without raising. See [train_sweep.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/train_sweep.py#L172) and [config_helpers.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/utils/config_helpers.py#L796).
- **Augmentation keys can be silently ignored.** The `quality_targeted_family` router only forwards YAML keys that match preset keys exactly. The documented `gamma_up_p` / `gamma_up_range` mismatch in `R13_A_trackA_teams_enhanced.yaml` is a real example of “looks configured, is not active.” See [combined_paired.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/data/sources/combined_paired.py#L3592), [SPATIAL_AUGMENTATION_PROPOSAL_2026-04-14.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/SPATIAL_AUGMENTATION_PROPOSAL_2026-04-14.md#L196), and [R13_A_trackA_teams_enhanced.yaml](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml#L60).
- **Backbone changes are fragile because hidden size, output dim, and SVD rank live in different places.** For OpenCLIP B/16, `hidden_size` is treated as output dim, while SVD rank is about the internal transformer dim. The current active registry does not expose your `laion/CLIP-ViT-B-16-laion2B-s34B-b88K` checkpoint as a first-class training option. See [effort_detector.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/detectors/effort_detector.py#L524), [backbone_registry.yaml](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/config/backbone_registry.yaml#L72), and [convert_and_package.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/intel/convert_and_package.py#L9).
- **Training selection and deployment selection are misaligned.** Training primary checkpointing uses holdout AUC; Track A docs prefer `best_ood_composite`; arena scorecards then judge fixed-threshold behavior at `0.5`. This can easily select the “wrong” checkpoint for low-FP Teams use. See [trainer.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/trainer/trainer.py#L1550), [trainer.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/trainer/trainer.py#L2002), [run_target_domain_validation_sequential.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/arena/run_target_domain_validation_sequential.py#L225), and [TRACK_A_HANDOFF_2026-04-06.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/TRACK_A_HANDOFF_2026-04-06.md#L32).
- **There is duplicated and partially dead infrastructure.** Checkpointing exists both inside `trainer.py` and in `trainer/mixins/checkpointing.py`; a `config_system/` exists with tests, but the active training path still uses `utils/config_helpers.py`. That increases the odds of reading the wrong abstraction and running the wrong experiment. See [trainer.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/trainer/trainer.py#L571), [checkpointing.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/trainer/mixins/checkpointing.py#L21), and [checklist.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/checklist.md#L98).

## 1. Active Entrypoints and What Actually Runs

The default container entrypoint still launches `train_sweep.py`, not the newer config-system abstractions. See [entrypoint.sh](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/entrypoint.sh#L5).

The active training job path is:

1. `entrypoint.sh` injects `--param-config` if a GCS YAML is provided, then calls `python -u train_sweep.py ...`. See [entrypoint.sh](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/entrypoint.sh#L56).
2. `train_sweep.py` loads defaults, detector config, train config, and dataloader config through `load_base_configs()`. See [train_sweep.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/train_sweep.py#L126) and [config_helpers.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/utils/config_helpers.py#L54).
3. The experiment YAML is loaded as `single_cfg`, passed to W&B, then manually re-applied for nested fields that W&B flattens away. See [train_sweep.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/train_sweep.py#L133) and [train_sweep.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/train_sweep.py#L172).
4. `apply_all_wandb_overrides()` then mutates both `config` and `data_config`, and also forces `metric_scoring='auc'`. See [config_helpers.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/utils/config_helpers.py#L796).
5. `train_sweep.py` merges `data_config` into the main config with a shallow `config.update(data_config)`, then builds the unified data pipeline. See [train_sweep.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/train_sweep.py#L413) and [train_sweep.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/train_sweep.py#L418).
6. The detector is constructed from the runtime config, the `Trainer` is created, and an optional base checkpoint is loaded afterward. See [train_sweep.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/train_sweep.py#L836).

The important negative fact: the documented `config_system/` is not the live execution path. The repo’s own checklist still says “Update `train_sweep.py` to use new loader” is unchecked. See [checklist.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/checklist.md#L98). That means any researcher reading `config_system/` as the source of truth will misunderstand current behavior.

## 2. Effective Config Flow, and Why It Is Fragile

The real config hierarchy is not just “defaults -> experiment -> W&B.” It is:

1. `defaults.yaml` + detector YAML + train YAML + dataloader YAML, merged by shallow `dict.update()`. See [config_helpers.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/utils/config_helpers.py#L54).
2. `single_cfg` from experiment YAML.
3. `wandb.init(config=single_cfg)`.
4. Manual pre-W&B direct application for nested keys such as `augmentation`, `combined_paired`, `backbone`, and `checkpointing`. See [train_sweep.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/train_sweep.py#L181).
5. `apply_all_wandb_overrides()` for flat and semi-flat keys. See [config_helpers.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/utils/config_helpers.py#L796).
6. Post-W&B `gcs_assets` override and base-checkpoint path override. See [train_sweep.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/train_sweep.py#L312) and [train_sweep.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/train_sweep.py#L365).
7. Final shallow `config.update(data_config)`. See [train_sweep.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/train_sweep.py#L413).
8. For validation and some inference flows, checkpoint `model_config` can override the runtime config again before model creation. See [validate_custom_sources.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/validate_custom_sources.py#L27) and [simple_inference.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/simple_inference.py#L172).

Critical implications:

- `load_base_configs()` is a shallow merge, so nested maps are replaced wholesale, not deep-merged. That is fast, but brittle for partial nested overrides.
- `apply_wandb_dataset_methods_override()` only actively handles `manifest` data source. `combined_paired` survives because `train_sweep.py` manually copies `single_cfg['combined_paired']` into `data_config` before the helper runs. See [config_helpers.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/utils/config_helpers.py#L460) and [train_sweep.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/train_sweep.py#L216).
- The only explicit fail-fast mismatch checks are for the quality head fields. Backbone, augmentation, combined-paired subtrees, and some sampling settings do not get the same protection. See [train_sweep.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/train_sweep.py#L283).

For experimentation, this is convenient because one script accepts almost everything. For reliability, it means you cannot trust a YAML by inspection alone; you have to trust the effective runtime logs.

## 3. Data Path for Teams: `combined_paired`

For your Teams-focused work, `combined_paired` is the center of gravity. It is not just a loader; it is the mechanism that decides source mixture, identity sampling, holdout definition, augmentation routing, quality-domain labels, and OOD monitoring.

The active data trace is:

`experiment YAML` -> `train_sweep.py` direct-copy of `combined_paired` -> `create_data_pipeline()` -> split/holdout in `combined_paired.py` -> optional `quality_targeted_family` router -> iterable dataset per source -> `combined_paired_collate_fn()` -> detector forward/loss.

Key runtime behavior:

- `CombinedPairedIterableDataset` groups by identity and can sample one method per identity per epoch, which is directly relevant to your generalization goal. See [combined_paired.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/data/sources/combined_paired.py#L1702).
- Teams passthrough samples are loaded from GCS JPG pairs and emitted as `source='deeplive_teams'` with `quality_domain=1`, which is exactly the branch closest to your target deployment domain. See [combined_paired.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/data/sources/combined_paired.py#L2385).
- External unpaired reals are also supported and go through the same transform routing. See [combined_paired.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/data/sources/combined_paired.py#L2461).
- Holdout can be by identity or by method. Method-holdout is much more relevant to “new manipulation family” stress, but it explicitly allows identity overlap. See [combined_paired.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/data/sources/combined_paired.py#L3195).
- OOD monitoring can build an external loader at startup and exclude training identities. That makes it the closest thing to a domain-generalization signal inside the training loop. See [combined_paired.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/data/sources/combined_paired.py#L3393).

This is also where a lot of hidden dependencies live:

- The quality-domain mapping in `combined_paired.py` is supposed to stay in sync with `QualityDomainHead.DOMAIN_MAP`, but the mappings are not literally identical anymore. `combined_paired.py` includes extra source names such as `visomaster_enhanced`, `visomaster_res_variant`, and `visomaster_teams_enhanced` that do not appear in the detector map. See [combined_paired.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/data/sources/combined_paired.py#L62) and [effort_detector.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/detectors/effort_detector.py#L250). This is survivable because the dataset emits integer labels directly, but it is a maintenance smell.
- `combined_paired_collate_fn()` hardcodes CLIP mean/std instead of reading backbone normalization from config. See [combined_paired.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/data/sources/combined_paired.py#L2522). For CLIP-like backbones this is probably harmless today, but it is exactly the kind of hidden assumption that breaks when experimentation broadens.

## 4. Augmentation Routing and the Specific Silent-Failure Pattern

The Teams-relevant augmentation path currently goes through `_create_combined_transform()`. See [combined_paired.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/data/sources/combined_paired.py#L3512).

Important behaviors:

- `augmentation.version: quality_targeted_family` dispatches to a metadata-aware router that sees `label`, `source`, and `method`. This is the right place for domain-family-aware augmentation.
- YAML overrides are not free-form. The router collects `preset_overrides` by filtering keys against the preset dictionary. Unknown keys are ignored silently. See [combined_paired.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/data/sources/combined_paired.py#L3592).
- The repo already documents one concrete failure: `gamma_up_p` and `gamma_up_range` in `R13_A_trackA_teams_enhanced.yaml` appear intended to affect the context-variation block, but the valid keys are `context_variation_gamma_up_p` and `context_variation_gamma_up_range`. See [SPATIAL_AUGMENTATION_PROPOSAL_2026-04-14.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/SPATIAL_AUGMENTATION_PROPOSAL_2026-04-14.md#L196) and [R13_A_trackA_teams_enhanced.yaml](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/experiments/phase2_round13/R13_A_trackA_teams_enhanced.yaml#L79).

For your next experiment, this matters more than it looks. If a Teams augmentation ablation appears neutral, it may be because the key never actually propagated.

## 5. EFFORT / SVD Adaptation and Where Backbone Choice Really Bites

`EffortDetector` is the active model definition. It does three things that matter here:

1. Builds the backbone.
2. Wraps it with either a linear head or ArcFace head.
3. Adds EFFORT regularization and the optional quality-domain adversarial head.

See [effort_detector.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/detectors/effort_detector.py#L280).

Backbone-specific observations:

- Hidden size is taken from `config['backbone']['hidden_size']` if present. If that key is wrong, the detector trusts it. See [effort_detector.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/detectors/effort_detector.py#L524).
- For OpenCLIP / LAION sources, `_build_openclip_backbone()` uses `openclip_model` or `model_name`, and `openclip_pretrained` or `pretrained`. This means the experiment YAML can drive your new backbone even without a registry entry, but then you are relying on the loose override path rather than a curated registry path. See [effort_detector.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/detectors/effort_detector.py#L642).
- OpenCLIP SVD coverage is configurable: fused `in_proj`, selective block indices, and optional MLP coverage. See [effort_detector.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/detectors/effort_detector.py#L723). This is a major experimental lever for stability and generalization.
- For OpenCLIP B/16 DataComp, the repo consistently treats the output feature size as `512`, while documenting that the internal transformer `embed_dim` is `768` and the recommended rank is `767`. See [backbone_registry.yaml](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/config/backbone_registry.yaml#L74).

Your chosen backbone issue:

- The checked-in active backbone registry in `defaults.yaml` / `backbone_registry.yaml` does **not** expose `laion/CLIP-ViT-B-16-laion2B-s34B-b88K` as a first-class selectable training backbone. See [defaults.yaml](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/config/defaults.yaml#L54) and [backbone_registry.yaml](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/config/backbone_registry.yaml#L72).
- The repo does know about that checkpoint in packaging utilities. See [convert_and_package.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/intel/convert_and_package.py#L9).
- Therefore, right now, your backbone change is supported more by convention than by a stable training-system contract.

This is the practical risk: if the experiment YAML sets `source: laion`, `model_name: ViT-B-16`, and `pretrained: laion2b_s34b_b88k`, but the registry / asset path / hidden-size fields are not coherent, the model may still build, yet not from the backbone you think you are testing.

One more concrete fragility: in `forward()`, the detector logs a feature-dimension mismatch once, but does not immediately abort there. See [effort_detector.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/detectors/effort_detector.py#L1118). That is good for debugging visibility, but not strong enough as a safety rail for expensive runs.

## 6. Losses and Metrics: Data -> Model -> Loss -> Validation

The full active path is:

`combined_paired_collate_fn()` emits `image`, `label`, `method_id`, `quality_domain` -> `EffortDetector.forward()` flattens `[B, T, C, H, W]` into frames, extracts backbone features, applies head, returns frame probabilities -> `get_losses()` computes classification loss + EFFORT regularization + optional quality-domain adversarial loss.

See [combined_paired.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/data/sources/combined_paired.py#L2522), [effort_detector.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/detectors/effort_detector.py#L1097), and [effort_detector.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/detectors/effort_detector.py#L796).

Loss details that matter for Teams generalization:

- OpenCLIP fused attention projections are counted specially in regularization via `SVDInProjLinear`, so OpenCLIP and HuggingFace CLIP do not regularize through exactly the same module topology. See [effort_detector.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/detectors/effort_detector.py#L857).
- The optional quality head is only active during training, and it requires the dataloader to supply `quality_domain`. If labels or logits are missing and `quality_domain_require_labels=True`, the run fails. See [effort_detector.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/detectors/effort_detector.py#L877).

This is useful for your objective because the quality head is one of the few built-in mechanisms explicitly aimed at invariance to capture/domain quality. But it also means the data path and the label map must be trusted, not just the model config.

## 7. Checkpoint Selection and How Target-Domain Metrics Actually Influence Decisions

Inside `Trainer`, evaluation order is:

1. `val_in_dist`
2. `val_holdout`
3. extract the in-dist EER threshold
4. evaluate holdout at that in-dist threshold
5. compute a legacy unified threshold across validation pools
6. run lesson gate
7. run OOD monitoring
8. optionally compute OOD-composite checkpoint ranking

See [trainer.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/trainer/trainer.py#L1550).

What actually drives checkpoint saving:

- The primary checkpoint lane is `val_holdout` and uses `current_metric = overall_metrics.get(self.metric_scoring)`. See [trainer.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/trainer/trainer.py#L2002).
- `metric_scoring` is forcibly set to `auc` in `apply_all_wandb_overrides()`. See [config_helpers.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/utils/config_helpers.py#L836).
- OOD monitoring is logged separately and can support a second checkpoint lane based on harmonic mean of holdout AUC and OOD AUC. See [trainer.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/trainer/trainer.py#L1658) and [trainer.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/trainer/trainer.py#L2222).
- Track A handoff docs explicitly tell the operator to use the `best_ood_composite` checkpoint, not the default best-holdout checkpoint. See [TRACK_A_HANDOFF_2026-04-06.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/TRACK_A_HANDOFF_2026-04-06.md#L32).

Then the final decision layer changes objective again:

- Arena scorecards evaluate per-suite metrics at threshold `0.5`, not at the learned EER threshold and not by AUC. Real-only suites score `real_fpr_at_0p5`, fake-only suites score `fake_recall_at_0p5`, and mixed suites score `accuracy_at_0p5`. See [run_target_domain_validation_sequential.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/arena/run_target_domain_validation_sequential.py#L225).

This is the single biggest systems-level reason you can get a strong “best checkpoint” offline and still be unhappy with Teams low-FP behavior.

## 8. Inference and Revalidation Are Not Symmetric

The repo has two important downstream consumers:

- `validate_custom_sources.py` restores checkpoint `model_config` before model creation and also merges checkpoint `clip_backbone` assets if needed. See [validate_custom_sources.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/validate_custom_sources.py#L27).
- `simple_inference.py` also restores model config from the checkpoint, but operationally it still defaults to threshold `0.5`, and when `--recrop` is false it always resizes to `224x224` directly. See [simple_inference.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/simple_inference.py#L161) and [simple_inference.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/simple_inference.py#L226).

For a Teams detector where low FP matters, `validate_custom_sources.py` is the more trustworthy path for model comparison. `simple_inference.py` is fine for spot checks, but it is not a reliable proxy for deployment selection.

## 9. Duplicated Logic, Dead Paths, and Maintenance Hazards

The repo currently contains three kinds of structural duplication:

- **Checkpointing duplication.** `Trainer` has its own `load_ckpt`, `_validate_model_config`, `compute_model_checksum`, and `save_ckpt`, while `trainer/mixins/checkpointing.py` contains another implementation of the same responsibilities. See [trainer.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/trainer/trainer.py#L571) and [checkpointing.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/trainer/mixins/checkpointing.py#L126). This is a classic “which version is authoritative?” risk.
- **Config-system duplication.** `config_system/` exists and is tested, but the live training path still depends on `utils/config_helpers.py` and manual patching in `train_sweep.py`. See [checklist.md](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/checklist.md#L104).
- **Legacy script duplication.** `train_deeplive_v2.py` is marked deprecated and explicitly tells users to run `train_sweep.py` instead. See [train_deeplive_v2.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/train_deeplive_v2.py#L1).

These are not just cleanup issues. They increase experiment risk because repo inspection can easily focus on the wrong abstraction layer.

## 10. What This Means for Next Experiments

For **fast experimentation**, the current system is good at:

- trying new `combined_paired` source mixes,
- changing family weights and holdout mode,
- turning OOD monitoring on quickly,
- switching OpenCLIP variants if the experiment YAML and asset paths are correct,
- logging useful data-mix summaries to W&B.

For **reliable production-style selection**, the current system is weak because:

- the effective config is assembled through multiple override stages,
- the “best checkpoint” definition changes by stage,
- threshold choice is not unified between training, validation, arena, and simple inference,
- some silent config-drop patterns already exist in the augmentation path,
- backbone selection is not yet fully productized for your new LAION B/16 checkpoint.

## Researcher Guidance: What To Trust, What To Verify

Trust these as the main sources of truth:

- `train_sweep.py` runtime logs and W&B summaries for the effective config and data mixture. See [train_sweep.py](/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/train_sweep.py#L475).
- `combined_paired.py` for understanding what data the model actually sees.
- `trainer.py` for deciding what “best checkpoint” means in a given run.

Verify manually before expensive runs:

- backbone asset path, `hidden_size`, and OpenCLIP `pretrained` tag when using `laion/CLIP-ViT-B-16-laion2B-s34B-b88K`;
- whether each augmentation key is part of the allowed preset key set;
- whether the run is being selected by holdout AUC or `best_ood_composite`;
- whether the checkpoint that wins arena is also acceptable under the threshold you will actually use for Teams.

If the immediate goal is the next best Teams experiment rather than long-term cleanup, the strongest system-level move is not a broad refactor. It is to make the selection lane explicit and consistent: one backbone contract, one verified augmentation contract, and one checkpoint-ranking contract tied to low-FP Teams behavior rather than generic holdout AUC.
