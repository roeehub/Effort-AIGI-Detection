# Phase 2 HEAD — face-pool head-only retrain on Slot A v2 step3500

Status as of 2026-05-22 PM CEST: smoke validated (cancelled for overrun before
yaml-bug fix landed); full training run RUNNING on Vertex AI us-east1.
Scoring + final verdict will land here after the full job reaches
`JOB_STATE_SUCCEEDED`.

This document records the FACTS only — what we built, what ran, what numbers
came back. The judgment call (deploy / iterate / abort) lives in
`AGENT_PROPOSAL_2026-05-22.md`.

> Banned-word policy: this FACTS doc uses no opinion verbs about outcomes
> (no "succeeds", "fails", "wins", "loses", "promotes", "deployment-grade",
> "kill", "best", "worst", "confirmed", "refuted"). Token matches in the
> body are literal API enums (`JOB_STATE_SUCCEEDED`), GCS paths
> (`best_checkpoints/`, `first_best_effort_*`), or meta-references to the
> AGENT_PROPOSAL decision options. All interpretation lives in the
> AGENT_PROPOSAL doc.

---

## Code edits

| File | Change |
|---|---|
| `detectors/effort_detector.py` | Added `face_pool_readout` parameter to `OpenCLIPVisionModelWrapper.__init__`; wires through to `_build_openclip_backbone` (reads top-level yaml key). When enabled, hooks the configured resblock, captures the full 197-token sequence, in `forward` drops CLS, index-selects the centered 7x7 patches (radius=3), mean-pools to (B, 768), applies `visual.ln_post` then `@ visual.proj` (768 -> 512). Backwards compatible — disabled by default. |
| `train_sweep.py` | Added `face_pool_readout` to the W&B-flattening bypass allowlist (line ~417, after `head_only_retrain`). Necessary because `wandb.init(config=single_cfg)` flattens nested dicts; without re-apply the detector would read `None` and silently fall back to CLS pool. |
| `experiments/phase2_round13/R13_FACE_POOL_HEAD_2026-05-25.yaml` | New full-launch yaml. Clones Slot A v2 ancestor (`R13_T5C_ANCHOR_AWARE_2026-05-16.yaml`). Single-lever delta: enables `face_pool_readout {enabled: true, layer: 11, subgrid_radius: 3}` AND `head_only_retrain {enabled: true, reinit_head: true}`. seed=9914. max_train_steps=2500. periodic_saves=[100, 500, 1000, 1500, 2500]. |
| `experiments/phase2_round13/R13_FACE_POOL_HEAD_SMOKE_2026-05-25.yaml` | 200-step smoke variant for pre-launch verification. |
| `analysis/phase_2_head_2026-05-22/` | New analysis folder for sentinels, smoke verdict, scoring scripts, FACTS + AGENT_PROPOSAL. |

### Yaml diff vs Slot A v2 ancestor (R13_T5C_ANCHOR_AWARE_2026-05-16.yaml)

Single-lever deltas (added lines):

```yaml
face_pool_readout:
  enabled: true
  layer: 11
  subgrid_radius: 3

head_only_retrain:
  enabled: true
  reinit_head: true
  trainable_param_substrings:
    - "head.weight"
    - "head.bias"
```

Other deltas (mechanical):
- `name`, `seed: 9914`, `wandb.tags` updated for the new packet
- `total_training_steps: 2500` + `max_train_steps: 2500` (down from 3500 for shorter head-only retrain)
- `lr_scheduler_warmup_steps: 250` (proportional to total_training_steps)
- `periodic_saves.step_list: [100, 500, 1000, 1500, 2500]` (down from 3500)
- `combined_paired.split_seed: 9914` and `external_training_reals[].identity_split_seed: 9914`
- All other levers (anchor_aware, multi_axis_grl, augs, sampling) inherited unchanged

---

## Smoke test (cancelled)

| Field | Value |
|---|---|
| Vertex job id | `4897381264362831872` |
| W&B run id | `s1hihyi7` |
| W&B URL | `https://wandb.ai/dtect-vision/effort-r13-phase2/runs/s1hihyi7` |
| Image | `1.3.297` (face_pool wrapper, NO `max_train_steps` fix) |
| Region | us-east1 |
| Pending duration | ~10 min |
| Running duration | ~57 min (cancelled at 13:52:03 UTC) |
| Steps reached before cancel | ~500 (step 100 and step 200 ckpts saved) |
| Estimated GPU cost | ~$2.94 (within $5 threshold) |

### Smoke pass criteria (all met)

| Criterion | Verdict | Evidence |
|---|---|---|
| Loss decreasing | met | step 100 train_loss=2.7767; step 401 train_loss=2.4407 — monotonically decreasing on sampled batches |
| head_only_retrain freeze | met | trainable=1,026 of 235,488,010 params (0.0004%) — matches nn.Linear(512, 2) exactly |
| head reinit | met | log: "re-initialised nn.Linear head with PyTorch default Kaiming uniform" |
| face_pool wrapper active | met | log: "OpenCLIPVisionModelWrapper: face_pool_readout ACTIVE at block 11 \| subgrid_radius=3 (centered 7x7 of 14x14, 49 face patches)" |
| first val holdout_auc ≥ 0.40 | met | holdout_auc=0.9896 at step 100; far above floor |
| no NaN losses | met | all reported losses finite; anchor_aware composite logs NaN but anchor_mean=0.4574 valid (composite uses a separate stat that initializes NaN) |

### Smoke cancellation reason

Discovered after launch that my smoke yaml used `total_training_steps: 200`
WITHOUT `max_train_steps: 200`. In `trainer/trainer.py:567`,
`max_train_steps` is the actual training-loop cap;
`total_training_steps` is only consumed by the LR scheduler (cosine target).
With `nEpochs: 2` and a 2987-batch epoch, the smoke would have run for
~6.4 h instead of the planned ~10-15 min — projected ~$18-20 cost vs the
$3 plan. Cancelled at 57 min wall ($2.94 actual) after periodic ckpts at
step 100 + 200 were saved. The bug fix landed in BOTH smoke and full
yamls before the full launch (image rebuilt as `1.3.298`).

### Smoke artifacts (kept)

```
gs://training-job-outputs/best_checkpoints/s1hihyi7/periodic_effort_20260522_step100_auc0.9896_eer0.0490.pth
gs://training-job-outputs/best_checkpoints/s1hihyi7/periodic_effort_20260522_step200_auc0.9895_eer0.0490.pth
gs://training-job-outputs/best_checkpoints/s1hihyi7/value_composite_effort_20260522_step100_auc0.9896_eer0.0490.pth
gs://training-job-outputs/best_checkpoints/s1hihyi7/top_n_effort_20260522_step100_auc0.9896_eer0.0490.pth
gs://training-job-outputs/best_checkpoints/s1hihyi7/first_best_effort_20260522_ep1_auc0.9896_eer0.0490.pth
```

---

## Full training run

| Field | Value |
|---|---|
| Vertex job id | `3087778639090024448` |
| W&B run id | `kwhju7im` |
| W&B URL | `https://wandb.ai/dtect-vision/effort-r13-phase2/runs/kwhju7im` |
| Image | `1.3.298` (face_pool wrapper + max_train_steps fix) |
| Region | us-east1 |
| GPU | NVIDIA_TESLA_A100 x 1 |
| Submitted | 2026-05-22T13:52:00Z |
| RUNNING | 2026-05-22T13:52:21Z (no pending queue) |
| Total training steps | 2500 (max_train_steps=2500) |
| Periodic saves attempted | [100, 500, 1000, 1500, 2500] |
| Periodic saves expected to fire | [500, 1000, 1500, 2500] — step 100 will be skipped because `evaluate_every_steps: 250` means no eval at step 100 → trainer's `holdout_auc_p is None` short-circuits the periodic save (`trainer/trainer.py:2761`). The smoke run `s1hihyi7` step 100 ckpt covers this data point in spirit (same base + yaml, with the smoke's `evaluate_every_steps: 100` triggering the eval). |
| Status | _pending — populated post-run_ |

### Scorecard (per-ckpt)

_Pending — will be populated after full reaches `JOB_STATE_SUCCEEDED` and the
face-pool scorer runs on each periodic ckpt against the 9 promotion-contract
suites, then applied through `arena/score_teams_promotion_contract.py` under
both lex and composite λ=1.0 policies._

Expected columns:

| ckpt | step | val_holdout_auc | teams_real_all_dev fpr | teams_real_all_lockbox fpr | teams_fake_all_lockbox recall | visomaster_enhanced_macro_dev recall | composite (λ=1.0) | lex passed |
|---|---|---|---|---|---|---|---|---|

### Comparison baselines (already on file)

These come from `analysis/face_pool_scorecard_2026-05-22/RESULTS_FACTS_2026-05-22.md`:

| ckpt + readout | lockbox_real_fpr | lockbox_fake_recall | viso_enhanced_dev recall | composite λ=1.0 |
|---|---|---|---|---|
| Slot A v2 step3500 + CLS pool | 0.0191 | 0.6877 | 0.1673 | 0.331 |
| Slot A v2 step3500 + face-pool inference | 0.0154 | 0.7668 | 0.0673 | 0.249 |

---

## Reproducibility

- Code: `detectors/effort_detector.py` (face_pool_readout) + `train_sweep.py` (allowlist)
- Yaml: `experiments/phase2_round13/R13_FACE_POOL_HEAD_2026-05-25.yaml`
- Base ckpt: `gs://training-job-outputs/best_checkpoints/hp35c51p/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth`
- Image: `us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:1.3.298`
- Launcher: `./scripts/launch/launch_experiment.sh -y effort-r13-phase2 us-east1 experiments/phase2_round13/R13_FACE_POOL_HEAD_2026-05-25.yaml`
