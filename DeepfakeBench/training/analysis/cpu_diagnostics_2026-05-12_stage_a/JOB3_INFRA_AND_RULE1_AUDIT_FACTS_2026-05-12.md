# CPU Job 3 — Infrastructure audit + AGENT_GUIDE Rule 1 validate-before-suggest grep — FACTS (2026-05-12)

> **Status: factual-only.** Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, unfortunately, remarkably.
> Numbers + tables + cross-references only. Interpretation belongs in `STAGE_A_PLUS_PACKET_PLAN_OPINIONS_2026-05-12.md`.
>
> **Scope**: enumerate existing codebase infrastructure for anchor losses + parameter-efficient FT, and confirm that the three GPU slots proposed in `STAGE_A_PLUS_PACKET_PLAN_OPINIONS_2026-05-12.md` have not been previously tested (AGENT_GUIDE.md Rule 1 — Validate-before-suggest).

---

## 1. Existing loss + auxiliary infrastructure

| Element | File / lines | Notes |
|---|---|---|
| `AnchorAwarePenalty` class | `loss/anchor_aware_penalty.py:37` | Output-mean hinge: `weight * max(0, mean(prob_fake) - target)^2` on cached anchor pool frames. Currently applied only to single `ANCHOR_POOL = "dor-real-webcam-false-flag-no-virtual-bg"` per pool list in `analysis/teams_pool_rescore.py:64`. |
| `cache_anchor_pools_locally` | `analysis/teams_pool_rescore.py:117` | Downloads + indexes frames under `gs://live-fakes-teams-prod/<pool>/` into `cache_dir/<pool>/`. Supports arbitrary list of pools. |
| `_LocalAnchorDataset` | `analysis/teams_pool_rescore.py:177` | Yields `(tensor, metadata)` from a cached pool index for in-step sampling. |
| `MultiAxisGRLBlock` | `detectors/effort_detector.py:370` | Multi-axis adversarial GRL on encoder pooled L11 features. Configurable axes (chronic_flag, is_dor, sharpness_laplacian_high, color_a_approx_dev_high), `hidden_dim` (default 256, T5C uses 1024), `bottleneck_dim` (default 128). |
| `GradientReversalLayer` | `detectors/effort_detector.py:254` | Implements gradient sign-flip for DANN-style adversarial training. |
| `QualityDomainHead` | `detectors/effort_detector.py:266` | Single-axis quality-domain head with GRL (legacy, used by P15). |
| `SVDResidualLinear` | `detectors/effort_detector.py:1850` | Decomposes a linear layer's weight as top-r SVD components (frozen) + trainable residual. Per-axis classes for `out_proj` and `in_proj`. |
| `SVDInProjLinear` | `detectors/effort_detector.py:2207` | SVD-residual variant for attention in-projection (`q_proj`, `k_proj`, `v_proj` decomposed separately). |
| `correlation_penalty` | `loss/correlation_penalty.py` | Existing penalty class (used by PD packet); not a feature-anchor mechanism. |
| `pair_rank_loss` | `loss/contrastive_regularization.py` (or similar) | Existing ranked-pair loss (used by P1 packet). |
| `consistency_loss` | `loss/consistency_loss.py` | Existing; not currently wired into the standard recipe. |

## 2. Trainer wiring of `AnchorAwarePenalty`

| Element | File / lines | Notes |
|---|---|---|
| Loss init | `trainer/trainer.py:671-672` | `self.anchor_aware_penalty = AnchorAwarePenalty(...)`; config block read from `config.get('anchor_aware')`. |
| Loss application | `trainer/trainer.py:1824-1828` | `anchor_loss = self.anchor_aware_penalty.compute(model, device); losses['overall'] += anchor_loss; losses['anchor_aware'] = anchor_loss.detach()`. Applied every training step. |
| Yaml block | per packet's `anchor_aware:` block (see e.g. `experiments/phase2_round13/R13_T5C_T4_BIG_CLASSIFIER_2026-05-11.yaml:165-166` — T5C disables it: `anchor_aware: enabled: false`) | Keys: `enabled`, `weight`, `target_mean_prob`, `samples_per_step`, `pool_names`. |

## 3. Existing loss classes in `loss/`

```
__init__.py
abstract_loss_func.py
am_softmax.py
anchor_aware_penalty.py        ← the only "anchor" loss; output-mean hinge
bce_loss.py
capsule_loss.py
classNseg_loss.py
consistency_loss.py
contrastive_regularization.py
correlation_penalty.py
cross_entropy_loss.py
det_loss.py
id_loss.py
js_loss.py
l1_loss.py
vgg_loss.py
```

## 4. Infrastructure required for the 3 proposed GPU slots

### 4.1 Slot 1 — T5C + L11 feature anchor loss to P8A on 5-identity cohort

| Component | Status | Required work |
|---|---|---|
| Anchor frame cache infra | ✓ existing (`teams_pool_rescore.py`) | Add 5 new pool names: `dor_shkedi_real`, `bla_bla_chow_real`, `roy_d_real`, `xiang_real`, `dor_real` (or 1 combined pool) under `gs://live-fakes-teams-prod/anchor_chronic_5/` |
| Per-frame P8A L11 feature target | ✗ does not exist | New pre-compute script: load P8A ckpt, run forward on anchor frames, save 512-d L11 CLS feature per frame to parquet |
| L11AnchorLoss class | ✗ does not exist | New `loss/l11_anchor_loss.py`: load target feature parquet, sample per-step batch, compute MSE between model's L11 CLS and target |
| Trainer hook | ✗ does not exist | Wire `compute()` call into `trainer.py` like `anchor_aware_penalty.compute()`; access to L11 features requires either (a) returning them from `EffortDetector.forward(..., return_features=True)` or (b) hooking the post-ln_post tensor — option (b) is cleaner because L11 is already accessible to `MultiAxisGRLBlock` |
| Yaml block | n/a | New `l11_anchor:` block with `enabled`, `weight`, `samples_per_step`, `target_parquet`, `lambda_warmup_steps` |

### 4.2 Slot 2 — T5C + per-frame output-distillation anchor on broad real cohort

| Component | Status | Required work |
|---|---|---|
| Anchor frame cache infra | ✓ existing | Extend `pool_names` to include healthy-real pools (e.g., `dor-real-laptop-correct-*`, `roee-real-windows-laptop-correct`) + curated chronic-5 pools |
| Per-frame P8A `prob_fake` target | ✗ does not exist | New pre-compute script: load P8A ckpt, run forward on anchor frames, save scalar prob_fake per frame to parquet |
| OutputDistillAnchor class | ✗ does not exist directly | Extend `AnchorAwarePenalty`: add `target_distribution_mode` switch that, instead of hinge on mean, computes per-frame BCE between model's `prob_fake` and stored target |
| Trainer hook | ✓ existing (`AnchorAwarePenalty.compute()` already wired) | No change needed if extended in-place |
| Yaml block | ✓ existing schema | Add `mode: per_frame_distillation`, `target_parquet`, `loss_fn: bce|mse|kl` keys |

### 4.3 Slot 3 — T3_SLOT1 base + 1024 classifier (yaml-only compose)

| Component | Status | Required work |
|---|---|---|
| Init from T3_SLOT1_step1500 | ✓ existing (T6, T7 already init from T3_SLOT1) | Just change `gcs_base_checkpoint` to T3_SLOT1_step1500 path |
| `hidden_dim: 1024` in `multi_axis_grl` block | ✓ existing (T5C uses 1024) | Copy block from T5C yaml |
| Data filter (T3 keep-list) | ✓ existing (T3, T6, T7 all use) | Already in inherited config |
| No new code | n/a | Yaml-only fork |

## 5. AGENT_GUIDE Rule 1 grep audit

Per `AGENT_GUIDE.md` §"Rule 1 — Validate-before-suggest": before proposing any packet with a "has never been tested" hypothesis, grep for the key toggle in `experiments/phase2_round13/*.yaml` and the registered-runs config, and search MEMORY.md.

### 5.1 Has L11 feature anchor / feature-level distillation been tried?

Command run: `grep -lriE "l11_anchor|feature_anchor|feature.distill|kd_loss|kd loss|representation.loss|teacher" experiments/phase2_round13/`

| Pattern | Matches | Disposition |
|---|---:|---|
| `l11_anchor` | 0 | never tried |
| `feature_anchor` | 0 | never tried |
| `feature_distill` | 0 | never tried |
| `kd_loss` | 0 | never tried |
| `representation_loss` | 0 | never tried |
| `teacher` | 0 | never tried |

### 5.2 Has per-frame output distillation been tried?

Command run: `grep -lriE "output.distill|output_anchor|per_frame_target|prob_target|reference_prob|kl.divergence|kld_loss" experiments/phase2_round13/`

| Pattern | Matches | Disposition |
|---|---:|---|
| `output_distill` | 0 | never tried |
| `output_anchor` | 0 | never tried |
| `per_frame_target` | 0 | never tried |
| `prob_target` | 0 | never tried |
| `reference_prob` | 0 | never tried |
| `kl_divergence` | 0 | never tried |
| `kld_loss` | 0 | never tried |

### 5.3 Has T3_SLOT1 init + multi-axis GRL `hidden_dim: 1024` been composed?

Commands run:
- `grep -l "T3_SLOT1\|t3_slot1\|9601" experiments/phase2_round13/*.yaml`
- `grep -l "hidden_dim: 1024" experiments/phase2_round13/*.yaml`

| Yamls matching T3_SLOT1 init | Yamls with `hidden_dim: 1024` | Intersection |
|---|---|---|
| `R13_T3_SLOT1_DROP_HIGH_IQ_TEAMS_REALS_2026-05-09.yaml` (T3 itself) | `R13_T5C_T4_BIG_CLASSIFIER_2026-05-11.yaml` (P8A init) | `R13_T5C_T4_BIG_CLASSIFIER_2026-05-11.yaml` matches both greps but T5C inits from `9lmvb5b4` (P8A), not from T3_SLOT1 ckpt — the T3_SLOT1 match is only in yaml comments referencing the inherited DATA filter. |
| `R13_T6_T3_PLUS_JITTER_2026-05-11.yaml` (T6: T3 SLOT1 init + jitter, NO GRL) | | |
| `R13_T4_MULTI_AXIS_GRL_2026-05-10.yaml` (T4: P8A init, T3 data filter) | | |
| `R13_T7_T4_PLUS_JITTER_2026-05-11.yaml` (T7: T4 init + jitter) | | |
| `R13_T5C_T4_BIG_CLASSIFIER_2026-05-11.yaml` (T5C: P8A init, T3 data filter, GRL hidden_dim 1024) | | |

Detailed yaml inspection confirms: no R13_T*.yaml file simultaneously satisfies (a) init from T3_SLOT1 ckpt AND (b) `multi_axis_grl.hidden_dim: 1024`. T6 has T3_SLOT1 init but no GRL (NO_GRL recipe); T5C has GRL hidden_dim=1024 but P8A init. The composition is untested.

### 5.4 Memory search for these directions

Pattern: `l11.anchor|anchor.loss|distillation|teacher|t3.slot1.*1024|1024.*t3.slot1`

Memory entries from `MEMORY.md` mentioning relevant directions:

| Memory | Quote | Relevance |
|---|---|---|
| `project_stage2_all_levers_regress_p8a_2026-05-09` | "Untested lever classes: LoRA / parameter-efficient adapter FT (preserves base by construction), hard-output-preservation losses on a reference cohort, multi-objective training with Pillar-2 explicit constraint, frozen-encoder regimes other than head-only retrain." | Flags Slot 1 (L11 anchor) and Slot 2 (output distillation) lever classes as **explicitly untested**. |
| `project_stage2_all_levers_regress_p8a_2026-05-09` | "L11 anchor-loss FT (option 3a) remains a candidate next intervention" | Directly names Slot 1's mechanism as a candidate. |
| `project_t6_t7_t5c_scorecard_2026-05-12` | (recent) | Records T5C step3500 contract rank-3 placement, the partial chronic_6 recovery, and the open question of whether classifier capacity scales. |
| `project_t5c_classifier_capacity_mechanism` (open loop, severity medium) | Open loop close criterion: "L11 atlas inv_mean recompute on T5C step1500 + step3500 + step3750 — if T5C step3500 chronic_6 inv_mean ≥ P8A's..." | **CLOSED 2026-05-12** by `INV_MEAN_FACTS_2026-05-12.md` §3.3: T5C_step3500 chronic_6 inv_mean = +0.0224 vs P8A +0.0445; partial recovery (+0.0094 vs T4 baseline) but not at P8A's level. |

No memory entry records any prior R13 yaml implementing L11 feature anchor loss, per-frame output distillation, or T3_SLOT1+1024 composition.

## 6. Output artifacts

This audit doc has no CSV outputs; the grep commands are runnable and reproducible from §5. The infrastructure inventory in §1-§4 references file paths and line numbers in the current working tree (commit pending).

## 7. Caveats

- Grep matches are case-insensitive and pattern-based; structurally similar implementations under different names may have been missed if naming differed materially from the patterns listed.
- Memory entries reflect snapshots and may be incomplete; a structurally similar packet that was tested without a corresponding memory entry would not appear in §5.4.
- The infrastructure inventory in §1 lists only `loss/*.py` modules referenced by the current standard recipe; a deprecated or unused module may exist under a different path.

## 8. Direct observations

1. `AnchorAwarePenalty` exists as a global output-mean hinge; per-frame target distillation mode does not exist (§1, §5.2).
2. No `loss/l11_anchor_loss.py` or equivalent feature-level anchor module exists in the codebase (§3, §4.1).
3. SVD-residual FT regime (`apply_svd_to_in_proj=true`, `apply_svd_to_mlp=true`, `rank=736`) is the standard B16 fine-tuning lever, used by T3/T4/T5C/T6/T7 (§1, by yaml inspection).
4. Of R13 phase2_round13 yamls, no file has both T3_SLOT1 ckpt init and `multi_axis_grl.hidden_dim: 1024` (§5.3).
5. Memory entry `project_stage2_all_levers_regress_p8a_2026-05-09` explicitly lists "L11 anchor-loss FT (option 3a)" and "hard-output-preservation losses on a reference cohort" as untested lever classes (§5.4).
6. Open loop `t5c-classifier-capacity-mechanism` close criterion is met by `INV_MEAN_FACTS_2026-05-12.md` (partial recovery confirmed, but does not reach P8A's chronic_6 inv_mean) (§5.4).
