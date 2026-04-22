# R13 Relaunch — Packet 3.5 Experiment Plan (2026-04-22)

**Status**: authored 2026-04-22 alongside packet 3 mid-training.
**Relationship to packet 3**: parallel wave, not successor. Launched in `us-east1` while packet 3's 7 RLP3 runs complete in `asia-southeast1`.
**Driving question**: given that packet 3 will ceiling `value_composite` around 0.60–0.70 (AUC saturated, stability pinned by `score_jitter_max/teams_ood_fake ≈ 1.0`), what *training-time* levers can push it toward the deployment target of 0.90–0.95?

## 1. Motivation

Packet 3 mid-run diagnostics (`HANDOFF.md` and in-code inspection) exposed four structural caps on `value_composite`:

1. **`arcface_m: 0.0`** in every RLP yaml. ArcFace is running with zero margin — a fancy linear head. Never varied in packets 1–3.
2. **`stability_lambda: 0.0`** — training-time perturbation-consistency loss (`trainer/mixins/stability.py`) is disabled.
3. **`label_smoothing: 0.0`** — off.
4. **`anneal_steps: 15000` but `total_training_steps: 10000`** — ArcFace scale anneal from `s=6→12` never completes; training ends at `s≈10`.

Two metric-definition issues additionally matter:

5. **Jitter stability aggregator is hardcoded `max`** (`trainer/trainer.py:3068`). `teams_ood_fake` maxes at ≈1.0, pinning the 10% stability component to 0.
6. **FPR gate is hardcoded `target_mean_fpr=0.02, max_pool_fpr=0.04`**. The user has approved relaxing to `0.03, 0.05` as a deployment-defensible operating point.

Packet 3.5 turns on the four disabled training knobs (cautiously — the user flagged that this backbone collapses with aggressive ArcFace margins), applies the two metric-definition changes globally, and retro-scores packet 3 under the new definition.

## 2. Non-regressions from packets 1–2

Every slot inherits the `R13_RLP3_02_FT_proper_main` data recipe. Specifically:

- **Unenhanced proper-data only** (`include_lanes: [proper_visomaster_clean, proper_visomaster_teams]`). No enhanced proper-data. Packet 2 showed enhanced at 2.2× dose hurts; packet 3 slot 08 (dose-matched enhanced-teams) remains deferred and is **not** in this wave.
- **No hints.** `visomaster_hints` and `visomaster_hints_teams` stay disabled. Packets 1–2 confirmed hints are dead.
- **Seed 737** across all slots. Seed variance is deferred to packet 4.
- **`identity_split_mode: hash_stable`** (packet-2-onwards convention).
- **Base checkpoint**: `top_n_effort_20260310_step14000_auc0.9930_eer0.0253.pth` (same r12g-FP32 base every packet-3 slot used).

## 3. Global changes applied to every slot

Three metric-definition changes + one schedule fix:

| Change | Legacy | Packet 3.5 | Why |
|--------|--------|-----------|-----|
| `value_composite.target_mean_fpr` | `0.02` | `0.03` | User-approved relaxed deployment gate. |
| `value_composite.max_pool_fpr` | `0.04` | `0.05` | User-approved relaxed deployment gate. |
| `value_composite.stability_jitter_stat` | `max` | `p95` | One frame should not kill the stability term. `_aggregate_jitter_across_videos` already emits `p95`. |
| `anneal_steps` | `15000` | `8000` | Anneal finishes 80% through training, leaving 2k steps at full `s=12`. |

All three metric-definition changes are wired through a new config block:

```yaml
value_composite:
  target_mean_fpr: 0.03
  max_pool_fpr: 0.05
  stability_jitter_stat: "p95"
```

Backend code in `trainer/trainer.py` defaults all three keys to the legacy packet-3 values when absent, so runs without the block are bit-identical to pre-packet-3.5 trainers.

## 4. Slate — 6 single-lever slots + 1 contingent stacked

Each single-lever slot differs from RLP3_02 by **exactly one training knob** (plus the inherited global metric/schedule block). All launched in `us-east1` in parallel with packet 3 (`asia-southeast1`).

| Slot | Delta | Hypothesis | Expected composite lift | Abort signal |
|------|-------|-----------|------------------------|--------------|
| `R13_RLP35_01_arcface_m010` | `arcface_m: 0.10` | Small margin bites without collapse. | +0.03 → +0.08 | `val_holdout/auc < 0.95` in first 1k steps. |
| `R13_RLP35_02_arcface_m015` | `arcface_m: 0.15` | Middle of cautious range. | +0.05 → +0.12 | same. |
| `R13_RLP35_03_arcface_m020` | `arcface_m: 0.20` | Upper bound of cautious range; probes backbone-collapse threshold. | +0.08 → +0.15 if healthy. | same. |
| `R13_RLP35_04_stability_lambda_003` | `stability_lambda: 0.03` (+ `noise_std=0.02, crop_jitter=0.03`) | Training-time perturbation KL drops `teams_ood_fake` frame jitter from ~1.0 toward ≤0.5. | Stability term 0 → 0.3–0.6. Composite +0.03 → +0.06. | If loss divergence or `val_holdout/auc` drop. |
| `R13_RLP35_05_label_smoothing_005` | `label_smoothing: 0.05` | Smoother targets lower real-score tail → less strict τ at FPR gate → more fakes caught. | +0.02 → +0.06. | — |
| `R13_RLP35_06_family_rebalance_proper_up` | `family_weights`: proper_clean/teams fake 1.0→1.5; realpool/external real 2.5→3.5; df40_fake 0.15→0.10. | Deployment real surface gets more training attention → tighter real-pool FPR tails. | +0.03 → +0.07. | — |
| `R13_RLP35_07_stack_top3` | **contingent** — arcface_m (best healthy) + `stability_lambda=0.03` + `label_smoothing=0.05` + family rebalance. | Levers compound. | Composite 0.60 → 0.82–0.92. 0.95 is stretch. | — |

Slot 07 is authored now but launched only after early signals from 01–06 (~step 3–4k, ~2h after launch) confirm which single levers are net-positive. Tune its `arcface_m` field to match the highest-healthy slot (01/02/03) before launch.

## 5. Decision rules

After all 7 runs complete:

- **Single-lever ranking**: compute `Δvalue_composite` vs packet-3 RLP3_02 baseline (retro-scored under same `(0.03, 0.05, p95)` gate). Rank slots 01–06 by this Δ.
- **Backbone-collapse threshold**: if slot 03 (m=0.20) hits the abort signal and slot 02 (m=0.15) is clean, the safe margin cap is in [0.15, 0.20); packet 4 does not exceed 0.15.
- **Packet 4 design input** is the single-lever ranking. High-signal levers get retuned; null-signal levers get dropped.

## 6. Retro-scoring packet 3 under the new metric

Independent of the training wave: once image `1.3.192` (with the new code plumbing) lands, rerun the packet-3 top-3 checkpoints through `rerun_validation.py` with the new `value_composite` block. This produces:

- A baseline "packet 3 under new metric" composite number, separating "metric-defn-moved-the-number" from "training-knob-moved-the-number."
- Confirmation that `p95` vs `max` materially moves the stability term (expected ~0 → ~0.3–0.5 without any retraining).

## 7. Risks

- **Backbone collapse on slot 03** (m=0.20). Mitigated by cautious slots 01, 02.
- **Relaxed FPR gate inflates absolute numbers** independent of training. Mitigated by retro-scoring packet 3 under the same new gate.
- **Stability `p95 → max` dependency**: if `score_jitter_p95` is also near 1.0 (not just `max`), the stability term stays pinned regardless of stat choice. Only packet-3 retro-score will confirm.
- **Parallel compute burn** — 6 new us-east1 runs for ~8h. User confirmed quota.

## 8. Not in this wave (deferred to packet 4)

- Third-seed variance probe — need single-lever attribution first.
- Frame aggregator swap (`mean → median`) — breaks R12/R13 comparability.
- Dose-matched enhanced-clean — packet-2 null-negative; needs a new hypothesis.
- Temporal/determinism probes — infrastructure work; not composite-moving.
- Relaxing max_pool_fpr further (e.g. 0.06) — wait for packet-3.5 per-pool data.
- Multi-head architecture for enhanced/unenhanced capacity conflict.

## 9. File references

- **Plan**: this file (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_5_EXPERIMENT_PLAN_2026-04-22.md`).
- **Code**: `trainer/trainer.py` (value_composite config plumbing + jitter stat switch), `trainer/mixins/arcface.py` (anneal-mismatch warning).
- **Tests**: `tests/test_value_composite_config.py`, `tests/test_arcface_anneal_warning.py`.
- **YAMLs**: `experiments/phase2_round13/R13_RLP35_0{1..7}_*.yaml`.
- **Image**: new Vertex image at `us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:1.3.192` (to be built).
- **Baseline**: `experiments/phase2_round13/R13_RLP3_02_FT_proper_main.yaml`.
- **Retro-target checkpoints**: packet-3 top-3 per slot under `gs://training-job-outputs/phase2r13_experiments/<wandb_run_id>/`.

## 10. Handoff checklist for the next agent

- [ ] Code changes landed: trainer.py C1–C3, arcface.py C4. Tests green inside image.
- [ ] Image 1.3.192 built & pushed. VERSION bump committed.
- [ ] 6 slot yamls launched in `us-east1`, all `JOB_STATE_RUNNING`.
- [ ] Slot 07 yaml authored but not launched.
- [ ] Packet 3 retro-score job (rerun_validation.py on top-3 per-slot checkpoints with new metric block) queued for once packet 3 finishes.
- [ ] Early signals from slots 01–06 (~step 3–4k) feed slot-07 decision: fire or revise.
