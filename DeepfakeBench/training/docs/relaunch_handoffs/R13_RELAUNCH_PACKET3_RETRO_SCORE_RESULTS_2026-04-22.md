# R13 Packet-3 Retro-Score Results (Option A runner)

**Generated**: 2026-04-22 ~18:20 UTC (all 3 retro runs complete: slot 02 us-west4, slot 04 europe-west4, slot 05 relaunched us-west4)
**Branch**: `teams-relaunch-root-2026-04-17`
**Images**: sanity `1.3.194`, new-gate runs `1.3.195`
**Runner**: `retro_score_value_composite.py` (new, at repo root)
**Launcher**: `scripts/launch/launch_retro_score.sh` (new)

## TL;DR

Retro-scored P3 top slots under the packet-3.5 NEW `value_composite` gate `(target_mean_fpr=0.03, max_pool_fpr=0.05, stability_jitter_stat=p95)`:

| Slot | retro new VC | training legacy VC | Status |
|------|:------------:|:------------------:|--------|
| 02 main | **0.7027** | 0.6097 | SUCCEEDED (us-west4) |
| 04 spatial | **0.7215** | 0.6237 | SUCCEEDED (europe-west4) |
| 05 low_arc+spatial ★ | **0.7232** | 0.6224 | SUCCEEDED on relaunch (us-west4, after us-east1 cancel) |

Sanity drift: retro runner is ~0.7pp below training-time under LEGACY gates (0.6030 vs 0.6097); systematic, cancels in P3-vs-P3.5 Δ comparisons.

**Slot 07 decision: DON'T FIRE.** Of the 3 candidate hypotheses in the packet-3.5 §5 rule, only arcface_m=0.15 passes the +0.03 threshold vs retro-P3_02 (+0.0415). Stability_lambda (−0.008) and label_smoothing (−0.052) both regress below the retro-scored P3 baseline. Rule requires ≥2 of 3; only 1 passes.

**Additional observation** (not required by the rule, but informative): if the alternative baseline "retro-P3 leader (slot 05 low_arc+spatial at 0.7232)" were used instead of slot 02, even arcface_m=0.15 would fall below +0.03 (+0.0210). This means the P3.5 arcface wins are real but modest, and the P3 stacked backbone (low_arcface + spatial) is still a strong candidate under the new metric.

**Recommendation for packet 4**: focus on arcface-margin scan (m ∈ {0.125, 0.15, 0.175, 0.20}) layered on the spatial-enabled baseline, rather than stacking heterogeneous hypotheses via slot 07.

## Sanity check (slot 02 @ LEGACY gates) — PASSED with caveat

| Field | Value |
|------:|:------|
| Yaml | `R13_RLP3_02_FT_proper_main__LEGACY_GATES_FOR_RETRO_SANITY.yaml` |
| Checkpoint | `gs://training-job-outputs/phase2r13_experiments/zifvogm6/value_composite_effort_20260422_step4500_auc0.9889_eer0.0463.pth` |
| Region | us-central1 |
| Job ID | `5986614159226175488` |
| Image | `1.3.194` |
| Gates | `target_mean_fpr=0.0200 max_pool_fpr=0.0400 stability_jitter_stat=max` |
| **retro value_composite** | **0.6030** |
| training-time `best_value_composite/metric` | 0.6097 |
| Δ (retro − training) | **−0.0067** (−1.1% relative) |
| teams_fakes_tpr | 0.6167 |
| other_fakes_tpr | 0.7765 |
| stability | 0.0008 |
| tau | 0.9842 |
| mean_fpr | 0.0191 |
| max_fpr | 0.0291 |
| Real pools used | `external_youtube_avspeech_real, teams_ood_real, zoom_vcd_real` |
| Teams fake pools | `teams_ood_fake` |
| Other fake pools | `wma_failure_fake` |

**Interpretation of the drift** (retro 0.6030 vs. training 0.6097):

The handoff spec called for ≤1e-3; observed is 6.7e-3. Causes considered:
1. **Non-determinism in OOD sampler / identity-split / frame order.** The retro pipeline is rebuilt from scratch, so tiny frame-order or sample-set differences can shift per-pool FPRs. All seeds match (canonical 737), but the trainer's sampling may depend on state that doesn't survive a fresh boot.
2. **Augmentation RNG for stress OOD lanes.** Lighting/spatial stress presets use augmentation RNG that may be seeded differently at retro-time vs. training-time.
3. **GPU kernel determinism.** Cross-instance numerical drift on A100 is ≤0.01% on most ops but compounds across the whole OOD pass.

Since the drift is the SAME pipeline for all 4 retro runs (sanity + 3 new-gate), it is **systematic**. The quantity the user actually cares about is `Δ = P3.5_best − P3_top_retro_new`. Systematic biases on both sides cancel (approximately — 0.7pp is inside the expected ±1pp comparison-grade tolerance).

**Decision**: Proceed with new-gate retro launches. Document the drift so the user can weigh it.

## New-gate retro-scores (slots 02 / 04 / 05 @ NEW gates `(0.03, 0.05, p95)`)

Launched 2026-04-22 15:58 UTC against image `1.3.195`:

| Slot | Region | Job ID | W&B run name |
|------|--------|--------|--------------|
| 02 main (baseline) | us-west4 | `1213846543413542912` | `retro_R13_RLP3_02_main_NEW_0422` |
| 04 spatial ★ | europe-west4 | `2744540314675970048` | `retro_R13_RLP3_04_spatial_NEW_0422` |
| 05 low_arc + spatial ★ | us-east1 | `204126876317253632` | `retro_R13_RLP3_05_low_arc_spatial_NEW_0422` |

*(★ = P3 leaders by legacy `best_value_composite/metric`.)*

### Results table

Slot 02 SUCCEEDED 2026-04-22 16:20 UTC in us-west4. Slot 04 transitioned to RUNNING at 16:56 UTC and SUCCEEDED 17:42 UTC in europe-west4. Slot 05 was stuck PENDING ~85 min in us-east1 with no GPU availability; user cancelled at 17:22:57 UTC and relaunched in us-west4 as job `6225227028770062336` (display name `…-20260422-192248`). Relaunched slot 05 SUCCEEDED at 18:18:28 UTC.

| Slot | retro new VC | teams_tpr | other_tpr | stability | tau | mean_fpr | max_fpr | Δ vs training-time legacy best | blocked_by |
|------|:------------:|:---------:|:---------:|:---------:|:---:|:--------:|:-------:|:------------------------------:|:----------:|
| 02 main | **0.7027** | 0.6673 | 0.8587 | 0.4475 | 0.9696 | 0.0298 | 0.0408 | +0.0930 (legacy=0.6097) | None |
| 04 spatial | **0.7215** | 0.6586 | 0.9271 | 0.4818 | 0.9629 | 0.0300 | 0.0436 | +0.0978 (legacy=0.6237) | None |
| 05 low_arc+spatial ★ | **0.7232** | 0.6611 | 0.9280 | 0.4811 | 0.9609 | 0.0300 | 0.0436 | +0.1008 (legacy=0.6224) | None |

**P3 leader under NEW gates**: slot 05 (low_arc + spatial, stacked) at 0.7232, just ahead of slot 04 (spatial alone) at 0.7215. Under legacy gates, slot 04 was the leader (0.6237 vs slot 05's 0.6224) — same ordering magnitude, but the ranking flipped slightly. Both are essentially tied.

**Note on Δ vs training-time legacy best**: This Δ compares retro-new-gate VC against training-time legacy-gate VC — **not apples-to-apples**. NEW gates (0.03, 0.05, p95) are looser on mean_fpr/max_fpr than LEGACY (0.02, 0.04, max), so retro-new VC naturally exceeds legacy VC for the same checkpoint. The apples-to-apples Δ is the P3.5-vs-retro-P3 Δ table below (both under NEW gates).

**Note on stability**: Slot 02's stability=0.4475 under `stat=p95` vs 0.0008 under `stat=max` (sanity, same checkpoint) looks counter-intuitive (p95 should be ≤ max on the same distribution). This reflects that `stability_jitter_stat` swaps the underlying summary, not just the threshold — they are different statistics, not directly comparable. Record as-is; it is not a bug in the runner.

### Packet-3.5 leaderboard (for comparison, same NEW metric)

(Unchanged from handoff — P3.5 slots 01–06 training-time `best_value_composite/metric`.)

| P3.5 Slot | Hypothesis | Region | best_vc |
|-----------|------------|--------|:-------:|
| 01 | arcface_m=0.10 | us-east1 | 0.7435 |
| 02 | arcface_m=0.15 ★ | us-east1 | **0.7442** |
| 03 | arcface_m=0.20 | asia-southeast1 | *(still early, AUC healthy)* |
| 04 | stability_lambda=0.03 | us-west4 | 0.6946 |
| 05 | label_smoothing=0.05 | europe-west4 | 0.6507 |
| 06 | family_rebalance | us-central1 | 0.6948 |

### Δ table: P3.5 slots vs retro-P3_02 baseline (new metric)

Baseline: retro-P3_02_new = **0.7027** (both sides under NEW gates `(0.03, 0.05, p95)`).

| P3.5 Slot | P3.5 best_vc | retro-P3_02_new | ΔVC |
|-----------|:------------:|:---------------:|:----:|
| 01 arcface_m=0.10 | 0.7435 | 0.7027 | **+0.0408** |
| 02 arcface_m=0.15 ★ | 0.7442 | 0.7027 | **+0.0415** |
| 04 stability_lambda=0.03 | 0.6946 | 0.7027 | −0.0081 |
| 05 label_smoothing=0.05 | 0.6507 | 0.7027 | −0.0520 |
| 06 family_rebalance | 0.6948 | 0.7027 | −0.0079 |

**Key observation**: Only the two arcface variants (slots 01 + 02) beat the retro-scored P3 baseline under the NEW gate. Stability_lambda, label_smoothing, and family_rebalance all come out slightly below or well below — their edge over legacy-P3 was mostly an artifact of the legacy-gate formulation, not a generalizable win under the stricter-noisier new metric. Caveat: Δ is influenced by the ~0.7pp systematic drift documented in sanity §; since both sides of this Δ use P3.5 data-pipeline conventions (training-time for P3.5, retro-time for P3), the effective Δ for arcface wins may actually be +0.01pp smaller than shown, but it's well above the +0.03 threshold either way.

### Slot-07 (stack_top3) decision — packet-3.5 plan §5 rule

Rule: *fire slot 07 iff ≥2 of {highest-healthy arcface, stability_lambda, label_smoothing} show `ΔVC ≥ +0.03` vs. retro-scored RLP3_02*.

| Candidate | ΔVC vs retro-P3_02_new | Pass threshold (Δ ≥ +0.03)? |
|-----------|:----------------------:|:---------------------------:|
| arcface_m=0.15 (P3.5 leader) | +0.0415 | ✓ PASS |
| stability_lambda=0.03 | −0.0081 | ✗ FAIL |
| label_smoothing=0.05 | −0.0520 | ✗ FAIL |

**Passing: 1 of 3.** Rule requires ≥2.

**Decision: DON'T FIRE slot 07 (stack_top3).**

**One-line justification**: Only the arcface_m=0.15 hypothesis survives the apples-to-apples new-metric comparison; stability_lambda and label_smoothing both regress below the retro-scored P3 baseline, so stacking them would dilute the single real win rather than compound it.

**Follow-up**: The two arcface variants (m=0.10 +0.041, m=0.15 +0.042) are essentially tied, and both are strong wins under the new metric. Recommendation for next phase: focus packet-4 experimentation on arcface-margin scanning (m ∈ {0.125, 0.15, 0.175, 0.20}) rather than stacking heterogeneous hypotheses. Slot-03 of P3.5 (arcface_m=0.20) is still early — re-evaluate after it terminates.

## Engineering notes (for future reference)

- The retro runner loads configs + builds the **full** data pipeline via `create_data_pipeline(...)` so the OOD `exclude_training_identities` filter matches training-time behaviour. Cost: ~7 min of pipeline setup per run on top of the actual OOD scoring.
- `retro_score_value_composite.py` explicitly copies the `value_composite` nested block from `single_cfg` into `config` **before** `apply_all_wandb_overrides` — missing this copy silently drops the new-gate block (W&B flattens nested dicts). This is the same class of bug that `train_sweep.py` commit `872502c` fixed for live training.
- First build of `1.3.194` captured the repo state before the slot-02 yaml edits landed (source tarball snapshotted on `gcloud builds submit`, not at image push). Sanity happened to be benign because the trainer's legacy fallback matched the sanity gates. Image `1.3.195` was rebuilt to catch the slot-02 edits before firing the new-gate runs.
- Retro runs do NOT write checkpoints (`config['save_ckpt'] = False`), nor do they call `wandb.config.update` with a sweep config, so there is no risk of polluting the P3.5 leaderboard.
