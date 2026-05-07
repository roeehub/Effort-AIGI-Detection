# Thread: In-training canary signal — bridging the train-time / deployment-time gap

> **Template contract**: this thread follows `thread_template.md`. Status of the load-bearing claim is **active design + first-fire pending** as of 2026-05-07. The canary probe infrastructure has landed; whether the metrics it reports actually predict deployment-grade quality will be empirically tested by the P2 packet runs.

## The question

Train-time loss curves, val AUC, and the `value_composite` trainer metric have been empirically demonstrated to be decoupled from actual deployment quality on this project (memories `project_train_auc_not_valid_promotion_signal.md`, `project_class_sep_not_predictive.md`, `project_value_composite_semantics.md`). A model can hit 0.99 train AUC and still be a "pretty bad model" deployment-side. The agent does not know during training whether the run is heading toward a good ckpt or a P1-BUNDLE-style ROC-degenerate one until the post-train scorecard lands.

This thread tracks the design + evolution of an in-training signal that **is** correlated with deployment quality at near-zero cost: a fixed canary set scored every N steps, with deployment-shaped metrics logged to W&B.

## Initial belief

When a model is trained, the most reliable signal is post-training evaluation. Mid-training W&B logs (loss, AUC, val metrics) are watched but treated as advisory; promotion decisions wait for the contract scorecard. This is conservative and correct as a fallback, but expensive: a wasted overnight run is ~$60-80 and 10-12h that could have been intercepted by step 2000 if we had a deployment-shaped readout.

## What changed our mind

- **2026-05-07 (P1 evaluation, joint τ-sweep, FACTS doc `analysis/p1_pe_eval_2026-05-07/joint_tau_sweep_2026-05-07/JOINT_TAU_SWEEP_FACTS_2026-05-07.md`)**: F1 ∩ F4 ∩ F5 overlap region is EMPTY for P1 BUNDLE_step500. Roy_D real frames sit in the same score band as lockbox fakes; no τ-cut separates them. **Calibration-shifting is mathematically refuted as a fix** — distributions overlap at the per-frame level. The failure mode was undetectable in training-time loss curves; it surfaced only on the post-training scorecard 12h later.
- **2026-05-07 (Job 7, FACTS doc `analysis/pd_vs_p1_2026-05-07/PD_VS_P1_FACTS_2026-05-07.md`)**: PD's deeplive_corr_top_n_step4800 lifts dev_macro_recall by +0.149 over E2B but breaks dor invariance to FPR ≥ 0.46 on 5/6 ckpts. The PD training-side signal (per-axis Pearson r dropping cleanly) said the loss was biting; what it didn't say is "the model is moving the shortcut to a different population (dor) you'll only see at scorecard time."
- **2026-05-07 (cold-start agent simulation)**: a fresh agent reading the FACTS docs proposed a deployment-shaped canary as the next-best monitoring step. The user authorized building it at 2× the original size with diverse identities.

The structural finding both events point at: **per-axis loss values converge before deployment-grade behavior crystallizes**. Training-time signals describe what the optimizer is optimizing; they don't describe whether what's being optimized is what's actually going to deploy well. The fix is to compute deployment-shaped metrics on a fixed canary at the same cadence as periodic_saves.

## Current stance (2026-05-07)

The canary probe lives at [`trainer/mixins/canary_probe.py`](../../../trainer/mixins/canary_probe.py) and fires from the trainer's `_run_validation` callsite right after `_run_periodic_saves`. **Hard discipline**: bulletproof — every section is wrapped in try/except; on any failure it logs a warning and disables itself for the rest of the run. Cannot crash training.

**Canary composition** ([`arena/canaries/teams_chronic_diverse_800_2026-05-07.parquet`](../../../arena/canaries/teams_chronic_diverse_800_2026-05-07.parquet), 800 frames, deterministic seed=42):

| Cohort | Frames | Source |
|---|---:|---|
| 6 chronic reals (50 each) | 300 | `teams_real_all_dev` prefix-matched on chronic IDs (PC_Generator__s22/s45, Q__s6, bla_bla_chow + s2, Roy_D) |
| 5 healthy diverse reals (50 each) | 250 | `teams_real_all_dev` (Test_Cam, Md_noyn_Sharker, Xiang_Xiang2_Feng, dor) + `teams_real_dor_dev` (dor_shkedi) |
| HDTF clean reals | 50 | `proper_real_clean_lockbox` |
| lockbox fakes | 100 | `teams_fake_all_lockbox`, video-stratified |
| viso fakes | 50 | `visomaster_enhanced_macro_dev` |
| deeplive fakes | 50 | `deeplive_enhanced_dev` |

Total: 600 reals + 200 fakes = 800 frames. P8A reference scores pre-computed at calibrated τ for the Wilcoxon drift signal.

**Logged scalars per probe** (W&B keys, fired every `frequency_steps`):

1. `canary/score_p50_on_reals`, `canary/score_p95_on_reals`, `canary/score_mean_on_reals`, `canary/score_std_on_reals` — distribution shape (catches pair_rank tail-collapse).
2. `canary/score_p50_on_fakes`, `canary/score_p05_on_fakes`, `canary/score_mean_on_fakes` — fake-side distribution.
3. `canary/chronic_mean/<identity>` (6 keys) — per-chronic-identity mean prob_fake. Plus `canary/max_per_identity_mean_score` and `canary/mean_per_identity_mean_score`.
4. `canary/lockbox_recall_at_FPR_10pct`, `canary/lockbox_recall_at_FPR_5pct`, `canary/lockbox_tau_at_FPR_10pct` — F1-style FPR-calibrated recall (the deployment-relevant metric, monotone-invariant).
5. `canary/recall_at_tau05/<method_cohort>` (4 keys) — per-method (lockbox / viso / deeplive / teams) recall at τ=0.5.
6. `canary/wilcoxon_stat_vs_p8a_reals`, `canary/wilcoxon_pval_vs_p8a_reals`, `canary/mean_score_drift_vs_p8a_reals`, `canary/abs_mean_score_drift_vs_p8a_reals` — Wilcoxon signed-rank drift from P8A's known behavior on the canary's reals.
7. `canary/n_frames_evaluated`, `canary/n_reals`, `canary/n_fakes` — sanity counters.

Cost: ~12 sec per probe (forward pass on 800 frames, A100). 8 probes per 8000-step run = ~96 sec total overhead, < 0.3% of training time. First-call latency is higher (~1-3 min) due to GCS download of the 800 frames; cached on the instance for subsequent probes.

**This stance is hypothesis-active, not validated yet** because:
1. The canary's first fire is on the P2 packet runs launching 2026-05-07 evening. Whether the logged metrics correlate with actual scorecard outcomes will only be measured after P2 + scorecard land 2026-05-08.
2. The Wilcoxon-vs-P8A-reference assumes P8A's behavior on the canary's healthy reals is itself a signal of "good model." This is true for the healthy + HDTF reals (P8A correctly classifies, mean prob_fake 0.01-0.10) but NOT for chronic-6 reals where P8A scores ~0.93 mean — i.e., P8A is the wrong baseline on those 300 frames. The current implementation's aggregate Wilcoxon mixes both. A v2 should split the metric: signed mean drift on healthy ≤ 0 (good), signed mean drift on chronic ≤ −0.5 (improvement away from P8A's failures).
3. Per-method fake recall at τ=0.5 is a calibration-fragile metric. The canary's `lockbox_recall_at_FPR_10pct` is the calibration-invariant version and should be the primary read.

**The single most important thing the canary watches for**, given today's evidence: a rising `score_p95_on_reals` while `lockbox_recall_at_FPR_10pct` stays flat or falls. That's the pair_rank-tail-collapse signature (memory `project_p1_diagnostics_complete_*`) — it would give us 8-12h of warning that a run is heading to ROC-degeneracy before the post-train scorecard arrives.

## Packet timeline

- [P2](../packets/P2.md) (2026-05-07 → 2026-05-08) — first packet to ship with the canary probe enabled. Three slots, all from-scratch, all canary-monitored. The canary's first empirical fire; its predictive value is testable post-launch.

## Evidence locations

- `trainer/mixins/canary_probe.py` — the mixin (~440 LOC, bulletproof).
- `trainer/trainer.py` — `_run_canary_probe(step_cnt)` called from `_run_validation` after `_run_periodic_saves`.
- `train_sweep.py` — `canary_probe` re-apply in the wandb-flattening allowlist (memory `project_wandb_flattens_nested_dicts.md`).
- `arena/canaries/teams_chronic_diverse_800_2026-05-07.parquet` — the canary set.
- `arena/canaries/teams_chronic_diverse_800_2026-05-07.README.md` — composition table + provenance.
- `arena/canaries/build_canary_2026-05-07.py` — reproducible build script (seed=42).
- `experiments/phase2_round13/R13_P2_SCRATCH_*.yaml` — three yamls with `canary_probe.enabled: true`.

## Cross-thread refs

- [`promotion_contract_evolution`](promotion_contract_evolution.md) — defines the deployment-grade contract layer. The canary is an in-training proxy for that layer's outputs, NOT a replacement.
- [`anti_shortcut_bundle_decomposition`](anti_shortcut_bundle_decomposition.md) — discipline for stacking levers. The canary's per-method recall + per-identity mean score lets you see in-flight whether a stacked bundle is trading invariances (the PD failure mode).
- [`pair_rank_collateral`](pair_rank_collateral.md) — the Roy_D-class regression. The canary's per-chronic-identity mean score on Roy_D specifically is the early-warning signal for this regression class.
- [`processing_signature_shortcut`](processing_signature_shortcut.md) — the canary watches the chronic-6 + Roy_D outcomes most closely because those are the residual deployment-blocker identities.

## Open loops

### Open loop: canary-empirical-validation
status: open
severity: medium
first_seen: 2026-05-07
last_verified: 2026-05-07
close_criterion: at least 2 P2 ckpts have BOTH (a) a canary readout at the corresponding training step AND (b) a post-training contract scorecard verdict. Compute Pearson correlation between `canary/lockbox_recall_at_FPR_10pct` and the scorecard's `lockbox_fake_recall` at calibrated τ. Compute the same for `canary/max_per_identity_mean_score` and the F5 close-criterion's binding identity. If r > 0.7 on both, the canary is empirically validated as a deployment proxy. If r < 0.3, the canary's design is wrong and the metric set needs re-deriving from the P2 outcomes. Either result closes the loop.

### Open loop: canary-wilcoxon-cohort-split
status: open
severity: low
first_seen: 2026-05-07
last_verified: 2026-05-07
close_criterion: the Wilcoxon-vs-P8A metric is split into two cohort-specific stats: `canary/wilcoxon_stat_vs_p8a_healthy_reals` (where small drift = good) and `canary/wilcoxon_stat_vs_p8a_chronic_reals` (where large negative drift = good). The aggregate stat is preserved but augmented. Implementation is ~30 lines in `trainer/mixins/canary_probe.py:_aggregate_metrics`. Defer until P2 results validate the basic metric set; if the basic Wilcoxon already correlates with deployment quality, the cohort split may not be needed.

### Open loop: canary-finer-resolution
status: open
severity: low
first_seen: 2026-05-07
last_verified: 2026-05-07
close_criterion: a "tiny canary" companion of ~60 frames runs every 200 steps for finer resolution at the cost of ~0.1% extra training time. Useful specifically because P1 BUNDLE_step500 was already in the failed regime by step 500 — the current 1000-step cadence might miss the inflection. Implement only if the 1000-step cadence proves to be too coarse on the P2 runs. Tiny canary composition would be: 5 chronic identities × 6 frames + 30 lockbox fakes = 60 frames.
