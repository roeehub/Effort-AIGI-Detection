# R13 Relaunch Packet 3 — Handoff Plan (v3, final)

**Date:** 2026-04-21
**Status:** Planning complete, awaiting approval to implement.
**Audience:** The next agent picking up the training thread. Should be self-contained.

---

## 1. Context

Packets 1 and 2 are essentially finished. This document covers:

- What the packets taught us (stable, unstable, or muddled).
- What instrumentation the evaluation stack is missing, and why those gaps distort current interpretation.
- The nine code / config changes to land **before** packet 3 launches.
- The eight packet-3 experiments, their launch order, and the decision rules we'll apply when reading results.

### 1.1 What is stable after packets 1 and 2

- **Hints are out.** `RLP1_02`, `RLP1_03`, `RLP2_01` (no hints) all underperform the corresponding hint-bearing controls — hints have failed in two packets.
- **Unenhanced proper-data is the best new signal.** `RLP1_04` (hints + unenhanced proper) was packet-1's second-best arm. `RLP2_02` (no hints + unenhanced proper) is packet-2's top composite (`0.98937`).
- **Scratch is not competitive on the balanced metric.** `RLP1_08` best composite `0.98421` while running 3× the steps.
- **WT-C sidecars (GammaUp, Teams-shadow) and full-proper are rejected.** Packet 1 closed these; packet 2 didn't retry.

### 1.2 What is muddled, not stable

- **Enhanced proper-data vs unenhanced.** Packet 2's conclusion was "enhanced hurts," but the evidence is not that clean. `RLP2_03` added +1484 enhanced rows on top of +684 unenhanced (2.2× dose). `RLP2_04` same shape. `val_holdout` for `RLP2_02` contains **zero enhanced-proper content**, so we are measuring against an unenhanced-only validation slice. We cannot separate "enhanced training hurts the enhanced slice we don't measure" from "enhanced training hurts the unenhanced slice we do measure." The right reframe: we want a model that captures **both** enhanced and unenhanced in deployment. Packet 3 has to make that separation measurable.
- **Stability probes are inconclusive.** `RLP2_05` (spatial) and `RLP2_06` (low-ArcFace) are tied with `RLP2_02` on composite. Live jitter does not separate them. Without a per-video max-delta or a stacked test, we cannot say whether these probes do anything or just repeat `02`.
- **The packet-2 fresh control dropped.** `RLP2_01 = 0.98662` vs `RLP1_01 = 0.98915`, delta `−0.00253`. Packet 1 used `identity_split_mode: shuffle` (legacy); packet 2+ uses `hash_stable`. Part of the drop is split-mode artifact and cannot be attributed to mutable-source drift alone.

### 1.3 Instrumentation gaps that distort current interpretation

- **Mean jitter hides per-video spikes.** `mean(|Δp|)` across 7 sampled-frame diffs cannot detect a single 0.3–0.7 probability jump inside a video. Live numbers around 0.08 are arithmetically consistent with occasional large spikes.
- **The 5% test split is dormant.** It is aliased into `val_holdout_loader` and no frozen test-time number is logged per run. All "packet comparisons" are run-time snapshot comparisons.
- **No lighting-stress eval pool.** The prior "lighting aug is worse" conclusion was drawn against eval data with no lighting stress to reward robustness. That conclusion is not safe.
- **No enhanced-proper slice in OOD.** Enhanced-proper only exists inside val_primary and only for arms that train on it. No run is evaluated for "how well does this generalize to enhanced content it didn't train on?"
- **W&B run view is cluttered.** ~50 per-method metrics bury the ~10 numbers that drive decisions.
- **OOD starts too late for FT runs.** First OOD at step 5000 means 2 of 6 packet-2 best checkpoints landed exactly at the first OOD read — we may be promoting later checkpoints than we should.
- **No proper-data `build_id` in W&B.** Silent mismatches (packet-1 live proper-fake row counts diverged from yaml-documented counts) are undetectable after the fact.
- **No metric aligned to the actual deployment value hierarchy** (see §1.4).

### 1.4 The deployment value hierarchy (anchor for all decisions)

In descending priority:

1. **Minimum false-positive rate** across all real pools, equally weighted (df40 real, external_youtube_avspeech, zoom_vcd_real, teams_ood_real, proper_clean_real, proper_teams_real).
2. **Maximum recall on Teams fakes** — deeplive + visomaster, **both enhanced and unenhanced**.
3. **Maximum recall on non-Teams fakes.**
4. **The rest of the fakes.**
5. **Robustness** to lighting and spatial/movement perturbations is a cross-cutting multiplier on the above — a slightly weaker model that is materially more lighting-robust beats a slightly stronger fragile one.

Current `best_ood_composite` (which is ≈ `(holdout_auc + ood_auc)/2`) is AUC-based and threshold-independent. It is adjacent to this hierarchy but does not enforce equal-weight FPR across real pools or the 2× priority on Teams fakes. Packet 3 logs a new metric (A9 `value_composite`) that does enforce this, as a **readout only** — checkpoint selection stays on `best_ood_composite` in packet 3 so packet-3 runs remain comparable to packet-2 runs. If `value_composite` ranks arms sensibly, checkpoint selection switches to it in packet 4.

---

## 2. Part A — Code Changes Before Launch

Nine items. A1 through A9. Land all before launching packet 3; packet-3 yamls assume every one of them exists.

### A1. Per-video jitter percentiles, max, spike counts

**What.** The scalar `mean(|Δp|)` cannot see per-video spikes. Add: per-video `max`, `p95`, and `spike_count_0p3` (fraction of frame-pair deltas > 0.3), aggregated across videos per method. Keep existing `ood/score_jitter/<m>` for dashboard continuity. Emit a W&B histogram of all frame-pair deltas per method, **logged every 2500 steps** (not every eval) to bound payload.

**Performance.** Per-video math is O(T=8) numpy — negligible vs a training step. Histogram is the only measurable W&B cost; 2500-step cap keeps it < 0.5% of total run time.

**Files.**
- `DeepfakeBench/training/trainer/trainer.py:2295-2299` — per-video diff loop. Capture the full `|diff(fp)|` array per video, not only its mean.
- `DeepfakeBench/training/trainer/trainer.py:2355-2359` — aggregation. Emit `ood/score_jitter_max/<m>`, `ood/score_jitter_p95/<m>`, `ood/score_jitter_spike_rate_0p3/<m>`.
- `DeepfakeBench/training/trainer/trainer.py` near line 2376 — add `wandb.Histogram(...)` under `if step % 2500 == 0` gate.
- **Mirror on val_holdout.** Current jitter block runs only in the OOD path. Factor per-video jitter computation into a helper in the same file; call from both OOD and `val_holdout` paths (`trainer.py:1737-1758`).

**Reuse.** `tools/teams_frame_policy_analysis.py:251, 294` already computes per-video `max_jitter`. Mirror its definition exactly so training-time and post-hoc numbers agree byte-for-byte.

### A2. Final standardized evaluation on best checkpoint (dual-checkpoint)

**What.** Replace the dormant "test slice" idea with **dual final evaluation passes, one per selection criterion** (see §8.7). At training end:
1. Track TWO best checkpoints during training:
   - `best_ood_composite` (AUC-based, preserves packet-2 comparability)
   - `best_value_composite` (deployment-aligned per A9; still readout-only for selection purposes in packet 3, but saved so we can evaluate it)
2. For each checkpoint, run one clean eval pass over: `val_primary`, `val_holdout`, every OOD source (A3 lighting-stress, A3b spatial-stress, A6 enhanced-proper), the A10 held-out slices, and the 5% test slice.
3. Log under `final_eval/by_ood_composite/<pool>/<metric>` and `final_eval/by_value_composite/<pool>/<metric>` — AUC, EER, FPR-at-thresh, TPR-at-thresh, and the A1 jitter family per pool.
4. A third small block, `final_eval/agree/same_step`, flags whether both checkpoints landed at the same training step (strongly indicates the two criteria agree).

**The 5% test slice** is kept as a cross-check under `final_eval/by_*/test/*` but it is **not** the headline number — it is a slice of the training distribution and is not aligned with the deployment value hierarchy. The headline numbers are the **A10 held-out bench** and the A6 enhanced-proper slices.

**Why dual.** Packet 3 keeps `best_ood_composite` as the nominal selection (packet-2 comparability). But we need to know: if we HAD selected on `value_composite`, would the chosen checkpoint differ, and would its numbers on the deployment-aligned metric be better? Dual checkpoints answer this for free at end-of-training — no extra training cost, one additional eval pass (~5 min).

**Why this matters.** Every run produces TWO frozen, directly-comparable numbers blocks (one per selection criterion) on the checkpoints actually worth comparing. Two runs at different wall-clock times can be compared on exactly the same eval pools, augmentation state, AND selection philosophy.

**Files.**
- `DeepfakeBench/training/data/sources/combined_paired.py:4305-4310` — stop re-wrapping `test_loader` as `val_holdout_loader`. Keep them distinct.
- `DeepfakeBench/training/data/sources/combined_paired.py:4456` — return contract. Emit `test_loader` separately.
- `DeepfakeBench/training/trainer/trainer.py:84` — `__init__` wiring: add `self.test_loader`, add `self.best_value_composite_ckpt` alongside `self.best_ood_composite_ckpt`.
- `DeepfakeBench/training/trainer/trainer.py:1573-1577` — at training end: reload BOTH checkpoints → run final eval pass on all pools (including A10 held-out) → log under `final_eval/by_<metric>/*`.
- Factor a `run_final_eval(checkpoint_path, loaders)` helper so A2b can reuse it.

### A2b. Retroactive `final_eval` for past runs

**What.** Standalone script: given a W&B run id, pull `best_ood_composite/gcs_path`, reconstruct the eval pools from the run's recorded config, run `run_final_eval`, log results back to the same W&B run under `final_eval/*`. Lets us retroactively produce comparable numbers for `RLP1_01`, `RLP1_04`, `RLP2_01`, `RLP2_02`, `RLP2_06`, so packet 3 can be compared across both packets on the same frozen criterion.

**Files.**
- New: `DeepfakeBench/training/tools/retroactive_final_eval.py`. Imports `run_final_eval` from trainer. No change to trainer beyond factoring `run_final_eval` as a callable outside `__init__`.

### A3. Lighting-stress OOD pools (multi-preset)

**What.** A parallel OOD source list using the same underlying videos as `external_youtube_avspeech`, `teams_ood_real`, `teams_ood_fake`, but with stronger, deterministic lighting distortions applied at eval time. **Three presets, each a separate OOD lane** — breadth over depth, because a single preset measures that preset only, not lighting robustness generally (see §8.1):

- `ood_lighting_stress_general` — `vcd_targeted_stress` (moderate composite distortion)
- `ood_lighting_stress_backlight_dim` — strong backlight + low overall exposure (home-office windows, dim rooms)
- `ood_lighting_stress_warm_harsh` — warm color cast + high contrast (over-lit lamp-lit environments)

Log under `ood_lighting_stress/<preset>/*` at each OOD eval, and under `final_eval/ood_lighting_stress/<preset>/*` in A2's final pass.

Deterministic means: fixed aug seed, per-frame apply probability = 1.0 (every frame gets the distortion, no sampling), fixed parameter values (no randomization). This gives a fair cross-run comparison AND breadth of stress modes.

**Files.**
- `DeepfakeBench/training/dataset/dataloaders.py:464` — aug gate is `if mode == 'train' and config['use_data_augmentation']`. Extend to honor `eval_aug_spec` when set on the source.
- `DeepfakeBench/training/data/sources/combined_paired.py:1808-2000` — `_build_external_ood_videos()`. Add per-source `eval_augmentation` field support.
- `DeepfakeBench/training/data/augmentations/pipelines.py:941-996` — add three presets (`vcd_targeted_stress`, `backlight_dim_stress`, `warm_harsh_stress`), each with fixed params, per-frame p=1.0, deterministic seed.
- RLP3 yamls — new `ood_monitoring.lighting_stress_sources` block listing all three presets, mirroring `external_real_sources` / `external_fake_sources`.

### A3b. Spatial-stress OOD pools (deterministic crop/shift/rotation)

**What.** The §1.4 hierarchy and the user's robustness bar call out spatial robustness alongside lighting. Packet 2 and v3 of this plan had no spatial-stress eval pool — slot 04 ("spatial" probe) adds a training aug but was originally scored on pools that don't test spatial perturbation. Add a matched eval pool so "is the model spatially robust" becomes measurable independently of "did we train spatially." See §8.2.

**Three presets, each a separate OOD lane** (same underlying videos as A3):
- `ood_spatial_stress_crop_shift` — deterministic ±8% horizontal + ±5% vertical crop shift on the face bbox before resize (off-center faces)
- `ood_spatial_stress_scale` — deterministic 0.9× and 1.1× scale variants (tighter / looser crops than training)
- `ood_spatial_stress_rotation` — deterministic ±6° affine rotation (head-tilt / camera-tilt)

Deterministic means fixed parameter values applied per-frame, not sampled. Log under `ood_spatial_stress/<preset>/*` and `final_eval/ood_spatial_stress/<preset>/*`.

**Why this is load-bearing for slot 04.** Without A3b, promoting slot 04 means promoting "trained with spatial aug, measured on non-spatial pools" — an identity claim. With A3b, slot 04 promotion requires `final_eval/ood_spatial_stress/*` > slot 02 by more than σ_seed on ≥2/3 presets. See §4.4.

**Files.**
- `DeepfakeBench/training/data/augmentations/pipelines.py` — add three spatial presets alongside A3's lighting presets.
- `DeepfakeBench/training/data/sources/combined_paired.py:1808-2000` — same `eval_augmentation` hook as A3; bbox-level crop shift implemented before the standard resize.
- RLP3 yamls — new `ood_monitoring.spatial_stress_sources` block.

### A4. Proper-data artifact `build_id` in W&B config

**What.** Extract `wave_id` / build timestamp from the loaded proper-data manifest and write into `data_stats['proper_data_build_id']`, which already flows to W&B config. Prevents silent mismatches like packet-1 (live proper-fake count 684 vs yaml-documented 442).

**Files.**
- `DeepfakeBench/training/data/sources/combined_paired.py:3825-3854` — at manifest load, parse `wave_id` from manifest header and add to `data_stats`.
- `DeepfakeBench/training/trainer/trainer.py:50-90` — verify `data_stats` forwarded to W&B config (already is; confirm during smoke).

### A5. Split-mode caveat documentation

**What.** Not code. One pinned block at the top of the packet-1 results doc and the new packet-3 doc explaining that packet 1 used `identity_split_mode: shuffle` (legacy default) while packet 2+ uses `hash_stable`, so any packet-3-vs-packet-1 AUC delta has an unquantified split-mode component. Packet-3 treats `RLP3_01` as its own baseline.

**Files.**
- `DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_RESULTS_AND_PLANNING_CONTEXT_2026-04-21.md` — add "Split-mode caveat" block at top.
- `DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_EXPERIMENT_PLAN_2026-04-21.md` — new doc containing the relevant sections of this plan.

### A6. Enhanced / unenhanced slice evaluation (highest-leverage change)

**What.** The single most important code change for interpreting packet 3. Two pieces.

**Piece 1 — enhanced proper-data as OOD monitoring lanes for every run, independent of training.** New OOD sources:
- `ood_enhanced_proper_clean` — drawn from the enhanced-clean lane of the proper-data artifact
- `ood_enhanced_proper_teams` — drawn from the enhanced-teams lane

All 8 packet-3 runs log AUC and jitter on these lanes regardless of whether they train on enhanced data. This is what lets us finally ask: "does unenhanced-only training capture enhanced content, or not?"

**Selection-bias carve-out (see §8.3).** These enhanced-proper lanes are **monitored but excluded from `ood_composite`**. They flow into `value_composite` (readout) and `final_eval` (readout), but not into the AUC metric used to pick `best_ood_composite`. Without this, `best_ood_composite` could be chosen to maximize enhanced-proper AUC, making "unenhanced-only training captures enhanced content" unfalsifiable because the checkpoint was selected for exactly that. Carve-out brings these lanes closer to held-out status (A10 gives the fully held-out version).

**Piece 2 — slice existing OOD fake metrics by enhanced/unenhanced.**
- `wma_failure_fake` is explicitly enhanced → `ood/fake_enhanced/*`
- `teams_ood_fake` is mixed or unknown → log under its own name AND under a `fake_unenhanced` or `fake_mixed` bucket via a yaml-side mapping (don't force a false partition; where ambiguous, expose both raw and sliced).

Slice labels live in yaml, not trainer code — so method→slice mapping is a config choice per packet.

**Files.**
- `DeepfakeBench/training/data/sources/combined_paired.py:1808-2000` — extend `_build_external_ood_videos()` to accept proper-data enhanced lanes as OOD input (read from the same manifest that training proper-data uses).
- RLP3 yamls — new `ood_monitoring.enhanced_proper_sources` block (applies to ALL 8 slots uniformly).
- `DeepfakeBench/training/trainer/trainer.py:2355-2376` — in OOD aggregation, emit per-method AUC under both the raw name and a coarser slice, with slice labels read from yaml config.

**Why this is the linchpin.** After A6 lands, the `RLP2_02` vs `RLP2_03/04` debate is readable for the first time. If `RLP2_02` retrospectively scores well on `ood/enhanced_proper_*`, then unenhanced training already captures enhanced content and enhanced training adds nothing. If `RLP2_02` scores poorly on the enhanced slice while `RLP2_03/04` score better there but worse on the unenhanced slice, we are looking at a capacity tradeoff — and slot 08 (dose-matched enhanced) tests whether the tradeoff disappears at balanced dose.

### A7. Summary namespace + agent-reviewable artifacts

**What.** Two pieces.

**Piece 1 — `summary/` namespace.** At each OOD eval and in A2's final pass, duplicate-log exactly these 10 decision-driving metrics under `summary/...`:

| Summary key | Source |
|---|---|
| `summary/val_holdout/auc` | A6-aware val_holdout AUC |
| `summary/ood/overall/auc` | existing OOD AUC |
| `summary/val_primary/ood_composite` | existing composite |
| `summary/ood_lighting_stress/auc` | A3 |
| `summary/ood/enhanced_proper/auc` | A6 (mean of clean + teams lanes) |
| `summary/ood/score_jitter_max/teams_ood_fake` | A1 |
| `summary/ood/score_jitter_spike_rate_0p3/teams_ood_fake` | A1 |
| `summary/value_composite` | A9 (defined below) |
| `summary/final_eval/test/auc` | A2 (fires once) |
| `summary/best_checkpoint/step` | existing |

A default W&B filter on `summary/*` shows only these. The ~50 verbose per-method metrics remain but are filtered out of the default view.

**Piece 2 — per-video scores as W&B Artifacts.** At training end (after A2 runs), write a single `per_video_scores.parquet` with one row per `(pool, method, video_id)` holding `prob_per_frame` (list), `mean_jitter`, `max_jitter`, `p95_jitter`, `spike_count_0p3`, `mean_prob`. Upload as W&B **Artifact** (downloadable), not Table (bloats run view). Post-run, a review script pulls these, flags outlier videos (e.g. `max_jitter > 0.5`, mean-prob spread > 0.7), and writes flags into W&B run comments.

**Files.**
- `DeepfakeBench/training/trainer/trainer.py` near 2376 — `summary/` duplicate log block.
- `DeepfakeBench/training/trainer/trainer.py:1573-1577` — after A2 final eval, assemble and log per-video artifact.
- New: `DeepfakeBench/training/tools/review_run_artifacts.py` — agent-facing, pulls per-video parquet, emits flags.

### A8. Move first-OOD-step earlier for FT runs

**What.** **Correction to v1/v2 of this plan.** The current policy is "first OOD at step 5000, every 500 thereafter." For scratch (30k steps) that is reasonable. For FT (10k steps, warmup=400), first-OOD-at-5000 is the back half of training, and 2 of 6 packet-2 best checkpoints landed exactly at step 5000 (the first OOD read). We are likely missing earlier true peaks.

Revised policy:
- **FT runs: first OOD at step 1000, every 500 thereafter.** (Warmup ends at 400; add ~600 steps for peak-LR stabilization.)
- **Scratch runs: first OOD at step 5000, every 500 thereafter.** (Unchanged.)
- Expose as two yaml knobs: `ood_monitoring.first_ood_step` and `ood_monitoring.ood_cadence`.

Cost: ~8 additional OOD evals per FT run × ~60s per eval ≈ 8 min additional wall time. Negligible.

**Files.**
- `DeepfakeBench/training/trainer/trainer.py:1550-1600` — the OOD-eval gate. Add adaptive first-step/cadence using the two new knobs.
- RLP3 yamls — expose the knobs; default FT to `1000 / 500`.

### A9. `value_composite` readout metric (deployment-hierarchy-aligned)

**What.** A metric computed at each OOD eval and in A2's final pass, aligned to the §1.4 value hierarchy.

**Formula:**

1. **Fixed operating point (with worst-pool gate, see §8.4).** Compute the threshold `τ` such that BOTH conditions hold:
   ```
   pools = [FPR_df40_real, FPR_external_youtube_avspeech, FPR_zoom_vcd_real,
            FPR_teams_ood_real, FPR_proper_clean_real, FPR_proper_teams_real]
   mean_FPR(τ) = mean(FPR_i(τ) for FPR_i in pools)
   max_FPR(τ)  = max(FPR_i(τ)  for FPR_i in pools)
   ```
   - `mean_FPR(τ) == 0.02` (bisection on the held pool probs)
   - `max_FPR(τ) ≤ 0.04` (worst-pool FPR may not exceed 2× the mean target)

   **If no τ satisfies both,** log `value_composite = NaN` with `value_composite_blocked_by = "worst_pool_fpr"` in W&B summary, and record `max_FPR_at_mean_02 = <observed>`. Do not silently pick a looser τ — the NaN is the signal that the model is hiding a spiky real pool.

   **Why the max-gate.** §1.4 says "minimum FPR across all real pools, equally weighted." A pure mean satisfies the arithmetic but lets 0%, 0%, 0%, 0%, 0%, 12% pass. The 0.04 ceiling (2× the mean target) is the simplest guard; revisit the ceiling in packet 4 with observed per-pool spreads.
2. **Teams fakes TPR** at `τ`. Equal-weight mean of:
   - deeplive fakes (from val_primary deeplive lane)
   - visomaster_unenhanced_teams fakes (from val_primary + proper_teams unenhanced lanes)
   - visomaster_enhanced_teams fakes (A6 `ood_enhanced_proper_teams`)
   - `teams_ood_fake` (the live-cropped Teams lane)
3. **Other fakes TPR** at `τ`. Equal-weight mean of:
   - `wma_failure_fake` (A6 slices as fake_enhanced)
   - visomaster_clean fakes (unenhanced + enhanced)

   **df40 training fake methods are excluded from the composite** (see §8.5). simswap, facedancer, blendface, e4s, inswap, mobileswap, uniface are training-distribution — including them measures memorization, not generalization, and inflates `value_composite` for any decent model. They are still logged under `summary/training_fakes/tpr` as a separate readout; if packet-3 shows they vary meaningfully across arms, that's evidence they carry signal and should be reinstated with a lower weight in packet 4.
4. **Stability term.** `stability = 1 - clip(max(score_jitter_max across teams_ood_fake, teams_ood_real, external_youtube_avspeech), 0, 1)`.
5. **Composite.**
   ```
   value_composite = 0.6 * TPR_teams_fakes + 0.3 * TPR_other_fakes + 0.1 * stability
   ```

**Explicit readout-only policy (per user direction):**
- `value_composite` is **logged at every OOD eval and in `final_eval/`**.
- Checkpoint selection **stays on `best_ood_composite`** through packet 3. This preserves apples-to-apples comparability with packet-2 runs, which were selected the same way.
- A2b retroactively computes `value_composite` for `RLP1_01`, `RLP1_04`, `RLP2_01`, `RLP2_02`, `RLP2_06` so we have comparable history.
- If packet-3 `value_composite` ranks arms sensibly (the ranking is stable across seeds 737 and 239, and the top-`value_composite` arm also has strong `final_eval/*` numbers), **packet 4 switches checkpoint selection to `value_composite`**.
- If `value_composite` behaves erratically (e.g. flips ranking across the two seeds, or rewards arms we know are bad), we diagnose and either reweight or discard it; selection stays on `best_ood_composite`.

**Defaults encoded:**
- FPR operating point: `0.02` — user-confirmed.
- Teams-fake sub-weights: equal — user-confirmed.
- Fake family set in step (2): chosen from the §1.4 hierarchy + the A6 slices we're adding.
- Stability weight: 0.1 — conservative. If packet 3 shows jitter variance dominates meaningful AUC variance, revisit.

**Files.**
- `DeepfakeBench/training/trainer/trainer.py` — new helper `compute_value_composite(eval_state) → dict`, called inside the OOD aggregation block and inside A2's final eval.
- `DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_EXPERIMENT_PLAN_2026-04-21.md` — document the formula, weights, operating point, max-FPR gate, and the readout-only policy so future agents don't silently re-weight it.

### A10. Held-out OOD partition (validation-bias-free final_eval bench)

**What.** Every packet-3 OOD pool gets a **video-id-level 90/10 partition**:
- 90% → monitored during training, flows into `ood_composite` and `value_composite` as today.
- 10% → **never logged during training, never in any selection metric**. Evaluated only once, inside A2's end-of-training final_eval pass, under `final_eval/heldout/<pool>/*`.

This is the single bench that answers question D: a set of numbers that the checkpoint was definitionally not selected against. See §8.6.

**How the partition is chosen.**
- Partition hash: `blake2b(video_id, digest_size=8)`. Lowest 10% by hash → held-out.
- Fixed across all 8 packet-3 runs. Same hash rule applied retroactively by A2b to packet-1/2 runs.
- Partition seed + hash-rule logged in W&B config so a future critic can reproduce it.

**Magnitude of selection-bias estimate.** For each pool, compute `bias(pool) = auc_monitored_90(pool) − auc_heldout_10(pool)` on the same checkpoint. A small bias (<0.005) means the selection metric is not over-fitting that pool. A large bias (>0.02) means it is, and that pool's headline numbers should be discounted. This is packet 3's first direct measurement of "how much validation bias is actually in our reported AUCs."

**Subtle gotcha for cross-packet comparison.** Packet-3 checkpoints were selected against the 90%-monitored partition, so their `final_eval/heldout/*` numbers are unbiased. Packet-1/2 checkpoints were selected against 100% of the pool (no split existed then), so their `final_eval/heldout/*` numbers are still slightly upward-biased — the checkpoint DID see that 10% during selection. The bias is modest (held-out is 10%, pool-wide metrics are averages), but not zero. A careful comparison should report both `final_eval/heldout` and `final_eval/monitored` for packet-1/2 re-evals, and flag if the 10% bias estimate exceeds σ_seed.

**A6 interaction.** Enhanced-proper lanes (A6) are pool-level carved out of `ood_composite` already. They also get the A10 90/10 split, so we have two layers of held-out signal on enhanced: (i) full-lane out of selection, (ii) 10% of the lane out of even `value_composite` readout.

**A7 interaction.** `per_video_scores.parquet` includes a `split: monitored|heldout` column so post-run review can filter.

**Cost.** 10% of each OOD pool's videos stop flowing into training-time OOD evals — minor data loss. One extra eval pass at training end per held-out pool; total wall-time overhead < 2 min per run.

**Files.**
- `DeepfakeBench/training/data/sources/combined_paired.py:1808-2000` — `_build_external_ood_videos()`. Compute partition hash; return `(monitored_videos, heldout_videos)` per source.
- `DeepfakeBench/training/trainer/trainer.py` — OOD eval path uses `monitored_videos` only. A2 final_eval loops over `heldout_videos` separately and logs under `final_eval/heldout/<pool>/*`.
- `DeepfakeBench/training/tools/retroactive_final_eval.py` (A2b) — apply the same hash-partition when re-evaluating packet-1/2 checkpoints so cross-packet numbers are on matched held-out slices.
- `DeepfakeBench/training/tools/retroactive_final_eval.py` — **pool-superset assertion.** When re-running old checkpoints, assert every video-id in the old run's monitored pool is still present in the current pool (or was moved to held-out by the A10 hash). If the current pool is strictly smaller, flag the A2b number as `not_comparable=true` in W&B. Catches the "current pools are easier than prior pools" trap.

---

## 3. Part B — The Eight Experiments

### 3.1 Shared scaffold

Unless a slot says otherwise, every packet-3 run inherits:

- backbone: `vit_b_16_laion_datacomp` / `ViT-B-16-DataComp-XL`
- rank: `736`, `k=32`
- frames per video: `8`
- frames per batch: `32`
- resolution: `224`
- train / val / test split: `0.85 / 0.10 / 0.05`
- base checkpoint: Track C `R12_G_FP32`
- LR: `3e-5`
- total steps: `10000`
- warmup: `400`
- ArcFace schedule: `s 6 → 12`
- checkpoint selection: `best_ood_composite` (unchanged — see A9)
- gradient clipping: `1.0`
- `combined_paired.identity_split_mode: hash_stable`
- OOD cadence: `first_ood_step: 1000`, `ood_cadence: 500` (A8)
- OOD monitoring includes A3 lighting-stress lanes and A6 enhanced-proper lanes for **every run**
- Final eval (A2) + per-video artifact (A7) run at training end for every run
- seed: `737` unless varied

**Proper-data builder rerun immediately before launch.** All 8 runs share one manifest, one `wave_id`, pinned via A4.

### 3.2 The slate

Launch order = priority order. If slots are dropped, drop from the bottom.

| Slot | Config | Differs from `02` by | Question |
|---|---|---|---|
| `01` | `R13_RLP3_01_FT_control.yaml` | no proper-data (pure clean FT baseline) | packet-3 own baseline — did the `RLP2_01` drop persist or recover? |
| `02` | `R13_RLP3_02_FT_proper_main.yaml` | (main recipe, seed `737`) | reconfirm `RLP2_02` under full A1–A9 instrumentation |
| `03` | `R13_RLP3_03_FT_proper_low_arcface.yaml` | ArcFace `s 6→10` | does low-ArcFace stack with the main data bet, or was `RLP2_06` just `02` in disguise? |
| `04` | `R13_RLP3_04_FT_proper_spatial.yaml` | +spatial stability aug | does spatial stack, or was `RLP2_05` just `02`? |
| `05` | `R13_RLP3_05_FT_proper_low_arcface_spatial.yaml` | +low-ArcFace AND +spatial | do the stability probes compound or trade off? |
| `06` | `R13_RLP3_06_FT_proper_main_seedB.yaml` | seed `239` | seed variance on main hypothesis (with slot 02 → 2 seeds) |
| `07` | `R13_RLP3_07_FT_proper_lighting_aug.yaml` | +vcd_targeted lighting aug (stronger preset than baseline) | does lighting aug lose **when evaluated on the lighting-stress pool** (A3), or only on the previous unfair eval? |
| `08` | `R13_RLP3_08_FT_proper_main_plus_dose_matched_enhanced.yaml` | + `proper_visomaster_enhanced_teams` at ~342 rows (1:1 with unenhanced teams dose) | does dose-matched enhanced-proper recover enhanced-slice performance without hurting unenhanced-slice performance? |

### 3.3 Design logic per slot

- **`01` resolves the drift question.** If `RLP3_01` matches `RLP2_01` (~0.9866), drift is real and persistent — audit mutable sources before packet 4. If it rebounds toward `RLP1_01` (~0.9891), packet-2 control was an outlier and the packet-2 story is stronger than it looked. Either way, `RLP3_01` is the packet-3 baseline.
- **`02` and `06` anchor seed variance.** Two seeds is the floor. If the gap between them is larger than the gap between slots 03/04/05 and `02`, the stability-probe story is noise. If seeds disagree widely, budget a third seed in packet 4 rather than overloading packet 3.
- **`03` / `04` / `05` test whether stability probes stack with the main bet.** Packet 2 could not separate `05` (spatial) and `06` (low-ArcFace) from `02`. If packet-3 `03` and `04` land inside the 2-seed envelope of `02`/`06`, they add nothing. If `05` beats all seeds, stacking is real.
- **`07` tests lighting aug honestly** — only meaningful because A3 lands first. Prior conclusion ("lighting aug is worse") was drawn against eval pools with no lighting stress, so the old answer cannot be trusted. Slot `07` on the A3 lighting-stress pool gives the fair read.
- **`08` answers the enhanced/unenhanced coverage question directly.** Packet 2 added enhanced at 2.2× dose; slot 08 adds enhanced at 1:1 dose within the Teams slice (~342 enhanced_teams + 342 unenhanced_teams + 342 unenhanced_clean). Combined with A6's enhanced-proper OOD lanes (which evaluate ALL runs including `02`), this is the clean test. Possible outcomes:
  - Tie with `02` on both unenhanced and enhanced slices → unenhanced-only captures enhanced; no further work needed.
  - Tie on unenhanced, wins on enhanced → dose-matched enhanced is the new recipe; promote in packet 4.
  - Loses on unenhanced the way `RLP2_03/04` did → there is a real capacity conflict; next packet explores smaller enhanced doses or multi-head setups.

### 3.4 What packet 3 is NOT doing

- No scratch slot. Packet 1 resolved scratch is not competitive on the balanced metric.
- No hints. Packets 1 and 2 agreed hints are out.
- No full-proper packet. Packet 1 rejected this dose.
- No WT-C GammaUp / Teams-shadow sidecars. Packet 1 rejected these.
- No generic `stability_lambda`. Repo history rejected this.
- No third seed. Deferred to packet 4 if needed.
- No checkpoint-selection swap to `value_composite`. Deferred to packet 4 (see A9).

---

## 4. Part C — Verification

### 4.1 Pre-launch smoke (before firing the 8 runs)

1. Run `tests/test_phase4_family_pipeline.py` and `tests/test_build_visomaster_proper_data_artifacts.py`. Confirm pipeline integrity.
2. Launch `R13_STARTUP_SMOKE_PROPER_DATA_WTF_PROVISIONAL.yaml` as a ~500-step smoke (long enough to pass step 1000 if you want to verify A8 first-OOD-at-1000). In W&B, confirm:
   - [ ] `ood/score_jitter_max/<m>`, `ood/score_jitter_p95/<m>`, `ood/score_jitter_spike_rate_0p3/<m>` all present (A1)
   - [ ] `val_holdout/score_jitter_max/<m>` present (A1 mirror)
   - [ ] Frame-pair delta histogram appears at step 2500 (A1 gated histogram — need a longer smoke to verify)
   - [ ] `final_eval/<pool>/*` block fires exactly once at training end (A2) — this requires the smoke to reach the end-of-training hook; if short, confirm A2 plumbing via unit-level test instead
   - [ ] `ood_lighting_stress/overall/auc` present (A3)
   - [ ] `ood_lighting_stress/score_jitter_max/<m>` present (A3 × A1 interaction)
   - [ ] `data_stats.proper_data_build_id` populated in W&B config tab (A4)
   - [ ] `ood/enhanced_proper_clean/auc` and `ood/enhanced_proper_teams/auc` present (A6)
   - [ ] `ood/fake_enhanced/auc` present (A6 slicing)
   - [ ] `summary/*` 10-metric panel visible (A7)
   - [ ] `per_video_scores.parquet` artifact uploads at end (A7) — needs end-of-training
   - [ ] First OOD eval fires at step 1000, not step 5000 (A8)
   - [ ] `value_composite` appears in OOD summary (A9)
   - [ ] `identity_split_mode: hash_stable` prints in startup banner
3. Before launching all 8 runs, confirm the proper-data builder completed cleanly and the manifest `wave_id` is the one packet-3 yamls reference.

### 4.2 Live monitoring during runs

Watch only the `summary/*` panel and the following per-run health signals:
- `train/collapse/is_constant_output` should stay 0.
- `train/params_with_grad` should stay 145.
- `summary/val_holdout/auc` and `summary/ood/overall/auc` trajectories.
- `summary/value_composite` trajectory — we expect it to roughly track `ood_composite` but with more volatility on threshold changes.

Keep an eye on whether slots 02 and 06 (the two seeds) are producing consistent trajectories. Large early divergence is a signal to investigate before letting all 8 runs complete.

### 4.3 Post-training analysis

1. Run A2b retroactively on `RLP1_01`, `RLP1_04`, `RLP2_01`, `RLP2_02`, `RLP2_06` so their `final_eval/*` and `value_composite` numbers land in the same namespace as packet 3. **Only after this step** can we compare cross-packet numbers honestly.
2. For each packet-3 run, `tools/review_run_artifacts.py` (A7) pulls `per_video_scores.parquet`, flags any video with `max_jitter > 0.5` or mean-prob spread > 0.7, writes flags to W&B run comments.
3. Cross-check: training-time `ood/score_jitter_max/<m>` should match post-hoc `tools/teams_frame_policy_analysis.py` `max_jitter` column on the same checkpoint. Divergence means A1 and the policy tool disagree — investigate before trusting either.

### 4.4 Decision rules (promotion criteria)

The seed-variance envelope is defined as `|final_eval.ood_composite[02] − final_eval.ood_composite[06]|`. Call this `σ_seed`.

- **Slot 01 (baseline).** Report-only. Its purpose is framing, not promotion.
- **Slots 03, 04, 05 (stability probes) — shared requirement:**
  - `final_eval/by_ood_composite/ood_composite` within `σ_seed` of slot 02 (not worse)
  - `final_eval/by_ood_composite/ood/score_jitter_max/teams_ood_fake` beats slot 02 by more than `2 × σ_jitter_seed`, where `σ_jitter_seed = |max_jitter[02] − max_jitter[06]|` on the same pool. Calibrated from actual seed-to-seed noise, not hardcoded. See §8.8.

  Live jitter-mean wins alone do NOT count. The per-video max must move. **With only 2 seeds, σ_jitter_seed is a single-sample estimate and uncertain** — if σ_jitter_seed is comparable to the probe's effect, treat the probe as inconclusive and budget a third seed in packet 4.
- **Slot 04 (spatial probe) additionally:** `final_eval/by_ood_composite/ood_spatial_stress/*` beats slot 02 by more than `σ_seed` on **≥2 of 3 spatial presets** (crop_shift, scale, rotation). A spatial aug that doesn't improve the spatial-stress eval is not spatial robustness — it's jitter in the training signal.
- **Slot 05 (spatial + low-ArcFace):** Must pass BOTH slot 04's spatial-stress requirement AND slot 03's jitter-max requirement.
- **Slot 06 (seed B).** Not a promotion candidate. Its role is sigma. If `|02 − 06| > 0.005` (i.e., seed variance dwarfs the typical probe effect sizes we saw in packet 2), flag that packet-3 interpretation is fragile and queue a third seed in packet 4.
- **Slot 07 (lighting aug).** Promote lighting aug as default-for-packet-4 iff BOTH of:
  - `final_eval/by_ood_composite/ood_lighting_stress/*` beats slot 02 by more than `σ_seed` on **≥2 of 3 lighting presets** (general, backlight_dim, warm_harsh). Single-preset wins are rejected — breadth is the actual robustness claim.
  - `final_eval/by_ood_composite/ood_composite` within `σ_seed` of slot 02 (robustness isn't paid for by main-pool damage)
- **Slot 08 (dose-matched enhanced).** Promote iff ALL of:
  - `final_eval/by_ood_composite/ood/enhanced_proper/auc` beats slot 02 by more than `σ_seed`
  - `final_eval/by_ood_composite/ood/unenhanced/auc` loses by less than half the enhanced gain
  - `value_composite` ≥ slot 02's `value_composite` within `σ_seed` AND `value_composite` is not NaN (the max-FPR gate from A9 step 1 is not tripped)
- **A9 `value_composite` rank stability (packet-4 switch criterion).** If packet-3 ranking by `value_composite` matches ranking by `final_eval/by_ood_composite/ood_composite` to within `σ_seed`, AND the dual-checkpoint `final_eval/agree/same_step` is true for most slots, adopt `value_composite` as packet-4's checkpoint-selection metric. If rankings diverge badly OR `value_composite` is frequently NaN (max-FPR gate tripping), keep `best_ood_composite` and diagnose.
- **A10 held-out bias audit (gating the whole packet).** Compute `bias(pool) = auc_monitored_90 − auc_heldout_10` per pool per run. If the mean bias across pools exceeds `σ_seed` for slot 02, **all packet-3 AUC comparisons are compromised** — headline numbers are upward-biased by the selection loop more than the effect sizes we're trying to measure. In that case, re-read packet 3 off `final_eval/heldout/*` only and downweight the monitored numbers.

---

## 5. Critical Files (every path the next agent will touch)

| File | Change | Item |
|---|---|---|
| `DeepfakeBench/training/trainer/trainer.py` | jitter helper + call from OOD and val_holdout paths (2295-2299, 1737-1758); summary namespace + per-video artifact with `split` column (near 2376, 1573-1577); **dual-checkpoint** final eval on A10 held-out (1573-1577); OOD cadence knobs (1550-1600); `compute_value_composite` helper (with max-FPR gate); dual best-checkpoint tracking | A1, A2, A7, A8, A9 |
| `DeepfakeBench/training/data/sources/combined_paired.py` | separate test_loader (4305-4310, 4456); extend `_build_external_ood_videos()` for eval_aug, enhanced-proper lanes (carved out of `ood_composite`), and A10 90/10 hash partition (1808-2000); extract `wave_id` into data_stats (3825-3854) | A2, A3, A3b, A4, A6, A10 |
| `DeepfakeBench/training/dataset/dataloaders.py` | honor `eval_aug_spec` (464) | A3, A3b |
| `DeepfakeBench/training/data/augmentations/pipelines.py` | add three lighting presets (`vcd_targeted_stress`, `backlight_dim_stress`, `warm_harsh_stress`) and three spatial presets (`crop_shift`, `scale`, `rotation`), after 996 | A3, A3b |
| `DeepfakeBench/training/tools/retroactive_final_eval.py` | new script; applies A10 hash-partition retroactively; runs dual-checkpoint final_eval; asserts pool-superset | A2b, A10 |
| `DeepfakeBench/training/tools/review_run_artifacts.py` | new script | A7 |
| `DeepfakeBench/training/tools/teams_frame_policy_analysis.py` | read-only; reference for A1 jitter definition | A1 |
| `DeepfakeBench/training/experiments/phase2_round13/R13_RLP3_01_*.yaml` through `R13_RLP3_08_*.yaml` | 8 new yamls cloned from RLP2_02; each inherits A3 + A3b + A6 + A8 + A9 + A10 via a shared base | Part B |
| `DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_RESULTS_AND_PLANNING_CONTEXT_2026-04-21.md` | pinned split-mode caveat at top | A5 |
| `DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_EXPERIMENT_PLAN_2026-04-21.md` | new doc (this plan, reformatted) | A5 |
| `DeepfakeBench/training/docs/relaunch_handoffs/README.md` | add packet-3 entry | A5 |

---

## 6. Open Questions / Things To Revisit After Packet 3

- **Does `value_composite` with 2% FPR actually match deployment?** If production uses a different FPR target, the composite is re-weighted accordingly. Confirm with deployment ops before packet 4.
- **Teams fake sub-weighting.** Currently equal. If deployment traffic skews heavily toward one family (e.g. mostly deeplive, less visomaster), re-weight.
- **Stability term in `value_composite`.** Currently 0.1. Revisit after packet 3 based on whether jitter variance dominates or is dominated by AUC variance across arms.
- **Third seed.** If `|02 − 06|` is large (> 0.005 on ood_composite), packet-4 budgets seed C and possibly D.
- **Enhanced clean.** Slot 08 only tests enhanced-**teams** at dose-matched. If the dose-matched story works, a follow-up tests enhanced-**clean** at dose-matched in packet 4. If it doesn't, enhanced is closed.
- **Mutable-source audit.** If slot 01 confirms the `RLP2_01` drop is real, we need an explicit audit of `deeplive`, `teams`, and `external_training_reals` to quantify what drifted between 2026-04-19 and 2026-04-20.
- **Temporal-shift and same-frame determinism stability probes.** Packet 3 measures jitter across subsampled frames from pre-extracted videos. It does not measure (a) determinism across repeated forwards on the same frame, or (b) stability under sub-second temporal shifts (sampling window ±1 frame). For true streaming-deployment stability, both matter. Deferred to packet 4 if packet-3 jitter numbers show probes doing real work.
- **Max-FPR ceiling calibration.** A9's `max_FPR ≤ 0.04` gate is a first guess (2× the 0.02 mean target). If packet-3 runs consistently trip the gate (all slots log `value_composite = NaN`), the ceiling is too tight and must be raised. If no slot ever trips it, the gate is not binding and may not be needed. Record `max_FPR_at_mean_02` in every run so packet 4 can re-tune from observed per-pool spreads.
- **A10 held-out slice size.** 10% is ~4-20 videos per pool depending on pool size. If variance on the held-out slice dominates the bias signal we're trying to measure, revisit partition size (15% or 20%) in packet 4.

---

## 7. Quick-start for the Next Agent

1. Read §1 (context), §1.4 (value hierarchy), §2 (all A items including A10), §3.2 (the slate), §8 (revision log — understand WHY each recent change was made before changing it back).
2. Land **A10 → A1 → A2 → A2b → A4 → A6 → A7 → A8 → A9 → A3 → A3b → A5** in that order. A10 first because it changes the OOD pool wiring — landing it early avoids reworking A1/A2/A6's OOD-eval code paths. A3b alongside A3 because they share aug-pipeline plumbing. A5 is doc-only.
3. Run the §4.1 pre-launch smoke and tick every checklist box — including the new A3b spatial-stress lanes, A10 held-out slices, and A9 max-FPR gate behavior.
4. Rerun the proper-data builder; confirm the `wave_id` is the one all 8 yamls reference.
5. Launch the 8 runs in priority order (01 → 08). Use the live-monitoring panel (§4.2).
6. After all 8 finish, run A2b retroactively on the packet-1 / packet-2 runs listed in §4.3 — with A10 hash-partition applied. Check the pool-superset assertion passes; if not, flag cross-packet numbers as not-comparable.
7. Compute the A10 bias audit (§4.4) BEFORE reading other decision rules. If bias exceeds σ_seed, all AUC comparisons must be re-read off `final_eval/heldout/*`.
8. Apply §4.4 decision rules. Produce a packet-3 results doc in `docs/relaunch_handoffs/` summarizing which arms promoted, which didn't, and what packet-4 should do.

Everything in this plan is legible without reading packet-1 or packet-2 docs first, but if the next agent wants deeper context, the read order is:
1. `R13_RELAUNCH_TRAINING_SOURCE_OF_TRUTH_2026-04-21.md`
2. `R13_RELAUNCH_PACKET1_RESULTS_AND_PLANNING_CONTEXT_2026-04-21.md`
3. `R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN_2026-04-21.md`
4. This document.

---

## 8. Critic Revision Log — 2026-04-21

This section records every change made to this plan in response to the outside-critic review conducted 2026-04-21 against v3. Goal: let a future critic identify, evaluate, and reverse any decision below that turns out to be wrong. Each entry follows the format **what changed → why → risk if wrong → how to reverse.** Critique questions from the review (A/B/C/D below) were:

- **A.** Are we learning target-domain success, priority-ordered?
- **B.** Are we learning whether the model is stable (jitter)?
- **C.** Are we learning lighting and spatial robustness — not just trained, but actually robust?
- **D.** Do we have a single test bench not tied to training, free of validation bias?

The review concluded A was partial, B was narrower than the goal, C covered lighting weakly and spatial not at all, and D was presented as solved but wasn't. The changes below close those gaps. Items NOT changed are called out at the bottom of this section.

### 8.1 A3 extended from 1 preset to 3 lighting presets (answers C)

**Changed.** `vcd_targeted_stress` was the only lighting distortion. Added `backlight_dim` and `warm_harsh` as separate OOD lanes. Slot 07 promotion (§4.4) now requires improvement on ≥2/3 presets.

**Why.** §1.4's robustness bar is real-life lighting the user cannot enumerate. A single preset measures that preset only; a model can memorize a single distortion. Three presets with distinct characters (moderate, dim-backlight, warm-harsh) is the minimum to distinguish "model is robust" from "model memorized this pattern."

**Risk if wrong.** Three presets slightly increase lighting-eval payload (wall time <1 min per run). If the three I picked are poorly chosen (e.g., all fall in one region of the lighting-variation space), slot 07 is judged on a bad benchmark. Specifically, flicker and colored-gel lighting are not covered.

**How to reverse.** Drop `backlight_dim` and `warm_harsh` from `ood_monitoring.lighting_stress_sources`; restore slot 07's rule to single-preset.

### 8.2 New A3b — spatial-stress OOD pools (answers C)

**Changed.** Added three deterministic spatial eval pools (crop_shift ±8%/±5%, scale 0.9×/1.1×, rotation ±6°). Slot 04 promotion rule in §4.4 now gated on `ood_spatial_stress/*` improvement on ≥2/3 presets, not just `teams_ood_fake` jitter.

**Why.** The v3 plan trained a spatial probe (slot 04) but had no eval pool that rewards spatial robustness. The user explicitly called this gap out ("the same applies if the face is cropped slightly more to the right, left, or in another manner"). Matched eval makes slot 04's claim testable.

**Risk if wrong.** Three parameter settings picked from heuristic; may over- or under-stress relative to real deployment variation. If deployment spatial variation is subtler than ±8% shift, A3b could reject a probe that would have worked (false negative). If it's harsher (e.g., full profile views), A3b could pass a probe that wouldn't help deployment (false positive).

**How to reverse.** Remove A3b's three lanes; revert slot 04 rule to v3 (teams_ood_fake jitter only).

### 8.3 A6 — enhanced-proper lanes carved out of `ood_composite` (answers A, D)

**Changed.** Enhanced-proper OOD lanes are monitored and flow into `value_composite` and `final_eval`, but NOT into the AUC-based `ood_composite` selection metric.

**Why.** Without this carve-out, `best_ood_composite` could be chosen to maximize enhanced-proper AUC, making "unenhanced-only training captures enhanced content" unfalsifiable because the checkpoint was selected for exactly that. Carve-out brings these lanes closer to held-out status — they become packet 3's cleanest signal for the §1.3 enhanced-vs-unenhanced debate.

**Risk if wrong.** If enhanced-proper belongs in the primary selection (e.g. if we later decide deployment distribution is 50/50 enhanced/unenhanced), leaving it out means packet 3 picks a checkpoint that underperforms on the enhanced slice. Mitigation: `value_composite` does include it, and A2's `by_value_composite` dual checkpoint logs the counterfactual.

**How to reverse.** Re-include enhanced-proper in `ood_composite` (one config flag in trainer.py).

### 8.4 A9 step 1 — added `max_FPR ≤ 0.04` gate (answers A)

**Changed.** `value_composite` only computes if `max(FPR_i) ≤ 0.04` at the chosen τ. Otherwise logs NaN with `value_composite_blocked_by = "worst_pool_fpr"` and records `max_FPR_at_mean_02`.

**Why.** §1.4 specifies "minimum FPR across all real pools, equally weighted" — that reads as worst-case, not mean. The v3 formula used a pure mean, which allows one pool to be arbitrarily spiky (0%, 0%, 0%, 0%, 0%, 12% hits mean=2%). A deployment model cannot ship if one real-class source has 12% FPR while others are 0%. The max-gate enforces the actual priority from §1.4.

**Risk if wrong.** The 0.04 ceiling is a guess (2× the 0.02 mean target). If typical pool spread at `mean_FPR=0.02` is wider than 0.04, every run logs NaN and the metric becomes useless. Conversely, if 0.04 is too loose, it admits models with a truly bad pool. Logging `max_FPR_at_mean_02` per run lets packet 4 re-tune from real data. If packet 3 sees >50% of runs trip the gate, the ceiling is too tight.

**How to reverse.** Remove the max-gate (revert to pure mean), or raise ceiling to 0.06 if 0.04 blocks everything.

### 8.5 A9 step 3 — dropped df40 training methods from "Other fakes TPR" (answers A)

**Changed.** "Other fakes TPR" sum now includes only `wma_failure_fake` and visomaster_clean fakes. df40 training methods (simswap, facedancer, blendface, e4s, inswap, mobileswap, uniface) are logged separately under `summary/training_fakes/tpr` but do not enter `value_composite`.

**Why.** These are in the training distribution. A well-fit model scores high on them regardless of OOD generalization; including them inflates `value_composite` for any decent model and dilutes the composite's ability to rank arms by deployment value.

**Risk if wrong.** If some df40 methods are meaningfully OOD to our fine-tuned model (e.g., we froze backbone layers early, rely on recent-gen fakes), we discard usable signal. Mitigation: they're still logged under `summary/training_fakes/tpr`; if packet-3 shows them varying meaningfully across arms, reinstate with a lower weight in packet 4.

**How to reverse.** Restore step 3 to v3's formulation; `summary/training_fakes/tpr` was computed all along so nothing is lost.

### 8.6 New A10 — video-id-level 10% held-out OOD partition (answers D)

**Changed.** Every OOD pool gets a deterministic `blake2b(video_id)` hash-partition; 10% is never logged during training, only at end-of-training `final_eval/heldout/*`. Retroactively applied to packet 1/2 runs via A2b. Per-pool selection bias measurable as `auc_monitored_90 − auc_heldout_10`.

**Why.** This is the direct answer to question D. Nothing in v3 was validation-bias-free — `final_eval` scored the selected checkpoint on the pools that drove selection, so numbers were upward-biased by an unknown amount. A10 gives the first actual measurement of per-pool selection bias. The §4.4 "A10 held-out bias audit" rule uses it to gate the entire packet's interpretation: if bias > σ_seed, AUC comparisons are compromised and must be re-read off held-out.

**Risk if wrong.**
- 10% is ~4-20 videos per pool depending on pool size. For small pools, the held-out slice may be too noisy to distinguish bias from variance; bias estimates would be unusable.
- Cross-packet comparison has an asymmetric bias: packet-3 checkpoints were selected against 90%-monitored (unbiased held-out numbers); packet-1/2 checkpoints were selected against 100% (slightly biased held-out numbers). The plan notes this subtlety; if the next critic misses it, they'll misread cross-packet deltas.
- Adding A10 to the OOD pool wiring is the largest structural change in this revision. If A10 has a bug, it blocks everything downstream. Mitigation: §7 implementation order puts A10 first so it's validated before A1/A2 depend on it.

**How to reverse.** Set partition percentage to 0%; all videos monitored, zero held-out. `final_eval/heldout/*` becomes empty. Everything else still works.

### 8.7 A2 — dual best-checkpoint tracking (answers A)

**Changed.** Two checkpoints saved per run (`best_ood_composite` and `best_value_composite`); final_eval runs on both; results logged under `final_eval/by_ood_composite/*` and `final_eval/by_value_composite/*`. A `final_eval/agree/same_step` flag records whether the two criteria chose the same training step.

**Why.** The v3 plan kept selection on `ood_composite` for packet-2 comparability, correctly noting that `value_composite` readout-only meant packet 3 might pick the wrong checkpoint by its own stated priorities. Dual-checkpoint resolves this at no training cost — packet 3 preserves comparability AND learns whether value-composite selection would have chosen a different step. This is the data packet 4 needs to decide whether to switch selection metrics.

**Risk if wrong.** Two checkpoints take 2× disk (not binding). If `best_value_composite` thrashes across training (because `value_composite` is noisier than `ood_composite` — which it might be, because it depends on a threshold τ that shifts with FPR changes), the "best" could land on a lucky spike. Mitigation: log `value_composite` trajectory and check monotonicity in the pre-launch smoke.

**How to reverse.** Track only `best_ood_composite`; delete `final_eval/by_value_composite/*` logging.

### 8.8 §4.4 jitter threshold — calibrated from σ_jitter_seed, not hardcoded 0.01 (answers B)

**Changed.** Stability-probe promotion threshold on `score_jitter_max/teams_ood_fake` is now `2 × σ_jitter_seed`, where `σ_jitter_seed = |max_jitter[02] − max_jitter[06]|`.

**Why.** The v3 threshold (`0.01 absolute`) was picked without calibration to seed-to-seed noise. A probe could "beat" slot 02 by 0.015 and still be pure seed variance. Calibrating from the actual slot-02-vs-slot-06 gap grounds the threshold in measured noise.

**Risk if wrong.** With only 2 seeds, σ_jitter_seed is a single-sample estimate with wide uncertainty. If the two seeds happen to land on nearly-identical max-jitter, the threshold becomes over-tight and all probes fail. If they happen to land far apart, the threshold becomes over-loose. Explicitly documented: if σ_jitter_seed is comparable to probe effect sizes, treat probes as inconclusive and budget a third seed in packet 4.

**How to reverse.** Restore hardcoded `0.01 absolute` threshold.

### 8.9 Implementation order updated — A10 first (supports §8.6)

**Changed.** Implementation order: A10 → A1 → A2 → A2b → A4 → A6 → A7 → A8 → A9 → A3 → A3b → A5.

**Why.** A10 changes the wiring of `_build_external_ood_videos()`. Every downstream item's OOD eval path depends on whether a video is monitored or held-out, so landing A10 first avoids reworking A1/A2/A6/A7's OOD-eval code paths.

**Risk if wrong.** A10 first means an A10 bug blocks everything else. Mitigation: A10 is small (one function change, one yaml knob) and independently testable on a 10-video fixture before integration.

**How to reverse.** Move A10 later; land A1/A2/A6 against v3 OOD wiring, retrofit A10.

---

### 8.10 What was NOT changed in response to the critique, and why

Each of these was considered during the revision and deferred intentionally.

- **`value_composite` weights (0.6 / 0.3 / 0.1).** Critic noted these are unvalidated. Left as-is because any reweighting is speculative before packet-3 data lands. Packet 4 retunes from observed per-run numbers.
- **Switching checkpoint selection from `ood_composite` to `value_composite` in packet 3.** Still deferred to packet 4. Dual-checkpoint (§8.7) gives the data to decide without breaking packet-2 comparability.
- **Temporal-shift / same-frame determinism stability probes.** Added to §6 as an open question. Not added to packet 3 because subsample-jitter is structurally the right shape; adding a new stability mechanism mid-packet is scope creep.
- **Third seed.** Deferred to packet 4, conditional on σ_seed and σ_jitter_seed magnitudes.
- **Temporal stability weight in value_composite (0.1).** Critic flagged as possibly too low. Left because we don't have data yet on whether jitter variance dominates AUC variance. Revisit in packet 4.
- **Raising A10 partition from 10% to 15-20%.** Considered but rejected — larger held-out steals more monitoring signal. Revisit in packet 4 if 10% is statistically underpowered.

---

### 8.11 Directions a future critic should re-examine first

In descending order of "most likely to be wrong":

1. **The A10 partition size (10%)** — pure heuristic; the first thing to check against observed bias-estimate variance after packet 3.
2. **A9's max_FPR ceiling (0.04)** — also heuristic; should be re-tuned from observed per-pool spreads after packet 3.
3. **A3 and A3b preset choices** — three presets each, each picked by the revising critic based on intuition about lighting/spatial variation modes. Real deployment distributions may not match.
4. **Dropping df40 training methods from `value_composite`** — defensible but discards in-distribution performance signal. If packet-3 shows df40 TPR varies across arms meaningfully, reinstate with lower weight.
5. **The dual-checkpoint cost model** — assumes `best_value_composite` is stable. If it thrashes in training, the second checkpoint may be worse than useless.

If packet 3 lands and any of the above shows up as a real problem, revisit §8.x for that item and follow the "how to reverse" path.
