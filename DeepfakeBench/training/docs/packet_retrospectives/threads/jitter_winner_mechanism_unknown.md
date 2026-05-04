# Thread: Jitter@0.50 winner — mechanism unknown

> **2026-04-30 afternoon, opened by working agent.** The night's overnight slate produced a clear value_composite leader (`mclioexb` = `face_scale_jitter@0.50` ONLY, FT-from-P8A_step5000) at 0.661, beating the P14 bundle (0.116) and DATA_FIX (0.126) by 5.7×. The morning interpretation was that jitter@0.50 closed the face-pixel-area shortcut, with a "Why" paragraph in memory `project_face_scale_jitter_load_bearing.md` that proposed two mechanisms: face-size invariance + bundle regularization conflict. Direct measurement the same afternoon refuted the face-size-invariance half; the bundle-regularization half survives. The thread tracks the open question that emerges: **what mechanism actually moved value_composite by 5.7× when the obvious target metric (face-size invariance on production-honest frames) shows the model got *worse*?**

## The question

The jitter@0.50 isolated training run won the 2026-04-29 → 2026-04-30 overnight slate at trainer-side `value_composite=0.661`, with `other_fakes_tpr=0.591` (cross-method generalization preserved) and best ckpt at step 1500. The recipe was identical to the P14 bundle except (a) `face_scale_jitter.scale_limit` 0.25 → 0.50, and (b) `anchor_aware` + `pipeline_randomization` were DISABLED. Two mechanisms could in principle explain the 5.7× lift over the bundle: (1) jitter@0.50 attenuated the face-pixel-area shortcut (its design target) more than jitter@0.25 did; (2) the dropped anchor_aware + pipeline_random interventions were net drag in the FT-from-P8A regime. Direct measurement on 2026-04-30 afternoon refutes (1) on the production-honest substrate. (2) survives but does not by itself explain why jitter@0.50 alone produced `value_composite=0.661` — only that the bundle was strictly worse than its strongest single component. **What is the actual mechanism by which jitter@0.50 alone reached 0.661?**

## Initial belief

Through the design and launch of `R13_P14_FACE_SCALE_JITTER_ISOLATED.yaml` (2026-04-29 evening), the working agents framed the test as "is face-scale-jitter the LOAD-BEARING lever in the P14 bundle?". The yaml header said so explicitly. When `mclioexb` came back at value_composite 0.661 vs the bundle's 0.116, the team and memory entry leapt directly from "jitter is the load-bearing lever" to "jitter@0.50 *attenuated the face-pixel-area shortcut*" — i.e., they conflated "this lever is load-bearing in the value_composite metric" with "this lever achieved its design intention (face-size invariance)". The morning memory `project_face_scale_jitter_load_bearing.md` and the morning P14.md "Headline finding" subsection both presented the face-size-invariance interpretation as the leading hypothesis.

## What changed our mind

- **2026-04-30 afternoon — Face-size invariance probe ran and refuted the design-intent interpretation** (`analysis/face_size_invariance_2026-04-30/face_size_invariance.py`, log `analysis/_logs_2026-04-30/face_size_invariance.log`). Same 180 production-honest frames × 5 tightnesses for all four overnight ckpts:

  | Ckpt | flip rate | median \|Δprob_fake\| |
  |---|---:|---:|
  | P8A baseline | 39.4% | 0.128 |
  | **`mclioexb` (jitter@0.50, value_composite WINNER)** | **43.9%** | **0.303** |
  | `w5tky6ss` (P15 GRL) | 22.8% | 0.124 |
  | `xan4dfto` (DATA_FIX) | 22.8% | 0.143 |

  Jitter@0.50 made the model *more* sensitive to crop tightness on production-honest frames, not less. The median |Δprob_fake| almost tripled. So whatever lever jitter@0.50 is actually pulling, it is NOT the face-size invariance the design targeted.

- **2026-04-30 afternoon — Domain-confusion linear probe shows GRL did NOT bite the [CLS] manifold either** (`analysis/domain_confusion_probe_2026-04-30/domain_probe.py`, output `outputs/probe_p8a_slot2_slot3/summary.json`). Macro-OVR AUC P8A=0.9999 ≈ SLOT2_GRL=0.9994 ≈ SLOT3_JITTER=0.9969 (2-class fallback because lockbox parquet omits df40). Per the README's own interpretation table, ≤0.55 = GRL worked, ≥0.85 = GRL no-bite. **0.9994 is no-bite by ~7 standard deviations.** The companion mechanism for P15's `value_composite=0.516` (which was its design target, i.e. "make the [CLS] domain-invariant") is also not what was advertised. Caveats: linear probe is L2-normalized direction-only (norm-encoded domain info would be invisible); domain 0 absent in lockbox so the test is 2-class not 3-class; webcam-codec vs studio-capture differ at the pixel level so even untrained CLIP probably saturates — a no-FT-CLIP control would tell us whether 0.999 is a model property or a data property and is not yet run.

- **2026-04-30 afternoon — Embedding triptych corroborates** (`analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/triptych_grid_tsne.png`). 3×3 grid: rows = P8A | SLOT2_GRL | SLOT3_JITTER; cols = real/fake | clip_capture_mode | face_pixel_area bucket. The arc geometry is broadly conserved across all three rows. Real/fake separation: clean in all three (sanity check passed). Capture-mode coloring: similar mixing pattern across rows (no row shows webcam collapsing into the rest). Face-size quartile coloring: similar quartile structure across rows (no row shows buckets bleeding together). **The gross [CLS] geometry is preserved across both interventions.** Whatever moved value_composite is not visible at the [CLS] manifold's gross axes.

- **2026-04-30 afternoon — DATA_FIX collapsed at `other_fakes_tpr=0.047`** (already documented in P14.md and [`viso_bucket_gap`](viso_bucket_gap.md)). This is the third datapoint where the night's value_composite ranking is decoupled from the design-intent metric: face-size invariance, GRL domain confusion, and bucket-fix headline-recall lift are all NOT what the rankings reflect.

## Current stance (2026-04-30 evening)

The 5.7× value_composite gap of jitter@0.50-isolated over the P14 bundle is real and is reproducible across four wandb summary metrics (`other_fakes_tpr`, `max_pool_fpr`, `max_fpr_at_mean_02`, `stability_max`). What is unknown is the *mechanism*. The two design-intent mechanisms have been refuted on direct probes (face-size invariance: refuted by flip-rate sweep; GRL domain confusion: refuted by linear probe + triptych geometry conservation). The promotion-contract scorecard (2026-04-30 evening, see new subsection below) provides the first direct evidence among the three candidate mechanisms — **mechanism (1) decision-boundary / calibration shift now has direct evidentiary support**:

1. **Decision-boundary geometry / calibration shift, not feature-space rearrangement.** ✅ **NOW SUPPORTED** by the scorecard's recall-FPR Pareto curve. At FPR≤0.02 max macro_recall is 0.128; at FPR≤0.07 max is ~0.27; at FPR≤0.10 max is ~0.35. The model's prob_fake distribution at the high-confidence tail has very few fakes — any τ tight enough to hold FPR≤0.02 destroys recall. The trainer's `value_composite` aggregates over τ-conditioned components that don't isolate the production operating point. **What this does NOT prove**: that the [CLS] features themselves shifted in a way specific to `mclioexb` vs `9lmvb5b4`-step5000 (P8A baseline). The feature-vs-head decomposition probe (~1h CPU, head-only features on the same 180 frames) is still warranted — it would tell us whether the failure is "P8A's features were already inadequate at this operating point and FT-from-P8A inherits the cliff" vs "FT-from-P8A reshaped the features in a specifically calibration-hostile way".
2. **Intermediate-layer feature shift, not [CLS]-layer.** Still plausible but less load-bearing under (1)'s evidentiary support. GRL may have flattened a deeper layer's domain-discriminability without affecting the final [CLS] readout; jitter@0.50 may have invariantized intermediate textures (rather than [CLS]-level face-size). **Probe**: extract features from earlier transformer blocks (resblocks 0-5 or 6-9) and re-run the linear probes there.
3. **Test-substrate mismatch.** Still the cheapest probe (already drafted) but its operational value is reduced: even if jitter@0.50 invariantizes face-size on a training-distribution-stratified substrate, the scorecard says the model still doesn't promote. The mechanism question for the value_composite ranking is informational; for the deployment question it is moot.

The thread is *open*; the night's value_composite numbers stand, but the **deployment story has crystallized**: jitter@0.50 wins the trainer-side ranking by some mechanism not yet identified, AND that win does not survive the calibration to the production operating point. The empirical recommendation in memory `project_face_scale_jitter_load_bearing.md` survives but is weakened to "use jitter@0.50 over 0.25 *because it's strictly less bad*, not because it gets you to deployment". See companion memory `project_mclioexb_does_not_promote_2026-04-30.md`.

## 2026-04-30 evening — Production-grade scorecard refutes the deployment side of the puzzle

**The promotion-contract scorecard for `mclioexb` ran 2026-04-30 evening** (Vertex job `2402672943822798848`, us-east1, image 1.3.233; succeeded 08:59:05Z; local artifacts `analysis/scorecard_mclioexb_2026-04-30/`). The verdict is **does not promote**: at the calibrated τ=0.975031, per-method dev recalls collapsed (visomaster 0.013, deeplive 0.024, teams_fake 0.347), macro 0.128 vs 0.70 floor.

A scan of the full 5549-point τ grid (`threshold_grid.csv`) shows the recall-FPR Pareto frontier:

| recall target | min FPR achievable | min stress FPR | implied τ |
|---:|---:|---:|---:|
| 0.20 | 0.0492 | 0.0657 | 0.9263 |
| **0.30** | **0.0845** | **0.1042** | 0.7872 |
| 0.50 | 0.1399 | 0.1727 | 0.4831 |
| 0.70 | 0.2266 | 0.2991 | 0.2129 |

**Even under the in-tree relaxed v3 contract** (recall_min=0.30, FPR=0.07, stress=0.10 — see [`contract_policy_bug`](contract_policy_bug.md)), zero τ qualifies. To hit recall=0.30 the model needs FPR≥0.085 and stress≥0.104, both above the relaxed gates. *(Note: this scorecard ran the DEFAULT policy because the launcher omitted `--promotion_target_fake_recall_min 0.30`; the v3 fix is still uncommitted in working tree. The verdict is robust either way — the recall-FPR Pareto curve is too shallow in the operating region to clear either gate set.)*

**Implication for the open puzzle:**

The scorecard is direct evidence for candidate mechanism (1) (decision-boundary / calibration shift). The model's prob_fake distribution at the high-confidence tail (τ>0.97) has very few fakes; the model's `value_composite=0.661` is computed from metrics that aggregate over τ-conditioned components that don't isolate the production operating point. At τ=0.5, the model is decent on `teams_fake_all_dev` (recall 0.678) but poor on `visomaster` (0.211) and mediocre on `deeplive` (0.558) — already off-target. Lifting τ to satisfy FPR drives all recalls toward zero.

This **does not** by itself prove the [CLS] features moved in a `mclioexb`-specific way (the FT-from-P8A track may inherit a calibration cliff that P8A also exhibits at this operating point). The head-only-vs-feature decomposition probe (the cheapest of the three originally catalogued candidates, ~1h CPU) would resolve that ambiguity.

**Reframing the open puzzle from "deployment" → "diagnosis-only" terms:**

The original framing of this thread was "what mechanism produced the 5.7× value_composite gap, and is it deployable?" The scorecard answers the deployment half with NO. What remains is a strictly diagnostic question:
- Why is the trainer-side `value_composite=0.661` so misaligned with the production-honest scorecard?
- Did jitter@0.50 isolated reshape the features in any way that's relevant to a future packet's design, even if this specific checkpoint isn't deployable?
- Does the head-only vs feature decomposition tell us whether the next-packet move should target the head (calibration / loss design) or the features (substrate / FT base)?

These questions matter for P16 design but are no longer load-bearing for "ship this".

## 2026-04-30 evening — Intermediate-layer probe refutes mechanism (2) in the strong form

Probe `analysis/intermediate_layer_probe_2026-04-30/intermediate_layer_probe.py` ran on local Mac/MPS over the same 800-frame stratified-lockbox substrate as the triptych. Hooks captured the `[CLS]` token at `transformer.resblocks[0,3,6,9,11]` of both `9lmvb5b4` step 5000 (P8A) and `mclioexb` step 500. Per-layer cosine similarity P8A↔mclioexb and 5-fold CV LR rec@FPR (oracle head per layer):

| layer | cos_mean | frac<.90 | Fp_AUC | Fm_AUC | Fp@.05 | Fm@.05 |
|---|---|---|---|---|---|---|
| 0  | 1.0000 | 0.000 | 0.7421 | 0.7421 | 0.370 | 0.370 |
| 3  | 0.9995 | 0.000 | 0.9096 | 0.9097 | 0.503 | 0.478 |
| 6  | 0.9959 | 0.000 | 0.9924 | 0.9929 | 0.969 | 0.963 |
| 9  | 0.9947 | 0.000 | 0.9783 | 0.9840 | 0.880 | 0.898 |
| 11 | 0.9733 | 0.043 | 0.9738 | 0.9764 | 0.827 | 0.843 |

Reading: **the FT reshape is concentrated at the last block.** Layers 0/3/6/9 are essentially identical between checkpoints (`frac<.90 = 0`); only layer 11 carries non-trivial shift (4.25%, with min cos = 0.573 on a few outlier frames). Per-layer oracle-head AUCs are within 0.006 across ckpts at every layer — `Fm` is marginally ahead at 6/9/11 but the gap is not load-bearing. Compared to the head/feature decomposition's 9.4% shift at the final pooler `[CLS]` (one block downstream of resblock 11), 4.25% at layer 11 is consistent within ~5%; the residual ~5pp lives in the LN+`visual.proj`+head path.

**Mechanism (2) verdict (uniform-stack feature reshape): refuted in the strong form.** The lever's effect lives in the very last transformer block plus the head; representations one block earlier are within numerical noise of P8A's. This is consistent with mechanism (1) (decision-boundary / calibration shift, head-localized) as the dominant story, and tightens the design lever for P16-and-after: head-side / final-layer interventions are where the lever bites; full-backbone interventions are over-budget.

Outputs: `analysis/intermediate_layer_probe_2026-04-30/outputs/{summary.json, per_layer_table.csv}`.

## Packet timeline

- [P13](../packets/P13.md) — first packet to introduce `face_scale_jitter@0.25` as part of the bundle. Verdict γ; cross-domain collapsed (the from-scratch trade); jitter's effect undiagnosed because the bundle was the test.
- [P14](../packets/P14.md) — P14_FT_FROM_P8A bundle (jitter@0.25 + anchor_aware + pipeline_randomization) at value_composite 0.116; sister-variant `R13_P14_FACE_SCALE_JITTER_ISOLATED` (jitter@0.50 ONLY) at 0.661. Single-lever ablation showed bundle was net-negative against jitter alone, but the *mechanism* by which jitter@0.50 wins is not what the design intended.
- [P15](../packets/P15.md) — P15_GRL_FROM_P8A at value_composite 0.516; structural infrastructure success (cross-domain head room preserved); but the canonical DANN test (linear-probe macro-AUC drop toward chance) failed — GRL's design-intent mechanism (collapse the [CLS] domain manifold) is also not what the value_composite ranking reflects.

## Evidence locations

- `analysis/face_size_invariance_2026-04-30/face_size_invariance.py` — the probe script.
- `analysis/face_size_invariance_2026-04-30/outputs/*_invariance.{csv,png}` — 4-ckpt outputs.
- `analysis/_logs_2026-04-30/face_size_invariance.log` — run log with all 4 verdicts.
- `analysis/domain_confusion_probe_2026-04-30/domain_probe.py` — the linear probe.
- `analysis/domain_confusion_probe_2026-04-30/outputs/probe_p8a_slot2_slot3/{summary.json, confusion_matrices.png, macro_auc_bar.png, per_fold_per_class_auc.csv}` — outputs.
- `analysis/embedding_triptych_2026-04-30/triptych.py` + `triptych_postprocess.py` — feature extraction + 2D reduction (the latter added because sklearn TSNE silently segfaulted on Apple Silicon when run in the same Python process as torch — see [`#environment-note`](#environment-note) below).
- `analysis/intermediate_layer_probe_2026-04-30/intermediate_layer_probe.py` + `outputs/{summary.json, per_layer_table.csv}` — per-resblock [CLS] cosine + per-layer oracle-head AUC for P8A vs mclioexb on the 800-frame stratified-lockbox substrate. 2026-04-30 evening run shows reshape concentrated at resblock 11 (frac<.90 = 4.25% there; 0.0% at 0/3/6/9).
- `analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/triptych_grid_tsne.png` — the headline figure.
- W&B summaries: `dtect-vision/phase2-experiments/runs/{mclioexb,w5tky6ss,xan4dfto}` for the trainer-side metrics that constitute the value_composite ranking.
- Memory: `project_face_scale_jitter_load_bearing.md` — its 2026-04-30 afternoon "Why" addendum revises the face-size-invariance interpretation as refuted; its 2026-04-30 evening update integrates the scorecard verdict.
- Memory: `project_mclioexb_does_not_promote_2026-04-30.md` — load-bearing scorecard verdict for the deployment half of the puzzle.
- Scorecard artifacts: `analysis/scorecard_mclioexb_2026-04-30/{promotion_winner.json, promotion_contract.json, threshold_grid.csv, scorecard.wide.csv, selected_threshold_scorecard.csv, scorecard.json}`; GCS `gs://training-job-outputs/test_results/teams_promotion_contract/teams-promotion-contract-mclioexb-step500-20260430-102421/`.

## Open loops

### Open loop: jitter-winner-mechanism-unknown
status: open
severity: low
first_seen: 2026-04-30
last_verified: 2026-04-30
close_criterion: at least one of the three candidate mechanisms (decision-boundary / intermediate-layer / test-substrate) is empirically supported on a probe whose design isolates that mechanism, AND the supported mechanism produces a measurable signature on `mclioexb` that does NOT also appear on `9lmvb5b4` step 5000 baseline at comparable magnitude — i.e., the mechanism is specific to the value_composite winner, not a feature shared with the FT base

The morning interpretation in memory `project_face_scale_jitter_load_bearing.md` ("Why (1)") proposed face-size invariance as the lever's mechanism. The afternoon direct measurement refuted that on the production-honest 180-frame substrate. The evening scorecard provides direct evidence for candidate mechanism (1) (decision-boundary / calibration shift) — see "2026-04-30 evening" subsection above. **Severity downgraded medium → low** (2026-04-30 evening): the deployment half of the puzzle is settled (the scorecard says `mclioexb` does not promote, and the bundle is presumptively also non-promotable), so the mechanism question is now purely diagnostic. The three candidate next probes remain catalogued (in increasing cost order): (a) head-only diff vs feature diff between bundle and jitter@0.50 on the same 180 frames (~1h CPU) — would resolve whether the calibration cliff is FT-from-P8A-inherited or `mclioexb`-specific; (b) intermediate-layer linear probe on the same domain-confusion question (~2h CPU); (c) face-size sweep on a training-distribution-stratified frame substrate (~2-3h CPU + parquet sampling). The cheapest probe (a) is the highest-value one because its answer informs P16 design (head-side vs feature-side lever).

### Open loop: jitter-on-training-substrate-not-yet-tested
status: open
severity: low
first_seen: 2026-04-30
last_verified: 2026-04-30
close_criterion: the face-size invariance sweep is re-run on a frame substrate stratified across the training face-pixel-area distribution (e.g., 200 frames sampled from `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` with stratification by face-area quartile), with the same 5-tightness grid and same 4 ckpts as the 04-30 afternoon run; if `mclioexb` flip rate ≤ 10% on the training-distribution substrate while still being > 10% on the production-honest substrate, mechanism (3) (test-substrate mismatch) is supported

This is the cheapest of the three candidate mechanism probes and the most likely-to-be-decisive: if jitter@0.50 was trained against a different face-size range than the production-honest 180-frame cache exercises, the negative result on this thread's primary substrate may be the test, not the model. Expected runtime ~2-3h CPU. The frame substrate is already inventoried (the parquet exists). The script (`face_size_invariance.py`) accepts an arbitrary `--frames_dir` so the new run is a drop-in.

## Cross-thread refs

- [`face_size_label_leak`](face_size_label_leak.md) — companion thread; its 2026-04-30 afternoon update is the source of the refuted-on-direct-measurement evidence in this thread. The face-size leak's open loop's flip-rate close criterion is now affirmatively NOT MET on the leader; this thread's open question is "if not face-size, then what?".
- [`anti_shortcut_bundle_decomposition`](anti_shortcut_bundle_decomposition.md) — captures the discipline rule (single-lever ablation slot when stacking interventions). This thread's question lives one level below: granted the bundle was net-negative against its single load-bearing component (the discipline-rule level), what is the mechanism by which the single component wins (the causal-story level).
- [`processing_signature_shortcut`](processing_signature_shortcut.md) — the camera-signature shortcut is the parent shortcut framing; jitter@0.50's failure to bite face-size at inference is one slice of "interventions don't *transfer* their design-intent target to the production-honest evaluation substrate."

## Environment note

The triptych script's `--reducer tsne` path crashes on Apple Silicon when sklearn TSNE is called in the same Python process as torch. Two crash modes observed: (1) `threadpoolctl` 2.2.0's `get_version()` `AttributeError` from `get_config().split()` returning None on Apple Accelerate; (2) silent SIGSEGV after threadpoolctl was upgraded to 3.6.0. Workaround: extract features in the torch process (cached as `analysis/_features_cache_2026-04-30/triptych_features__*__n800.npz`), then run TSNE + plotting in a torch-free postprocess process (`analysis/embedding_triptych_2026-04-30/triptych_postprocess.py`). This is the substrate-level fact that makes the triptych figure reproducible — future agents reproducing the figure should use the postprocess path, not the in-process TSNE call.
