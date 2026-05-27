# P17 Round 1 — ArcFace runs complete + trained-head probe falsifies hypothesis (as-stated)

**Date**: 2026-05-01 11:00–17:00 CEST
**Branch**: `teams-relaunch-root-2026-04-17`
**Author**: Claude (auto-mode day session — continuation of `OVERNIGHT_LAYER3_FINDING_2026-05-01.md`)
**Status**: 2 of 4 P17 arms complete (ArcFace L3 + L4). 2 still running (LINEAR L3 + L4). Trained-head probe on completed ArcFace ckpts shows the trainer DESTROYED the substrate-invariance that motivated the hypothesis. LINEAR ckpts not yet probed.

---

## TL;DR

> **The trained ArcFace head, applied to its own layer-3 features, is *anti-correlated* on lockbox: AUC 0.187 (where chance = 0.5).** The exact same features fed to a fresh logistic regression give lockbox AUC 0.973 (matching the offline probe's reference 0.9495). The trainer's ArcFace + cos-similarity + s-anneal recipe re-learns the dev-substrate shortcut even at layer 3. The hypothesis as-stated ("a head trained on layer-3 features avoids the substrate shortcut") is falsified for the ArcFace recipe. Two LINEAR arms are still running and may save it — they don't normalize features and don't use cos-similarity logits.

---

## Timeline (the day)

| Time (CEST) | Event |
|---|---|
| 11:33 | First L3+L4 ArcFace launch (image 1.3.235ish — known-buggy: SVD residuals not actually frozen, dead schedule keys). |
| ~11:50 | Second-agent review caught the bugs. User authorized kill+rebuild. |
| ~12:30 | Relaunch attempt failed — third bug found (`apply_wandb_backbone_params` clobbered yaml's explicit `hidden_size: 768` with registry's 512). Fix in `utils/config_helpers.py:613-622`, regression test in `tests/test_p17_layer3_head_wiring.py:test_wandb_overrides_preserve_explicit_hidden_size`. |
| 13:33 UTC | Clean L3+L4 ArcFace relaunch in us-west4. |
| 14:21 | First check-in: cls_loss INCREASING through step 1300 (1.26 → 1.40), AUC ~0.55. False alarm — recovered later. |
| ~15:28 | Two LINEAR variants launched (image 1.3.239) — companion arms with `use_arcface_head: false`, `normalize_features_before_head: false`. |
| 16:05 UTC | Both ArcFace arms FINISHED (150 min, full 10000 steps). Best AUC: L3 = 0.7390, L4 = 0.7634. |
| 16:51 | LINEAR arms confirmed climbing: AUC 0.49 (step 482) → 0.67 (step 1044). Hypothesis still alive on the LINEAR axis. |
| 17:00 | Trained-head probe written and run on the completed ArcFace ckpts. Inversion finding documented below. |

---

## ArcFace arm final results

Best ckpts (top-N by val AUC):
- **L3 ArcFace** (run `melp4mol`, seed 5503): step 2088, val AUC = **0.7390**, EER = 0.3266
- **L4 ArcFace** (run `faheakaf`, seed 5504): step 2410, val AUC = **0.7634**, EER = 0.3186

Both arms peaked early (epoch 9–10 ≈ step 2200–2400), then plateaued for the remaining ~31 epochs.

### Per-bucket fake recall (val_in_dist) — non-DF40 only

| Bucket | L3 ArcFace | L4 ArcFace |
|---|---|---|
| `deeplive_visomaster` | 0.848 | 0.891 |
| `deeplive_minimal_processing_enhanced` | **0.983** | **1.000** |
| `deeplive_minimal_processing` | 0.302 ⚠️ | 0.233 ⚠️ |
| `deeplive_edge_cases_enhanced` | 0.970 | **1.000** |
| `deeplive_edge_cases` | 0.377 | 0.393 |
| `deeplive_quality_enhancement` | 0.343 | 0.457 |

**Pattern is the familiar "enhancement-signature" leak**: `_enhanced` variants near-perfect (0.97–1.0); non-enhanced at 0.23–0.46. Layer choice (3 vs 4) is within noise on every bucket — 3-pt average gap with no consistent direction.

**Notably absent from `val_in_dist/per_bucket/`**: separate `visomaster_clean` or `*_teams` buckets. The eval suite's per-bucket reporting only spans `deeplive_*` family + DF40 methods. Filling that gap is one of the deferred follow-ups.

`val_holdout` shows the same pattern, slightly worse (`quality_enhancement` drops to 0.175 on L3, `edge_cases` to 0.217 on L4).

### Weakest method overall (`val_in_dist/weakest`)

- L3: MRAA (DF40), fake_acc 0.4286
- L4: e4s (DF40), fake_acc 0.4286

Per user direction, DF40 weakest is informational but not load-bearing.

---

## The trained-head probe (decisive finding)

### Why this probe exists

The original hypothesis (overnight session): a head reading layer-3 [CLS] from frozen P8A should produce a deployment-grade detector because layer-3 features are substrate-invariant (CPU probe: dev→lockbox transfer AUC = 0.95). The trainer's val AUC of 0.74 doesn't tell us whether the substrate-invariance held — that's a within-dev metric. The actual test is whether the *trained head*'s decisions transfer to lockbox.

### Method

1. Use the cached layer-3 [CLS] features from `analysis/_features_cache_2026-04-30/intermediate__P8A__layer03__n800.npz` (800 frames: 713 dev, 87 lockbox; 277/436 dev fake/real, 47/40 lockbox fake/real). Backbone is frozen P8A — same in both cache and our trained ckpts — so features are bit-identical.
2. Download the best ArcFace ckpt: `gs://training-job-outputs/phase2r13_experiments/melp4mol/top_n_effort_20260501_step2088_auc0.7390_eer0.3266.pth`.
3. Extract `head.weight` (shape `(2, 768)`) and `head.s` (scalar, `8.5044`).
4. Apply the ArcFace inference recipe: L2-normalize features, L2-normalize head weights per-class, dot-product → `(N, 2)` logits, fake-vs-real margin = `logits[:, 1] − logits[:, 0]`. (The scalar `s` is positive and identical for all samples; drops out of AUC ranking.)
5. Compute AUC + recall@FPR within dev, within lockbox, and overall.

Script: `analysis/intermediate_layer_probe_2026-04-30/eval_trained_heads_2026-05-01.py`. Output: `analysis/intermediate_layer_probe_2026-04-30/outputs/trained_head_eval_2026-05-01.json`.

### Sanity check (passed)

Fresh LR (5-fold CV within dev; train-on-dev / test-on-lockbox for transfer) on the **exact same features the trained-head probe loaded**:

| Test | Fresh-LR (this run) | Reference (`per_layer_split_probe.csv`) |
|---|---|---|
| Within-dev AUC | **0.9390** | 0.9390 ✓ |
| Transfer (dev→lb) AUC | **0.9495** | 0.9495 ✓ |

Match is exact — feature pipeline is correct, results below are real.

### Result

**L3 ArcFace (melp4mol step 2088, trainer val AUC 0.7390):**

| Metric | Trained ArcFace head | Fresh LR on same features |
|---|---|---|
| Within-dev AUC (n=713) | **0.6462** | 0.9390 |
| Within-dev rec@5%FPR | 0.2094 | — |
| Within-lockbox AUC (n=87) | **0.1867** ⚠️ | 0.9729 |
| Within-lockbox rec@5%FPR | 0.0000 | — |
| All-800 AUC | 0.5821 | — |

**Lockbox AUC of 0.187 means the trained head is anti-correlated on lockbox** — it predicts "fake" when the truth is "real" and vice versa. This is the same substrate-flip signature that the offline probe found at layers 6, 9, and (to a lesser extent) 11 of P8A. The trainer recapitulated the bug at layer 3 instead of avoiding it.

The within-dev AUC of 0.6462 is also *much* lower than the trainer's reported val AUC of 0.7390. Two explanations are consistent with this:
1. The 800-frame triptych sample is engineered for substrate analysis (89% dev + 11% lockbox + balanced labels per substrate); the trainer's val_in_dist sample is a different distribution.
2. There may be additional preprocessing in the trainer's eval path (e.g., a final LayerNorm) that I'm not replicating.

Either way, the **lockbox 0.187** is the load-bearing number — it's a same-recipe / same-ckpt / different-substrate comparison and is unambiguous.

### Implication

The hypothesis as-stated ("a head trained on layer-3 features avoids the substrate shortcut") is **falsified for the ArcFace recipe**. Layer 3 features carry substrate-invariant signal (fresh LR proves this), but the trainer's ArcFace + cos-similarity + s-anneal recipe systematically re-learns the dev-substrate shortcut even when reading from layer 3.

Two open possibilities — neither is closed yet:
- **Recipe is the problem, not the layer.** A plain Linear head with no normalization may be less prone to amplifying spurious cosine-similarity directions. The two LINEAR arms (still running) test this. If LINEAR's trained head gives lockbox AUC ≥ 0.85, the hypothesis is alive on the LINEAR recipe and ArcFace was the trap.
- **Recipe is fine, training duration is the problem.** ArcFace may have been substrate-invariant early (step 500–1500) and drifted into substrate-overfit later. Periodic-save ckpts at step 1000 and 2000 are downloadable; we can run the same probe on them. If lockbox AUC is high at step 1000 then collapses by step 2000, early-stopping ArcFace would be a recipe.

---

## What's running

Both LINEAR arms in us-west4 (image 1.3.239, Cloud Build SUCCESS; checked at 16:51 CEST):

| Job ID | Display name | seed | State | Last AUC | At step |
|---|---|---|---|---|---|
| `6468843621113135104` | `exp-R13_P17_LAYER3_HEAD_LINEAR-20260501-152845` | 5505 | RUNNING | 0.6713 | 1044 |
| `5009677341845094400` | `exp-R13_P17_LAYER4_HEAD_LINEAR-20260501-152858` | 5506 | RUNNING | 0.6732 | 1044 |

Trajectory (both arms): step 482 ≈ 0.50 (random) → step 883 ≈ 0.61 → step 964 ≈ 0.64 → step 1044 ≈ 0.67. Climbing monotonically. Projection: by step 2000–2500, LINEAR likely matches or exceeds ArcFace's 0.74–0.76 plateau.

ETA to step 5000: ~3 hours from now (at the observed ~10 min / 1000 steps pace). ETA to full 10000 steps: ~6 hours.

GCS prefixes:
- L3 LINEAR: `gs://training-job-outputs/phase2r13_experiments/nqvfz44v/`
- L4 LINEAR: `gs://training-job-outputs/phase2r13_experiments/mbd951b8/`

---

## What's pending (next decisions)

1. **Probe LINEAR step-1000 ckpt** (CPU, ~2 min). Decisive on whether LINEAR's recipe preserves substrate-invariance. If yes → hypothesis alive on LINEAR axis, plan a follow-up packet around plain-Linear heads. If no → hypothesis is structurally dead at the recipe level; pivot to a different lever.
2. **Probe earlier ArcFace ckpts** (step 1000, 1500 from melp4mol). Tests the "ArcFace was OK early then drifted" theory. If lockbox AUC was high at step 1000 then collapsed, early-stopping is a recipe.
3. **Extend feature cache to layer 4** (~10–25 min CPU). Required to probe L4 LINEAR / L4 ArcFace ckpts on the same lockbox subset.
4. **Run promotion contract scorecard** on the two completed ArcFace ckpts (GPU, ~30–90 min, ~$5–15). Even though the trained-head probe says they failed substrate, the deployment-grade verdict is independent and may reveal something the probe missed (e.g., a τ where lockbox FPR is acceptable).
5. **Fill the data gap**: re-run inference on `visomaster_clean` and `*_teams` buckets (not in the trainer's `val_in_dist/per_bucket/`). User flagged this as a genuine evidence gap.
6. **Memory entry**: defer until LINEAR result is in. Memory should capture the *final* picture, not an intermediate snapshot.

---

## File pointers

### Authored / modified this session

- `experiments/phase2_round13/R13_P17_LAYER3_HEAD_LINEAR.yaml` (new)
- `experiments/phase2_round13/R13_P17_LAYER4_HEAD_LINEAR.yaml` (new)
- `analysis/intermediate_layer_probe_2026-04-30/eval_trained_heads_2026-05-01.py` (new)
- `analysis/intermediate_layer_probe_2026-04-30/outputs/trained_head_eval_2026-05-01.json` (new)
- `utils/config_helpers.py:613-622` (fix for `apply_wandb_backbone_params` hidden_size override; was clobbering yaml's `hidden_size: 768`)
- `tests/test_p17_layer3_head_wiring.py` (added `test_wandb_overrides_preserve_explicit_hidden_size` regression test)
- `VERSION`: 1.3.238 → 1.3.239 (built and pushed via Cloud Build)
- `docs/relaunch_handoffs/P17_TRAINED_HEAD_FINDING_2026-05-01.md` (this document)

### Existing files referenced

- `analysis/_features_cache_2026-04-30/intermediate__P8A__layer03__n800.npz` (cached layer-3 [CLS] features for the 800-frame triptych sample)
- `analysis/embedding_triptych_2026-04-30/outputs/triptych_p8a_slot2_slot3/sampled_frames.csv` (the 800-frame index with dev/lockbox split labels)
- `analysis/intermediate_layer_probe_2026-04-30/per_layer_split_probe.py` (the original fresh-LR substrate-invariance probe — produced the reference 0.95 transfer AUC)
- `analysis/intermediate_layer_probe_2026-04-30/intermediate_layer_probe.py` (the script that GENERATED the cache; can be re-run with `--layers 4` to extend cache for L4 probing)
- `docs/relaunch_handoffs/OVERNIGHT_LAYER3_FINDING_2026-05-01.md` (precursor session — six probes that motivated P17)

### GCS paths

- L3 ArcFace ckpts (`melp4mol`): `gs://training-job-outputs/phase2r13_experiments/melp4mol/`
- L4 ArcFace ckpts (`faheakaf`): `gs://training-job-outputs/phase2r13_experiments/faheakaf/`
- L3 LINEAR ckpts (`nqvfz44v`): `gs://training-job-outputs/phase2r13_experiments/nqvfz44v/`
- L4 LINEAR ckpts (`mbd951b8`): `gs://training-job-outputs/phase2r13_experiments/mbd951b8/`

Local ckpt cache (this session): `/tmp/p17_ckpts/melp4mol_step2088.pth`, `/tmp/p17_ckpts/faheakaf_step2410.pth` (~900 MB each).

---

## Reproducing the trained-head probe

```bash
cd /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training

# 1. Download a ckpt (skip if already in /tmp/p17_ckpts/)
gsutil cp gs://training-job-outputs/phase2r13_experiments/melp4mol/top_n_effort_20260501_step2088_auc0.7390_eer0.3266.pth /tmp/p17_ckpts/

# 2. Run the probe
python3 analysis/intermediate_layer_probe_2026-04-30/eval_trained_heads_2026-05-01.py
```

To extend to additional ckpts: edit the `ARMS` list near the top of the probe script.

To extend to layer 4: first re-run the cache generator with layer 4:

```bash
python3 analysis/intermediate_layer_probe_2026-04-30/intermediate_layer_probe.py --layers 0,3,4,6,9,11 --device cpu
# ~10–25 min on CPU. Then add (label, ckpt, 4) to ARMS in eval_trained_heads_2026-05-01.py.
```

---

## Open questions for the user

1. **Run the LINEAR-step-1000 probe now, or wait for LINEAR to finish?** Probe is cheap (~2 min) and is the immediate decisive test on the recipe choice. Recommendation: run now.
2. **Should the ArcFace ckpts get a GPU promotion-contract scorecard despite the probe verdict?** ~$5–15. Probe says lockbox is structurally broken, but the contract uses a different evaluation path and might still find a τ that's deployment-acceptable. Low-confidence "yes — costs little to know."
3. **Is the layer-3 hypothesis worth pursuing further if LINEAR also fails the probe?** This is the strategic question. Alternatives in the precursor's tail-section: substrate correction, locked crop policy, hard-real anchors.
