# Slot 1 (LoRA-P8A top_n_step2000) — L11 frozen-encoder probe on dor_shkedi cohort — FACTS (2026-05-13)

> **Status: factual-only.** Numbers + tables + cross-references. No interpretation, no verdict language.
> Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade, unfortunately, remarkably.

## 0. Scope and provenance

- **Question (verbatim from task brief)**: linear probe on Slot 1's frozen L11 CLS features for lockbox real-vs-fake on the dor_shkedi cohort specifically; compare against the same probe on P8A_REFERENCE_STEP5000.
- **Trigger**: today's partial scorecard reported Slot 1 `lb_real_fpr ≈ 0.40` (21× P8A) with 96.3% of lockbox FPs being dor_shkedi (45.8% dor FPR vs P8A 0.7%, 65×). The Slot 1 head-retrain GPU job is in-flight (Vertex 6221202060298158080); its success path depends on whether the encoder retains separation at L11.
- **Inputs**:
  - Slot 1 ckpt: `analysis/r13_overnight_atlas_2026-05-13/_cache/top_n_effort_20260513_step2000_auc0.9929_eer0.0229.pth` (LoRA-wrapped, rank 16 / α 32 on layers [10,11], target modules `attn.{in_proj,out_proj}`, `mlp.{c_fc,c_proj}` per `R13_LORA_*.yaml`).
  - P8A ckpt: `analysis/_features_cache_2026-04-30/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth`.
  - Manifest: `arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json`.
  - Local frame cache: `analysis/lockbox_tagging/_frame_cache/<md5(blob_path)[:2]>/<md5(blob_path)[2:4]>/<basename>`.
- **Code**: `run_lockbox_probe.py` in this dir. Re-runnable.
- **Device**: MPS (Apple M-series). Batch size 32.
- **Methodology**: hook `backbone.visual.transformer.resblocks[11]` for CLS embedding (mirrors `analysis/r13_overnight_atlas_2026-05-13/run_atlas.py`); L2-normalize features; 5-fold StratifiedKFold logistic regression (C=1.0, max_iter=3000, lbfgs, n_jobs=1).

## 1. Cohort construction note

The task brief stated: "1138 lockbox-real dor_shkedi + 62 lockbox-fake dor_shkedi". The manifest read on 2026-05-13 contradicts the second number:

| Identity_key      | split    | label | n_videos | n_frames |
|---|---|---|---:|---:|
| dor_shkedi        | lockbox  | real  | 1138     | 1170     |
| dor_shkedi        | dev      | real  | 70       | 81       |
| dor_shkedi__s16   | dev      | fake  | 57       | 78       |
| dor_shkedi__s16   | dev      | real  | 31       | 31       |
| deeplive_dor      | dev      | fake  | 545      | 545      |

Source: `arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json`.

There are **0 fake dor_shkedi videos in lockbox**. The full fake-lockbox set across all identities is 253 videos = `Cam_Test__s33` (191) + `PC_Generator__s15` (62). The "62" figure in the task brief matches `PC_Generator__s15`, not dor_shkedi.

Because lockbox-fake-dor is empty, the probe construction must source the fake-side anchor from elsewhere. I built three probes (A/B/C below), all using the same 275 lockbox-real dor_shkedi frames as the real-class anchor, varying only the fake-class composition.

## 2. Cache availability for the dor cohort

| Cohort                              | Manifest n_frames | Local cache n |
|---|---:|---:|
| dor_shkedi lockbox real             | 1170              | 275 (23.5%)   |
| dor_shkedi__s16 dev fake (teams)    | 78                | 78 (100%)     |
| dor_shkedi__s16 dev real            | 31                | 31 (100%)     |
| deeplive_dor dev fake (deeplive)    | 545               | 545 (100%)    |

To honor the speed mandate, the probe ran on 898 locally-cached frames (275 real + 78 + 545 fake) without GCS downloads. The 275 lockbox-real subset is randomly drawn from prior diagnostics' cache populations; per-identity it is all dor_shkedi.

## 3. Feature extraction

| Ckpt label                       | n_frames_input | n_valid_extracted | nonfinite | feature_shape | cache file |
|---|---:|---:|---:|---|---|
| SLOT1_LORA_P8A_top_n_step2000    | 898            | 898               | 0         | (898, 768)    | `_features/SLOT1_LORA_P8A_top_n_step2000__layer11__dor_cohort.npz` |
| P8A_REFERENCE_STEP5000           | 898            | 898               | 0         | (898, 768)    | `_features/P8A_REFERENCE_STEP5000__layer11__dor_cohort.npz` |

Both ckpts loaded under MPS without missing-key issues beyond the expected LoRA runtime construction (Slot 1 ckpt).

## 4. Probe results

5-fold StratifiedKFold logistic regression. AUC reported as fold-mean ± fold-std and 95% CI computed as mean ± 1.96·SE (SE = std / √k). `OOF_AUC` is single-pass on stitched out-of-fold predictions.

### 4.1 Probe A — lockbox-real (dor_shkedi) vs dev-fake (dor_shkedi__s16, teams pipeline)

| Ckpt | n_pos (fake) | n_neg (real) | AUC mean | AUC std | 95% CI | per-fold |
|---|---:|---:|---:|---:|---|---|
| **SLOT1_LORA_P8A_top_n_step2000** | 78  | 275 | **0.9972** | 0.0037 | [0.9940, 1.0004] | 1.0000, 0.9989, 0.9955, 1.0000, 0.9915 |
| **P8A_REFERENCE_STEP5000**        | 78  | 275 | **0.9991** | 0.0010 | [0.9982, 0.9999] | 1.0000, 0.9977, 0.9989, 1.0000, 0.9988 |

### 4.2 Probe B — lockbox-real (dor_shkedi) vs all dev-fake (s16 + deeplive_dor)

| Ckpt | n_pos (fake) | n_neg (real) | AUC mean | AUC std | 95% CI | per-fold |
|---|---:|---:|---:|---:|---|---|
| **SLOT1_LORA_P8A_top_n_step2000** | 623 | 275 | **0.9931** | 0.0054 | [0.9883, 0.9978] | 0.9939, 0.9849, 0.9965, 0.9988, 0.9912 |
| **P8A_REFERENCE_STEP5000**        | 623 | 275 | **0.9942** | 0.0053 | [0.9895, 0.9988] | 0.9943, 0.9857, 0.9983, 0.9991, 0.9935 |

### 4.3 Probe C — lockbox-real (dor_shkedi) vs dev-fake (deeplive_dor only, cross-method)

| Ckpt | n_pos (fake) | n_neg (real) | AUC mean | AUC std | 95% CI | per-fold |
|---|---:|---:|---:|---:|---|---|
| **SLOT1_LORA_P8A_top_n_step2000** | 545 | 275 | **0.9938** | 0.0051 | [0.9893, 0.9982] | n/a (see JSON) |
| **P8A_REFERENCE_STEP5000**        | 545 | 275 | **0.9946** | 0.0050 | [0.9902, 0.9990] | n/a (see JSON) |

Source: `_probe_results_2026-05-13.json`.

## 5. Cross-references

- Atlas chronic_6 forgery AUC=0.995 on the 800-frame triptych (ATLAS_FACTS_2026-05-13.md §): consistent with the AUC magnitudes here. The triptych contained 98 dor-related frames out of 800 (25 lockbox-real dor_shkedi + 5/9 dev s16 real/fake + 59 deeplive_dor); the probes here use the same encoder family on a 9× larger dor cohort and reproduce ~0.99 AUC.
- Partial scorecard `lb_real_fpr` for slot1_top_n_step2000 at video-level τ=0.4496: 0.3975 (vs P8A 0.0184). Source: `analysis/r13_overnight_partial_scorecard_2026-05-13/PARTIAL_RESULTS_FACTS_2026-05-13.md §5`.
- Shift-analysis Spearman r at frame level on `teams_real_all_lockbox`: 0.2020 between P8A and Slot 1 top_n_step2000 (`SHIFT_ANALYSIS_FACTS_2026-05-13.md §1.3`); contrasts with the 0.97 may6 figure.
- Memory `project_t5c_chronic6_partial_recovery_2026-05-12.md` and `project_chronic_offenders_partition_per_ckpt_2026-05-04.md` document chronic-6 / dor_shkedi as a load-bearing fragility axis.

## 6. Notes on probe limitations

- Real-class draws from **lockbox** split; fake-class draws from **dev** split. The split mismatch means the probe may also separate on split-correlated factors (capture-time-of-day, lighting, codec). However, both the encoder under test (Slot 1) AND the reference (P8A) face the same potential confound, so the cross-ckpt comparison remains controlled.
- The 275-frame real subset is the subset of dor_shkedi lockbox reals already cached locally (23.5% of the 1170 total). No identity-level stratification was applied (all rows are dor_shkedi).
- The probe sees **L11 CLS embedding only** (768-dim). Slot 1's head adds visual.proj → ln_post → MLP → ArcFace head on top of L11. The probe does not exercise the head; it only tests whether L11 alone carries the discriminative axis.

## 7. JSON output

Full per-probe per-fold AUC + cohort metadata at `_probe_results_2026-05-13.json`. Re-run via:

```
python analysis/r13_overnight_slot1_lockbox_probe_2026-05-13/run_lockbox_probe.py --batch_size 32
```

Re-run cost: ~5 min on MPS (300s extraction + <1s probes; results cached at `_features/`).
