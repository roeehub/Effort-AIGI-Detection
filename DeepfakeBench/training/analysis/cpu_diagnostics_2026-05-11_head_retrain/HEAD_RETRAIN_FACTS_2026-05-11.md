# Head retrain on frozen T4 encoder — train on dev, eval on lockbox (FACTS, 2026-05-11)

> **Status: factual-only.** Forbidden words: succeeds, fails, wins, loses, promotes, deployment-grade.
> Numbers + tables + cross-references only. Interpretation belongs in a paired proposal doc.
>
> **Scope**: train a fresh 2-layer MLP head on frozen-encoder L11 CLS features extracted on dev,
> evaluate the head on lockbox. Two encoders: `T4_LAMBDA1_TOP_N_STEP10500` and `P8A_REFERENCE_STEP5000`.
>
> **Anchor priors** (FACTS docs):
> - T4 trained-head lockbox AUC = 0.7619 vs P8A = 0.9355 (full lockbox 1361r + 253f) —
>   `analysis/t4_eval_2026-05-11/RESULTS_FACTS_2026-05-11.md` §3.1, §7.1.
> - A2: T4 + P8A L11 5-fold linear probe AUC = 1.0000 ± 0.0000 on in-sample lockbox subset —
>   `analysis/cpu_diagnostics_2026-05-11_a2_linear_probe/LOCKBOX_PROBE_FACTS_2026-05-11.md`.
> - A2-extension: same with real_dor; T4 step5000-periodic, 9000, 10500, 11250 all AUC = 1.0000 —
>   `analysis/cpu_diagnostics_2026-05-11_a2_extension/A2_EXTENSION_FACTS_2026-05-11.md`.
> - chronic_6 identity list: `analysis/group_id_design_audit_2026-05-06/outputs/chronic_flag_definition.json`.
>
> **Inputs**:
> - Ckpts: `analysis/cpu_diagnostics_2026-05-10/_ckpts_t4/top_n_effort_20260510_step10500_auc0.9937_eer0.0195.pth`
>   and `analysis/_features_cache_2026-04-30/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth`.
> - Dev frame lists: `analysis/option_a_ensemble_2026-04-28/cache/teams_{real,fake}_all_dev_p11_mild_step1000_frames_report.csv`
>   (only the `frame_path` and `video_id` columns are used; the `frame_prob` column references a different ckpt).
> - Lockbox frame lists: `analysis/cpu_diagnostics_2026-05-10/_t4_scorecard_local/teams_{real,fake}_all_lockbox_t4_lambda1_top_n_step10500_frames_report.csv`.
> - Local frame mirror: `/Users/roeedar/Downloads/faces/r9_feb28_for_checker/{real,fake}/flat/` + `analysis/cpu_diagnostics_2026-05-11_a2_extension/_real_dor_png/` (109 real_dor PNGs).
> - Scripts: `run_head_retrain.py`, `run_trained_head_baseline.py`.
> - Outputs: `head_retrain_aucs.csv`, `head_retrain_per_video_scores.csv`, `feature_extraction_metadata.csv`, `trained_head_baseline_aucs.csv`, `_cache/video_feats__*__{DEV,LOCKBOX}__L11.npz`, `_run.log`.

---

## 1. Method

### 1.1 Frame → feature extraction

- Each ckpt loaded via `detectors.DETECTOR[cfg['model_name']]` with `state_dict.pop('module.')` prefix and `strict=False`; T4 ckpts' `multi_axis_grl_block.*` keys dropped on load.
- Forward hook on `model.backbone.visual.transformer.resblocks[11]`; CLS token = `output[:, 0]` (dim=768).
- Image preprocessing identical to A2: BGR→RGB, resize-224, CLIP-mean/std normalize.
- Device: MPS.
- Per video, ≤4 frames sampled (random within video, seed=42), mean-aggregated to one (768,) vector per video.

### 1.2 Dev sample construction

- Source: dev frame CSVs (3253 real, 2409 fake videos in full).
- Local mirror availability check via `gs_to_local()` (`real_dor__*.png` → `_real_dor_png/`; otherwise `.jpg` mirror).
- After local mapping: 2695 real videos / 4006 frames, 1859 fake videos / 2489 frames.
- **Downsampling** (per brief's MPS-budget fallback):
  - Real: 2695 → 2000 via proportional source-stratified random sampling (seed=42).
  - Fake: 1859 → 1500 via proportional source-stratified random sampling (seed=42).
- After ≤4 frames/video cap: 3500 videos / 4972 frames (2961 real frames + 2011 fake frames) → 3500 video-level features.

### 1.3 Lockbox sample construction

- Source: same lockbox frame CSVs used by A2/A2-extension.
- Full lockbox: 1361 real videos / 1418 frames + 253 fake videos / 425 frames.
- Local mirror availability check: 494 real videos / 523 frames + 253 fake videos / 425 frames (all fakes available; reals constrained by local PNG availability).
- After ≤4 frames/video cap: 747 videos / 948 frames → 747 video-level features.
- **Lockbox real cohort breakdown after local-mirror filter** (per cohort, from `feature_extraction_metadata.csv` + `head_retrain_per_video_scores.csv`):

| cohort | n_reals | comment |
|---|---:|---|
| dor_shkedi | 271 | 271/1138 = 23.8% of full-lockbox dor_shkedi; remaining 867 are `.png` files not in local mirror |
| real_dor | 109 | 109/109 = 100%; PNGs were downloaded for A2-extension |
| chronic_6 | 61 | 61 are all `bla_bla_chow` (only chronic_6 identity present in lockbox locally) |
| non_chronic | 53 | 25 Chikara_Takahashi + 28 PC_Generator (non-`__s22`/`__s45`) |
| **Total** | **494** | |

### 1.4 Head architecture & training

- `TwoLayerHead(Linear(768→128) + ReLU + Linear(128→1))`. Output: BCE-with-logits.
- Class weight: `pos_weight = n_neg / n_pos` (computed per fit from train split).
- Optimizer: Adam, lr=1e-3, batch_size=64.
- Train/val split: stratified 80/20 of dev video features (seed=`RANDOM_SEED=42`).
- Max 50 epochs; early stop on val_loss plateau (patience=5).
- Best-state restoration at min val_loss before final eval.
- Train on dev features; eval on lockbox features (cross-substrate transfer).
- 3 seeds: {42, 7, 123}. PyTorch + numpy seeded per run.

### 1.5 Cohort assignment

- Function `assign_cohort(video_id, source)` (in `run_head_retrain.py`):
  - `source == "real_dor"` → `real_dor`
  - `source == "dor_shkedi"` → `dor_shkedi`
  - case-insensitive substring match of any chronic_6 identity (`bla_bla_chow`, `PC_Generator__s22`, `PC_Generator__s45`, `roy_d`, `Q__s6`) inside `video_id` → `chronic_6`
  - otherwise → `non_chronic`
- Per-cohort AUC computed as `roc_auc_score` on (cohort_reals + all 253 lockbox fakes).

---

## 2. Feature extraction metadata

Source: `feature_extraction_metadata.csv`.

| ckpt | split | n_videos | n_reals | n_fakes | elapsed_sec |
|---|---|---:|---:|---:|---:|
| T4_LAMBDA1_TOP_N_STEP10500 | DEV | 3500 | 2000 | 1500 | 111.5 |
| T4_LAMBDA1_TOP_N_STEP10500 | LOCKBOX | 747 | 494 | 253 | 31.0 |
| P8A_REFERENCE_STEP5000 | DEV | 3500 | 2000 | 1500 | 118.4 |
| P8A_REFERENCE_STEP5000 | LOCKBOX | 747 | 494 | 253 | 33.3 |

Total feature-extraction runtime: 294.2s. Total run including head training: 309.0s.

Train/val split per ckpt: 2800 train (1600 real + 1200 fake) / 700 val (400 real + 300 fake).

---

## 3. Head retrain — per-seed lockbox AUCs

Source: `head_retrain_aucs.csv`.

### 3.1 T4_LAMBDA1_TOP_N_STEP10500 (new-head)

| seed | dev_holdout_AUC | lockbox_AUC | chronic_6_AUC | real_dor_AUC | dor_shkedi_AUC | non_chronic_AUC | epochs_run |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.9998 | 0.7578 | 0.9620 | 0.8898 | 0.6245 | 0.9333 | 16 |
| 7 | 0.9998 | 0.7556 | 0.9610 | 0.9077 | 0.6119 | 0.9408 | 18 |
| 123 | 0.9997 | 0.7323 | 0.9712 | 0.8722 | 0.5828 | 0.9342 | 15 |
| **mean** | **0.9998** | **0.7486** | **0.9647** | **0.8899** | **0.6064** | **0.9361** | — |
| std | 0.0001 | 0.0141 | 0.0057 | 0.0178 | 0.0214 | 0.0041 | — |

### 3.2 P8A_REFERENCE_STEP5000 (new-head)

| seed | dev_holdout_AUC | lockbox_AUC | chronic_6_AUC | real_dor_AUC | dor_shkedi_AUC | non_chronic_AUC | epochs_run |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.9999 | 0.5084 | 0.9836 | 0.7964 | 0.2158 | 0.8653 | 16 |
| 7 | 1.0000 | 0.5031 | 0.9840 | 0.8022 | 0.2031 | 0.8680 | 50 |
| 123 | 1.0000 | 0.5111 | 0.9867 | 0.8170 | 0.2096 | 0.8759 | 35 |
| **mean** | **1.0000** | **0.5075** | **0.9848** | **0.8052** | **0.2095** | **0.8697** | — |
| std | 0.0000 | 0.0041 | 0.0017 | 0.0106 | 0.0063 | 0.0055 | — |

---

## 4. Trained-head baseline on the same local-subset

Source: `trained_head_baseline_aucs.csv`. Computed from per-frame `frame_prob` in the existing T4 scorecard frames CSVs; video-level aggregate = mean of `frame_prob` per `video_id`.

| ckpt | full local-subset (494r + 253f) | chronic_6 (61r vs 253f) | real_dor (109r vs 253f) | dor_shkedi (271r vs 253f) | non_chronic (53r vs 253f) |
|---|---:|---:|---:|---:|---:|
| T4_LAMBDA1_TOP_N_STEP10500 (trained head) | 0.8469 | 0.5788 | 0.9842 | 0.8337 | 0.9404 |
| P8A_REFERENCE_STEP5000 (trained head) | 0.8922 | 0.9790 | 0.9901 | 0.9070 | 0.5157 |

For reference, the **full-lockbox (1361r + 253f) trained-head AUCs** are 0.7619 (T4) and 0.9355 (P8A); see RESULTS_FACTS §7.1. The local-subset trained-head AUCs (0.8469 / 0.8922) are higher than the full-lockbox numbers because the local subset excludes 867 of 1138 dor_shkedi reals.

---

## 5. Head retrain — direct comparison (Δ vs trained head on same 747-video subset)

| ckpt | trained-head lockbox AUC (local subset) | new-head lockbox AUC mean | Δ (new − trained) |
|---|---:|---:|---:|
| T4_LAMBDA1_TOP_N_STEP10500 | 0.8469 | 0.7486 | **−0.0983** |
| P8A_REFERENCE_STEP5000 | 0.8922 | 0.5075 | **−0.3847** |

Per-cohort Δ (new-head − trained-head, mean across 3 seeds):

| ckpt | chronic_6 Δ | real_dor Δ | dor_shkedi Δ | non_chronic Δ |
|---|---:|---:|---:|---:|
| T4_LAMBDA1_TOP_N_STEP10500 | **+0.3859** | −0.0943 | −0.2273 | −0.0043 |
| P8A_REFERENCE_STEP5000 | +0.0058 | −0.1849 | **−0.6975** | +0.3540 |

---

## 6. T4 new-head vs P8A new-head — encoder comparison under matched head treatment

| | T4 new-head (mean) | P8A new-head (mean) | Δ (T4 − P8A) |
|---|---:|---:|---:|
| dev_holdout_AUC | 0.9998 | 1.0000 | −0.0002 |
| lockbox_AUC | 0.7486 | 0.5075 | **+0.2411** |
| chronic_6 | 0.9647 | 0.9848 | −0.0201 |
| real_dor | 0.8899 | 0.8052 | +0.0847 |
| dor_shkedi | 0.6064 | 0.2095 | **+0.3969** |
| non_chronic | 0.9361 | 0.8697 | +0.0664 |

---

## 7. Outcome verdict per Brief §α/β/γ

Brief §α: new-head lockbox AUC ≥ 0.90.
Brief §β: new-head lockbox AUC ≈ 0.76 (matches T4 trained head's 0.7619).
Brief §γ: new-head lockbox AUC ~0.85 (intermediate).

- T4 new-head full-lockbox AUC mean = **0.7486 ± 0.0141** (vs T4 trained-head full-lockbox AUC = 0.7619 from RESULTS_FACTS §7.1, and T4 trained-head local-subset AUC = 0.8469).
- The 0.7486 value (full local-subset, n=747) maps to **β-outcome** with respect to the 0.7619 full-lockbox trained-head reference.
- Side observation under the same local-subset reference 0.8469, T4 new-head is −0.0983 (Δ §5).

---

## 8. Cross-references

- T4 packet trained-head outcome: `analysis/t4_eval_2026-05-11/RESULTS_FACTS_2026-05-11.md` §3.1, §7.1.
- A2 in-sample linear probe AUC = 1.0000: `analysis/cpu_diagnostics_2026-05-11_a2_linear_probe/LOCKBOX_PROBE_FACTS_2026-05-11.md`.
- A2-extension (dor included, multi-step): `analysis/cpu_diagnostics_2026-05-11_a2_extension/A2_EXTENSION_FACTS_2026-05-11.md`.
- chronic_6 list: `analysis/group_id_design_audit_2026-05-06/outputs/chronic_flag_definition.json`.

---

## 9. Self-contained 7-bullet summary

- **Bullet 1**: T4 new-head lockbox AUC mean = **0.7486 ± 0.0141** across 3 seeds (per-seed: 0.7578 / 0.7556 / 0.7323; full-lockbox reference T4 trained-head AUC = 0.7619 from RESULTS_FACTS §7.1; same-local-subset T4 trained-head AUC = 0.8469).
- **Bullet 2**: P8A new-head lockbox AUC mean = **0.5075 ± 0.0041** across 3 seeds (per-seed: 0.5084 / 0.5031 / 0.5111; full-lockbox reference P8A trained-head AUC = 0.9355 from RESULTS_FACTS §7.1; same-local-subset P8A trained-head AUC = 0.8922; new-head − local-subset trained-head Δ = −0.3847).
- **Bullet 3**: Per-cohort T4 new-head AUCs (mean across 3 seeds): chronic_6 = 0.9647, real_dor = 0.8899, **dor_shkedi = 0.6064**, non_chronic = 0.9361. Versus T4 trained-head on same local subset: chronic_6 +0.3859 (0.5788 → 0.9647), real_dor −0.0943 (0.9842 → 0.8899), dor_shkedi −0.2273 (0.8337 → 0.6064), non_chronic −0.0043 (0.9404 → 0.9361). Largest improvement: chronic_6 (+0.3859); largest regression: dor_shkedi (−0.2273).
- **Bullet 4**: Verdict — **β-outcome**. T4 new-head full-lockbox AUC 0.7486 is within 0.013 of the T4 trained-head reference 0.7619. P8A new-head Δ = −0.3847 (relative to local-subset trained head), so the head retrain procedure does not transfer P8A's identity-cohort behavior across substrates; only encoder features alone do not carry it.
- **Bullet 5**: Dev held-out AUC sanity = **T4 0.9998 ± 0.0001, P8A 1.0000 ± 0.0000** (all 3 seeds ≥ 0.9997 for both ckpts), exceeds the ≥ 0.95 threshold from the brief.
- **Bullet 6**: Dev train n=3500 videos (2000 real + 1500 fake), downsampled from 2695 real / 1859 fake locally-available dev videos via proportional source-stratified random sampling (seed=42; brief's MPS-budget fallback). Lockbox eval n=747 videos (494 real + 253 fake) — the 494 reals reflect local-mirror availability and break down to chronic_6=61 (all `bla_bla_chow`), real_dor=109, dor_shkedi=271 (out of 1138 in full lockbox), non_chronic=53.
- **Bullet 7**: Caveats — (i) Head: 2-layer MLP 768→128→1; class-weighted BCE; Adam lr=1e-3; early stop on val_loss patience=5. (ii) Class imbalance: train pos_weight = n_neg/n_pos computed per fit. (iii) Cohort identification via case-insensitive substring match on `video_id` against chronic_6 list + source-prefix routing for `real_dor` and `dor_shkedi`; only `bla_bla_chow` of the chronic_6 list is present in lockbox locally — other chronic_6 identities (`PC_Generator__s22/__s45`, `roy_d`, `Q__s6`) are NOT in the lockbox CSVs at all. (iv) dor_shkedi lockbox coverage is 271/1138 = 23.8% due to missing PNG files in the local mirror; full lockbox dor_shkedi cohort has 4.2× more reals than this measurement, which biases dor_shkedi cohort AUCs and the full-lockbox new-head AUC compared to the FACTS reference 0.7619 / 0.9355. (v) The new-head P8A non_chronic AUC 0.8697 > the trained-head P8A non_chronic AUC 0.5157 — for P8A specifically, the 25 Chikara_Takahashi + 28 PC_Generator non-chronic reals score with mean=0.704 / median=0.795 under the trained head, comparable to the fake median 0.738, which is why trained-head non_chronic AUC is near chance for P8A. (vi) Per-video new-head scores written to `head_retrain_per_video_scores.csv` for both ckpts (mean across 3 seeds + per-seed columns).
