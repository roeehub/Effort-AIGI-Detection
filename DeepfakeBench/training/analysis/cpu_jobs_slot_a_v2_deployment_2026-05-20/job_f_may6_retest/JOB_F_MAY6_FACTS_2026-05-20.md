# Job F — may6 production-drift retest of Slot A v2 step3500 (+ P8A harness sanity)

Status: factual-only. No interpretation language. Per `docs/packet_retrospectives/AGENTS.md` and `docs/packet_retrospectives/eval_folder_template.md` — words forbidden: "succeeds", "fails", "wins", "promotes", "deployment-grade". Pass/fail is mechanical against pre-stated bars.

## Question answered

What is the count of fired frames (prob_fake > 0.5) and the score-distribution percentiles on the 92-frame may6 fresh-real Xinhe cohort (`analysis/xinhe_cross_camera_audit_2026-05-06/raw/may6/`) for `SLOT_A_V2_STEP3500` (run `hp35c51p`, ckpt `periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth`), compared to the 5 baseline ckpts measured in the 2026-05-13 retest (P8A, E2B, T3_SLOT1_step1500, T3_SLOT1_step2500, T5C_PERIODIC_STEP3500)? Independently, does the same harness re-score the P8A reference within numerical noise?

## Method

### Harness
- Adapted from `analysis/r13_overnight_may6_retest_2026-05-13/run_t5c_step3500_may6.py`.
- Model loader: `batch_inference_gcs.load_model(ckpt, detector_cfg, train_cfg, device)`. This loader handles ArcFace-scale restore, LoRA auto-detect from state_dict, and `strict=False` for unexpected aux-head keys. Slot A v2's `model_config.use_arcface_head=False`, so ArcFace scale is not restored. Slot A v2's state_dict contains 0 `lora_*` tensors (no LoRA cfg is auto-inferred). Slot A v2's state_dict contains 18 `multi_axis_grl_block.*` tensors; these are absorbed via `strict=False` (the detector's `multi_axis_grl` config defaults to `enabled: False`, so the aux block is not constructed at inference). State_dict load report (verified out-of-script): `len(missing)=0`, `len(unexpected)=18`, all 18 in `multi_axis_grl_block.*`, zero `lora_*`, zero `anchor_*`.
- Forward call: `model({"image": images}, inference=True)["prob"]`. `inference=True` routes around the ArcFace label-required branch and skips any aux GRL heads.
- Device: MPS (Mac).
- Threshold: `prob_fake > 0.5`.

### Ckpts scored
- `SLOT_A_V2_STEP3500` — local: `analysis/manual_canary_2026-05-20/ckpts/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth` (GCS `gs://training-job-outputs/best_checkpoints/hp35c51p/...`).
- `P8A_STEP5000` (harness sanity) — local: `analysis/manual_canary_2026-05-20/ckpts/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth` (GCS `gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/...`).

### Frame cohort
- 92 PNGs at `analysis/xinhe_cross_camera_audit_2026-05-06/raw/may6/`.
- Preprocessing: `cv2.IMREAD_COLOR` → `INTER_LINEAR` resize to 224×224 → `BGR2RGB` → CLIP normalization (`CLIP_MEAN`/`CLIP_STD` from `arena/model_arena.py`).
- Frame ordering: `sorted(glob("*.png"))` — same ordering used by the precedent harness `run_local_inference.py` and `run_may6_retest.py`.

### Configs
- Detector: `config/detector/effort.yaml`.
- Train: `config/defaults.yaml` (matches the prior may6 harness).

### Output artifacts
- Per-frame scores Slot A v2: `outputs/scores_slot_a_v2_step3500_may6.csv` (columns: `population,frame_path,frame_basename,prob_fake`).
- Per-frame scores P8A re-score: `outputs/scores_p8a_sanity_may6.csv` (same schema).
- Combined table (prior baselines + Slot A v2 + P8A sanity row): `outputs/may6_retest_table_with_slot_a_v2.csv`.
- Summary JSON: `outputs/slot_a_v2_may6_summary.json`.
- Run log: `_run.log`.
- Script: `run_may6_slot_a_v2.py`.

## Numbers

### may6 fired-frame counts at τ=0.5 — Slot A v2 step3500 row appended to the 2026-05-13 table

| Ckpt | n_fired @ τ=0.5 | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|
| P8A_step5000 (baseline, 2026-05-13 retest) | 0/92 | 0.0093 | 0.0368 | 0.1479 | 0.2535 |
| E2B_top_n_step3200 (baseline, currently-deployed) | 53/92 | 0.5794 | 0.9451 | 0.9890 | 0.9896 |
| T3_SLOT1_step1500 (baseline) | 6/92 | 0.0197 | 0.3171 | 0.7437 | 0.8076 |
| T3_SLOT1_step2500 (baseline) | 71/92 | 0.8574 | 0.9832 | 0.9883 | 0.9898 |
| T5C_PERIODIC_STEP3500 (baseline) | 16/92 | 0.1702 | 0.5818 | 0.8003 | 0.8287 |
| **SLOT_A_V2_STEP3500 (this retest, job F)** | **4/92** | **0.0967** | **0.2896** | **0.5737** | **0.7826** |

Source CSVs: `outputs/scores_slot_a_v2_step3500_may6.csv` and `outputs/may6_retest_table_with_slot_a_v2.csv`. Baselines reproduced from `analysis/r13_overnight_may6_retest_2026-05-13/outputs/may6_retest_table.csv`.

### P8A re-score harness sanity check (n=92)

| Quantity | Job F re-score (job_f harness) | Prior may6 (precedent harness) | Δ |
|---|---:|---:|---:|
| n_fired @ τ=0.5 | 0 | 0 | 0 |
| p50 | 0.009322 | 0.009315 | +7.1e-06 |
| p90 | 0.036755 | 0.036755 | (within tolerance) |
| p99 | 0.147898 | 0.147904 | -6e-06 |
| max | 0.253487 | 0.253485 | +2e-06 |

Per-frame delta (joined on `frame_basename`, n=92):
- `max_abs_delta = 7.25e-06`
- `mean_abs_delta = 4.32e-07`

Source: `outputs/scores_p8a_sanity_may6.csv` vs `analysis/xinhe_cross_camera_audit_2026-05-06/outputs/scores_P8A.csv`; full diff JSON at `outputs/slot_a_v2_may6_summary.json::p8a_harness_drift_vs_prior`.

### Score-band distribution — Slot A v2 step3500

| Score band | Slot A v2 count |
|---|---:|
| [0.0, 0.1) | 49 |
| [0.1, 0.3) | 34 |
| [0.3, 0.5) | 5 |
| [0.5, 0.7) | 3 |
| [0.7, 1.0] | 1 |

Computed from `outputs/scores_slot_a_v2_step3500_may6.csv`; reproduced via `pd.read_csv(...); np.histogram(df.prob_fake, bins=[0,0.1,0.3,0.5,0.7,1.0001])`.

## Mechanical pass/fail vs pre-stated bars

Bars are stated by the job specification:

| Bar | Definition | Threshold | Slot A v2 result | Verdict |
|---|---|---:|---:|---|
| Bar 1 (P8A parity) | n_fired ≤ 5/92 → preserves P8A-level production-drift robustness | ≤ 5 | 4 | **MET** |
| Bar 2 (T5C parity) | n_fired ≤ 16/92 → at least as robust as its T5C base | ≤ 16 | 4 | **MET** |
| Bar 3 (E2B parity) | n_fired < 53/92 → better than currently-deployed E2B | < 53 | 4 | **MET** |
| Bar 4 (Score-collapse signature) | p50 ≤ 0.05 → real-side healthy | ≤ 0.05 | 0.0967 | **NOT MET** |

P8A harness-sanity bar (implicit, from spec step 4):

| Bar | Definition | Threshold | Result | Verdict |
|---|---|---:|---:|---|
| P8A sanity (numerical noise) | P8A re-score yields 0/92 fires within numerical noise (max abs Δ < 1e-4 conventionally) | 0 fires; Δ small | 0 fires; max_abs_delta = 7.25e-06 | **MET** |

## Direct observations

- Slot A v2 step3500 fires on 4/92 may6 frames at τ=0.5. The 4 fired frames sit in score bands [0.5, 0.7) (3 frames) and [0.7, 1.0] (1 frame, max = 0.783).
- 83/92 frames score below 0.3.
- p50 = 0.097 is above the 0.05 floor but well below the τ=0.5 firing threshold; the distribution does not exhibit the right-skewed score-collapse signature observed in `T3_SLOT1_step2500` (p50 = 0.857) or `T5C_PERIODIC_STEP1500` (p50 = 0.742).
- State_dict load for Slot A v2 produced 0 missing, 18 unexpected — all 18 in the `multi_axis_grl_block.*` namespace (training-only aux head, dropped via `strict=False`). No `lora_*` keys present. No `anchor_*` keys present in the state_dict (the `anchor_aware` feature documented in the Slot A v2 training recipe operates loss-side via the existing `is_dor` GRL head and does not produce a saved aux module).
- P8A re-score via the same harness reproduced n_fired=0/92 with `max_abs_delta = 7.25e-06` and `mean_abs_delta = 4.32e-07` against the precedent CSV. Floating-point variation is consistent with MPS non-determinism noise level (the precedent CSV's saved precision is 6 decimal digits; the observed deltas are at the 5th-6th decimal).
- ArcFace scale was NOT restored for Slot A v2 (`model_config.use_arcface_head=False`); it WAS restored for P8A (s=9.749).
- No infrastructure failures: 92/92 frames scored for both ckpts; ~16 s inference per ckpt on MPS.

## Reproducibility

Run from repo root:
```
python analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_f_may6_retest/run_may6_slot_a_v2.py
```

Inputs that affect the result:
- `analysis/xinhe_cross_camera_audit_2026-05-06/raw/may6/*.png` (92 files).
- `config/detector/effort.yaml`, `config/defaults.yaml`.
- Slot A v2 ckpt: `analysis/manual_canary_2026-05-20/ckpts/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth`.
- P8A ckpt: `analysis/manual_canary_2026-05-20/ckpts/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth`.
- `batch_inference_gcs.load_model` (commit captured by git status at the time of the run).
- Slot A v2's `multi_axis_grl_block.*` keys in the saved state_dict are dropped via `strict=False`; the (training-only) aux head is not built because `cfg['multi_axis_grl']` is absent / `enabled: False` in the merged inference config.
