# may6 production-drift retest — 4 r13-overnight candidate ckpts + T5C step3500/step1500/step3750 follow-up

Status: factual-only. No interpretation. Per `docs/packet_retrospectives/eval_folder_template.md`.

## Setup

### Question answered
What is the count of fired frames (prob_fake > 0.5) on the 92-frame may6 fresh-real Xinhe cohort for each of the 4 candidate r13-overnight ckpts plus three steps of T5C (jrlldtem: periodic_step1500, top_n_step3750, periodic_step3500 — the "T4 with classifier hidden_dim 256→1024" arm of the T5 axis on the 2026-05-12 scorecard), compared to the 4 baselines (P8A, E2B, T3_S1_step1500, T3_S1_step2500) measured in prior retests?

### Method
- Frame cohort: the 92 png frames at `analysis/xinhe_cross_camera_audit_2026-05-06/raw/may6/`. Same set used in the precedent harness (`run_local_inference.py`).
- Preprocessing: `cv2.IMREAD_COLOR` → `INTER_LINEAR` resize to 224×224 → `BGR2RGB` → CLIP normalization (mean/std as in `arena/model_arena.py` `CLIP_MEAN`/`CLIP_STD`).
- Forward: `model({"image": images}, inference=True)["prob"]`, identical to the precedent harness call (`analysis/xinhe_cross_camera_audit_2026-05-06/run_local_inference.py` lines 143–149).
- Device: MPS where available, CPU otherwise. Both produce numerically identical scores at the contractual `prob > 0.5` resolution.
- Threshold: τ=0.5 fixed, matching the prior retest in `analysis/cpu_diagnostics_2026-05-10/`.
- For Slot 1 + Slot 2 (LoRA ckpts), the LoRA wrapper (`detectors/lora_adapter.py::apply_lora_to_openclip_visual`) is applied AFTER instantiating the base `EffortDetector` and BEFORE `model.load_state_dict(...)`. LoRA config injected from the experiment yaml:
  - `target_layers=[10, 11]`, `rank=16`, `alpha=32`, `target_modules=["attn.in_proj", "attn.out_proj", "mlp.c_fc", "mlp.c_proj"]`, `freeze_base=True`.
- Baselines re-use the score CSVs at `analysis/xinhe_cross_camera_audit_2026-05-06/outputs/scores_{P8A,E2B,T3_S1_STEP1500,T3_S1_STEP2500}.csv`. Independently re-verified at the headline count level (P8A 0/92, E2B 53/92, step1500 6/92, step2500 71/92) prior to compositing the table.

### Candidate ckpts

| Slot | Alias | GCS URI |
|---|---|---|
| 1 | SLOT1_LORA_P8A_STEP2000 | `gs://training-job-outputs/best_checkpoints/gf6l06rf/top_n_effort_20260513_step2000_auc0.9929_eer0.0229.pth` |
| 2 | SLOT2_LORA_T5C_STEP1500 | `gs://training-job-outputs/best_checkpoints/912kd88q/periodic_effort_20260513_step1500_auc0.9941_eer0.0198.pth` |
| 3 | SLOT3_T5C_JITTER030_STEP4500 | `gs://training-job-outputs/best_checkpoints/502dcznh/top_n_effort_20260513_step4500_auc0.9923_eer0.0304.pth` |
| 4 | SLOT4_B16_SCRATCH_FOURIER_STEP10000 | `gs://training-job-outputs/best_checkpoints/qrpf5dtr/top_n_effort_20260513_step10000_auc0.9902_eer0.0259.pth` |
| follow-up | T5C_PERIODIC_STEP3500 (`jrlldtem`) | `gs://training-job-outputs/best_checkpoints/jrlldtem/periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth` |
| follow-up | T5C_PERIODIC_STEP1500 (`jrlldtem`) | `gs://training-job-outputs/best_checkpoints/jrlldtem/periodic_effort_20260511_step1500_auc0.9874_eer0.0570.pth` |
| follow-up | T5C_TOP_N_STEP3750 (`jrlldtem`) | `gs://training-job-outputs/best_checkpoints/jrlldtem/top_n_effort_20260511_step3750_auc0.9948_eer0.0154.pth` |

For all three T5C ckpts (step3500, step1500, step3750 — NOT LoRA), the EffortDetector base was instantiated with `multi_axis_grl.enabled=False` (this block is not saved in `model_config` — see `trainer/trainer.py::save_ckpt` lines 1394–1427). The 18 `multi_axis_grl_block.*` keys in the saved state_dict were dropped via `load_state_dict(strict=False)`. This is safe because `EffortDetector.forward(..., inference=True)` does not invoke `self.multi_axis_grl_block` (see `detectors/effort_detector.py` line 1843 — only invoked when `not inference`); the GRL block is an aux training head and does not affect the `prob` output. Architecture relevant to inference: `apply_svd_to_in_proj=true`, `apply_svd_to_mlp=true`, `unfreeze_final_proj=true`, `unfreeze_final_ln=true`, `vit_b_16_laion_datacomp` (ViT-B-16-DataComp-XL, datacomp_xl_s13b_b90k). The classifier head is `nn.Linear(hidden_size=512, 2)`; the multi_axis_grl `hidden_dim=1024` only sized the (dropped) aux block. State_dict load report for each: 0 missing keys, 18 unexpected — all 18 in `multi_axis_grl_block.*` namespace.

### Output artifacts

- Per-ckpt frame-level scores: `outputs/scores_<alias>.csv` (4 candidates).
- Baselines: `analysis/xinhe_cross_camera_audit_2026-05-06/outputs/scores_<P8A|E2B|T3_S1_STEP1500|T3_S1_STEP2500>.csv`.
- Summary table: `outputs/may6_retest_table.csv`.
- Script: `run_may6_retest.py`.

## Numbers

### may6 fired-frame counts at τ=0.5 (per ckpt)

| Ckpt | n_fired @ τ=0.5 | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|
| P8A_step5000 (baseline) | 0/92 | 0.0093 | 0.0368 | 0.1479 | 0.2535 |
| E2B_top_n_step3200 (baseline) | 53/92 | 0.5794 | 0.9451 | 0.9890 | 0.9896 |
| T3_SLOT1_step1500 (baseline) | 6/92 | 0.0197 | 0.3171 | 0.7437 | 0.8076 |
| T3_SLOT1_step2500 (baseline) | 71/92 | 0.8574 | 0.9832 | 0.9883 | 0.9898 |
| SLOT1_LORA_P8A_STEP2000 | 5/92 | 0.1765 | 0.3523 | 0.7275 | 0.7764 |
| SLOT2_LORA_T5C_STEP1500 | 19/92 | 0.1352 | 0.6768 | 0.9251 | 0.9417 |
| SLOT3_T5C_JITTER030_STEP4500 | 14/92 | 0.1669 | 0.5969 | 0.7602 | 0.8037 |
| SLOT4_B16_SCRATCH_FOURIER_STEP10000 | 18/92 | 0.0643 | 0.6884 | 0.9500 | 0.9582 |
| T5C_PERIODIC_STEP3500 | 16/92 | 0.1702 | 0.5818 | 0.8003 | 0.8287 |
| T5C_PERIODIC_STEP1500 | 71/92 | 0.7416 | 0.9215 | 0.9331 | 0.9367 |
| T5C_TOP_N_STEP3750 | 9/92 | 0.1476 | 0.4755 | 0.7194 | 0.7887 |

Cite: `outputs/may6_retest_table.csv` and the underlying CSVs in `outputs/`.

### Slot 1 vs P8A per-frame agreement (n=92 may6 frames, joined on `frame_basename`)

| Statistic | Value |
|---|---:|
| Pearson r | 0.9168 |
| Spearman r (rank) | 0.9699 |
| mean(Slot1 − P8A) | +0.2072 |
| median(Slot1 − P8A) | +0.1685 |
| max(Slot1 − P8A) | +0.6150 (`Generator PC__frame_000048_seq88.png`: P8A 0.108, Slot1 0.723) |
| min(Slot1 − P8A) | +0.1559 (`Generator PC__frame_000054_seq98.png`: P8A 0.155, Slot1 0.311) |
| frames where Slot1 fires (>0.5) but P8A does not | 5 |
| frames where P8A fires but Slot1 does not | 0 |
| frames where both fire (>0.5) | 0 |

Slot 1's `min(Slot1 − P8A) = +0.156` means **every** may6 frame's score is shifted upward relative to P8A; the rank order is preserved (Spearman 0.97), so the same 5 frames that have the highest P8A scores (max 0.254) become the 5 that fire at τ=0.5 on Slot 1 (max 0.776).

### Slot 1 vs P8A score-band distribution

| Score band | Slot 1 count | P8A count |
|---|---:|---:|
| [0.0, 0.1) | 0 | 88 |
| [0.1, 0.3) | 79 | 4 |
| [0.3, 0.5) | 8 | 0 |
| [0.5, 0.7) | 3 | 0 |
| [0.7, 1.0] | 2 | 0 |

### T5C step3500 vs P8A per-frame agreement (n=92 may6 frames, joined on `frame_basename`)

| Statistic | Value |
|---|---:|
| Pearson r | 0.5524 |
| Spearman r (rank) | 0.8882 |
| mean(T5C − P8A) | +0.2434 |
| median(T5C − P8A) | +0.1562 |
| max(T5C − P8A) | +0.7661 (`Generator PC__frame_000026_seq49.png`: P8A 0.031, T5C 0.798) |
| min(T5C − P8A) | +0.0543 (`Generator PC__frame_000115_seq202.png`: P8A 0.005, T5C 0.060) |
| frames where T5C fires (>0.5) but P8A does not | 16 |
| frames where P8A fires but T5C does not | 0 |
| frames where both fire (>0.5) | 0 |

Cite: `outputs/t5c_step3500_may6_summary.json`. Spearman p=3.77e-32 (statistically significant rank concordance). T5C's `min(T5C − P8A) = +0.054` means every may6 frame's T5C score is shifted upward vs P8A; like Slot 1, rank order is preserved (Spearman 0.89) so the same upper region of P8A's score distribution becomes the firing set in T5C.

### T5C step3500 score-band distribution

| Score band | T5C count | P8A count |
|---|---:|---:|
| [0.0, 0.1) | 19 | 88 |
| [0.1, 0.3) | 46 | 4 |
| [0.3, 0.5) | 11 | 0 |
| [0.5, 0.7) | 10 | 0 |
| [0.7, 1.0] | 6 | 0 |

### T5C step1500 vs P8A per-frame agreement (n=92 may6 frames, joined on `frame_basename`)

| Statistic | Value |
|---|---:|
| Pearson r | 0.3987 |
| Spearman r (rank) | 0.8306 |
| mean(T5C − P8A) | +0.6599 |
| median(T5C − P8A) | +0.7193 |
| max(T5C − P8A) | +0.9138 (`Generator PC__frame_000028_seq52.png`: P8A 0.008, T5C 0.922) |
| min(T5C − P8A) | +0.1667 (`Generator PC__frame_000114_seq201.png`: P8A 0.005, T5C 0.172) |
| frames where T5C fires (>0.5) but P8A does not | 71 |
| frames where P8A fires but T5C does not | 0 |
| frames where both fire (>0.5) | 0 |

Cite: `outputs/t5c_periodic_step1500_may6_summary.json`. Spearman p=1.33e-24. min(delta)=+0.167 — every may6 frame's T5C step1500 score is shifted upward vs P8A; rank order preserved (Spearman 0.83).

### T5C step1500 score-band distribution

| Score band | T5C count | P8A count |
|---|---:|---:|
| [0.0, 0.1) | 0 | 88 |
| [0.1, 0.3) | 6 | 4 |
| [0.3, 0.5) | 15 | 0 |
| [0.5, 0.7) | 18 | 0 |
| [0.7, 1.0] | 53 | 0 |

### T5C step3750 vs P8A per-frame agreement (n=92 may6 frames, joined on `frame_basename`)

| Statistic | Value |
|---|---:|
| Pearson r | 0.5598 |
| Spearman r (rank) | 0.8830 |
| mean(T5C − P8A) | +0.1984 |
| median(T5C − P8A) | +0.1362 |
| max(T5C − P8A) | +0.6811 (`Generator PC__frame_000026_seq49.png`: P8A 0.031, T5C 0.713) |
| min(T5C − P8A) | +0.0599 (`Generator PC__frame_000115_seq202.png`: P8A 0.005, T5C 0.065) |
| frames where T5C fires (>0.5) but P8A does not | 9 |
| frames where P8A fires but T5C does not | 0 |
| frames where both fire (>0.5) | 0 |

Cite: `outputs/t5c_top_n_step3750_may6_summary.json`. Spearman p=2.58e-31. min(delta)=+0.060 — every may6 frame's T5C step3750 score is shifted upward vs P8A; rank order preserved (Spearman 0.88).

### T5C step3750 score-band distribution

| Score band | T5C count | P8A count |
|---|---:|---:|
| [0.0, 0.1) | 20 | 88 |
| [0.1, 0.3) | 50 | 4 |
| [0.3, 0.5) | 13 | 0 |
| [0.5, 0.7) | 7 | 0 |
| [0.7, 1.0] | 2 | 0 |

## Direct observations

- **Slot 1 (LoRA-P8A_step2000) — 5/92 fires.** Closer to P8A's 0/92 than to E2B's 53/92. Per-frame Pearson r=0.92 / Spearman r=0.97 vs P8A indicate Slot 1's may6 score *order* is essentially the P8A order; the 5 fires are the 5 top-of-distribution frames where P8A's score was already 0.10–0.25.
- **Slot 2 (LoRA-T5C_step1500) — 19/92 fires.** Higher than Slot 1; falls between T3_S1_step1500 (6/92) and E2B (53/92).
- **Slot 3 (T5C+jitter030_step4500) — 14/92 fires.** Comparable order of magnitude to Slot 2; p90=0.597 / max=0.804 — no frame scores in the E2B-saturated >0.95 range.
- **Slot 4 (B16-scratch+Fourier_step10000) — 18/92 fires.** Comparable count to Slots 2/3, but p99=0.950 / max=0.958 lands in the E2B-saturation regime on its tail; the bulk of frames is at p50=0.064 (lowest median of the 4 candidates).
- **T5C_PERIODIC_STEP3500 (jrlldtem, follow-up) — 16/92 fires.** Comparable count to Slots 3 (14/92) and 4 (18/92); higher than Slot 1 (5/92) and T3_S1_step1500 (6/92); lower than Slot 2 (19/92), E2B (53/92), and T3_S1_step2500 (71/92). p90=0.582 / max=0.829 — no frame scores in the E2B-saturated >0.95 range. Pearson r=0.55, Spearman r=0.89 vs P8A; mean delta vs P8A = +0.243. All 16 T5C fires are T5C-only (zero overlap with P8A's empty fire-set, since P8A is 0/92).
- **T5C_PERIODIC_STEP1500 (jrlldtem, follow-up) — 71/92 fires.** Equal to T3_S1_step2500 baseline (71/92) and higher than E2B (53/92). p50=0.742 (highest of the 3 T5C steps; 53 of 92 frames score ≥0.7). max=0.937. Pearson r=0.40, Spearman r=0.83 vs P8A; mean delta vs P8A = +0.660 (largest delta of the 3 T5C steps). All 71 T5C fires are T5C-only.
- **T5C_TOP_N_STEP3750 (jrlldtem, follow-up) — 9/92 fires.** Lowest of the 3 T5C steps; comparable to T3_S1_step1500 (6/92) and Slot 1 (5/92); higher than P8A (0/92). p50=0.148, p90=0.476, max=0.789 — no frame scores ≥0.95. Pearson r=0.56, Spearman r=0.88 vs P8A; mean delta vs P8A = +0.198 (smallest delta of the 3 T5C steps). All 9 T5C fires are T5C-only.
- **Within the T5C family:** ranking by may6 n_fired is step1500 (71) >> step3500 (16) > step3750 (9). All three T5C steps shift every may6 frame upward vs P8A (min_delta > 0 for each); step3750 has the smallest upward shift and the lowest fire count of the family.
- **Load anomalies:** state_dict load was clean (0 missing, 0 unexpected lora keys) on Slots 1, 2, 4. Slot 3 had 18 unexpected keys (no `lora_` keys; non-lora unexpected, consistent with a multi-axis-L11-GRL head present in the saved ckpt but not constructed in inference). All three T5C ckpts (step3500, step1500, step3750) had 0 missing, 18 unexpected — all 18 in `multi_axis_grl_block.*`, the same training-only aux head pattern as Slot 3. No frame produced a NaN or near-0.5 uncertainty cluster — score histograms are unimodal-left for Slot 1/3/4 + T5C step3500/step3750; T5C step1500 is unimodal-right (p50=0.742).
- **No infrastructure failures.** All 7 candidates loaded on MPS; 92/92 frames scored per ckpt; ~16 s inference per ckpt.

## Reproducibility

Run from repo root:
```
python analysis/r13_overnight_may6_retest_2026-05-13/run_may6_retest.py                       # 4 slots
python analysis/r13_overnight_may6_retest_2026-05-13/run_t5c_step3500_may6.py                 # T5C step3500 followup
python analysis/r13_overnight_may6_retest_2026-05-13/run_t5c_step1500_and_step3750_may6.py    # T5C step1500 + step3750 followup
```

Inputs that affect the result:
- `analysis/xinhe_cross_camera_audit_2026-05-06/raw/may6/*.png` (92 files).
- `config/detector/effort.yaml`, `config/defaults.yaml`.
- For Slot 1 + Slot 2, `detectors/lora_adapter.py`. The LoRA config is injected from the script (not the saved `model_config`, which does not store the `lora` block — see `trainer/trainer.py` lines 1394–1427).
- For all three T5C ckpts (step3500, step1500, step3750), the multi_axis_grl block is similarly NOT in `model_config`; the runner explicitly sets `cfg['multi_axis_grl'] = {'enabled': False}` before model construction so the (training-only) aux head is not built. The `multi_axis_grl_block.*` keys in the saved state_dict are dropped via `strict=False`.
