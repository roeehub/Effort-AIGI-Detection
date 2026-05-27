# Job G — Per-axis perturbation sweep on Slot A v2 step3500, FACTS 2026-05-20

> **Context.** Replicates the 2026-05-19 Teams-account natural-experiment per-axis perturbation sweep on the Slot A v2 step3500 checkpoint (anchor_aware, FT from T5C, no LoRA, classifier hidden_dim=1024). The 2026-05-19 sweep on T5C showed `G_scale` is the dominant single-axis driver of `prob_fake` on Roy_D. The question this job answers: does the anchor_aware training in Slot A v2 reduce that per-axis sensitivity.

## 1. Method

- **Script.** `analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_g_perturbation/run_perturbation_slot_a_v2.py`.
- **Reference sweep.** `analysis/teams_account_natural_experiment_2026-05-19/perturbation_sweep.py` — perturbation functions, axis grids, IQ metrics, preprocessing pipeline used verbatim.
- **Checkpoint.** `analysis/manual_canary_2026-05-20/ckpts/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth` (Slot A v2 step3500). GCS: `gs://training-job-outputs/best_checkpoints/hp35c51p/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth`.
- **Detector / train config.** `config/detector/effort.yaml`, `config/defaults.yaml`. Loaded via `arena.model_arena.load_model` on CPU.
- **Input crop.** `analysis/teams_account_natural_experiment_2026-05-19/crops/face_roy_d.png`. Reference comparator: `face_guest.png`.
- **Preprocessing.** `cv2.resize → BGR2RGB → torchvision ToTensor → Normalize(mean=CLIP_MEAN, std=CLIP_STD)` at 224×224.
- **Sweep grids.** 8 axes × the same per-axis level lists as the 2026-05-19 script (lines 104-113):
  - `gamma` [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7]
  - `blur_sigma` [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
  - `R_scale`, `G_scale`, `B_scale` [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
  - `saturation`, `contrast` [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7]
  - `brightness_add` [-50, -30, -20, -10, 0, 10, 20, 30, 50]
- **T5C comparator swings.** Sourced from `analysis/teams_account_natural_experiment_2026-05-19/CHEAP_FOLLOWUPS_FACTS_2026-05-19.md §1`, also reproducible from the perturbation_sweep.csv in that folder.
- **Outputs.** `outputs/perturbation_sweep_slot_a_v2.csv` (79 rows, schema matches 2026-05-19 perturbation_sweep.csv), `outputs/swing_comparison.csv`, `outputs/operational_single_axis_slot_a_v2.csv`, `outputs/summary.json`, `outputs/run_stdout.log`.

## 2. Baselines

| Crop | Slot A v2 step3500 prob_fake | T5C step3500 prob_fake (2026-05-19) |
|---|---:|---:|
| Roy_D | **0.7668** | 0.7950 |
| Guest | **0.5987** | 0.6280 |
| Δ (Roy_D − Guest) | **0.1681** | 0.1670 |

The Slot A v2 Roy_D → Guest delta (0.168) is within 0.001 of the T5C delta (0.167).

## 3. Per-axis prob_fake table (Slot A v2 step3500)

### 3.1 `gamma`

| value | prob_fake | lap_var | luma_mean |
|---:|---:|---:|---:|
| 0.6 | 0.7741 | 157.92 | 188.3 |
| 0.7 | 0.7431 | 180.49 | 180.1 |
| 0.8 | 0.7615 | 200.64 | 172.4 |
| 0.9 | 0.7740 | 217.85 | 165.4 |
| 1.0 | 0.7668 | 231.43 | 159.4 |
| 1.1 | 0.8080 | 245.68 | 152.9 |
| 1.2 | 0.8247 | 258.27 | 147.2 |
| 1.3 | 0.8201 | 268.41 | 141.9 |
| 1.5 | 0.8111 | 285.39 | 132.6 |
| 1.7 | 0.8068 | 298.58 | 124.4 |

### 3.2 `blur_sigma`

| value | prob_fake | lap_var |
|---:|---:|---:|
| 0.00 | 0.7668 | 231.43 |
| 0.25 | 0.7668 | 231.43 |
| 0.50 | 0.8704 | 132.49 |
| 0.75 | 0.8716 | 60.89 |
| 1.00 | 0.8805 | 34.05 |
| 1.25 | 0.8630 | 20.87 |
| 1.50 | 0.8231 | 14.15 |
| 2.00 | 0.7519 | 7.74 |
| 2.50 | 0.8699 | 5.06 |
| 3.00 | 0.9256 | 3.77 |

### 3.3 `R_scale`

| value | prob_fake | R_mean |
|---:|---:|---:|
| 0.70 | 0.8534 | 126.71 |
| 0.80 | 0.8148 | 144.81 |
| 0.85 | 0.7822 | 153.86 |
| 0.90 | 0.7593 | 162.91 |
| 0.95 | 0.7615 | 171.96 |
| 1.00 | 0.7668 | 181.01 |
| 1.05 | 0.6700 | 190.06 |
| 1.10 | 0.5513 | 199.11 |
| 1.15 | 0.5505 | 208.16 |
| 1.20 | 0.5721 | 217.21 |

### 3.4 `G_scale`

| value | prob_fake | G_mean |
|---:|---:|---:|
| 0.70 | **0.1148** | 101.65 |
| 0.80 | 0.1608 | 116.17 |
| 0.85 | 0.2563 | 123.43 |
| 0.90 | 0.4493 | 130.69 |
| 0.95 | 0.6319 | 137.95 |
| 1.00 | 0.7668 | 145.21 |
| 1.05 | **0.8128** | 152.47 |
| 1.10 | 0.7981 | 159.73 |
| 1.15 | 0.7907 | 166.99 |
| 1.20 | 0.7792 | 174.25 |

### 3.5 `B_scale`

| value | prob_fake | B_mean |
|---:|---:|---:|
| 0.70 | **0.8689** | 93.53 |
| 0.80 | 0.8606 | 106.89 |
| 0.85 | 0.8492 | 113.57 |
| 0.90 | 0.8450 | 120.25 |
| 0.95 | 0.8145 | 126.93 |
| 1.00 | 0.7668 | 133.61 |
| 1.05 | 0.7553 | 140.29 |
| 1.10 | 0.6606 | 146.97 |
| 1.15 | 0.5221 | 153.65 |
| 1.20 | **0.4056** | 160.33 |

### 3.6 `saturation`

| value | prob_fake | sat_mean |
|---:|---:|---:|
| 0.50 | 0.6583 | 26.31 |
| 0.70 | 0.7606 | 36.83 |
| 0.80 | 0.7464 | 42.08 |
| 0.90 | 0.7598 | 47.34 |
| 1.00 | 0.7760 | 52.60 |
| 1.10 | 0.7843 | 57.85 |
| 1.20 | 0.8061 | 63.11 |
| 1.30 | 0.7851 | 68.37 |
| 1.50 | 0.7912 | 78.88 |
| 1.70 | 0.7795 | 89.40 |

### 3.7 `contrast`

| value | prob_fake | lap_var |
|---:|---:|---:|
| 0.50 | **0.2338** | 59.40 |
| 0.70 | 0.3592 | 115.36 |
| 0.80 | 0.4356 | 149.98 |
| 0.90 | 0.6909 | 189.06 |
| 1.00 | 0.7668 | 231.43 |
| 1.10 | 0.7577 | 278.20 |
| 1.20 | 0.6513 | 325.47 |
| 1.30 | 0.6358 | 374.77 |
| 1.50 | 0.6856 | 475.55 |
| 1.70 | 0.6179 | 573.32 |

### 3.8 `brightness_add`

| value | prob_fake | luma_mean |
|---:|---:|---:|
| -50 | 0.6065 | 110.4 |
| -30 | 0.7038 | 129.6 |
| -20 | 0.7083 | 139.6 |
| -10 | 0.7366 | 149.5 |
|   0 | 0.7668 | 159.4 |
|  10 | 0.7724 | 168.7 |
|  20 | 0.5998 | 177.4 |
|  30 | 0.5198 | 185.6 |
|  50 | 0.5478 | 200.7 |

## 4. Swing comparison: Slot A v2 vs T5C

Swing = max(prob_fake) − min(prob_fake) across the 10 perturbation levels for that axis.

| Axis | T5C swing (prior) | Slot A v2 swing (new) | Δ swing | Δ / T5C |
|---|---:|---:|---:|---:|
| `gamma`          | 0.092 | 0.0816 | −0.0104 |  −11.3% |
| `blur_sigma`     | 0.373 | 0.1737 | −0.1993 |  −53.4% |
| `R_scale`        | 0.259 | 0.3030 | +0.0440 |  +17.0% |
| `G_scale`        | 0.678 | **0.6980** | +0.0200 |   +3.0% |
| `B_scale`        | 0.497 | 0.4633 | −0.0337 |   −6.8% |
| `saturation`     | 0.030 | 0.1478 | +0.1178 | **+392.7%** |
| `contrast`       | 0.428 | 0.5330 | +0.1050 |  +24.5% |
| `brightness_add` | 0.199 | 0.2526 | +0.0536 |  +26.9% |
| **mean across 8 axes** | **0.3195** | **0.3316** | **+0.0121** | **+3.8%** |

Source: `outputs/swing_comparison.csv`.

## 5. Direction check on operational measurement (Step 5)

Apply each Roy_D → Guest measured factor singly to the Roy_D crop, score Slot A v2:

| Recipe | prob_fake | Δ vs Roy_D baseline | lap_var | R_mean | G_mean | B_mean |
|---|---:|---:|---:|---:|---:|---:|
| Roy_D baseline | 0.7668 | 0 | 231.4 | 181.0 | 145.2 | 133.6 |
| `R ×0.85` | 0.7822 | +0.0154 | 212.4 | 153.4 | 145.2 | 133.6 |
| `G ×0.87` | **0.3355** | **−0.4312** | 199.2 | 181.0 | 125.8 | 133.6 |
| `B ×0.85` | 0.8492 | +0.0824 | 224.7 | 181.0 | 145.2 | 113.1 |
| `blur σ=1.0` | 0.8805 | +0.1137 | 34.1 | 181.0 | 145.2 | 133.6 |
| Guest crop (reference) | 0.5987 | −0.1681 | 110.9 | 153.7 | 126.5 | 113.5 |

Source: `outputs/operational_single_axis_slot_a_v2.csv`.

- Guest direction from Roy_D baseline: **down** (0.7668 → 0.5987).
- Strongest single-axis driver towards "real": `G ×0.87` (Δ = **−0.4312**). Same direction and same axis as T5C in 2026-05-19's `CHEAP_FOLLOWUPS_FACTS_2026-05-19.md §1` "Direction check" subsection.
- `R ×0.85` and `B ×0.85` and `blur σ=1.0` individually move prob_fake **up** (towards fake), opposite the Roy_D → Guest direction. Same per-axis sign pattern as observed for T5C in 2026-05-19.

## 6. Mechanical bars

- **Bar 1** — Slot A v2 `G_scale` swing < 50% of T5C 0.678.
  - Threshold: 0.339. Slot A v2: **0.698**. → **NOT MET** (0.698 > 0.339; Slot A v2 is +3.0% relative to T5C, not reduced).

- **Bar 2** — Slot A v2 mean 8-axis swing < 50% of T5C mean 0.3195.
  - Threshold: 0.1598. Slot A v2 mean: **0.3316**. → **NOT MET** (0.3316 > 0.1598; mean swing is +3.8% relative to T5C).

- **Bar 3** — Any axis with Slot A v2 swing > T5C swing × 1.5.
  - **TRIGGERED** on 1 axis:
    - `saturation`: Slot A v2 0.1478 vs T5C 0.0300 → ratio **4.93×**.
  - No other axis exceeds the 1.5× threshold (next-highest ratios: `brightness_add` +26.9%, `contrast` +24.5%, `R_scale` +17.0%; all below 1.5×).

## 7. Files

- `analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_g_perturbation/run_perturbation_slot_a_v2.py` — script.
- `analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_g_perturbation/outputs/perturbation_sweep_slot_a_v2.csv` — full sweep (79 rows; same schema as the 2026-05-19 CSV).
- `analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_g_perturbation/outputs/swing_comparison.csv` — swing table.
- `analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_g_perturbation/outputs/operational_single_axis_slot_a_v2.csv` — Step 5 single-axis Roy_D → Guest checks.
- `analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_g_perturbation/outputs/summary.json` — machine-readable summary including bars.
- `analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/job_g_perturbation/outputs/run_stdout.log` — full stdout from the run.
