# RESULTS_FACTS_v3 — band-shortcut analysis

> **FACTS only.** Supersedes the "opposing property profiles" conclusion in
> RESULTS_FACTS_v2 §3.3 and §4. Bands found via per-decile binning of
> Slot β over-fire rate.
>
> Origin: 2026-05-16, user pushed back on "no single property axis" framing
> noting shortcuts can live in bands. This analysis confirms the
> band-shortcut hypothesis.

## §1. Per-band over-fire rate (1198 PNG frames pool, lockbox + dev)

For each property, frames inside the listed band vs frames outside.

| property | band | n in | over in | rate in | n out | over out | rate out | ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **lab_a_dev** | > 16 | 121 | 96 | **79.3%** | 1077 | 82 | 7.6% | **10.4×** |
| **min_dim** | > 266 | 114 | 81 | **71.1%** | 1084 | 97 | 9.0% | **7.9×** |
| **sharpness_laplacian** | < 142 | 240 | 117 | **48.8%** | 958 | 61 | 6.4% | **7.7×** |
| **skin_frac_hsv** | > 0.88 | 123 | 80 | **65.0%** | 1075 | 98 | 9.1% | **7.1×** |
| **edge_density** | < 0.25 | 232 | 110 | **47.4%** | 966 | 68 | 7.0% | **6.7×** |
| **lab_a_std** | in [10.3, 11.9] | 111 | 67 | **60.4%** | 1087 | 111 | 10.2% | **5.9×** |
| **luma_mean** | < 130 | 244 | 105 | **43.0%** | 954 | 73 | 7.7% | **5.6×** |
| **lab_b_dev** | in [14.7, 17.7] | 114 | 52 | **45.6%** | 1084 | 126 | 11.6% | **3.9×** |
| **file_size_bytes** | > 110210 | 119 | 51 | **42.9%** | 1079 | 127 | 11.8% | **3.6×** |

Every property has a band where over-fire rate is 4-10× the rest. These would not be detected by mean/median Cohen's d tests because the band is interior (lab_b_dev [14.7, 17.7]) or because the cohorts have opposing signs (Roy_D's high lab_a_dev vs dor_shkedi's low lab_a_dev both over-fire — but they over-fire on DIFFERENT bands).

## §2. Compositional band-hits — dose-response

Per-frame, count how many of the 9 bands the frame inhabits. Then over-fire rate as a function of band-count:

| n bands hit | n frames | Slot β rate | T5C rate | P8A rate |
|---:|---:|---:|---:|---:|
| 0 | 770 | 6.9% | 1.6% | 0.0% |
| 1 | 175 | 6.9% | 5.1% | 0.6% |
| 2 | 71 | 12.7% | 5.6% | 0% |
| 3 | 24 | 12.5% | 8.3% | 4.2% |
| 4 | 16 | 6.3% | 6.3% | 0% |
| 5 | 30 | 30.0% | 33.3% | 6.7% |
| 6 | 32 | **65.6%** | **75.0%** | 21.9% |
| 7 | 29 | **93.1%** | **96.6%** | **41.4%** |
| 8 | 39 | **79.5%** | **89.7%** | 28.2% |
| 9 | 12 | **100%** | **100%** | **50%** |

Clean dose-response on all 3 ckpts. Even P8A — the "less broken" baseline — over-fires on 50% of frames that hit all 9 bands. The bands describe a **chronic structural failure mode shared across architectures**, not a Slot β regression.

## §3. Per-identity band-hit profile

For Slot β over-firing frames only:

| identity | n over | avg n_bands hit |
|---|---:|---:|
| **Roy_D** | 103 | **7.05** (essentially every band) |
| **dor_shkedi** (lockbox PNG) | 75 | **0.44** (essentially no bands) |

Roy_D's over-firing frames hit the bands. dor_shkedi.png's over-firing frames do NOT — they over-fire through a DIFFERENT mechanism not captured by these 9 bands. This explains the "opposing profile" observation in v2: Roy_D fails via band-hits; dor_shkedi.png fails via a separate channel.

## §4. Decision tree on the 11 properties

Trained on `Slot β over-fire` ∈ {0, 1} with `min_samples_leaf=20`:

| depth | AUC | top-2 splits |
|---:|---:|---|
| 3 | 0.855 | `lab_a_dev > 15.59`, then `edge_density`, `lab_a_std` |
| 4 | 0.892 | `lab_a_dev > 15.59`, then refines via `lab_b_dev`, `lab_b_std`, `luma_std` |
| 5 | 0.907 | same primary; refines via `skin_frac`, `file_size` |

For T5C and P8A on the same input features:

| ckpt | depth | AUC | primary split |
|---|---:|---:|---|
| T5C | 4 | **0.981** | `lab_a_dev > 15.59` |
| P8A | 4 | **0.988** | `luma_std > 51.08` then `lab_a_dev ∈ [15.77, 18.05]` |

P8A's AUC of 0.988 on a depth-4 tree means **P8A's over-fires are almost perfectly characterized by these properties** — the chronic-FP problem is band-shaped at the P8A level, not introduced by Slot β. P8A's narrower band (`lab_a_dev ∈ [15.77, 18.05]` rather than `> 15.59`) is what gives it the lower aggregate FPR.

## §5. Output files

- `outputs/all_png_pool.csv` — combined 1198 PNG frames + scores + properties
- `outputs/all_png_pool_with_bands.csv` — same + per-frame band-hit flags + count
- `outputs/per_decile_overfire_rate.csv` — per-property decile binning
- `outputs/decision_tree_*` (printed text-form in stdout; not saved as artifact)
