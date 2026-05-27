# Amp vs Phase Probe — FINDINGS
**Date**: 2026-05-06
**Question**: Where in the FFT does the deepfake signal live on `teams_fake_all_dev` vs `teams_real_all_dev`?

## Verdict: **OTHER**

AUCs do not match any clean rule (amp=0.946, phase=0.861, pixel=0.671). Manual interpretation required.

## Setup
- Total frames scored: **5000** (2500 fake, 2500 real)
- Unique videos / groups: **4120**
- Resolution: 224x224 grayscale
- FFT bands: 16 radial x 8 angular = 128 features per representation
- Classifier: LogisticRegression (liblinear, C=1.0, class_weight=balanced) inside StandardScaler pipeline
- CV: 5-fold StratifiedGroupKFold (group=video_id)

## AUC table
| Representation | Mean AUC | Std | Folds | Per-fold AUCs |
|---|---|---|---|---|
| amplitude | 0.9461 | 0.0054 | 5 | 0.944, 0.954, 0.947, 0.947, 0.939 |
| phase | 0.8608 | 0.0080 | 5 | 0.861, 0.853, 0.859, 0.857, 0.874 |
| pixel-baseline | 0.6711 | 0.0162 | 5 | 0.684, 0.684, 0.645, 0.677, 0.666 |
| amp-shuffle | 0.4972 | 0.0105 | 5 | 0.499, 0.491, 0.491, 0.515, 0.491 |

## Per-band interpretation
Radial bands are indexed 0 (DC / lowest freq) to 15 (highest freq). Coarse classes:
- LF = bands 0..3
- MF = bands 4..11
- HF = bands 12..15

### Amplitude — top-20 |coef| radial-band class breakdown
- LF: 3 of 20
- MF: 10 of 20
- HF: 7 of 20

Top-5 amplitude features:
- band r=12, theta=2, coef=+2.3583
- band r=12, theta=6, coef=+2.1260
- band r= 8, theta=4, coef=+2.1060
- band r=11, theta=1, coef=-1.9480
- band r=13, theta=3, coef=-1.9092

### Phase — top-20 |coef| radial-band class breakdown
- LF: 0 of 20
- MF: 16 of 20
- HF: 4 of 20

Top-5 phase features:
- band r=11, theta=1, coef=-1.3245
- band r=11, theta=2, coef=-1.2858
- band r=11, theta=6, coef=+1.1204
- band r=12, theta=3, coef=+1.0235
- band r= 9, theta=3, coef=+0.9641

## Decision rule recap
- **POISON**: amp >= 0.85 AND phase < 0.70 -> Fourier amp aug destroys signal
- **GOLD**: phase >= 0.85 AND amp < 0.70 -> Fourier amp aug greenlit
- **MIXED**: both >= 0.85 within 0.05 -> conservative, mid-band only

## Files
- `outputs/amp_phase_aucs.csv` — fold AUCs and summary
- `outputs/amp_top_features.csv` — top-20 amplitude coefficients
- `outputs/phase_top_features.csv` — top-20 phase coefficients
- `outputs/run.log` — full run log
