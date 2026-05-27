# Job A — Slot A v2 step3500 on Roy_D / Guest natural-experiment crops (FACTS)

> Per `docs/packet_retrospectives/AGENTS.md` eval-folder authoring contract: this is a FACTS doc; interpretation lives in `AGENT_PROPOSAL_2026-05-20.md`.

## 1. Question

The 2026-05-19 Teams-account natural experiment (`analysis/teams_account_natural_experiment_2026-05-19/`) measured prob_fake on two face crops captured at the same physical moment with the same camera, only the Teams account differs:

- T5C step3500: Roy_D 0.795 / Guest 0.628, Δ = −0.168, flips at τ=0.70
- P8A step5000: Roy_D 0.376 / Guest 0.211, Δ = −0.166, no flip at standard τ
- Slot B 6-axis GRL step3500: Roy_D 0.813 / Guest 0.610, Δ = −0.203, flips at τ=0.70 and τ=0.80

**Slot A v2 step3500 (anchor_aware, W&B run `hp35c51p`) was not in that table.** Question: does the anchor_aware mechanism reduce the per-frame transport-shortcut Δ, or is the mechanism content-bounded to the dor anchor pool (PC1=+2.67) and irrelevant to the Teams-account-encoded shift?

## 2. Method

- **Input crops**: `analysis/teams_account_natural_experiment_2026-05-19/crops/face_roy_d.png` (210×210) and `face_guest.png` (208×208). Cropped via production YOLO face detector (`yolov8s-face.pt`, conf=0.20).
- **Preprocessing**: `cv2.imread (BGR) → cv2.resize 224×224 INTER_LINEAR → BGR2RGB → CLIP normalize` (mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276]); matches `combined_paired.py:3455` exactly.
- **Ckpts**:
  - `T5C_STEP3500_CONTROL`: `analysis/manual_canary_2026-05-20/ckpts/periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth` (W&B `jrlldtem`). Re-scored to confirm harness parity with 2026-05-19.
  - `SLOT_A_V2_STEP3500`: `analysis/manual_canary_2026-05-20/ckpts/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth` (W&B `hp35c51p`).
- **Inference call**: `model({"image": x}, inference=True)["prob"]`, `torch.inference_mode()`, CPU device.
- **Script**: `run_slot_a_v2_inference.py`; raw output `outputs/job_a_inference_results.json`, tabular `outputs/job_a_inference_results.csv` and `outputs/job_a_combined_table.csv`.

## 3. Numbers

### 3.1 Harness parity sanity (T5C control)

| Source | Roy_D | Guest | Δ |
|---|---:|---:|---:|
| 2026-05-19 prior | 0.795 | 0.628 | −0.168 |
| 2026-05-20 Job A | **0.7952** | **0.6275** | **−0.1677** |

Reproduced to ≤0.001 absolute on every cell. Harness verified.

### 3.2 Slot A v2 step3500 result

| ckpt | Roy_D | Guest | Δ(Guest − Roy_D) |
|---|---:|---:|---:|
| T5C step3500 (2026-05-19 prior) | 0.7951 | 0.6275 | **−0.168** |
| **Slot A v2 step3500** (this job) | **0.7668** | **0.5987** | **−0.168** |
| Δ(SlotAv2 − T5C) | −0.028 | −0.029 | +0.000 |

### 3.3 Flip table (τ ∈ {0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90})

| τ | T5C Roy_D | T5C Guest | T5C flip | SlotAv2 Roy_D | SlotAv2 Guest | SlotAv2 flip |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.50 | FAKE | FAKE | — | FAKE | FAKE | — |
| 0.55 | FAKE | FAKE | — | FAKE | FAKE | — |
| 0.60 | FAKE | FAKE | — | FAKE | real | **⚠ FLIP** |
| 0.65 | FAKE | real | **⚠ FLIP** | FAKE | real | **⚠ FLIP** |
| 0.70 | FAKE | real | **⚠ FLIP** | FAKE | real | **⚠ FLIP** |
| 0.80 | real | real | — | real | real | — |
| 0.90 | real | real | — | real | real | — |

The flip window for T5C is τ ∈ [0.65, 0.70). For Slot A v2 the flip window is τ ∈ [0.60, 0.70) — wider (the lower bound dropped by 0.05).

## 4. Mechanical pass/fail

Pre-stated bars (defined before the result was known):

| Bar | Definition | Result |
|---|---|---|
| **Bar 1 (Δ reduction)** | abs(Δ_SlotAv2) ≤ 0.5 × abs(Δ_T5C) — anchor mechanism reduces transport-axis swing by at least half | **NOT MET** (abs(Δ_SlotAv2) = 0.168 = abs(Δ_T5C); ratio = 1.000) |
| **Bar 2 (no flip at production τ=0.65)** | SlotAv2 produces same verdict at τ=0.65 on both crops | **NOT MET** (FLIP at τ=0.65) |
| **Bar 3 (flip window narrower than T5C)** | SlotAv2 flip window ⊆ T5C flip window | **NOT MET** (SlotAv2 flip window is WIDER: lower bound moved from 0.65 → 0.60) |
| **Bar 4 (vertical shift only)** | abs(SlotAv2_Roy_D − T5C_Roy_D) ≈ abs(SlotAv2_Guest − T5C_Guest) AND Δ unchanged → mechanism is a uniform downward calibration, not transport-invariance | **MET** (Roy_D shift = −0.028; Guest shift = −0.029; Δ identical to 3 decimal places) |

## 5. Raw absolute scores

```
T5C_STEP3500_CONTROL:
  Roy_D  prob_fake = 0.795211   (cls = [-0.663, +0.694])
  Guest  prob_fake = 0.627506   (cls = [-0.244, +0.277])

SLOT_A_V2_STEP3500:
  Roy_D  prob_fake = 0.766795   (cls = [-0.585, +0.605])
  Guest  prob_fake = 0.598729   (cls = [-0.190, +0.210])
```

## 6. Artifacts

| Path | Contents |
|---|---|
| `outputs/job_a_inference_results.json` | Per-ckpt × per-crop prob_fake + cls logits + flip table |
| `outputs/job_a_inference_results.csv` | Tabular version of above |
| `outputs/job_a_combined_table.csv` | Merged with 2026-05-19 prior table |
| `run_slot_a_v2_inference.py` | Reproducible script |

## 7. Caveats

- n = 1 crop pair. Δ stability across multiple frames per account is not measured here. The 2026-05-19 followups (Cheap_Followups §6 #1 multi-account capture sweep) remain the proper N>>2 test.
- Slot A v2 also has step1500, step2500, step4500 saved; this job scored step3500 only (the contract rank-2 ckpt). Per-step trajectory not measured.
- CPU floats only; ckpts were trained on GPU. The harness has been verified to reproduce the 2026-05-19 T5C numbers within 0.001, so CPU↔GPU drift is negligible at this resolution.

## 8. Cross-references

- 2026-05-19 prior: `analysis/teams_account_natural_experiment_2026-05-19/TEAMS_ACCOUNT_NATURAL_EXPERIMENT_FACTS_2026-05-19.md`
- 2026-05-19 cheap followups (G-channel dominance, σ=2.5 + scale 0.80 recipe): `analysis/teams_account_natural_experiment_2026-05-19/CHEAP_FOLLOWUPS_FACTS_2026-05-19.md`
- 2026-05-16 anchor mechanism encoder probe (anchor pool 53% Roy_D-adjacent → 0%; Roy_D 29→81% dev FPR): memory `project_band_shortcut_ood_hypothesis_2026-05-16`
- Thread: `docs/packet_retrospectives/threads/processing_signature_shortcut.md` (open loop class)
