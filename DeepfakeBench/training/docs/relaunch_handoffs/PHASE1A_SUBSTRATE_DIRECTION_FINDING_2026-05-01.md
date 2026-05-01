# Phase 1A finding — P17 OQ1 substrate-classifier direction probe

**Date**: 2026-05-01 (post-P17 verdict)
**Author**: working agent
**Probe**: `analysis/intermediate_layer_probe_2026-04-30/substrate_classifier_direction_2026-05-01.py`
**Outputs**: `analysis/intermediate_layer_probe_2026-04-30/outputs/substrate_classifier_direction_2026-05-01.{json,csv}`

## TL;DR

The trained P17 heads (ArcFace + LINEAR, all 8 trajectory ckpts) are **essentially orthogonal to capture-mode and lockbox-vs-dev substrate axes** (cosines |c|<0.075). They are modally aligned with **`is_dor_shkedi`** (cos +0.07→+0.14 monotonic across training) and secondarily with **`is_deeplive_enhanced`** (cos +0.06→+0.10), which form one tightly correlated identity-fake-method cluster (`cos(is_dor_shkedi, is_deeplive_enh) = +0.86`).

**The destructor is not capture-mode.** It's the identity/fake-method cluster. This invalidates the implicit assumption behind P15 GRL @ λ=0.20 (and ramped-λ extension) that capture-mode quality-domain adversarial gradient is the right intervention.

## Method

L3 [CLS] features cached from P8A backbone (n=800: 713 dev + 87 lockbox; bit-identical to features the P17 heads were trained on). For each substrate axis, fit logistic regression on L2-normalized features → substrate direction. For each trained-head ckpt, compute cosine similarity between (head_decision_direction, substrate_direction) in normalized feature space.

ArcFace decision direction: `l2norm(head.weight[1]) - l2norm(head.weight[0])`.
LINEAR decision direction: `head.weight[1] - head.weight[0]` (raw space; cross-recipe approximation).

## Substrate axes probed (with in-sample LR AUC)

| Axis | n_pos | n_neg | LR AUC | ‖w‖ |
|---|---:|---:|---:|---:|
| is_lockbox | 87 | 713 | 0.876 | 3.7 |
| is_webcam | 238 | 562 | 0.851 | 6.9 |
| is_phone_screen | 255 | 545 | 0.734 | 5.5 |
| is_screen_any | 43 | 757 | 0.865 | 3.7 |
| is_normal_photo | 264 | 536 | 0.781 | 6.4 |
| is_teams_real | 476 | 324 | 0.923 | 9.1 |
| is_teams_capture | 251 | 549 | 0.944 | 10.4 |
| is_deeplive_enhanced | 59 | 741 | 0.988 | 7.4 |
| **is_dor_shkedi** | **39** | **761** | **0.906** | **3.9** |
| is_pc_generator | 115 | 685 | 0.874 | 5.2 |
| is_chikara | 6 | 794 | 1.000 | 0.7 (tiny n) |
| face_size_above_med | 390 | 410 | 0.925 | 9.2 |
| sharpness_above_med | 383 | 417 | 0.908 | 9.9 |
| brightness_above_med | 380 | 420 | 0.847 | 7.3 |
| Fresh-LR fake-vs-real (dev) | — | — | (ref) | 8.3 |

All substrate axes are real, well-separated directions in the L3 feature space. Most achieve AUC ≥ 0.85 in-sample.

## Key trained-head alignment table

Cosines per ckpt with each substrate direction. **Bold** = the most-aligned axis per ckpt.

| Arm | recipe | is_lockbox | is_webcam | is_phone | is_normal | is_dor_shkedi | is_deeplive | face_size | fresh_LR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L3_ARCFACE_ep1 | ARCFACE | -0.020 | -0.015 | +0.029 | +0.009 | -0.010 | -0.009 | -0.005 | +0.027 |
| L3_ARCFACE_s1687 | ARCFACE | +0.000 | +0.006 | -0.055 | +0.044 | **+0.092** | +0.076 | -0.011 | +0.044 |
| L3_ARCFACE_s1928 | ARCFACE | +0.022 | +0.006 | -0.067 | +0.052 | **+0.120** | +0.092 | -0.007 | +0.066 |
| L3_ARCFACE_s2088 | ARCFACE | +0.044 | +0.004 | -0.075 | +0.056 | **+0.142** | +0.103 | -0.001 | +0.086 |
| L3_LINEAR_ep1 | LINEAR | -0.020 | -0.015 | +0.029 | +0.009 | -0.009 | -0.009 | -0.005 | +0.027 |
| L3_LINEAR_s1044 | LINEAR | -0.012 | +0.016 | -0.047 | +0.029 | **+0.070** | +0.059 | -0.022 | +0.027 |
| L3_LINEAR_s1205 | LINEAR | +0.004 | +0.017 | -0.055 | +0.033 | **+0.089** | +0.069 | -0.021 | +0.044 |
| L3_LINEAR_s1285 | LINEAR | +0.016 | +0.016 | -0.060 | +0.036 | **+0.102** | +0.076 | -0.017 | +0.056 |

## What this tells us

1. **Trained heads do NOT pick up the capture-mode substrate axis.** All cosines vs `is_webcam`, `is_phone_screen`, `is_normal_photo` are |c|<0.08. The previous P15 GRL ran on `clip_capture_mode` quality-domain labels (PLAN.md §7.4 mapping) — it was reversing a gradient the model wasn't using. The static-λ=0.20 no-bite result (`domain_confusion_probe` SLOT2_GRL AUC 0.9994) is now mechanistically explained: there was no shortcut to reverse on this axis.

2. **The destructor is the identity/fake-method cluster.** `is_dor_shkedi` and `is_deeplive_enh` are 86% correlated as feature directions. Trained-head alignment with this cluster grows monotonically from ep1 (≈0) to step 2088 (+0.10–+0.14). This matches the trajectory P17 documented (lockbox AUC 0.72→0.07) and the prior memory on `dor_shkedi`-vs-`real_dor` flip (`project_signature_shortcut_finding.md`).

3. **Face-size is NOT a head-axis** even though `project_face_size_label_leak.md` documented training-data face-size leak. Trained-head cosine vs `face_size_above_med` is uniformly ≈0. The face-size leak may be a property of the training distribution that the L3 head doesn't pick up — it picks up identity-fake-method clustering instead.

4. **Even the modal alignment is small (max +0.14).** The trained-head direction is mostly NOT captured by any single substrate axis I probed. It's a high-dimensional combination that has small projections onto multiple axes. The largest is the `is_dor_shkedi` cluster, but most of the head's variance is along directions not tested.

5. **The fresh-LR fake-vs-real direction has substantial component along `is_lockbox`** (cos +0.55), explaining why fresh-LR transfers (0.95) while the trained head doesn't (0.04–0.19). Fresh-LR's substrate-discriminative axis is part of what makes it work. The trained head doesn't capture that axis.

## Implication for next-move design

**Capture-mode GRL @ ramped-λ is unlikely to bite either.** The trained head is essentially orthogonal to capture-mode at λ=0.20; ramping λ harder on the same axis won't introduce alignment that wasn't there.

**Better-targeted interventions:**
- **Identity-conditional GRL** — gradient-reverse against identity_key prediction. Directly attacks the dominant alignment axis.
- **Method-conditional GRL** — gradient-reverse against fake-method prediction (df40 / deeplive_* / visomaster_* / teams_capture_* / etc.). The eval probe shows method directions are highly discriminable in L3 features.
- **Paired same-identity training (Move 4)** — explicitly forbids identity as a shortcut by construction (paired clean[id_X] real, teams[id_X] real, swapped[id_X] fake). proper-data wave has 705 paired identities; companion_bucket plumbing exists at `data/sources/visomaster.py:748,1085,1105,1176,1523`.
- **Hard-negative mining** around `dor_shkedi` (and analogous high-density identity clusters) — pull the within-identity boundary tighter.

## Caveats

- Linear probe is L2-direction-only; norm-encoded shortcut info is invisible.
- Sample is 800 frames from one eval bucket (`teams-faces-data-test-2914-...`). Identity coverage limited (39 dor_shkedi, 6 chikara, 115 pc_generator); other identities not represented in this sample.
- The 800-frame sample is the dev+lockbox eval substrate, NOT the training distribution. The trained heads were trained on features from the actual training pool. Projection of their direction onto eval-features-substrate axes is informative but not perfectly representative.
- Cosines for LINEAR are cross-space (raw vs normalized) approximations.

## What this DOES NOT tell us

- Whether a Vertex run with identity/method-conditional GRL would actually bite at the [CLS] manifold. Need a smoke run with the analog domain-probe AUC checks to verify.
- Whether the same alignment pattern appears in the FINAL [CLS] layer of the lineage models (P8A, mclioexb, w5tky6ss/P15, P16). Plausible but not directly tested.
- Whether `is_dor_shkedi` is the SOLE driver or a representative of a larger identity-cluster pattern.

## Files

```
A  analysis/intermediate_layer_probe_2026-04-30/substrate_classifier_direction_2026-05-01.py  (the probe)
A  analysis/intermediate_layer_probe_2026-04-30/outputs/substrate_classifier_direction_2026-05-01.json
A  analysis/intermediate_layer_probe_2026-04-30/outputs/substrate_classifier_direction_2026-05-01.csv
A  docs/relaunch_handoffs/PHASE1A_SUBSTRATE_DIRECTION_FINDING_2026-05-01.md  (this document)
```
