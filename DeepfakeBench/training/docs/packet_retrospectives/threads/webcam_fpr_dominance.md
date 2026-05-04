# Thread: Lockbox FPR is dominated by webcam-style captures (modern_lockbox_v2 subset)

> **⚠ Critical-reading note (added 2026-05-04 night)**: this thread reports the substrate-level filter (modern_lockbox_v2) which IS deployable (filters which frames are scored, NOT how each frame is scored), AND it reports per-capture-mode τ analysis which is OFFLINE-ONLY (per-mode τ requires detecting capture mode at inference, which is impossible in Teams deployment per `feedback_per_mode_tau_not_deployable.md`). Read modern_v2 numbers as deployment-relevant; read per-mode τ recall lifts (e.g. the 24.5pp lift in the 2026-05-04 update section) as analysis-only. Do NOT propose per-mode τ as a forward production lever.

> **Slice 6 finding**: 105 of 414 lockbox real frames are `clip_capture_mode == webcam` and produce 65.7% FPR vs 8.3% / 1.5% for normal_photo / phone_screen on P8A step 5000. The headline 4.6% lockbox FPR is a data-hygiene artifact dominated by older / lower-quality webcam captures that are not representative of modern Teams deployment conditions. Memory `project_lockbox_fpr_dominated_by_webcam_mode.md` is the auto-memory anchor; this thread is the deliberated synthesis.

## The question

When the lockbox real pool reports a headline FPR (e.g. 4.6% on P8A at calibrated τ), how much of that FPR is **deployment-relevant** vs how much is **artifact of the lockbox composition**? Specifically, which `clip_capture_mode` slices are deployment proxies and which are legacy capture modes that bias the headline upward without informing the deployment risk?

## Initial belief

Through Slice 5 the lockbox real pool was treated as the canonical deployment proxy; the headline FPR (e.g. P8A 6.2% at τ=0.5, 4.6% at τ=0.9741) was used as the load-bearing readout for crowning protocols. The `PACKET_9_MID_FLIGHT_HANDOFF_2026-04-26.md:134-147` crowning protocol named `lockbox_real_fpr ≤ 0.441%` as the hard gate without slicing by capture mode; the implicit assumption was that the lockbox composition was already a deployment-honest filter.

That assumption was wrong on the FPR axis: the lockbox carries a substantial webcam-mode tail that drives most of the headline FPR but does not reflect modern Teams deployment.

## What changed our mind

- **2026-04-27 22:40 CEST — Lockbox FPR is highly concentrated by capture mode** (`april-26-training-master-plan-v2.LOG.md:864`). On P8A step 5000 against 414 lockbox real frames (parquet `analysis/lockbox_tagging/full_tags_2026-04-27.parquet`):

  | capture_mode | n | FPR @ τ=0.5 | FPR @ τ=0.9741 (5%-prod) |
  |---|---:|---:|---:|
  | normal_photo | 240 | 8.3% | 0.83% |
  | phone_screen | 68 | 1.5% | 0% |
  | **webcam** | **105** | **65.7%** | **15.2%** |
  | screen | 1 | 100% | 100% |
  | **all** | **414** | **22.0%** | **4.6%** |

  The webcam mode (n=105 of 414) drives nearly all of the headline 4.6% FPR. Source: memory `project_lockbox_fpr_dominated_by_webcam_mode.md`.

- **2026-04-27 23:25 CEST — Modern lockbox v2 subset built** (`april-26-training-master-plan-v2.LOG.md:1088-1133`). Filter sweep on P8A step 5000:

  | filter | n_real | n_fake | FPR τ=0.5 | FPR τ=0.9741 |
  |---|---:|---:|---:|---:|
  | baseline (all) | 414 | 425 | 21.98% | 4.59% |
  | v2a_drop_webcam | 309 | 390 | 7.12% | 0.97% |
  | v2b_drop_webcam_screen | 308 | 390 | 6.82% | 0.65% |
  | v2c_drop_webcam_tiny | 302 | 390 | 7.28% | 0.99% |
  | **v2_recommended** | **281** | **367** | **6.76%** | **0.71%** |

  v2_recommended definition (`analysis/modern_lockbox_v2_2026-04-27/`):
  ```
  clip_capture_mode not in {webcam, screen}
  AND face_area_ratio >= 0.10
  AND not is_pose_extreme
  AND not is_no_face
  ```
  Headline: at calibrated 5% production τ (0.9741), modern_lockbox_v2 lockbox FPR is **0.71%** — a 6.5× reduction from baseline 4.59% with no retraining. **The FPR side of the 90/5 target is solved by data hygiene at P8A baseline.** The recall side (≥ 90% on viso/deeplive/teams_fake) remains the gating concern.

- **`is_likely_screen_capture` filter dropped from v2_recommended** (LOG `:1110`). Empirical reason: it flags ~90% of lockbox fakes (over-aggressive on this set) and breaks paired recall measurement (n_fake from 390 → 41).

- **Caveat: v2_recommended is dor_shkedi-skewed** (`april-26-training-master-plan-v2.LOG.md:1124`, memory `project_lockbox_fpr_dominated_by_webcam_mode.md`). Of 281 v2 reals, **211 (75%) are dor_shkedi**. PC_Generator__s15 and Chikara_Takahashi__s22 are nearly fully filtered out (they are nearly-all webcam mode). Per-identity FPR breakdown is essential before treating v2 numbers as load-bearing.

- **Per-identity P8A lockbox FPR (5 identities)** (`april-26-training-master-plan-v2.LOG.md:858-863`):
  - bla_bla_chow__s1 (n=68, 45k px²): FPR=1.47%
  - dor_shkedi (n=275, 22k px²): FPR=10.55%
  - Chikara_Takahashi__s22 (n=42, 17k px²): FPR=83.3%
  - PC_Generator__s15 (n=29, 2.3k px²): FPR=89.7% — tiny artifact crops, the catastrophic outlier

  The two highest-FPR identities (Chikara, PC_Generator__s15) are exactly the ones almost-fully filtered by the webcam-drop rule; the v2 FPR drop is partly a survivor-bias effect.

- **Day-2 score_p11_modern_v2 scorer landed** (`analysis/modern_lockbox_v2_2026-04-27/score_p11_modern_v2.py`, `april-26-training-master-plan-v2.LOG.md:1200-1201`). Day-2 readout assets are staged: `modern_lockbox_real_v2_frames.yaml` (281 URIs), `modern_lockbox_fake_v2_frames.yaml` (367 URIs), `modern_lockbox_real_v2_videos.yaml` (270 video IDs), `modern_lockbox_fake_v2_videos.yaml` (217 video IDs), `p8a_lockbox_subsets_fpr_recall.csv`, `p8a_lockbox_v2_per_identity_fpr.csv`, `p8a_lockbox_v2_summary.json`. P11 day-2 verdict β (`april-26-training-master-plan-v2.LOG.md:1275-1277`) used the v2 scorer to read out the P11 step-1000 inference; HEAVY beat P8A by +16/+25/+9 pp on viso/deeplive/teams_fake at τ=0.5 — at the cost of 4× FPR on v2.

## Current stance (2026-04-29)

The lockbox real pool has a **webcam-mode tail that should not be in the headline FPR** for any deployment-relevant readout. modern_lockbox_v2 is the canonical filter for deployment-relevant FPR claims, with the dor_shkedi-skew caveat tracked alongside. On P8A step 5000:

- Baseline lockbox FPR @ τ=0.9741 (5%-prod): 4.6% (headline; misleading).
- modern_v2 FPR @ τ=0.9741: **0.71%** (deployment-relevant).
- 6.5× reduction is data hygiene, not model improvement.

**Operational rule** (memory `project_lockbox_fpr_dominated_by_webcam_mode.md`): when evaluating a new checkpoint against the lockbox, always report v2 FPR alongside baseline FPR. The v2 number is what's deployment-relevant. **Per-identity FPR breakdown is mandatory before treating v2 numbers as load-bearing**.

The webcam-mode tail is **not noise** — it represents the failure mode the camera-signature shortcut produces (memory `project_signature_shortcut_finding.md`), so dropping it from the headline does not mean "the model is fine on webcam." It means: when reporting deployment FPR, don't conflate the lockbox's legacy webcam captures with modern Teams capture conditions. The shortcut still needs to be broken; the v2 filter is a measurement-side correction, not a fix.

### Slice 7 update (2026-04-29) — modern_v2 readout used routinely; P13 verdict makes the gap stark

P13_FROM_SCRATCH Day-4 verdict (`docs/relaunch_handoffs/P13_DAY4_VERDICT_2026-04-29.md`) reads modern_v2 FPR as a first-class column at τ=0.5 alongside the baseline `tdev_FPR`. The verdict shows P13_step18000 sitting at modern_v2 FPR = **30.2%** at τ=0.5 (vs P8A 3.6% and RLP6_04 20.3%). This is the worst modern_v2 FPR of any candidate to date — the from-scratch trade gave back the data-hygiene gain that P8A had earned, and the verdict makes the regression starkly visible. **The modern_v2 readout is now routine across packets**; the open loop's close criterion (the contract scorecard reports v2 FPR as a first-class metric alongside baseline) is partially met at the verdict-doc layer but not at the contract-scorecard layer itself. Loop stays `open`; the contract-scorer integration is not done.

The Slice-7 reframing of "low recall" via the frame-level AUC evidence (memory `project_p8a_frame_level_auc_2026-04-29.md`) does NOT downgrade this loop's relevance: the FPR axis is independent of the recall reframing. Any future contract scorecard run with the corrected policy (`--promotion_target_fake_recall_min 0.30`) and the bucket-fix retrain `R13_P14_DATA_FIX.yaml` would still benefit from v2 FPR being a first-class contract metric — otherwise a candidate with 5% baseline lockbox FPR but 0.7% v2 FPR is read as "deployment-FPR ≤ 5%" instead of the more accurate 0.7% headline.

### Post-Slice-7 update (2026-04-29 afternoon) — crop-tightness gap means even modern_v2 isn't a strict production-FPR floor

A 2026-04-29 morning visual audit of eval-substrate slices (the very-sharp-FP and is_no_face slices from the 2026-04-27 investigation) surfaced a structurally-upstream finding: **eval frames carry more background context around the face than production crops do** (see [`eval_production_crop_tightness_gap`](eval_production_crop_tightness_gap.md)). The implication for this thread is direct: modern_lockbox_v2 FPR is the deployment-relevant number under the *capture-mode* caveat, but under the *crop-tightness-gap* caveat even modern_v2 may not be a strict production-FPR floor. The two caveats compound. Concretely:

- The webcam-mode tail this thread documents drives FPR via camera/ISP-pipeline cues. If the eval substrate gives those cues a larger surface (more background, more virtual-background content, more body-silhouette region) than production does, then dropping the webcam tail removes one axis of bias but leaves the residual eval-vs-production crop-tightness mismatch on every remaining slice.
- modern_v2's `not is_no_face` already excludes the data-degenerate is_no_face slice (audit Finding 1 — see [`eval_substrate_data_hygiene`](eval_substrate_data_hygiene.md)). modern_v2 does **not** apply a source-image-resolution floor; the audit Finding 4 (99×110-pixel source crops in the substrate) suggests adding `min(width, height) >= 200` as an additional v2 clause or as a side-by-side reporting column.
- The audit's `sharpness_laplacian` finding (see [`sharpness_metric_bug`](sharpness_metric_bug.md)) does not directly impact v2's filter set — v2 does not slice on `sharpness_laplacian` — but the sharpness metric bug does mean that any per-property quartile analysis someone ran on the v2 subset using `sharpness_laplacian` is partially confounded.

**This thread's close criterion (contract reports v2 FPR alongside baseline) is unchanged** — it still matters and still closes the loop when the contract scorecard wires v2 in. But under the crop-tightness gap, the v2 number itself carries an additional caveat: it is the cleanest available eval-substrate FPR, but the eval-to-production FPR translation is suspect for any failure mode that depends on background content. **The webcam-mode tail is exactly such a failure mode** (camera/ISP signal lives in pipeline metadata that surrounds the face), so the residual caveat applies most strongly to the slice this thread's open loop addresses. Do not silently downgrade the loop's severity; do log the cross-thread caveat for future agents.

## Packet timeline

- [WS_probes](../packets/WS_probes.md) — earlier per-pool / per-identity slicing on the lockbox; the camera/ISP shortcut is documented in [`processing_signature_shortcut`](processing_signature_shortcut.md). The webcam-mode dominance was implicit in the per-identity readouts (PC_Generator__s15, Chikara_Takahashi__s22) but not yet quantified at the capture-mode-tag level.
- [P8A](../packets/P8A.md) — the checkpoint the modern_lockbox_v2 filter sweep is applied to; P8A step 5000 lockbox FPR 4.6% headline / 0.71% v2 is the first number to clearly show the data-hygiene gap.
- [P11](../packets/P11.md) — first packet whose inference verdict is read out through the modern_v2 scorer (`score_p11_modern_v2.py`); HEAVY's +16/+25/+9 pp recall lift came at 4× v2 FPR cost, anchoring the verdict-β β/γ ASSESS framing.
- [P12](../packets/P12.md) — would have been read out through v2 if any usable mid-step checkpoints had been saved; periodic_saves silent failure means P12 never reached the v2 readout stage.
- *(Slice 7)* — modern_v2 filter audit is named in Plan v4 Track H (`april-26-training-master-plan-v2.LOG.md:1357`) as a Day-3 parallel-no-GPU task; the audit is the close criterion for the open loop below.

## Evidence locations

- `analysis/modern_lockbox_v2_2026-04-27/` — full subset infrastructure: `build_modern_subset.py` (filter sweep), `export_subset.py` (yaml emitter), `modern_lockbox_real_v2_frames.yaml` (281 URIs), `modern_lockbox_fake_v2_frames.yaml` (367 URIs), `modern_lockbox_real_v2_videos.yaml`, `modern_lockbox_fake_v2_videos.yaml`, `p8a_lockbox_subsets_fpr_recall.csv` (full filter sweep), `p8a_lockbox_v2_per_identity_fpr.csv` (per-identity breakdown), `p8a_lockbox_v2_summary.json` (machine-readable summary), `score_p11_modern_v2.py` (the post-hoc scorer).
- `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` — n=7,334 dev+lockbox tagged frames; the substrate parquet for capture-mode classification.
- Master plan LOG: `april-26-training-master-plan-v2.LOG.md:858-866` (per-identity readout + capture-mode breakdown), `:1088-1133` (modern_v2 build session), `:1200-1201` (score_p11_modern_v2 readout asset), `:1275-1277` (P11 verdict β using v2).
- Memory: `project_lockbox_fpr_dominated_by_webcam_mode.md` — auto-memory anchor with the capture-mode FPR table and the v2 definition.

## Open loops

### Open loop: webcam-mode-fpr-dominance-headline-misleading
status: open
severity: medium
first_seen: 2026-04-27
last_verified: 2026-04-29
close_criterion: the promotion contract scorecard (`arena/score_teams_promotion_contract.py` or its successor contract yaml) reports v2-filtered lockbox FPR as a first-class metric alongside baseline lockbox FPR — i.e., a candidate that has 5% baseline lockbox FPR but 0.7% v2 FPR is read as "deployment-FPR ≤ 5%" by the contract, not the other way around. Includes per-identity breakdown to surface the dor_shkedi-skew caveat as a hard sub-gate (e.g. "no single v2 identity has FPR > 30%" — the catastrophic outlier filter).

The data-hygiene story is now empirically clean: 6.5× headline FPR reduction comes from filtering legacy webcam captures (modern_v2 vs baseline). The contract still reads baseline FPR as the gate. Slice 6's P11 day-2 verdict β was the first time a packet read its results out through the v2 scorer (`score_p11_modern_v2.py`), but the contract scorecard pipeline did not. **The headline 4.6% lockbox FPR will continue to dominate crowning conversations until the contract reports v2 first.** The Plan v4 Track H modern_v2 filter audit (`april-26-training-master-plan-v2.LOG.md:1357`) is the canonical Slice-7 work that closes this; the audit's job is to confirm the v2 filter is correctly characterizing the deployment surface (the LOG entry calls the audit "30% likelihood this changes everything").

The dor_shkedi-skew caveat is the most likely failure mode: if 75% of v2 reals are one identity, the v2 number is partly an identity-specific FPR rather than a deployment-distribution FPR. The close criterion above incorporates this as a sub-gate.

### 2026-05-04 evening update — per-substrate τ-calibration tool quantifies the offline lift but confirms it is NOT deployable

A reusable per-substrate τ-calibration tool landed at `analysis/per_substrate_tau_calibration_2026-05-05/` (script `run_calibration.py`, `substrate_manifest.json`, `reference_run_p8a/`, `USAGE.md`). The tool takes a per-frame score CSV, joins against `analysis/lockbox_tagging/full_tags_2026-04-27.parquet`'s `clip_capture_mode` column to bucket reals into 5 substrates (normal_photo, webcam, phone_screen, screen, screen_recording), sweeps a 100-point τ grid plus quantile anchors, and reports per-substrate FPR + per-suite recall at each τ.

**Job-7 21pp lift validated and refined to 24.5pp on P8A**: Oracle per-mode dev-calibrated τ on P8A gives lockbox `teams_fake_lockbox` recall **78.6% at 16.9% FPR** vs single global TAU_F0_DEV (0.705) **54.1% recall at 4.3% FPR** — a **24.5pp recall lift at 4× the FPR**. This exceeds the memory-noted 21pp by 3.5pp.

**Critical deployability caveat (established by user constraint, this session)**: the 24.5pp lift requires detecting `clip_capture_mode` at deployment time (per-mode τ assigned to incoming frames at inference). **There is no way to detect capture-mode in production** — Teams does not surface the mode, and a learned mode classifier reaches only 52% CV accuracy on 4 IQ features (`analysis/viso_capture_mode_proxy_2026-05-05/summary.json:1`). Per-mode τ is therefore an OFFLINE ANALYSIS lever only — useful for understanding score distributions and substrate stratification, NOT a deployment policy.

**Deployable single-τ-substrate-aware results (P8A, the strongest available ckpt for viso)**:

| target FPR ceiling | τ | lockbox FPR | teams_fake_lockbox recall | viso recall | deeplive recall |
|---|---|---|---|---|---|
| 5% (strict) | 0.9941 | 0.07% | 16.2% | 0.18% | 0.0% |
| 10% (moderate) | 0.9891 | 0.35% | 24.9% | 1.6% | 4.0% |
| 20% (loose) | 0.8409 | 2.68% | 46.8% | 20.9% | 31.0% |

The numbers are bleak relative to the 90% across-the-board target. This thread's modern_v2 0.71% FPR finding (which IS deployable — modern_v2 filters by capture-mode at the data-substrate level, not at inference time per-frame) remains the cleanest deployment-relevant FPR claim. The per-substrate τ tool's main contribution is making the SIZE of the offline lift legible, which informs the calibration vs training-aug debate ([`calibration_vs_training_aug`](calibration_vs_training_aug.md)) by quantifying what calibration alone CAN'T deliver in production.

**Implication for this thread's open loop**: `webcam-mode-fpr-dominance-headline-misleading` close criterion is unchanged. modern_v2 (substrate-level filter) remains the right reporting fix. Per-substrate τ stays in the analysis layer. **Do NOT propose per-mode τ as a forward production lever**; the user's deployment constraint prohibits it.

### Cross-thread refs

- [`processing_signature_shortcut`](processing_signature_shortcut.md) — the webcam-mode tail is where the camera/ISP-signature shortcut produces its highest false-flag rate. Dropping webcam from the lockbox does not fix the shortcut; it just removes the lockbox slice that surfaces it. The two close criteria are paired but the underlying remediation is different: shortcut breaks via training-time aug, headline-FPR misleadingness fixes via reporting-side filter.
- [`face_size_label_leak`](face_size_label_leak.md) — `face_area_ratio >= 0.10` is part of the v2 filter; face_pixel_area is the underlying confound. The v2 filter is a deployment-domain question; the face-size leak is a leakage question. They share the variable but treat it differently.
- [`promotion_contract_evolution`](promotion_contract_evolution.md) — the per-identity reducer (WS-P2.b) integration as a contract gate is paired with this open loop; both are about reporting-side hygiene that the contract does not yet enforce. Loop `per-identity-reducer-not-a-contract-gate` (low) is the contract-side codification axis.
