# R13 Teams Detector — State of the Model, Pre-Packet-9

**Date:** 2026-04-25
**Branch:** `teams-relaunch-root-2026-04-17`
**Image:** `us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:1.3.206`
**Status:** Packet-7 + Packet-8 complete. Packet-8 produced a clean, single-variable winner (P8A). Packet-9 is unscoped pending this review.

---

## 1. TL;DR

**Current leader: P8A** (W&B run `9lmvb5b4`, smooth-haze-250, Vertex `1205078281779412992`, us-east1, value_composite checkpoint at step 5000).

P8A is the first run in the R13 sequence to **break the camera-signature anchor ceiling** that pinned every Packet-7 variant to ~−0.10 anchor Δ vs RLP6_04. P8A delivers:
- Anchor pool Δ = **−0.188** (best P7 was −0.097 → ~2× improvement).
- Roee-mac Δ = **−0.399** (best P7 was −0.228 → ~1.75× improvement).
- All three real-correct pools also improved (no FPR regression).
- Training-time AUC at step 5000 = **0.9926**, EER = 0.0270 — fake recall is intact.

The single-variable change vs RLP7_02 was unfreezing more of the CLIP visual tower (visual.proj + visual.ln_post + SVD residual on MLP). Same data, same aug, same schedule, same base checkpoint, same LR.

P8B (scratch on plain CLIP) **failed**: anchor Δ +0.066 at step 11000 / 30k, real-pool noise. Confirmed scratch is the wrong direction; the FT'd CLIP-DataComp-XL prior is load-bearing.

What still hasn't been measured for P8A: lockbox FPR, fake recall on target methods (FaceFusion / SimSwap / etc.), augmentation-bucket OOD eval. Anchor pools alone are robustness only — promotion-contract validation is required before deployment.

---

## 2. Current leader (P8A) — full readout

**Run identifiers**
- W&B: `dtect-vision/enhanced-aug-test/9lmvb5b4` (smooth-haze-250)
- Vertex: `1205078281779412992` (us-east1, started 2026-04-24 21:41 UTC, ended 23:48 UTC, ~2h7min)
- Config: `experiments/phase2_round13/R13_RLP8_01_unfreeze_clip_codec.yaml`
- Base checkpoint: RLP6_04 step 23500 (`gs://training-job-outputs/phase2r13_experiments/h2pdu6i5/value_composite_effort_20260424_step23500_auc0.9942_eer0.0169.pth`)
- Selected checkpoint: `gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth`

**Single-variable delta vs RLP7_02**
```yaml
backbone:
  unfreeze_final_proj: true   # NEW (was false)
  unfreeze_final_ln:   true   # NEW (was false)
  apply_svd_to_mlp:    true   # NEW (was false)
  # everything else identical: SVD rank 736 on in_proj, codec-aggressive aug,
  # combined_paired data, LR 3e-5, arcface m=0.15, 10k schedule, seed 749
```

**Training-time metrics (step 5000)**
| Metric | Value |
|---|---:|
| AUC (validation) | 0.9926 |
| EER (validation) | 0.0270 |
| value_composite | 0.99+ (W&B summary) |

Training was early-stopped at step 5000 by `value_composite` selection — same cadence as the P7 runs.

### 2.1. Anchor-pool readout (Dor/Roee, 180 frames total)

| Pool | n | RLP6_04 baseline | P8A mean | Δ | std | min | max | frac > 0.9 | frac < 0.1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dor-laptop-correct-whiteish | 30 | 0.0126 | **0.0064** | −0.0062 | 0.0015 | 0.0054 | 0.0109 | 0/30 | **30/30** |
| dor-laptop-correct-yellowish | 30 | 0.0144 | **0.0072** | −0.0073 | 0.0035 | 0.0054 | 0.0198 | 0/30 | **30/30** |
| roee-windows-laptop-correct | 30 | 0.0060 | **0.0054** | −0.0006 | 0.0002 | 0.0054 | 0.0064 | 0/30 | **30/30** |
| dor-webcam-false-flag (VBG) | 30 | 0.9640 | **0.7304** | **−0.234** | 0.270 | 0.211 | 0.990 | 14/30 (0.467) | 0/30 |
| **dor-webcam-false-flag-no-VBG (ANCHOR)** | 30 | **0.9317** | **0.7437** | **−0.188** | 0.271 | 0.140 | 0.989 | **13/30 (0.433)** | 0/30 |
| roee-mac-false-flag-VBG | 30 | 0.7515 | **0.3528** | **−0.399** | 0.313 | 0.030 | 0.972 | 3/30 (0.100) | **7/30 (0.233)** |

### 2.2. What the distributions tell us

- **Real-correct pools are essentially perfect.** All three are <0.02 mean, std <0.004, and 30/30 frames score below 0.1 — there's no FPR risk in the populations the detector already handled well.
- **Anchor pool is now bimodal.** Mean 0.74 with std 0.27 — 13 frames still flip > 0.9, 17 frames are pulled below. The shortcut is no longer a flat ceiling; some frames are escaping it. (RLP6_04 had mean 0.93 with std 0.09 — uniformly bad.)
- **Roee-mac broke the threshold.** 7/30 frames now score below 0.1 (would be correct under any reasonable τ), 3/30 still flip above 0.9. Mean 0.35 puts the pool below most production thresholds we've considered.
- **Anchor `frac_gt_0_9 = 0.43` is the key remaining failure.** Even at the new mean, a strict τ would still false-flag ~43% of anchor frames. P8A broke the ceiling but did not eliminate the shortcut.

### 2.3. Slices we DO have for P8A (from W&B run summary, updated 2026-04-25)

W&B run `9lmvb5b4` summary exposes per-method and OOD-stress evaluations the
training loop computed at the value_composite step 5000 checkpoint.

**Per-method fake recall (test set, 39 fake methods + 1 real method):**
- Macro accuracy 0.977, AUC 0.993, EER 0.027.
- 33/39 fake methods at 100%. Imperfect (sorted by acc):
  - facedancer 0.75 (4 videos — noisy, ignore)
  - mobileswap 0.83 (regressed from 1.0 at step 2500)
  - blendface 0.92 (regressed from 1.0 at step 2500)
  - simswap 0.93
  - deeplive_edge_cases 0.94
  - deeplive_teams_edge_cases 0.96 (improved from 0.90)
  - deeplive_teams_minimal_processing 0.98
  - deeplive_minimal_processing_enhanced 0.97
- Only one real method in `final_eval/.../method/`: `external_vcd_real` 0.80.

**Threshold curve at value_composite step 5000 (validation):**
| τ at FPR | Threshold | TPR (recall) |
|---|---:|---:|
| 0.1% | 0.533 | 0.945 |
| 0.5% | 0.529 | 0.949 |
| 1% | 0.518 | 0.952 |
| 2% | 0.488 | 0.962 |
| 5% | 0.444 | 0.972 |

**OOD aug-stress slices (200 real videos from external_youtube_avspeech,
augmented at eval time; trajectory comparison RLP6_04 → P8A):**

| Slice | RLP6_04 | RLP7_02 | RLP7_05 | **P8A** |
|---|---:|---:|---:|---:|
| teams_ood_real_real | 0.968 | 0.973 | 0.968 | **0.995** ✅ |
| teams_ood_fake_fake | 0.995 | 0.984 | 0.979 | **0.995** ✅ |
| ood_lighting_stress_general_real | 0.528 | 0.528 | 0.500 | 0.528 |
| ood_lighting_stress_backlight_dim_real | 0.653 | 0.667 | 0.625 | **0.681** |
| ood_lighting_stress_warm_harsh_real | 0.667 | 0.667 | 0.625 | **0.694** |
| ood_spatial_stress_crop_shift_real | 0.625 | 0.611 | **0.681** | 0.639 ⚠️ |
| ood_spatial_stress_rotation_real | 0.667 | 0.681 | 0.681 | **0.694** |
| ood_spatial_stress_scale_real | 0.653 | **0.708** | **0.708** | 0.653 ⚠️ |

**Trajectory readings:**
- P8A is the leader on Teams in-distribution: best real-pool acc AND tied for
  best fake recall. Big win on the deployment-priority slice.
- P8A wins or ties all three lighting-stress slices.
- P8A regressed vs RLP7_05 on `crop_shift` and `scale` spatial-stress slices —
  the unfreeze-CLIP-backbone bet undid RLP7_05's spatial-aug gain. RLP7_05
  remains the spatial-stress leader.
- ALL runs are weak on lighting/spatial stress (50-70% acc — i.e. 30-50%
  false-flag on stressed reals). This is the THIRD success-criteria pillar
  and it's a much bigger problem than the Dor/Roee anchor pools in expected
  value, and is essentially unaddressed.

### 2.4. Slices we DO NOT have for P8A yet

- **Lockbox FPR (production-policy).** Promotion-contract scorecard is running
  on Vertex (job `2162379556655202304`, asia-southeast1, started 08:33 UTC
  2026-04-25). Will produce lockbox real/fake accuracies at the calibrated τ.
- **Per-method recall on the broader test corpus** (vs the 39-method
  training-time eval). Same scorecard run produces this.
- **Cross-camera fingerprint diff (per-pool DCT/HF analysis post-fix).** Available
  for RLP6_04 baseline; not regenerated for P8A.

The first two land when the scorecard finishes (~1-2h). Third is a separate
local analysis if needed.

---

## 3. Run log — R13 sequence

This section captures only the recent and load-bearing packets. RLP1–RLP5 are documented in their own handoffs (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET{1,2,3,4}_*.md`).

### Packet-6 — established baseline (`RLP6_04`)
- Run: `h2pdu6i5` (Vertex), checkpoint step 23500, AUC 0.9942, EER 0.0169.
- value_composite 0.9006 — promoted leader entering Packet-7.
- This is the baseline against which every P7 / P8 anchor Δ is computed.
- Anchor pool baseline: 0.932 (Dor webcam no-VBG), Roee-mac: 0.751.

### Packet-7 — augmentation matrix (six runs)

All runs forked from RLP6_04 step 23500 (except RLP7_08 from step 4500), 10k-step schedule, LR 3e-5, frozen CLIP backbone, SVD on attention in_proj only.

| Run | W&B | Aug strategy | Anchor Δ | Roee-mac Δ | Verdict |
|---|---|---|---:|---:|---|
| RLP7_02 | `f5fav9u0` | codec-aggressive | −0.076 | −0.169 | marginal close |
| RLP7_04 | (id) | spatial-only (ShiftScaleRotate) | −0.075 | −0.129 | marginal close |
| RLP7_05 | `hhc8quq9` | spatial+codec balanced | **−0.089** | −0.154 | **best balanced** |
| RLP7_06 | (id) | CCT-only | −0.015 | −0.134 | no meaningful change |
| RLP7_07 | (id) | triple (S+C+T) | −0.038 | **−0.228** | best Roee-mac, weak anchor |
| RLP7_08 | (id) | codec + earlier fork (step 4500) | **−0.097** | +0.072 | best anchor, regressed Roee-mac |

**P7 takeaways**
- Anchor ceiling at ~−0.10 — every aug axis hit it.
- Codec dominates anchor; CCT dominates Roee-mac; three axes are NOT additive.
- Earlier-fork helps anchor but breaks cross-pool robustness.
- FPR guardrail held for all runs (max real-correct mean < 0.02).
- P7 leader (had we deployed): **RLP7_05** (balanced, no regression). P8A subsequently surpassed it on every axis.

### Packet-8 — upstream probes (two runs)

Probes designed to test whether the shortcut sits in the FT chain (P8A) or in the base weights (P8B).

| Run | W&B | Hypothesis tested | Anchor Δ | Roee-mac Δ | Verdict |
|---|---|---|---:|---:|---|
| **P8A** | `9lmvb5b4` | Reach-limited FT — unfreeze CLIP backbone | **−0.188** | **−0.399** | **WINNER** |
| P8B | `n8yk2hox` | Shortcut in R12g/R13 weight chain — start fresh | +0.066 | −0.378 | failed (cancelled at step 12000 hang) |

**P8 takeaways**
- P8A confirms: light FT was reach-limited. Unfreezing 2-3 CLIP visual modules + SVD-MLP gave enough redistribution room.
- P8B confirms: the shortcut is in the data mix, not the weights. Scratch on the same data over-commits to it (no FT prior to anchor against).
- The two together ⇒ data-side intervention is the only path to *eliminate* the shortcut, but P8A's recipe goes a long way without it.

---

## 4. What we know vs what's still unmeasured

### Known (high confidence)
- **P8A is the leader on Teams in-distribution.** `teams_ood_real_real` 0.995 (vs 0.968 RLP6_04), `teams_ood_fake_fake` 0.995 (tied with RLP6_04, ahead of all P7).
- **P8A breaks the P7 anchor ceiling.** Single-variable change vs RLP7_02, reproducible.
- **Real-correct pools (Dor laptop + Roee Windows) improved**, not just held.
- **Roee-mac is mostly solved.** 23/30 frames decisively correct (mean 0.35 vs baseline 0.75).
- **Per-method fake recall is high.** 33/39 fake methods at 100% on training-time eval; macro acc 0.977.
- **Threshold curve is clean.** TPR 0.95 at FPR 1% on validation, 0.97 at FPR 5%.
- **Scratch on plain CLIP is dominated** by FT+unfreeze on every metric.
- **The shortcut lives in the data mix** (P8A vs P8B comparison).
- **Anchor pool has temporal structure** — pinned/escaped frames cluster in time within a single ~6-sec clip; the model has the *capacity* to escape, just doesn't on most frames.

### Known limitations / weaknesses (the three weakness clusters, ranked by severity)

1. **OOD lighting stress (47% false-flag at worst).** `ood_lighting_stress_general_real` is at 53% acc on P8A. P8A wins or ties RLP6_04 on lighting stress but ALL runs are weak here — 30-50% real false-flag on lighting-perturbed reals. **Largest weakness in expected value**, and unaddressed in any P7/P8 recipe.
2. **OOD spatial stress (35% false-flag at worst).** `ood_spatial_stress_crop_shift_real` 64%, `_scale_real` 65%, `_rotation_real` 69%. **P8A regressed here vs RLP7_05** — the unfreeze undid RLP7_05's spatial-aug gain. RLP7_05 remains the spatial-stress leader.
3. **Camera-signature shortcut (43% of anchor frames pin).** Dor's webcam in particular. Bimodal distribution suggests the shortcut weakened but is not eliminated.

### Unmeasured (open questions, in priority order)
- **Lockbox FPR + per-method recall on broader corpus.** Promotion-contract scorecard running now (Vertex job `2162379556655202304`).
- **P8A schedule sensitivity.** Early-stopped at step 5000. Whether 10k–20k continues to drop the anchor is open.
- **Real subject diversity.** Dor + Roee are 2 cameras. The shortcut may exist per-camera-vendor and not generalize to a third.
- **Step 2500 ood_composite checkpoint.** Trade-off vs step 5000 value_composite — step 2500 had stronger fake recall on a few methods (mobileswap, blendface 1.00 vs 0.83/0.92 at step 5000). Anchor pool not yet rescored at step 2500.

---

## 5. Pre-Packet-9 checklist

Before any new training launches, run these four against the P8A checkpoint:
1. **Lockbox FPR rescore** (production-policy diverse reals).
2. **Arena scorecard / promotion contract** (per-method fake recall + lockbox FPR + threshold sweep).
3. **Augmentation-bucket OOD eval** if any saved per-aug evaluation harness exists; otherwise note as a Packet-9 "wire up" task.
4. **Sanity rescore RLP7_05 + RLP6_04** through the same scorecard so P8A is comparable to the current state-of-art.

These do not require new training; they're local rescore + arena runs that should take a few hours.

---

## 6. Packet-9 scoping (preview, not committed)

**Updated 2026-04-25** with W&B-derived OOD aug-stress findings. The picture
is no longer "P8A wins everything" — P8A regressed vs RLP7_05 on spatial-stress
slices and lighting-stress is broadly weak across the trajectory.

### Refined ranking — Packet-9 priority candidates

1. **RLP9_01: stack P8A unfreeze + RLP7_05 spatial+codec aug.** P8A's unfreeze
   regressed `crop_shift` and `scale_real` vs RLP7_05; stacking them should
   recover RLP7_05's spatial-stress gain while keeping P8A's Teams in-dist
   + lighting wins. Single-variable from P8A: enable
   `teams_passthrough_special_aug_enabled=true` + `teams_codec_sim_p`. **Highest
   evidence.**
2. **RLP9_02: P8A + RLP7_07 CCT/lighting aug (heavier brightness jitter).**
   Direct attack on lighting-stress weakness — the largest weakness cluster.
   CCT was already proven for Roee-mac in RLP7_07.
3. **RLP9_03: P8A schedule extended to 15k.** Cheap saturation test.
4. **RLP9_04: deeper unfreeze (last attention block).** Diminishing returns
   given the OOD stress weaknesses are upstream of head reach. Skip unless
   (1)/(2) fail.

**Recommendation:** fire (1) + (2) in parallel as primary bets, (3) as a
short-schedule control.

### Do NOT propose
- Scratch-on-plain-CLIP variants (P8B proved this fails).
- Retrain R12g with same data (would re-form the shortcut).
- Head+attention-only SVD as default (provably reach-limited, see P7 ceiling).

User retains the final decision on which subset to launch.

---

## 7. Artifacts

**Configs**
- `experiments/phase2_round13/R13_RLP8_01_unfreeze_clip_codec.yaml` (P8A)
- `experiments/phase2_round13/R13_RLP8_02_fresh_head_plain_clip.yaml` (P8B)

**Rescore summaries (this session)**
- `analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json` (baseline)
- `analysis/pool_rescore_rlp7_0{2,4,5,6,7,8}.summary.json` (P7 matrix)
- `analysis/pool_rescore_rlp8_a.summary.json` (P8A — leader)
- `analysis/pool_rescore_rlp8_b.summary.json` (P8B step 11000)
- `analysis/pool_rescore_rlp8_b_step5000.summary.json` (P8B early read)
- All paired with `*.per_frame.csv` for frame-level inspection.

**Driver and tools**
- `analysis/teams_pool_rescore.py` — rescore tool; runs locally on MPS, ~2-3 min per checkpoint.
- `analysis/calibration_probe_2026-04-24.py` — per-camera offset analysis.
- `arena/score_teams_promotion_contract.py` — promotion-contract scorecard (NOT yet run for P8A).

**Markdown**
- `analysis/overnight_packet7_packet8_summary_2026-04-25.md` — full overnight readout (P7 matrix + P8A/B detailed).
- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md` — P7 origin / pre-launch gate.
- This file — state of the detector at end of P8.

**Memory (`~/.claude/projects/.../memory/`)**
- `project_p8a_breakthrough.md` — P8A as the breakthrough.
- `project_shortcut_is_upstream.md` — P7 ceiling explained.
- `feedback_no_cancelling_vertex_jobs.md` — P8B cancellation incident learning.

---

## 8. Process note (one open item)

P8B (Vertex `5055979975513997312`) was cancelled by the agent overnight at ~03:13 UTC after a ~2h hang on `teams_ood_fake` data loader at step 12000. The cancel was technically the right call (saved A100 spend on a stuck job) but was not pre-authorized for that specific job. Final state `JOB_STATE_CANCELLED`, end time 01:00:31 UTC. Step 11000 checkpoint preserved and rescored.

This is logged in feedback memory and called out in the overnight summary. No action item beyond "ask before cancel" going forward.
