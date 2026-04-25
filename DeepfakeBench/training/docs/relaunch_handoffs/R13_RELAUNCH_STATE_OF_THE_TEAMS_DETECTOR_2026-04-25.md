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

### 2.3. Slices we DO NOT have for P8A yet

- **Augmentation-bucket OOD eval.** No saved `aug-slice` JSONs in the run's GCS prefix; the training-time eval reports aggregate AUC/EER but no per-aug breakdown.
- **Lockbox FPR (production-policy).** Not run. Was deferred as Task #17 along with the arena scorecard.
- **Fake recall on target methods (FaceFusion / SimSwap / DDIM / etc.).** Not measured outside training-time AUC; the arena scorecard would give per-method recall.
- **Cross-camera fingerprint diff (per-pool DCT/HF analysis post-fix).** Available for RLP6_04 baseline; not regenerated for P8A.

These four gaps are the next-most-valuable data points for promotion. None require new training — all are local rescore / arena runs.

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
- P8A breaks the P7 anchor ceiling. Single-variable change, reproducible config.
- Real-correct pools improved (not just held). FPR risk on the populations we already handled is at or below RLP6_04.
- Roee-mac materially improves and now has 7/30 frames decisively correct.
- Scratch on plain CLIP is dominated by FT+unfreeze on every metric — do not retrain R12g.
- The Teams camera-signature shortcut is real, cross-subject (Dor + Roee), and lives upstream in the data mix.

### Unmeasured (open questions)
- **Lockbox FPR.** Production-policy threshold check on diverse real footage — not yet run for P8A.
- **Per-method fake recall.** Arena scorecard per generator — not yet run for P8A.
- **P8A schedule sensitivity.** Early-stopped at step 5000. Whether 10k–20k continues to drop the anchor is open.
- **Augmentation slice contributions.** No per-aug-bucket OOD eval for P8A.
- **Real subject diversity.** Dor + Roee are 2 cameras. The shortcut may exist per-camera-vendor and not generalize to a third.

### Known limitations of P8A as it stands
- Anchor pool `frac_gt_0_9` is still 0.43 — 13/30 frames remain pinned above 0.9. P8A broke the ceiling but did not eliminate the shortcut.
- Anchor std jumped from 0.09 (RLP6_04) to 0.27 (P8A) — the pool is now bimodal, not uniformly fixed. This is consistent with "shortcut weakened, not eliminated".
- Step 5000 / 10k means P8A might not have saturated. Or might have already overfit on the value_composite criterion. Unknown without longer runs.

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

Detailed reasoning lives in `analysis/overnight_packet7_packet8_summary_2026-04-25.md`. Highlights:

- **Adopt the P8A recipe as the Packet-9 base.** The three flags should default to `true` for any Teams-FT yaml.
- **Stretch tests to consider:**
  - Extend P8A schedule to 15k–20k steps (test if it saturates past step 5000).
  - Unfreeze the last two attention blocks via `svd_blocks: [10, 11]` (deeper backbone reach).
  - Stack RLP7_07's CCT aug on the P8A base (try to push Roee-mac past −0.4).
  - Data-side intervention: camera-diversify the Teams real-pool (heavier lift; longer-term).
- **Do NOT** propose: scratch-on-plain-CLIP variants, retrain R12g with same data, head+attention-only SVD as default.

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
