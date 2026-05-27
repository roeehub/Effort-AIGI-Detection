# Experiment Lineage & Checkpoint History

**Date prepared:** 2026-04-26
**Branch:** `teams-relaunch-root-2026-04-17`
**Scope:** Every Effort-detector training experiment in this codebase from CLIP-base to current P10 packet, with checkpoint lineage and cumulative FT-step accounting.

This is a fact-collection report. Conclusions are not synthesized; where I am uncertain about an attribution I prefix the row with "**UNCERTAIN:**".

---

## 0. Quick orientation

The architecture has been **unchanged since R1**: OpenCLIP ViT-B/16 with `pretrained="datacomp_xl_s13b_b90k"` (LAION DataComp-XL, ~13B images) as the frozen visual backbone, an **SVD residual** of trainable rank-`k` (default `k=32`, `rank=736` since R8) wired onto the per-block attention `in_proj`, and an **ArcFace** classification head. The variants explored have been mostly k, rank, augmentation, data composition, fine-tune base, and (since P8A) which extra weights to unfreeze.

There is **one thread of frozen-CLIP-as-base** that runs through every training run from R1 through to the most recent P10 runs. Within that thread, however, almost every R13 experiment fine-tunes from the **previously-FT'd checkpoint**. The cumulative trail of FT steps starting from CLIP base is large (see §3) and is the user's central concern.

---

## 1. Every experiment

Notation: "Δ from parent" lists the *single config knob* that changes vs the parent (per the agent-discipline of single-axis variation). When the parent is "scratch on plain CLIP," I list it as "scratch (no FT base)." Evidence sources for each row are provided in §1.7. Where the verdict cannot be cleanly recovered from on-disk artifacts I write "UNCERTAIN" and state what we *do* know.

The repo is consistent that **all experiments use** `vit_b_16_laion_datacomp` / `datacomp_xl_s13b_b90k` **except** R10_D, R11_G (ViT-L/14 — non-deployable per product rules; see §1.6).

### 1.1 Phase 1 baseline + R1–R7 (Jan–Feb 2026)

These are mostly pre-program-start exploration. They are documented in `experiments/WINNING_RUNS_REGISTRY.md` (a 720-line registry maintained until ~Mar 15).

| ID | Hypothesis | Δ from parent | Parent ckpt | Base CLIP | Outcome (W&B / score) |
|---|---|---|---|---|---|
| **B16-old / Phase 1** | Frozen-CLIP + SVD k=8 baseline on DF40+DeepLive | scratch (no FT base) | none | DataComp-XL | step 14000, AUC 0.9947, ExtReal FPR 8.6%@95%TPR. Ckpt: `gs://training-job-outputs/best_checkpoints/corrected/top_n_effort_20260119_step14000_auc0.9947_eer0.0218_B16_LAION.patched.pth`. Was **superseded** by R25_F1. |
| **R1 (P2_C3, `t1zpnv9s`)** | Cosine softmax beats CE | scratch + cosine softmax + rank=760 | none | DataComp-XL | step 8000, AUC 0.9844. **Invalidated** — config-passthrough bug meant all R1 runs trained on identical data. |
| **R2 (R2_A3, `8urk1cmw`)** | All-data CE, k=8, rank=760 | scratch + DF40+DL+VisoMaster | none | DataComp-XL | step 10000, AUC 0.9866. |
| **R2.5 (R25_F1, `5w453our`)** | k=32 (vs k=8) | scratch + k=32 | none | DataComp-XL | step 18500, AUC **0.9893**. New leader; replaced B16-old. |
| **R3 (R3_FT3, `kzfu116l`)** | Hold-out champ FT | FT R25_F1 | R25_F1 step 18500 | DataComp-XL | step 2500, AUC 0.9966 (holdout). OOD not validated. |
| **R4 (R4_FT7, `udgwsu7o`)** | Deploy-candidate FT | FT R3_FT3 | R3_FT3 step 2500 | DataComp-XL | step 500, AUC 0.9935; WMA 88.35%, ExtReal FPR 4.38%, DeepLive TPR 99.53%. |
| **R5** | Scratch did not displace FT7 | (various) | (various) | DataComp-XL | No new winner; FT7 retained. |
| **R6 (R6_S1, `s3tx3fk4`)** | VCD reals + aug | scratch + VCD reals + aug | none | DataComp-XL | step 6000, AUC 0.9691, VCD real acc 0.698, OOD AUC 0.954. (UNCERTAIN: per memory `project_signature_shortcut_finding`, R6 had GRL plumbing but `quality_domain_loss=0.0` due to a config bug — GRL was not effectively trained in R6.) |
| **R7** | (subsumed by R8) | — | — | — | R7 configs ran but R8 superseded; no winner registered. |

### 1.2 R8–R11 (Feb 23 – Mar 8 2026): the early FT chain

This is where the lineage that becomes RLP6_04 starts. Detailed in `experiments/PHASE2_SUMMARY_R8_R11.md`.

**R8 (~Feb 24, 8 runs)**:

| ID | Hypothesis | Δ from parent | Parent ckpt | Outcome |
|---|---|---|---|---|
| **R8_E (`hu7cen3m`)** ★ | Scratch, target-heavy (DF40+VM+DL) | scratch (no FT base) | none | step 8000, AUC **0.9925**, EER 0.0207, VCD real 82.1%, Viso 97.1%, DL 93.3%. **Champion of R8**. |
| (other R8 slots) | various scratch/FT | various | various | R8_E was champion; details in `R8_SESSION_SUMMERY.txt`. |

**R9 (Feb 27–28, 8 runs)** — Teams v1 (288 pairs) introduced:

| ID | Hypothesis | Δ from parent | Parent ckpt | Outcome |
|---|---|---|---|---|
| **R9_D (`m7etxxnp`)** ★ | Scratch, no Teams data | scratch | none | step 4000, AUC **0.9942**, EER 0.0325, OOD 0.9689. **Best B-16 ever recorded**. |
| **R9_F (`ueoziuou`)** | FT + codec sim | FT R9_A or scratch (UNCERTAIN — not labeled) | (UNCERTAIN) | step 500, AUC 0.9926, EER 0.0266 (best EER). |
| **R9_A (`1551zxa8`)** | FT + Teams v1 | FT R8_E + Teams v1 | R8_E step 8000 | step 6000, AUC 0.9891, OOD 0.9692. **Production deploy**. |
| (R9_B/C/G/H) | various | various | various | none won. |

R9 shipped **with a stability-λ config bug** (λ never reached the trainer); all R9 runs effectively had λ=0.

**R9.5 (Mar 2)**: 6 runs to retest stability-λ properly. Stability-λ monotonically *hurt*. Label-smoothing also hurt. R9_A retained as production.

**R10 (Mar 6–7, 7 runs)** — Teams v2 (1,346 pairs, 4.7× more) introduced:

| ID | Hypothesis | Δ from parent | Parent ckpt | Outcome |
|---|---|---|---|---|
| **R10_C** | FT R9_A + Teams v2, narrow aug | FT R9_A | R9_A step 6000 | step 8167, AUC 0.9836, OOD 0.9502. **R10 winner**. |
| R10_G | FT R9_A + Teams v2, wide aug | FT R9_A | R9_A step 6000 | step 8167, AUC 0.9829, OOD 0.9547. |
| R10_A/B/E/F | scratch + Teams v2 + various aug | scratch | none | All crashed at ~8.5K steps (24h Vertex timeout). |
| **R10_D (ViT-L/14)** | Different backbone | scratch + L/14 | none | Catastrophic AUC 0.9042 + best-OOD 0.9807. Config bug (`λ=1.0`, `rank=1023`). |

**R11 (Mar 7–8, 8 runs)** — FT-from-R9_D:

| ID | Hypothesis | Δ from parent | Parent ckpt | Outcome |
|---|---|---|---|---|
| R11_A | FT R9_D + narrow aug | FT R9_D | R9_D step 4000 | step ~6618, AUC 0.9827, OOD 0.9485. |
| R11_B | + Group DRO | FT R9_D + Group DRO | R9_D step 4000 | identical to R11_A; DRO had no effect. |
| R11_C | + wide aug | FT R9_D + wide aug | R9_D step 4000 | step ~5626, AUC 0.9821, OOD 0.9623. |
| R11_D | FT R9_F (best EER) | FT R9_F | R9_F step 500 | step ~6409, AUC 0.9829, OOD 0.9405. |
| R11_E | scratch 30K (ceiling probe) | scratch | none | step ~6401, AUC 0.9787, OOD 0.9177. |
| **R11_F (`wc3jv0ls`)** | FT R9_D + high Teams weight | FT R9_D | R9_D step 4000 | step 3000, AUC 0.9828, EER 0.0569, OOD 0.9615, **Teams holdout 88.7%** (best Teams competence). |
| **R11_G (ViT-L/14)** ★ | L/14 fix (λ=0.01, rank=768) | scratch + L/14 | none | step 5500, AUC **0.9918**, EER **0.0263**, OOD **0.9819** — best overall but **not production-eligible** (B-16 only). |
| R11_H | FT R9_D, ultra-low LR (1e-5) | FT R9_D | R9_D step 4000 | step ~5626, AUC 0.9815, OOD 0.9613. |

**R11 takeaway** (per `PHASE2_SUMMARY_R8_R11.md`): six B-16 FT runs *all* plateaued at AUC 0.982–0.983, 1.1pp below R9_D. "B-16's rank-32 residual subspace doesn't have the capacity to hold both DF40 discrimination and Teams domain knowledge." (This framing was later complicated by the camera-signature shortcut hypothesis — see §1.5.)

### 1.3 R12 (Mar 8–11): the R12_G/0xxqwhxg checkpoint that becomes the program's load-bearing FT base

R12 introduced **augmentation fixes** targeting the Teams brightness/CCT gap and the Teams OOD eval pool. Eight slots; two notable:

| ID | Hypothesis | Δ from parent | Parent ckpt | Steps | Base CLIP | Outcome |
|---|---|---|---|---|---|---|
| R12_A (`4v8av986`) | Scratch + R12 aug fixes (CCT, asymmetric brightness, wider gamma) — **k=32 baseline** | scratch (no FT base); +R12 aug fixes; Teams v2 in mix; OOD eval pool added | none | 30000 | DataComp-XL | step 7500 (ood_composite), AUC 0.9705, OOD 0.9658, VCD real 83.9%. |
| R12_B (`wupk4909`) | + k=64 (capacity) | scratch + k=64 | none | 30000 | DataComp-XL | step 7500 (ood_composite), AUC 0.9805, OOD 0.9656. |
| R12_C/F | FT R8_E + GRL | FT R8_E + GRL λ=0.1 | R8_E step 8000 | 8000 | DataComp-XL | converged instantly to 0.9832 holdout, plateaued. GRL "not worth it" verdict. |
| R12_D (k=128) | + k=128 | scratch + k=128 | none | 30000 | DataComp-XL | (UNCERTAIN: outcome not in registry summary) |
| R12_E (`ys8z1div`) | scratch + GRL + Teams OOD | scratch + GRL λ=0.1 + Teams OOD | none | 30000 | DataComp-XL | step 7500, AUC 0.9774, OOD 0.9614. GRL stalled at EWI=11. |
| **R12_G ★ (`0xxqwhxg`)** | UNCERTAIN-PARENT — see note below | UNCERTAIN | none (scratch) | 30000 (per yaml) | DataComp-XL | step 12500 (ood_composite), AUC 0.9923, OOD 0.9607, **Composite 0.9738** — leader. |
| R12_H | + GRL strong (λ=0.3) | scratch + GRL λ=0.3 | none | 30000 | DataComp-XL | "actively hurt" per registry. |

**UNCERTAIN — R12_G yaml vs W&B id mismatch**: `experiments/phase2_round12/R12_G_scratch_seed_control.yaml` declares `seed: 1337` and `description: "scratch + R12 aug fixes — seed sensitivity control (seed=1337 vs R12_A seed=737)"`. However `WINNING_RUNS_REGISTRY.md` line 33 maps `0xxqwhxg` to "**R12_G** (scratch, **seed=737**)". This is a discrepancy. The most likely explanation is that `R12_G` in the registry naming refers to the *seed=737 scratch* run that ended up under W&B id `0xxqwhxg` while the yaml in repo with `R12_G` in the filename was a re-purposed seed-control. The R12 EXPERIMENT_PLAN matrix lists R12_G as `seed=1337`. The R13 packets all consume `gs://training-job-outputs/phase2r12_experiments/0xxqwhxg/top_n_effort_20260310_step14000_auc0.9930_eer0.0253.pth` — i.e. **the W&B id is `0xxqwhxg` and the chosen checkpoint is `step14000`**, not the registry's listed `step12500` ood_composite. Receiving agent should verify which yaml actually launched `0xxqwhxg` against the W&B run record.

**R12_G/0xxqwhxg is the load-bearing scratch ancestor for everything in R13 RLP1–RLP6 and the P9 R12G fork experiments.** Cited as "R12g" in handoffs.

### 1.4 R12.5 (Mar ~10): lighting augmentation ablation

| ID | Hypothesis | Δ from parent | Parent ckpt | Outcome |
|---|---|---|---|---|
| R125_A (`gamma_up`) | + GammaUp + DirectionalShadow + wider CCT (lighting envelope close) | scratch + lighting envelope | none | (UNCERTAIN — outcomes not consolidated in WINNING_RUNS_REGISTRY) |
| R125_B (`gamma_up_shadow`) | + lighting + shadow only | scratch | none | (UNCERTAIN) |
| R125_C (`max_lighting_envelope`) | + maximum-envelope variant | scratch | none | (UNCERTAIN) |

(All R12.5 runs were `load_base_checkpoint: false`, so they are scratch-from-CLIP and do not feed the R13 RLP chain.)

### 1.5 R13 — the relaunch packets (RLP1–RLP8 + P9 + P10)

The R13 sequence is a *single deep FT chain* on the R12_G `0xxqwhxg` checkpoint, with the following segmentation. **All `gs_base_checkpoint:` values verified by direct `grep` over the yamls**.

#### 1.5.1 R13 pre-relaunch slots (R13_A, B, C, D, E, F, G, H, I + R13_FT1..FT17 + R13_TB1/2 + R13_WTB/WTC*)

These yamls live in `experiments/phase2_round13/` from Apr 8 onward (see git log). They reference earlier R13 work (R13_A "champion plus enhanced", FT1–FT17 "trackA scorecard base", etc.). For brevity I do not table every slot here — the key facts:

- Many R13_A/FT* yamls were created Apr 8–13 2026 (per `mtime`).
- R13 entered a "relaunch" phase on **2026-04-17** (`teams-relaunch-root-2026-04-17` branch creation).
- The relaunch reset: chose `0xxqwhxg/step14000` as the canonical FT base for the entire RLP1+ packet sequence (see §1.5.2).
- (UNCERTAIN: I have not enumerated FT1–FT17 outcomes for this sub-report; they pre-date the camera-signature shortcut work and the user's hypothesis is about the *recent* (P7+) lineage. The receiving agent can pull these on request.)

#### 1.5.2 RLP1 — relaunch packet 1 (8 slots, launched ~Apr 19–20)

All slots fork from `gs://training-job-outputs/phase2r12_experiments/0xxqwhxg/top_n_effort_20260310_step14000_auc0.9930_eer0.0253.pth` (= R12_G/R12g step 14000) **except slot 08 which is scratch**.

Documented in `docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_*.md`.

| ID | Hypothesis | Δ from R12g/0xxqwhxg | Steps | Outcome |
|---|---|---|---|---|
| RLP1_01 | FT control — no hints, no proper-data, live discovery | base data only | 10000 | (UNCERTAIN — listed in monitoring handoffs, individual outcome not pulled) |
| RLP1_02 | + WTB2 hints only | + hints | 10000 | (same) |
| RLP1_03 | + WTB3 hints + Teams hints | + hints + Teams hints | 10000 | (same) |
| RLP1_04 | + retained unenhanced proper-data | + proper-data unenhanced lanes | 10000 | (same) |
| RLP1_05 | + full proper-data snapshot | + full proper-data snapshot | 10000 | (same) |
| RLP1_06 | + truthful GammaUp | + truthful gamma | 10000 | (same) |
| RLP1_07 | + Teams shadow aug | + teams shadow | 10000 | (same) |
| RLP1_08 | **SCRATCH** hedge — full proper-data, scratch | scratch (no FT base) | **30000** | (UNCERTAIN — was the scratch hedge; outcome not consolidated here) |

#### 1.5.3 RLP2 (6 slots, launched ~Apr 21)

Same parent (`0xxqwhxg/step14000`). Iterated on RLP1 results.

| ID | Δ from R12g | Outcome |
|---|---|---|
| RLP2_01 | refresh of RLP1_01 (no hints) | (UNCERTAIN) |
| RLP2_02 | + proper-data unenhanced | (UNCERTAIN) |
| RLP2_03 | + proper-data + Teams enhanced | (UNCERTAIN) |
| RLP2_04 | + proper-data + clean enhanced | (UNCERTAIN) |
| RLP2_05 | + spatial stability aug | (UNCERTAIN) |
| RLP2_06 | + low arcface | (UNCERTAIN) |

#### 1.5.4 RLP3 (8 slots — `RLP3_00` smoke + 01–08, launched ~Apr 21–22)

Same parent. RLP3_02 became the "FT proper main" reference. There are three derivative RLP3_02 yamls in repo (legacy gates, new gates) for retro-scoring sanity.

| ID | Δ from R12g | Outcome |
|---|---|---|
| RLP3_00_PRELAUNCH_SMOKE | 1200-step smoke verifying instrumentation | passed |
| RLP3_01_FT_control | drift probe vs RLP2_01 | (UNCERTAIN) |
| RLP3_02_FT_proper_main | reconfirm RLP2_02 under A1–A10 instrumentation | (UNCERTAIN) |
| RLP3_03_FT_proper_low_arcface | s_end=10 | (UNCERTAIN) |
| RLP3_04_FT_proper_spatial | + stronger spatial aug | (UNCERTAIN) |
| RLP3_05 | low_arcface + spatial | (UNCERTAIN) |
| RLP3_06_FT_proper_main_seedB | seed variance | (UNCERTAIN) |
| RLP3_07_FT_proper_lighting_aug | + lighting aug | (UNCERTAIN) |
| RLP3_08_FT_proper_main_plus_dose_matched_enhanced | + enhanced lane | (UNCERTAIN) |

(Retrospective scoring at `R13_RELAUNCH_PACKET3_RETRO_SCORE_RESULTS_2026-04-22.md`.)

#### 1.5.5 RLP3.5 (7 slots — arcface margin + stability λ + label smooth + family rebalance, ~Apr 22)

Same parent. Probes margin/stability/LS axes that the R9.5 era refuted but were re-litigated under the new gates.

| ID | Δ from R12g | Outcome |
|---|---|---|
| RLP35_01 | arcface m=0.10 | (UNCERTAIN — not consolidated) |
| RLP35_02 | arcface m=0.15 | (UNCERTAIN) |
| RLP35_03 | arcface m=0.20 | (UNCERTAIN — abort criterion was AUC < 0.95 at 1k) |
| RLP35_04 | stability λ=0.03 | (UNCERTAIN) |
| RLP35_05 | label_smoothing 0.05 | (UNCERTAIN) |
| RLP35_06 | family rebalance proper-up | (UNCERTAIN) |
| RLP35_07 | stack top-3 | (UNCERTAIN) |

#### 1.5.6 RLP4 (8 slots, ~Apr 22 evening)

Same parent. ArcFace margin × seed × spatial × Teams reweight crosses.

(All 8 fork from `0xxqwhxg/step14000`. Outcomes mostly unrecorded in current repo state; documented in `R13_RELAUNCH_PACKET4_PLANNING_HANDOFF_2026-04-22.md`.)

#### 1.5.7 RLP5 (8 slots, ~Apr 23)

Same parent. **Slot RLP5_07 (E3 seedB)** became the packet-5 "leader" referenced in PACKET6 docs (composite 0.7736 @ step 20500 → run id `6jwwb526`).

| ID | Δ from R12g | Outcome |
|---|---|---|
| RLP5_01..06,08 | E3/E1 + teams/enh weight axes | (UNCERTAIN — not all consolidated) |
| **RLP5_07 (`6jwwb526`) ★** | E3 seedB | **step 20500, AUC 0.9908, EER 0.0304, value_composite 0.7736 (packet-5 leader).** This is the ckpt where the camera-signature shortcut was first cleanly identified per memory `project_signature_shortcut_finding`. |

#### 1.5.8 RLP6 (8 slots, ~Apr 23)

**Same parent (`0xxqwhxg/step14000`)**, 8 slots. The PACKET6 plan (`R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md`) specifies that RLP6 inherits from RLP5_07 *conceptually* (i.e., applies the gate-alignment recipe over the same fork point). RLP6 introduces **gate alignment** (drop avspeech + zoom_vcd_real from external_real_sources, `path_exclude_contains: ["/visomaster_"]` on teams_ood_fake, drop wma_failure_fake) and then explores adding proper-data lanes.

| ID | Δ from R12g | Outcome |
|---|---|---|
| RLP6_01 | gate align canary (3 levers stacked) | (UNCERTAIN) |
| RLP6_02 | hint-clean only | (UNCERTAIN) |
| RLP6_03 | drop wma only | (UNCERTAIN) |
| **RLP6_04 (`h2pdu6i5`) ★** | canary + add proper_visomaster_enhanced_clean lane @ weight 2.0 | **step 23500, AUC 0.9942, EER 0.0169, value_composite 0.9006**. *This is the load-bearing FT base for every P7/P8/P9/P10 experiment that uses RLP6_04 as parent.* |
| RLP6_05 | seed variance | (UNCERTAIN) |
| RLP6_06 | avspeech readout-only | (UNCERTAIN) |
| RLP6_07 | heavy enh (proper_enh_clean weight 4.0) | (UNCERTAIN) |
| RLP6_08 | E1 backbone gate align | (UNCERTAIN) |

#### 1.5.9 RLP6B (8 slots, parent: R12g) — second-line probes

These are also forked from `0xxqwhxg/step14000`. They explore quality-domain/GRL strength, mixup, label-smoothing on the fake side, feature-norm regularization, and small "combo-lite/aggressive" stacks.

**RLP6B is the first time `use_quality_domain_head: true` is used in R13** (verified by direct grep over yamls):

| ID | Δ from R12g | quality_domain_loss_weight | Outcome |
|---|---|---|---|
| RLP6B_01 | + quality λ 2x | 0.2 | (UNCERTAIN) |
| RLP6B_02 | + quality λ 4x | 0.4 | (UNCERTAIN) |
| RLP6B_03 | + arcface high margin | — | (UNCERTAIN) |
| RLP6B_04 | + mixup strong | — | (UNCERTAIN) |
| RLP6B_05 | + label smooth fake-only | — | (UNCERTAIN) |
| RLP6B_06 | + feat norm reg | — | (UNCERTAIN) |
| RLP6B_07 | combo lite | 0.2 | (UNCERTAIN) |
| RLP6B_08 | combo aggressive | 0.4 | (UNCERTAIN) |

(UNCERTAIN whether RLP6B was actually scored and if any RLP6B slot promoted; no outcome doc retrieved here. The fact that RLP7+ all use RLP6_04 as base — not an RLP6B slot — suggests RLP6B did not produce a promotable winner.)

#### 1.5.10 RLP7 — Packet 7 (8 slots, ~Apr 24): augmentation-only attack on the camera-signature shortcut

**Parent: RLP6_04 step 23500** (`h2pdu6i5`) for slots 01–07. Slot 08 forks from `step 4500` of the same run (early-fork hypothesis test).

| ID | Hypothesis | Δ from RLP6_04 | Anchor Δ vs RLP6_04 | Roee-mac Δ | Verdict |
|---|---|---|---:|---:|---|
| RLP7_01 (UNCERTAIN — yaml present but not in P8A pre-doc summary) | lighting aggressive | + lighting aug | (UNCERTAIN) | (UNCERTAIN) | (UNCERTAIN) |
| **RLP7_02 (`f5fav9u0`)** | codec-aggressive aug | quality_p 0.72 / jpeg_lower 30 / webcam_codec_p 0.35 | −0.076 | −0.169 | marginal |
| RLP7_03 | combined | combined aug axes | (UNCERTAIN) | (UNCERTAIN) | (UNCERTAIN) |
| RLP7_04 | spatial-only | ShiftScaleRotate | −0.075 | −0.129 | marginal |
| **RLP7_05 (`hhc8quq9`)** ★ | spatial + moderate codec | spatial+codec balanced | **−0.089** | −0.154 | **best balanced of P7** |
| RLP7_06 | CCT-only | CCT shift only | −0.015 | −0.134 | weak |
| RLP7_07 | triple (spatial + codec + CCT) | three axes | −0.038 | −0.228 | best Roee-mac, weak anchor |
| RLP7_08 | codec aug + earlier fork | parent: `h2pdu6i5/step4500 ood_composite` (NOT step 23500) | −0.097 | +0.072 | best anchor; refuted "late consolidation" |

P7 runs are 10000 steps each on top of the 23500 (or 4500) RLP6_04 step.

#### 1.5.11 RLP8 — Packet 8 (2 slots, Apr 24–25): backbone unfreeze + scratch on plain CLIP

| ID | Hypothesis | Δ from RLP7_02 | Parent ckpt | Steps | Outcome |
|---|---|---|---|---|---|
| **P8A = RLP8_01 (`9lmvb5b4`)** ★ | Unfreeze CLIP backbone partially | `unfreeze_final_proj=true` + `unfreeze_final_ln=true` + `apply_svd_to_mlp=true` (vs RLP7_02's all-false) | RLP6_04 step 23500 | 10000 (early-stopped at 5000) | step 5000, AUC 0.9926, EER 0.0270, **lockbox_real_fpr 0.147%** (vs 0.441% RLP6_04), anchor Δ −0.188 (best-ever). **But −13.6 pp aggregate fake-recall regression on hard methods**. |
| **P8B = RLP8_02 (`n8yk2hox`)** | Scratch on plain CLIP — refute "shortcut in R13 chain" | `load_base_checkpoint: false` + LR 2e-4 + 30k steps | scratch (no FT base) | 30000 | step 11000/30000 (cancelled — `teams_ood_fake` data loader hang at step 12000), anchor Δ **+0.066** (worse than RLP6_04). Confirmed shortcut is in data mix, not weights. |

P8A's checkpoint at step 5000 is the second key fork point in this lineage (alongside RLP6_04 step 23500). The P10_SYM_on_P8A run (§1.5.13) forks from it.

#### 1.5.12 RLP9 / Packet 9 (10 slots, launched 2026-04-25 to 2026-04-26)

P9 is "softened P8A" + chain probes + reseed + schedule. Documented in `R13_FULL_STORY_PRE_PACKET_9_2026-04-25.md` and `PACKET_9_MID_FLIGHT_HANDOFF_2026-04-26.md`. As of 2026-04-26 morning, the P9 promotion-contract scorecard was still running and outcomes are not yet recorded on disk.

| ID | Hypothesis | Δ from P8A | Parent ckpt | Outcome (per Apr 26 handoff) |
|---|---|---|---|---|
| **P9_01** (`R13_P9_01_softened_p8a`) | Magnitude — too-hot training caused regression | `optimizer.adam.backbone_lr_mult: 0.3` (else identical to P8A) | RLP6_04 step 23500 | RUNNING at handoff time |
| **P9_03** (`R13_P9_03_no_mlp_svd`) | Topology — MLP-SVD reaches too deep | `apply_svd_to_mlp: false` (else identical to P8A) | RLP6_04 step 23500 | RUNNING |
| **P9_05** (`5oc7zyzw`) | Data — real/fake codec asymmetry | + `augmentation.real_codec_uplift: true` | RLP6_04 step 23500 | step 9000 finished |
| **P9_R** (`a5w629y1`) | Replication — is P8A reproducible? | seed 749→938 + split_seed 737→938 | RLP6_04 step 23500 | step 3500 finished |
| **P9_freeze** (`ku6fdljz`) | Disentangle magnitude vs topology — head-only | `backbone_lr_mult: 0.0` | RLP6_04 step 23500 | step 9000 finished |
| **P9_AB_stack** | All-in: lr_mult 0.3 + no MLP-SVD + real_codec_uplift | three axes stacked | RLP6_04 step 23500 | RUNNING |
| **P9_FORK_p8a_on_r12g** | Chain probe — P8A recipe rooted on R12g (skip RLP6_04) | base ckpt → R12g step 14000 (NOT RLP6_04) | R12g/0xxqwhxg step 14000 | PENDING quota |
| **P9_FORK_AB_stack_on_r12g** | Max longshot: AB stack on R12g | base ckpt → R12g step 14000 + AB stack | R12g step 14000 | RUNNING |
| **P9_FORK_DATA** (`6pd7o8iz`) | Pure data probe — only real_codec_uplift, no P8A flags | + real_codec_uplift, drop all P8A unfreeze flags | R12g step 14000 | step 5500 finished |
| **P9_LONG_extended** | Schedule probe — P8A recipe to 16000 steps | + total_training_steps 16000 | RLP6_04 step 23500 | RUNNING (europe-west4) |

#### 1.5.13 RLP10 / Packet 10 (6 slots, defined 2026-04-26, anti-shortcut track)

**The critical packet for the user's hypothesis.** P10 explicitly tests the "shortcut is installed by training data + augmentation routing" hypothesis. Five slots fork from RLP6_04, one (`SYM_on_P8A`) forks from P8A.

| ID | Hypothesis tested | Δ from RLP6_04 FT recipe | Parent ckpt | Backbone-trainable surface | Outcome |
|---|---|---|---|---|---|
| **P10_SYM_baseline** | H1: family-aware aug router installs the shortcut by giving fakes heavier degradation than reals | `augmentation.routing.mode: "symmetric"` (else identical) | RLP6_04 step 23500 | `apply_svd_to_in_proj` only (no unfreeze) | Defined; checkpoint map exists `teams_target_domain.p10_partial_2026-04-26.yaml`. (UNCERTAIN — running/done?) |
| **P10_SYM_light** | H1 sub-question: is recovery from symmetry per se or from lower total degradation budget? | symmetric + `quality_p 0.40 / jpeg_lower 55` (lower budget) | RLP6_04 step 23500 | same as above | (UNCERTAIN) |
| **P10_GRL_baseline** | H2: GRL on quality-domain head makes embedding invariant to quality dim, even with asymmetric router | `use_quality_domain_head: true` + `quality_domain_loss_weight: 0.1` (else identical, family_aware router still on) | RLP6_04 step 23500 | same as above | (UNCERTAIN) |
| **P10_GRL_strong** | GRL λ axis: 0.1 → 0.25 (R6_S7's attempted-but-broken value) | quality_domain_loss_weight 0.25 | RLP6_04 step 23500 | same | (UNCERTAIN) |
| **P10_SYM_GRL** | H3: do SYM and GRL stack? | symmetric + GRL λ=0.1 | RLP6_04 step 23500 | same | (UNCERTAIN) |
| **P10_SYM_on_P8A** | H4: P8A's per-method regression caused by the asymmetric router teaching the unfrozen backbone a stronger version of the shortcut | symmetric + P8A unfreeze flags (`unfreeze_final_proj`+`unfreeze_final_ln`+`apply_svd_to_mlp` all true) | **P8A step 5000** (NOT RLP6_04) | **proj + ln_post + MLP-SVD trainable** | (UNCERTAIN — partial, per checkpoint map name `p10_partial_2026-04-26.yaml`) |

### 1.6 Backbone variant rollcall

The CLIP base for *every* run in this codebase except R10_D / R11_G is OpenCLIP **ViT-B-16 with `pretrained="datacomp_xl_s13b_b90k"`** (LAION DataComp-XL pretraining, ~13B image-text pairs, batch 90k). This is `gs://base-checkpoints/effort-aigi/CLIP-ViT-B-16-DataComp.XL-s13B-b90K/` in the gcs_assets section of every R13 yaml.

R10_D, R11_G are ViT-L/14 (different pretraining); R11_G ran successfully but is *not production-eligible*.

### 1.7 Evidence sources for §1

- `experiments/PHASE2_SUMMARY_R8_R11.md` (R8–R11 final summary).
- `experiments/WINNING_RUNS_REGISTRY.md` (pre-program-relaunch winning-runs registry).
- `experiments/EXPERIMENT_JOURNEY.md` (early R-rounds narrative).
- `experiments/R8_SESSION_SUMMERY.txt` (R8 session log).
- `experiments/phase2_round12/R12_EXPERIMENT_PLAN.md` (R12 plan and motivation).
- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET{1,2,3,3_5,4,6,7}_*.md` (RLP plans + monitoring).
- `docs/relaunch_handoffs/R13_FULL_STORY_PRE_PACKET_9_2026-04-25.md` (full P7/P8 detail + P9 design).
- `docs/relaunch_handoffs/R13_RELAUNCH_STATE_OF_THE_TEAMS_DETECTOR_2026-04-25.md` (P8A breakthrough doc).
- `docs/relaunch_handoffs/PACKET_9_MID_FLIGHT_HANDOFF_2026-04-26.md` (P9 mid-flight state).
- `experiments/phase2_round13/R13_*.yaml` (every R13 yaml — `gcs_base_checkpoint:` directly verified by grep).
- `analysis/pool_rescore_rlp{6_04,7_*,8_a,8_b}.summary.json` (anchor-pool rescores — mentioned, not opened here).
- Memory files at `/Users/roeedar/.claude/projects/.../memory/`: `project_p8a_breakthrough.md`, `project_shortcut_is_upstream.md`, `project_signature_shortcut_finding.md`.
- Git log over `teams-relaunch-root-2026-04-17` branch since 2026-01-01.

---

## 2. Checkpoint lineage tree

Forking points are marked with `*`. "Steps trained" is the run's *own* steps; cumulative trained steps are in §3. ASCII tree is rooted at OpenCLIP ViT-B-16-DataComp-XL — the *only* CLIP base in this program (excluding L/14 dead-end).

```
OpenCLIP ViT-B-16 / DataComp-XL (datacomp_xl_s13b_b90k)
│   weights file: gs://base-checkpoints/effort-aigi/CLIP-ViT-B-16-DataComp.XL-s13B-b90K/
│   13B image-text pairs, batch 90k. Frozen, except for SVD residuals + ArcFace head.
│
├── (Phase 1 ~Jan 19) B16-old, k=8, scratch — step 14000, AUC 0.9947 [DEAD; superseded]
│
├── (R1 Feb 11) P2_C3 cosine softmax, k=8 rank=760 — INVALIDATED (data bug)
│
├── (R2 Feb 12) R2_A3 (CE, k=8 rank=760) — step 10000, AUC 0.9866
│
├── (R2.5 Feb 12) R25_F1 (k=32, scratch) — step 18500, AUC 0.9893
│   │
│   ├── (R3 Feb 13) R3_FT3 — FT R25_F1 — step 2500, AUC 0.9966 holdout
│   │   │
│   │   └── (R4 Feb 17) R4_FT7 (`udgwsu7o`) — FT R3_FT3 — step 500, AUC 0.9935
│   │
│   └── (R5) [no winner displaced FT7]
│
├── (R6 Feb 19) R6_S1 (`s3tx3fk4`) — scratch + VCD reals — step 6000, AUC 0.9691 [GRL plumbing broken]
│
├── (R7) [subsumed by R8]
│
├── (R8 Feb 24) R8_E (`hu7cen3m`) ★ — scratch, target-heavy — step 8000, AUC 0.9925
│   │
│   ├── (R9 Feb 28) R9_A (`1551zxa8`) — FT R8_E + Teams v1 — step 6000, AUC 0.9891 [PRODUCTION 2026-02]
│   │   │
│   │   └── (R10 Mar 6–7) R10_C — FT R9_A + Teams v2 — step 8167, AUC 0.9836
│   │
│   └── (R12_C/D/F Mar 8) R12_C/D/F — FT R8_E + R12 aug fixes [+ GRL for D/F] — converged to AUC 0.9832 instantly
│
├── (R9 Feb 28) R9_D (`m7etxxnp`) ★ — scratch, NO TEAMS DATA — step 4000, AUC 0.9942 [best B-16 ever]
│   │
│   └── (R11 Mar 7–8) R11_A/B/C/F/H — FT R9_D + Teams v2 — all plateaued AUC ~0.982–0.983
│       │
│       └── R11_F (`wc3jv0ls`) — high Teams weight — step 3000, Teams holdout 88.7%
│
├── (R9 Feb 28) R9_F (`ueoziuou`) — FT + codec sim — step 500, EER 0.0266
│   │
│   └── (R11 Mar 8) R11_D — FT R9_F — step ~6409, AUC 0.9829
│
├── (R10 Mar 7) R10_D — ViT-L/14, lambda_reg=1.0, rank=1023 [BROKEN config]
│
├── (R11 Mar 8) R11_G — ViT-L/14, FIXED config — step 5500, AUC 0.9918, OOD 0.9819
│   [DEAD-END for production: B-16 only]
│
├── (R12_A Mar 9) R12_A (`4v8av986`) — scratch + R12 aug fixes (CCT, asym brightness, wider gamma)
│                                       + Teams v2 + Teams OOD eval
│                                     — 30k steps, ood_composite at step 7500, AUC 0.9705
│
├── (R12_B Mar 9) R12_B — scratch + k=64 — step 7500, AUC 0.9805
│
├── (R12_E Mar 9) R12_E (`ys8z1div`) — scratch + GRL λ=0.1 + Teams OOD — step 7500, AUC 0.9774
│
├── (R12_G Mar 10) R12_G ★★★ (`0xxqwhxg`) — scratch — step 14000, AUC 0.9930, EER 0.0253
│   │   [UNCERTAIN: registry says seed=737, repo yaml R12_G_scratch_seed_control says seed=1337
│   │    — see §1.3 note. The W&B id 0xxqwhxg's actual yaml-of-record is the question.]
│   │
│   │   * THIS IS THE FT BASE FOR EVERYTHING IN R13 RLP1–RLP6 AND THE R12g-FORK P9 SLOTS *
│   │
│   ├── (RLP1 ~Apr 19) RLP1_01..07 — FT 10k from R12g step 14000 [+ varied data axes]
│   │   └── RLP1_08 — SCRATCH 30k hedge (separate; not in this chain)
│   │
│   ├── (RLP2 ~Apr 21) RLP2_01..06 — FT 10k from R12g [+ refreshed live snapshot, varied data axes]
│   │
│   ├── (RLP3 + RLP3.5 ~Apr 21–22) RLP3_00..08 + RLP35_01..07 — FT 10k from R12g
│   │     [proper-data + arcface margin + stability + label smooth + family rebalance probes]
│   │
│   ├── (RLP4 ~Apr 22 night) RLP4_01..08 — FT 10k from R12g [arcface × seed × spatial × Teams reweight]
│   │
│   ├── (RLP5 ~Apr 23) RLP5_01..08 — FT 10k from R12g
│   │   └── RLP5_07 (`6jwwb526`) ★ — E3 seedB — step 20500, AUC 0.9908, value_composite 0.7736
│   │       (camera-signature shortcut FIRST DIAGNOSED on this checkpoint)
│   │
│   ├── (RLP6 ~Apr 23) RLP6_01..08 — FT 10k from R12g
│   │   └── RLP6_04 ★★★ (`h2pdu6i5`) — gate-align canary + add proper_visomaster_enhanced_clean lane
│   │       — step 23500, AUC 0.9942, EER 0.0169, value_composite 0.9006
│   │       │
│   │       │   * THIS IS THE FT BASE FOR ALL P7/P8/P9/P10 RUNS (except scratch hedges) *
│   │       │
│   │       ├── (RLP7 Apr 24, 8 slots) — FT 10k from RLP6_04 step 23500
│   │       │   ├── RLP7_01 (lighting agg)
│   │       │   ├── RLP7_02 (`f5fav9u0`) — codec aggressive
│   │       │   ├── RLP7_03 (combined)
│   │       │   ├── RLP7_04 (spatial only)
│   │       │   ├── RLP7_05 ★ (`hhc8quq9`) — spatial+codec (best balanced)
│   │       │   ├── RLP7_06 (CCT only)
│   │       │   ├── RLP7_07 (S+C+T triple)
│   │       │   └── RLP7_08 — codec, FORKED FROM RLP6_04 step 4500 (not 23500)
│   │       │
│   │       ├── (P8A = RLP8_01 Apr 24) (`9lmvb5b4`) ★★★ — RLP7_02 aug + UNFREEZE
│   │       │   `unfreeze_final_proj=true`, `unfreeze_final_ln=true`, `apply_svd_to_mlp=true`
│   │       │   step 5000 (early-stopped from 10k), AUC 0.9926, lockbox_real_fpr 0.147%
│   │       │   anchor Δ −0.188 (best ever) BUT −13.6 pp aggregate fake-recall regression
│   │       │   │
│   │       │   * SECONDARY FT BASE for P10_SYM_on_P8A and P9_R replication targets *
│   │       │   │
│   │       │   ├── (P9 Apr 25–26, 5 RLP6_04-based slots) — FT 10k from RLP6_04 step 23500
│   │       │   │   ├── P9_01 — + backbone_lr_mult 0.3 (magnitude)
│   │       │   │   ├── P9_03 — − apply_svd_to_mlp (topology)
│   │       │   │   ├── P9_05 — + real_codec_uplift (data)
│   │       │   │   ├── P9_R — seed reseed (replication)
│   │       │   │   ├── P9_freeze — backbone_lr_mult 0.0 (head-only disentangle)
│   │       │   │   ├── P9_AB_stack — three-axis combo
│   │       │   │   └── P9_LONG_extended — 16k schedule
│   │       │   │
│   │       │   └── (P10_SYM_on_P8A Apr 26) — FT 10k from P8A step 5000
│   │       │       + symmetric routing (overrides family_aware)
│   │       │
│   │       └── (P10 Apr 26, 5 RLP6_04-based slots) — FT 10k from RLP6_04 step 23500
│   │           ├── P10_SYM_baseline — symmetric router (no unfreeze)
│   │           ├── P10_SYM_light — symmetric + lighter degradation budget
│   │           ├── P10_GRL_baseline — quality-GRL λ=0.1 (family_aware router still on)
│   │           ├── P10_GRL_strong — quality-GRL λ=0.25
│   │           └── P10_SYM_GRL — symmetric + GRL λ=0.1 stacked
│   │
│   ├── (RLP6B Apr 23, 8 slots) — FT 10k from R12g step 14000 [GRL/mixup/feat-norm/etc probes]
│   │
│   └── (P9 R12g-fork branch Apr 25–26, 3 slots)
│       ├── P9_FORK_p8a_on_r12g — P8A recipe rooted on R12g step 14000
│       ├── P9_FORK_AB_stack_on_r12g — AB stack on R12g
│       └── P9_FORK_DATA_codec_on_r12g — pure data probe (no P8A flags) on R12g
│
├── (R12.5 Mar ~10) R125_A/B/C — scratch + lighting envelope augmentations [DEAD: not in R13 chain]
│
├── (R13_A..I, R13_FT1..17, R13_TB1..2, R13_WTB*..WTC* yaml slots Apr 8–17)
│   [Many yamls present pre-relaunch; not detailed in this report — pre-camera-shortcut era]
│
├── (P8B = RLP8_02 Apr 24) (`n8yk2hox`) — scratch on plain CLIP (NOT in R13 chain!)
│   30k steps planned, hung at step 12000 (data loader on teams_ood_fake), CANCELLED step 11000
│   anchor Δ +0.066 (worse than RLP6_04). Confirms shortcut is in data mix, not weights.
│
└── (RLP1_08 Apr 19) SCRATCH 30k hedge — also scratch, not in R13 FT chain.
    (UNCERTAIN: outcome; may have failed or not promoted)
```

**Sibling sets** (runs that share the *same* parent and differ on a single axis):

- **RLP7_01..07** are siblings of each other (parent: RLP6_04 step 23500, axis: aug strategy).
- **RLP7_08** is *not* a sibling (different fork: RLP6_04 step 4500).
- **P8A vs RLP7_02** are siblings (parent: RLP6_04 step 23500, axis: backbone unfreeze flags).
- **P8A vs P8B** are *not* siblings — P8A inherits R13 chain, P8B is scratch on plain CLIP.
- **P9_01, P9_03, P9_05, P9_R, P9_freeze, P9_LONG** are siblings of each other & of P8A (parent: RLP6_04 step 23500, axis: one-of {lr-mult, MLP-SVD on/off, real-codec aug, seed, backbone_lr_mult=0, schedule}).
- **P9_AB_stack** is a 3-axis combo, not single-variable.
- **P9_FORK_*_on_r12g** are siblings of each other (parent: R12g step 14000) but *not* of the rest of P9 — different fork point.
- **P10_SYM_baseline, P10_SYM_light, P10_GRL_baseline, P10_GRL_strong, P10_SYM_GRL** are siblings of each other (parent: RLP6_04 step 23500, frozen-style backbone).
- **P10_SYM_on_P8A** is alone in its sub-sibling set: it forks from **P8A step 5000** and applies symmetric routing on top of the unfreeze recipe.

---

## 3. Cumulative FT-step accounting

This is the central number for the user's hypothesis. Every R13 P-series experiment is *itself* the result of multiple stacked FT phases. The cumulative trained steps from CLIP-base to the most recent runs:

### 3.1 The "main" recent path: CLIP → R12g → RLP6_04 → P-series

| Stage | Run | Trained steps in this stage | Cumulative since CLIP |
|---|---|---:|---:|
| 0. CLIP ViT-B-16 / DataComp-XL | base | 0 (frozen) | 0 |
| 1. R12_G / `0xxqwhxg` (scratch with R12 aug fixes) | scratch from CLIP | **14000** (the chosen ckpt step) | **14000** |
| 2. RLP6_04 / `h2pdu6i5` (FT from R12g + gate align + proper_enh_clean) | FT from R12g step 14000 | **23500** (the chosen ckpt step) | **37500** |
| 3a. P8A / `9lmvb5b4` (FT from RLP6_04 + 3 unfreeze flags) | FT from RLP6_04 step 23500 | **5000** (early-stopped) | **42500** |
| 3b. RLP7_05 / `hhc8quq9` (FT from RLP6_04 + spatial+codec aug) | FT from RLP6_04 step 23500 | **~10000** (UNCERTAIN: which step the `hhc8quq9` checkpoint was selected at — `value_composite` or `ood_composite`; 10k is the schedule, actual selection may be earlier) | up to 47500 |
| 4. P10_SYM_on_P8A | FT from P8A step 5000 | up to 10000 (schedule; UNCERTAIN whether finished) | up to **52500** |

So **a P10_SYM_on_P8A finished checkpoint is the result of ~52500 cumulative FT steps starting from frozen CLIP**, broken across three FT phases (R12g 14k → RLP6_04 23.5k → P8A 5k → P10 10k).

### 3.2 The "RLP6_04-anchored" P-series path

For all P7/P8/P9/P10 *non-fork* slots that train from RLP6_04 step 23500:

| Stage | Run | Steps | Cumulative |
|---|---|---:|---:|
| 0. CLIP base | — | 0 | 0 |
| 1. R12_G (scratch, R12 aug) | `0xxqwhxg` step 14000 | 14000 | **14000** |
| 2. RLP6_04 (FT R12g, gate align, +proper_enh_clean) | `h2pdu6i5` step 23500 | 23500 | **37500** |
| 3. Any P-series slot at full schedule | 10000 steps | 10000 | **47500** |

So **a finished P10 slot (e.g. P10_SYM_baseline at step 10000) inherits ~47500 cumulative FT steps before its own 10000 begin**. After it finishes its own 10000 steps, that's **~47500 + 10000 = ~47500 if we count "its" 10000 as already in the 47500 total**, OR **57500** if we count "the full chain inclusive of the new run". Phrasing is ambiguous; the inclusive number is **~47500 trained steps from CLIP (14k + 23.5k + 10k)**.

### 3.3 P9 R12G-fork path (skips RLP6_04)

For `P9_FORK_p8a_on_r12g`, `P9_FORK_AB_stack_on_r12g`, `P9_FORK_DATA_codec_on_r12g`:

| Stage | Steps | Cumulative |
|---|---:|---:|
| 0. CLIP base | 0 | 0 |
| 1. R12g step 14000 | 14000 | 14000 |
| 2. P9 fork-slot (10k schedule) | up to 10000 | up to **24000** |

These are the **minimum-FT-depth** experiments in the R13 P9/P10 family — they have ~24000 cumulative FT steps after their own training, vs ~47500 for the RLP6_04-anchored slots.

### 3.4 The "old" R8/R9/R11 path (now-historical)

For comparison only — the production-2026-02 model (R9_A) and the registry's R12_C/D/F:

| Path | Stages | Cumulative |
|---|---|---:|
| R8_E (scratch) → R9_A (FT) | 8000 + 6000 | 14000 |
| R8_E → R12_C/D/F (FT) | 8000 + 8000 | 16000 |
| R9_D (scratch) → R11_F (FT) | 4000 + 3000 | 7000 |

These chains are **~3–7× shorter** in cumulative FT-steps than the current R13 P-series chain.

### 3.5 Summary numbers (the user's central concern)

| Recent experiment | Total cumulative FT steps from frozen CLIP base | # of FT phases |
|---|---:|---:|
| R12_G / R12g (scratch with R12 aug) | **14000** | 1 |
| RLP6_04 step 23500 | **37500** | 2 (R12g → RLP6_04) |
| P7 family (RLP7_01..07 at 10k schedule) | up to **47500** | 3 |
| P7 RLP7_08 (forked from RLP6_04 step 4500) | up to **27500** | 3 (R12g → RLP6_04 step 4500 → P7) |
| **P8A step 5000** | **42500** | 3 |
| P8B step 11000 | 11000 | 1 (scratch) |
| **P9 (RLP6_04-anchored, 10k schedule)** | up to **47500** | 3 |
| **P9 (R12g-fork, 10k schedule)** | up to **24000** | 2 |
| **P10 (RLP6_04-anchored, 10k schedule)** | up to **47500** | 3 |
| **P10_SYM_on_P8A (10k schedule)** | up to **52500** | 4 (R12g → RLP6_04 → P8A → P10) |

**The user's hypothesis ("shortcuts get baked in early") is consistent with these numbers**: the P-series experiments inherit 37500 cumulative FT steps before any "anti-shortcut" intervention is applied.

---

## 4. What's frozen vs what's tunable in each FT pass

This determines whether the inherited shortcut *can* be unlearned by the FT phase, mechanically.

### 4.1 Default ("no unfreeze flags") — the recipe used in R12g, RLP6_04, all of P7, all of P9 except FORK variants, and P10_SYM/_GRL

```yaml
backbone:
  apply_svd_to_in_proj: true
  # all other flags unset / false
```

Trainable (per `detectors/effort_detector.py`):
- ArcFace head (final classifier weights + bias, per-class).
- SVD residual on `attention.in_proj` per transformer block: `U_residual ∈ ℝ^(d×r)`, `S_residual ∈ ℝ^r`, `V_residual ∈ ℝ^(r×d)`. With rank=736, `r=736`. Trainable params per block ≈ `2·d·r + r = 2·512·736 + 736 ≈ 754k` per block × 12 blocks ≈ 9.05M trainable.

Frozen (everything else in CLIP visual tower):
- Patch embedding (`visual.conv1`, `visual.class_embedding`, `visual.positional_embedding`).
- Per-block: `attention.in_proj_weight` (additive base — the SVD residual is *added*, not replacing), `attention.out_proj`, MLP `c_fc` and `c_proj`, layer norms.
- Final `visual.proj` (768→512), `visual.ln_post`.

**Mechanical implication for shortcut removal**: in this default recipe, the *only* path the model has to "unlearn" a backbone-level shortcut is via the SVD-on-attention-in_proj residuals. The frozen CLIP attention input projection itself cannot be modified, only counter-adjusted by the residual. This is the "FT is reach-limited" hypothesis (memory `project_shortcut_is_upstream`).

### 4.2 P8A recipe — partial unfreeze (used in P8A, all P9 RLP6_04-anchored slots except `P9_03`/`P9_FORK_DATA`, `P10_SYM_on_P8A`)

```yaml
backbone:
  apply_svd_to_in_proj: true
  unfreeze_final_proj: true   # visual.proj (768→512 final projection)
  unfreeze_final_ln:   true   # visual.ln_post (final LayerNorm)
  apply_svd_to_mlp:    true   # SVD residual on MLP c_fc + c_proj per block
```

Trainable additions vs default:
- `visual.proj`: 768·512 + 512 ≈ 393k params.
- `visual.ln_post`: 2·512 = 1024 params.
- SVD residual on MLP `c_fc` (768→3072 per block): per block ≈ `2·d·r + r = 2·768·736 + 736 ≈ 1.13M`, possibly times 12 blocks (but the YAML doesn't restrict — UNCERTAIN whether this applies to all 12 blocks or last-N; per the R13_FULL_STORY_PRE_PACKET_9 doc it's "all blocks" but I have not verified this against `effort_detector.py`).

Estimated total trainable params ≈ 9M (in_proj SVD) + 13.6M (MLP SVD) + 393k (proj) + 1k (ln_post) ≈ **23M trainable**, vs ~9M for the default — roughly **2.5× more degree of freedom**.

**Mechanical implication**: P8A is the first recipe in R13 that gives FT enough reach to redistribute features in the visual tower itself, not just inject a residual on the attention input. This is what "broke the camera-signature ceiling."

### 4.3 P9 sub-variants (most relax one of the three P8A flags, or set lr-mult)

| Slot | Trainable surface vs P8A |
|---|---|
| **P9_01** | identical to P8A; `optimizer.adam.backbone_lr_mult: 0.3` caps the *learning rate* on the unfrozen-backbone params (proj + ln_post + SVD residuals) to 0.3× base 3e-5 = 9e-6. Head still at base LR. |
| **P9_03** | `apply_svd_to_mlp: false` — drops MLP-SVD reach. Keeps proj + ln_post unfreeze + attention-in_proj-SVD. ~10M trainable. |
| **P9_05** | identical to P8A trainable surface; data-side change only (`real_codec_uplift: true`). |
| **P9_R** | identical to P8A; only seed+split_seed changed (749→938). |
| **P9_freeze** | identical to P8A trainable surface; `backbone_lr_mult: 0.0` — backbone params receive zero LR; effectively head-only training. |
| **P9_AB_stack** | apply_svd_to_mlp: false + lr_mult 0.3 + real_codec_uplift. Reduced reach + capped LR + data axis. |
| **P9_LONG** | identical to P8A; schedule extended to 16k. |
| **P9_FORK_p8a_on_r12g** | identical to P8A trainable surface; only base ckpt changed. |
| **P9_FORK_AB_stack_on_r12g** | AB-stack surface + R12g base. |
| **P9_FORK_DATA_codec_on_r12g** | **default (no unfreeze)** trainable surface (all P8A flags off); R12g base + real_codec_uplift. |

### 4.4 P10 sub-variants (anti-shortcut track)

| Slot | Trainable surface |
|---|---|
| P10_SYM_baseline / P10_SYM_light | **default** (apply_svd_to_in_proj only). The data-side fix (symmetric router) is the entire intervention. |
| P10_GRL_baseline / P10_GRL_strong | **default + GRL quality-domain head** (the head adds ~70k params). Backbone reach unchanged. |
| P10_SYM_GRL | **default + GRL head** (combines symmetric data + GRL embedding regularizer). |
| P10_SYM_on_P8A | **P8A recipe** (proj + ln_post + MLP-SVD trainable) + symmetric router. *This is the only P10 slot that has the wide trainable surface; all others test "can the shortcut be removed at FT time *without* unfreezing the backbone?".* |

**Mechanical implication for the user's hypothesis**: if the shortcut was already baked into RLP6_04 step 23500 (or earlier — into R12g step 14000, or even into CLIP itself), then **P10's default-recipe slots have no mechanical way to remove backbone-level shortcut features**. They can only counter-modulate via the SVD-on-attention residual. This *is* the limitation the user is asking about — and the same limitation that ostensibly drove P8A to unfreeze in the first place.

---

## 5. Major recipe transitions

What changed *outside the parent-checkpoint axis* across the program:

### 5.1 Data composition

| Phase | First introduced | Where |
|---|---|---|
| DF40 + DeepLive | Phase 1 | `combined_paired` since R1 |
| VisoMaster | R2 | added to combined_paired Feb 12 |
| Teams v1 (288 pairs) | R9 (Feb 28) | first Teams data in training mix |
| Teams v2 (1346 pairs, 4.7×) | R10 (Mar 6) | larger Teams corpus |
| Teams OOD eval (300 real + 300 fake holdouts) | R12 (Mar 9) | per `R12_EXPERIMENT_PLAN.md` |
| External AVSpeech reals (`external_youtube_avspeech`) | UNCERTAIN — pre-R13 (referenced from RLP3 onward) | external_real_sources |
| External VCD reals (`external_vcd_real`) | R8 era | training reals in combined_paired |
| **"Proper data" lanes** (`proper_visomaster_*`) | R13 RLP1 (~Apr 19) | Re-curated VisoMaster-Teams pipeline; commit `77facfc Add proper-data runtime` 2026-04-19 |
| `proper_visomaster_enhanced_clean` | R13 RLP6_04 (~Apr 23) | Added explicitly in RLP6_04, weight 2.0 |
| Identity-specific anchor pools (Dor + Roee) | RLP5/RLP6 era | as eval/anchor pools, not training |
| Symmetric routing (data-pipeline change) | R13 P10 (~Apr 26) | added to `data/augmentations/pipelines.py:1421` (`_build_symmetric_quality_pipeline`); routing flag in `pipelines.py:1535` |
| `real_codec_uplift` flag | R13 P9_05 (Apr 25) | commit `ebce585 Add real_codec_uplift flag` |

### 5.2 Augmentation routing

| Phase | Routing mode |
|---|---|
| Phase 1 – R7 | basic aug, no family routing |
| R8–R11 | `vcd_targeted` aug strength, family-aware routing of fakes vs reals |
| R12 | `vcd_targeted` + R12 fixes (CCT, asymmetric brightness ±0.20→[-0.20,+0.60], wider gamma [80,120]→[70,130]) |
| R13 RLP1–RLP9 | `quality_targeted_family` aug version, `routing.mode: "family_aware"` (the asymmetric router) |
| R13 P10 | `routing.mode: "symmetric"` introduced as opt-in |

**The augmentation routing change (family_aware → symmetric) is the load-bearing P10 hypothesis** (per H1 in `R13_P10_SYM_baseline.yaml` header comment): "the family-aware augmentation router installs the quality-shortcut at training time by giving fake samples heavier degradation than real samples."

### 5.3 GRL / Gradient Reversal Layer

| Phase | GRL state |
|---|---|
| R6 (Feb 19) | GRL plumbing added but `quality_domain_loss=0.0` due to a config bug; never trained. |
| R12_C/D/E/F/H (Mar 9) | First *intentional* GRL test. Verdict: λ=0.1 marginal, λ=0.3 actively harmful. "GRL not worth it" per WINNING_RUNS_REGISTRY observation 12. |
| R13 RLP6B (~Apr 23) | First R13-era GRL revisit (8 slots, λ=0.2/0.4 + combos). UNCERTAIN whether any RLP6B slot promoted. |
| R13 P10 (Apr 26) | GRL re-introduced as primary anti-shortcut lever (P10_GRL_baseline λ=0.1, P10_GRL_strong λ=0.25, P10_SYM_GRL stack). |

**The current GRL re-introduction is the second time the program tries this lever in R13** — RLP6B was the first.

### 5.4 Teams data introduction

| Generation | Teams data | Quantity | First used in |
|---|---|---|---|
| Teams v1 | original Teams capture pool | 288 pairs | R9_A (Feb 28) |
| Teams v2 | enlarged Teams capture pool | 1346 pairs | R10 (Mar 6) |
| Teams v2 + OOD holdout | + 300 real + 300 fake held out for eval | + 600 OOD videos | R12 (Mar 9) |
| Teams v2 + proper-data | + ~13 lanes of "proper VisoMaster" Teams-codec-passed and clean-companion variants | varies per inventory | R13 RLP1 (Apr 19) |
| Teams v2 + dor/roee anchors | individual subjects' webcam/laptop captures | tiny (~30 frames per pool, 6+ pools) | RLP5/RLP6 era as eval-only |

### 5.5 Promotion-contract scorecard

- Trainer's `value_composite` metric: introduced ~R12; refined in R13 RLP3 with A1–A10 instrumentation (`value_composite: target_mean_fpr 0.03, max_pool_fpr 0.05, stability_jitter_stat: p95`).
- Lockbox-anchored "promotion contract" scorecard (`arena/score_teams_promotion_contract.py`) — introduced R13 WT-E (~Apr 17): commit `213df81 add authoritative teams promotion contract path`.
- Per memory `project_promotion_contract`: trainer's `value_composite` is **not deployment-grade**; the scorecard is.

### 5.6 Backbone unfreeze

- All R-rounds before P8A: backbone fully frozen except for SVD residuals + ArcFace head.
- **P8A (Apr 24)**: first run to unfreeze `visual.proj` + `visual.ln_post` + add MLP-SVD residuals.
- P9 / P10_SYM_on_P8A continue this; P10_SYM/_GRL revert to frozen-backbone-style.

### 5.7 New code knobs added in R13 P-series

| Knob | Added in | Purpose |
|---|---|---|
| `optimizer.adam.backbone_lr_mult` | P9 (commit `0f2f342` 2026-04-25) | Cap LR on unfrozen-backbone params separately from head. Default 1.0. |
| `augmentation.real_codec_uplift` | P9_05 (commit `ebce585` 2026-04-25) | Apply codec aug to real samples (not just fakes). |
| `augmentation.routing.mode: symmetric` | P10 (commit `2c9778b` 2026-04-26) | Symmetric-quality-pipeline routing (label-blind aug). |
| `use_quality_domain_head` (re-instated) | RLP6B / P10 | GRL on quality-domain (4-class: df40/external_vcd/studio_capture/youtube). |

---

## 6. Missing experiments — the user's gap

### 6.1 "Has there been a clean-CLIP-base retrain in the recent (past 3 months) effort?"

**Yes, three:**

1. **R12_G / `0xxqwhxg` (March 10 2026)** — scratch from CLIP-DataComp-XL with R12 aug fixes, 30k steps, ckpt selected at step 14000. *This is the run that becomes the load-bearing FT base for everything since.* Whether this run **counts as "recent" is subjective**: it is **~47 days old** at the time of writing. It pre-dates the whole "camera-signature shortcut diagnosis" arc that began in RLP5_07 (~Apr 23). So in the user's framing, the clean-CLIP retrain *did* happen, but **not since the shortcut hypothesis was formed**.
2. **R12.5 (R125_A/B/C, ~March 10)** — three scratch-from-CLIP runs with extended lighting envelope. Outcomes are unconsolidated in the WINNING_RUNS_REGISTRY; UNCERTAIN whether these were ever scored against the current evaluation suites. They do *not* feed the R13 chain.
3. **P8B = RLP8_02 (April 24–25)** — scratch from plain CLIP with the same Teams-era data + codec-aggressive aug, 30k step plan, **hung at step 12000 and was cancelled at step 11000**. Anchor Δ +0.066 (worse than RLP6_04). This was the deliberate "is the shortcut in the data mix or in the R13 weight chain?" probe; verdict per memory `project_p8a_breakthrough` was that **the shortcut is in the data mix** (P8B-from-scratch was *worse*, not better, than RLP6_04 on the anchor pool — but P8B was cancelled before completing 30k steps, so this conclusion rests on a partial run).
4. **RLP1_08 SCRATCH** (April 19–20) — scratch hedge with WTB3 hints + full proper-data, 30k steps. UNCERTAIN — outcome not in the consolidated R13_FULL_STORY doc.

**So the gap is more specific than "has there been a clean-CLIP retrain":**

- There has been **no clean-CLIP-base scratch run with the modern (RLP6_04-era) data composition** that has been **fully trained AND fully scored against the lockbox-anchored promotion contract**.
- P8B was the closest attempt; it hung mid-run.
- RLP1_08 SCRATCH was on an older data mix (WTB3 hints + proper-data, but pre-dates the gate-alignment + proper_enh_clean fix from RLP6_04).
- All other R13 P-series experiments inherit ≥37500 cumulative FT steps before they begin.

### 6.2 Other potentially-missing experiments

These would sharpen the "shortcut is baked in early" investigation but I cannot find evidence they were run:

- **A scratch-from-CLIP run with the *exact* RLP6_04 data composition** (R12g aug fixes + gate alignment + `path_exclude_contains: ["/visomaster_"]` + `proper_visomaster_enhanced_clean` lane @ weight 2.0 + the symmetric router from P10). If shortcut-causation is *only* the data mix (per P8B's interpretation), this would be the cleanest test. (UNCERTAIN: this may be subsumed by P10_SYM_baseline which uses `family_aware → symmetric` routing on RLP6_04 base — but `P10_SYM_baseline` is FT-from-RLP6_04, not scratch.)
- **A scratch-from-CLIP with symmetric routing and the modern data mix**. Closest candidate: P8B — but P8B used family_aware aug (not symmetric).
- **A scratch-from-R8_E retrain on the modern data mix**. R8_E was the clean scratch ancestor that several R12_C/D/F runs forked from but never completed properly. None of the R13 chain forks from R8_E — every R13 run forks from R12g or later.
- **A clean rank-axis sweep at `apply_svd_to_in_proj=true` with `rank=512`/`rank=384`/`rank=256`**. Memory `project_p8a_breakthrough` flags rank=736 as "inherited from R12g, not re-validated under current data mix." UNCERTAIN: B16_svd_rank_sweep / B16_capacity_ceiling / B16_lambda_sweep / B16_diagnostic_sweep / arcface_scale_ablation directories exist in `experiments/` but I did not enumerate their configs in this report.

### 6.3 What experiments *did* probe shortcut origin

- **RLP7_08** (codec aug from `RLP6_04 step 4500`) ruled out "the shortcut is consolidated late in RLP6_04" — same anchor ceiling as forking from step 23500.
- **P8A** (unfreeze backbone) ruled out "FT is sufficient to remove the shortcut at default reach" — the unfreeze broke the ceiling (anchor Δ −0.188), confirming the prior was reach-limited not data-limited.
- **P8B** (scratch on plain CLIP + same data) gave evidence — though incomplete (cancelled at step 11000) — that scratch on plain CLIP is *worse* than FT on this data, because the RLP6_04 (and CLIP-prior) is a regularizer that scratch lacks.
- **P9_FORK_*_on_r12g** are testing *partial* shortcut origin: skip RLP6_04 (= skip 23500 of FT) but keep R12g (= keep 14000 of scratch).
- **P10_SYM_baseline / P10_GRL_baseline / P10_SYM_GRL** test whether the shortcut can be unlearned at FT time *without* changing the inherited weights' representational floor.

What's *not* in the experiment record is a run that strips both R12g *and* RLP6_04 priors simultaneously — i.e., scratch on plain CLIP with the modern (RLP6_04) data mix, fully trained.

---

## 7. Caveats / explicit uncertainties

1. **R12_G yaml ↔ W&B id mismatch** (§1.3): registry says `0xxqwhxg` is seed=737; repo yaml `R12_G_scratch_seed_control.yaml` is seed=1337. Most likely the W&B id `0xxqwhxg` was launched by a different (now-overwritten or unkept) yaml. Receiving agent should confirm via the W&B run record.
2. **R12_G chosen step (14000 vs 12500)**: registry says "step 12500 ood_composite" was the leader; R13 yamls all consume `step 14000 top_n` (not ood_composite). So the R13 chain forks from a *different* checkpoint of `0xxqwhxg` than the registry's recorded leader. This is a 1500-step difference — small, but present.
3. **RLP6_04 and 0xxqwhxg checkpoints' actual training-step cumulative count** is reconstructed from yaml `total_training_steps:` and the explicit step numbers in checkpoint filenames; I have not opened the W&B runs themselves. UNCERTAIN whether the chosen checkpoints are early-stopped or ran to schedule.
4. **Outcomes for many RLP1/RLP2/RLP3/RLP3.5/RLP4/RLP5/RLP6/RLP6B/RLP7_01/RLP7_03 slots are not consolidated in this report** — they exist as yamls + W&B runs but I did not pull per-run outcomes for each. The receiving agent can request specific slots.
5. **RLP1_08 SCRATCH outcome**: not surfaced in any of the relaunch handoffs I read. UNCERTAIN whether it failed, was cancelled, or just didn't promote.
6. **R12.5 outcomes**: `WINNING_RUNS_REGISTRY.md` does *not* surface R125_A/B/C results in its leader column; UNCERTAIN whether the 3 runs ever finished or were promoted.
7. **MLP-SVD scope** (which transformer blocks): per `R13_FULL_STORY_PRE_PACKET_9_2026-04-25.md` §2.2 it's "all blocks" but the doc also mentions `apply_svd_to_mlp adds SVD residuals to the MLP c_fc / c_proj weights inside each transformer block (in addition to the default attention in_proj residuals)". I did not grep `effort_detector.py` to confirm this is uniform across all 12 blocks vs restricted to last 2 (per a comment in P8A doc).
8. **Earlier R13 yamls (R13_A, B, C, D, E, F, G, H, I, FT1–FT17, TB1, TB2, WTB*, WTC*)**: these pre-date the relaunch (~Apr 8–17). They live in the same `phase2_round13/` folder as RLP* yamls. I have *not* enumerated their parents/outcomes — they are pre-camera-signature-shortcut-era and out of scope for the user's recent-3-months investigation. Request if needed.
9. **Phase 1 ↔ R1 interleaving**: B16-old (Phase 1, ~Jan 19) and R1 (~Feb 11) are both early-program experiments; the R1 data-passthrough bug means R1's outcomes were invalidated, not that R1 didn't run. Phase 1 is documented separately.
10. **RLP6_04's P95 stability metric, value_composite=0.9006**: claimed in `R13_FULL_STORY_PRE_PACKET_9_2026-04-25.md`. UNCERTAIN whether this was measured under the gate-aligned metric definition (RLP6_04 *is* the gate-alignment canary, so its own value_composite is computed under the new gate); cross-checking against the trainer config would confirm.

---

## 8. One-screen summary

- **CLIP base**: ViT-B/16-DataComp-XL, frozen except for trainable SVD residuals on attention `in_proj` (rank 736) + ArcFace head. Same since R1.
- **R12_G / `0xxqwhxg` step 14000** (Mar 10) — scratch with R12 aug fixes — is the load-bearing FT base for the entire R13 RLP chain.
- **RLP6_04 / `h2pdu6i5` step 23500** (Apr 23) — FT 23.5k from R12g, gate-aligned + proper_enh_clean — is the FT base for all P7/P8/P9/P10 RLP6_04-anchored slots.
- **P8A / `9lmvb5b4` step 5000** (Apr 24) — FT 5k from RLP6_04 with three-flag backbone unfreeze — is the FT base for `P10_SYM_on_P8A` and the comparison anchor for the P9 sweep.
- A finished P10 RLP6_04-anchored slot has **~47500 cumulative FT steps** from CLIP. A finished P10_SYM_on_P8A slot has **~52500**.
- **Frozen-backbone P10 slots** (SYM, GRL, SYM_GRL) **mechanically cannot remove backbone-level features**; they can only counter-modulate via the SVD-on-attention residual.
- **No fully-trained, fully-scored, scratch-from-plain-CLIP run exists with the modern (RLP6_04-era) data mix.** P8B was the closest attempt and it hung at step 11000.
- The user's "shortcut baked in early" hypothesis is structurally consistent with the lineage: every recent experiment inherits ≥37500 cumulative FT steps before its anti-shortcut intervention is applied.
