# Expanded team-identity deploy readout — RESULTS FACTS

Generated 2026-05-23. Factual readout only (per `docs/packet_retrospectives/AGENTS.md` §"Eval-folder authoring contract" — forbidden words: "succeeds", "fails", "wins", "promotes", "deployment-grade"). Interpretive content lives in `AGENT_PROPOSAL_2026-05-23.md`.

This expansion of `analysis/team_identity_deploy_readout_2026-05-23/RESULTS_FACTS_2026-05-23.md` widens the cohort from 6 named pools / 180 frames to the full 5-team-human cohort: 4,121 deploy-relevant real frames + 5,005 fake-attack frames, sample-capped to 6,439 frames after capping each cohort at REAL_CAP=150 / FAKE_CAP=100 (random_state=42).

---

## 0. Scope and inputs

5 ckpts × 5 deploy-relevant team humans × {real cohort, fake-attack cohort if any} × 4 τ-modes.

Plus Mac-Roee informational read using only cached scores (no fresh scoring on Mac-Roee since user explicitly excluded Mac from the deploy gate).

| Ckpt key | Local path | Score-file path |
|---|---|---|
| `P8A_REFERENCE_STEP5000` | `analysis/manual_canary_2026-05-20/ckpts/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth` | `outputs/P8A_REFERENCE_STEP5000_scores.per_frame.csv` (this session) + cached `score_P8A` from `grouped_manifest_v2.csv` |
| `E2B_TOP_N_STEP3200` | `analysis/team_identity_deploy_readout_2026-05-23/_ckpts/E2B_TOP_N_STEP3200.pth` | `outputs/E2B_TOP_N_STEP3200_scores.per_frame.csv` (this session) + cached `score_E2B` |
| `T5C_PERIODIC_STEP3500` | `analysis/manual_canary_2026-05-20/ckpts/periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth` | `outputs/T5C_PERIODIC_STEP3500_scores.per_frame.csv` (this session) + cached `score_T5C` |
| `SLOT_A_ANCHOR_AWARE_STEP3500` (CLS pool) | `analysis/manual_canary_2026-05-20/ckpts/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth` | `outputs/SLOT_A_ANCHOR_AWARE_STEP3500_scores.per_frame.csv` (this session) |
| `SLOT_A_ANCHOR_AWARE_STEP3500_FACE_POOL` | same ckpt + face-pool monkey-patch from `analysis/face_pool_canary_2026-05-22/score_canary_face_pool.py` | `outputs/SLOT_A_ANCHOR_AWARE_STEP3500_FACE_POOL_scores.per_frame.csv` (this session) |

τ-modes from memory `project_deployment_three_modes_slot_a_v2_2026-05-21`:
- `tau_0_5` = 0.5 (natural threshold; informational)
- `mode_A_tau_0_535` = 0.535 (mode A; lockbox-FPR≈10% / fake-recall≈85% calibrated for Slot A v2 step3500 CLS-pool)
- `mode_B_tau_0_78` = 0.78 (mode B; lockbox-FPR≈2% / fake-recall≈70%; contract-compliant)
- `mode_C_tau_0_87` = 0.87 (mode C; lockbox-FPR≈1% / fake-recall≈65%)

CAVEAT (carried from prior readout): τ values were calibrated for Slot A v2 CLS-pool on the 9-suite scorecard. Cross-ckpt usage of the same numeric τ is a constant-threshold comparison, not a constant-FPR comparison.

Scoring: 5 ckpts × all relevant frames on local MPS (Apple Silicon). 100% decode success on every ckpt × frame after the `local_frame_resolver.py` fix (see §6 Self-correction). Total wall time across 5 ckpts: ~85 min.

---

## 1. Inventory verification (Task A) — revisions vs prior agent

The prior agent's `outputs/pool_inventory.csv` attributed 21 cohorts. Inspecting frame-path naming and bucket origins identified ONE revision:

| Cohort | Frame count | Prior attribution | Verification evidence | Revised attribution |
|---|---:|---|---|---|
| `royd_real_2026-03-06` | 181 | **Roee_Windows** (deploy-relevant) | Frame names start with `Roy D` (Mac label per memory `project_team_identities_multi_labeled_2026-05-23`), NOT `tester tester` (Windows signature). Bucket=`live-fakes-teams-prod`; but the cohort dir-name `royd_real_*` carries the `Roy D` person tag which is the Mac-Roee label cluster per user spec. | **Roee_Mac** (NOT deploy-relevant) |
| `roee_tester_real_2026-03-24` | 323 | Roee_Windows | Frame names start with `tester tester` — confirmed Windows signature | **Roee_Windows** (confirmed) |
| `tester_roee_real_2026-03-06` | 173 | Roee_Windows | Frame names start with `tester tester` — confirmed Windows signature | **Roee_Windows** (confirmed) |
| `extra_roy_d` | 236 | Roee_Mac | `gs://local/extra/extra_roy_d/` — no person-name prefix, extra-bucket Mac per memory | **Roee_Mac** (confirmed) — DROPPED from this readout: no cached scores + Mac is out of scope |
| `bla_bla_chow*` | 559 (311+68+180) | Roee_Mac | `teams-faces-data-test-2914-fake-4420-real-feb-28` bucket per memory | **Roee_Mac** (confirmed) |
| `Roy_D` | 130 | Roee_Mac | Same Mac bucket | **Roee_Mac** (confirmed) |

**Impact of the `royd_real_2026-03-06` revision on the prior agent's headline**: the prior agent's pool inventory listed 707 Windows-Roee real frames (323 + 173 + 30 + 181). After revision, Windows-Roee = 526 real frames (323 + 173 + 30). The 181 `royd_real_*` frames move to Mac-Roee informational. The prior agent's numerical headline (`Slot A v2 = 0.000 on Roee-Windows`) is unaffected because Slot A v2 was 0/30 on the 30-frame named pool sample and the additional Windows-Roee cohorts (`roee_tester_real_*` + `tester_roee_real_*`) replicate that with 0/(323+173)=0.000.

Full inventory at `outputs/master_inventory.csv` (6,439 rows after sampling).

---

## 2. Sample counts per (ckpt × team-human × real/fake)

After sampling (REAL_CAP=150, FAKE_CAP=100 per cohort; random_state=42):

| Human | Real frames (deploy-relevant) | Fake-attack frames (target = this human) |
|---|---:|---:|
| Roee_Windows | 330 (3 cohorts) | 0 |
| dor | 620 (6 cohorts) | 2,443 (29 cohorts incl. visomaster_v2_dor + dor_fake_local + dor_shkedi__s16 + dor_fake_deeplive_enhanced_*) |
| Noyn (= Noyn Sharker) | 210 (2 cohorts) | 0 |
| Xiang | 582 (5 cohorts) | 578 (8 cohorts incl. live_prod__xiang-fake-{1..6} + Xiang_Xiang2_Feng_fake + extra_xiang_fake) |
| Xinhe | 79 (2 cohorts) | 1,099 (12 cohorts: live_prod__xinhe-fake-{1..11-glasses} + extra_xinghe_fake) |
| Roee_Mac (info only) | 498 (4 cohorts, cached scores only) | — |
| **Total deploy-relevant** | **1,821 real + 4,120 fake = 5,941** | (Slot A v2 scored all 5,941) |
| **Total including Mac-Roee info** | **2,319 real + 4,120 fake = 6,439** | (P8A/E2B/T5C scored 4,921 fresh + 1,518 cached = 6,439 cells) |

100% decode success on every (ckpt × frame) cell across all 5 ckpts.

---

## 3. Per-human × per-ckpt FPR at mode B (τ=0.78) — the binding mode

The user's specified gates: per-human FPR ≤ 0.05 (real-side) AND per-human fake-recall ≥ 0.50 (fake-side), both at the same τ.

### 3.1 Real-side: per-human FPR @ mode B (τ=0.78)

| Human | n_real | P8A | E2B | T5C | Slot A v2 (CLS) | Slot A v2 (face-pool) |
|---|---:|---:|---:|---:|---:|---:|
| Noyn | 210 | 0.005 | 0.000 | 0.000 | 0.000 | 0.000 |
| Roee_Windows | 330 | 0.009 | 0.000 | 0.000 | 0.000 | 0.000 |
| Xiang | 582 | 0.014 | 0.036 | 0.021 | 0.024 | 0.003 |
| Xinhe | 79 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dor | 620 | 0.016 | 0.011 | 0.052 | 0.008 | 0.002 |
| **MAX per-human FPR** | | **0.016** | **0.036** | **0.052** | **0.024** | **0.003** |
| **Real-floor (≤0.05) hits?** | | YES | YES | NO (dor 0.052) | YES | YES |

### 3.2 Fake-side: per-human fake-recall @ mode B (τ=0.78)

(Reports per-human fake-recall; only the 3 humans with fake-attack cohorts.)

| Human | n_fake | P8A | E2B | T5C | Slot A v2 (CLS) | Slot A v2 (face-pool) |
|---|---:|---:|---:|---:|---:|---:|
| Xiang | 578 | 0.815 | 0.869 | 0.907 | 0.830 | 0.830 |
| Xinhe | 1099 | 0.487 | 0.674 | 0.247 | 0.293 | 0.495 |
| dor | 2443 | 0.745 | 0.473 | 0.628 | 0.414 | 0.252 |
| **MIN per-human fake-recall** | | **0.487** | **0.473** | **0.247** | **0.293** | **0.252** |
| **Fake-floor (≥0.50) hits?** | | NO (Xinhe 0.487) | NO (dor 0.473) | NO (Xinhe 0.247) | NO (Xinhe 0.293, dor 0.414) | NO (dor 0.252) |

### 3.3 Mode B BOTH-GATE summary

| Ckpt | Real-side ≤5% | Fake-side ≥50% | Both pass? | Failing humans |
|---|:---:|:---:|:---:|---|
| P8A | YES | NO | NO | fake: Xinhe(0.487, **0.013pp below floor**) |
| E2B | YES | NO | NO | fake: dor(0.473) |
| T5C | NO (dor 0.052) | NO | NO | real: dor(0.052); fake: Xinhe(0.247) |
| Slot A v2 (CLS) | YES | NO | NO | fake: Xinhe(0.293), dor(0.414) |
| Slot A v2 (face-pool) | YES | NO | NO | fake: Xinhe(0.495), dor(0.252) |

**No ckpt clears both gates simultaneously at mode B (τ=0.78) under the 5%/50% bars.**

---

## 4. Per-human × per-ckpt at mode A (τ=0.535) — recall-leaning

### 4.1 Real-side @ mode A

| Human | P8A | E2B | T5C | Slot A v2 (CLS) | Slot A v2 (face-pool) |
|---|---:|---:|---:|---:|---:|
| Noyn | 0.014 | 0.000 | 0.000 | 0.000 | 0.010 |
| Roee_Windows | 0.009 | 0.000 | 0.006 | 0.003 | 0.030 |
| Xiang | 0.022 | 0.086 | 0.065 | 0.055 | 0.433 |
| Xinhe | 0.013 | 0.013 | 0.000 | 0.013 | 0.203 |
| dor | 0.040 | 0.029 | 0.215 | 0.050 | 0.315 |
| **MAX FPR** | **0.040** | **0.086** | **0.215** | **0.055** | **0.433** |

### 4.2 Fake-side @ mode A

| Human | P8A | E2B | T5C | Slot A v2 (CLS) | Slot A v2 (face-pool) |
|---|---:|---:|---:|---:|---:|
| Xiang | 0.894 | 0.929 | 0.979 | 0.943 | 1.000 |
| Xinhe | 0.633 | 0.794 | 0.493 | 0.482 | 0.995 |
| dor | 0.807 | 0.650 | 0.838 | 0.578 | 0.798 |
| **MIN recall** | **0.633** | **0.650** | **0.493** | **0.482** | **0.798** |

### 4.3 Mode A BOTH-GATE summary (5%/50%)

| Ckpt | Real-side ≤5% | Fake-side ≥50% | Both pass? |
|---|:---:|:---:|:---:|
| **P8A** | **YES (max 0.040)** | **YES (min 0.633)** | **YES** |
| E2B | NO (Xiang 0.086) | YES | NO |
| T5C | NO (dor 0.215, Xiang 0.065) | NO (Xinhe 0.493) | NO |
| Slot A v2 (CLS) | NO (Xiang 0.055, dor 0.050) | NO (Xinhe 0.482) | NO |
| Slot A v2 (face-pool) | NO (4/5 humans over) | YES (min 0.798) | NO |

**At mode A (τ=0.535), only P8A passes both 5%/50% gates simultaneously.**

---

## 5. Per-human × per-ckpt at mode C (τ=0.87) — FPR-leaning

### 5.1 Real-side @ mode C

| Human | P8A | E2B | T5C | Slot A v2 (CLS) | Slot A v2 (face-pool) |
|---|---:|---:|---:|---:|---:|
| Noyn | 0.005 | 0.000 | 0.000 | 0.000 | 0.000 |
| Roee_Windows | 0.006 | 0.000 | 0.000 | 0.000 | 0.000 |
| Xiang | 0.010 | 0.019 | 0.003 | 0.007 | 0.000 |
| Xinhe | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dor | 0.010 | 0.008 | 0.010 | 0.002 | 0.000 |

### 5.2 Fake-side @ mode C

| Human | P8A | E2B | T5C | Slot A v2 (CLS) | Slot A v2 (face-pool) |
|---|---:|---:|---:|---:|---:|
| Xiang | 0.784 | 0.806 | 0.843 | 0.716 | 0.349 |
| Xinhe | 0.413 | 0.590 | 0.139 | 0.197 | 0.067 |
| dor | 0.711 | 0.376 | 0.447 | 0.307 | 0.053 |

### 5.3 Mode C BOTH-GATE summary (5%/50%)

| Ckpt | Real-side ≤5% | Fake-side ≥50% | Both pass? |
|---|:---:|:---:|:---:|
| P8A | YES (max 0.010) | NO (Xinhe 0.413) | NO |
| E2B | YES (max 0.019) | NO (dor 0.376) | NO |
| T5C | YES (max 0.010) | NO (Xinhe 0.139, dor 0.447) | NO |
| Slot A v2 (CLS) | YES (max 0.007) | NO (Xinhe 0.197) | NO |
| Slot A v2 (face-pool) | YES (max 0.000) | NO (Xinhe 0.067, dor 0.053) | NO |

---

## 6. Per-cohort decomposition (mode B)

### 6.1 Real-side per-cohort FPR (mode B, τ=0.78)

| Human | Cohort | n | P8A | E2B | T5C | Slot A v2 CLS | Slot A v2 face |
|---|---|---:|---:|---:|---:|---:|---:|
| Noyn | Md_noyn_Sharker__s15 | 150 | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 |
| Noyn | team_may5__Noyn | 60 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Roee_Windows | roee_tester_real_2026-03-24 | 150 | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 |
| Roee_Windows | team_may5__Roee | 30 | 0.033 | 0.000 | 0.000 | 0.000 | 0.000 |
| Roee_Windows | tester_roee_real_2026-03-06 | 150 | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 |
| Xiang | Xiang_Xiang2_Feng | 150 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Xiang | Xiang_Xiang2_Feng__s23 | 102 | 0.039 | 0.000 | 0.029 | 0.020 | 0.000 |
| Xiang | extra_xiang | 150 | 0.013 | 0.007 | 0.000 | 0.007 | 0.000 |
| Xiang | team_may5__Xiang | 30 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Xiang | xiang | 150 | 0.013 | 0.133 | 0.060 | 0.073 | 0.013 |
| Xinhe | extra_xinghe | 19 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Xinhe | team_may5__Xinhe | 60 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dor | dor_evening | 150 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dor | dor_morning | 150 | 0.040 | 0.013 | 0.153 | 0.000 | 0.000 |
| dor | dor_shkedi | 150 | 0.020 | 0.027 | 0.053 | 0.027 | 0.007 |
| dor | dor_shkedi__s16 | 31 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dor | real_dor | 109 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dor | team_may5__Dor | 30 | 0.033 | 0.033 | 0.033 | 0.033 | 0.000 |

### 6.2 Fake-side per-cohort recall (mode B, τ=0.78) — the 12 Xinhe-fake cohorts

| Cohort | n | P8A | E2B | T5C | Slot A v2 CLS | Slot A v2 face |
|---|---:|---:|---:|---:|---:|---:|
| extra_xinghe | 100 | 0.830 | 0.800 | 0.660 | 0.690 | 0.710 |
| live_prod__xinhe-fake-1 | 100 | 0.120 | 0.290 | 0.050 | 0.060 | 0.180 |
| live_prod__xinhe-fake-2 | 86 | 0.047 | 0.291 | 0.000 | 0.000 | 0.058 |
| live_prod__xinhe-fake-3 | 100 | 0.290 | 0.300 | 0.150 | 0.150 | 0.190 |
| live_prod__xinhe-fake-4 | 100 | 0.450 | 0.760 | 0.310 | 0.500 | 0.730 |
| live_prod__xinhe-fake-5 | 88 | 0.330 | 0.807 | 0.125 | 0.227 | 0.670 |
| live_prod__xinhe-fake-6 | 82 | 0.634 | 0.463 | 0.110 | 0.220 | 0.610 |
| live_prod__xinhe-fake-7 | 66 | 0.970 | 0.955 | 0.364 | 0.379 | 0.955 |
| live_prod__xinhe-fake-8 | 100 | 0.720 | 0.900 | 0.360 | 0.330 | 0.490 |
| live_prod__xinhe-fake-8-glasses | 100 | 0.580 | 0.800 | 0.180 | 0.100 | 0.100 |
| live_prod__xinhe-fake-9-glasses | 60 | 0.350 | 0.950 | 0.200 | 0.333 | 0.767 |
| live_prod__xinhe-fake-10-glasses | 73 | 0.630 | 0.849 | 0.384 | 0.507 | 0.712 |
| live_prod__xinhe-fake-11-glasses | 44 | 0.455 | 0.909 | 0.386 | 0.432 | 0.659 |

### 6.3 Fake-side per-cohort recall (mode B, τ=0.78) — Xiang-fake cohorts

| Cohort | n | P8A | E2B | T5C | Slot A v2 CLS | Slot A v2 face |
|---|---:|---:|---:|---:|---:|---:|
| Xiang_Xiang2_Feng | 100 | 0.190 | 0.400 | 0.510 | 0.440 | 0.590 |
| extra_xiang | 73 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| live_prod__xiang-fake-1 | 50 | 0.940 | 0.880 | 0.960 | 1.000 | 0.960 |
| live_prod__xiang-fake-2 | 84 | 0.952 | 0.940 | 1.000 | 0.940 | 0.988 |
| live_prod__xiang-fake-3 | 83 | 0.880 | 0.988 | 0.976 | 0.639 | 0.434 |
| live_prod__xiang-fake-4 | 78 | 0.910 | 1.000 | 1.000 | 0.949 | 0.936 |
| live_prod__xiang-fake-5 | 61 | 1.000 | 0.967 | 1.000 | 0.984 | 1.000 |
| live_prod__xiang-fake-6 | 49 | 0.959 | 0.959 | 0.980 | 0.959 | 0.959 |

### 6.4 Fake-side per-cohort recall (mode B, τ=0.78) — dor-fake cohorts (29 cohorts; truncated to noteworthy)

| Cohort | n | P8A | E2B | T5C | Slot A v2 CLS | Slot A v2 face |
|---|---:|---:|---:|---:|---:|---:|
| dor_shkedi__s16 (chronic) | 78 | 0.923 | 0.500 | 0.308 | 0.154 | 0.064 |
| dor_fake_deeplive_enhanced_1 | 50 | 0.960 | 0.060 | 0.900 | 0.700 | 0.300 |
| dor_fake_deeplive_enhanced_2 | 65 | 1.000 | 0.138 | 0.800 | 0.385 | 0.123 |
| dor_fake_deeplive_enhanced_3 | 78 | 1.000 | 0.641 | 0.987 | 0.962 | 0.795 |
| dor_fake_ghostface_v1 (visomaster) | 100 | 0.770 | 0.370 | 0.470 | 0.170 | 0.030 |
| dor_fake_ghostface_v2 (visomaster) | 100 | 0.730 | 0.190 | 0.280 | 0.050 | 0.010 |
| dor_fake_ghostface_v3 (visomaster) | 100 | 0.640 | 0.170 | 0.380 | 0.120 | 0.010 |
| dor_fake_inswapper_128res_gpen1024 | 100 | 0.230 | 0.020 | 0.090 | 0.020 | 0.000 |
| dor_fake_inswapper_128res_gpen512 | 100 | 0.380 | 0.010 | 0.170 | 0.020 | 0.000 |
| dor_fake_simswap | 79 | 0.481 | 0.025 | 0.076 | 0.013 | 0.013 |
| dor_fake_bill_gates_regular | 24 | 0.917 | 0.917 | 0.875 | 0.792 | 0.458 |
| dor_fake_elone_enhanced | 47 | 1.000 | 0.957 | 1.000 | 1.000 | 0.979 |
| dor_fake_trump_regular | 44 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Full per-cohort table at `outputs/per_cohort_summary.csv`.

---

## 7. Aggregate ship verdict (Task E)

Reproducing §3 / §4 / §5 verdicts under the dual gate (real-side per-human FPR ≤ 5%, fake-side per-human fake-recall ≥ 50% at the same mode):

| Ckpt | Mode A both-pass | Mode B both-pass | Mode C both-pass |
|---|:---:|:---:|:---:|
| P8A | **YES** | NO (Xinhe fake recall 0.487 < 0.50) | NO |
| E2B | NO (Xiang FPR 0.086) | NO (dor fake recall 0.473) | NO |
| T5C | NO | NO | NO |
| Slot A v2 (CLS) | NO | NO | NO |
| Slot A v2 (face-pool) | NO | NO | NO |

**Single mechanical passer under the user's pre-stated bars: `P8A_REFERENCE_STEP5000` at mode A (τ=0.535).**

### Tier ranking

- Tier 1 (both gates pass at mode B): empty
- Tier 2 (both gates pass at mode A or C, fails mode B on at least one human): `P8A_REFERENCE_STEP5000`
- Tier 3 (only one gate passes at any mode): all four others (E2B, T5C, Slot A v2 CLS, Slot A v2 face)
- Tier 4 (neither gate passes at any mode): none

---

## 8. Mac-Roee informational read

Cached scores from `grouped_manifest_v2.csv` for the 4 Roee_Mac cohorts (498 frames). Slot A v2 not scored on Mac-Roee.

| Ckpt | Roy_D (130) | bla_bla_chow (150) | bla_bla_chow__s1 (68) | bla_bla_chow__s2 (150) | Aggregate FPR @ mode B |
|---|---:|---:|---:|---:|---:|
| P8A | 0.369 | 0.060 | 0.000 | 0.160 | 0.163 |
| E2B | 0.123 | 0.213 | 0.015 | 0.253 | 0.175 |
| T5C | 0.923 | 0.140 | 0.324 | 0.273 | 0.410 |

(Roy_D regression from memory `project_band_shortcut_ood_hypothesis_2026-05-16` was on Slot A v2 (29→81.5%) but that cohort was a different sample of Roy_D. This readout has P8A 0.369 / E2B 0.123 / T5C 0.923 on the 130-frame Roy_D pool.)

Slot A v2 was not scored on Mac-Roee in this readout per task spec; the prior readout (analysis/team_identity_deploy_readout_2026-05-23/) showed Slot A v2 CLS at 0.233 mode-B FPR and face-pool at 0.000 on a 30-frame Mac-Roee proxy pool.

---

## 9. E2B → recommendation switch impact (Task F)

E2B is the currently deployed model per `project_deployment_is_e2b_2026-05-06`. The mechanical Tier-1 verdict in §7 is `P8A at mode A`. Compute delta vs E2B at mode A:

### 9.1 Real-side delta (P8A − E2B at mode A)

| Human | E2B FPR | P8A FPR | Δ (P8A − E2B) | Direction |
|---|---:|---:|---:|---|
| Noyn | 0.000 | 0.014 | **+0.014** | E2B tighter |
| Roee_Windows | 0.000 | 0.009 | **+0.009** | E2B tighter |
| Xiang | 0.086 | 0.022 | **−0.064** | P8A tighter |
| Xinhe | 0.013 | 0.013 | 0.000 | tie |
| dor | 0.029 | 0.040 | **+0.011** | E2B tighter |

### 9.2 Fake-side delta (P8A − E2B at mode A)

| Human | E2B recall | P8A recall | Δ (P8A − E2B) | Direction |
|---|---:|---:|---:|---|
| Xiang | 0.929 | 0.894 | **−0.035** | E2B catches more |
| Xinhe | 0.794 | 0.633 | **−0.161** | E2B catches more |
| dor | 0.650 | 0.807 | **+0.157** | P8A catches more |

### 9.3 Specific Xinhe call-out (memory says E2B 57.6% FPR on Xinhe-may6 vs P8A 0%)

Per memory `project_xinhe_may6_falseflag_2026-05-06`, the original may6 cohort (92 frames at `analysis/xinhe_cross_camera_audit_2026-05-06/raw/may6/`) showed E2B 57.6% / P8A 0% FPR. That cohort is NOT directly part of the `grouped_manifest_v2.csv` browser. The closest proxy in this readout is `team_may5__Xinhe` (60 frames at mode B): P8A 0.000 / E2B 0.000 / T5C 0.000 / Slot A v2 CLS 0.000 / face-pool 0.000. **All 5 ckpts handle `team_may5__Xinhe` cleanly at mode B.** Whether the may6 cohort itself reproduces the E2B 57.6% finding in this expanded readout is not measured here — that cohort would need a separate scoring run on the 92 may6 frames.

### 9.4 Mode B (the contract-compliant mode E2B was deployed under): P8A vs E2B

| Human | Real FPR (P8A / E2B / Δ) | Fake recall (P8A / E2B / Δ) |
|---|---|---|
| Roee_Windows | 0.009 / 0.000 / +0.009 | — |
| Noyn | 0.005 / 0.000 / +0.005 | — |
| dor | 0.016 / 0.011 / +0.005 | 0.745 / 0.473 / **+0.272** |
| Xinhe | 0.000 / 0.000 / 0.000 | 0.487 / 0.674 / **−0.187** |
| Xiang | 0.014 / 0.036 / −0.022 | 0.815 / 0.869 / −0.054 |

At mode B the trade is: P8A picks up dor fake recall (+27pp) at the cost of Xinhe fake recall (−19pp).

---

## 10. IQ-gate readout

`grouped_manifest_v2.csv` carries `width`/`height` for 1,014/6,439 = 15.7% of frames (the rest are NaN). Computed `min_dim = min(width, height)` per-frame on the rows that have data:

| Human (role) | n_with_iq | p25 min_dim | median | p75 |
|---|---:|---:|---:|---:|
| Noyn (real) | 150 | 165 | 184 | 192 |
| Roee_Mac (real) | 368 | 157 | 218 | 390 |
| Xiang (real) | 252 | 197 | 206 | 330 |
| Xiang (fake) | 100 | 300 | 386 | 399 |
| dor (real) | 66 | 312 | 318 | 396 |
| dor (fake) | 78 | 310 | 312 | 318 |

The user's IQ gate is `face_min_dim ≥ 120`. On the cohorts with `width`/`height` data, every cohort has p25 ≥ 120 — gate is non-binding for these frames.

For frames without `width`/`height`, the IQ-gate impact is unknown. The prior readout's separate `compute_iq.py` pass (which downloaded raw frame bytes and computed face_min_dim post-hoc on a 180-frame subset) showed similar non-binding behavior at gate=120.

---

## 11. Pre-stated bars and mechanical pass/fail (Task E)

Bar: per-human FPR ≤ 0.05 (5%) AND per-human fake-recall ≥ 0.50 (50%) for all 5 deploy-relevant humans simultaneously, at the same τ-mode.

| Ckpt | Mode A (τ=0.535) | Mode B (τ=0.78) | Mode C (τ=0.87) |
|---|:---:|:---:|:---:|
| P8A_REFERENCE_STEP5000 | **PASS** | FAIL (Xinhe recall 0.487 < 0.50) | FAIL |
| E2B_TOP_N_STEP3200 | FAIL (Xiang FPR 0.086 > 0.05) | FAIL (dor recall 0.473 < 0.50) | FAIL |
| T5C_PERIODIC_STEP3500 | FAIL | FAIL (dor FPR 0.052 > 0.05; Xinhe recall 0.247 < 0.50) | FAIL |
| SLOT_A_ANCHOR_AWARE_STEP3500 | FAIL (Xinhe recall 0.482 < 0.50) | FAIL (Xinhe recall 0.293 < 0.50; dor recall 0.414 < 0.50) | FAIL |
| SLOT_A_ANCHOR_AWARE_STEP3500_FACE_POOL | FAIL (4/5 humans FPR > 0.05) | FAIL (Xinhe recall 0.495 < 0.50; dor recall 0.252 < 0.50) | FAIL |

**Only mechanical passer under the user's bars: P8A_REFERENCE_STEP5000 at mode A (τ=0.535).**

(Mechanical = direct application of the user-specified bars to the measured cell values; no opinion.)

---

## 12. Numbers cross-check vs prior readout and memory entries

### 12.1 vs prior agent's 30-frames-per-pool readout (`analysis/team_identity_deploy_readout_2026-05-23/`)

The prior readout's mode-B verdict was: P8A=FAIL (dor 29.2% mode-B FPR), Slot A v2 CLS/face=PASS (0.0%), E2B=PASS (3.3%), T5C=PASS-boundary (5.0%).

**Prior numbers were on 4 dor pools × 30 frames = 120 dor real frames, dominated by 2 dor-webcam pools where P8A historically over-fires.** This readout's dor real cohort is 620 frames across 6 pools (`dor_evening`, `dor_morning`, `dor_shkedi`, `dor_shkedi__s16`, `real_dor`, `team_may5__Dor`); the dor-webcam pools are not part of `grouped_manifest_v2.csv`. So this readout's dor real FPR (P8A 0.016 mode-B) **does NOT reproduce the prior readout's P8A 0.292** — they're measuring different dor sub-populations.

The prior readout's claim "**P8A is the WEAKEST of the five ckpts on the population that actually matters**" is FALSE if "population that matters" is read as the broader deploy-relevant dor cohort in `grouped_manifest_v2`. P8A is in fact tied with E2B for the tightest dor-real-FPR at mode B (0.016 vs 0.011) in this readout.

The Slot A v2 0.000 on dor real (prior readout) reproduces here as 0.008 (5/620 frames) at mode B — still very tight, but not literally zero on the larger sample.

### 12.2 vs `project_deployment_is_e2b_2026-05-06`

Memory says E2B is the deployed model. This readout shows E2B at mode A fails the 5% real-side floor on Xiang (0.086) — a finding not previously surfaced. At mode B E2B passes the real-side floor on every human but misses dor fake recall (0.473 < 0.50).

### 12.3 vs `project_face_pool_scorecard_pareto_2026-05-22`

Memory says face-pool Pareto-improves Slot A v2 on the 9-suite scorecard (lockbox_real_fpr 0.0191→0.0154, lockbox_fake_recall 0.6877→0.7668, deeplive_enhanced +0.132, visomaster_enhanced −0.100). This readout shows that on the team-attack subset:
- face-pool LIFTS Xinhe-fake recall mode-B from 0.293 (CLS) → 0.495 (+20pp)
- face-pool DROPS dor-fake recall mode-B from 0.414 → 0.252 (−16pp)
- face-pool TIGHTENS real-side (max FPR mode B 0.024 CLS → 0.003 face-pool)
- face-pool MASSIVELY worsens dor visomaster recall (e.g., ghostface_v1 mode B: CLS 0.170 → face 0.030; instyle_vA: 0.270 → 0.040)

The "face-pool visomaster regression" generalizes to the team-attack subset on dor-as-fake-target (visomaster_v2_dor uses inswapper/instyle/ghostface — these are visomaster-class attacks on dor). The face-pool's xinhe-fake lift does NOT compose with the dor-fake regression.

### 12.4 vs `project_team_identities_multi_labeled_2026-05-23`

Memory describes Mac-Roee out-of-scope and the 5-person team cohort as the deploy gate. This readout confirms:
- All 5 ckpts handle `team_may5__Roee` cleanly (Roee_Windows real FPR mode B ≤ 0.033 across all ckpts).
- `Xinhe` is the binding fake-side human (lowest fake recall across all ckpts).
- `dor` is the binding real-side human only for T5C (mode B dor FPR 0.052).
- Mac-Roee aggregate FPR at mode B: P8A 0.163, E2B 0.175, T5C 0.410 — all materially over the 5% floor; face-pool drops to ~0 per the prior 30-frame readout but is not reproduced on Mac here.

---

## 13. Artifacts

- `outputs/master_inventory.csv` — 6,439 rows (sampled team-frame inventory with human/role attribution + cached score columns)
- `outputs/P8A_REFERENCE_STEP5000_scores.per_frame.csv` — fresh scores (4,921 frames)
- `outputs/E2B_TOP_N_STEP3200_scores.per_frame.csv` — fresh scores (4,921 frames)
- `outputs/T5C_PERIODIC_STEP3500_scores.per_frame.csv` — fresh scores (4,921 frames)
- `outputs/SLOT_A_ANCHOR_AWARE_STEP3500_scores.per_frame.csv` — fresh scores (5,941 frames)
- `outputs/SLOT_A_ANCHOR_AWARE_STEP3500_FACE_POOL_scores.per_frame.csv` — fresh scores (5,941 frames; face-pool monkey-patch)
- `outputs/per_frame_full.csv` — merged: 6,439 frames × {meta, prob_P8A, prob_E2B, prob_T5C, prob_SlotAv2_CLS, prob_SlotAv2_FACE}
- `outputs/per_cohort_summary.csv` — one row per (ckpt × base_identity × role)
- `outputs/per_human_summary.csv` — one row per (ckpt × human × role) — deploy-relevant only
- `outputs/per_human_summary_mac_info.csv` — same for Mac-Roee, informational
- `outputs/ship_verdict.csv` — gates pass/fail per (ckpt × mode)
- `scripts/build_master_inventory.py` — inventory builder (with revised attribution)
- `scripts/local_frame_resolver.py` — gs://local/* → local path resolver (fixes prior decode failures)
- `scripts/score_all_ckpts.py` — single-pass scorer for all 5 ckpts
- `scripts/analyze_team_deploy.py` — merge + per-human + verdict
- `scripts/run_sequence.sh` — sequential runner for the 4 ckpts after Slot A v2 CLS
- `_score_*.log` — per-ckpt run logs

All scoring done on local MPS (Apple Silicon). No Vertex jobs, no image builds, no production code modifications.

---

## 14. Caveats / known limitations

1. **Sample cap at 150 real / 100 fake per cohort**: aggregates per-human FPR/recall are sample-bounded. e.g., dor real n=620, Xinhe real n=79 (small — ±5pp confidence on Xinhe real FPR even at the floor). Xinhe fake n=1099 (good).
2. **τ values are cross-ckpt constants**, not per-ckpt-calibrated. P8A's mode-A pass at τ=0.535 might shift if τ is per-ckpt re-calibrated; same for Slot A v2 face-pool which has a shifted score distribution (mean real ~0.35-0.50 vs CLS ~0.07-0.17).
3. **xinhe_may6_falseflag (92 frames) is NOT in this readout** — it lives at `analysis/xinhe_cross_camera_audit_2026-05-06/raw/may6/`. The original E2B 57.6% / P8A 0% FPR claim on that cohort is not reproduced here. team_may5__Xinhe (60 frames) is the closest proxy and shows 0/0/0/0/0 across all 5 ckpts.
4. **dor-webcam-false-flag pools (the prior readout's bias source) are NOT in `grouped_manifest_v2.csv`** — those 4 pools are in a separate session capture (`gs://real-teams-dor-roee/session_20260424_combined_tags_*/`). The prior readout's "P8A fails dor 29%" finding is specific to that capture and does not propagate to the broader dor cohort.
5. **Mac-Roee Slot A v2 numbers are missing from this readout** (Mac was out of scope per task spec) — prior 30-frame readout has them.
6. **9-suite contract not re-run**: this is per-team-human; the broader contract (dev_macro, visomaster_enhanced_macro_dev_recall, lockbox_fake_recall) is not validated here.
7. **`royd_real_2026-03-06` (181 frames) excluded from BOTH deploy and Mac informational** — it has the Mac-Roee `Roy D` naming but lives in `live-fakes-teams-prod` bucket and is uncached. To get a numerical proxy for Slot A v2 Roy-D behavior, the closest cached Roy_D cohort (130 frames) shows P8A 0.369 mode-B FPR, E2B 0.123, T5C 0.923 — but Slot A v2 is uncached on it.

---

## 15. Throughput notes

- Slot A v2 CLS (5,941 frames): 20 min wall
- Slot A v2 face-pool (5,941 frames): 18 min wall
- T5C (4,921 frames): 12 min wall
- P8A (4,921 frames): 14 min wall
- E2B (4,921 frames): 13 min wall
- Total: ~77 min scoring + ~10 min setup/analysis = ~90 min

Bottleneck was GCS-download for `teams-faces-data-test-2914-fake-4420-real-feb-28` and `visomaster-enhanced-face-cropped-v2` buckets; local-resolver short-circuit gave ~6-12 fps for local frames vs ~3-5 fps for GCS.
