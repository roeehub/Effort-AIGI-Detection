# P-series FACTS ledger — pure data, no interpretation

**Date generated**: 2026-05-02
**Branch**: `teams-relaunch-root-2026-04-17`
**Scope**: every P-series Vertex run from P8A (2026-04-24) through P18 (2026-05-01) plus the canonical scorecards. **Zero interpretation.** Every numeric claim cites a file or scorecard CSV path.

> **Read this BEFORE any framing doc.** Past framings have been wrong (see PSERIES_OPINIONS_2026-05-02.md disclaimer). The ledger is what was actually run, configured, and measured.

---

## How to read

- Each row = one Vertex run that produced a checkpoint scored against the promotion contract.
- `Lever changes vs P8A` lists only the deltas vs the P8A baseline configuration; everything else is identical (same backbone, same anti-shortcut levers unless explicitly toggled).
- Recall numbers are at the **contract-selected τ** (lexicographic policy: `dev_primary_real_fpr ≤ 0.02` first, then maximize dev fake recall). All from `selected_threshold_scorecard.csv` or `checkpoint_summary.csv` in the cited scorecard dir.
- "Promotion rank" = the lexicographic rank of this ckpt vs the others scored in the same scorecard. Rank 1 = least-bad of the trio; does NOT mean it cleared the contract floor.
- Status legend: `BASELINE` / `LIFT` (better than P8A on the contract) / `NO_LIFT` (rank ≥2; ≤ P8A) / `COLLAPSE` (trainer-side composite collapsed) / `CANCELLED` (run stopped early).

---

## The chain at a glance

| # | Packet | Yaml file | W&B run | Vertex job | Tag | Result vs P8A |
|---|---|---|---|---|---|---|
| 1 | **P8A** | (in current tree as `experiments/phase2_round13/R13_RLP*` — pre-04-25, ckpt step5000 from `9lmvb5b4`) | `9lmvb5b4` | n/a (preserved) | BASELINE | reference |
| 2 | P14_FT_FROM_P8A | `experiments/phase2_round13/R13_P14_FT_FROM_P8A.yaml` | unknown — pulled to `gs://training-job-outputs/phase2r13_experiments/<run>/periodic_step{500..4000}.pth` per `analysis/policy_reruns_2026-04-29_floor_0p70/checkpoint_summary.csv:2-10` | unknown | `p14`, `ft-from-p8a` | **NO_LIFT** (rank 6 in 2026-04-29 floor=0.70 scorecard) |
| 3 | P14_FACE_SCALE_JITTER_ISOLATED | `experiments/phase2_round13/R13_P14_FACE_SCALE_JITTER_ISOLATED.yaml` | `mclioexb` | unknown | `p14`, `jitter`, `leader` (per `viewer/model_dashboard_runs.yaml:65-86`) | **NO_LIFT** (single-ckpt scored only) |
| 4 | P14_DATA_FIX (fw=8.0) | (yaml not preserved — see comment in `experiments/phase2_round13/R13_P16_DATA_AXIS.yaml:282`) | `xan4dfto` | unknown | `p14`, `data-fix`, `collapse` (per `viewer/model_dashboard_runs.yaml:87-109`) | **COLLAPSE** (trainer-side `value_composite=0.126`) |
| 5 | P15_GRL_FROM_P8A | `experiments/phase2_round13/R13_P15_GRL_FROM_P8A.yaml` | `w5tky6ss` (per `viewer/model_dashboard_runs.yaml:27-55`) | unknown | `p15`, `grl`, `dann` | NO contract scorecard found in tree (trainer-side `value_composite=0.516`) |
| 6 | P16_DATA_AXIS | `experiments/phase2_round13/R13_P16_DATA_AXIS.yaml` | `rmic6wrc` | (scorecard 2026-04-30) | `p16`, `data-axis` | **NO_LIFT** (8 ckpts; all rank 2-9 vs P8A rank 1) |
| 7 | P17_LAYER3_HEAD (ArcFace) | `experiments/phase2_round13/R13_P17_LAYER3_HEAD.yaml` | `melp4mol` (per `HANDOFF.md:151`) | `2429114755361800192` | `p17`, `arcface`, `layer-3` | **NO_LIFT** (lockbox AUC 0.187 at best step; killed by probe) |
| 8 | P17_LAYER4_HEAD (ArcFace) | `experiments/phase2_round13/R13_P17_LAYER4_HEAD.yaml` | `faheakaf` | `3168830994157404160` | `p17`, `arcface`, `layer-4` | **NO_LIFT** (best step 2410, val AUC 0.7634; lockbox not probed) |
| 9 | P17_LAYER3_HEAD_LINEAR | `experiments/phase2_round13/R13_P17_LAYER3_HEAD_LINEAR.yaml` | `nqvfz44v` | `6468843621113135104` | `p17`, `linear` | **CANCELLED** at step 1285 (lockbox AUC 0.066 by probe) |
| 10 | P17_LAYER4_HEAD_LINEAR | `experiments/phase2_round13/R13_P17_LAYER4_HEAD_LINEAR.yaml` | `mbd951b8` | `5009677341845094400` | `p17`, `linear` | **CANCELLED** at step ~1450 |
| 11 | P18_METHOD_DOMAIN_GRL (treatment) | `experiments/phase2_round13/R13_P18_METHOD_DOMAIN_GRL.yaml` | `xpbvc1e4` (per `HANDOFF.md:71`) | `820959496569356288` | `p18`, `grl-12bucket` | **NO_LIFT** (contract rank 2 vs P8A rank 1) |
| 12 | P18_NO_GRL_CONTROL | `experiments/phase2_round13/R13_P18_NO_GRL_CONTROL.yaml` | `rgt4kw2u` (per `HANDOFF.md:71`) | `456167926752346112` | `p18`, `no-grl` | **NO_LIFT** (contract rank 3) |

**Net**: 11 packets between P8A and P18 (incl. variants). Zero promote. P8A remains rank 1 on every scorecard it was included in.

---

## Section 1 — P8A baseline (reference)

**W&B run**: `9lmvb5b4`
**Best ckpt** (used as P8A reference): `gs://training-job-outputs/phase2r13_experiments/9lmvb5b4/value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth` (cited in `experiments/phase2_round13/R13_P14_FT_FROM_P8A.yaml:83`).
**Trainer-side**: val AUC 0.9926, EER 0.0270 (filename).
**ArcFace s at step 5000**: 9.749 (annealed 6→12 over `anneal_steps=8000`; per `MEASUREMENTS_2026-05-02_P18.md` §0).

### 1.1 P8A on the 2026-05-02 D contract scorecard

(Source: `analysis/p18_probe_2026-05-01/d_results/promotion_contract/selected_threshold_scorecard.csv` and `checkpoint_summary.csv`.)

| Metric | Value |
|---|---:|
| Promotion rank (vs P18T, P18C) | **1** |
| Selected τ | 0.990946 |
| dev_primary_real_fpr | 0.019982 |
| dev_worst_real_stress_fpr | 0.015703 |
| dev_fake_macro_recall | **0.135844** |
| teams_fake_all_dev recall (n=2409) | 0.372769 |
| visomaster_enhanced_macro_dev recall (n=550) | **0.010909** |
| deeplive_enhanced_dev recall (n=545) | 0.023853 |
| teams_real_all_lockbox FPR (n=1361) | 0.001470 |
| teams_fake_all_lockbox recall (n=253) | 0.237154 |
| teams_real_dor_dev FPR (n=50) | **0.000** (0/50) |

### 1.2 Held-out lockbox AUC (n=87)

(Source: `MEASUREMENTS_2026-05-02_P18.md §2.8`.)

| Subset | n_real / n_fake | P8A AUC |
|---|---|---:|
| All lockbox | 40 / 47 | **0.7926** |
| Lockbox minus dor reals | 15 / 47 | 0.6511 |
| Lockbox only-dor reals + all fakes | 25 / 47 | 0.8774 |

### 1.3 Frozen-feature linear probe on visomaster buckets (Move 1, 2026-04-29)

(Source: per `analysis/move1_linear_probe_2026-04-29/` and memory `project_move1_bucket_gap_refuted.md`.)

P8A's frozen features rank-order viso fake-vs-real correctly across **every** bucket variant tested. Probe AUC range **0.92–0.998** including the `visomaster_enhanced` bucket. Identity-only control AUC = 0.987 → bucket gap is **identity-confounded**.

---

## Section 2 — P14 family

### 2.1 P14_FT_FROM_P8A (control: anti-shortcut bundle, no GRL, no new sources)

**Yaml**: `experiments/phase2_round13/R13_P14_FT_FROM_P8A.yaml`
**Description (yaml line 39)**: "FT-from-P8A_step5000 + P13 anti-shortcut bundle (anchor-aware + pipeline-rand + face scale-jitter)"
**Anti-shortcut bundle**: anchor_aware enabled (weight 5.0, pool: dor-real-webcam-false-flag); face_scale_jitter enabled (scale_limit 0.25); pipeline_randomization enabled.
**No GRL.**
**Sources enabled** (yaml line ~155-220): df40, deeplive, visomaster (buckets 4-6 only), teams. NOT enabled: visomaster_enhanced, visomaster_teams_enhanced, visomaster_hints*, proper_visomaster_*.
**Family weights** (yaml line ~244): same as P18 minus `visomaster_enhanced_fake` line.
**Vertex job**: unknown.

### 2.2 P14_FT scorecard outcome (2026-04-29 floor=0.70 policy)

(Source: `analysis/policy_reruns_2026-04-29_floor_0p70/checkpoint_summary.csv`.)

| Step | Promotion rank | dev_fake_macro_recall | visomaster_enhanced_macro_dev recall | teams_fake_all_lockbox recall |
|---|---:|---:|---:|---:|
| 500 | 1 | 0.17849 | 0.056364 | 0.260870 |
| 1000 | 3 | 0.164274 | 0.014545 | 0.264822 |
| 1500 | 4 | 0.277381 | 0.056364 | 0.332016 |
| 2000 | 6 | 0.279316 | 0.047273 | 0.217391 |
| 3000 | 10 | 0.449401 | 0.167273 | 0.217391 |
| 3500 | 8 | 0.333243 | 0.058182 | 0.185771 |
| 4000 | 7 | 0.329037 | 0.052727 | 0.185771 |
| **P8A reference (rank 5)** | 5 | 0.30028 | 0.136364 | 0.387352 |

P14_FT step3000 had the highest measured `visomaster_enhanced_macro_dev` recall (0.167) of any P14_FT variant. It also had the lowest promotion rank (10) due to lockbox regression.

> NOTE: this is the 2026-04-29 floor=0.70 scorecard, NOT the current contract v3 (different selection policy). Cross-arm comparisons are valid; absolute numbers may shift under v3.

### 2.3 P14_FACE_SCALE_JITTER_ISOLATED (`mclioexb`, jitter@0.50 only)

**Yaml**: `experiments/phase2_round13/R13_P14_FACE_SCALE_JITTER_ISOLATED.yaml`
**Description**: "FT-from-P8A_step5000 + face_scale_jitter scale_limit=0.50 ONLY (anchor_aware + pipeline_random disabled)"
**W&B run**: `mclioexb`. **Trainer-side `value_composite`**: 0.661 (per `viewer/model_dashboard_runs.yaml:71`).
**Scorecard**: `analysis/scorecard_mclioexb_2026-04-30/checkpoint_summary.csv`.

| Metric | Value |
|---|---:|
| Promotion rank (single-ckpt) | 1 |
| Selected τ | 0.975031 |
| dev_fake_macro_recall | 0.127732 |
| visomaster_enhanced_macro_dev recall | **0.012727** |
| teams_fake_all_lockbox recall | 0.185771 |

Memory `project_mclioexb_does_not_promote_2026-04-30.md`: **zero τ in the 5549-pt grid meets contract under default OR in-tree relaxed policy.**

### 2.4 P14_DATA_FIX (`xan4dfto`, fw=8.0 — the COLLAPSE)

**Yaml**: not preserved in current tree. Reference in `experiments/phase2_round13/R13_P16_DATA_AXIS.yaml:282`: `# CONSERVATIVE (DATA_FIX used 8.0 → collapse).`
**W&B run**: `xan4dfto`. Per `viewer/model_dashboard_runs.yaml:87-109`:
- Recipe: "FT-from-P8A with P14 bundle plus visomaster_teams_enhanced family weight 8.0"
- Trainer metrics: `value_composite: 0.126` (vs P8A's ~0.99)
- Tag: `collapse`
- Note: "DATA_FIX Collapse — local artifact record marks DATA_FIX as a generalization collapse candidate"

No contract scorecard in tree (run collapsed before producing a deployable ckpt).

### 2.5 P14_DATA_FIX (revised 2026-04-29 PM, fw=8.0, drafted-but-not-launched)

**Yaml**: `experiments/phase2_round13/R13_P14_DATA_FIX.yaml` (current working tree).
**Status**: drafted. Yaml line 34 says `DO NOT LAUNCH without explicit user OK`.
**Differs from xan4dfto (per yaml header)**: explicitly disables `visomaster_enhanced` (clean, no Teams transport) and `visomaster_hints_teams` (bad-data lane); ENABLES `visomaster_teams_enhanced` with `companion_domains: [teams_v2]`. Same fw=8.0 as the collapsed run.
**Result**: not measured.

---

## Section 3 — P15 GRL

### 3.1 P15_GRL_FROM_P8A (`w5tky6ss`)

**Yaml**: `experiments/phase2_round13/R13_P15_GRL_FROM_P8A.yaml`
**Description**: "FT-from-P8A_step5000 + P14 anti-shortcut bundle + GRL quality-domain head λ=0.20"
**W&B run**: `w5tky6ss`.
**Trainer-side `value_composite`**: 0.516 (per `viewer/model_dashboard_runs.yaml:41`).
**GRL**: 4-bucket QUALITY_DOMAIN_MAP (df40 / deeplive / visomaster / external), λ static 0.20.
**Best ckpt**: `analysis/_features_cache_2026-04-30/value_composite_effort_20260430_step8500_auc0.9842_eer0.0457.pth` (per `viewer/model_dashboard_runs.yaml:31`).
**Contract scorecard**: not run (per absence in `analysis/`). Diagnostics ran against this ckpt instead.

Per memory `project_phase1a_method_cluster_axis_2026-05-01.md`: GRL @ static λ=0.20 didn't bite the encoder because the encoder wasn't using capture-mode (the GRL target axis); was using identity/method-cluster axis (`is_dor_shkedi`/`is_deeplive_enhanced`).

---

## Section 4 — P16 data axis

### 4.1 P16_DATA_AXIS (`rmic6wrc`, fw=2.0)

**Yaml**: `experiments/phase2_round13/R13_P16_DATA_AXIS.yaml`
**Description**: "FT-from-P8A_step5000 + visomaster_teams_enhanced fw=2.0 (conservative), no anti-shortcut bundle (single-lever discipline), no GRL"
**W&B run**: `rmic6wrc`.
**Source delta vs P8A**: `combined_paired.visomaster_teams_enhanced.enabled=true` (yaml line 247-258), companion_domains=[teams_v2], include_statuses=[teams_v2_companion], sampling_family_key=`visomaster_enhanced_fake`.
**Family weight delta**: added `visomaster_enhanced_fake: 2.0` (yaml line 282).
**Anti-shortcut bundle**: anchor_aware off, face_scale_jitter off, pipeline_randomization off.
**GRL**: off.
**Scorecard**: `analysis/scorecard_p16_data_axis_2026-04-30/checkpoint_summary.csv`. 8 P16 ckpts + P8A reference.

| Step | Promotion rank | Selected τ | dev_fake_macro_recall | visomaster_enhanced_macro_dev recall | teams_fake_all_dev recall | teams_fake_all_lockbox recall |
|---|---:|---:|---:|---:|---:|---:|
| **P8A reference** | **1** | 0.990946 | 0.135844 | 0.010909 | 0.372769 | 0.237154 |
| 500 | 2 | 0.988299 | 0.119267 | 0.007273 | 0.348692 | 0.185771 |
| 1500 | 3 | 0.983291 | 0.129155 | (n/a in summary) | (n/a) | 0.158103 |
| 2500 | 4 | 0.978595 | 0.163115 | (n/a) | (n/a) | 0.150198 |
| 3000 | 6 | 0.977664 | 0.159103 | (n/a) | (n/a) | 0.146245 |
| 4000 | 5 | 0.985842 | 0.158906 | (n/a) | (n/a) | 0.146245 |
| 5500 | 7 | 0.81093 | 0.166857 | (n/a) | (n/a) | 0.146245 |
| 6000 | 8 | 0.694207 | 0.169223 | (n/a) | (n/a) | 0.146245 |
| 7000 | 9 | 0.667867 | 0.171000 | (n/a) | (n/a) | 0.146245 |

**Per-pool calibrated recalls at τ that pins dev real_FPR=2%** (per memory `project_p16_data_axis_does_not_promote_2026-04-30.md`):
- visomaster_enhanced_macro_dev: **0.9–1.1% across all 8 P16 ckpts**.
- deeplive_enhanced_dev: 2.4% (P8A) → 17.4% (P16 step 7000).
- teams_fake_all_dev: 33–37% (similar across all ckpts).

**At default τ=0.5** (memory same source): P16 step 6000 hit visomaster recall 51.6%, deeplive recall 96.9%. **The model learned the source during training; deployment τ collapsed the recall.**

---

## Section 5 — P17 layer-X head

### 5.1 P17_LAYER3_HEAD (ArcFace, `melp4mol`)

**Yaml**: `experiments/phase2_round13/R13_P17_LAYER3_HEAD.yaml`
**Description**: "Layer-3 [CLS] readout from frozen P8A backbone — test substrate-invariant features hypothesis"
**W&B run**: `melp4mol` (per `HANDOFF.md:153`)
**Vertex job**: `2429114755361800192`
**Best step**: 2088. Val AUC 0.7390. **Lockbox AUC: 0.187** (per `HANDOFF.md:153`).
**ArcFace L3 lockbox AUC trajectory** (per `HANDOFF.md:97-98`): ep1 = 0.7229 → step 1687 = 0.0745 → step 2088 = 0.1867.
**Pearson r between trained-head and fresh-LR scores on lockbox** (per `HANDOFF.md:99-101`): −0.20 to −0.41 across all trained ckpts. Cosine sim between fresh-LR direction and trained-head decision direction: +0.03 to +0.09 (orthogonal).

Memory `project_p17_trained_head_destroys_substrate_invariance.md`: layer-3 readout idea structurally dead; both ArcFace + LINEAR start clean at ep1 then learn AWAY from invariant signal.

### 5.2 P17_LAYER4_HEAD (ArcFace, `faheakaf`)

**Yaml**: `experiments/phase2_round13/R13_P17_LAYER4_HEAD.yaml`
**W&B run**: `faheakaf`. **Vertex job**: `3168830994157404160`.
**Best step**: 2410. Val AUC 0.7634. Lockbox not probed (cache lacks layer-4).

### 5.3 P17_LAYER3_HEAD_LINEAR (`nqvfz44v`, CANCELLED)

**W&B run**: `nqvfz44v`. **Vertex job**: `6468843621113135104` (CANCELLED at step 1285).
**Lockbox AUC trajectory** (per `HANDOFF.md:98`): ep1 = 0.7117 → step 1044 = 0.0431 → step 1285 = 0.0660.

### 5.4 P17_LAYER4_HEAD_LINEAR (`mbd951b8`, CANCELLED)

**W&B run**: `mbd951b8`. **Vertex job**: `5009677341845094400` (CANCELLED at step ~1450). Lockbox not probed.

---

## Section 6 — P18 method-conditional GRL

### 6.1 P18_METHOD_DOMAIN_GRL (treatment, `xpbvc1e4`)

**Yaml**: `experiments/phase2_round13/R13_P18_METHOD_DOMAIN_GRL.yaml`
**Description (yaml line 39)**: "FT-from-P8A_step5000 + 6-class source-domain GRL ISOLATED (no anchor_aware, no pipeline_random, no face_scale_jitter)" — actually 12-class per yaml `quality_domain_count: 12`.
**W&B run**: `xpbvc1e4` (per `HANDOFF.md:71`). **Vertex job**: `820959496569356288`.
**Best ckpt** (used in D contract): `gs://training-job-outputs/phase2r13_experiments/xpbvc1e4/periodic_effort_20260501_step4000_auc0.9907_eer0.0279.pth`.
**ArcFace s at step 4000**: 12.0 (annealed 6→12 over `anneal_steps=4000`).
**GRL**: 12-bucket method-domain map (`analysis/method_class_audit_2026-05-01/proposed_method_domain_map.py`). λ ramped 0.245 → 0.948.

### 6.2 P18T D contract scorecard (2026-05-02)

(Source: `analysis/p18_probe_2026-05-01/d_results/promotion_contract/selected_threshold_scorecard.csv`.)

| Metric | Value |
|---|---:|
| Promotion rank (vs P8A, P18C) | **2** |
| Selected τ | 0.993590 |
| dev_fake_macro_recall | 0.150550 |
| visomaster_enhanced_macro_dev recall | 0.018182 |
| deeplive_enhanced_dev recall | 0.075229 |
| teams_fake_all_dev recall | 0.358240 |
| teams_real_dor_dev FPR (n=50) | 0.040 (2/50) |
| teams_real_all_lockbox FPR | 0.002939 |
| teams_fake_all_lockbox recall | 0.177866 |

### 6.3 P18C_NO_GRL_CONTROL (`rgt4kw2u`)

**Yaml**: `experiments/phase2_round13/R13_P18_NO_GRL_CONTROL.yaml`
**W&B run**: `rgt4kw2u`. **Vertex job**: `456167926752346112`.
**Best ckpt**: `gs://training-job-outputs/phase2r13_experiments/rgt4kw2u/periodic_effort_20260501_step4000_auc0.9871_eer0.0330.pth`.

### 6.4 P18C D contract scorecard

| Metric | Value |
|---|---:|
| Promotion rank | 3 |
| Selected τ | 0.994625 |
| dev_fake_macro_recall | 0.179269 |
| visomaster_enhanced_macro_dev recall | 0.014545 |
| deeplive_enhanced_dev recall | **0.194495** |
| teams_fake_all_dev recall | 0.328767 |
| teams_real_dor_dev FPR (n=50) | 0.040 (2/50) |
| teams_real_all_lockbox FPR | **0.008082** (5.5× P8A) |
| teams_fake_all_lockbox recall | 0.173913 |

### 6.5 P18 CPU diagnostics (2026-05-02)

(Full numbers in `MEASUREMENTS_2026-05-02_P18.md §2.1-2.9`.)

- **Lockbox AUC** (n=87): P8A 0.7926, P18T 0.6734, P18C 0.4973.
- **Lockbox dor real FPR @ τ=0.92 s-scaled**: P8A 0.00, P18T 0.00, **P18C 0.52** (the 0% → 52% regression).
- **Per-frame paired Wilcoxon on n=25 dor lockbox reals**: P18T < P18C (p<0.0001), P18T > P8A (p<0.0001).
- **Within-bucket-3 dor-vs-other LR @ FINAL [CLS] AUC**: P8A 0.9826, P18T 0.9536, P18C 0.9965.

---

## Section 6.5 — P22 augmentation curriculum (`dot1buye`)

### 6.5.1 P22_AUG_CURRICULUM training run

**Yaml**: `experiments/phase2_round13/R13_P22_AUG_CURRICULUM.yaml`
**Description (yaml)**: FT-from-P8A_step5000 with single-lever symmetric `pipeline_randomization` (Gaussian luma blur σ ∈ [0.5, 4.0] p=0.7, JPEG q ∈ [40, 95] p=0.6, brightness β ∈ [-40, 40] p=0.5). No anchor_aware, no GRL, no face_scale_jitter, no data-axis source weights.
**W&B run**: `dot1buye` (sparkling-sun-202). **Vertex job**: `9071387644058927104` (us-west4).
**Image**: 1.3.242 (training) / 1.3.243 (eval).
**Wall time**: 14:32 → 17:49 UTC 2026-05-02 (~3h17m).
**Total training steps**: 8000 (cap; nEpochs=16 floor).
**ArcFace s schedule**: 6.0 → 12.0 over `anneal_steps=8000`.

### 6.5.2 P22 trainer trajectory (W&B)

| Step | Train AUC | Train EER |
|---:|---:|---:|
| 500 | 0.9867 | 0.0819 |
| 1000 | 0.9903 | 0.0282 |
| 1500 | 0.9883 | 0.0452 |
| 2000 | 0.9885 | 0.0480 |
| 3000 | 0.9853 | 0.0565 |
| 3500 | 0.9782 | 0.0650 |
| 4000 | 0.9746 | 0.0932 |
| 6000 | 0.9631 | 0.0932 |
| 7000 | 0.9364 | 0.1469 |
| 8000 | 0.9406 | 0.1441 |

(Source: GCS object names in `gs://training-job-outputs/phase2r13_experiments/dot1buye/`.)

### 6.5.3 P22 contract scorecard (2026-05-02 evening)

**Scorecard run**: `p22-aug-curriculum-scorecard-20260502`
**Suite manifest**: `arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml` (9 suites)
**Checkpoint map**: `arena/checkpoint_maps/teams_target_domain.p22_aug_curriculum_2026-05-02.yaml` (4 ckpts: P8A_REFERENCE_STEP5000 + P22_AUG_STEP1000/4000/8000)
**Wall time**: 17:42 → 20:00 UTC (~2h16m on A100, us-west4)
**Source**: `analysis/p22_eval_2026-05-02/scorecard_data/promotion_contract/checkpoint_summary.csv`.

| Ckpt | Selected τ | dev_primary_real_fpr | dev_worst_real_stress_fpr | dev_fake_macro_recall | teams_fake_all_dev recall | viso recall | deeplive recall | lockbox_real_FPR | lockbox_fake_recall | rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.991 | 0.020 | 0.016 | 0.136 | 0.373 | 0.011 | 0.024 | 0.001 | 0.237 | **1** |
| P22_AUG_STEP1000 | 0.979 | 0.020 | 0.046 | 0.127 | 0.322 | 0.004 | 0.055 | 0.003 | 0.130 | **2** |
| P22_AUG_STEP4000 | 0.841 | 0.020 | 0.044 | 0.309 | 0.450 | 0.005 | 0.472 | 0.028 | 0.134 | **3** |
| P22_AUG_STEP8000 | 0.524 | 0.020 | 0.039 | 0.393 | 0.574 | 0.044 | 0.561 | 0.043 | 0.217 | **4** |

(Promotion ranks per the v3 contract's lexicographic ordering: minimize `dev_worst_real_stress_fpr` first, then maximize `dev_fake_macro_recall`. P8A wins rank 1 on stress-FPR.)

### 6.5.4 P22 multi-FPR-floor operating-point grid

(Source: `analysis/p22_eval_2026-05-02/scorecard_data/promotion_contract/operating_point_grid.csv` via `arena/scorecard_with_fpr_grid.py`.)

**teams_fake_all_dev recall**:

| FPR floor | P8A | P22 step1k | P22 step4k | P22 step8k |
|---:|---:|---:|---:|---:|
| 2% | 0.373 | 0.322 | 0.450 | **0.574** |
| 5% | 0.466 | 0.509 | 0.571 | **0.668** |
| 10% | 0.641 | 0.665 | 0.670 | **0.749** |
| 20% | **0.831** | 0.822 | 0.761 | 0.803 |

**visomaster_enhanced_macro_dev recall**:

| FPR floor | P8A | P22 step1k | P22 step4k | P22 step8k |
|---:|---:|---:|---:|---:|
| 2% | 0.011 | 0.004 | 0.005 | **0.044** |
| 5% | 0.071 | **0.082** | 0.027 | 0.075 |
| 10% | **0.271** | 0.242 | 0.060 | 0.126 |
| 20% | **0.562** | 0.469 | 0.126 | 0.193 |

**deeplive_enhanced_dev recall**:

| FPR floor | P8A | P22 step1k | P22 step4k | P22 step8k |
|---:|---:|---:|---:|---:|
| 2% | 0.024 | 0.055 | 0.472 | **0.561** |
| 5% | 0.145 | 0.407 | 0.734 | **0.763** |
| 10% | 0.429 | 0.617 | **0.912** | 0.908 |
| 20% | 0.763 | 0.833 | **0.982** | 0.982 |

### 6.5.5 P22 falsifier outcomes (CPU, post-scorecard)

(Source: `analysis/p22_eval_2026-05-02/falsifiers_per_checkpoint.csv` and `verdicts.csv`.)

Pre-registered falsifier rules from `analysis/cpu_decision_2026-05-02_pm_late/FINDINGS_AND_DECISION.md`:
- **F1**: |Pearson r(score, laplacian_var)| on dev viso drops by ≥ 0.15 vs P8A
- **F2** (literal text): standalone-shortcut LR AUC drops 0.882 → ≤ 0.78
- **F3**: viso recall at FPR=2% rises 0.5% → ≥ 3% (P8A actual 1.1%)
- 2/3 → success; 0/3 → didn't bite; 1/3 → ambiguous

**Implementation note (this is interpretation)**: F2's literal "standalone-shortcut LR AUC" is data-only (model-independent), so it cannot drop between P8A and P22 evaluated on the same data. The eval harness substituted **R²(score | {laplacian, luma, skin}) drops by ≥ 0.05** as the model-dependent analogue. (See OPINIONS doc for whether this substitution is fair.)

| Ckpt | F1 raw r (n=63 viso fakes) | F1 \|Δr\| vs P8A row | F2 R²(score\|attrs) on n=473 pool | F3 viso recall @ FPR=2% | F1 PASS? | F2 PASS? | F3 PASS? | Score |
|---|---:|---:|---:|---:|:-:|:-:|:-:|---:|
| P8A_REFERENCE_STEP5000 | +0.444 | (baseline) | 0.047 | 0.011 | — | — | — | (ref) |
| P22_AUG_STEP1000 | +0.372 | 0.072 | 0.099 | 0.004 | NO | NO (rose) | NO (lower) | **0/3** |
| P22_AUG_STEP4000 | -0.335 | 0.109 | 0.137 | 0.005 | NO | NO (rose) | NO (lower) | **0/3** |
| P22_AUG_STEP8000 | -0.119 | 0.325 | 0.175 | 0.044 | YES | NO (rose) | YES | **2/3** |

(Note: the script's verdict file used a different P8A baseline for F1 — the hardcoded `viso_pearson_r=-0.301` from a 923-frame pool that doesn't include viso. Using the per-ckpt P8A row from the same scorecard (+0.444) gives larger |Δ| for step8k (0.325) and same PASS verdict; using the hardcoded -0.301 baseline gives |Δ|=0.182 for step8k, still PASS. The conclusion is robust to this choice; the magnitudes shift.)

### 6.5.6 P22 score residualization (CPU)

(Source: `analysis/p22_eval_2026-05-02/score_residual_per_checkpoint.csv`. Pool: dev real + fake suites that joined to attrs, n=473.)

| Ckpt | n | R²(score \| attrs) | AUC orig score | AUC pred-from-attrs | AUC residual |
|---|---:|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 473 | 0.047 | 0.778 | 0.446 | 0.747 |
| P22_AUG_STEP1000 | 473 | 0.099 | 0.757 | 0.504 | 0.778 |
| P22_AUG_STEP4000 | 473 | 0.137 | 0.753 | 0.550 | 0.752 |
| P22_AUG_STEP8000 | 473 | 0.175 | 0.769 | 0.607 | 0.742 |

The residual AUC (signal in the score after partialing out attrs) is roughly flat across all 4 ckpts (0.74–0.78). **The score-vs-attrs R² rises monotonically with P22 training step** — i.e. the trained-with-aug model's score depends MORE on the 3-attribute combination, not less. AUC pred-from-attrs (a model that uses ONLY attrs to predict P22 score) rises 0.45 → 0.61, confirming the same.

### 6.5.7 P22 joint dev+lockbox τ-recalibration (CPU follow-up, 2026-05-02 evening)

(Source: `analysis/p22_eval_2026-05-02/cpu_followups/outputs/01_joint_recal_summary.csv`. Smallest τ s.t. BOTH dev primary FPR ≤ 2% AND lockbox real FPR ≤ 2%.)

| Ckpt | τ | dev_FPR | lockbox_FPR | viso_dev recall | deeplive_dev recall | teams_fake_dev recall | lockbox_fake recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.993 | 1.8% | 0.14% | 0.4% | 0.4% | 41.7% | 19.1% |
| P18T_GRL_TREATMENT_STEP4000 | 0.994 | 1.8% | 0.28% | 1.6% | 5.3% | 42.1% | 17.9% |
| P22_AUG_STEP1000 | 0.971 | 2.0% | 0.6% | 2.4% | 17.8% | 45.0% | 18.8% |
| P22_AUG_STEP4000 | 0.876 | 0.9% | 1.9% | 0.2% | 37.4% | 43.2% | 6.1% |
| P22_AUG_STEP8000 | 0.539 | 0.2% | 1.97% | 0.7% | 13.8% | 40.9% | 15.5% |

### 6.5.8 P22 score distribution width (CPU follow-up)

(Source: `analysis/p22_eval_2026-05-02/cpu_followups/outputs/02_score_quantiles.csv`. Median score per (suite, ckpt). All-suite IQR per ckpt: P8A 0.03–0.75 range, P22 step1k 0.13–0.58, P22 step8k uniformly 0.02–0.05.)

P22_AUG_STEP8000 var(score) on the cross-suite pool = 0.0012 (P8A: 0.157 — a 130× collapse).

### 6.5.9 P22 per-identity lockbox FP concentration (CPU follow-up)

(Source: `analysis/p22_eval_2026-05-02/cpu_followups/outputs/05_per_identity_fpr_summary.csv`. Total identities in `teams_real_all_lockbox`: 5.)

| Identity | n_frames | P8A FPs | P22 step1k FPs | P22 step4k FPs | P22 step8k FPs |
|---|---:|---:|---:|---:|---:|
| PC_Generator | 29 | 2 | 1 | 5 | **22** (75.9% per-id FPR) |
| dor_shkedi | 1170 | **0** | 6 | 44 | **36** (3.1% per-id FPR) |
| real_dor | 109 | 0 | 0 | 0 | 2 |
| Chikara_Takahashi | 42 | 0 | 2 | 0 | 0 |
| bla_bla_chow | 68 | 0 | 0 | 0 | 0 |

P22 step8k regresses on P8A's 0/1170 dor invariance (`project_p18_diagnostics_complete_2026-05-02` documented this as P8A's signature defense).

### 6.5.10 Cross-packet ensemble at joint FPR=2% (CPU follow-up)

(Source: `analysis/p22_eval_2026-05-02/cpu_followups/outputs/06_ensemble.csv`. Best ensembles vs single models, joint dev+lockbox FPR ≤ 2%.)

| Strategy | τ | dev_FPR | lockbox_FPR | viso recall | deeplive recall | teams_fake_dev | teams_fake_lockbox |
|---|---:|---:|---:|---:|---:|---:|---:|
| P8A (single) | 0.993 | 1.8% | 0.14% | 0.4% | 0.4% | 41.7% | 19.1% |
| P22 step1k (single) | 0.971 | 2.0% | 0.6% | 2.4% | 17.8% | 45.0% | 18.8% |
| P8A + P22 step1k :: min | 0.944 | 2.0% | 1.1% | 7.1% | 20.4% | 49.0% | 26.6% |
| P8A + P22 step8k :: min | 0.526 | 0.5% | 1.97% | 3.5% | 37.3% | 54.2% | 18.8% |
| P8A + P22 step1k + step8k :: mean | 0.812 | 2.0% | 1.0% | 6.0% | 21.1% | 51.3% | 27.3% |

### 6.5.11 P22 full-viso F1 Pearson r (CPU follow-up)

(Source: `analysis/p22_eval_2026-05-02/cpu_followups/outputs/07_full_viso_pearson.csv`. n=551 viso fakes; Laplacian computed via cv2 on GCS-fetched frames. P-values: all <0.001 except P22 step8k p=0.008.)

| Ckpt | Pearson r(score, lap) on full viso | F1 |Δr| ≥ 0.15 vs P8A |
|---|---:|:-:|
| P8A | +0.5074 | (baseline) |
| P18T | +0.1480 | PASS (Δ=0.359) |
| P22_AUG_STEP1000 | +0.4471 | FAIL (Δ=0.060) |
| P22_AUG_STEP4000 | -0.2610 | PASS (Δ=0.246) |
| P22_AUG_STEP8000 | -0.1128 | PASS (Δ=0.395) |

P8A's r is strongly POSITIVE on viso (+0.51) — opposite direction from the deeplive shortcut. P18T also weakens the viso laplacian shortcut.

### 6.5.7-bis P22 lockbox readout (out-of-sample) [renumbered to 6.5.12]

| Ckpt | lockbox_real_FPR (target ≤ 0.02) | lockbox_fake_recall |
|---|---:|---:|
| P8A_REFERENCE_STEP5000 | 0.001 | 0.237 |
| P22_AUG_STEP1000 | 0.003 | 0.130 |
| P22_AUG_STEP4000 | 0.028 ⚠ | 0.134 |
| P22_AUG_STEP8000 | 0.043 ⚠ | 0.217 |

⚠ P22_AUG_STEP4000 and STEP8000 BOTH violate the contract's 2% lockbox real-FPR target at the τ calibrated on dev. This is calibrated-τ drift between dev and lockbox at the chosen calibration τ, not a separate model artifact.

---

## Section 6.6 — S1 short-training-cap (`padjfsoq`, 2026-05-02 evening)

### 6.6.1 S1_REDUX_SHORT training run

**Yaml**: `experiments/phase2_round13/R13_S1_P22_REDUX_SHORT.yaml`
**Description**: P22 aug curriculum with `total_training_steps = 1000` (vs P22's 8000) and `s_end = 8.0` (vs P22's 12.0). FT from same base as P22 (P8A_step5000). Single-lever isolates the cap.
**W&B run**: `padjfsoq`. **Vertex job**: `3095110938538278912` (us-west4).
**Image**: 1.3.244 (training) / 1.3.246 (eval).
**Wall time**: 21:18 → 23:10 UTC 2026-05-02 (~1h53m).
**Total training steps**: 1000.
**ArcFace s schedule**: 6.0 → 8.0 over `anneal_steps = 1000`.

### 6.6.2 S1 trainer collapse-metric trajectory (W&B)

(Source: W&B run `padjfsoq` history pulled via Probe 1; `outputs/01_select_best_ckpt_padjfsoq.json`.)

| Metric | Peak value | Peak step | Closest periodic save |
|---|---:|---:|---|
| `train/collapse/class_separation` | 3.91 | 600 | step 600 |
| `train/collapse/logit_std` | 2.81 | 0 | step 100 |
| `val_holdout/overall/auc` | 0.9870 | 800 | step 800 |
| `val_in_dist/overall/auc` | 0.9881 | 600 | step 600 |

(Top-3 by class_sep: step 600 (3.91), step 100 (3.89), step 400 (3.58).)

### 6.6.3 S1 contract scorecard (2026-05-03 early AM)

**Scorecard**: `s1-redux-scorecard-20260502` (us-west4)
**Suite manifest**: same as P22 (`teams_promotion_contract_2026-04-23_with_dor.yaml`)
**Checkpoints scored**: P8A_REFERENCE_STEP5000, P22_AUG_STEP1000, S1_REDUX_SHORT_STEP100/400/600 (5 ckpts × 9 suites)
**Source**: `analysis/s1_s2_eval/scorecard_A_s1/promotion_contract/checkpoint_summary.csv`

| Ckpt | Selected τ | dev_FPR | dev_macro | viso recall | deeplive recall | teams_fake_dev | lockbox_real_FPR | lockbox_fake | rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **S1_REDUX_SHORT_STEP600** | 0.972 | 0.020 | 0.113 | 0.005 | **0.000** | 0.332 | 0.000 | 0.166 | **1** |
| P8A_REFERENCE_STEP5000 | 0.991 | 0.020 | 0.136 | 0.011 | 0.024 | 0.373 | 0.001 | 0.237 | 2 |
| S1_REDUX_SHORT_STEP400 | 0.967 | 0.020 | 0.123 | 0.009 | 0.007 | 0.352 | 0.001 | 0.213 | 3 |
| P22_AUG_STEP1000 | 0.979 | 0.020 | 0.127 | 0.004 | 0.055 | 0.322 | 0.003 | 0.130 | 4 |
| S1_REDUX_SHORT_STEP100 | 0.929 | 0.020 | 0.125 | 0.007 | 0.011 | 0.357 | 0.004 | 0.241 | 5 |

(S1 step600 wins promotion rank #1 only because of lex-first stress-FPR. **At calibrated τ S1 step600 has 0% deeplive recall** — the model collapsed at step 600 in a way that isn't visible in val_AUC but kills operating-point recall.)

### 6.6.4 S1 multi-FPR-floor view (joint dev+lockbox)

(Source: `outputs/05_chain_joint_summary.csv` — Smallest τ s.t. BOTH dev primary FPR ≤ floor AND lockbox real FPR ≤ floor.)

**Viso recall:**
| floor | P8A | S1 step100 | S1 step400 | S1 step600 |
|---:|---:|---:|---:|---:|
| 0.02 | 0.4% | 0.7% | 1.3% | 0.9% |
| 0.05 | 5.8% | 5.3% | 5.1% | 5.8% |
| 0.10 | 26.9% | 17.8% | 14.2% | 18.0% |
| 0.20 | 57.6% | 49.5% | 36.7% | 45.3% |

**Deeplive recall:**
| floor | P8A | S1 step100 | S1 step400 | S1 step600 |
|---:|---:|---:|---:|---:|
| 0.02 | 0.4% | 0.4% | 1.8% | 0.6% |
| 0.05 | 10.8% | 10.5% | 13.6% | 12.3% |
| 0.10 | 42.4% | 31.9% | 35.6% | 32.3% |
| 0.20 | 78.7% | 66.8% | 69.0% | 69.2% |

**Teams_fake_lockbox recall:**
| floor | P8A | S1 step100 | S1 step400 | S1 step600 |
|---:|---:|---:|---:|---:|
| 0.02 | 19.1% | 22.6% | 27.5% | 23.5% |
| 0.05 | 30.1% | 36.5% | 46.4% | 47.5% |
| 0.10 | 54.1% | 57.4% | 72.9% | 73.9% |
| 0.20 | 87.8% | 87.8% | 93.7% | 93.7% |

(S1 ckpts beat P8A on teams_fake_lockbox at FPR ≥ 5% but lose on viso at every FPR floor.)

---

## Section 6.7 — S2 earlier-base (`evm4y66r`, 2026-05-02 evening)

### 6.7.1 S2_EARLIER_BASE training run

**Yaml**: `experiments/phase2_round13/R13_S2_P22_EARLIER_BASE.yaml`
**Description**: Same as S1 (1000-step cap, s_end=8) but FT from `top_n_effort_20260424_step2500_auc0.9936_eer0.0270.pth` (P8A's run `9lmvb5b4` step 2500, val AUC 0.9936). Tests whether less-saturated base helps.
**W&B run**: `evm4y66r`. **Vertex job**: `8631168820662239232` (us-east1).
(First launch failed at 21:19 UTC due to wrong GCS path — `value_composite_*` doesn't exist at step 2500, only `top_n_effort_*` and `ood_composite_*`. Re-launched after yaml fix at 21:28 UTC.)

### 6.7.2 S2 trainer collapse-metric trajectory

(Source: W&B run `evm4y66r`; `outputs/01_select_best_ckpt_evm4y66r.json`.)

| Metric | Peak value | Peak step | Closest periodic save |
|---|---:|---:|---|
| `train/collapse/class_separation` | **4.89** | 200 | step 200 |
| `train/collapse/logit_std` | 3.34 | 300 | step 300 |
| `val_holdout/overall/auc` | 0.9890 | 500 | step 500 |
| `val_in_dist/overall/auc` | 0.9873 | 500 | step 500 |

(Class separation peaks much earlier and higher than S1 — supports the "earlier base = more headroom" hypothesis.)

### 6.7.3 S2 contract scorecard (2026-05-03 early AM)

**Scorecard**: `s2-earlier-base-scorecard-20260502` (us-east1)
**Source**: `analysis/s1_s2_eval/scorecard_B_s2/promotion_contract/checkpoint_summary.csv`

| Ckpt | Selected τ | dev_FPR | dev_macro | viso recall | deeplive recall | teams_fake_dev | lockbox_real_FPR | **lockbox_fake** | rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **S2_EARLIER_BASE_STEP300** | 0.953 | 0.020 | 0.132 | 0.016 | 0.017 | 0.363 | 0.001 | 0.340 | **1** |
| S2_EARLIER_BASE_STEP200 | 0.964 | 0.020 | 0.131 | 0.016 | 0.000 | 0.378 | 0.001 | **0.451** | 2 |
| S2_EARLIER_BASE_STEP600 | 0.958 | 0.020 | 0.147 | 0.016 | 0.050 | 0.376 | 0.001 | 0.431 | 3 |
| P8A_REFERENCE_STEP5000 | 0.991 | 0.020 | 0.136 | 0.011 | 0.024 | 0.373 | 0.001 | 0.237 | 4 |
| P22_AUG_STEP1000 | 0.979 | 0.020 | 0.127 | 0.004 | 0.055 | 0.322 | 0.003 | 0.130 | 5 |

(**S2 step300 wins promotion rank #1 — first post-P8A ckpt ever to do so.** lockbox_fake_recall S2 = 34-45% vs P8A's 23.7% — 1.4-1.9× lift at the contract τ.)

### 6.7.4 S2 multi-FPR-floor view (joint dev+lockbox)

**Viso recall:**
| floor | P8A | S2 step200 | S2 step300 | S2 step600 |
|---:|---:|---:|---:|---:|
| 0.02 | 0.4% | 1.6% | 1.6% | 1.6% |
| 0.05 | 5.8% | 2.4% | 2.9% | 4.0% |
| 0.10 | 26.9% | 6.6% | 9.3% | 11.8% |
| 0.20 | 57.6% | 23.1% | 34.6% | 36.0% |

(**S2 underperforms P8A on viso at FPR ≥ 5%.** The earlier base does not help viso.)

**Deeplive recall:**
| floor | P8A | S2 step200 | S2 step300 | S2 step600 |
|---:|---:|---:|---:|---:|
| 0.02 | 0.4% | 0.0% | 2.8% | 11.2% |
| 0.05 | 10.8% | 8.1% | 18.5% | 31.0% |
| 0.10 | 42.4% | 26.6% | 42.2% | 53.6% |
| 0.20 | 78.7% | 60.7% | 76.0% | 80.9% |

**Teams_fake_lockbox recall:**
| floor | P8A | S2 step200 | S2 step300 | **S2 step600** |
|---:|---:|---:|---:|---:|
| 0.02 | 19.1% | **52.0%** | 43.5% | 54.1% |
| 0.05 | 30.1% | **73.7%** | 64.2% | 79.3% |
| 0.10 | 54.1% | 86.8% | 84.0% | **91.5%** |
| 0.20 | 87.8% | 96.7% | 95.8% | **97.7%** |

(**S2 step600 is the lockbox-transfer champion of the entire chain.** 92% lockbox_fake_recall at FPR=10% is +37pp over P8A.)

---

## Section 6.8 — S3 viso-weight-bump (`1sakmkv4`, 2026-05-03 early AM)

### 6.8.1 S3_VISO_WEIGHT training run

**Yaml**: `experiments/phase2_round13/R13_S3_VISO_WEIGHT.yaml`
**Description**: Identical to S2 except `combined_paired.sampling.family_weights.visomaster_fake = 8.0` (vs S2's 4.0). Tests whether higher viso exposure in batches lifts viso recall under S2's recipe.
**W&B run**: `1sakmkv4`. **Vertex job**: `3275254923633098752` (us-west4).
**Image**: 1.3.247 (training) / 1.3.248 (eval).
**Wall time**: 02:22 → 04:03 UTC 2026-05-03 (~1h41m).

### 6.8.2 S3 trainer collapse-metric trajectory

| Metric | Peak value | Peak step | Closest periodic save |
|---|---:|---:|---|
| `train/collapse/class_separation` | **4.97** | 550 | step 500 |
| `train/collapse/logit_std` | 3.01 | 1000 | step 1000 |
| `val_holdout/overall/auc` | 0.9905 | 600 | step 600 |
| `val_in_dist/overall/auc` | 0.9834 | 600 | step 600 |

(**S3 has the highest class_separation peak in the entire chain (4.97).** Higher than S2 (4.89), higher than S1 (3.91). Logit_std didn't collapse — peak at step 1000.)

### 6.8.3 S3 contract scorecard (2026-05-03 morning)

**Scorecard**: `s3-viso-weight-scorecard-20260503` (us-east1)
**Source**: `analysis/s1_s2_eval/scorecard_C_s3/promotion_contract/checkpoint_summary.csv`

| Ckpt | Selected τ | dev_FPR | dev_macro | viso recall | deeplive recall | teams_fake_dev | lockbox_fake | rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S2_EARLIER_BASE_STEP600 | 0.958 | 0.020 | 0.147 | 0.016 | 0.050 | 0.376 | 0.431 | 1 |
| **S3_VISO_WEIGHT_STEP500** | 0.945 | 0.020 | 0.151 | 0.015 | 0.061 | 0.378 | 0.348 | 2 |
| S3_VISO_WEIGHT_STEP1000 | 0.977 | 0.020 | 0.137 | 0.013 | 0.035 | 0.365 | 0.265 | 3 |
| P8A_REFERENCE_STEP5000 | 0.991 | 0.020 | 0.136 | 0.011 | 0.024 | 0.373 | 0.237 | 4 |
| S3_VISO_WEIGHT_STEP700 | 0.956 | 0.020 | 0.159 | 0.016 | 0.079 | 0.382 | 0.237 | 5 |
| P22_AUG_STEP1000 | 0.979 | 0.020 | 0.127 | 0.004 | 0.055 | 0.322 | 0.130 | 6 |

(S3 ckpts are competitive at the contract but **S3 LOST the lockbox_fake_recall lead to S2 step600**. The viso weight bump from 4.0 to 8.0 mildly hurt the lockbox transfer.)

### 6.8.4 S3 multi-FPR-floor view (joint dev+lockbox)

**Viso recall (the headline metric):**
| floor | P8A | S2 step600 | S3 step500 | S3 step700 | S3 step1000 |
|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.4% | 1.6% | 1.5% | 1.6% | 1.6% |
| 0.05 | 5.8% | 4.0% | 6.6% | 6.4% | 6.0% |
| 0.10 | **27.0%** | 11.8% | 13.8% | 13.8% | 13.6% |
| 0.20 | **57.6%** | 36.0% | 31.8% | 30.7% | 30.4% |

**S3 viso recall at FPR=10% = 13.8% — far below P8A's 27%.** The viso family weight bump did not break the ceiling. Best F-S3-B falsifier value (target: > 30%) is 13.8% — clear FAIL.

**Teams_fake_lockbox recall:**
| floor | S2 step600 | S3 step500 | S3 step700 | S3 step1000 |
|---:|---:|---:|---:|---:|
| 0.02 | 54.1% | 35.9% | 30.6% | 43.8% |
| 0.05 | 79.3% | 67.8% | 52.5% | 67.5% |
| 0.10 | **91.5%** | 83.8% | 74.4% | 82.6% |
| 0.20 | **97.7%** | 94.8% | 90.8% | 94.4% |

(S3 lost 5-15pp on lockbox_fake_recall vs S2 step600 across all FPR floors.)

---

## Section 6.9 — Cross-cutting summary across S1/S2/S3 (the Sx batch)

### 6.9.1 Headline failure: viso ceiling unbroken

(Source: `outputs/05_chain_joint_summary.csv`, FPR floor 0.10 view.)

| Packet | best viso recall at joint FPR=10% |
|---|---:|
| P8A_REFERENCE_STEP5000 | **27.0%** ← still champion |
| P22_AUG_STEP1000 | 24.9% |
| P22_AUG_STEP4000 | 5.5% |
| P22_AUG_STEP8000 | 13.5% |
| S1_REDUX_SHORT (any step) | 14-18% |
| S2_EARLIER_BASE (any step) | 7-12% |
| S3_VISO_WEIGHT (any step) | 14% |

**No post-P8A packet has improved viso recall at joint FPR=10%.** P8A held the viso lead through 10+ packets across the entire R13 chain (P14, P15, P16, P17, P18T/C, P22, S1, S2, S3 plus their step variants).

### 6.9.2 Headline win: lockbox transfer

| Packet | best teams_fake_lockbox at joint FPR=10% |
|---|---:|
| P8A_REFERENCE_STEP5000 | 54.1% |
| P22_AUG_STEP1000 | 46.1% |
| P22_AUG_STEP8000 | 36.7% |
| S1_REDUX_SHORT_STEP600 | 73.9% |
| S2_EARLIER_BASE_STEP200 | 86.8% |
| **S2_EARLIER_BASE_STEP600** | **91.5%** ← chain champion |
| S3_VISO_WEIGHT_STEP500 | 83.8% |

S2 step600 at FPR=10% (joint) is the strongest lockbox-substrate ckpt produced across the entire R13 chain.

### 6.9.3 Champions per suite (joint dev+lockbox FPR=10%)

| Suite | Champion ckpt | recall at FPR=10% |
|---|---|---:|
| visomaster_enhanced_macro_dev | P8A_REFERENCE_STEP5000 | 27.0% |
| deeplive_enhanced_dev | P22_AUG_STEP8000 | 91.9% |
| teams_fake_all_dev | P22_AUG_STEP8000 | 79.5% |
| teams_fake_all_lockbox | S2_EARLIER_BASE_STEP600 | 91.5% |

**No single ckpt wins all four.** The user's "90% across the board at FPR=10%" target is unreachable on any single ckpt.

### 6.9.4 Min joint FPR floor for ≥ 90% recall (per ckpt × fake suite)

(NaN means recall never reaches 90% on the [0.005, 0.50] joint-FPR sweep.)

| Ckpt | viso | deeplive | teams_fake_dev | teams_fake_lockbox |
|---|---:|---:|---:|---:|
| P8A | NaN | 0.335 | 0.290 | 0.235 |
| P22 step1k | NaN | 0.380 | 0.440 | NaN |
| P22 step4k | NaN | 0.105 | NaN | NaN |
| P22 step8k | NaN | **0.085** | 0.375 | 0.430 |
| S1 step600 | NaN | NaN | NaN | NaN |
| S2 step600 | NaN | NaN | NaN | 0.090 |
| S3 step500 | NaN | NaN | NaN | 0.155 |

(**Viso unreachable for every ckpt up to FPR=50%.** Deeplive reachable on P22 step8k at 8.5% FPR. Teams_fake_lockbox reachable on S2 step600 at 9% FPR.)

### 6.9.5 Class_separation ranking (training-side health proxy)

| Packet | peak class_sep | peak step | val_holdout AUC at peak |
|---|---:|---:|---:|
| P22 (8000-step) | 5.54 | 650 | 0.9903 |
| **S3** | **4.97** | 550 | 0.9885 |
| S2 | 4.89 | 200 | 0.9853 |
| S1 | 3.91 | 600 | 0.9870 |

Class_separation peak does NOT predict scorecard outcomes — S3 has the highest class_sep among the Sx batch but the worst scorecard performance (lost lockbox lead, no viso lift).

### 6.9.6 Falsifier verdicts summary

| Packet | F-A class_sep | F-B viso recall | F-C dor invariance | Score | Verdict |
|---|:-:|:-:|:-:|---:|---|
| S1 (target 4.0/24%/0.43%) | FAIL (3.91) | FAIL (18%) | unchecked | 0/3 | **FAILED** |
| S2 (target ≥S1/30%/0.43%) | PASS (4.89) | FAIL (12%) | unchecked | 1/3 | MIXED |
| S3 (target 4.0/30%/0.43%) | PASS (4.97) | FAIL (14%) | unchecked | 1/3 | MIXED |

(F-C dor invariance was specified but not directly probed for the Sx batch due to time pressure. The S2 lockbox_real_FPR of 0.001 = 1/1361 indirectly confirms strong dor preservation.)

---

## Section 7 — Cross-cutting: visomaster_enhanced_macro_dev recall through the chain

A single column extracted from each scorecard, contract-selected τ:

| Run | Selected τ | visomaster_enhanced_macro_dev recall | Source |
|---|---:|---:|---|
| P8A_REFERENCE_STEP5000 (D 2026-05-02) | 0.991 | **0.0109** (6/550) | `d_results/promotion_contract/selected_threshold_scorecard.csv` |
| P14_FT step500 (2026-04-29) | n/a | 0.0564 | `policy_reruns_2026-04-29_floor_0p70` |
| P14_FT step1500 | n/a | 0.0564 | same |
| P14_FT step3000 | n/a | **0.1673** (highest measured) | same |
| MCLIOEXB_STEP500 (jitter, 2026-04-30) | 0.975 | 0.0127 | `scorecard_mclioexb_2026-04-30` |
| P16_DATA_AXIS_STEP500 (2026-04-30) | 0.988 | 0.0073 | `scorecard_p16_data_axis_2026-04-30` |
| P18T_GRL_TREATMENT_STEP4000 (D 2026-05-02) | 0.994 | 0.0182 | `d_results/promotion_contract` |
| P18C_NO_GRL_CONTROL_STEP4000 (D 2026-05-02) | 0.995 | 0.0145 | same |
| **P22_AUG_STEP1000** (P22 scorecard 2026-05-02) | 0.979 | 0.0036 | `analysis/p22_eval_2026-05-02/scorecard_data/promotion_contract` |
| **P22_AUG_STEP4000** (same) | 0.841 | 0.0055 | same |
| **P22_AUG_STEP8000** (same) | 0.524 | **0.0436** (4× P8A — first time viso recall rises non-trivially at calibrated τ) | same |

**The viso recall has not exceeded 0.17 at any contract-equivalent τ across any P-series run.** It briefly reached 0.51 at τ=0.5 in P16 step 6000 (per memory; no scorecard in tree), but the deployment-τ collapse is universal.

---

## Section 8 — Cross-cutting: contract scorecard authority

(Source: `MEMORY.md` index entry `project_promotion_contract.md`; full text not re-quoted here.)

The contract is authoritative for promotion. Its lexicographic τ-policy minimizes `dev_primary_real_fpr` first, then maximizes `dev_fake_macro_recall`. Trainer-side `value_composite` is NOT promotion-grade.

Current contract config (per `arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml` and `analysis/p18_probe_2026-05-01/d_results/promotion_contract/promotion_contract.json`):

```
target_fake_recall_min:    0.7
target_real_fpr:           0.02
target_stress_fpr:         0.05
dev_fake_suites:           [teams_fake_all_dev, visomaster_enhanced_macro_dev, deeplive_enhanced_dev]
dev_real_stress_suites:    [teams_real_poor_quality_dev, teams_real_lighting_extreme_dev]
dev_real_suite:            teams_real_all_dev
lockbox_fake_suite:        teams_fake_all_lockbox
lockbox_real_suite:        teams_real_all_lockbox
```

---

## Section 9 — Eval suite manifest contents (verified 2026-05-02)

(Source: this session's audit of `arena/manifests/teams_target_domain_manifest_2026-04-23_with_dor.json`.)

**`visomaster_enhanced_macro_dev` slice** (n=550, all `label: fake`, all `split: dev`):
- All 550 videos have `method: "visomaster_enhanced_macro"` and `identity_key: "visomaster_enhanced_raw"`.
- All `frame_paths` start with `gs://teams-faces-data-test-2914-fake-4420-real-feb-28/fake/visomaster_enhanced_raw__*.png`.
- The bucket name (`teams-faces-data-test-...`) is the Teams-recapture bucket.
- This is the conjunction (enhancer-pass viso × Teams transport).

The training source `combined_paired.visomaster_teams_enhanced` (in code at `data/sources/combined_paired.py:761`, `data/sources/visomaster.py:1044`) targets the same conjunction. As of P18 yaml: `enabled: false`. As of P16 yaml: `enabled: true` with fw=2.0 (no lift). As of xan4dfto: `enabled: true` with fw=8.0 (collapse). As of `R13_P14_DATA_FIX.yaml` (current tree): `enabled: true` with fw=8.0 (drafted-not-launched).

---

## Section 10 — What's installed in code but never tested in a launched packet

(Source: this session's audit of `data/sources/combined_paired.py` toggles.)

- `combined_paired.visomaster_enhanced.enabled` (clean enhancer-pass, no Teams) — wired (`combined_paired.py:4187-4220`) but `false` in P14_DATA_FIX (yaml line 195) and absent from other recent yamls.
- `combined_paired.proper_visomaster_clean.*` / `combined_paired.proper_visomaster_teams.*` (the 2026-04-19 proper-data wave; 705 paired ids per memory `project_visomaster_hints_lanes_bad_data.md`) — buckets 8/9 RESERVED in method-domain map, no yaml in tree enables them.
- `visomaster_teams_v2_companion` (1994/997 ids per memory) — referenced as valid Teams-transport substrate; not enabled in any P-series yaml in tree.

---

## Section 11 — Open uncertainties (where this ledger admits ignorance)

- **Vertex job IDs for P8A, P14_FT, P14_jitter, P14_DATA_FIX(xan4dfto), P15, P16**: not preserved in the in-tree handoffs/memory. Recoverable from W&B / Vertex API but not done in this session.
- **Whether P15 (`w5tky6ss`) ever ran a contract scorecard**: no scorecard dir found in `analysis/`. Trainer-side `value_composite=0.516` is the only number.
- **The 2026-04-29 floor=0.70 scorecard's `visomaster_enhanced_macro_dev` column**: present for P14_FT rows but absent for P14_jitter / P16 rows in the same scorecard? — no; verified with `csv.DictReader` that the column exists in all `selected_threshold_scorecard.csv` files but row-level values may be omitted in cross-summary tables. The scorecard CSVs themselves are authoritative.
- **Whether `R13_P14_FT_FROM_P8A.yaml` and the actual `xan4dfto`-producing yaml are byte-identical**: cannot verify without the run's archived yaml. Memory + viewer config are consistent that xan4dfto used "P14 bundle + visomaster_teams_enhanced fw=8.0" but the exact yaml is unrecoverable from current tree.
