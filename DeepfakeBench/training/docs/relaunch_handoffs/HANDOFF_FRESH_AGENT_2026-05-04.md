# Handoff — Fresh Agent Engagement, 2026-05-04 (pre-Job-B)

**Author:** prior agent (2026-05-04 evening)
**Audience:** you, a fresh ML researcher / engineer picking up this work cold.
**Time budget:** the user (Roee) wants to land a deployment-grade detector in the next few days.
**Your role:** not to take dictation from this document. Treat me as a colleague who has been deep in this for one session, has good data on disk, and probably has subtle biases you should question. The user has explicitly asked that you investigate independently.

---

## 0. How to use this handoff

This document is a map, not a verdict. Below you will find:
- A goal statement (with one **NEW** constraint added today that should change your search space).
- A domain taxonomy that is essential to interpret any number you read elsewhere.
- A factual summary of the 14 CPU diagnostics + 1 in-flight GPU job that landed today.
- Four candidate checkpoints described **symmetrically** (no leader designated).
- A list of directions that are closed (don't propose those without articulating a structural difference) and a list of directions that remain genuinely open.
- Operational rules you should follow.
- A section listing what I have deliberately **not** included (my opinions, recommended actions, etc.) so you can choose to consult me explicitly if you want them.

**Bias warning — read this before reading anything else.** The prior agent (me) anchored on `P8A` as "the leader" multiple times in this session, and was pushed back by the user for it. That was inertia from `P8A` being the most-cited checkpoint in past memory entries, retros, and packet docs — its name appears more often in `~/.claude/.../memory/MEMORY.md` than any other checkpoint, which primes any reader toward treating it as canonical. The data does not support that framing as an unconditional one. Do not read `MEMORY.md` first; if you do, read it with active skepticism toward "leader" framings.

**Recommended read order** (each step has a deferred-judgment-until-after-this-step gate):

1. This handoff (sections 1–11).
2. `analysis/job_8_anchor_audit_2026-05-04/anchor_decision_matrix.csv` — the symmetric 3-checkpoint × 4-lens decision matrix. Read the numbers; do not read any FINDINGS_INTERPRETATION until you have your own first reaction.
3. `analysis/cpu_followups_2026-05-04/FINDINGS_FACTS.md` — broader 14-section facts dump from a prior CPU pass; identifies the 3 candidate checkpoints by name and gives baseline numbers.
4. The 14 individual `FINDINGS_FACTS` docs from today's diagnostic pass (paths in §4).
5. `docs/relaunch_handoffs/AGENT_GUIDE_2026-05-02.md` — operational rules; this is load-bearing.
6. `docs/relaunch_handoffs/PSERIES_FACTS_2026-05-02.md` — the broader R-chain ledger.
7. `~/.claude/.../memory/MEMORY.md` — read **last**, with skepticism toward "leader" / "winner" framings. Each project memory entry has a `How to apply` section; trust the data, not the language.

If a claim in this handoff conflicts with what you read in the FACTS docs, the FACTS docs win. Flag the conflict to the user.

**Disclaimer on prior framings (concrete examples of past wrong reads).** Use these as evidence that this domain has surprised every prior agent and will likely surprise you too:

- Job 6 (frozen-feature linear probe) hit AUC=0.97 predicting `caught/uncaught` on viso fakes. The prior agent (me) read this as "encoder has the signal, head-only retrain will lift recall." Job 7 retrained 6 candidate heads and refuted that claim — every retrained head over-fired on lockbox reals at 80–92% FPR. Substrate-invariance is a trained property of the existing head, not a frozen-feature property.
- Job 1 (suite composition audit) and Job 11 (identity audit) referenced viso as "~22 video sequences" and "2 identities" respectively. Job 13 reconstructed grouping from filename patterns and corrected this to **8 source sequences × {raw, teams} = 16 derived videos**. Job 11's `__s\d+`-token approach over-counted.
- The "viso ceiling 27% unbroken across 13+ packets" finding was treated as a model-capacity ceiling for weeks. Job 14 cleaned the eval substrate (drop 6 chronic-FP-prone identities + low-resolution frames + no-face frames) and found viso recall jumps to 67–78% per checkpoint with no model change. The "ceiling" was substrate pollution.
- Memory entry `project_p18_method_grl_does_not_bite_2026-05-01.md` was annotated `⚠ ORIGINAL VERDICT WAS WRONG` after a corrective probe. The pattern is: confident verdicts on this project keep needing revision. Stay tentative.

If your work follows this pattern (and it likely will), say so in your own handoff at the end. The user values intellectually honest pushback over confident wrong answers.

---

## 1. The goal

Roee is building a deepfake detector for Microsoft Teams deployment. Three pillars (memory: `project_success_criteria.md`):

1. **Fake recall on target methods** — visomaster, deeplive, teams_capture (and ideally df40). The deployment goal: cross 90% fake recall across the board.
2. **FPR < 5%** in production; 10% is currently acceptable as an interim.
3. **Robustness across lighting, camera, codec, color** — not just headline numbers on the dev pool.

All three pillars are load-bearing. Optimizing one at the expense of others is not the goal.

### NEW constraint (added 2026-05-04 evening) — lightweight deployment

The detector must run on a **lightweight** software stack. Practical implications you should treat as binding unless explicitly relaxed by the user:

- **L14-class backbones are out** for production. The `E3_6600` checkpoint (L14 scratch + CE + aug) remains useful as a diagnostic reference but is not a production-deployment candidate. B16-class (or smaller) only.
- **Ensemble deployments are out.** A 3-model deployment (e.g., P8A + E2B + E3 with score fusion) is structurally not viable in this stack. Single-model deployment is the constraint. (Job 12 also showed label-free ensembles across these 3 specific checkpoints don't break the operational ceilings — see §4.)
- **Runtime input-quality gating is on the table as a deployment lever.** If you can identify cheap-at-inference input-quality features (frame size, sharpness, brightness) that:
  1. Are easily measurable on the input frame at deployment time,
  2. Have **extreme-enough** values when the model fails, that
  3. The "low-bandwidth conditions" framing is defensible to a stakeholder,
  then the detector can either reject those inputs upfront or apply quality-conditional decision rules. This is a **new strategic direction** opened by the user today and has not yet been tested.

### Auxiliary product option

Memory `project_deeplive_specialist_option.md`: Roee may also ship a **deeplive-only specialist detector** (clean AND enhanced variants) as a separate product, in addition to or instead of the unified detector. Don't ignore this branch when reading per-method numbers — `E2B_3200` already gets 94% deeplive recall on dev, suggesting that specialist branch is closer to shippable than the unified one.

### Data-creation budget

Creating new training or eval data is expensive and slow. The user explicitly asks: **use what you have**. Inventory of what's available is in Job 5 (`analysis/inventory_audit_2026-05-04/`). Re-evaluating existing checkpoints on already-collected-but-unwired data (e.g., the `visomaster_enhanced_v2` bucket — see §3) is much cheaper than collecting new data.

---

## 2. Domain taxonomy — read this before any number

This is the single most-likely-to-trip-you-up section. Past agents (including me) have written reports that are technically correct but type-confused because of taxonomy errors.

### Methods are creators × enhancement × transport

- **deeplive** and **visomaster** are face-swap *creation tools*. Each has two production variants:
  - **clean / none** — lower face quality, native creator output.
  - **enhanced** — artificially upscaled, sharpened, smoothed (using GFPGAN, codeformer, GPEN, etc.).
- **df40** is a third creator family (other deepfake methods).
- **teams** is **NOT a creator**. It is a *transport pipeline* — the Microsoft Teams capture / codec stack. Any source content (real footage, deeplive-fake, visomaster-fake, df40-fake) can be passed through teams, picking up codec / capture / compression artifacts.

Implication: a frame can be `(visomaster, enhanced, teams-passthrough)` or `(deeplive, clean, no-passthrough)` etc. Eval suites named `*_enhanced_*` are the *enhanced* variant only — the clean variant may not be in your eval substrate at all.

### What's actually in the production scorecard substrate

Per Job 1 (`analysis/suite_composition_audit_2026-05-04/FINDINGS.md`):

- `visomaster_enhanced_macro_dev` (550 frames) is enhanced viso only. Per Job 13 + Job 4: it is **8 source sequences × {raw flat-upload, teams transport} = 16 derived videos**. The "raw" subtype is post-creator pre-transport; the "teams" subtype is creator → teams transport. There is no clean (non-enhanced) viso in this suite.
- `deeplive_enhanced_dev` (545 frames) is enhanced deeplive only. No clean deeplive is currently scored anywhere in the production substrate.
- `teams_fake_all_dev` (3,039 frames) is a **transport mixture** — 40.5% (1,230 frames) are non-teams-captured (flat-upload deeplive_enhanced + visomaster_enhanced + a Xiang clip); only 59.5% (1,809 frames, 12 identities) are actual teams-captured sessions. The headline "teams_fake_all recall" mixes two distinct signals.
- `teams_fake_all_lockbox` (425 frames) collapses to **2 sessions / 2 identities** (`Cam_Test__s33` + `PC_Generator__s15`). Lockbox-fake recall headlines are brittle.
- Real-side suites: `teams_real_all_dev` (4,564 reals), `teams_real_all_lockbox` (1,418 reals), plus stress slices.

Lockbox is harder than dev and is the user's preferred shortcut-detection signal. Memory: `project_lockbox_identity_looseness.md` — same person can appear as both real and fake across dev/lockbox; identity leakage is intentional eval design.

### The `visomaster_enhanced_v2` bucket (eval-substrate expansion candidate)

Bucket: `gs://visomaster-enhanced-face-cropped-v2/`. Manifest: `arena/manifests/visomaster_enhanced_v2_manifest_2026-04-13.json`. Per Job 5 + user eyeball confirmation 2026-05-04 (memory `project_v2_substrate_is_dor_diverse_swap.md`):

- v2 is **Dor in his standard setup × 16 swap-model families × small-N swap-target masks**.
- v2 is **NOT identity-fresh** (Dor is heavily in training as both real and fake; chronic FP-prone identity).
- v2 IS **swap-model-fresh** — at least 4 swap-model families (`inswapper128_*`, `inswapper_*_codeformer`, `inswapper_*_gfpgan`, `inswapper_*_gpen*`, `instyle_swapper_256_v[A-C]`, `simswap`) do not appear by name in the v1 manifest's training methods.
- v2 is visually less sharp than v1 enhanced (consistent with v2 being raw swap output, no enhancer pass).
- Total: 2,073 frames across the 16 method folders.

v2 is the cleanest "$0 cross-swap-model generalization test" available. Naive scoring with the existing P8A/E2B/E3 heads is part of Job 7's already-completed work — see `analysis/job7_head_retrain_2026-05-04/v2_held_out_per_family.csv` for per-family breakdowns.

---

## 3. Candidate checkpoints — described symmetrically

Four candidates have meaningful evidence as of this handoff. The lightweight constraint takes one off the production-candidate list (E3); it remains a useful diagnostic reference. Job B (in flight, see §6) will add four more.

| Alias | Backbone | Training recipe | W&B run | Where the canonical pth lives |
|---|---|---|---|---|
| **P8A** | B16 (CLIP-init) | 5-stage FT chain CLIP → R12g → RLP6_04 → RLP7_02 → P8A | `9lmvb5b4` | `gs://training-job-outputs/phase2_round11/.../` (search W&B run for path) |
| **E2B_3200** | B16 (CLIP-init) | scratch + CrossEntropy + heavy aug | `rmat8lwx` | `gs://training-job-outputs/phase2r13_experiments/.../top_n_step3200_auc0.9863_eer0.0310.pth` |
| **E3_6600** | **L14** | scratch + CE + heavy aug | `jzroefab` | `gs://training-job-outputs/phase2r13_experiments/.../top_n_step6600_auc0.9972_eer0.0093.pth` |
| **P22_step1000** | B16 (P8A FT) | P8A + pipeline_randomization aug, step 1000 | (cited in `project_p22_cpu_followups_reframe_2026-05-02`) | search the W&B run |

**E3_6600 is OUT for production deployment** under the lightweight constraint. Keep it as a reference for "what L14 on this data substrate gives" — Job 14 results show E3 reaches 78% viso recall on the cleaned substrate (highest of the three), so it tells you what the upper-bound looks like if L14 were viable. But you should not propose E3 as the production anchor.

### Symmetric per-property summary (from Job 8 + Jobs 7/9/11/12/13/14)

Read these as **observations**, not rankings.

| Lens | P8A (B16) | E2B_3200 (B16) | E3_6600 (L14) | Source |
|---|---|---|---|---|
| Viso enhanced recall @ FPR=10%, F0 substrate | 0.269 | 0.084 | 0.138 | `cpu_followups_2026-05-04/outputs/11_threshold_relaxation_curves.csv` |
| Deeplive enhanced recall @ FPR=10%, F0 substrate | 0.424 | **0.939** | 0.908 | same |
| Teams_fake_dev recall @ FPR=10% | 0.699 | 0.794 | **0.799** | same |
| Teams_fake_lockbox recall @ FPR=10% | 0.541 | 0.831 | **0.944** | same |
| Substrate-invariance Δmean (real_dev vs real_lockbox; smaller=more invariant) | 0.0382 | **0.0162** | 0.0655 | Job 8 `substrate_invariance_per_ckpt.csv` |
| Substrate-invariance Δp50 | 0.0090 | 0.0122 | **0.00025** | Job 8 |
| Per-substrate τ-cal lift @ FPR=10% on lockbox-fake | **+0.214** | +0.075 | +0.049 | Job 8 `calibration_lift_per_ckpt.csv`; reproduces Job 7's "21pp" finding |
| IQ-shortcut CV AUC on viso (higher = more IQ-bound) | 0.7474 | **0.8665** | 0.7651 | Job 8 `iq_shortcut_per_ckpt.csv` |
| Unique viso fake catches in DISAGREE bands | **122** | 1 | 56 | Job 9 `unique_catches_viso_per_ckpt.csv` |
| Unique FPs on `teams_real_all_dev` | 312 | 63 | 389 | Job 9 `unique_fps_real_per_ckpt.csv` |
| Chronic-6 share of unique FPs | 86% / 83% | 66% / 44% | **93% / 97%** | Job 9 |
| Viso recall under Job 14 F4-cleaned substrate @ FPR=10% | 27% → **67%** | 8% → 31% | 14% → **78%** | Job 14 `per_ckpt_filter_recall_lift.csv` |
| Viso recall under Job 14 F1-only (drop chronic-6 only) | TBD if you want it | TBD | TBD | rerun Job 14 with F1 isolation if needed |
| Score-distribution shape on real-class | mean ≈ p50 close (head-substrate-invariant per Job 7) | substrate-invariant by Δmean | very compressed: p10–p90 = 0.006–0.85 | Job 8 + Job 7 |

Each candidate has ckpt-specific strengths and ckpt-specific failure modes. The "no single anchor wins" framing is **factually correct on a per-suite basis**: P8A, E2B, E3 each own a different fake suite at FPR=10% on the F0 substrate. The Job 14 cleaned-substrate numbers re-shuffle the ordering somewhat but no checkpoint dominates uniformly on the cleaned substrate either.

P22_step1000 is plausibly a candidate but has not been put through the full Job 8 / Job 9 / Job 11 lens. Memory `project_p22_cpu_followups_reframe_2026-05-02.md` describes its known properties; if you want it included in the symmetric audit, you would need to find or extract its frozen scores.

---

## 4. Today's diagnostic substrate (Jobs 1–14)

These ran 2026-05-04 between morning and evening. All facts. Each has its own `FINDINGS_FACTS` doc and a separate `FINDINGS_INTERPRETATION` doc — read facts first, interpretation second. All output paths are under `analysis/<job-name>_2026-05-04/`.

| Job | Question answered | Load-bearing fact | Output dir |
|---|---|---|---|
| 1 | What's actually in each eval suite? | `teams_fake_all_dev` is 40% non-teams-captured; clean viso/deeplive variants not scored anywhere in production substrate; `teams_fake_all_lockbox` collapses to 2 sessions | `suite_composition_audit_2026-05-04/` |
| 2 | Per-cell recall by (method × enhancement × passthrough) | Per-method specialization is real and Venn-supported (P8A=viso, E2B=deeplive, E3=lockbox-fake-AUC); 3-way union recall on viso enhanced caps at 33.8% on F0 substrate | `per_cell_catch_2026-05-04/` |
| 3 | Is the FP tail concentrated or diffuse? | Concentrated. Top-5 of 24 dev identities own 80–95% of FPs per ckpt; 6 chronic offenders dominate every ckpt; 90.1% of P8A FPs come from `min(W,H)<200` | `fp_tail_characterization_2026-05-04/` |
| 4 | Is viso → teams a transport-specific signature or IQ-compounded? | IQ-compounded (HIGH confidence). Transport shifts image quality in webcam-failure direction; sharpness-coefficient flips +1.00 (raw) → −0.11 (teams) | `viso_subtype_analysis_2026-05-04/` |
| 5 | What unwired data exists in inventory? | 49 candidate cells; only 16 are not in P8A training (visomaster_enhanced_v2 ~2073 frames); proper viso teams + enhanced teams exist (~23k frames) but are P8A training; df40 → teams does not exist; pre-RLP6_04 ckpts have full proper viso pool clean held-out | `inventory_audit_2026-05-04/` |
| 6 | Is `caught/uncaught` linearly separable in frozen P8A features? | Yes (AUC=0.969). CLIP-B16 raw also high (0.870); FT chain sharpens by +0.099. **Caveat: Job 7 refuted the practical implication** — see Job 7 below | `viso_frozen_feature_probe_2026-05-04/` |
| 7 | Does a head-only retrain on frozen P8A features lift viso recall? | **REFUTED.** All 6 retrained heads over-fire 80–92% on lockbox reals at dev-calibrated FPR=10%; substrate-invariance lives in the trained head, not the frozen encoder. **Side finding: P8A has +21pp recall available via per-substrate τ calibration alone, no retrain.** | `job7_head_retrain_2026-05-04/` |
| 8 | Multi-checkpoint anchor-selection audit (4 lenses) | No checkpoint dominates uniformly. See per-lens table in §3. Substrate-invariance ordering is metric-dependent. | `job_8_anchor_audit_2026-05-04/` |
| 9 | Cross-checkpoint per-frame disagreement structure on viso | P8A 122 unique viso fake catches vs E3 56 vs E2B 1 in DISAGREE bands; "scratch has more viso headroom" hypothesis refuted; each checkpoint's unique-FPs concentrate on different chronic-identity subsets | `job_9_disagreement_audit_2026-05-04/` |
| 11 | Per-identity FPR per checkpoint | Chronic-6 do NOT fail uniformly. PC_Generator__s22 P8A 0.91 vs E3 0.06 (Δ=0.85). Each chronic identity has at least one checkpoint that handles it cleanly. | `job_11_identity_audit_2026-05-04/` |
| 12 | Can a label-free ensemble break the ceilings? | **NO.** Best non-oracle ensemble on viso = 16.2% < P8A 27%; only teams_fake_dev gets a positive lift (+1.9pp). Oracle ceiling = 77% on viso shows information IS in the streams but no label-free combiner extracts it. Out-of-stream router (using IQ / capture mode / identity features) is the structural lever. | `job_12_ensemble_ceiling_2026-05-04/` |
| 13 | Are viso failures video-bound or scattered? | Viso = 8 source sequences × {raw, teams} = 16 videos. No video reaches catch_rate >0.90 for any checkpoint. Failures are concentrated near 0 with a heavy partial-catch tail to ~0.78. Teams subtype is uniformly hard with E2B↔E3 r=0.94; raw subtype is where ckpt-disagreement lives. | `job_13_per_video_catch_2026-05-04/` |
| 14 | Does cleaning the eval substrate break the viso ceiling? | **YES.** F4 (drop chronic-6 + lowres + no-face): P8A viso 27→67%, E3 viso 14→78%, E2B viso 8→31%; FPR collapses to 0.53–3.16% on cleaner reals. Chronic-6 axis alone explains 96–108% of the FPR drop. | `job_14_substrate_clean_simulation_2026-05-04/` |

### The two findings to dwell on

**(a) Job 14 — the viso "ceiling" is a substrate-pollution artifact.** This is the single largest reframe of the day. The "27% viso recall ceiling unbroken across 13+ packets" claim (memory `project_viso_ceiling_unbroken_10_packets.md`) holds **only on the F0 substrate**. On a substrate where the 6 over-represented chronic identities + small-source-image frames are dropped, every existing checkpoint hits 67–78% viso recall with no model change.

Critical caveats from Job 14's interpretation (do not ignore):
- F4 throws away 54% of dev reals. F1-only (drop chronic-6 only, keep ~78% of reals) gets ~80–100% of the F4 win for E2B and E3.
- The F4-recalibrated τ is much smaller (~0.01–0.07 vs F0's ~0.5–0.85). The decision boundary in score units is narrower; real-world deployment noise has less margin.
- The lowres<200 filter is **not** deployment-defensible — production webcams produce sub-200 frames in low-bandwidth conditions. F1 (chronic-6 alone) is the more defensible filter.
- 12.2% of dev frames are parquet-uncovered; F4 numbers are conservative if those polluters disproportionately sit in the uncovered slice.

**(b) Job 12 — label-free ensembling does not break the ceilings.** Every score-fusion strategy across (P8A, E2B, E3) leaves viso recall ≤ 16% at FPR=10% — worse than P8A alone (27%) on F0. The information is in the streams (oracle ceiling = 77%) but no label-free combiner extracts it. The structural lever is an out-of-stream router (using IQ / capture mode / identity), not another correlated score stream. Combined with the lightweight constraint, multi-model deployment is doubly off the table.

---

## 5. Closed directions — don't loop here without articulating structural difference

The user has been burned by agents who proposed already-refuted experiments. AGENT_GUIDE_2026-05-02 was written precisely after a prior agent proposed P19 (a relaunch of P14_DATA_FIX) without checking that the same lever had been pulled twice and failed both times. Read `docs/relaunch_handoffs/AGENT_GUIDE_2026-05-02.md` before proposing any new experiment.

The following classes of experiment have been run and are **closed for now** unless you can explicitly articulate what's structurally different about your proposal:

| Class | Concrete refutations | Memory entries |
|---|---|---|
| Architecture variation (B16 FT vs B16 scratch+CE vs L14 scratch+CE) | E1 / E2b / E3 — viso ceiling holds at F0 substrate, only deeplive moves | `project_e2b_breaks_deeplive_ceiling`, `project_l14_does_not_break_viso_ceiling` |
| Aug curriculum stacking | P14 anti-shortcut bundle (net-negative); P22 + S1/S2/S3 (no viso lift); E1 (lost everything except teams) | `project_face_scale_jitter_load_bearing`, `project_s1_s2_s3_2026-05-03` |
| Data-axis fw weighting (force-up viso training weight) | P14_DATA_FIX (xan4dfto, fw=8.0, collapsed); P16_DATA_AXIS (rmic6wrc, fw=2.0, doesn't promote) | `project_data_axis_lever_pulled_twice_no_lift` |
| GRL / domain-confusion | P15 static λ=0.20 (works structurally, weaker than jitter); P18 12-class method-conditional (defensive against FT regression, not additive over P8A) | `project_p18_diagnostics_complete_2026-05-02` |
| Head-only retrain on frozen features | Job 7 (today) refuted; substrate-invariance destroyed | `project_job7_head_retrain_REFUTED_2026-05-04` |
| Label-free ensemble across (P8A, E2B, E3) | Job 12 (today) refuted | `project_job12_ensemble_ceiling_2026-05-04` |
| Calibration-loss as primary lever | Job 3 (today) refuted via FP-tail concentration; FP tail is concentrated on 6 identities, not diffuse | `project_eval_substrate_reframe_2026-05-04` |

**Trainer-side metrics that are NOT promotion signals:**
- `value_composite` (memory `project_promotion_contract.md`).
- Train AUC (memory `project_train_auc_not_valid_promotion_signal.md` — P22 trained AUC dropped 0.99→0.94 but operational fake recall went UP 3×).
- `class_separation` peak (memory `project_class_sep_not_predictive.md` — S3 had the highest peak in the chain but worst scorecard).

Promotion goes through the lockbox-anchored contract scorecard or it doesn't go.

---

## 6. Open directions — genuinely untested, with what each would test

These are starting points for your own thinking. None is endorsed; each has trade-offs you should evaluate.

| Direction | What it tests | Cost | Notes |
|---|---|---|---|
| **Eval substrate redesign** (drop chronic-6, recalibrate τ, re-evaluate every existing checkpoint) | Whether the F0 ceilings are substrate-bound or model-bound; whether the production-honest picture changes the candidate ordering | $0 CPU | Job 14 implies this is the highest-leverage move. F1-only (chronic-6 drop) is more defensible than F4 (which adds the controversial lowres filter). The agent should run this separately if not already done. |
| **Per-substrate τ-calibration policy** | Free 21pp on P8A's lockbox-fake recall (Job 7 side finding). Whether E2B and E3 also benefit (Job 8 says they get +7.5pp and +4.9pp respectively, smaller). | $0 CPU + deployment-time policy change | The "calibrate τ on lockbox-shaped reals at deployment" idea. Ship-as-policy without retraining. |
| **Runtime input-quality gating** (NEW, lightweight-aware) | Whether IQ features (laplacian_var, source resolution, luma percentiles) are extreme-enough at model failure to defensibly reject low-bandwidth inputs at inference time | $0 CPU diagnostic | This is the new strategic direction the user opened today. See §7 for the question structure. |
| **Hard-negative mining on chronic-6 identities** (training-side) | Whether explicit oversampling of `bla_bla_chow` + `bla_bla_chow__s2` + `PC_Generator__s22/s45` + `roy_d` + `Q__s6` in training breaks the FP-tail concentration | ~$25–50 GPU | Training-side. Different from Job 7 (which retrained the head on frozen features and failed). This retrains the encoder + head with the chronic-6 upweighted. |
| **Wire visomaster_enhanced_v2 + viso → teams as standalone scorecard suites** | Cross-swap-model generalization on a known-difficult identity (Dor); deployment-relevance of the existing "viso enhanced" headline | $0 yaml change + small re-evaluation | Job 5 inventory found these are wireable; Job 7 already produced v2 scores per swap-family. |
| **Pre-RLP6_04 trajectory test** | Whether the FT chain itself created the ceiling, or whether earlier checkpoints have similar properties on the same eval | Job B (~$8–12, in flight) | Result will land before you take the next deployment-relevant decision. |
| **Wire `teams_manifest.breakdown_2026-04-07.yaml`** | Per-session per-identity recall as first-class scorecard rows | $0 yaml change (already merged in working tree per Job C; not yet committed) | Improves observability of every future scorecard run. |

---

## 7. The runtime input-quality gating direction (the new lever)

The user opened this direction today. The question structure:

For the model to ship as a single B16 detector with deployment-time runtime input-quality gating, you need to verify:

1. **Are there cheap-at-inference IQ features that correlate with model failure?** Job 4 showed yes on viso: 7 IQ features (luma_p10/p25/p50/p90, laplacian_var, sobel_edge_mean, saturation_mean, skin_frac) explain caught/uncaught at 72–76% CV accuracy on raw subtype, 76% on teams subtype, with sign-stable predictors.
2. **Are the failure-conditions extreme enough to defensibly reject inputs?** This is the gate. If the model fails when laplacian_var is in the bottom 5% of the distribution, and 5% of production frames also fall there, you can defensibly say "those frames are outside operational envelope" and reject. If model failure straddles the median, you cannot. **This has not been quantified.**
3. **Are quality-conditional decision rules viable?** Given a quality bucket (e.g., laplacian_var quartile), can you set a per-bucket τ that achieves higher recall on high-quality buckets while raising τ on low-quality ones? This is a calibration-policy intervention, not a model change.

A first CPU-only experiment to inform this direction: take the candidate B16 checkpoints (P8A, E2B, P22_step1000), the existing per-frame scores in `cpu_followups_2026-05-04/raw_reports/`, and the IQ tags in `analysis/score_distribution_2026-05-02/outputs/crop_attributes.csv` (extend coverage to deeplive / teams suites if feasible). Compute:

- For each (checkpoint, suite, IQ feature, IQ-quartile), recall and FPR at FPR=10% τ.
- Identify the IQ-quartile boundary at which recall collapses below 50% (or whatever you decide is the operational floor).
- Cross-check whether the "rejected" boundary (frames below this IQ floor) is a cohort that is (a) rare in production-realistic distributions and (b) a defensible exclusion class.
- Compare across checkpoints — the ckpt that has the cleanest quality-conditional behavior (high recall on high-quality buckets, robust failure on low-quality buckets) is the one most amenable to this deployment policy.

You decide whether this is worth doing. The user has not committed to it.

---

## 8. What's in flight — Job B

Vertex job `8771378543436234752`, region us-east1, submitted 2026-05-04 evening, ETA ~3h. Outputs at `gs://training-job-outputs/test_results/job_b_pre_rlp604/job-b-pre-rlp604-2026-05-04/`.

Tests pre-RLP6_04 checkpoints on the held-out HDTF proper viso pool:

| Alias | GCS path |
|---|---|
| R12g_step14000 | `gs://training-job-outputs/phase2r12_experiments/0xxqwhxg/top_n_effort_20260310_step14000_auc0.9930_eer0.0253.pth` |
| RLP3_05_step2500 | `gs://training-job-outputs/phase2r13_experiments/w92amaaa/value_composite_effort_20260422_step2500_auc0.9895_eer0.0116.pth` |
| RLP5_07_step20500 | `gs://training-job-outputs/phase2r13_experiments/6jwwb526/value_composite_effort_20260423_step20500_auc0.9908_eer0.0304.pth` |
| RLP6_04_step23500 | `gs://training-job-outputs/phase2r13_experiments/h2pdu6i5/value_composite_effort_20260424_step23500_auc0.9942_eer0.0169.pth` |

R12g is a pre-R13 anchor; RLP3_05 and RLP5_07 are pre-RLP6_04 (held-out on proper viso); RLP6_04 is the boundary checkpoint (first ckpt trained on proper viso — its recall on proper viso is the trained-on reference, leaky for this test).

What Job B answers: whether the viso ceiling is FT-trajectory-induced (would reproduce on earlier B16-FT checkpoints) or universal across the chain. What Job B does **not** answer: whether the lightweight constraint changes anything (all 4 ckpts are B16-eligible) or whether substrate-cleaning (Job 14) interacts with trajectory.

Preprocessing-parity caveat: pre-RLP6_04 ckpts are pre-fix `INTER_AREA`; arena scoring is post-fix `INTER_LINEAR`. Documented ~0.7pp systematic drift; below threshold for the directional question Job B is asking. Memory `project_preprocessing_parity_bug` for context.

---

## 9. Operational rules — read AGENT_GUIDE_2026-05-02 fully, but the headline rules

From `docs/relaunch_handoffs/AGENT_GUIDE_2026-05-02.md`:

- **Validate-before-suggest.** Before proposing any packet that includes a hypothesis like "X has never been tested," grep `experiments/phase2_round13/*.yaml` for the toggle, read every yaml that matches, check W&B for whether the run actually launched, check memory for the result. If you find a match, the hypothesis has been tested. Do not propose it without articulating what's structurally different about your proposal.
- **Single-lever discipline.** When stacking multiple anti-shortcut levers, include a single-lever ablation slot in the same packet. Memory `project_face_scale_jitter_load_bearing` shows the P14 bundle was net-negative against jitter@0.50 alone, by 5.7×.
- **CPU-first-then-GPU.** Identify the cheapest CPU diagnostic that would update your beliefs about a proposed GPU experiment. Run it first.
- **US regions only on Vertex** (CLAUDE.md). `us-east1`, `us-west4`, `us-central1`. No `asia-*` or `europe-*` without explicit user authorization. If a region pends >30 min, switch to another US region per CLAUDE.md retry pattern.
- **No /tmp secrets.** WANDB_API_KEY etc. inline on the launcher invocation, never in `/tmp/*.sh` files. Memory `feedback_no_secrets_in_tmp_files`.
- **Don't cancel Vertex jobs without explicit user authorization.** Memory `feedback_no_cancelling_vertex_jobs`.
- **Image rebuild discipline.** Yamls and checkpoint_maps are loaded from `/workspace/` inside the container; new files require `./dev.sh build-prod -y` before the launcher sees them. Auto-bumps VERSION patch. Memory `reference_image_rebuild`.
- **Viewer integration.** `viewer/server.py` + `viewer/templates/index.html` is the user's visual inspection tool. CPU analysis scripts that produce per-frame results SHOULD output viewer-compatible CSVs and SHOULD register as run entries in `viewer/model_dashboard_runs.yaml`. Memory `reference_agent_guide_2026_05_02`.
- **Facts-vs-opinions separation in handoffs.** Past handoffs mixed "we ran X" facts with "we should pivot to Y" opinions, and the next agent inherited both. Separate them. The OPINIONS file should start with a disclaimer listing concrete examples of past wrong framings. See `PSERIES_FACTS_2026-05-02.md` and `PSERIES_OPINIONS_2026-05-02.md` for the template.

---

## 10. State of the working tree at handoff

Useful context for understanding what is and isn't committed.

- **Job C merge** applied to `arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml` (77 lines → 237 lines; +20 net-new sub-suites for per-session breakouts). Validation script passes. **Not committed.** The user authorized the merge; commit when convenient.
- **Job B checkpoint_map** at `arena/checkpoint_maps/teams_target_domain.pre_rlp6_04_baseline_2026-05-04.yaml`. **Not committed.** Currently inside the running Vertex container's image (1.3.253).
- **VERSION** bumped to 1.3.253 by the Job-B image build.
- **Working tree** also has older modifications to `arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml` and `arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json` (manifest paths went absolute-Mac-host vs repo-relative; manifest content rebuilt locally). These are the reason Job B's launch needed `SKIP_IMAGE_CURRENCY_CHECK=1`. Look before committing anything in those files.
- Memory entries created today (auto-loaded in your session): `project_data_domain_taxonomy`, `project_v2_substrate_is_dor_diverse_swap`, `project_eval_substrate_reframe_2026-05-04`, `project_viso_subtype_iq_compounded_2026-05-04`, `project_viso_head_boundary_finding_2026-05-04`, `project_job7_head_retrain_REFUTED_2026-05-04`, `project_chronic_offenders_partition_per_ckpt_2026-05-04`, `project_job9_disagreement_audit_2026-05-04`, `project_job13_per_video_viso_2026-05-04`, `project_job12_ensemble_ceiling_2026-05-04`, `project_job14_substrate_clean_2026-05-04`, plus `project_deeplive_specialist_option`.

---

## 11. Things deliberately NOT in this handoff (transparency)

The user has explicitly asked that you investigate independently and reach your own conclusions. To support that, I have left out:

- **My personal opinion on which checkpoint should be the production anchor.** I have one. The user has heard it. They specifically asked me not to put it in this handoff because they don't want to bias your read of the data. If you want my opinion after you've formed your own first reaction, ask the user — they can relay it.
- **A "first-day action plan" with my recommended sequencing of experiments.** Same reason. The data is on disk; the candidate set is enumerated; the open directions are listed; you decide the priority.
- **A declared verdict on whether the viso ceiling is broken or not.** The data shows it depends on the substrate (Job 14 yes; F0 no). I have not collapsed this into a single claim.
- **A recommended packet design.** Per the operational rules above, packets need single-lever discipline and validate-before-suggest checks. I will not pre-stage a packet you haven't yet thought through.

What I have included is the substrate of facts, the reasoning chains that produced today's reframes, and the load-bearing constraints (lightweight deployment, data-creation budget, three pillars, AGENT_GUIDE rules). Everything else is your call.

---

## 12. First step suggestion (the only one I'm offering)

If you want a starting point you can react against, here is a single suggestion. Treat it as adversarial; reject it if you find a better path.

> **Suggested first step:** before proposing any new experiment, read `analysis/job_8_anchor_audit_2026-05-04/anchor_decision_matrix.csv` and `analysis/job_14_substrate_clean_simulation_2026-05-04/per_ckpt_filter_recall_lift.csv` cold (no FINDINGS_INTERPRETATION first). Form your own first reaction to two questions:
>
> 1. Given the lightweight constraint (single B16 model, no L14, no ensembles), which checkpoint do *you* think is the most plausible production anchor? Cite the specific lens(es) that drive your call.
> 2. Given Job 14's "F4 breaks the ceiling but F4 isn't deployment-defensible" + the new runtime-IQ-gating direction the user opened today, what's the single highest-leverage CPU experiment to run next?
>
> Then, and only then, read FINDINGS_INTERPRETATION docs and check whether your read agrees or disagrees with the prior agent's. Write down where you disagree.

If you want to skip my suggestion entirely, that is also fine. The user has asked for an independent audit.

---

**End of handoff.**

When you produce your own next-handoff document at the end of your session, follow the same facts-vs-opinions separation, list the directions you closed and opened, and include a concrete "examples of my framings that turned out wrong" section in your interpretation file. The pattern of confident verdicts being wrong continues; institutionalize the pushback.

Good luck.
