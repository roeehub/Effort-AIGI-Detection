# State — current rolling snapshot

> **Last refreshed**: 2026-05-10 ~07:00 local. Headline: **T3 (Stage 3) ran 2026-05-09 → 2026-05-10. T3_SLOT1_PERIODIC_STEP2500 is the highest-capability candidate seen across all R13 packets — 79.27% F4 v2 viso recall (vs P8A 67.09%, PA 72.36%) AND 84.97% HDTF `proper_visomaster_enhanced_teams_dev` at FPR-cal 5% (NOT a PA-style v2-bound collapse — step2500 partially generalizes to HDTF, beats step1500 on every HDTF cell). Step2500 fails the F0 v2 dev recall floor (22.95% < 30%) but the failure is a contract-calibration artifact: with chronic-6 in F0 reals, step2500 over-confidently scores Roy_D as fake → auto-τ clips at 0.971 → dev fakes collapse at that τ. T3_SLOT1_PERIODIC_STEP1500 is the secondary candidate that PASSES the F0 floor at 37.6%.**
>
> **Reading carried forward**: under the user's "abstain below IQ threshold" deployment policy (= F4 substrate), step2500 is the strongest deployable candidate. Under strict F0 contract policy (chronic-6 included), step1500 is the only floor-passing T3 candidate; P8A wins lex-policy rank 1 because all T3 ckpts hit ~9.92-9.99% `dev_worst_real_stress_fpr` (target ≤10%; P8A 6.85%). The 9.92% T3 stress FPR is **entirely Roy_D-on-lighting_extreme** — ex-Roy_D, T3 (3.2%) beats P8A (5.0%) on stress; F4 deployment lens makes the regression invisible.
>
> **What this means for the next agent**: deployment can ship step2500 with FPR-calibrated τ on a production-realistic real cohort. The remaining open question is whether to launch a refinement packet (T4 — face_scale_jitter or Roy_D hard-negative mining) before shipping. T4 directions documented in `analysis/cpu_diagnostics_2026-05-09/MORNING_BRIEF_2026-05-10.md` §13. Memory `project_t3_slot1_step1500_lockbox_lift_2026-05-09.md` should be amended (top-line headline now points at step2500, not step1500). Memory `project_data_axis_lever_pulled_twice_no_lift.md` is now refuted twice over (PA + T3_SLOT1).
>
> **Reading order for the next agent**: (1) `MORNING_BRIEF_2026-05-10.md` §15 TL;DR (30-second read), then (2) `packets/T3.md` (full packet doc), then (3) `T3_SCORECARD_FACTS_2026-05-09.md` + `_t3_f4_outputs/*.json` for the raw numbers, then (4) `_t3_hdtf/reports/` + `_t3_hdtf_step2500/reports/` for HDTF per-suite reports.
>
> ---
>
> **Earlier 2026-05-09 13:00 snapshot (preserved for context)**: Stage 2 ALL 3 SUCCEEDED. CPU analytics: 3 probes × 9 ckpts on Dor cohort + Roy_D + L11 features. **Headline: ALL Stage 2 step4500 ckpts regress P8A's chronic-identity Pillar 2 — Roy_D real-FPR at τ=0.5 is S1 99.2% / S2 96.9% / S3 100% vs P8A 45.4%.** L11 cos-distance vs P8A on Roy_D: S2_step500 0.21 → S3_step4500 0.82; Spearman r=0.91 with score regression across n=9 sibling Stage 2 ckpts.
> 
> **Reading (provisional, 2026-05-09 audit-corrected):** Across the levers tested (data-aug-toggle / LR / pair-rank-λ / Fourier-aug / pair-rank+GroupDRO bundle), every FT-from-P8A regime drifts the dor cluster within 500-2500 steps. The strongest reading from this convergent pattern is "FT-from-P8A under the tested regimes drifts L10-L11 dor invariance"; this was previously phrased as "FT itself is the binding constraint" — that universal framing is overstated and has been corrected (see `analysis/stage2_cpu_2026-05-09/STAGE2_SCORE_PROBE_OPINIONS_2026-05-09.md` 2026-05-09 caveats and `project_stage2_all_levers_regress_p8a_2026-05-09.md` Why-section). Untested lever classes: LoRA / parameter-efficient adapter FT (preserves base by construction), hard-output-preservation losses on a reference cohort, multi-objective training with Pillar-2 explicit constraint, frozen-encoder regimes other than head-only retrain.
>
> **L11 anchor-loss FT (option 3a)** remains a candidate next intervention — its targeting is supported by the L11-distance ↔ regression correlation, but the correlation alone is not dispositive (training-step amount is a confounder; alternative anchor layers like L9/L10 / output-side anchor have not been compared). Under the user's 2026-05-09 reframe (production target genuinely unknown; Pillar 3 robustness must come from forgery signal), preserving L11 also preserves the IQ-shortcut representation co-located there — so the anchor lever is not unambiguously net-positive without first establishing that L11 carries a forgery channel separable from the IQ + identity shortcuts. The forgery-signal atlas CPU diagnostic at `analysis/cpu_diagnostics_2026-05-09/` is the cheap test of that.
>
> Promotion contract scorecard NOT recommended for any Stage 2 ckpt — saves $45-60 GPU. FACTS + OPINIONS at `analysis/stage2_cpu_2026-05-09/`.
>
> **2026-05-09 audit-pass artifacts at `analysis/cpu_diagnostics_2026-05-09/`** — under user reframe (production target unknown; robustness must come from forgery signal not other-property shortcuts), four CPU diagnostics ran on the 800-frame triptych for all 12 ckpts (3 reference + 9 Stage 2):
>  - **Forgery-signal atlas** (per-layer × per-signal AUC matrix): all signals (real/fake, is_dor, is_chronic_6, lap_var-high, min_dim-high, face_size-high) extractable at AUC 0.95+ from L11 across all ckpts; L0 already gives 0.72 AUC for real/fake from frozen patch+pos embed alone.
>  - **F4 substrate filter recompute** on the triptych: dropping chronic-6 + is_no_face + min_dim<200 cuts Stage 2 step4500 real-FPR from 0.149-0.163 (F0_full) to 0.042-0.049 (F4) while preserving fake-recall within ≤0.02.
>  - **IQ R² + residual AUC decomp**: LOCKBOX R² varies 12× across Stage 2 ckpts (0.07-0.84); DEV_NO_CHRONIC residual AUC is 0.90-0.95 across all Stage 2 ckpts.
>  - **Per-identity score table**: same-root-identity-different-session sub-identities span 100×-280× score range under same ckpt (`dor_shkedi` 0.298-0.859 vs `dor_shkedi__s16` 0.003-0.080).
> 
> FACTS doc: `analysis/cpu_diagnostics_2026-05-09/CPU_DIAGNOSTICS_FACTS_2026-05-09.md`. No OPINIONS doc written this pass — by user request, conclusions are user-gated.)
>
> **Purpose**: single-page current-state snapshot. Always-current; rolling. Older dated snapshots archived in [`archive/`](archive/) for historical reference.
>
> **How to use this page**: a new agent reads this AFTER `AGENTS.md` and BEFORE the user's specific task. It tells you what's running, what just landed, what's open, and what decision the user might want next. If the wall-clock is materially after the "Last refreshed" date, verify against `git status`, the tail of [`TIMELINE.md`](TIMELINE.md), and the latest [`packets/`](packets/) entry — state moves quickly.
>
> **FRESH-AGENT GUIDANCE (2026-05-10 — START HERE)**: the user has explicitly asked for an independent fresh-agent take on the T3 packet results. Read [`HANDOFF_2026-05-10.md`](HANDOFF_2026-05-10.md) FIRST — it contains the user's framing constraint (production robustness across cameras/lighting/rooms/people), a load-bearing reading order (FACTS docs first, OPINIONS / `packets/T3.md` interpretation only AFTER you form your own view), and the specific deliverable (state your own reading + propose a next step before opening any interpretive doc). Do not skip the production-robustness constraint section — it changes how F4 substrate-cleaning numbers should be interpreted.
>
> ---
>
> **PRIOR FRESH-AGENT GUIDANCE (2026-05-08, preserved for context)**: the IQ-shortcut deconvolution program is mid-flight. The user's explicit instruction was to organize the artifacts so a new agent can form their own view of the next-step decision **without being biased by the prior agents' OPINIONS docs**. Reading order for an unbiased fresh take:
>
> 1. Read the **FACTS docs FIRST** (factual-only, forbidden-words enforced):
>    - [`analysis/p2_eval_2026-05-08/P2_PHASE_A_VERDICT_FACTS_2026-05-08.md`](../../analysis/p2_eval_2026-05-08/P2_PHASE_A_VERDICT_FACTS_2026-05-08.md) — promotion contract scorecard
>    - [`analysis/p2_eval_2026-05-08/P2_DEEPER_ANALYSIS_FACTS_2026-05-08.md`](../../analysis/p2_eval_2026-05-08/P2_DEEPER_ANALYSIS_FACTS_2026-05-08.md) — J1-J5 characterization
>    - [`analysis/iq_data_atlas_2026-05-08/IQ_ATLAS_FACTS_2026-05-08.md`](../../analysis/iq_data_atlas_2026-05-08/IQ_ATLAS_FACTS_2026-05-08.md) — cross-pool IQ measurement
>    - [`analysis/iq_shortcut_decomp_2026-05-08/IQ_DECOMP_FACTS_2026-05-08.md`](../../analysis/iq_shortcut_decomp_2026-05-08/IQ_DECOMP_FACTS_2026-05-08.md) — 23-cell R² decomposition (Stage 1 of IQ-deconvolution program)
>    - [`analysis/iq_perlayer_probe_2026-05-08/IQ_PERLAYER_PROBE_FACTS_2026-05-08.md`](../../analysis/iq_perlayer_probe_2026-05-08/IQ_PERLAYER_PROBE_FACTS_2026-05-08.md) — check (a) per-layer IQ probe (NEW)
>    - [`analysis/dor_encoder_axis_2026-05-08/DOR_ENCODER_AXIS_FACTS_2026-05-08.md`](../../analysis/dor_encoder_axis_2026-05-08/DOR_ENCODER_AXIS_FACTS_2026-05-08.md) — check (c) Dor encoder-axis (NEW)
>    - [`analysis/p2_d_hdtf_2026-05-08/P2_D_HDTF_FACTS_2026-05-08.md`](../../analysis/p2_d_hdtf_2026-05-08/P2_D_HDTF_FACTS_2026-05-08.md) — check (b) D step3000 HDTF Phase C (NEW)
>    - [`docs/packet_retrospectives/MODEL_GOALS.md`](MODEL_GOALS.md) — three pillars + promotion bar
>    - [`docs/packet_retrospectives/SCORECARD_GUIDE.md`](SCORECARD_GUIDE.md) — when to use which scorecard mode
> 2. Form your own reading of what the data shows.
> 3. THEN, optionally, read the OPINIONS docs to compare your reading to the prior agents'. **You are explicitly free to disagree.**
>    - [`analysis/p2_eval_2026-05-08/P2_DEEPER_ANALYSIS_OPINIONS_2026-05-08.md`](../../analysis/p2_eval_2026-05-08/P2_DEEPER_ANALYSIS_OPINIONS_2026-05-08.md) — synthesis + proposed pre-Stage-2a sequence
>    - [`analysis/iq_shortcut_decomp_2026-05-08/IQ_DECOMP_OPINIONS_2026-05-08.md`](../../analysis/iq_shortcut_decomp_2026-05-08/IQ_DECOMP_OPINIONS_2026-05-08.md) — sister-agent's Stage 2 reading
>    - [`docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md`](threads/iq_shortcut_deconvolution_program_2026-05-08.md) — the program proposal thread
> 4. (No pending check artifacts as of 2026-05-08 evening — all three pre-Stage-2a checks landed; Stage 2 decision is now user-gated.)
>
> The fresh agent's task is to look at the convergent FACTS, optionally challenge the existing OPINIONS, and make their own call on Stage 2 (or whatever direction they propose). The user reserves the GPU spend authorization regardless.

---

## Where we stand right now (one paragraph)

The most recently completed packet is **[T3](packets/T3.md) (Stage 3 IQ-shortcut single-lever)** — 3 slots × 6 step ckpts each, FT-from-P8A_step5000, drop / IQ-match / per-method-IQ-match the training real distribution. Trained 2026-05-09; promotion-contract scorecards on 5 candidate ckpts on 2026-05-09 PM; F4 substrate-cleaning re-eval + HDTF cross-substrate eval overnight 2026-05-09 → 2026-05-10. **T3_SLOT1_PERIODIC_STEP2500 is the highest-capability candidate** seen across all R13 packets: F4 v2 viso 79.27% (vs P8A 67.09%, PA 72.36%); HDTF `proper_visomaster_enhanced_teams_dev` 84.97% at FPR-cal 5% (vs step1500 77.38%, vs P8A 94.62%). **Step2500 generalizes BETTER than step1500 on every HDTF teams cell** — NOT a PA-style v2-bound collapse. Step2500 fails the F0 v2 dev recall floor (22.95% < 30%) only because the contract auto-calibration on F0 reals (which include chronic-6) clips τ to 0.971 to control Roy_D over-confidence. **Step1500 (37.63% F0 macro) is the secondary candidate that passes the strict F0 contract floor**; both are valid candidates at different operating points. P8A wins lex-policy rank 1 in the contract scorecards because all T3 ckpts hit 9.92-9.99% stress FPR (P8A 6.85%); the T3 stress FPR is entirely Roy_D-on-lighting_extreme, ex-Roy_D T3 beats P8A on stress. **Deployment recommendation pending user decision**: ship step2500 with FPR-calibrated τ on production-realistic reals + optional T4 refinement packet (face_scale_jitter @0.50 + Slot 1 keep-list). Sub-headline: **PA's mechanism is encoder-level, not memorization-bound** — T3_SLOT1 (which inherited PA's `visomaster_enhanced` + `visomaster_teams_enhanced` data sources and added the keep-list lever) preserves more HDTF capability than PA alone (PA collapsed to 7.87% on HDTF; T3_SLOT1 step2500 at 84.97%).

---

## Older "where we stood" paragraph (P1 era, preserved for context)

The most recently completed packet was **[P1](packets/P1.md) (PE_PAIR_RANK_DRO)** — pair_rank λ=0.2 + multi-axis GroupDRO with `chronic_flag` (BUNDLE) vs pair_rank-only (PAIRRANK), both FT-from-P8A_step5000 on the post-`2feea58` codepath. Phase A (29-suite contract scorecard) **SUCCEEDED**; Phase C (16-suite HDTF cross-substrate) **FAILED** at the promotion_contract step (per-suite diagnostic completed; verdict bundle never written; root cause not yet investigated). At the contract-selected τ, **no P1 ckpt clears F1 (90% lockbox recall)**; the contract's rank-1 winner is `P1_PAIRRANK_PERIODIC_STEP500` with 70.8% lockbox recall at τ=0.768. **F1 is reachable at non-contract τ** — `BUNDLE_step500` hits 96.5% lockbox recall at FPR ≤ 10% (τ=0.989); `PAIRRANK_step500` hits 91.5% at τ=0.50. **F5 PASSES big** for the BUNDLE arm: PC_Generator chronic FPR drops 62.9% → 0-3%. **F4 PASSES** universally at calibrated τ (max HDTF real FPR 1.4%). **F3 is partial**: 3 of 4 untargeted IQ axes show decoupling (sharpness, min_dim, color_b_dev) but face_area_fraction amplifies (+106-266%). **P1 introduced a Roy_D regression**: 130 frames flip from 29% FPR (P8A) to 78-93% FPR (P1); the regression is `color_b_dev`-aligned (Δr=+0.71 mirror of P8A's r=−0.71) AND **shared between BUNDLE and PAIRRANK arms** (Wilcoxon p=0.875), refuting the initial GroupDRO-balloon hypothesis. **Deployment is still E2B** per memory `project_deployment_is_e2b_2026-05-06.md`; E2B beats every P1 ckpt on `dev_fake_macro_recall` (0.508 vs 0.354). This session also delivered two bug fixes (trainer.py:1727 W&B logging gap + phase_d/run_chronic_filter.py regex) and a wiki contract upgrade (rolling STATE, FACTS/OPINIONS file convention, hard stop on per-session HANDOFF docs).

---

## In flight / running right now

**Stage 2 status as of 2026-05-09 03:30 local** — image `1.3.274`:

| slot | run_id | terminal state | runtime | best ckpt (holdout AUC) | DOR_REAL_LOCKBOX p50 (probe) |
|---|---|---|---:|---|---:|
| S2-1 REAL_AUG_OFF | `4pf24vo7` (us-west4) | SUCCEEDED 03:22:29Z | 5h 35min | step500 auc 0.9914 | 0.536 (step4500) |
| S2-2 LOW_LR_FT | `rravpdb9` (us-east1) | SUCCEEDED 00:05:48Z | 2h 18min | step2500 auc 0.9929 | **0.285 (step4500)** ← best |
| S2-3 WEAK_PAIRRANK | `wc6hvodm` (us-central1) | SUCCEEDED 00:17:02Z | 2h 31min | step3500 auc 0.9935 | 0.517 (step4500) |

P8A reference: `DOR_REAL_LOCKBOX p50 = 0.019`.

**CPU score-distribution probe COMPLETE** (FACTS + OPINIONS at [`analysis/stage2_cpu_2026-05-09/`](../../analysis/stage2_cpu_2026-05-09/)). 9 ckpts × 388-frame Dor cohort × 5 reference cohorts on MPS, ~3min total. All three slots regress P8A's signature dor invariance by 15-28× at step4500. Per-slot reading:

- **S1 REAL_AUG_OFF — IQ-prior FLIP hypothesis REFUTED.** Trajectory monotonically degrades: 0.27 → 0.35 → 0.54.
- **S2 LOW_LR_FT — Preservation PARTIALLY SUPPORTED.** Best preserver among Stage 2 slots; Pearson r 0.62 with P8A on DOR_REAL_LOCKBOX (vs S1 0.53, S3 0.53, P2D 0.46). Step4500 candidate for promotion contract scorecard.
- **S3 WEAK_PAIRRANK — Dose-response hypothesis REFUTED.** λ=0.05 still produces step500 cluster collapse (DOR_REAL_DEV std 0.077 = 4.5× compressed vs P8A 0.347), AND bidirectionally regresses Dor (real-FPR up + fake-recall DOWN — only Stage 2 ckpt with both-side regression).

Convergent reading: **the binding constraint is FT itself, not any of the three single levers.** All R13 packets that FT from P8A drift the dor cluster within 500-2500 steps. Next packet candidate (per OPINIONS doc §3a): L11 anchor-loss FT — the only structurally distinct intervention left. Promotion contract scorecard for `S2_step4500` (~$15-20 GPU) is the cheap next measurement.

**Pre-Stage-2 context** — P2 Phase A scorecard SUCCEEDED (verdict at [`P2_PHASE_A_VERDICT_FACTS_2026-05-08.md`](../../analysis/p2_eval_2026-05-08/P2_PHASE_A_VERDICT_FACTS_2026-05-08.md): P8A rank 1, P2D rank 4 with major Pillar-2 regressions). All three pre-Stage-2a checks (a/b/c) landed CPU+GPU; (b) HDTF Phase C verdict is at [`P2_D_HDTF_FACTS_2026-05-08.md`](../../analysis/p2_d_hdtf_2026-05-08/P2_D_HDTF_FACTS_2026-05-08.md) (recovered locally; cloud aggregator hit known suite-name-map bug — see open loop `phase-c-hdtf-promotion-contract-failure` below, root cause now identified).

---

## P2 training reference (historical)

**P2 packet** (PREVENT_NOT_UNLEARN, 4 slots) — **all 4 slots `JOB_STATE_SUCCEEDED` 2026-05-07/08**. Eval folder at [`analysis/p2_eval_2026-05-08/`](../../analysis/p2_eval_2026-05-08/) carries the populated FACTS docs (`RESULTS_FACTS_2026-05-08.md`, `CANARY_TRAJECTORY_FACTS_2026-05-08.md`). Per-slot summary:

| slot | yaml | wandb run | wall clock | terminal | final `_step` | canary fires | training-AUC peak |
|---|---|---|---|---|---:|---:|---|
| A — BUNDLE | `R13_P2_SCRATCH_BUNDLE.yaml` | `u22wz1vf` | 2h 11m | 23:49:26 UTC | 7508 | 1 | 0.6802 (step500) → declines to 0.5100 (step7000) |
| B — CORR_ONLY | `R13_P2_SCRATCH_CORR_ONLY.yaml` | `mlo5vfe8` | 2h 57m | 00:39:28 UTC | 6508 | 1 | 0.6800 (step500) → drops below chance to 0.3815 (step3500) |
| C — PAIRRANK_ONLY | `R13_P2_SCRATCH_PAIRRANK_ONLY.yaml` | `oaur8odo` | 3h 17m | 00:53:42 UTC | 13013 | 4 | 0.8840 → 0.9862 (step7000) monotone |
| D — FOURIER | `R13_P2_SCRATCH_FOURIER.yaml` | `89tt9xyz` | 6h 43m | 05:31:20 UTC | 25013 | 8 | 0.8743 → 0.9908 (top_n step19000) monotone |

Slot D's wall-clock is ~3× the other slots; consistent with the per-frame FFT overhead from `data/augmentations/fourier_band_aug.py` (~3-channel FFT2 + IFFT2 per fired frame at p_apply=0.5).

Canary trajectory headlines (from [`CANARY_TRAJECTORY_FACTS_2026-05-08.md`](../../analysis/p2_eval_2026-05-08/CANARY_TRAJECTORY_FACTS_2026-05-08.md)):

- Slot A canary single-fire: `score_p95_on_reals=0.4994`, all `recall_at_tau05` = 0 — uniform-output state.
- Slot B canary single-fire: `score_p95_on_reals=0.5005`, `lockbox_recall@FPR_10pct=0.01` but `recall_at_tau05/{viso,deeplive}_fake = 1.00` — every frame scored ~0.500, fakes barely above τ=0.5.
- Slot C 4 fires: `score_p95_on_reals` saturates to 0.998+ at first fire; chronic-6 means rise from 0.07-0.55 (step3000) to 0.66-0.97 (step12000); `lockbox_recall@FPR_10pct` peaks at 0.26 (step12000).
- Slot D 8 fires: `score_p95_on_reals` lower (0.88-0.99 oscillating); **non-monotonic chronic dip at `_step=18000`** — Roy_D 0.94→0.68, bla_bla_chow 0.91→0.41, max_per_id 0.94→0.68 — coinciding with `lockbox_recall@FPR_10pct=0.41` peak (top among all 4 slots).

**Open observations** flagged in the FACTS docs (no interpretation, no resolution at this time):
- Canary cadence anomaly: yaml configured `frequency_steps: 1000` but observed cadence is every 3000 W&B `_step`s for C/D, only 1 fire each for A/B. Cause unverified.
- All 4 slots terminated before `nEpochs: 16` cap; early_stopping_patience=12 likely fired but stop-condition not verified at this level.
- Slot A and Slot B both early-stopped at the SAME `_step` band (6500-7500) at epoch 3 with similar final score distributions.

**D1-D5 CPU diagnostics** at [`analysis/p2_eval_2026-05-08/d1_d4_cpu/`](../../analysis/p2_eval_2026-05-08/d1_d4_cpu/) — COMPLETED 2026-05-08 morning. Five FACTS docs:
- `D1_AB_DEGENERACY` — Slot A `top_n_step500` real-fake gap +0.072 (mean 0.43→0.51); Slot B gap +0.016 (means cluster ~0.56-0.58); both ckpts have weak discrimination → excluded from Phase A scorecard.
- `D2_SLOTD_PAIR` — Slot D step6000 vs step19000 per-cohort: PC_Generator__s22 0.22 → 0.78 (largest absolute shift); healthy_dor_shkedi 0.62 → 0.29; canary `_step=18000` chronic dip NOT reproduced by `top_n_step19000`.
- `D3_HISTOGRAM` — 15 cohort × 6 ckpt distribution table; P8A reference is OUTSIDE the P2-ckpt envelope on 7 of 15 cohorts (different failure pattern).
- `D4_WEIGHT_DELTA` — pairwise per-layer L2/cosine across 5 P2 ckpts + P8A. A/B nearly parameter-identical (cos > 0.998); `visual_proj` A/B vs C/D near-orthogonal (cos < 0.01); D step19000 resblock_11 uniquely orthogonal to A/B/P8A (min_cos = -0.27).
- `D5_CKPT_MAPPING` — verifies wandb `_step` == optimizer step (canary `_step=6000` chronic-6 EXACT match to local `top_n_step6000`); identifies `D_periodic_step3000` as the highest `lockbox_recall_at_FPR_10pct` ckpt across all 6 saved D ckpts (0.37 vs 0.06-0.20 at later ckpts; trajectory non-monotone in optimizer step) and the closest savable approximation to canary `_step=18000` chronic dip.

**Phase A scorecard for P2** — **LAUNCHED 2026-05-08 10:13 local on image 1.3.273** (Cloud Build `3737151f-c949-4326-8a4d-7ee1753eab79`, 17m29s, commit `eb815dd`). Vertex `6226906326623059968` (us-east1, `JOB_STATE_PENDING` since 08:13:43 UTC). Display name `p2-scratch-scorecard-2026-05-08`.

Curated 7-ckpt set per [`arena/checkpoint_maps/teams_target_domain.p2_scratch_2026-05-08.yaml`](../../arena/checkpoint_maps/teams_target_domain.p2_scratch_2026-05-08.yaml):
- `P8A_REFERENCE_STEP5000` (production anchor)
- `E2B_TOP_N_STEP3200` (current deployment)
- `P2_C_PAIRRANK_TOP_N_STEP7000` (C training-AUC peak)
- `P2_C_PAIRRANK_PERIODIC_STEP3000` (matched-step counterfactual to D step3000)
- `P2_D_FOURIER_PERIODIC_STEP3000` ⭐ (D5-identified low-saturation operating point; lockbox@10=0.37 on canary)
- `P2_D_FOURIER_PERIODIC_STEP8000` (yaml's nominal-cap)
- `P2_D_FOURIER_TOP_N_STEP19000` (D late-saturation peak; uniquely orthogonal layer-11)

Slots A and B excluded per D1 (weak discrimination at top_n_step500).

Suite manifest: 29-suite `target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml`. Promotion policy: target_real_fpr=0.07, target_stress_fpr=0.10, target_fake_recall_min=0.30 (matches P1 settings).

Output paths under `gs://training-job-outputs/test_results/teams_promotion_contract/p2-scratch-scorecard-2026-05-08/`:
- `reports/` — per-suite frame-level reports
- `diagnostic_scorecard/scorecard.{csv,wide.csv,json}` — fixed-threshold 0.5 sidecar
- `promotion_contract/{threshold_grid.csv, selected_threshold_scorecard.csv, checkpoint_summary.csv, promotion_contract.json, promotion_winner.json}`

**Bug fixes committed (2026-05-07, commits `2e3c26b` + `fb6c38f`)**:
- `trainer/trainer.py:1727` W&B logging gap when `use_group_dro=true` — RESOLVED.
- `analysis/p1_pe_eval_2026-05-07/phase_d/run_chronic_filter.py` regex bug — RESOLVED.
- Wiki contract upgrade (`608dbe2`) — RESOLVED.

**Canary probe infrastructure (NEW 2026-05-07)**:
- [`trainer/mixins/canary_probe.py`](../../trainer/mixins/canary_probe.py) — bulletproof in-training deployment-quality monitor; logs ~15 deployment-relevant scalars to W&B every N steps.
- [`arena/canaries/teams_chronic_diverse_800_2026-05-07.parquet`](../../arena/canaries/teams_chronic_diverse_800_2026-05-07.parquet) — 800-frame canary (600 reals + 200 fakes) covering 6 chronic + 5 healthy + HDTF reals + lockbox + viso + deeplive fakes.
- See thread [`in_training_canary_signal`](threads/in_training_canary_signal.md) for the design rationale.

---

## Most recent eval folder

[`analysis/p2_eval_2026-05-08/`](../../analysis/p2_eval_2026-05-08/) — P2 packet outcomes (4 slots SUCCEEDED). Read in order:

1. `RESULTS_FACTS_2026-05-08.md` — per-slot training outcome + ckpt inventory
2. `CANARY_TRAJECTORY_FACTS_2026-05-08.md` — full per-slot canary trajectory tables
3. `SLOT_D_SMOKE_FAIL.md` — Slot D Fourier-aug 3-tier smoke (Tier-3 floor relaxation rationale)
4. `d1_d4_cpu/` — CPU diagnostics (D1 A/B degeneracy, D2 Slot D step18000 deep-dive, D3 4-slot histogram, D4 weight-delta) — in flight
5. `slot_{A,B,C,D}_canary_history.csv` — raw W&B history pulls

Phase A scorecard for the P2 ckpts has NOT been authorized yet; the user reviews the canary trajectories + D1-D4 CPU diagnostics before deciding GPU spend on the P2 ckpt pool.

[`analysis/p1_pe_eval_2026-05-07/`](../../analysis/p1_pe_eval_2026-05-07/) — prior packet (P1) eval. Read in order:

1. `RESULTS_FACTS_2026-05-07.md` — raw scorecard
2. `RESULTS_F1_F5_FACTS_2026-05-07.md` — close-criterion verdicts
3. `DEEP_DIVE_FACTS_2026-05-07.md` — consolidating record (includes mid-session Self-correction log §16)
4. `FOLLOWUPS_FACTS_2026-05-07.md` — 7 additional CPU jobs (color_b_dev, lockbox τ curves, roy_d axis attribution, counter-theory probe, etc.)
5. Sub-investigation FACTS docs in `f2_pair_rank/`, `f3_color_b_dev/`, `roc_degeneracy/`, `roy_d_regression/`

The single OPINION doc is `AGENT_PROPOSAL_2026-05-07.md` — read AFTER user authorizes per `AGENTS.md` "Reading order for forming an independent view". The agent retracted one mechanism claim mid-session (GroupDRO balloon → refuted by counter-theory probe); §4 of the proposal documents this for reviewer skepticism calibration.

---

## High-severity / load-bearing open loops (snapshot — see `OPEN_LOOPS.md` for full list)

(Regenerate `OPEN_LOOPS.md` via `python tools/regenerate_open_loops.py` after thread edits.)

- **`trainer-py-1727-wandb-logging-gap`** (resolved 2026-05-07; commit pending) — fix applied locally, needs commit + a future GroupDRO training run to verify scalars flow.
- **`phase-c-hdtf-promotion-contract-failure`** — Phase C 2026-05-07 failed at the contract scoring step; per-suite diagnostic completed but `promotion_contract/` directory never written. Root cause not yet investigated. F4 verdict for P1 was recovered locally by applying Phase A τ to Phase C per-frame reports.
- **`f2-not-testable-from-phase-a`** — F2(a) close criterion ("≥30% relative lift on ≥2 of 6 paired lanes among previously-missed fakes") cannot be evaluated from Phase A substrate: only 1 of 6 yaml-named paired lanes (`deeplive_teams`) has a Phase A proxy + P8A baseline saturated at 0.971-1.000 → 30% lift bar arithmetically unreachable. Tracked in `threads/eval_substrate_layering.md`.
- **`pair-rank-non-paired-lane-collateral`** (NEW 2026-05-07) — Roy_D regression mechanism: pair_rank loss appears to lift fake-class scores on real frames belonging to identities NOT in any paired training lane. Δr=+0.71 mirror of P8A's color_b_dev signal; shared between BUNDLE and PAIRRANK arms. Owning thread: [`threads/pair_rank_collateral.md`](threads/pair_rank_collateral.md).
- **`contract-policy-bug-fix-not-committed` component 3** (resolvable 2026-05-07) — recall-floor tier mechanism worked correctly under P1; τ=0.999x for BUNDLE_step3750/4000 was ROC degeneracy (0 grid points hit recall ≥ 0.30 within FPR ≤ 0.07) not a contract bug. Component 3 of this loop closes; thread `threads/contract_policy_bug.md` to update.
- **`grouped-manifest-v2-stale-paths`** — 5 of 9 Phase A.5 diagnostic substrates have stale `gs://local/...` placeholder paths; cannot be re-evaluated until manifest is regenerated.

---

## Decisions pending (mapped to user)

1. **Authorize commits** for the trainer.py:1727 fix + phase_d regex fix. Both are low-risk; trainer fix is a logging-only change; phase_d fix matches the `_FIXED.csv` outputs the agent already verified.
2. **Authorize next packet** direction. Three candidate frames live in `AGENT_PROPOSAL_2026-05-07.md` §6:
   - **Loss-function**: per-identity score-distribution preservation auxiliary loss (catches Roy_D-class regressions before training ends).
   - **Training-base**: multi-FT-base study (P8A vs E2B vs CLIP-direct) — tests whether FT-from-FT chain depth matters.
   - **Investigation-only**: P8A train-set membership audit for chronic-6 + Roy_D — cheap; refines the §3 mechanism claim.
3. **Authorize Phase C re-run** (or local F4-from-cached-reports) — current F4 verdict was recovered without the cloud `promotion_contract/` step. A re-run would close the loop with proper artifacts; or accept the local verdict and close the open loop.
4. **Authorize wiki contract finalization** — this session's restructure (rolling STATE, FACTS/OPINIONS, eval_folder_template, AGENT_GUIDE move, AGENTS.md extension, hard stop on HANDOFFs). The user pre-authorized the plan; commit at end-of-session if everything works.

---

## Eval-substrate caveats (carried over; still apply uniformly)

These caveats apply to every FPR / recall number on this page and the eval folder:

1. **Eval-vs-production crop-tightness mismatch** ([`eval_production_crop_tightness_gap`](threads/eval_production_crop_tightness_gap.md)) — eval frames carry more background context than production crops; structurally upstream of face-size leak + camera-signature shortcut + webcam FPR investigations.
2. **`sharpness_laplacian` is computed on full image, not face crop** ([`sharpness_metric_bug`](threads/sharpness_metric_bug.md)) — per-quartile FPR analyses indexed on this column are partially confounded.
3. **Eval substrate has known data-quality issues** ([`eval_substrate_data_hygiene`](threads/eval_substrate_data_hygiene.md)) — is_no_face slice (n=219) data degeneracy + 99×110-pixel source crops below any reasonable production resolution floor.
4. **Quality-enhancement routing bug** affects ALL R13 packets pre-2026-05-05 (memory `project_quality_enhancement_routing_2026-05-05.md`). Relative comparisons remain valid (uniform contamination); absolute claims about GFPGAN-enhanced fakes need post-fix retraining to revisit.

---

## Cross-references

- [`AGENTS.md`](AGENTS.md) — read-first protocol; "Reading order for forming an independent view" is the canonical pass-1 / pass-2 split.
- [`TIMELINE.md`](TIMELINE.md) — append-only chronological master index; tail tells you what just landed.
- [`OPEN_LOOPS.md`](OPEN_LOOPS.md) — mechanically-generated open-issue inventory.
- [`packets/P1.md`](packets/P1.md) — most recent packet retro.
- [`AGENT_GUIDE.md`](AGENT_GUIDE.md) — the 6-rule contract for proposing the next packet.
- [`eval_folder_template.md`](eval_folder_template.md) — the FACTS/OPINIONS file-level split for `analysis/<packet>_eval_<date>/` folders.
- [`packet_template.md`](packet_template.md) — the structure every per-packet retro under `packets/` follows.
- [`archive/`](archive/) — older dated state snapshots.
