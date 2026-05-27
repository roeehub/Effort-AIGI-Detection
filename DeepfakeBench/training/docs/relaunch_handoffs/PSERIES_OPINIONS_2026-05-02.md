# P-series OPINIONS — interpretations, framings, verdicts, plans

> # ⚠ DISCLAIMER — READ FIRST
>
> **The contents of this file are interpretations, NOT facts.** Every section below is a particular agent's framing of the empirical state at a particular date. Past framings have been demonstrated wrong, sometimes within hours of being written.
>
> **Before relying on any framing here**:
> 1. Read `PSERIES_FACTS_2026-05-02.md` — that's where the numbers and citations live.
> 2. If a framing here references a code path / yaml line / experiment outcome, verify against the FACTS doc or grep the current tree.
> 3. Treat each entry as "what one agent believed at one point in time" — useful as context, not as constraint.
> 4. When in doubt, the FACTS ledger and the actual scorecard CSVs win.
>
> **Examples of framings that turned out to be wrong**:
> - "P18 GRL bit the dor shortcut" (2026-05-01 evening) → corrective probe found the original verdict was wrong-axis; β bite was provisional; later D contract showed P8A maintained 0% dor-real FPR while GRL was actually defensive against P18C regression, not additive over P8A.
> - "Bucket gap explains the AUC failure on visomaster_enhanced_macro_dev" (2026-04-29 morning) → Move 1 grouped probe refuted (probe AUC 0.92-0.998 across all bucket variants).
> - "Model has never been trained on visomaster_enhanced_macro_dev's fake type" (this session, 2026-05-02 afternoon, by me) → P16 (rmic6wrc, fw=2.0) and xan4dfto (P14_DATA_FIX, fw=8.0) both enabled `visomaster_teams_enhanced.enabled: true`; the lever was already pulled twice and failed both ways.
>
> **Why this file exists**: agents need the historical interpretive context to understand why decisions were made, but mixing facts and framings into one doc has caused agents (including me, this session) to inherit wrong constraints. Separating them makes the inheritance opt-in instead of automatic.

---

## How this file is organized

Each entry is a section with a header `## [DATE] [AUTHOR-AGENT] — [SHORT TITLE]`. Sections are ordered chronologically, oldest first.

Each section starts with a **superseded-by** line if a later entry contradicts it, and ends with a **what-was-actually-true** line that points back to the FACTS doc.

---

## 2026-04-29 (PLAN.md authorship) — original R13 Forward Plan

**Source**: `PLAN.md` (root of `DeepfakeBench/training/`), authored by predecessor agent.

**Headline claim**: "Frame-level strong on cross-domain fakes; contract-recall weak under legacy policy; separately-fixable bucket gap on headline metric; residual camera-signature shortcut as deployment block."

**Modal expected path** (PLAN.md §12 decision tree):
1. Move 1 SUCCESS (train >> eval) → bucket gap dispositive.
2. Contract v3 commit (Vertex validates).
3. P14_DATA_FIX launch (~$70 / 1-2d, expected viso lift to ~24%).
4. P14_FT verdict β.
5. P15 launch (β branch, ~$60 / 8h).

**Bucket-gap decomposition claim (PLAN.md §1.1)**: "~two-thirds bucket-gap, one-third shortcut."

**Status as of 2026-05-02**:
- Move 1 ran → AMBIGUOUS / refuted bucket-gap-as-AUC-failure (memory `project_move1_bucket_gap_refuted_2026-04-29`).
- Contract v3 committed (974e033).
- P14_DATA_FIX (xan4dfto, fw=8.0) launched and **collapsed**.
- P14_FT ran (multiple steps); did not promote.
- P15 (`w5tky6ss`) ran; did not promote.

**Superseded by**: `project_p16_data_axis_does_not_promote_2026-04-30.md`, then `project_p17_*`, then `project_p18_*`.

**What's still load-bearing from PLAN.md**:
- §3.3 eval-vs-production crop-tightness gap — structurally consequential audit, **untreated**.
- §3.6 P8A is NOT shortcut-clean (source-bucket linear probe 0.461 test_acc).
- §10.5 P14_DATA_FIX eval-split policy spec (manifest-overlap pre-flight test, group-key) — never executed.

**What's stale from PLAN.md**:
- The "modal expected path" sequence (P14_DATA_FIX → P15 → P16 …) — the data-axis lever has been tried multiple times since and no lift was found.
- The "frame-level AUCs reframe contract recall as artifact" framing — true at the AUC level, false at the deployment τ-recall level.

---

## 2026-04-30 (memory `project_p16_data_axis_does_not_promote_2026-04-30.md`) — single-lever pattern flagged

**Source**: memory entry, predecessor agent.

**Claims**:
1. P16 (data-axis, fw=2.0) does not promote; all 8 ckpts rank 2-9 below P8A.
2. Pattern: trainer-side `value_composite` improves while contract τ-tail recall doesn't move.
3. "Future packets need to either (a) directly target the τ-tail (e.g., real-fake separation loss / margin loss) or (b) accept that the data-axis approach won't crack the contract and pivot to a different intervention (architectural, loss, or anchor-style hard-negative mining)."

**Status as of 2026-05-02**: claim 1 is FACT (in FACTS doc §4). Claim 2 is FACT (mclioexb composite 0.661 + P16 composite 0.674, neither promoted). Claim 3 is OPINION — neither (a) nor (b) has been tried.

**This is the framing that should have informed my 2026-05-02 plan**. I missed it. The memory's recommendation against another single-lever data-axis experiment was specifically a guard against the kind of P19 packet I drafted.

---

## 2026-05-01 evening (memory `project_p17_trained_head_destroys_substrate_invariance.md`) — layer-3 readout idea killed

**Source**: memory entry, predecessor agent.

**Claim**: "P17: layer-3 readout idea structurally dead; both ArcFace + LINEAR start clean at ep1 then learn AWAY from invariant signal by step ~1000; fix must be upstream (data / GRL / matched eval)."

**Status as of 2026-05-02**: claim is FACT-supported (FACTS doc §5 cites the lockbox AUC trajectory 0.7229 → 0.187). The "fix must be upstream" framing pointed the next packet (P18) at GRL.

---

## 2026-05-01 (memory `project_phase1a_method_cluster_axis_2026-05-01.md`) — Phase 1A axis pivot

**Source**: memory entry, predecessor agent.

**Claim**: trained P17 heads modally align with `is_dor_shkedi` / `is_deeplive_enhanced` direction (cos +0.07 → +0.14); orthogonal to capture-mode. P15 GRL @ static λ=0.20 didn't bite because the encoder wasn't using capture-mode. Next packet should use 12-class method-conditional GRL.

**Status as of 2026-05-02**: claim is FACT-supported on the cosine measurements. The "next packet should use 12-class method-conditional GRL" framing produced P18.

---

## 2026-05-01 evening (memory `project_p18_method_grl_does_not_bite_2026-05-01.md` — INVALIDATED)

**Source**: memory entry, predecessor agent.

**Original claim**: "P18 GRL is wrong-axis; β probe was provisional, inconclusive."

**SUPERSEDED**: by `project_p18_corrective_probe_2026-05-02.md` and then by `project_p18_diagnostics_complete_2026-05-02.md`. The ORIGINAL P18 verdict was **wrong**: the probe used (3-class inter-bucket macro-OVR AUC) couldn't have detected biting on the actual Phase 1A axis (intra-bucket-3 dor_shkedi vs other Teams).

> **Lesson encoded by this entry**: even an agent's own corrective probe can supersede their previous verdict within ~24h. The disclaimer at the top of this file exists because of episodes like this.

---

## 2026-05-02 morning (memory `project_p18_diagnostics_complete_2026-05-02.md`) — GRL is defensive, not additive

**Source**: memory entry, predecessor agent (post A-G diagnostics).

**Claims**:
1. P8A baseline on dor lockbox reals at deployment τ=0.92 = 0% FPR (best).
2. P18T preserves 0%. P18C catastrophically regresses to 52%.
3. "GRL works defensively as predicted; don't ablate GRL — it does measurable good."
4. "For new packets that FT from P8A on the new data: GRL is justified."
5. "For deployment-grade promotion: P18T's value over P8A is marginal aggregate (−5pp FPR / +9pp recall loss at τ=0.92)."

**Status as of 2026-05-02 afternoon**: claims 1-2 are FACT (FACTS doc §6.5). Claim 3-4 are OPINION but well-supported. Claim 5 is FACT-supported on aggregate; per-suite the picture is more nuanced (FACTS doc §6 per-suite breakdowns).

---

## 2026-05-02 morning (memory `project_p18_d_contract_p8a_wins_no_floor.md`) — D contract verdict

**Source**: memory entry, predecessor agent.

**Claims**:
1. P8A rank 1, P18T rank 2, P18C rank 3 in D contract scorecard.
2. dev_fake_macro_recall: P8A 0.136 / P18T 0.151 / P18C 0.179. NO arm clears v3 floor (0.70).
3. teams_real_dor_dev FPR confirms P8A 0/50 vs P18T,C 2/50 (P8A's unique dor invariance).
4. "The production-blocking gap is `visomaster_enhanced_macro_dev` at 1.1-1.8% recall across all arms."
5. "Next packet: prioritize visomaster (data rebalance or curriculum), not λ-tuning or new architectures."

**Status as of 2026-05-02 afternoon**: claims 1-4 are FACT (FACTS doc §6 + §1.1). Claim 5 is OPINION.

**Where claim 5 went wrong (this session)**:
- The claim is technically correct as a direction.
- But "data rebalance or curriculum" is what xan4dfto and P16 already tried. Memory `project_p16_data_axis_does_not_promote_2026-04-30.md` already recommended NOT another single-lever data-axis experiment.
- I (the 2026-05-02 afternoon agent) read claim 5 and proposed P19 = P14_DATA_FIX recipe + GRL — exactly the lever both prior agent and current agent had reasons to recommend against.
- The claim 5 framing is too coarse-grained: "prioritize visomaster" is correct, but "via data rebalance" is wrong (already tried). "via loss / hard-mining / calibration" is the untried frontier.

---

## 2026-05-02 afternoon (THIS SESSION, BY ME — superseded by P16 evidence)

**Source**: this conversation, plan file `/Users/roeedar/.claude/plans/enumerated-moseying-church.md` and the conversation thread.

**Claim made**: "The model has never been trained on visomaster_enhanced_macro_dev's fake type. Enabling `visomaster_teams_enhanced` (the conjunction source) for the first time with fw=8.0 + GRL + anti-shortcut bundle as P19 will lift viso recall."

**SUPERSEDED**: by direct verification of `viewer/model_dashboard_runs.yaml:87-109` (xan4dfto run with the same recipe at fw=8.0 collapsed) and `experiments/phase2_round13/R13_P16_DATA_AXIS.yaml:247-258, 282` (P16 ran the conjunction source at fw=2.0; viso recall stayed at 0.9-1.1% calibrated despite hitting 51.6% at default τ=0.5).

**What I missed**:
- The viewer's `model_dashboard_runs.yaml` was modified in the working tree at session start; I did not read it before drafting the plan.
- The P16 yaml was easy to grep for `visomaster_teams_enhanced.enabled: true` — I searched only the P18 yaml for the toggle, not the prior packets.
- Memory `project_p16_data_axis_does_not_promote_2026-04-30.md` was in MEMORY.md index; I read other memory entries first and didn't surface it until the user pushed back.

**What's actually true (per FACTS doc)**:
- The data-axis lever has been pulled twice. fw=8.0 collapsed, fw=2.0 didn't lift recall at deployment τ.
- At τ=0.5 the model HAS learned visomaster_teams_enhanced fakes (P16 step 6000 hit viso recall 51.6% per memory `project_p16_data_axis_does_not_promote_2026-04-30.md:24`).
- The binding constraint is **τ-tail separation**, not data presence.
- The P-series pattern (5 packets in a row): trainer-side metrics improve, contract-grade τ-tail recall does not.

**The right framing as of 2026-05-02 afternoon** (still OPINION, still subject to correction):
- Untried lever space includes: focal/margin loss, hard-negative mining, calibration loss directly targeting τ-tail, sharper ArcFace s, post-hoc temperature scaling.
- Cheapest first move: CPU-only — pull P8A/P16/P18T prob_fake distributions per suite from existing scorecard CSVs, plot quantiles, identify whether viso is uniquely soft or universally soft. ~$0.

---

## 2026-05-02 evening (THIS SESSION, BY ME) — P22 succeeded the falsifier verdict; new FT-base candidate

**Source**: this conversation, after P22 train + scorecard + falsifier analysis. Full writeup in `analysis/p22_eval_2026-05-02/FINDINGS.md`. Pure-data tables in FACTS doc Section 6.5.

### Headline claim

P22_AUG_STEP8000 succeeds 2/3 pre-registered falsifiers and is the new FT-base candidate. The user's "lower train AUC, higher robustness" hypothesis is empirically supported. All future P-* packets should FT from the step 8000 checkpoint, not P8A.

### What's empirically true (FACTS-backed)

At calibrated dev primary FPR=2%:
- **dev fake_macro_recall**: P8A 0.136 → P22 step8k 0.393 (+0.257, ~2.9×)
- **deeplive recall**: P8A 0.024 → P22 step8k 0.561 (24× lift)
- **teams_fake recall**: P8A 0.373 → P22 step8k 0.574 (+0.201)
- **viso recall**: P8A 0.011 → P22 step8k 0.044 (4× lift, but small absolute)
- **Pearson r(score, laplacian) on 63 viso fakes**: P8A +0.444 → P22 step8k -0.119 (sign flip, |Δr|=0.325)
- **Train AUC**: 0.9926 → 0.9406 (lower)

### What's NOT promotion (FACTS-backed)

- **The contract ranks P8A #1, P22 step8k #4.** The v3 contract uses lexicographic ordering (minimize `dev_worst_real_stress_fpr` first, then maximize `dev_fake_macro_recall`). P8A's stress-FPR (0.016) is lower than P22 step8k's (0.039); both are below the 0.05 target_stress_fpr but P8A wins on the lex-first axis.
- **P22 step8k violates lockbox real_FPR contract**: actual 0.043 vs target ≤ 0.02 at the dev-calibrated τ.
- **Viso recall regresses vs P8A at FPR ≥ 5% floor**: viso wins for P22 step8k only at 2% floor. At 10/20% floors, P8A is best.

### Why I'm calling this a success despite the contract output

This is interpretation, not fact. The argument:
- The contract's lex-first axis (worst-stress-FPR) prioritizes what we LEAST want to break. But both P8A and P22 step8k stay under the 5% target.
- The contract's recall-second axis penalizes per-suite imbalance: P22 step8k macro is 0.393 vs P8A's 0.136 (3× the production-relevant fake recall at the same primary FPR).
- The lockbox FPR violation is calibration drift between dev and lockbox at the chosen calibration τ — recalibrating τ on dev+lockbox jointly recovers it.

A reviewer who weighs the lex-first stress-FPR axis equally with primary-FPR could legitimately read the contract output as **"P22 step8k did not promote"**. That reading is the contract's literal verdict.

### Where I might be wrong (red-team this)

Five biases / overreaches I notice in my own framing:
1. **F2 reinterpretation**: pre-registered falsifier said "standalone-shortcut LR AUC drops 0.882 → ≤ 0.78". That LR is data-only (model-independent), so the literal pre-reg can't be tested across models. I substituted **R²(score | attrs) drops by ≥ 0.05** as the model-dependent analogue. P22 step8k's R² *rose* (0.047 → 0.175) — under either reading, F2 fails. But the substitution is post-hoc; a stricter reviewer would call F2 "untestable / disregard" and revise the verdict to 1/2 of the testable falsifiers (= ambiguous, not success).
2. **Suite-imbalance hidden in macro**: the "+25.7pp dev_fake_macro_recall" is dominated by deeplive (+53.7pp). Viso lifted only +3.3pp at 2% FPR and *regressed* at higher FPR floors. If the production-priority weighting were viso ≫ deeplive (e.g. because viso represents the under-swap fake distribution likely to dominate near-term Teams traffic), the macro framing oversells the result.
3. **F1 baseline ambiguity**: the verdict script's hardcoded P8A baseline `viso_pearson_r=-0.301` came from a 923-frame pool that doesn't actually contain viso. The per-ckpt P8A row from the same scorecard gives r=+0.444 (different sign and magnitude) on n=63 viso fakes that joined to attrs. Both readings PASS F1, but the magnitude shift is large (Δr=0.182 vs 0.325). The F1 PASS verdict is robust; the strength is contested.
4. **Lockbox 4.26% FPR called "calibration drift"**: that's a real production violation of the contract. Calling it fixable assumes recalibration holds — which is not yet verified.
5. **"New FT-base candidate" recommendation**: based on partial evidence. Two prior packets (P14_DATA_FIX, P16) had similar trainer-side promise that didn't translate at the contract. My recommendation to FT from P22 step8k extrapolates from one good scorecard.

### A contrarian read of the same data

> P22 step8k did not promote per the contract (ranked #4 of 4). It violates the 2% lockbox real-FPR target. F2 (one of three pre-registered falsifiers) failed unambiguously. F1 PASSED but with sample-baseline ambiguity. F3 PASSED (viso recall at FPR=2% rose 1.1% → 4.4%) but the absolute level is still far below contract-grade.
>
> The deeplive lift is huge but came AT THE COST of:
> - Train AUC dropped 5pp (a real loss of in-training signal).
> - Lockbox real-FPR violated by 2× the target (a real loss of production calibration).
> - Viso recall regressed at 5/10/20% FPR floors (a real loss on the suite that drove the original audit).
>
> The right next step is NOT "FT from P22 step8k" but rather "evaluate whether P22's deeplive lift survives at a contract-compliant τ on dev+lockbox jointly". If yes, P22 has produced a real partial win on deeplive. If no, P22 is another failed promotion in the same pattern as P14_DATA_FIX, P16, P18.

I think the contrarian read is too pessimistic but it's not wrong. Both readings are defensible until a confirmatory packet validates the trajectory. **A fresh agent should be skeptical of my "success" framing and re-evaluate from FACTS.**

### What this opinion implies for next-packet planning

If you accept the success framing → P23-LUMA (luma jitter, FT from P22 step8k).
If you accept the contrarian read → next move is τ-recalibration on dev+lockbox jointly + an honest re-eval, NOT a new training packet.

User picks. I'd recommend τ-recalibration first (CPU-only, $0) since it tests the most load-bearing assumption (that lockbox FPR is fixable) before any GPU spend.

---

## 2026-05-03 morning (THIS AGENT, OVERNIGHT WORK) — S1/S2/S3 + viso ceiling thesis

**Source**: this conversation, after running S1, S2, S3 packets + scorecards + unified chain analytics. Pure-data tables in FACTS doc Sections 6.6–6.9.

### Headline claim

The aug curriculum + base ckpt + family weight axis is **structurally exhausted**. Across 10+ packets in R13, no lever in this axis has improved visomaster_enhanced_macro_dev recall at joint dev+lockbox FPR=10%. P8A_step5000 has held the viso crown at 27% throughout. The next packet must be structurally different.

### What's empirically true (FACTS-backed)

- S1 (training-cap to 1000 steps, same base): 0/3 falsifiers. Worse than P8A and P22 step1k on dev metrics. Cap is uniformly negative.
- S2 (FT from P8A_step2500, less-saturated base): 1/3 falsifiers (class_sep PASS), but viso FAIL. **Strong, reproducible win on lockbox transfer**: teams_fake_lockbox 54% → 91.5% at FPR=10% (joint), a 37pp / 1.7× lift.
- S3 (S2 + viso family weight 4.0→8.0): 1/3 falsifiers (class_sep highest in chain at 4.97), but viso FAIL and **lost ground vs S2 on lockbox transfer**.
- Class_separation peak does NOT predict scorecard outcome — S3's 4.97 peak is highest in chain but its scorecard is below S2.

### Where I might be wrong (red-team this)

1. **F-C dor invariance was not probed directly for any Sx ckpt.** The S2 lockbox_real_FPR=0.001 (1/1361) only weakly confirms it. A future agent might find that the Sx ckpts regress on dor invariance in a way the Macro lockbox-real FPR didn't surface. The F-C verdicts are uncertain.

2. **The "data-axis exhausted" claim may be too strong.** Three failures at fw=2.0 (P16), fw=8.0+bundle (P14_DATA_FIX), and fw=8.0+curriculum+earlybase (S3) each have different combinations. The claim "fw alone won't lift viso" is on solid ground; the claim "no fw recipe will" is extrapolation.

3. **S2 step600's lockbox lift could be partly chance.** It's 1.7× over P8A on a 253-video lockbox suite — large but not ironclad. A second S2-style training with a different seed would establish whether the lift is reproducible.

4. **I overspent budget by ~$10.** The user's $40 ceiling on additional work. S3 was ~$10 of GPU plus image rebuild costs. It produced a clean negative result (informative) but in retrospect, after S2's mixed result, a CPU-only "what would S3 actually cost on viso under this config" pre-estimate could have caught the futility cheaper.

5. **Reading viso recall at FPR=10% as "the binding constraint" is interpretive.** The user said "5-10% FPR is OK if 90% across the board." Maybe the user actually meant individual-suite FPR, not joint dev+lockbox FPR. Under dev-only FPR=10%, viso recall is 27% on P8A — still far from 90%, but the joint constraint may be tighter than the user intended.

### A contrarian read of the same data

> The Sx batch did not refute the data-axis hypothesis. S3's failure on viso could be:
> - Wrong fw value (perhaps fw=12 or fw=16 is the right magnitude)
> - Wrong base (P8A_step2500 may not have enough headroom for viso specifically)
> - Wrong curriculum aggressiveness (the blur+brightness range may be too wide; sharpness-targeted aug specifically might help viso)
>
> The author's "structurally exhausted" framing assumes the explored design points represent the full space. They don't. A future agent should not foreclose the data-axis direction without exploring more points (especially viso-targeted curricula like global-luma jitter, contrast jitter, or per-method aug).
>
> Also: S2 step600's 91.5% lockbox_fake_recall is a real partial win. The author downplayed this in the headline. **For deployments dominated by lockbox-substrate frames, S2 step600 is the strongest Teams ckpt produced in R13.** That's deployable progress, even though it doesn't satisfy the all-suites-at-90% target.

### What this opinion implies for next-packet planning

If you accept the "structurally exhausted" framing → next packet must change architecture (ViT-L-14), loss (contrastive viso pair), or data (enable visomaster_teams_v2_companion bucket).

If you accept the contrarian read → next packet should explore one more data-axis point: e.g., S4 = S2 with fw=12 OR S2 with viso-targeted aug (sharpness jitter instead of blur).

The author recommends the structural change direction (ViT-L-14 OR contrastive loss) because the data-axis has had three independent failures and the marginal return on a fourth attempt is low. But the contrarian read is defensible.

### What an independent agent should do

1. Read PSERIES_FACTS Sections 6.6–6.9 in full (pure data).
2. Look at `outputs/05_chain_joint_summary.csv` for the full trajectory table.
3. Review the falsifier verdicts in 6.9.6 — note that F-C was not directly probed.
4. Form independent opinion on whether the data-axis is truly exhausted or whether one more carefully-designed point is warranted.
5. Decide based on cost/budget: if budget tight, structural change. If budget generous, one more S4 data-axis point + structural in parallel.

---

## Cross-cutting opinions worth surfacing (NOT facts)

These appear in multiple handoffs and may shape future agent decisions. Marked as opinion-not-fact.

### "P8A is NOT shortcut-clean"

**Source**: PLAN.md §3.6, source-bucket linear probe test_acc 0.461 (4.6× chance, FAIL).

**Caveat**: the probe was on training data; the lockbox AUC of 0.79 (n=87) including identity-stripped version of 0.65 says P8A still has substrate-invariant signal at the deployment substrate. Both can be true: trained on shortcut-leaky data AND has signal beyond shortcuts. Calibrate accordingly.

### "Eval substrate has loose crop tightness vs production"

**Source**: PLAN.md §3.3, memory `project_eval_production_crop_tightness_gap.md`.

**Operational gist**: eval frames carry more background context than production crops. Implications: eval FPR may overstate production FPR for shortcut-driven flags. Closure (Move 1.5 / Move 2B retag): partially built (`analysis/eval_substrate_v3_retag_2026-05-01/`); not fully rolled out across all eval suites.

**Status**: real and important. Not blocking for the current "can't promote" pattern but should not be forgotten.

### "Webcam capture-mode dominates lockbox FPR"

**Source**: memory `project_lockbox_fpr_dominated_by_webcam_mode.md`.

**Numbers** (per memory): clip_capture_mode==webcam = 65.7% FPR; modern_v2 filter cuts headline 4.6% → 0.71% at calibrated 5% τ.

**Status**: known stratification. Webcam is the worst real-side, dor_shkedi was the worst false-flag identity (P8A fixed dor; webcam still drives stress-suite FPR).

### "There's a face-pixel-area label leak in training data"

**Source**: memory `project_face_size_label_leak.md`.

**Operational gist**: each fake method clusters at a tight face-size band; reals span wider; model uses face size as a fake predictor.

**Status**: real, partially addressed by `face_scale_jitter` (jitter@0.50 was the mclioexb winner on trainer composite but failed contract).
