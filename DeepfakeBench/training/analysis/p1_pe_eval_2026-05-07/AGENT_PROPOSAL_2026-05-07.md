# AGENT_PROPOSAL — Agent's staked view on P1 + next training regime, 2026-05-07

**This doc is OPINION, not RECORD.** It is the agent's interpretation, justification, and proposal for next steps. It is intentionally separate from the factual records (`DEEP_DIVE_FACTS_2026-05-07.md`, `FOLLOWUPS_FACTS_2026-05-07.md`) so that a reviewer can either:
- Read the factual docs first, form their own view, then compare to this proposal.
- Read this proposal first, then audit the cited evidence.

The recommendations here are the agent's best read of the data after one full evaluation cycle. **The reviewer is invited to disagree.** Each load-bearing claim cites specific numbers from the factual docs so the disagreement can be specific.

The doc is structured to make the agent's reasoning auditable: starts with confident headline claims, descends into more speculative ones, and ends with explicit counter-experiments that would falsify the proposal.

---

## 1. Executive summary

P1 (pair_rank + GroupDRO with chronic_flag, FT-from-P8A) **moves the model meaningfully but does not eliminate shortcuts — it shifts them**. The lever class is doing real work on 3 of 4 measured IQ axes (decoupling) and on the targeted chronic cluster (PC_Generator FPR collapses). It also introduces a new failure mode (Roy_D inflates from 29% FPR to 78-93%) which is **shared by both arms**, telling us the failure is in the shared lever (`pair_rank_loss`) or shared FT codepath, not in GroupDRO specifically.

**The single most underweighted finding from this evaluation:** F1 is reachable. BUNDLE_step500 hits 96.5% lockbox recall at FPR ≤ 10% (τ=0.989). PAIRRANK_step500 hits 91.5% at FPR ≤ 10% (τ=0.50). The contract algorithm picks a different τ that prioritizes dev_fake_macro_recall and gives up 12pp of lockbox recall. The strict "P1 fails F1" reading is contract-shape-dependent, not model-capability-dependent.

The proposal: **the next training regime should treat the FT base's behavior — not just its weights — as a constraint to preserve.** Specifically, add a "do-no-harm" auxiliary loss that penalizes per-identity score-distribution drift from the FT base, except on explicitly-targeted chronic groups. This would catch Roy_D-class regressions without sacrificing PC_Generator-class gains.

---

## 2. The 5 things I am most confident about

In descending confidence:

### 2.1 F1 is reachable at non-contract τ — high confidence
**Citations**: `FOLLOWUPS_FACTS_2026-05-07.md` §4 (lockbox τ curves). 86-point τ sweep × 8 ckpts × 1418 reals × 425 fakes.
- BUNDLE_step500: lockbox recall = 96.5% at τ=0.989, FPR=10.0%. F1 PASSES.
- PAIRRANK_step500: lockbox recall = 91.5% at τ=0.50, FPR=10.0%. F1 PASSES.
- The contract τ-selection algorithm (per `arena/score_teams_promotion_contract.py:485-499`) optimizes `(tier, macro_recall, threshold, primary_fpr)` lex-sort, NOT lockbox recall. BUNDLE_step500's contract τ=0.992 gives up 12pp lockbox recall vs the F1-passing τ=0.989.

**Implication**: The pre-registered F1 verdict ("FAIL for everyone") is a property of the contract's τ-objective alignment, not the model's underlying capability. If the goal is deployable lockbox detection, BUNDLE_step500 is in-band.

### 2.2 F5 (chronic-FP `pc_generator` cluster drop ≥ 0.10) PASSES big for BUNDLE — high confidence
**Citations**: `phase_d/pc_generator_cluster_fpr.csv` (after bug fix); per-identity table in `DEEP_DIVE_FACTS_2026-05-07.md` §6.
- P8A pc_generator FPR @ calibrated τ = 0.629 (200/318 chronic frames flagged false-positive).
- BUNDLE_step500 = 0.000 (0/318). Δ = +0.629 absolute drop.
- BUNDLE_step3750 = 0.006. BUNDLE_step4000 = 0.031.
- All 3 BUNDLE ckpts clear the +0.10 absolute drop bar by 60+pp.
- PC_Generator__s22 (the worst-P8A identity at 78.8%) drops to 0.0% in BUNDLE_step500. PC_Generator__s45 (P8A 23.1%) drops to 0.0%.

**Implication**: GroupDRO with `chronic_flag` is genuinely effective at suppressing the targeted chronic cluster. The mechanism works as designed.

### 2.3 Roy_D regression is `color_b_dev`-aligned, not axis-orthogonal — high confidence
**Citations**: `roy_d_regression/roy_d_with_axes.csv` (130 frames × 4 axes × 8 ckpts).
- P8A r(score, color_b_dev) on roy_d = −0.714 (highest single-axis r magnitude observed for P8A on this identity).
- For all 6 P1 ckpts, r(Δscore vs P8A, color_b_dev) = +0.582 to +0.714. The Δ-r magnitude on `color_b_dev` matches P8A's r magnitude. On `min_dim` and `sharpness` the Δ-r magnitudes are <0.30 and <0.13 respectively.
- The arithmetic implication: P1 ckpts unwind exactly the `color_b_dev`-related signal P8A learned on roy_d.

**Implication**: The Roy_D regression is not collateral noise. It has a specific feature-axis signature. Whatever P8A learned to use on roy_d (a relationship between B-channel std and real-vs-fake decision), the FT process erased it.

### 2.4 Roy_D regression is NOT GroupDRO-specific — high confidence (refutes my prior claim)
**Citations**: `counter_theory_per_identity_deltas.csv` (25 base_identities × 6 P1 ckpts at calibrated τ).
- Roy_D Δ FPR vs P8A: BUNDLE 500/3750/4000 = +0.638/+0.577/+0.562. PAIRRANK 500/6000/6750 = +0.492/+0.523/+0.485.
- Both arms regress on Roy_D with similar magnitude.
- On 9 non-target base_identities, mean |Δ| BUNDLE = 0.0058, PAIRRANK = 0.0080. Wilcoxon signed-rank p=0.875.
- BUNDLE has GREATER non-target regression than PAIRRANK on only 3 of 9 identities.

**Implication**: My earlier hypothesis "GroupDRO has balloon effect on non-targets" is REFUTED by this data. The regression mechanism is shared across both arms — most likely `pair_rank_loss` itself or the post-`2feea58` FT codepath, both of which apply equally to BUNDLE and PAIRRANK. **This is the most important self-correction in this evaluation.** I name it explicitly so a reviewer doesn't take my earlier framings at face value.

### 2.5 Shortcut-shift, not shortcut-removal — medium-high confidence
**Citations**: F3 audit across 4 axes; `FOLLOWUPS_FACTS_2026-05-07.md` §5 for sign-preservation detail.
- 3 of 4 measured untargeted IQ axes show DECOUPLING under P1 (real-side mean |r| reductions vs P8A): sharpness (decouples or stays flat), min_dim (-26 to -73%), color_b_dev (-75 to -100%).
- 1 of 4 axes shows AMPLIFICATION: face_area_fraction (+106 to +266% in absolute |r| terms).
- BUNDLE_step500 specifically FLIPS the sign of face_area_fraction on real-side (+0.052 vs P8A −0.098). Other P1 ckpts keep P8A's negative direction but with larger magnitude.

**Implication**: P1 doesn't eliminate IQ-axis reliance; it redistributes it. The strongest P8A shortcut (`min_dim`, |r|=0.491) is reduced. A weaker axis (`face_area_fraction`, |r|=0.117 P8A baseline) is amplified, in some cases with sign-flip. This is the "shortcut budget shift" pattern I claim — but with one P1 ckpt (step500) executing it via direction-flip rather than magnitude-only amplification, which complicates the simple "swap one axis for another" reading. (See §3 for the proposed mechanism.)

---

## 3. The mechanism I propose

**Core claim**: P1's `pair_rank_loss` is a fake-score-raising lever that operates over an eval substrate where pairs are loose. At training time, pair_rank fires on tight `(sample_id, frame_idx)` pairs in 6 paired training lanes. At eval time on Phase A's substrate, only `deeplive_teams` lane is represented, and within it pair_rank's training-time signal generalizes by pulling fake scores up everywhere — including on real frames belonging to identities NOT in the paired lanes (e.g., Roy_D).

**Why I think this is the right read**:
1. PAIRRANK-only and BUNDLE both regress on Roy_D with similar magnitude → the regression is shared across the two arms' common pieces.
2. The shared common piece besides pair_rank is the FT-from-P8A path itself (anchor_aware, stability, post-`2feea58` SVD lever) — but those are present in PAIRRANK_ONLY's predecessor (PC) packets without similar regressions.
3. Pair_rank specifically reshapes the loss landscape to push fake-class score above real-class score within a pair. If the model's representation collapses paired-lane patterns onto non-paired-lane identities (because those identities resemble the paired-lane ones in some feature subspace), pair_rank pulls those non-paired identities' scores up too.

**Predicted secondary effect**: pair_rank's score-raising should be larger for identities that are more out-of-distribution relative to the paired training lanes. Roy_D may be exactly that.

**What this mechanism does NOT explain by itself**:
- Why BUNDLE_step500 specifically flips the face_area_fraction sign while step3750/4000 amplify in the same direction (sign-preserving). This suggests BUNDLE_step500's score distribution is in a different regime than later steps — possibly an under-converged calibration that hasn't yet specialized to face_area magnitude.
- Why color_b_dev's correlation reverses sign in BUNDLE_step500 (+0.195) but not in others. Same calibration explanation may apply.

**What I am unsure about**:
- Whether the Roy_D color_b_dev signal in P8A is generalization (if Roy_D was OUT of P8A's training set) or memorization (if Roy_D was IN). I haven't checked. If Roy_D is in P8A's training, the color_b_dev=>real association may be a memorization that's no longer robust under FT. If Roy_D is held-out, the association is generalization, and losing it under FT is a more substantive regression.

---

## 4. The error I made earlier — the GroupDRO balloon hypothesis

In an earlier message I wrote "GroupDRO has a balloon effect — it inflates non-target identities like Roy_D as a side-effect of suppressing PC_Generator". The counter-theory probe (§6 of `FOLLOWUPS_FACTS_2026-05-07.md`) refutes this:

- BUNDLE arm (with GroupDRO + chronic_flag) and PAIRRANK arm (without GroupDRO) have indistinguishable non-target regressions: Wilcoxon p=0.875.
- The Roy_D Δ magnitude is similar between arms (BUNDLE +0.59 mean, PAIRRANK +0.50 mean).

This was a mechanism claim I made before computing the matched-arm comparison. I had data on chronic-target gains (BUNDLE > PAIRRANK on PC_Generator) and treated it as evidence for a BUNDLE-specific balloon effect. The corresponding test on non-targets refuted the directional asymmetry.

**What this teaches me to do differently**:
- When proposing a mechanism that says "X causes Y because of property Z", explicitly run the counter-test where Z is absent or controlled (here: the PAIRRANK-only arm).
- Don't treat "I observe a coincidence between A and B" as evidence for "A causes B" without a control.
- A reviewer should weight my mechanism claims in §3 and §6 with this calibration in mind. I have one demonstrated track record of mechanism-claim retraction in this same evaluation cycle.

---

## 5. Confidence-tiered claims

Tagged so a reviewer can disagree at the appropriate level. "High" = direct from data with effect size ≥ 5x noise. "Medium" = data plus one inference step. "Low" = inference plus extrapolation.

| claim | confidence | basis |
|---|:---:|---|
| F1 (lockbox recall ≥ 90% at FPR ≤ 10%) is reachable at non-contract τ for BUNDLE_step500 and PAIRRANK_step500 | HIGH | direct τ-sweep on 1418×425 frames |
| F5 (pc_generator drop ≥ 0.10) PASSES for all 3 BUNDLE ckpts by 60+pp | HIGH | 318 frames, large effect |
| Roy_D regression is `color_b_dev`-aligned (Δr magnitude matches P8A's r magnitude) | HIGH | 130 frames, Δr=+0.71 vs P8A r=−0.71 |
| Roy_D regression is NOT GroupDRO-specific | HIGH | Wilcoxon p=0.875 |
| F4 (HDTF FPR ≤ 5%) PASSES universally at calibrated τ | HIGH | 4 HDTF real suites × 8 ckpts, max=1.4% |
| P1 reduces real-side coupling on 3 of 4 IQ axes (sharpness, min_dim, color_b_dev) | HIGH | F3 audit + min_dim + color_b_dev extension |
| `pair_rank_loss` is the load-bearing lever for both gains and losses on non-targets | MEDIUM | inferred from arm-symmetry under counter-theory probe |
| The Roy_D color_b_dev association in P8A is genuinely-learned generalization (not memorization) | LOW | not directly tested; needs P8A train-set audit |
| Shortcut budget conservation across FT chains is a structural property of FT-from-FT | LOW | n=1 packet; would need historical replication |
| FT-from-CLIP-direct (skipping P8A) would yield different regression patterns | LOW | counterfactual; not run |

---

## 6. Proposed next steps — broken down by lever class

The user asked for broad-strokes proposals. I list them here with my confidence in each. **The reviewer should weight against the per-claim confidence in §5.**

### 6.1 Loss-function changes (highest leverage)

1. **(HIGH) Per-identity score-distribution preservation auxiliary loss.**
   For each base_identity in the train data, the FT model's score distribution on real frames should not drift more than ε (e.g. KS distance ≤ 0.10) from the FT-base's distribution, EXCEPT on identities marked as chronic-target.
   - Catches Roy_D-class regressions before training ends.
   - Doesn't sacrifice PC_Generator-class gains because chronic-targets are explicitly exempt.
   - Implementation: precompute FT-base scores once on training reals, then add a small auxiliary loss term (KL or KS) on per-identity score histograms during FT.
   - Risk: identity-stratified loss is sensitive to small-n identities. Add a min-frame threshold (e.g., n ≥ 30 per identity).

2. **(MEDIUM) Pair_rank with non-paired-lane anchor.**
   Augment pair_rank_loss with a "preserve" term that fires on real frames from identities NOT in any paired training lane. The constraint: such frames' scores should not move more than δ from FT-base.
   - Targets the failure mode I propose in §3 directly.
   - Risk: this can over-anchor the model and prevent useful FT updates.

### 6.2 Eval & monitoring (cheap to add, high signal)

3. **(HIGH) ROC trade-width metric per training step.**
   At each eval step during training, compute "max recall achievable in the FPR ≤ 0.07 budget on dev_fake_macro" — the in-budget max-recall point on the threshold grid.
   - Catches BUNDLE_step3750/4000 ROC degeneracy mid-training.
   - Single scalar; trivial to log.

4. **(HIGH) Per-identity score-distribution drift monitor.**
   At each eval step, log the KS distance of per-identity score distributions vs FT-base on a held-out diagnostic substrate.
   - Catches Roy_D-class regressions before final ckpts get saved.
   - Could be combined with #1 as the loss-side signal.

5. **(MEDIUM) F2 close-criterion redesign.**
   The current F2 ("≥30% relative lift on ≥2 of 5 paired lanes among previously-missed-fakes") is structurally not testable on Phase A substrate (1 of 6 lanes represented; baseline saturated). Alternatives:
   - "Lift on per-frame pair gaps where P8A is confident-wrong" — uses P8A's score itself, doesn't depend on training-loader pair structure.
   - "Per-identity pair gap distribution drift" — test whether the (fake_score − real_score) gap distribution shifts in the desired direction per identity.

### 6.3 Training regime changes (longer horizon, higher cost)

6. **(MEDIUM) Multi-FT-base study.**
   Run pair_rank+GroupDRO from THREE FT bases as a single packet:
   - From P8A (current).
   - From E2B (different lineage; from-scratch B16+CE).
   - From CLIP-DataComp-XL pretrained directly (no intermediate FT).
   Compares regression patterns and tests whether FT-from-FT depth matters per se.
   - Cost: ~3x training time. Need GPU.
   - Counter-bias: this is more balanced than just running another P-from-P8A.

7. **(LOW-MED) Synthetic IQ-axis grid as training data.**
   Generate fake/real pairs that match across an IQ-axis lattice (sharpness × min_dim × face_area × color_b_dev × …). Train on this only.
   - Tests whether shortcut-elimination is achievable with controlled data alone (no loss-axis interventions).
   - Cost: substantial data engineering. Probably out of scope for short-horizon planning.

### 6.4 Investigation / instrumentation (cheap, valuable)

8. **(HIGH) Audit P8A's training-set membership of Roy_D and the chronic-6 identities.**
   If Roy_D was in P8A's train, the color_b_dev association may be memorization. If held-out, it's generalization and the regression is more concerning.
   - Cheap: needs only a train-manifest grep.
   - Can change my §3 mechanism interpretation.

9. **(MEDIUM) trainer.py:1727 fix + replay 100 batches against BUNDLE checkpoint.**
   The fix (already applied this session) preserves diagnostic loss components. To recover the lever-activation magnitude for the existing P1 BUNDLE run without retraining: replay a few batches against the checkpoint with the same data-loader seed. This tells us whether `pair_rank_loss` actually fired non-trivially in BUNDLE.
   - Required to compare BUNDLE vs PAIRRANK lever activation directly, not just via downstream effect.

---

## 7. Counter-experiments that would falsify my proposal

The user explicitly asked for these. I list experiments whose outcomes — if they came out a particular way — would mean my §3 mechanism is WRONG. A reviewer who suspects I'm biased should weight these heavily.

1. **PAIRRANK-only arm at additional τ-points and on different substrates.**
   If PAIRRANK_step500 has Roy_D regression that vanishes at lower τ (i.e., Roy_D scores stay below 0.5 even though they're above τ_selected=0.768), the regression is τ-dependent rather than feature-representation-dependent. That would falsify my "pair_rank pulls scores up" claim.

2. **Cross-substrate Roy_D test.**
   Run the color_b_dev axis attribution on a different real substrate (e.g., a held-out identity-fresh substrate) for P8A and one P1 ckpt. If P8A's color_b_dev=>real association is roy_d-specific (not present on other identities), then my "P8A learned a useful identity-specific feature" framing is wrong; instead it's a statistical artifact of roy_d being unusual.

3. **From-scratch FT-CLIP run with pair_rank only.**
   If a from-scratch CLIP+pair_rank model (no P8A intermediary) ALSO regresses on Roy_D with similar magnitude, then the regression is in the data/lever, not the FT-from-FT chain depth. My "FT-from-FT accumulates baggage" framing weakens.

4. **Shortcut-substitute-vs-shortcut-eliminate test on a SYNTHETIC dataset.**
   Train a model on a synthetic dataset where IQ axes are perfectly orthogonalized (each axis has equal fake/real distributions). If the model still produces a measurable shortcut on some axis, then "shortcut budget conservation" is intrinsic to the model class — not a property of the data or the FT regime. My "data-axis balancing would eliminate shortcuts" claim weakens.

5. **Distillation-augmented FT vs vanilla FT head-to-head.**
   If adding a distillation-from-P8A loss does NOT prevent the Roy_D regression, my §6.1 #1 proposal is wrong on the mechanism even if it's right on the diagnosis.

---

## 8. Open questions I cannot answer from this data

- Is the Roy_D regression unique to P1, or did earlier packets (PA, PC, PD) show the same pattern on Roy_D? Not run; would need PD per-identity audit at calibrated τ.
- Did any P1 BUNDLE training-step show "unstable" lever activation (e.g., warmup-active transition catching the model in a bad regime)? Not testable until trainer.py:1727 fix is rerun.
- What identities BESIDES Roy_D regressed under P1? The counter-theory probe surfaced only top-25-by-frame-count. There may be smaller-n identities with worse regressions invisible at this aggregation.
- Is the `face_area_fraction` direction-flip in BUNDLE_step500 a transient training-stage artifact or a stable property of the early-FT regime? Would need finer step-resolution between step100 and step1000.
- What does P8A's score distribution look like on the F1-passing-but-not-contract-selected τ=0.989 region vs τ=0.50? If P8A's lockbox recall plateaus across τ in [0.5, 0.95] but P1's recall gains are concentrated in [0.97, 0.999], the "F1 reachability" of P1 is qualitatively different from P8A's — possibly a more useful operating mode, possibly a more brittle one.

---

## 9. Suggested CPU jobs not yet run (balanced — some test my view, some don't)

The reviewer can pick any of these. Listed roughly by signal-per-token:

### Tests that would refine my proposal (confirm or falsify):

A. **PD-vs-P1 per-identity comparison**: same per-identity regression analysis applied to PD ckpts. If PD also regresses on Roy_D, the regression is older than P1; my "pair_rank caused it" claim weakens. If PD doesn't, P1 introduced it.

B. **P8A train-set membership audit for chronic-6**: grep the train manifest for `roy_d`, `bla_bla_chow`, `PC_Generator`, etc. Tells us whether P8A's behavior is generalization or memorization.

C. **Cross-substrate axis attribution for color_b_dev**: re-run roy_d's color_b_dev computation on a different substrate (live_reals_teams_prod or any substrate where Roy_D doesn't appear). Tests whether the color_b_dev pattern is roy_d-specific or P8A-general.

### Tests that are more theory-neutral:

D. **Weight-delta-vs-axis-decoupling correlation**: do Phase E weight-delta magnitudes per layer correlate with axis-decoupling magnitudes? Bridges "what changed in the weights" and "what changed in feature reliance".

E. **Per-identity ROC-degeneracy probe**: same trade-width analysis (Task D) but stratified by identity. Would tell us whether ROC degeneracy is global or identity-specific.

F. **Joint axis decomposition**: what's the correlation MATRIX between the 4 IQ axes (sharpness, min_dim, face_area_fraction, color_b_dev) on training data? If two are highly correlated, "model trades A for B" may just be redirecting through correlated dimensions, not finding new shortcuts.

### Tests that lean toward my proposal:

G. **PAIRRANK-only τ-sweep on Roy_D**: at what τ does PAIRRANK-only's Roy_D FPR drop below P8A's 29%? If lowering τ recovers Roy_D for PAIRRANK but not BUNDLE, the regression is calibration vs. feature.

H. **Per-identity score-distribution KL FT-vs-FT-base**: compute the actual KL divergence per identity. Confirms whether Roy_D's KL is an outlier or just one of many.

---

## 10. What a reviewer should do with this doc

Suggested workflow:

1. **First, read `DEEP_DIVE_FACTS_2026-05-07.md` and `FOLLOWUPS_FACTS_2026-05-07.md`** without reading this proposal. Form your own interpretation.
2. Then read this proposal and note: where does the agent's interpretation match yours, and where does it diverge?
3. **For divergences, audit the cited evidence.** Each load-bearing claim in §2 and §5 cites a specific data file and effect size. The evidence may be weaker than the agent thinks.
4. **Test the §7 falsifiers.** If any of those experiments would change your view, propose them as next-priority CPU jobs.
5. **Calibrate against the §4 self-correction.** I demonstrated one mechanism-claim retraction in this evaluation. There is a non-zero probability of more retractions hiding in §3 and §6. The reviewer is warmly invited to find them.

The agent commits to:
- Updating this proposal if a reviewer surfaces evidence that changes the picture.
- Re-running any of the §7 or §9 experiments if the reviewer flags them as decision-relevant.
- Treating the reviewer's independent read as the higher-priority view when there's disagreement on interpretation, not when there's disagreement on numbers.

---

## 11. Cross-references

- Factual records: `RESULTS_FACTS_2026-05-07.md`, `RESULTS_F1_F5_FACTS_2026-05-07.md`, `DEEP_DIVE_FACTS_2026-05-07.md`, `FOLLOWUPS_FACTS_2026-05-07.md`.
- Sub-investigations: `f2_pair_rank/F2_PAIR_RANK_FACTS_2026-05-07.md`, `f3_color_b_dev/F3_COLOR_B_DEV_FACTS_2026-05-07.md`, `roc_degeneracy/ROC_DEGENERACY_FACTS_2026-05-07.md`, `roy_d_regression/ROY_D_REGRESSION_FACTS_2026-05-07.md`.
- Bug-fixed scripts: `trainer/trainer.py:1727+` (W&B logging), `phase_d/run_chronic_filter.py` (chronic-id matching).
- Memory entries that this proposal would propose REVISITING (after reviewer pass):
  - `project_p8a_breakthrough.md` — should P8A be the FT base, or has its accumulated FT chain become a liability?
  - `project_deployment_is_e2b_2026-05-06.md` — E2B beats P1 on dev_fake_macro; P1 BUNDLE_step500 beats E2B on lockbox at non-contract τ. Deployment should be re-evaluated against P1 candidates at non-contract τ.
  - `project_promotion_contract.md` — the contract τ-objective alignment is the bottleneck for F1 reachability. Should be re-examined.
