# Independent Review — Structural Reframe Proposal (2026-05-23)

**Reviewer**: independent agent, dispatched 2026-05-23 PM
**Reviewing**: `docs/packet_retrospectives/plans/STRUCTURAL_REFRAME_PROPOSAL_2026-05-23.md`
**Stance asked of reviewer**: adversarial, no rubber-stamp

---

## 1. Executive verdict

**Partial agree on the diagnosis, modify the experiment, and there is a higher-priority operational action that the proposal under-weights.**

The structural-reframe diagnosis is *directionally* correct — six weeks of single-lever shortcut suppression have not moved the binding gate, and the Probe 1 finding (per-ckpt substrate axes at ~98% accuracy, ~85-88° from frozen prior) is a genuinely load-bearing piece of evidence for a structural ceiling. But the proposal **overstates** the inevitability claim (the lever space is not exhausted — at least three structurally-distinct levers inside the current data + task remain untested), and it **understates** how much of today's failure traces to the cross-ckpt τ comparison being unfair to specific candidates rather than to any of them being deployable.

On the experiment: **B1+C1 combined is the wrong shape**. IRM is known-unstable, VIB is known-prone-to-encoding-shortcut-in-the-mean, both have multiple sensitive hyperparameters, and combining them obscures attribution. Spend ~$30 on a one-arm IRM smoke first (per-method environment partition, not per-substrate), and gate the C1 / combined runs on whether IRM smoke shows the penalty actually biting (gradient variance reduction) and learning hasn't collapsed.

Highest-priority *operational* item the proposal under-weights: **xinhe_may6 T5C revisit is in and the result is bad** (17.4% FPR @ τ=0.5; 16.3% @ mode A). T5C is the currently-deployed model and it is materially worse than P8A on the same cohort that motivated the 2026-05-06 "retire E2B" memo. Switching production to P8A_step5000 (or face-pool-on-Slot-Av2 with caveats) is a same-day no-GPU-cost action and matters more than the structural experiment.

---

## 2. Diagnosis review (§5 of proposal — is "structural ceiling" well-supported?)

### What the evidence does support

1. **Probe 1 is real and load-bearing.** `analysis/substrate_pair_geometry_2026-05-22/FALLBACK1_PROBE1_FACTS_2026-05-22.md` §1.2: P8A 0.9804, SlotAv2 0.9836, T5C 0.9836 held-out accuracy of a per-ckpt substrate discriminator; cosines 0.04-0.10 with the frozen-CLIP axis; per-ckpt projection 9-23× stronger than frozen axis projection. Three independent FT recipes converged on substrate-axis classifiers that are nearly orthogonal to the frozen prior and rotate differently from each other. This is consistent with "FT preferentially finds shortcuts."

2. **D7 confirms the IQ block dominates FPR variance.** `analysis/cpu_diagnostics_2026-05-12_d7_fpr_decomposition/D7_FACTS_2026-05-12.md` §5.2: block-drop ΔR² for IQ is +0.236 (P8A) / +0.314 (T5C); substrate-distance contributes ≤0.015 / 0.0001; chronic contributes 0.014 / 0.062. IQ is ~10-300× more load-bearing than the other two blocks. The shortcut taxonomy in §3 of the proposal is well-grounded in this kind of decomposition.

3. **The refuted-lever catalog is mostly accurate.** I spot-checked three: LoRA L10-L11 (`project_lora_l10_l11_chronic_fp_hard_coupled_2026-05-15` — confirms 88% viso vs 24% lockbox_real_fpr; rank/data variation doesn't move it), resolution_chain_aug Slot α (`project_overnight_resolution_chain_2026-05-16` — confirms 0.07 per-size swing and dev_fake_macro_recall failure at the v3 floor of 0.30), HEAD face-pool (`project_head_face_pool_verdict_iterate_2026-05-23` — confirms the plateau at composite 0.235). The "anchor_aware = the one win" framing in the catalog is also accurate per `project_band_shortcut_ood_hypothesis_2026-05-16`.

### What the evidence does NOT support as strongly as the proposal claims

1. **"Whack-a-mole with structural inevitability" overstates.** The proposal §5.1 claims the encoder will always find shortcuts "as long as any shortcut is easier to learn than the true signal." That's a tautology — but the actual question is whether the *specific* easier-to-learn shortcuts have been suppressed *jointly* yet. The catalog in §4 is mostly single-lever interventions. There is no entry for **multi-lever suppression at training time** (e.g., anchor_aware + resolution_chain_aug + face_scale_jitter composed). The "stacking jitter on T3/T4 regresses dev_macro" finding in memory `project_face_scale_jitter_load_bearing` rules out one specific stacking — but the proposal generalizes that to "compose doesn't work," which the evidence doesn't support broadly.

2. **The Probe 1 reading deserves a counter-interpretation.** Probe 1 shows FT moves the encoder in directions different from the frozen prior AND different from each other. The proposal reads this as "different FT recipes find different substrate-shortcuts." An equally valid reading: each FT recipe partially learns the substrate axis AND partially learns the task axis, and the residuals (the differences between encoders) reflect which non-shortcut signal each recipe picked up. The KLIEP-fitted-on-each-ckpt classifier achieves 98% accuracy on the *substrate* task — but that doesn't tell you what fraction of the encoder's capacity is going to substrate vs task. The proposal needs a complementary probe: "in the same per-ckpt-space, how accurately can we predict the true binary label?" If accuracy is ≥99%, the encoder isn't capacity-constrained between the two — substrate-discrimination is *additional* capacity, not crowding out task-discrimination.

3. **D9 cited support is imprecise.** Proposal §6.A says "100%-YouTube-origin real-side training distribution (per D9 2026-05-12)." D9 §5 actually shows training reals split across three substrate classes: youtube_origin (`teams-v2` lane, 4,908 frames, 7.8%), hdtf_corpus (35,008 frames, 55.3%), qclips_corpus (23,424 frames, 37.0%). All are YouTube-derived, but they sit in three different capture pipelines. This **strengthens** the data-acquisition argument (the 92% in proper_data lane is *not* the production substrate either) but also means **the IRM environment partition has more structure available than the proposal acknowledges** — see §3.3 of this review.

4. **"6 weeks, no movement" is partially true.** P8A → E2B switch (some date pre-2026-05-23) was movement. Slot A v2 anchor_aware fixed Chikara/PC_Generator/Q chronic FPs (26→0%, 28→0%) which is real, durable improvement at the per-identity level. The "no movement" framing is true *on the team-identity gate established today*, but it's the gate that was just defined today — making the failure-against-it 6-week-old is partly retrospective. This doesn't refute the structural argument but does temper the urgency framing.

### Verdict on diagnosis

Partial-agree. The structural ceiling claim is supported well enough that "try a structural reframe" is rationally next. But the claim of *inevitability* — that no remaining single-lever inside the current paradigm could work — is not as well-supported as the proposal presents. Specifically, three things have not been tried that fit inside the current paradigm: (a) multi-lever stacking at training time with proper hyperparameter search, (b) output-preservation aux loss against a frozen anchor (proposed never run, per catalog), (c) substantially-different base ckpts (P22 step1k, which is the robust P22 winner per `project_p22_cpu_followups_reframe_2026-05-02`, has not been used as the base for any anchor_aware or substrate-suppression variant).

The diagnosis is "structural reframe is one rational next step among 2-3 candidates," not "structural reframe is the only rational next step."

---

## 3. Experiment review (§7 — is B1+C1 the right specific experiment?)

### Combining B1+C1 in one run is wrong

The proposal acknowledges (§10.1.4) that "compound experiment makes attribution harder" and proposes ~$30-50 follow-up ablations. But the deeper problem is that **both methods have known instability and known failure modes that interact**:

- **IRM**: the penalty can dominate CE, especially with narrow environment partitions. The proposal mitigates with β-annealing, but doesn't specify the schedule precisely, doesn't specify abort criteria, and doesn't address the well-documented finding (Rosenfeld 2020, "The Risks of Invariant Risk Minimization") that IRM can be brittle to environment count and recovery requires either V-REx or REx-like extensions.

- **VIB**: the known failure mode is "encoder encodes shortcut in the mean, saturates variance to satisfy KL" — the proposal lists this as risk 2 but its detection mechanism ("MI between μ and known shortcuts at end of training") is post-hoc, not a training-time abort criterion. If VIB is doing the wrong thing, the IRM penalty is the only thing keeping the run honest, and you've stacked two penalties whose interaction is undocumented.

- **Interaction**: IRM applies its penalty to gradients of classifier outputs; VIB applies its penalty to the encoder representation distribution. The IRM penalty's environment-invariance signal is computed on samples z ~ N(μ, σ²). With VIB active, the IRM gradient itself becomes stochastic — increasing variance of the very gradient quantity IRM is trying to minimize. This is **mathematically incompatible without care**: IRM's penalty is sensitive to gradient noise; VIB injects gradient noise by design. Neither paper's empirical results combined them; combining is novel and the combined behavior is not predicted by either theory.

### What I'd run instead

**One-arm IRM smoke first ($25-30, ~1.5h), with:**
- Per-method environment partition, not per-substrate (12 method classes per `project_phase1a_method_cluster_axis_2026-05-01` — IRM with 12 environments is in DomainBed's reliable regime; 2-environment IRM is the known-pathological regime per Arjovsky 2019 §6)
- β=1.0 starting, with anneal-down on loss explosion (not anneal-up — the literature consensus is that anneal-up exposes the penalty to a representation that already learned shortcuts, and the penalty can't reverse them; anneal-down lets the model first satisfy the constraint, then learn)
- Two abort criteria: (1) train CE loss diverges (>0.5 increase from baseline), (2) substrate-pair gradient variance reduction < 10% by step 500
- Base: P22 step1k (per `project_p22_cpu_followups_reframe_2026-05-02`, the robust P22 ckpt with widened distributions). Slot A v2 step3500 is a fine alternative if the user prefers continuity, but P22 step1k has the substrate-representation property IRM most needs to bite on.

If IRM smoke shows the penalty bites without collapsing CE, **then** consider C1 as a separate $30 run, NOT a combined run.

### Alternative experiments worth considering at similar cost

- **Output-preservation aux loss** ($50-80): freeze a reference encoder (Slot A v2 step3500), regularize the FT encoder's penultimate-layer features toward the reference on a held-out reference set. This is structurally distinct from IRM (preserves what was learned, doesn't add invariance constraint) and is proposed-never-run per the catalog.
- **P22 step1k + anchor_aware** ($50): the proposal lists P22 in the catalog but only mentions step1k as forbidden ensemble. Step1k as an FT *base* for the next packet has not been tried. Cheap.
- **Per-method IRM-only** ($30) as outlined above.

None of these are slam-dunks. But "1 specific compound experiment with high attribution cost" is not the right shape when the proposal itself acknowledges (§10.2) that the diagnosis could be wrong about exhaustion.

### Substrate-pair environment partition concern

The proposal's environment partition is per-substrate (2 envs, "small partition, most direct"). But:
- Probe 1 already showed substrate-discriminability is high; the IRM penalty on 2 envs will land on whatever residual the encoder hasn't yet collapsed
- The 1,880 substrate-pair instances are ~0.2% of training data per D9 inventories
- 2-environment IRM is the configuration most prone to penalty-domination (DomainBed empirical: IRM is stable above ~5 envs, brittle below)

Per-method partition (~12 envs from `project_phase1a_method_cluster_axis_2026-05-01`) is structurally better-motivated.

### Success criteria critique

The "informative-even-if-failure" criterion (§7) is well-designed in spirit but specifies "substrate-pair gradient variance reduced by ≥30% vs baseline." Per the IRM literature, gradient-variance reduction is not directly predictive of OOD-generalization — DomainBed shows IRM frequently passes its own internal penalty target while failing the OOD AUC. The proper informative-failure criterion should be "team-identity bar fails AND held-out OOD AUC on substrate-pair test set improves vs baseline." If both, IRM is doing the right thing on the wrong data → data-acquisition is the binding constraint. If first but not second, IRM is gaming its own penalty → IRM isn't the right method.

---

## 4. Alternatives review (§8 — was alternative-consideration exhaustive?)

### What §8 covered well

HEAD ALT, additional GroupDRO, additional LoRA placement, Roy_D-specific anchor, temporal (B2), per-method aux (B3), self-supervised pretrain (B4), MI-regularizer (C2), small bottleneck (C3) — these cover most of the obvious lever space.

### Missing alternatives that should have been considered

1. **Output-preservation aux loss against a frozen anchor** (the proposal lists it in §4 as "proposed never run" but doesn't engage with it in §8). Cost: ~$50. Mechanism: prevents FT from drifting away from a known-good representation. The Probe 1 finding *directly motivates* this — if FT moves the encoder 85-88° away from the frozen prior, regularize against that drift on a held-out reference pool. This is structurally distinct from IRM (not a per-environment invariance constraint) and from VIB (not a bandwidth constraint), and addresses the Probe 1 finding more directly than either.

2. **P22 step1k as the FT base** for anchor_aware (or substrate-balanced or any new lever). Memory `project_p22_cpu_followups_reframe_2026-05-02` shows P22 step1k has 140× wider score variance than step8k and is the "robust" P22 ckpt — but it has never been used as a base for downstream FT. The base ckpt is itself a load-bearing lever that the catalog under-weights.

3. **Replace training data, not just augment** — the proposal mentions ~7000 webcam-video dataset acquisition but frames it as data-side (A). What's *not* mentioned: filtering the existing 63,340-frame training real pool by per-frame similarity to the deploy distribution, then training only on the in-distribution subset. This is essentially Estimator C from D8 (KLIEP-on-features at the per-frame level) applied as a training-time filter, not a head-retrain post-hoc reweight. Cost: ~$60 (one FT run). D8 already showed the KLIEP discriminator has effective sample size 269/2000 — i.e., ~13% of training reals are "near" the deploy distribution. Training on those frames only (with appropriate class balancing) is testable without new data acquisition.

4. **Contrastive pretrain on the substrate-pair data itself** (different from B4's self-supervised pretrain on raw video). The 1,825 matched (clean, teams) pairs are *a labeled invariance signal* — pretrain the encoder with a contrastive objective that pulls (clean_i, teams_i) closer and pushes apart different identities, regardless of substrate. Then FT for binary classification. Cost: ~$50 pretrain + $50 FT. Mechanism: directly encodes the invariance Probe 1 says we want, without depending on IRM's specific gradient-variance penalty.

5. **Smaller / different architecture as the encoder.** B16 is the default; L14 was a capacity test that closed. But the proposal never asks "would a smaller-capacity B16-frozen encoder + larger trainable head help?" or "would B32 (smaller patch) help?" Smaller encoders are less prone to shortcut overfitting (per the IB literature). Cost: ~$80 for a B32 FT trial.

### Verdict on alternatives

§8 covered the obvious near-neighbors but missed:
- Output-preservation aux loss as a specific structural lever
- Different base ckpt (P22 step1k) for the same lever
- Training-data subsetting as a training-time filter
- Substrate-pair contrastive pretrain
- Architecture choice as a lever

Three of these (1, 3, 4) directly address the Probe 1 finding more pointedly than B1+C1.

---

## 5. Probe-outcome integration

### Xinhe-may6 T5C revisit — COMPLETE, RESULT IS LOAD-BEARING

`analysis/xinhe_may6_t5c_revisit_2026-05-23/outputs/fpr_by_mode.txt` is in. Headline:

| Ckpt | may6_falseflag (92 frames) FPR@τ=0.5 | mode A | mode B | mode C |
|---|---:|---:|---:|---:|
| P8A | 0.0% | 0.0% | 0.0% | 0.0% |
| E2B | 57.6% (reproduced) | 55.4% | 31.5% | 20.7% |
| **T5C** | **17.4%** | **16.3%** | **3.3%** | **0.0%** |
| Slot A v2 (CLS) | 4.3% | 4.3% | 1.1% | 0.0% |
| Slot A v2 (face-pool) | **91.3%** | **85.9%** | **0.0%** | **0.0%** |

Two implications the proposal needs to absorb:

1. **T5C is materially broken on Xinhe-may6.** Not at the catastrophic level of E2B, but at 17.4% @ τ=0.5 / 16.3% @ mode A. The deployed model false-flags Xinhe in 1 in 6 frames at the recall-leaning τ. The structural reframe conversation is correct to proceed, but **switching production off T5C is the urgent same-day item**, not a deferred concern.

2. **Slot A v2 face-pool is *catastrophically* broken on Xinhe-may6** (91.3% @ τ=0.5; 85.9% @ mode A; 0% @ mode B). The face-pool inference baseline that's been treated as a $0 Pareto improvement (per memory `project_face_pool_scorecard_pareto_2026-05-22`) does *not* generalize to Xinhe-may6. The "deploy face-pool inference at $0" recommendation needs a Xinhe-may6 gate before being acted on.

The proposal §9.2 anticipated this probe and listed two outcomes; the result is closer to the catastrophic outcome than the at-least-stable one. Operational urgency is real. The structural experiment is fine to run alongside, but **the production-switch decision is the higher-priority item.**

### Frozen-CLIP team-identity baseline — STILL EXTRACTING (1,280/5,941 at write time, ETA ~15 min)

Not landed yet. The proposal's §9.1 decision-relevance framing is reasonable. If the result lands with frozen-CLIP within ~10pp of T5C on team-aggregate metrics, **the data argument moves up substantially** — would shift this reviewer's recommendation toward prioritizing (A) data acquisition over the B1+C1 experiment. The proposal accommodates this contingency adequately.

Recommend the user wait ~15-20 min for this probe before final-decision on B1+C1, since the result materially changes which lever-class is highest-EV.

---

## 6. Risks the proposal under-weights

1. **Cross-ckpt τ comparison is unfair to specific candidates.** RESULTS_FACTS §3 explicitly caveats (Caveat 2): "τ values are cross-ckpt constants, not per-ckpt-calibrated." Slot A v2 face-pool has shifted scores (mean real ~0.35-0.50 vs CLS ~0.07-0.17). The "no ckpt passes both gates at mode B" headline is partly a τ-calibration artifact. Per-ckpt τ-recalibration would likely move Slot A v2 face-pool through the gate — and is the same kind of work that gets called "promotion contract" elsewhere in the program. The proposal accepts the cross-ckpt comparison at face value and concludes "structural failure," when it should first try per-ckpt re-calibration ($0, 1-hr CPU job).

2. **The team-identity bar (5%/50%) is brand new and not user-validated.** Per the proposal's own §10.3, the bar may be misspecified. AGENT_PROPOSAL §6 of the team-deploy readout shows P8A at mode B misses Xinhe fake recall by 0.013pp on n=1099 — within sample noise of the 0.50 floor. Relaxing the floor to 0.40 (a reasonable user-allowed revision per the readout's own discussion) gives P8A at mode B both-pass. The proposal treats the bar as load-bearing for "structural ceiling" — but the bar's 50% floor on a single human is arbitrary, and the conclusion is fragile to that arbitrariness.

3. **Operational sequence risk.** Per proposal §1, the $80 B1+C1 experiment is "cheap-to-medium GPU cost." But §10.1.4 also notes that compound-experiment ablation requires "$30-50 each" follow-up runs. Realistic total cost trajectory if B1+C1 lands partial-positive: $80 + 2×$40 = $160. Realistic if it lands ambiguous: $80 + 3-4 ablations = $200-240. The proposal's cost framing is the floor, not the expected value.

4. **6 weeks of "20 single-lever interventions" is itself evidence of a process problem.** The proposal frames this as data-establishing-structural-failure. An equally valid framing: the program has been running too many independent low-power experiments without enough cross-design discipline. The "lever stacking has not been tried" gap (per §2 of this review) suggests an attribution-bias issue: every packet was scoped to one lever-vs-baseline, so cross-lever combinations never got proposed. A structural reframe doesn't fix that process problem; switching to a more deliberately-staged search regime does.

5. **The "B1+C1 will fail informatively" argument is sneaky.** The proposal frames B1+C1's downside as "even if it fails, it informs the data-acquisition decision." This is the kind of framing that makes ill-specified experiments feel safe. A failure of B1+C1 with the current design can plausibly be attributed to (a) wrong environment partition, (b) wrong β schedule for IRM, (c) wrong β for VIB, (d) interaction between IRM and VIB stochasticity, (e) wrong base ckpt, or (f) actual data insufficiency — and the design doesn't differentiate among these. The "informative failure" framing is only valid if there's exactly one plausible failure mechanism.

---

## 7. Direct answers to the 6 decision asks

### Ask 1 — Verdict on the structural-reframe diagnosis

**Partial-agree.** The structural-reframe direction is rational. The "inevitability" framing is overstated. Specifically: three structurally-distinct levers inside the current paradigm have not been tried (output-preservation aux loss, P22 step1k as FT base, training-data subsetting). The diagnosis should be "structural reframe is one of 3-4 rational next steps," not "structural reframe is the only rational path."

### Ask 2 — Verdict on B1+C1

**Reject the combined-run shape.** Modify to one-arm IRM smoke first ($30, ~1.5h), per-method environment partition (~12 envs), β-anneal-down, with two abort criteria (CE divergence and gradient-variance non-reduction). Gate any C1 or combined run on the smoke's outcome.

### Ask 3 — Specific changes

1. **Sequential, not combined.** IRM-only first, VIB-only second if IRM smoke is informative, combined run third only if both standalone runs land partial-positive.
2. **Per-method environment partition (12 envs)**, not per-substrate (2 envs). DomainBed shows IRM is brittle below ~5 envs.
3. **β-anneal-down for IRM**, not anneal-up. Anneal-up is the proposal's framing; literature consensus is anneal-up exposes the penalty to a shortcut-loaded representation and the penalty can't reverse it.
4. **Base ckpt should be P22 step1k**, not Slot A v2 step3500 — the wider-variance distribution is the regime IRM most needs.
5. **Better informative-failure criterion**: held-out OOD AUC on substrate-pair test set, not penalty-internal gradient variance.

### Ask 4 — If reject (the combined run as specified is what I reject)

Alternative experiment plan in priority order:
1. **Same-day, $0**: per-ckpt τ-recalibration on the 5 candidates against the team-identity bar. Likely reveals Slot A v2 face-pool passes the bar after recalibration. ~1 hr CPU.
2. **Same-day, $0**: Xinhe-may6-gated production switch — pick between P8A_step5000 (Xinhe-may6 0%, conservative) and Slot A v2 CLS (Xinhe-may6 4.3%, recall-leaning) for current production. T5C must go.
3. **This week, $30**: one-arm IRM smoke per (3) above.
4. **This week, $50**: output-preservation aux loss against Slot A v2 step3500 reference encoder, FT base = P22 step1k. Structurally distinct from IRM, motivated by the Probe 1 finding more directly.
5. **Conditional**: contingent on (3) and (4) — if both null, escalate to VIB-only ($30) or to the contrastive-pretrain-on-substrate-pair option ($100) or to data acquisition (A).

Cumulative cost to a clearer verdict: $80-130 vs the proposal's $80 single shot. Slower but better-attributed.

### Ask 5 — 7000-webcam dataset acquisition

**Pursue in parallel.** Even if it's "not guaranteed to represent production webcams," it's the only data-side lever currently available, and the structural diagnosis (Probe 1, D9 100%-non-team-data) implies the binding constraint *could* be data. Running it in parallel costs operator-attention not GPU, and the result feeds whatever post-B1+C1 (or post-modified-plan) decision is made. If the frozen-CLIP baseline (probe still extracting) lands within 10pp of T5C, the data-acquisition decision becomes urgent rather than parallel.

### Ask 6 — Operational urgency contingent on probes

**Yes, the operational ship-the-best-current-candidate action is high-priority and should run immediately.** The xinhe_may6 T5C probe is in. T5C @ may6 = 17.4% FPR @ τ=0.5 / 16.3% @ mode A. The currently-deployed model is materially worse than P8A on the same cohort that motivated the 2026-05-06 "retire E2B" memo. The structural experiment can proceed independently, but **switching production off T5C is the same-day item** — and Slot A v2 face-pool is *not* the answer (91.3% Xinhe-may6 FPR at τ=0.5, 85.9% at mode A). The candidate set narrows to P8A (mode A or B) or Slot A v2 CLS (4.3% Xinhe-may6 FPR, 1.1% at mode B). Per-ckpt τ-recalibration ($0, 1 hr) is the prerequisite for that production-switch decision.

---

## 8. Self-correction log

- **Initial draft framing**: I was ready to write "reject B1+C1 entirely." On second pass through the proposal's §10.2 (which honestly enumerates the diagnosis's failure modes), I downgraded to "reject the combined-run shape, modify to sequential." The proposal's self-criticism is real and disarms some of the adversarial angles.

- **D9 reading**: my first read interpreted D9 as supporting "100% YouTube training reals" (matching the proposal). On second read, D9 §5 clearly distinguishes three substrate classes within the training reals, all YouTube-derived but with different capture pipelines. The proposal's framing is slightly imprecise. I added §2.3 noting this, which strengthens (not weakens) the data-side argument and adds a usable IRM partition.

- **Xinhe-may6 probe**: I initially planned to note the probe was running and integrate when it landed. It had landed before I wrote §5; I integrated it as a same-day operational item, which materially changed §7's Ask 6 from "low-priority status quo OK" to "high-priority production switch."

- **Frozen-CLIP probe**: still extracting at write time (~15 min ETA). I noted the contingency in §5 rather than waiting; the proposal's framing of this probe's decision-relevance is sound enough that waiting wouldn't change my verdict materially.

- **One thing I considered and didn't pursue**: a more aggressive critique of MODEL_GOALS.md's "no ensemble" rule. The face-pool inference monkey-patch and the proposal's discussion of dual-readout head are both technically inside the rule but spiritually adjacent to ensemble. I decided this was out of scope — the user has been clear the rule is hard, and challenging it is a different conversation.

---

## 9. Bibliography of what I actually read

- `docs/packet_retrospectives/plans/STRUCTURAL_REFRAME_PROPOSAL_2026-05-23.md` — the primary proposal, read end-to-end
- `docs/packet_retrospectives/MODEL_GOALS.md` — verified hard rules; "no ensemble" + "no per-substrate τ" are in force; the 0.30 fake-recall floor + 5% real-FPR + per-identity 30% bound are documented
- `docs/packet_retrospectives/AGENTS.md` — FACTS/OPINIONS split rules; verified the readout's documentation discipline
- `analysis/team_identity_deploy_readout_expanded_2026-05-23/RESULTS_FACTS_2026-05-23.md` — the binding measurement; confirms no ckpt passes both gates at mode B under cross-ckpt τ; supports the proposal's framing of today's gap
- `analysis/team_identity_deploy_readout_expanded_2026-05-23/AGENT_PROPOSAL_2026-05-23.md` — prior agent's verdict (P8A @ mode A is single mechanical passer); useful for grounding the "operational ship" question
- `~/.claude/projects/.../memory/project_team_identities_multi_labeled_2026-05-23.md` — the 5-team-human cohort definition + Mac-out-of-scope rule
- `~/.claude/projects/.../memory/project_production_is_t5c_not_e2b_2026-05-23.md` — confirms T5C is current production; supersedes the 2026-05-06 E2B claim
- `~/.claude/projects/.../memory/project_backbone_slotav2_groupdro_abort_2026-05-23.md` — confirms GroupDRO abort mechanism is structural, not bug-driven
- `~/.claude/projects/.../memory/project_backbone_t5c_blocked_pair_sampling_design_2026-05-23.md` — confirms T5C asymmetric pair-loss is blocked on sampler-redesign, not just bug
- `~/.claude/projects/.../memory/project_head_face_pool_verdict_iterate_2026-05-23.md` — confirms HEAD face-pool plateau and the mechanism (face-pool drops 147 non-face patches where viso signal lives)
- `analysis/substrate_pair_geometry_2026-05-22/FALLBACK1_PROBE1_FACTS_2026-05-22.md` — the load-bearing Probe 1 finding; verified the per-ckpt substrate axis is 98% accurate and 85-88° from frozen prior; supports the proposal's structural argument
- `analysis/cpu_diagnostics_2026-05-12_d8_head_retrain_substrate_balanced/D8_FACTS_2026-05-12.md` — prior frozen-CLIP partial baseline (DEV→LOCKBOX AUC 0.587 unweighted, 0.679 KLIEP-weighted); useful for the "what's frozen-CLIP at?" framing
- `analysis/cpu_diagnostics_2026-05-12_d7_fpr_decomposition/D7_FACTS_2026-05-12.md` — confirms IQ block dominates FPR variance (ΔR² +0.236-0.314 vs substrate +0.015-0.0001); supports the shortcut-taxonomy framing in proposal §3
- `analysis/cpu_diagnostics_2026-05-12_d9_source_substrate_inventory/D9_FACTS_2026-05-12.md` (partial) — verified the training-real composition is 92% HDTF/QCLIPS + 8% teams-v2 YouTube, all of which is YouTube-derived but in three distinct substrate classes (correcting the proposal's "100% YouTube" framing as imprecise)
- `~/.claude/projects/.../memory/project_lora_l10_l11_chronic_fp_hard_coupled_2026-05-15.md` — spot-check on the LoRA refuted-lever; confirms the catalog's verdict claim is accurate
- `~/.claude/projects/.../memory/project_overnight_resolution_chain_2026-05-16.md` — spot-check on resolution-chain Slot α; confirms the catalog claim
- `analysis/phase_3_scorecard_2026-05-23/RESULTS_FACTS_2026-05-23.md` — confirms today's Phase 2/3 verdicts (HEAD ITERATE, SlotAv2 ABORT, T5C BLOCKED)
- `analysis/xinhe_may6_t5c_revisit_2026-05-23/outputs/fpr_by_mode.txt` (probe complete) — the operational-urgency input: T5C @ may6 = 17.4% FPR @ τ=0.5; Slot A v2 face-pool @ may6 = 91.3% FPR @ τ=0.5
- `analysis/frozen_clip_team_identity_baseline_2026-05-23/outputs/_extract.log` (probe still running, 1280/5941 at last check) — not enough to integrate; flagged the contingency

---

End of review.
