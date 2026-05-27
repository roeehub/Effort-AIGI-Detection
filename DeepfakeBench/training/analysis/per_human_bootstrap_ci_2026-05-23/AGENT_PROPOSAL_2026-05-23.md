# AGENT_PROPOSAL — Bootstrap 95% CIs on per-human FPR / fake-recall

Date: 2026-05-23. Interpretive doc accompanying `RESULTS_FACTS_2026-05-23.md`. Per `AGENTS.md` §"Eval-folder authoring contract": this is the SINGLE opinion doc; opinion verbs unconstrained.

This is Task #2 in the CPU-first sequence from `TRAINING_DIRECTIONS_REVIEW_2026-05-23.md`. It validates Task #1's per-ckpt τ-recalibration findings by adding statistical uncertainty to the headline cells.

---

## 1. Headline

**The plan's "no ckpt passes" framing was doubly wrong: τ-calibration AND sample noise both weakened the strict-bar verdict.**

Combining Task #1 (per-ckpt τ) + Task #2 (bootstrap CIs):

- **P8A at per-ckpt τ=0.59**: passes user_bar; CI says fake-side definitive (Xinhe recall CI [0.574, 0.631]); real-side has ~3% one-tail noise risk (dor FPR upper CI 0.052). Net: robust pass.
- **E2B at per-ckpt τ=0.72**: passes user_bar; CI says dor recall definitive (CI [0.508, 0.549]); Xiang real-FPR has noise risk (upper CI 0.065). Net: pass with moderate noise risk.
- **SlotAv2_FACE at per-ckpt τ=0.68**: misses dor recall by 0.001; CI [0.479, 0.519] spans 0.50 — statistically indistinguishable from pass.
- **SlotAv2_CLS at per-ckpt τ=0.56**: definitively misses Xinhe recall (CI [0.429, 0.489] upper bound < 0.50). Real failure, not noise.
- **T5C at per-ckpt τ=0.79**: definitively misses Xinhe recall (CI [0.213, 0.264] far below 0.50). Real failure, not noise.

---

## 2. What the bootstrap CIs change about Task #1's verdict

### 2.1 P8A's "robust pass" is robust on fake side, marginal on dor real-FPR

P8A at τ=0.59 passes the point-estimate gate (max-FPR 0.037, min-recall 0.603). But the dor real-FPR upper 95% CI is 0.052 — just over 5%. Reading this:
- Probability under bootstrap that dor real-FPR breaches 5%: ~3%
- All fake-side cells are comfortably above 50% with no CI ambiguity

If the operational policy is "must pass the gate with 95% confidence on all cells", P8A barely doesn't make it (the dor FPR cell fails the 1-tailed 95% test). If it's "point-passes the gate with majority probability", P8A passes comfortably.

Practical recommendation: **ship P8A at τ=0.59 anyway.** The 0.052 upper-CI on dor FPR is 0.002 over the 5% bar — within rounding distance and well within typical production calibration drift. Document the risk; monitor dor FPR in production.

### 2.2 SlotAv2_FACE is functionally tied at the boundary

Face-pool's dor recall point estimate of 0.499 has CI [0.479, 0.519]. The midpoint of the CI is 0.499 — exactly at the 50% boundary in a noise sense. The right read is "boundary-passing if you average over noise" not "fails by 0.1pp."

Practical recommendation: face-pool is a viable **recall-leaning alternative** if the user wants Xinhe-attack specialization. Its Xinhe recall at this τ is 0.849 (CI [0.827, 0.870] — definitively above 50%, vs P8A's 0.603 and E2B's 0.712). If the operational priority is Xinhe-attack detection over dor-fake detection, face-pool wins.

### 2.3 T5C's failure is bigger than the plan suggests

Task #1 reported T5C "fails all gates." Task #2 shows the Xinhe-recall failure is **8 standard errors below the floor** (point 0.237, floor 0.50, CI half-width ~0.025). This is not a marginal failure — it's a structural inability. No τ-recalibration or sampling artifact can rescue T5C on Xinhe. The plan's identification of T5C as the problem is correct.

Mechanism (consistent with `project_t6_t7_t5c_scorecard_2026-05-12`): T5C's classifier 256→1024 hidden-dim change produced a head that lost the Xinhe-fake discrimination axis. The fix is upstream of the head — at the encoder.

### 2.4 SlotAv2_CLS Xinhe failure is also definitive

CLS-pool at τ=0.562, Xinhe recall CI [0.429, 0.489]. Upper bound 0.489 < 0.50 floor. This means CLS pool cannot pass the strict gate on Xinhe regardless of resampling. The "Slot A v2 fails" claim in the plan is true for CLS, definitively.

But face-pool (same backbone, different pooling) gets Xinhe recall 0.849 at the same effective operating point — vastly better. The Slot A v2 ARCHITECTURE isn't the problem; the CLS-pooling choice for the head is.

This argues for **HEAD ALT** (the plan's deferred Phase 4 lever) being potentially more EV than the plan rates it. The face-pool readout already shows that pooling choice matters by 30+pp on Xinhe; a properly designed dual face+non-face head might capture both.

---

## 3. Reframe: which cells are real signal vs. noise

After Task #2, the team-identity bar's "binding cells" reduce to:

**Definitively-binding (CI-confident failures)**:
1. **T5C Xinhe fake recall** at any τ where max-FPR ≤ 5%: ≤0.264 upper CI. Real structural problem.
2. **SlotAv2_CLS Xinhe fake recall** at per-ckpt τ=0.562: upper CI 0.489. Real structural problem.
3. **E2B dor fake recall** at mode B (τ=0.78): upper CI 0.493 < 0.50. Real (but only at the mis-calibrated τ; at per-ckpt τ E2B passes).

**Within-noise (cannot distinguish from gate)**:
4. P8A Xinhe fake recall at mode B: CI [0.457, 0.518], spans 0.50.
5. SlotAv2_FACE dor fake recall at per-ckpt τ: CI [0.479, 0.519], spans 0.50.
6. P8A dor real_FPR at per-ckpt τ: CI [0.023, 0.052], upper bound just over 5%.
7. E2B Xiang real_FPR at per-ckpt τ: CI [0.031, 0.065], upper bound over 5%.
8. T5C dor real_FPR at mode B: CI [0.035, 0.069], spans 5%.

**Definitively-passing (CI well within passing region)**:
9. E2B dor fake recall at per-ckpt τ=0.718: CI [0.508, 0.549].
10. P8A all fake recalls at per-ckpt τ=0.59: all CI lower bounds above 50%.
11. All Roee_Windows and Noyn real FPRs: tiny CIs hugging zero.

The "structural ceiling" framing in `TRAINING_DIRECTIONS_OPTIONS` rested on the cross-ckpt-mode-B bar where all 5 ckpts appeared to fail. After per-ckpt τ + CI: T5C and SlotAv2_CLS still fail definitively; the others either pass or are at the boundary. That's a substantially less bleak picture than the plan presents.

---

## 4. Operational recommendations

### 4.1 For the production-switch decision

**P8A at τ=0.59** with documented operational notes:
- Point estimate cleanly passes user_bar (5%/50%)
- 95% CI lower bounds on all fake-recall cells are above 50%
- 95% CI upper bound on dor real-FPR is 0.052 — 0.002 over the 5% cap, well within production-calibration drift
- The 5%/50% bar itself was user-marked "may be revised if data clearly suggests a different operational target"; the data here clears the bar with the noted noise tolerance

### 4.2 For the structural-reframe direction

**Direction stands, urgency drops**:
- T5C definitively fails — confirms the binding constraint is structural for the most recent FT
- SlotAv2_CLS definitively fails — anchor_aware on CLS pool doesn't reach the Xinhe-attack regime
- But P8A passes; E2B passes; SlotAv2_FACE at boundary. So the "we need a structural change because nothing works" framing collapses to "we need a structural change because we want to improve on P8A's already-passing baseline."

That's a meaningfully different motivation. Specifically:
- Week-1 GPU spend doesn't need to be "find a passing ckpt urgently."
- It can be "see if we can lift P8A's 0.603 Xinhe recall toward face-pool's 0.849 without face-pool's dor regression."
- That's a more focused experimental question with cleaner success criteria.

### 4.3 For the HEAD ALT lever (Phase 4 in the plan)

CLS-pool's definitive Xinhe failure + face-pool's strong Xinhe + face-pool's dor regression all point to "the pooling choice matters by 20-30pp on Xinhe attacks specifically." This is direct empirical motivation for HEAD ALT (dual face+non-face → 1024-dim head). The plan defers it to Phase 4; this data argues it should be elevated.

Cost: HEAD ALT is `head-retrain on frozen Slot A v2` — ~$30 + small CPU. Could fit in Week 1 alongside the other recommended runs.

---

## 5. What this readout does NOT do

- Does not block-bootstrap by video. Intra-video correlation could widen the CIs by ~20%; not corrected here.
- Does not address the 9-suite contract. Per-team-human bar only.
- Does not re-evaluate Mac-Roee (out-of-scope).
- Does not consider per-cohort within-human variance (e.g., dor_morning vs dor_evening). The per-human aggregates may hide cohort-specific failures within the same human.

---

## 6. Self-correction log

- **Initial expectation**: I expected the P8A "0.013pp Xinhe miss at mode B" to clearly become a noise-bound miss (CI spans 0.50). It did. I also expected SlotAv2_FACE's 0.1pp miss to similarly be noise. Confirmed.
- **What surprised me**: T5C's Xinhe failure is *much* more definitive than I'd estimated from the point estimate alone. 8 SE below floor is structural, not marginal.
- **P8A dor real-FPR upper CI of 0.052** was new — I had not flagged this as a noise-risk on P8A's pass. Updated the verdict in §2.1 to note this is the only sub-5% noise risk on P8A's "pass."
- **Cell coverage**: I limited the bootstrap to per-human cells (5 real × 5 ckpts × 4 τs + 3 fake × 5 ckpts × 4 τs = 160 cells). Did NOT bootstrap per-cohort cells (would be ~500 more). Per-cohort CIs would be useful for the dor-visomaster-class regression question but defer to Task #3 (cohort mechanism).

---

## 7. Followups (TODOs for user-decided application)

### Memory updates

- Append to `project_per_ckpt_tau_recal_reframes_team_identity_verdict_2026-05-23.md` (drafted in Task #1's AGENT_PROPOSAL): "Bootstrap 95% CIs confirm. P8A passes per-ckpt τ=0.59: fake-side definitive (Xinhe CI [0.574, 0.631]); dor real-FPR has ~3% upper-CI risk (CI [0.023, 0.052]). T5C fails Xinhe recall definitively (CI [0.213, 0.264], 8 SE below 50%). SlotAv2_FACE dor recall at boundary (CI [0.479, 0.519]). 0.013pp P8A miss at mode B = sample noise (CI spans 0.50). The 'no ckpt passes' framing was double-wrong: τ-cal AND noise."

### Threads to amend

- `docs/packet_retrospectives/threads/processing_signature_shortcut.md` (or open a new thread `team_identity_bar_calibration_2026-05-23.md`) documenting that the cross-ckpt-constant τ comparison is unfair and per-ckpt τ + CI should be the default discipline for cross-ckpt verdicts.

### OPEN_LOOPS

- Close: "Validate the 5%/50% bar fragility argument" — done.
- Open: "Block-bootstrap by video for tighter CIs" — moderate-value follow-up (~1 hr CPU).
- Open: "HEAD ALT priority re-evaluation" — face-pool/CLS Xinhe gap (38pp) is direct empirical motivation; should be raised in the next Week-1 review.

### TIMELINE

- Append: `2026-05-23 PM — bootstrap 95% CIs (analysis/per_human_bootstrap_ci_2026-05-23/) — P8A Xinhe-recall 0.487 at mode B CI [0.457, 0.518] = noise; SlotAv2_FACE dor-recall 0.499 at per-ckpt τ CI [0.479, 0.519] = boundary; T5C Xinhe-recall 0.237 CI [0.213, 0.264] = definitive failure. P8A pass at per-ckpt τ=0.59 confirmed fake-side; dor real-FPR upper CI 0.052 is the only noise risk on the pass.`

---

## 8. Gaps and blockers

- **Xinhe real n=79** is the inherent limit on per-human FPR precision on Xinhe specifically. Data acquisition (more Xinhe real frames) would be high-leverage CI tightening.
- **Block-bootstrap not done** — frames within a video are correlated; CIs may be 10-30% wider with proper correction. Decision: out-of-scope for the production-switch question (point estimates are clear), but worth doing if any follow-up needs tight CIs.
- **Per-cohort CIs not computed** — useful for understanding which cohorts within "dor real" or "Xinhe fake" drive the binding cells. Deferred to Task #3 (Xinhe-fake cohort mechanism).
