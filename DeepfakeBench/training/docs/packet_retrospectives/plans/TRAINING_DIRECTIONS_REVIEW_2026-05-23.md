# Independent Review — Training Directions Options (2026-05-23)

**Reviewer**: independent agent (Opus 4.7), dispatched 2026-05-23 PM by user
**Reviewing**: `docs/packet_retrospectives/plans/TRAINING_DIRECTIONS_OPTIONS_2026-05-23.md`
**Reading order completed**: this document end-to-end; `STRUCTURAL_REFRAME_PROPOSAL_2026-05-23.md`; `STRUCTURAL_REFRAME_REVIEW_2026-05-23.md`; `MODEL_GOALS.md`; today's three binding measurements (`team_identity_deploy_readout_expanded_2026-05-23/RESULTS_FACTS` + `AGENT_PROPOSAL`, `xinhe_may6_t5c_revisit_2026-05-23/RESULTS_FACTS`, `frozen_clip_team_identity_baseline_2026-05-23/RESULTS_FACTS`)
**Stance**: adversarial, no rubber-stamp

---

## 1. Executive verdict

**Mostly agree with structure and pacing; disagree with the headline Week-1 pick; one major missing implication; one foregrounded item underweighted.**

The author has cleanly absorbed the morning reviewer's corrections — the sequential-not-combined IRM/VIB split, the per-method environment partition, the P22 step1k base option, the output-preservation aux loss, the 50% bar fragility, the cross-ckpt τ caveat. The four-tier menu structure (untried-in-paradigm / structural reframe / big commitments / lower-EV) is correctly framed. The §6 risks section is honest.

What's wrong:

1. **The Week-1 pick (B.I.1 + B.I.2 combined) commits the exact compound-experiment sin the morning reviewer caught for B1+C1.** Two independent levers in one run = the same attribution problem rebranded.
2. **The frozen-CLIP probe's OPTB result is the most decision-relevant new evidence today, and the plan absorbs only a fraction of what it implies.** Specifically for B.III.3 (dataset acquisition) and B.II.3 (contrastive pretrain).
3. **A-1 (per-ckpt τ-recalibration) is undersold as bookkeeping when it's a same-day, $0 lever that could resolve part of the production-switch decision and unblock the structural-ceiling framing's load-bearingness.**
4. **The output-preservation aux loss is recommended without a concrete spec.** Cost estimates without that spec are not defensible.

The structural-reframe *direction* (as one of 2-3 rational next steps) is fine. The push-back is on specific Week-1 commits, prioritization, and bookkeeping work the plan elides.

---

## 2. Where I agree with the plan

- **Diagnosis scoping**: "structural ceiling is one of 2-3 rational next steps" (not the only). The plan correctly walks back the morning proposal's overstatement.
- **IRM smoke spec is right**: per-method 12-env partition, β-anneal-down, P22 step1k base, explicit abort criteria (CE divergence + gradient-variance non-reduction). This is the corrected shape per the morning reviewer.
- **Lower-EV exclusions (B.IV) are well-justified**: B32, LoRA placement, Roy_D anchor, more HEAD-only — each correctly downgraded with citation to a refuting experiment.
- **The "$200-300 over 1 week" cumulative framing is more honest** than the morning's $80 single-shot framing. Expected-cost transparency is up.
- **The §7 decision-asks are well-scoped** for the operator (3 primary + 2 secondary).
- **The MODEL_GOALS NO-ENSEMBLE constraint is faithfully respected.** No proposed lever crosses it.

---

## 3. Critical pushback

### 3.1 B.I.1 + B.I.2 combined is the wrong shape, for the same reason B1+C1 combined was

The morning reviewer wrote: *"both methods have known instability and known failure modes that interact... combining them obscures attribution."* The plan accepts that critique for B1+C1 (rejects combined VIB+IRM, splits into sequential IRM-only smoke first).

But then it commits the same fault structure for the Week-1 top pick: **P22 step1k base + output-preservation aux loss in one run**. If it lands on the team-identity bar, you don't know whether the base change or the aux loss did the work. If it fails, you don't know whether the right base + wrong loss or the wrong base + right loss failed.

Both ingredients are interesting; they should be split. Concretely:
- **$50 — P22 step1k + anchor_aware** (no aux loss): tests "is the base ckpt the lever?" Anchor_aware is the one packet-proven win (per `project_band_shortcut_ood_hypothesis_2026-05-16`); keeping it constant and varying base is a clean attribution.
- **$50 — Slot A v2 step3500 + output-preservation aux loss against frozen-CLIP penultimate**: tests "does aux loss against a known-good representation help?"

Same $100, two clean attributions, runs in parallel on different US regions per CLAUDE.md. The combined run can come in Week 2 if either bites.

The plan's defense — *"if output-preservation is right but Slot A v2 was wrong base, this catches both"* — is exactly the framing the morning reviewer rejected for IRM+VIB. The author has internalized the critique for B.II but not for B.I.

### 3.2 The OPTB negative result has bigger implications than the plan absorbs

§1.5.6 records that adding 6,000 training-corpus frames to the frozen-CLIP head training **made every per-human metric strictly worse**: best Option B head min recall 0.037 vs best Option A 0.205. **LR_OPTB lockbox AUC 0.345** (below chance) — the head's separator points the wrong way on the deploy distribution.

The plan reports this faithfully but treats it as one bullet in §1.5. It needs to propagate into:

- **B.III.3 (7000-webcam dataset acquisition) recommendation.** The plan says "pursue in parallel." But if generic-domain training data degrades the head's deploy-distribution behavior, **unmatched webcam data could too**. The recommendation should be: "pursue acquisition; do NOT train on it without a substrate-matching check (e.g., per-frame KLIEP fit against deploy distribution) first." A naive "add data, retrain" attempt could net-negative — at hundreds of GPU dollars. **The OPTB result is the cheapest piece of evidence we will ever get for the "more data ≠ help" hypothesis on this problem.**
- **B.II.3 (substrate-pair contrastive pretrain) priority upgrade.** The OPTB result is paradoxically a *positive* signal for B.II.3. Generic data hurts; aligned data (the 1,825 paired clean↔teams captures from A0.1) is precisely the *not-generic* invariance signal. The plan rates B.II.3 at "20-30% P(materially better)" but doesn't pull it into Week 1. Given the OPTB finding, I'd elevate this to **25-35%** and consider it Week-1 instead of (or alongside) the IRM smoke. Mechanism-to-evidence fit is now better for B.II.3 than for IRM.
- **B.I.3 (KLIEP per-frame training subset) priority upgrade.** Same logic — the OPTB result is direct evidence that filtering to the in-distribution subset is more valuable than adding diversity. Should also be elevated from 20-30% to 25-35%.

In short: the OPTB finding flips the prior on "more data ≈ help." It says **substrate-matched data ≈ help, generic data ≈ hurt**. That changes which Week-1 lever has the best mechanism-to-evidence fit and the plan should re-rank accordingly.

### 3.3 A-1 per-ckpt τ-recalibration is doing more work than its $0 / 1hr cost suggests

The plan files A-1 under "deployable today" §2 and says "Prerequisite for any ship decision." But it doesn't engage with the operational implications. From the data already in hand:

- The cross-ckpt τ headline — **P8A 0.487 Xinhe-fake-recall at mode B, 0.013pp below 50% floor** — is *within sample noise* on n=1099 (~±1.5pp at 95% CI). Per-ckpt re-calibration could plausibly tip P8A through the mode-B gate, which would resolve the "no ckpt passes" framing and remove the urgency that motivates much of §3.II.
- Slot A v2 face-pool's "catastrophic" 91% Xinhe-may6 at τ=0.5 is partly score-distribution drift. **But — critically — the internally-calibrated drift table in `xinhe_may6_t5c_revisit/RESULTS_FACTS §3` shows face-pool STILL has 76.1% may6 FPR at the per-ckpt p95 cut** (vs P8A 4.3%). So per-ckpt τ-recalibration does NOT rescue face-pool on Xinhe-may6. That's an important asymmetric result that should be explicitly called out in the plan — **A-1 will likely promote P8A mode B as the deploy answer, not Slot A v2 face-pool.**

A-1 is potentially the highest-EV item in §2: it resolves the production-switch decision AND tightens the case for whether the "no ckpt passes" framing is sample-noise or signal. The plan files it as routine bookkeeping.

### 3.4 Output-preservation aux loss is recommended without a concrete spec

The plan lists output-preservation aux loss as a Week-1 top-pick and the morning reviewer's "top in-paradigm recommendation." Both treat it as well-understood, but neither answers:

- **Against which reference encoder?** Slot A v2 step3500 (plan's pick), or frozen-CLIP B16-DataComp.XL? Probe 1 says FT rotates 85-88° from frozen — so regularizing against the frozen prior is more directly motivated than against another FT'd ckpt.
- **On which layers' features?** Penultimate only? Last 4 transformer blocks? L11 CLS only?
- **What loss function?** MSE on raw features? Cosine similarity? KL on normalized distributions?
- **On which images?** Held-out reference pool? Training reals broadly? The 1,825 substrate-pair captures?
- **What weight relative to CE?** β=0.1, 1.0, 10.0? Sweep?

The plan estimates "$50, 4h, P(materially better) 15-25%" — these numbers are not defensible without the above specs. The packet could waste GPU on the wrong layer/weight combination and report "aux loss didn't bite" when the real verdict should be "aux loss with THIS spec didn't bite."

Recommendation: **before launching B.I.1, the agent owes a 1-page spec doc** that names these choices and justifies each from existing evidence. The Probe 1 finding directly motivates: penultimate-layer features, frozen-CLIP reference, cosine similarity, computed on the 1,825 substrate-pair reference pool, weight 0.1-1.0 with sweep — but that's me speccing it; the plan should do this work explicitly before committing GPU.

### 3.5 B.III.1 temporal pre-experiment at $30 is implausibly cheap

"Verify the temporal signal survives Teams compression on a small sample. $30 (CPU + small GPU)."

This is at minimum:
1. Acquire matched (raw video, Teams-pipeline-processed) pairs — operationally hard for the *fake* side per §6.D of the morning proposal; real-side is easier but still nontrivial.
2. Compute temporal-consistency features (optical flow consistency, frame-to-frame embedding drift, prediction stability) on both versions.
3. Quantify how much of the signal survives the pipeline.
4. Compare against a baseline (random / single-frame).

Honest estimate: **$100-250 of CPU/MPS analysis effort** (mostly free under standing greenlight) + 1-2 days of agent time. The $30 figure undersells what "verify temporal signal survives compression" actually requires for a defensible verdict. If the user picks up B.III.1 in Week 2 thinking the gating pre-experiment is $30, they will discover otherwise.

Either re-estimate honestly ($150-250 + agent-time) or scope the pre-experiment more narrowly (e.g., "single-pair optical-flow consistency on 10 video clips" → $30, but verdict is weak).

### 3.6 The 5%/50% bar fragility deserves a top-level decision, not a §6 caveat

The AGENT_PROPOSAL for the team-identity readout (§6) explicitly says:
> "60% Xinhe-fake gate: E2B passes at mode A. 40% Xinhe-fake gate: P8A passes at mode B... A reasonable revised floor might be 50% at mode A (where P8A passes) and 40% at mode B (where E2B + P8A both pass)."

The plan acknowledges this in §6.1 but treats it as a risk-to-monitor. It's actually a **decision the operator could make for $0**: relax the per-human floor to 40% (with documented justification) and the "no ckpt passes" headline goes away. That doesn't make the structural-reframe direction wrong — but it removes the urgency framing, weakens the "structural ceiling" claim's load-bearingness, and changes the Week 1 EV calculation. P8A mode B at a 40% Xinhe floor becomes a defensible ship and Week 1 can be more exploratory rather than blocking on "find an actually-passing ckpt."

This should be in §7 as a top-level decision ask: **"Decide whether the 50% per-human fake-recall floor is firm or revisable to 40% with documentation."**

---

## 4. Direct answers to §7 decisions

### Decision 1 — Week 1 plan

**Modify.** My Week 1, $130 total GPU + 8-10 hrs CPU:

| Step | Cost | Wall | Purpose |
|---|---:|---:|---|
| A-1 — per-ckpt τ-recalibration on 5 ckpts × {team-identity bar, 9-suite contract} | $0 | 1 hr CPU | Resolve production-switch + reveal whether face-pool is salvageable beyond mode B |
| **Bootstrap CI on per-human metrics** | $0 | 30 min CPU | Validate the 5%/50% bar fragility numerically |
| **KLIEP substrate-matching check (OPTB / team-identity / lockbox)** | $0 | 1 hr CPU | Sets the policy for B.III.3 dataset-acquisition gating |
| **Output-preservation aux loss spec doc + feature-distance analysis** | $0 | 2 hrs CPU | Prerequisite to B.I.1 GPU launch |
| **GPU run 1**: P22 step1k + anchor_aware (no aux loss) | $50 | 4 hrs | Tests "is the base ckpt the lever?" |
| **GPU run 2**: Slot A v2 step3500 + output-preservation aux loss (frozen-CLIP penult ref) | $50 | 4 hrs | Tests "does aux loss bite?" — runs in parallel in a different US region per CLAUDE.md |
| **GPU run 3**: IRM-only smoke (per plan spec; per-method 12-env, β-anneal-down, P22 step1k base) | $30 | 1.5 hrs | Tests structural-reframe class |

Skip the combined B.I.1+B.I.2 run. Skip the B.III.1 pre-experiment until you've re-scoped its cost honestly.

If user has appetite for one more $100 line item: add **B.II.3 substrate-pair contrastive pretrain** as a Week-1 candidate. The OPTB result moves its mechanism-to-evidence fit ahead of IRM's.

### Decision 2 — B.III.3 (7000-webcam dataset acquisition)

**Pursue acquisition, defer training-on-it.** The OPTB result is direct evidence that generic data hurts. Acquire the 7000-frame set — operator-time cost only — but BEFORE training:

1. Run a 1-day CPU job: KLIEP-fit a discriminator between the acquired set and the team-identity deploy distribution.
2. If the discriminator achieves <70% accuracy, the data is well-matched → train on it.
3. If ≥85%, it's not aligned with production → either don't train, or train only on the substrate-aligned subset (à la B.I.3 logic).
4. 70-85% → train on a stratified subset, validate carefully.

The KLIEP substrate-matching check is CPU-only and effectively free under standing greenlight; building it now (as part of Week-1 CPU work) means no waste if the dataset arrives later.

### Decision 3 — Production switch (A-1 + A-2 / A-3 / hold)

**Run A-1 first ($0, 1hr), then decide.** Expected outcomes:
- If A-1 reveals P8A mode B passes per-ckpt-calibrated bars (likely given the 0.013pp miss is within noise): switch to P8A mode B.
- If not: switch to P8A mode A (the mechanical winner per RESULTS_FACTS §11).
- Status quo (T5C) is the worst choice given Xinhe-may6 elevation (17.4% @ τ=0.5; 3.3% @ mode B) and 5.2% dor mode-B FPR (over 5% floor).

Slot A v2 face-pool is **not** a viable production switch. Per-ckpt-calibrated may6 FPR is 76.1% — the τ=0.78 zero is a calibration artifact that vanishes at any other operating point.

### Secondary 4 — Elevate B.III.1 (temporal) to Week 1?

**No.** Cost is genuinely $300-800 + 1-2 weeks of infra. The gating pre-experiment is $100-250 (not the plan's $30). Week 1 should run the cheaper attribution-clean experiments first; B.III.1 properly belongs in Week 2 contingent on Week-1 partial-positive + an honestly-scoped pre-experiment. Defer.

### Secondary 5 — Elevate B.II.3 (substrate-pair contrastive pretrain) to Week 1?

**Yes, if user has budget for one more $100 run.** The OPTB negative result is the most direct positive signal for contrastive pretraining on the 1,825 *aligned* pairs (vs. adding generic data). Mechanism-to-evidence fit is now better than IRM's. Total Week 1 would become $230 vs my $130 above.

If keeping Week 1 to $130: drop the IRM smoke in favor of B.II.3 contrastive pretrain.

---

## 5. What's missing from the plan

1. **A concrete output-preservation aux loss spec** (see §3.4).
2. **A re-stated production-switch decision** that incorporates the per-ckpt-τ recalibration result (which will arrive within the same hour as the decision is made — A-1 unblocks A-2).
3. **An honest re-estimate of the B.III.1 temporal pre-experiment cost.**
4. **A foreground decision-ask on the 50% floor itself** (firm or relaxable to 40%).
5. **A "data acquisition guard rail"**: explicit policy that any externally-sourced dataset goes through a substrate-matching check before training, given the OPTB finding.
6. **A note about the Slot A v2 face-pool / Xinhe-may6 asymmetry**: per-ckpt τ-recalibration doesn't rescue face-pool on may6 (76.1% per-ckpt-p95 FPR). This contradicts the implicit "calibration would fix it" reading and should be explicit.

---

## 6. What I would NOT change

- The IRM smoke spec (per-method 12-env, β-anneal-down, P22 step1k base, abort criteria).
- The B.IV exclusions (LoRA placement, B32, Roy_D anchor, more HEAD variants).
- The "no ensemble" / "no per-substrate τ at deploy" hard constraints stand.
- The structural-reframe direction as one of 2-3 rational next steps.
- The §6 risks section's framing.

---

## 7. Honest uncertainty on my own pushback

- **My B.II.3 elevation** rests on the OPTB result generalizing as "generic data hurts → aligned data helps." The OPTB sample is 6,000 frames; the substrate-pair pool is 1,825 pairs. Both are small. The "more aligned data is better" inference is plausible but not proven; B.II.3 could fail too.
- **My recommendation to split B.I.1+B.I.2** trades $0 net cost for cleaner attribution. The plan's framing ("catches both") is not crazy — just inferior when you can afford parallel runs (which on Vertex you usually can per CLAUDE.md region-parallelism rules).
- **My "$100-200 not $30" estimate for the B.III.1 pre-experiment** depends on what "verify temporal signal survives" actually means. If you only need a 10-clip optical-flow-consistency probe, $30 is OK. If you need a defensible enough verdict to commit $500 of follow-up GPU, you need more.
- **The 5%/50% bar relaxation** is a user-judgment call; I can flag the option but cannot pick it.
- **A-1 might not promote any ckpt through mode-B gates** even after recalibration. My expectation that it will lift P8A through is based on the 0.013pp gap being noise-sized; could be wrong.

---

## 8. Self-correction log

- **Initial draft framing**: I was tempted to write "approve the plan with modifications" because it correctly absorbed the morning reviewer's catches. On re-reading, the B.I.1+B.I.2 combined Week-1 pick repeats the structural error the plan had just rejected. Downgraded to "modify."
- **OPTB implications**: my first pass treated the OPTB finding as one of several pieces of evidence. On reflection, it's the *most* decision-relevant new piece — it directly contradicts the prior on "data acquisition helps" and reweights B.III.3, B.II.3, and B.I.3. I expanded §3.2 to give it proper weight.
- **Face-pool reading**: I initially questioned whether the Slot A v2 face-pool 91% Xinhe-may6 finding was a τ-calibration artifact. On reading `xinhe_may6_t5c_revisit/RESULTS §3` (internally-calibrated drift table), the 76.1% per-ckpt-p95 FPR confirms face-pool is broken on may6 even after calibration. I retracted that thread and added it as a positive observation in §3.3.
- **B.II.3 elevation**: initially treated as a "yes-and" suggestion. After §3.2 expansion, it became a serious Week-1 candidate. Adjusted Decision 1 + Secondary 5 to reflect.
- **What I considered and didn't pursue**: an aggressive critique of the MODEL_GOALS NO-ENSEMBLE rule (the morning reviewer noted this too). The face-pool monkey-patch is technically inside the rule but spiritually adjacent to a "pool-fusion" choice. Decided against challenging the rule itself — it's hard, user-set, and challenging it is a separate conversation.

---

## 9. Working log

This section is updated each session we touch this review. Append new entries; do not edit historical ones.

### 2026-05-23 (PM) — initial review authored

- **Author**: independent agent (Opus 4.7), dispatched by user (Roee) after reading `TRAINING_DIRECTIONS_OPTIONS_2026-05-23.md`
- **Reading completed**: plan end-to-end; both prior plans (morning proposal + adversarial review); `MODEL_GOALS.md`; today's three binding measurements (team_identity_deploy_readout_expanded + xinhe_may6_t5c_revisit + frozen_clip_team_identity_baseline RESULTS_FACTS + the team-identity AGENT_PROPOSAL).
- **Verdict**: modify (not approve). Six specific pushbacks, three of which are net-new vs the morning reviewer: (a) B.I.1+B.I.2 combined repeats the compound-experiment error, (b) the OPTB finding has wider implications than the plan absorbs, (c) the output-preservation aux loss needs a spec doc before GPU. The other three (A-1 under-sold, B.III.1 cost suspect, 50% floor fragility) extend or sharpen the morning reviewer's points.
- **No code/config/data changes made**. Review-only artifact.
- **Next**: (per user) suggest CPU tasks for the time before GPU launch.

### 2026-05-23 (PM, ~5 hrs CPU) — sequence 1→2→7→3→4 completed; major review-changing findings landed

All 5 CPU tasks from §4 Decision 1 of this review completed. Each produced RESULTS_FACTS + AGENT_PROPOSAL pair under `analysis/<task>_2026-05-23/`. Headline-changing findings landed in #1 (production switch) and #3 (Xinhe-fake mechanism).

**Task #1 — Per-ckpt τ-recalibration** (`analysis/per_ckpt_tau_recal_2026-05-23/`):
- The plan's "no ckpt passes the team-identity bar" headline was a **τ-calibration artifact**. Under per-ckpt fair calibration:
  - **P8A passes user_bar (5%/50%) at τ ∈ [0.427, 0.753]** (width 0.33, best τ=0.59, max-FPR 3.7%, min-recall 60.3%)
  - **E2B passes at τ ∈ [0.687, 0.748]** (width 0.06)
  - SlotAv2_FACE misses by 0.1pp at boundary (within sample noise)
  - T5C does not pass any gate (Xinhe recall 0.237 << 50%)
  - SlotAv2_CLS passes the relaxed 5%/40% gate
- **Recommendation crystallizes**: switch production T5C → P8A at τ=0.59, same-day $0.
- This **weakens the plan's structural-reframe urgency** significantly (direction unchanged).

**Task #2 — Bootstrap CIs** (`analysis/per_human_bootstrap_ci_2026-05-23/`):
- P8A's mode-B Xinhe miss (0.487) CI [0.457, 0.518] — definitively sample noise.
- SlotAv2_FACE dor-recall at per-ckpt τ (0.499) CI [0.479, 0.519] — within noise of pass.
- **T5C's Xinhe failure is structural** (CI [0.213, 0.264], 8 SE below 50%). Not noise.
- P8A's per-ckpt-τ pass has one noise risk: dor real-FPR upper CI 0.052 (just over 5% cap).

**Task #7 (now #3) — Xinhe-fake cohort mechanism** (`analysis/xinhe_fake_cohort_mechanism_2026-05-23/`):
- **Hard-vs-easy is perfectly CLIP-separable** (LR 1.0 5-fold CV AUC).
- Hard *geometric* cluster covers 10 of 13 cohorts (77% of Xinhe-fake frames); easy is just {6, 7, 8}.
- **Face-pool's +37pp Xinhe gain over P8A concentrates on the hard cluster** (P8A hard=0.341 → face=0.709; P8A easy=0.956 → face=0.844, a 11pp regression on easy).
- LR coefficient saved at `outputs/hard_vs_easy_lr_coef.npy` — usable directly as the aux-loss target for a hard-axis-aware FT run.
- **Adds two GPU candidates to Week 1**: hard-axis-aware FT ($50-80), HEAD ALT elevation ($30).

**Task #4 — KLIEP substrate-matching** (`analysis/kliep_substrate_match_2026-05-23/`):
- **OPTB ↔ team-identity is highly mismatched** (99.9% CV accuracy, ESS_on_OPTB=0.031). Concrete confirmation of the OPTB-negative hypothesis.
- OPTB ↔ each team-human's real cohort all at 0.999-1.000 CV accuracy.
- **Each team-human's real cohort is 100% CV-separable from every other**. The team-identity "population" is 5 distinct subdistributions.
- Substrate-match policy thresholds proposed for B.III.3 dataset acquisition (table at §4).
- **Adds new IRM partition candidate**: 5-env per-team-human partition (or 17-env per-method + per-human).

**Task #5 — Output-preservation aux loss spec** (`analysis/output_preservation_spec_2026-05-23/`):
- Per-layer feature distance: **~1000× more cross-ckpt drift at L11 than at L0** (cos 0.40-0.75 at L11 vs ≥0.997 at L0).
- P8A is the geometric outlier at L11 vs SlotAv2/T5C cluster (cos 0.48 vs SlotAv2/T5C 0.74).
- **Spec landed**: reference=P8A, layer=L11+L8 (β_8=β_11/4), loss=cosine, pool=1,825 substrate-pair (50/50 batch), base=P22 step1k, β_11=0.5 with 500-step warmup. Cost $50-60 + $30 ablation.
- The spec is now defensible for a GPU launch (was not before — §3.4 of this review flagged it as missing).

**Changes to my own §4 recommendations after these tasks**:
- Decision 1 (Week 1 plan): unchanged — but the OUTPUT-PRESERVATION spec is now concrete, the IRM partition has a new 5-env per-human candidate, and a NEW lever (hard-axis-aware FT) joins the menu.
- Decision 2 (B.III.3): unchanged — the KLIEP gate threshold table from Task #4 §4 is the concrete policy.
- Decision 3 (production switch): SHARPENED — **switch to P8A at τ=0.59**, with documented dor real-FPR upper-CI noise risk (~3%). E2B is also a viable per-ckpt-cal pass.

**Working-log totals**: 5 RESULTS_FACTS + 5 AGENT_PROPOSAL docs written; 5 analysis folders created; ~5 hrs wall ($0 cost).

**Open follow-ups** (CPU, all $0, all under standing greenlight):
- Visual inspection of hard-cluster Xinhe-fake vs easy-cluster frames (15 min) — would inform H1/H2/H3 for face-pool mechanism
- OPTB ↔ lockbox substrate-match (1 hr) — completes the substrate-distance map
- Project hard-axis on lockbox cohorts (1 hr) — tests if the Xinhe-fake hard-cluster axis generalizes
- Re-evaluate Xinhe-may6 (92 frames) at per-ckpt-calibrated τ for all 5 ckpts (~5 min)
- Score 4 missing dor-webcam-false-flag pools (~30 min) — cross-checks P8A production switch
- 9-suite contract re-evaluation under per-ckpt τ (~1 hr)
- Pre-extract P8A reference features on substrate-pair pool (~5 min CPU/MPS) — prerequisite to the B.I.1 GPU launch

### 2026-05-23 (PM, +~30 min CPU) — three quick follow-ups landed

After the main sequence 1→2→7→3→4, three additional CPU follow-ups completed under standing greenlight:

**Memory updates** (`~/.claude/projects/.../memory/`):
- `project_per_ckpt_tau_recal_reframes_verdict_2026-05-23.md` — locks in the per-ckpt-τ + P8A=production verdict for future sessions; prevents repeating the cross-ckpt-const-τ mistake. Index entry added to MEMORY.md.
- `project_xinhe_fake_hard_cluster_clip_axis_2026-05-23.md` — locks in the LR axis + face-pool concentration finding; pointer to the saved LR coefficient. Index entry added.

**Xinhe-may6 re-evaluation at per-ckpt-calibrated τ** (`analysis/xinhe_may6_per_ckpt_tau_2026-05-23/`):
- **P8A is bulletproof on Xinhe-may6 at τ=0.59: 0.0% FPR on both may5 (60 frames) and may6 (92 frames).** Confirms it's the only ckpt that passes BOTH the team-identity bar AND the Xinhe-may6 false-flag check at per-ckpt-cal τ.
- E2B: 35.9% may6 FPR at per-ckpt τ=0.72 (lower than the 57.6% at τ=0.5; still bad). Structural Xinhe fragility persists across τ.
- SlotAv2_FACE: 15.2% may6 FPR at per-ckpt τ=0.68 (much lower than 91.3% at τ=0.5; the 91.3% was largely calibration artifact, but the 15.2% is still 3× over a 5% floor). Face-pool genuinely fragile on may6.
- T5C, SlotAv2_CLS: clean on may6 at per-ckpt τ, but fail team-identity bar anyway.

**P8A reference feature cache built** (`analysis/output_preservation_spec_2026-05-23/outputs/p8a_substrate_pair_reference_features.npz`, 59.5 MB):
- Combined cache of P8A L11 + L8 features on the 1,825-pair substrate pool (5,475 clean + 5,478 teams frames)
- Keyed by blob_path + pair_id + identity_id + source
- README at `analysis/output_preservation_spec_2026-05-23/P8A_REFERENCE_CACHE.md` with usage snippet
- **Prerequisite to the B.I.1 GPU launch satisfied** — when the trainer implements the cosine-distance aux loss per the spec, it can load this cache directly.

**OPTB ↔ lockbox substrate-match** (`analysis/kliep_substrate_match_2026-05-23/outputs/discriminator_results_with_lockbox.csv`, appended to the RESULTS_FACTS §5):
- OPTB ↔ lockbox-real: 100% CV-separable — OPTB is far from BOTH team-identity AND lockbox.
- **dev_real ↔ team_id_real: 87.4% CV accuracy** — LOWEST of all pool-pair comparisons tested today. dev is genuinely the closest training pool to team-identity (consistent with the frozen-CLIP MLP_DEV being the best Option A head).
- team_id_real ↔ lockbox_real: 98.1% — moderate distance (team-identity and lockbox share some capture-pipeline overlap but are still highly mismatched).
- **Updated substrate-match policy context**: nothing achievable today qualifies as "well-matched" (≤65%); dev at 87% is the floor. Any acquired webcam set needs <80% CV vs team-identity to materially help.

**Substrate-distance ordering** (closest → farthest):
1. dev_real ↔ team_id_real: 87% (closest pair)
2. team_id_real ↔ lockbox_real: 98%
3. {OPTB, dev} ↔ lockbox + {OPTB} ↔ everything: 100%

### 2026-05-23 (PM, deployment-side CPU work) — three new analyses landed; recommended option changed

User raised the operational concern: "majority vote solves my problem if per-frame FPR truly stays under 15% per identity in production — but I don't trust the 5-human sample, fear some users will land at 50-70% per-frame FPR and majority vote breaks."

Three new CPU analyses + the shipment-options doc were produced.

**Shipment-options doc** (`docs/packet_retrospectives/plans/SHIPMENT_OPTIONS_2026-05-23.md`): operator-facing decision menu. Initially 3 options (τ=0.59 / τ=0.10 + abstain / two-tier); after MV-robustness analysis, a fourth Option 0 was added that strictly dominates Option 1.

**Majority-vote robustness analysis** (`analysis/majority_vote_robustness_2026-05-23/` + `analysis/mv_window_weighting_2026-05-23/`):
- Per-cohort within-human variance is real: dor aggregate 15% but `dor_morning` cohort alone is 36.7% per-frame FPR; Xiang aggregate 7.9% but `xiang` cohort 20%. User's fear validated.
- Mac-Roee (out-of-scope per deploy spec but same person, different device): per-frame FPR 67.7% on `Roy_D` and 64.0% on `bla_bla_chow__s2` at τ=0.10. **Majority vote inverts when per-frame FPR > 50%** (Roy_D MV-failure 92.4% at simple-maj τ=0.10).
- Window-size sweep: larger N (50, 100) makes simple-majority WORSE on Mac-OOD (per-frame FPR doesn't change; majority just becomes more stable in the wrong direction).
- **Supermajority-70% rule (flag if >70% of frames over τ) is the load-bearing finding**: at τ=0.15, gives Xinhe per-segment recall 90.9% AND cuts Mac-OOD MV-failure dramatically (Roy_D 92.4% → 15.9%; bla_bla_chow__s2 86.5% → 7.5%; in-distribution worst cohort dor_morning to 0%).
- Pareto-dominance table: τ=0.15 + supermaj-70% **strictly dominates** simple-majority τ=0.59 on every axis (better recall, better OOD safety, lower in-dist MV failure).

**Device-OOD detector** (`analysis/device_ood_detector_2026-05-23/`): NEGATIVE RESULT.
- LR on frozen-CLIP L11 features, Roee-Mac (498) vs Roee-Windows (330): 1.0 5-fold CV AUC.
- BUT projection of all team frames shows the detector overfit to "Roee on his specific Windows laptop" — all other team-humans' Windows captures (Xinhe, Xiang, Noyn extras) classified as "Mac-like" (margin > 0).
- The detector is essentially an identity-AND-device detector, not a device-only detector.
- **Implication**: Can't use this as the abstain trigger for SHIPMENT_OPTIONS Option 2. The fraction-over-τ heuristic (used in Option 0 / 2 / 3) is the better trigger.
- A useful device-OOD detector would need multi-person training data — in scope for the broader B.III.3 (7000-webcam dataset) lever.

**The NEW shipment recommendation (after this session's work)**:

Option 0 (NEW): τ=0.15 + supermajority-70% vote rule
- Per-segment recall: Xinhe 90.9%, dor 98.6%, Xiang 100%
- In-distribution MV-failure: 0% on all cohorts including dor_morning
- Mac-OOD MV-failure: Roy_D 15.9%, bla_bla_chow__s2 7.5%, bla_bla_chow 0%
- **Strictly dominates** Option 1 (τ=0.59 simple-majority) on recall + OOD safety.
- Engineering: minimal — same N-frame window, just change vote-threshold from 0.5 to 0.7.

This is now the recommended ship in SHIPMENT_OPTIONS.

### 2026-05-23 (PM, score-shift sensitivity analysis) — Option 0 recommendation REVERSED

User raised follow-on concern: even with the supermajority-70% MV rule, what if production users have systematically higher per-frame scores than the 5-human sample? Specifically, "some users at 50-70% per-frame FPR" would break MV.

**Score-shift sensitivity analysis** (in `analysis/mv_window_weighting_2026-05-23/`): simulate adding a constant Δ to per-frame scores per cohort, measure MV-failure rate at each Δ.

| Option | Static recall | Static Mac:b_b_c__s2 MV-fail | Critical Δ to break Mac:b_b_c__s2 | Critical Δ to break Roy_D |
|---|---:|---:|---|---|
| Opt 0 (τ=0.15 supermaj-70%) | 91% Xinhe | 7.5% | **+0.10** | +0.05 |
| Opt 1 (τ=0.59 simple-maj) | 75% Xinhe | 0.4% | **≥ +0.30 (robust)** | +0.125 |

**Mechanism**: high τ + simple-majority is robust to drift because the high τ filters out the drift-affected score tail. Low τ + supermajority lives at the edge of the score distribution where small drifts dominate.

**SHIPMENT_OPTIONS doc updated**: Option 1 (τ=0.59 simple-majority) is now the recommended ship. Option 0 is downgraded to "upgrade once production telemetry confirms user population is well-represented by the 5-human sample." The Pareto-dominance claim from the previous working-log entry only held under the no-drift assumption.

**IQ-based device-OOD detector** (Task #11): **POSITIVE RESULT.** `analysis/iq_device_detector_2026-05-23/`.
- 25 IQ features computed per frame (sharpness, brightness, color cast, LAB, edge density, aspect ratio, etc.).
- LR trained on Roee-Mac (498) vs Roee-Windows (330) — 1.0 5-fold CV AUC.
- **Generalization**: 11 of 15 non-Roee in-dist cohorts correctly classified as Windows-like (vs CLIP detector where 0/15 were).
- Per-frame Spearman ρ with P8A score: 0.52 overall, 0.35 within deploy-only.
- Per-decile P8A FPR rises monotonically: decile 0 (most Windows-like) 4% → deciles 8-9 (most Mac-like) 36-43%.
- Top features (LR coefficients): edge_density, sharpness_laplacian, aspect_ratio, lab_a_std/mean (color cast), color_cast_gb. Physically interpretable device properties.
- The 4 in-dist cohorts that DON'T classify as Windows-like (dor_shkedi__s16, dor_morning, Md_noyn_Sharker__s15, xiang) are 3-of-4 cohorts with elevated P8A FPR — so the "false positives" of the detector concentrate on cohorts that ARE genuinely problematic for P8A.

**Open caveat**: fake frames were not extracted. If fakes have systematically high IQ-margin (= Mac-like), an IQ-margin-based abstain rule could suppress fake detection. Need IQ extraction on fakes before any production-ship of an IQ-margin abstain rule. Estimated cost: another ~30 min CPU + GCS download for ~4100 fake frames.

**SHIPMENT_OPTIONS doc updated**: Option 2's abstain-trigger discussion now references both detectors; the IQ-margin trigger is flagged as the more promising path pending the fake-frame verification.

### 2026-05-24 — fake-frame IQ-axis cross-check landed; IQ-abstain rule REJECTED

Closed the open caveat from the previous working-log entry. Extracted IQ features for all 4,120 fake frames (13 min wall, 0 failures), projected onto the same IQ-margin axis (trained on Roee-Mac vs Roee-Windows REAL frames; fakes held out). Full readout: `analysis/iq_device_detector_2026-05-23/RESULTS_FAKES_2026-05-24.md`.

**Headline**: fakes are NOT uniformly Windows-like as the hopeful outcome would have required.

- **fake_target_dor is bimodal**: 61% Mac-like (margin>0), 18.6% extreme-Mac (margin>+3). At an abstain threshold of T=+3, **7 specific dor fake-cohorts become COMPLETELY undetectable**: dor_shkedi__s16, dor_fake_trump_regular, dor_fake_bill_gates_regular, dor_fake_chinese_regular, dor_fake_face_1_enhanced, dor_fake_face_1_regular, dor_fake_chinese_enhanced. All 330 frames are margin>+6 AND 95-100% currently detected by P8A. Abstaining silences them entirely.
- **fake_target_Xinhe** (strongly Windows-like, margin -4 to -8) and **fake_target_Xiang** (mostly Windows-like, 5.5% Mac-like) are safe from any IQ-abstain rule.
- **Per-decile pathology**: decile 8 (margin +1.86 to +7.35) has real_FPR=40% AND fake_recall=100%. The abstain rule drops both signals together — IQ-margin is NOT orthogonal to true fakeness.
- **Per-frame Spearman ρ within fakes = +0.40** (same direction as within reals, +0.35-0.52). Mac-like fakes are P8A's MORE-confident fake detections, not borderline cases.

**Abstain-threshold cost curve (deploy-relevant)**:
- Peak selectivity at T=+2 (real:fake abstain ratio 1.37×). At T=+2: real_FPR@τ=0.59 drops 2.36% → 1.4% (40% reduction); fake_recall@τ=0.59 drops 75.66% → 72.7% (4% reduction).
- At T=+5 or higher, the rule **inverts** — suppresses MORE fakes than reals (ratio 0.46×).
- Aggregate trade looks tolerable on the surface, but the recall loss is **concentrated on specific fake methodologies**, not uniform. Total blindness to "trump/bill-gates/chinese deepfake swaps of dor" is operationally worse than aggregate-recall would suggest.

**Verdict**: IQ-margin abstain rule REJECTED for hard-abstain use. The IQ signal is still useful in soft forms:
1. Operator telemetry / alert (flag OOD users, don't auto-act)
2. Soft down-weighting in MV (weight Mac-like frames less, don't drop)
3. Per-user τ shift (Mac-like frames need higher confidence to flag)
4. Per-user calibration aid (from longitudinal telemetry)

**SHIPMENT_OPTIONS doc updated**: the "Note on Option 2's abstain trigger" section now reflects the REJECTED verdict. Option 2 falls back to its original form (fraction-over-τ as the only abstain trigger). The top-of-doc recommendation stays at Option 1 (τ=0.59 simple-majority); this round's work confirms rather than changes that choice.

**Files**:
- `analysis/iq_device_detector_2026-05-23/scripts/extract_iq_fakes.py` — fake-IQ extraction
- `analysis/iq_device_detector_2026-05-23/scripts/analyze_fakes_on_iq_axis.py` — projection + cost-curve
- `analysis/iq_device_detector_2026-05-23/outputs/per_frame_iq_fakes_v2.parquet` — 4,120 fake frames × 30 cols
- `analysis/iq_device_detector_2026-05-23/outputs/per_fake_cohort_iq_margin.csv` — 52 fake-cohort summary
- `analysis/iq_device_detector_2026-05-23/outputs/per_decile_fake_vs_real.csv` — combined decile table
- `analysis/iq_device_detector_2026-05-23/outputs/abstain_threshold_cost_curve{,_deploy}.csv` — abstain curves
- `analysis/iq_device_detector_2026-05-23/RESULTS_FAKES_2026-05-24.md` — full factual readout

### [reserved for future sessions]

When work resumes on this review (e.g., responses from author, additional evidence, decision-changes), append a dated entry here describing what was done, what changed, and any follow-ups created.

---

End of review.
