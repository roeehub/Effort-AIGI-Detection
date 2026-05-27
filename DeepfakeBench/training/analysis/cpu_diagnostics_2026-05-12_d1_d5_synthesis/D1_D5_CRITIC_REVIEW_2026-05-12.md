# D1-D5 critic-review — independent agent (2026-05-12)

> **Status**: OPINION, not RECORD. Critic-review per the task spec at [`docs/packet_retrospectives/D1_D5_CRITIC_REVIEW_TASK_2026-05-12.md`](../../docs/packet_retrospectives/D1_D5_CRITIC_REVIEW_TASK_2026-05-12.md). Authored as Pass-1-then-Pass-4 independent reviewer: FACTS docs read before opening the planning agent's OPINIONS doc. Disclaimer per `AGENT_GUIDE.md` Rule 5 — past framings in this project have been demonstrably wrong; this critique is itself another candidate framing for the ledger and should be treated skeptically. The user wants sharper next-step framing from any disagreement, not deferential agreement nor contrarianism for sport.

---

## 1. My "deeper story" paragraph (formed BEFORE reading OPINIONS §1)

The forgery signal lives in raw CLIP B16's L11: D2 §2 reports 5-fold CV AUC = 1.000 on chronic-6 with no FT at all. FT does not add the signal — FT *rotates* the readout direction. The honest size-of-effect is **a 4.8°-8.5° drift from CLIP-frozen's 83.68° toward IQ-PC1**, NOT 9-13°: the latter only appears if you compare FT'd ckpts to a *theoretical* 87.5° random baseline (D2 §7) rather than to the empirical CLIP-frozen counterfactual. The chronic-6 vs non-chronic split is the load-bearing axis: D5 §4.4 cleanly partitions 5/5 chronic above and 5/6 non-chronic below the IQ-predicted line **for all three measured ckpts**, and the same 3 identities (`real_dor`, `Cam_Test`, `PC_Generator`) recur in every top-5 residual list. Whatever "chronic-6" tags, it persists across recipes and is not captured by 6 IQ axes — so it is identity-level signal, not an IQ shortcut per se. D1's per-identity sign flips on `saturation_mean` (β=−6.15 on Roy_D vs +3.33 on dor_shkedi) say the same thing from the other side. **D4 is the most disruptive single FACT**: at dev-calibrated 5% τ, lockbox FPR is *below* dev for all three ckpts (0.92% / 2.4% / 2.2%) — the contract's `lockbox_real_fpr` tiebreak is measuring per-quartile residual structure / τ-tail-density, NOT lockbox difficulty. D3's encoder-substrate-specificity is real at the feature level but is operationally benign at dev-cal τ at the OUTPUT level. The path to a single-model candidate is **data-side first** (ingest lockbox-style + v2-substrate-diverse frames so the encoder cannot keep substrate-specific directions; D3's resolution criterion (b) already names this), training-time bundles second.

---

## 2. Side-by-side: planning agent's §1 claims vs my verdict

| Tag | Planning-agent claim | My verdict | One-sentence justification |
|---|---|---|---|
| **C1** | Forgery signal is already in raw CLIP B16 | **Agree (caveat)** | D2 §2 supports it on chronic-6 at n=282 (41 fakes); robustness at larger fake-class n is the open loop `clip-frozen-chronic-6-auc-robustness` and is load-bearing for Slot 2. |
| **C2** | Standard FT does not add the signal — FT modifies how it's projected | **Partial agree** | D2 alone shows FT'd ckpts are also AUC ≥ 0.997 on chronic-6, but "does not add" overreaches — non_chronic + full cohort don't have the AUC=1.0 baseline anyway, so D2 doesn't *prove* the negative; it shows FT redirects a signal CLIP already separates *within chronic-6*. |
| **C3** | FT pulls the chronic-6 encoder separation direction **9°-13°** toward color-axis IQ alignment | **Disagree on magnitude** | The 9-13° figure compares FT to a **theoretical** 87.5° random-vector baseline that D2 §7 itself flags as not empirically validated; the empirically grounded counterfactual is CLIP-frozen at 83.68°, against which the FT drift is **4.8°-8.5°**. Smaller, still real. |
| **C4** | P8A learns identity-cluster **memorization** | **Disagree on the framing** | D5 R²=0.22 and D1 sign-flips are equally consistent with "P8A learned identity-level invariance (uses fewer per-identity cues that aggregate to consistent IQ rules)" — a positive property. "Memorization" is value-laden interpretation, not data-forced; structured per-identity IQ usage is a better neutral characterization. |
| **C5** | T5C step3500's IQ-correlation lives at the head, not the encoder | **Agree (with caveat)** | D5 R²=0.60 + D2 angle 75° are reconciled cleanly only if magnitude (head amplification) does the work; the magnitude-vs-direction confound the brief flagged is real but the alternative reading (encoder direction has high IQ projection magnitude despite mostly orthogonal angle) is geometrically incompatible with 75° on a unit-normed probe. |
| **C6** | Every FT'd encoder retains DIFFERENT separations per substrate | **Agree, but its operational meaning is partially overstated** | D3 §2.1 supports it cleanly at the feature level (4/4 ckpts fail substrate-agnostic > 0.95 both directions). But D4 shows this DOES NOT manifest at dev-cal τ — lockbox FPR is *lower* than dev. The encoder is substrate-specific in direction; the output is operationally substrate-stable. |
| **C7** | P8A-output distillation and P8A-L11 anchor levers are off the table | **Agree on distillation, disagree on L11 anchor retraction** | Distillation retraction is sound (per-identity sign-flips would propagate). The L11 anchor retraction rests on JOB1 §3 "uniform L11 cosdist 0.5053 vs 0.5063 chronic vs healthy" — but that's median-mixed-across-identities; JOB2 §4 shows per-identity overfire rates of 30%/17%/44% on dor/bla/Roy_D. Median uniformity ≠ direction uniformity. The retraction is premature. |

---

## 3. Each contested claim in detail

### 3.1 C3 — the size of the FT-induced angle drift is overstated by 2×

D2 §3 reports angles to IQ-PC1 within chronic-6: CLIP-frozen 83.68°, P8A 78.54°, E2B 74.95°, T5C 75.19°, T3 77.09°. The brief flagged "9°-13°" as the planning agent's claim; D2 §7 explicitly states the 87.5° baseline is theoretical (uniform random unit vectors, which trained LR-probe coefficients are NOT). The empirical counterfactual for "what FT changed" is CLIP-frozen at 83.68°, which gives **4.8° (P8A), 6.6° (T3), 8.5° (T5C), 8.7° (E2B)** of drift. That's half the planning agent's headline. The planning agent acknowledges this in §5 uncertainty #2 ("The 87.5° random-orthogonality baseline is theoretical, not empirical") — so by the doc's own caveats, C3 should be MEDIUM confidence, not HIGH.

This matters operationally: a 5° drift in 512-d space is small enough that a counterfactual GRL experiment (Slot 1 in the plan) might "succeed" by moving the angle 2° while not actually changing FPR behavior. The falsifier "post-training chronic-6 IQ-PC1 angle ≥ 82°" (1.7° below CLIP-frozen, ~3° above P8A) is too narrow a target relative to the noise floor the planning agent itself flags.

### 3.2 C4 — "memorization" is one interpretation, not the only one

D5 §3.3 ranks `real_dor`/`Cam_Test`/`PC_Generator` as recurring top-residual identities across all 3 ckpts. The planning agent reads this as "P8A learned identity-specific rules." An equally valid reading: **`real_dor` and `Cam_Test` are non-chronic identities where the model scores BELOW the IQ-predicted line** (D5 §4: `real_dor` P8A residual = −2.325, observed_mean 0.0084, predicted 0.0793). These identities are *over-classified-as-real* by every ckpt — meaning the model has additional information beyond IQ that lets it correctly call them real. That's invariance, not memorization. The recurring pattern across ckpts says **the data has a structure (identity-level forgery cue) that every recipe is picking up**.

What would actually distinguish these readings: a train-manifest grep for `real_dor`/`Cam_Test`/`PC_Generator`. The planning agent's §5 uncertainty #5 names this exact test as untested. Until run, "memorization" is an interpretive choice, not a fact.

This matters for the slot plan: if P8A is "memorizing", then distilling its outputs is poison. If P8A "learned identity-level invariance from training data that the chronic-cohort identities are *in*", then distilling P8A's outputs on chronic-cohort reals is exactly the transfer signal we want — and the retraction in C7 is unsound.

The MAY6 production-drift evidence (memory `project_xinhe_may6_falseflag_2026-05-06`, packet T3.md per-cohort retest) is the load-bearing test. **P8A: 0/92 may6 false-flags. E2B (deployed): 53/92. T3_S1_step2500: 71/92.** If P8A were truly identity-memorizing, fresh real Xinhe frames from a new capture day should not be handled cleanly. They are. The planning agent's framing should reconcile with this; it doesn't.

### 3.3 C6 — the dev↔lockbox transfer gap may not be operationally binding

D3 establishes substrate-specific encoder directions at the feature level. D4 says: at dev-calibrated 5% FPR, lockbox FPR is 0.92% (P8A) / 2.40% (T5C) / 2.19% (T3) — **all below 5%**. The encoder cannot be a deployment problem if the head still produces operationally lockbox-safe output at the operating τ.

The planning agent acknowledges this in §5 uncertainty #8 ("contradicts the contract-scorecard reading and the user's hands-on experience") but doesn't propagate the implication to §3 (the slot plan motivation). The slot plan's central premise — that the encoder gap must be repaired — depends on framing C6 as load-bearing. D4 says it's not load-bearing at deployment τ. **The right resolution is not "Tier 2 escalation on D3 sample size"; it's "the substrate gap framing has moved from operational to mechanistic."**

The mechanistic question is still interesting (why does the encoder retain substrate-specific directions?) — but it's no longer the primary deployment blocker. The deployment blocker is the contract's `lockbox_real_fpr` tiebreak, which D4's quartile analysis suggests is measuring **τ-tail-density on specific IQ-quartile cells** (T5C `color_a_dev` Q4 dev 18.5% vs lockbox 4.2% — a 14pp dev-side over-fire that drags the dev-cal τ tightness with it), not lockbox substrate difficulty.

### 3.4 C7 — the L11 anchor retraction overshoots

The planning agent retracts Slot 1 (L11 anchor on 5 identities) on the basis of JOB1 §3: T5C cosdist median ≈ 0.50 on chronic-6 reals and healthy reals (0.5053 vs 0.5063, less than 0.001 apart). The argument: "drift is uniform, not chronic-specific, so anchoring on chronic-6 cohort is mis-targeted."

This conflates two claims. JOB1's median is over frames within a cohort. JOB2 §4 shows the *real-side score* overfire rate per identity (T5C step3500 − P8A > 0.5): **dor_shkedi 30.09%, bla_bla_chow 17.37%, Roy_D 44.31%**, with the top-5 covering 95.4% of all 766 overfires. The per-frame L2 distance and the per-identity score-difference distribution are different quantities. A small per-frame L2 in a direction aligned with the head's decision boundary can produce a large score shift on a SPECIFIC identity even when distance medians are uniform across identities.

The retraction is sound IF you accept that anchoring should be uniform-cohort. But the original Slot 1 design from `STAGE_A_PLUS_PACKET_PLAN_OPINIONS_2026-05-12.md` §2.1 anchored on the 5 high-overfire identities precisely because that's where T5C's *output* diverged from P8A. JOB1 doesn't dispute that. The retraction confuses the *L11 distance* signal with the *head output* signal — and L11 anchor's job was to constrain the latter via the former.

---

## 4. The slot plan — my version

I keep 2 of the planning agent's slots, modify 1, drop 1, and add 1. Net: 4 slots, structurally distinct, with the data-side lever the planning agent omitted.

### Slot A (HIGHEST) — Lockbox-style data ingestion into training (the slot the planning agent missed)

**Hypothesis grounded in evidence**: D3 establishes substrate-specific encoder directions. D4 shows lockbox distributes very differently from dev across IQ axes (lockbox 68.5% mass in `lap_hi · md_hi`; dev distributes 18.6% there). HEAD_RETRAIN_FACTS shows training on dev features alone fails to transfer to lockbox. The most direct fix for "encoder retains different separations per substrate" is to train on the missing substrate.

**Mechanism**: identify the v2-substrate-diverse and lockbox-style data (per memory `project_v2_substrate_is_dor_diverse_swap`), add them to the training pool, FT from P8A_step5000. No new loss code.

**Falsifier**: post-training DEV→LOCKBOX encoder probe transfer AUC stays ≤ 0.70, OR lockbox_real_fpr at FPR-cal τ ≥ 0.025.

**Confirmation**: DEV→LOCKBOX transfer AUC ≥ 0.85 AND lockbox_real_fpr ≤ 0.018 AND lockbox_fake_recall ≥ 0.55 AND no per-identity FPR > 30%.

**Cost**: ~$40 Vertex. **No new code** — just data-pool reconfiguration.

**Why this isn't in the planning agent's plan**: the planning agent's slot framework leans on loss interventions (GRL variants, anchor losses, LoRA). D3's resolution candidate explicitly mentions "data-side (substrate-diverse ingestion)" as one of the two structural fixes; the planning agent took the other path. The brief's §3.2 explicitly flagged this omission.

### Slot B (HIGH) — Continuous-axis-GRL on T5C step3500 base [= planning agent's Slot 1]

**Keep as-is in mechanism**, but **revise the falsifier**. Planning agent's falsifier "chronic-6 IQ-PC1 angle stays at T5C's 75.19° or worse" sets the bar at no movement at all; the better falsifier is **angle ≤ 80°** (haven't recovered the empirical CLIP-frozen baseline of 83.68° within ~4°). Confirmation: angle ≥ 82°, real-side |r| ≤ 0.15 on each axis, lockbox_real_fpr ≤ 0.020, lockbox_fake_recall ≥ 0.55.

**Cost**: ~$30-40. Seed 9603.

**Why I keep it**: even if my interpretation in §3.1 (5° drift, not 13°) is right, this slot is the cleanest test of whether the IQ-angle is *causally* responsible for chronic-FPR. The open loop `chronic-6-encoder-iq-angle-drift-during-ft` (per thread §865) names this as the close criterion. Cheap counterfactual.

### Slot C (MEDIUM) — L11 anchor on T5C, 5-identity cohort [restored from prior plan, not retracted]

**Mechanism**: per `STAGE_A_PLUS_PACKET_PLAN_OPINIONS_2026-05-12.md` §2.1 — pull T5C's L11 CLS features toward P8A's L11 CLS features on real frames from the 5-identity overfire cohort (dor_shkedi/bla_bla_chow/Roy_D/xiang/dor); λ schedule 0 → 0.5 linear warmup; FT-from-P8A base.

**Why I disagree with the retraction**: see §3.4 above. The JOB1-cited "uniform L11 drift" doesn't address the question "does pulling features on 5 identities help"; it just says the average drift magnitude is similar across cohorts. JOB2's 95.4%-on-5-identities concentration is the directly-targeting evidence.

**Falsifier**: chronic_6 inv_mean ≤ T5C's +0.0224 (no recovery) OR HDTF teams_dev regression > 5pp vs E2B (P8A's specific behavior didn't transfer).

**Confirmation**: chronic_6 inv_mean ≥ +0.035, dor_real_lockbox FPR ≤ 1.0% at FPR-cal τ, lockbox_fake_recall ≥ 0.50, HDTF teams_dev within 5pp of E2B.

**Cost**: ~$50. New code: `loss/l11_anchor_loss.py` (~80 lines) + P8A reference precompute (~30 min CPU). Seed 9604.

### Slot D (MEDIUM-LOW) — LoRA on P8A layers 10-11 [= planning agent's Slot 4, with caveat]

**Keep at MEDIUM-LOW**, not MEDIUM. The planning agent's own evidence is internally inconsistent here: D3 says P8A's encoder is the MOST substrate-specific (DEV→LOCKBOX 0.594), and LoRA freezes that encoder. The motivation in planning agent §3.3 acknowledges this tension but doesn't resolve it. My read: LoRA on P8A is worth running because the *adapter delta* might add the substrate-diverse direction the frozen base lacks, and Stage 2 found every full-FT regime drifts the dor cluster — so a parameter-efficient method is the structurally distinct alternative. But the prior on "this will lift lockbox AUC by 15pp" is low.

**Falsifier**: lockbox AUC stays at ~0.76 (head-retrain β-outcome level — FT regime is not the binding constraint), OR dor-cluster regression reproduces (Roy_D real-FPR > 30%).

**Cost**: ~$30-40. Code agent already in flight per `STATE.md`. Seed 9601.

### Drop: planning agent's Slot 2 (B16-scratch + Fourier-aug bands 8-13 + continuous-axis-GRL + T3 keep-list)

**Why drop**: this slot stacks **four** interventions on a known-fragile training base. Memory `project_l14_does_not_break_viso_ceiling` documents that B16-scratch+CE (E2B) doesn't break the viso ceiling. PD packet experience (`packets/PD.md`) shows corr-penalty can shift shortcuts to untargeted axes; continuous-axis-GRL on 6 axes is closely related — adding 6 axes doesn't eliminate the "7th unmeasured axis" risk. This stack violates AGENT_GUIDE.md Rule 6's single-lever discipline directly: the planning agent §1.1 marks Slot 2 as MEDIUM but the slot's design has 4 simultaneously-tested levers.

If the planning agent insists on a scratch-bundle slot, the cleaner version is **B16-scratch + the single anti-shortcut lever with the strongest individual prior** (Fourier-aug bands 8-13 from memory `project_fourier_band_overlap_2026-05-06`'s greenlight) and nothing else. Even then, the priors against full-recipe-scratch in this project are accumulated (E2B, L14).

### Drop: Slot 3 from planning agent's plan (T3_SLOT1 + 1024 classifier)

Agree with planning agent's recommendation to drop, but **for a different reason**. Planning agent drops on D4 r=−0.03 alignment. My §3 critique notes per-quartile lockbox n is 14-32 in some cells; the r=−0.03 is fragile. The real reason to drop: T3_SLOT1 already exists as the rank-4 ckpt per memory `project_t6_t7_t5c_scorecard_2026-05-12.md`; composing two known-rank-3-or-4 ckpts has low ceiling (P8A is still rank-1).

---

## 5. Things the planning agent missed

1. **The MAY6 production-drift evidence is decisive against the "P8A memorization" framing** (per packet `T3.md` post-packet diagnostics §may6: P8A 0/92, E2B 53/92, step1500 6/92, step2500 71/92). If P8A learned identity-specific rules on training-set identities, fresh Xinhe frames from a new capture day should fail. They don't. This needs to be reconciled before C4 is accepted.

2. **The data-side lever is structurally indicated by D3 itself** (resolution candidate (b) explicitly names "data-side ingestion or training-time anti-shortcut") and is absent from the 4-slot plan. The brief flagged this gap.

3. **P22 / `pipeline_randomization` step1k success** (memory `project_p22_cpu_followups_reframe_2026-05-02`) is the prior example of an augmentation-axis intervention that DID move the needle (3× macro fake recall). It's not cited in the synthesis. Slot 2's Fourier-aug-bands rationale should be measured against P22's effect size, not against a clean-slate prior.

4. **Stage A's specialist routing finding** (P8A on chronic, T5C else gives +37.9pp lockbox fake, preserves all may6/dor invariance) is rejected on NO-ENSEMBLE grounds — correctly. But the FACT it surfaces is load-bearing for slot design: **the encoder pair carries complementary information**. The single-model slot that captures this is "FT T5C with explicit preservation loss against P8A's chronic-cohort behavior" — that's Slot C (L11 anchor) in my plan. The planning agent had this slot, dropped it.

5. **D4's lockbox-easier-than-dev finding deserves its own thread/open-loop**. The implication that the contract's `lockbox_real_fpr` tiebreak is measuring τ-tail-density rather than substrate difficulty is structurally important for how to read all future scorecards. The planning agent flags it as uncertainty #8 in §5; it deserves a thread amendment, not just an in-doc footnote.

6. **The 5-identity overfire concentration in JOB2 (95.4% on top-5) is consistent across ckpts**: same identities recur in D5 top-5 residual list. This convergence — same identities flagged by different methods — strengthens the per-identity-targeting case that the planning agent retracted. JOB1's cosdist uniformity (the retraction basis) is a distance-magnitude measurement that doesn't address direction.

---

## 6. Unresolved questions / next CPU diagnostic

If I had to authorize one $0 CPU diagnostic before any GPU spend:

**D6 — Empirical orthogonality baseline + larger CLIP-frozen sample (~3h MPS, $0)**: Compute the empirical angle distribution between LR-probe weight vectors trained on independent random 50/50 splits of the same 800-frame triptych (e.g., 50 random bootstraps). This is the "what's the noise floor on angle measurements" calibration that D2 §7 admits is missing. Without it, the 4.8°-8.5° (or 9°-13°) drift can't be sized against measurement noise. Simultaneously, extend the CLIP-frozen probe to a 500-1000 chronic-6 sample with more fakes (per open loop `clip-frozen-chronic-6-auc-robustness`) to test whether AUC=1.000 holds at scale.

If D6 returns: empirical baseline ≥ 80° (current 87.5° was overstated) → the FT drift to 75-79° is even less impressive → Slot B becomes lower priority. If empirical baseline still around 86°+ → planning agent's framing is robust.

Before authorizing any GPU: also run the train-manifest grep for `real_dor` / `Cam_Test` / `PC_Generator` per §5 uncertainty #5. This is the single load-bearing test that distinguishes "memorization" from "learned identity-level invariance." 30 minutes; gates whether Slot C (L11 anchor) and Slot A (data ingestion) are well-targeted.

---

## 7. Self-criticism

Things in this review that might be wrong, and how a future agent could check:

1. **My "4.8°-8.5° not 9°-13°" complaint depends on which baseline is correct.** If a future empirical-baseline calibration (D6 above) shows the angle distribution between independent probes is ~83°, then the FT drift relative to that baseline IS in the 4-8° range. If empirical baseline is ~87°, the planning agent's 9-13° framing is closer to right. I assumed the conservative comparison; the underlying argument (re-grade C3 from HIGH to MEDIUM until empirical baseline is run) holds regardless of which way D6 falls.

2. **My "P8A identity-invariance vs memorization" reframe leans on the may6 evidence.** That evidence is convergent but n=92 and was generated by a non-trained-on cohort that *might* still share latent factors with training. The train-manifest grep is the clean test. Without it, I'm asserting "convergent indirect evidence" against the planning agent's "OLS R² differential", and neither is dispositive.

3. **My Slot A (data ingestion) is light on the engineering specifics.** I haven't specified which lockbox-style frames, in what mix, with what label discipline. A future agent should treat my Slot A as the right *target* (close the substrate gap data-side) with the recipe to be worked out from `project_v2_substrate_is_dor_diverse_swap` + the lockbox manifest. The "no new code" claim assumes existing data-pool config supports this; if it doesn't, the cost estimate is wrong.

4. **My Slot C (L11 anchor restoration) might still have the issue the planning agent flagged.** If P8A's L11 features on those 5 identities encode P8A-specific (non-generalizable) per-identity sign-flipped IQ rules, pulling toward them inherits the same problem the planning agent retracted distillation for. The mechanistic difference (L11 features vs output prob) should attenuate this (features are higher-dim and contain more than the decision boundary) but doesn't eliminate it. A future agent should design the close-criterion to test HDTF transfer, which is the cleanest test of "did we inherit P8A's specific quirks vs principled invariance."

5. **I haven't verified the Stage A specialist routing numbers against the contract scorecard.** Specifically, the +37.9pp lockbox fake recall at FPR-cal τ assumes the FPR-cal τ on the chronic_6 sub-population for P8A and on the non-chronic for T5C aren't double-spent. The MODEL_GOALS NO-ENSEMBLE rule makes this moot for deployment, but the FACT carries weight in slot design. Worth a sanity-check.

6. **I'm rejecting Slot 2 (B16-scratch + bundle) partly on AGENT_GUIDE Rule 6 grounds.** A counter-argument: the bundle IS a single experimental unit if you treat "anti-shortcut training" as the lever rather than its individual components. The planning agent could (and should) reframe it. My critique is that the falsification structure as written doesn't decompose what was responsible for any lift or regression — the slot isn't WRONG, but it's not single-lever-clean, and Rule 6 is in place because we've burned compute on muddled-lever runs before.

---

**Bottom-line headline for the user**:

My version of the deeper story agrees with the planning agent that the forgery signal lives in raw CLIP and FT rotates the readout direction, but **disagrees on three load-bearing characterizations**: (a) the FT-induced IQ-angle drift is ~5°, not ~12°, when compared to the right baseline; (b) "P8A identity-cluster memorization" is one of several equally-supported readings, and the convergent indirect evidence (may6 production-drift) actually favors "learned identity-level invariance"; (c) D4's lockbox-easier-than-dev result implies the dev-to-lockbox transfer gap is **mechanistically real but operationally non-binding** at the deployment τ — which reframes the slot-plan motivation.

My 4 slots are: **(A) lockbox-style data ingestion** (the slot the planning agent missed; structurally indicated by D3's own resolution candidate), **(B) continuous-axis-GRL on T5C** (keep, revise falsifier), **(C) L11 anchor on T5C, 5-identity cohort** (restored; the planning agent's retraction confused distance-magnitude with direction), **(D) LoRA on P8A** (keep at MEDIUM-LOW). I drop the planning agent's Slot 2 (single-lever-discipline violation) and Slot 3 (low ceiling).

The one thing I'd most want investigated next: **train-manifest grep for `real_dor` / `Cam_Test` / `PC_Generator`** ($0, 30 min). It distinguishes "memorization" from "invariance" and gates whether L11 anchor / output distillation are pollutants or assets. Currently the entire slot plan rests on an interpretive choice that this grep would resolve.

---

## §addendum (2026-05-12 evening) — train-overlap audit results

After the user's go-ahead, I ran the train-manifest grep ($0, ~45 min CPU). Full FACTS at [`../train_overlap_audit_2026-05-12/TRAIN_OVERLAP_FACTS_2026-05-12.md`](../train_overlap_audit_2026-05-12/TRAIN_OVERLAP_FACTS_2026-05-12.md). Headline:

**At the frame/bucket level**: all 3 critic-review identities' eval frames live in `gs://teams-faces-data-test-2914-fake-4420-real-feb-28`. That bucket has **0 references across all 158 R13 training yamls**. P8A's Teams training source is a different bucket (`live-deepfake-methods-real-and-fake-frames-cropped-teams-v2`). The 2026-05-04 inventory audit's `also_in_training="yes (teams-v2 bucket is training+ood)"` annotation is set by **method-string match in audit code** (`run_audit.py:340-348`), not by per-identity bucket-content verification. The annotation is a soft bucket-family inference; it is not load-bearing for the memorization claim.

**At the identity (person) level**: unresolved by local-only audit. The training-side bucket `teams-v2` may contain the same humans under different sample_ids / sessions; verification requires a GCS-side audit (~10 min + GCS list-API quota). For `real_dor` specifically, the closest training-side reference is `gs://real-teams-dor-roee/session_20260424_110139/uniform30` — declared `readout_only_external_real_sources` in P8A's yaml (line 262), and `readout_only_*` per `analysis/generate_packet6_yamls_2026-04-23.py:308-309` is "FPR is logged but does NOT gate τ" — i.e., not training data.

**Implications for §3.2 (C4 — "memorization vs invariance"):** the framing tension shifts. My critique argued the may6 invariance evidence (P8A 0/92 false-flags on fresh real Xinhe) favored "invariance" but conceded that the train-manifest grep was the dispositive test. The grep now shows that the eval-substrate frames are **not** in P8A's training pool at the bucket level. This is supporting evidence for the invariance reading, not the memorization reading — though it is not dispositive at the person level. The planning agent's "P8A is identity-cluster memorizing" framing should therefore be re-graded from MEDIUM-HIGH confidence to **LOW-MEDIUM** confidence: bucket-level evidence is now negative, person-level evidence is unverified.

**Implications for the slot plan:**
- **Slot C (L11 anchor on T5C, 5-identity cohort restored)**: the worry that L11 anchor inherits "P8A's identity-specific memorization rules" is **weaker** post-audit. The audit does not confirm those rules are memorization in the strict sense. Keep Slot C at MEDIUM.
- **Slot A (data ingestion)**: unchanged. Independent of the memorization question.
- **The planning agent's C7 retraction of distillation/anchor** rests partially on the (now weakened) memorization claim. The retraction should be re-litigated, not accepted at the planning agent's confidence level. **Output distillation may still be a valid lever class** if P8A's per-identity rules are learned invariance rather than memorization. A bounded experiment (low λ, narrow anchor cohort) is a reasonable test.
- **The contract's `lockbox_real_fpr` tiebreak** (D4 §1) — at dev-cal τ, lockbox FPR is below dev for all 3 ckpts. Combined with this audit's finding that lockbox identities are NOT in training-bucket frames: P8A's low lockbox FPR is genuine generalization, not memorization-induced low FPR. This further weakens the planning agent's framing.

**Open loop opened**: `train-bucket-identity-overlap-gcs-audit` (LOW severity) — closing requires a GCS-side `gsutil ls gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2/samples/ | head -1000` plus identity-name extraction. Cost <10 min. If the result shows the same humans appear in teams-v2 under different sessions, the memorization framing becomes plausible again. If teams-v2 contains entirely different humans, the invariance framing is strengthened. The open loop is LOW severity because the slot plan does not gate on its resolution — Slots A, B, C, D are all worth running regardless.

**Self-criticism of this addendum**: I now have slightly weaker grounds for the §3.2 framing than the original critique stated — the may6-evidence-favors-invariance argument has gained bucket-level support, but the person-level question remains open. If the GCS audit later shows person-level training overlap, my critique's confidence on C4 should swing back toward the planning agent's reading. I have updated the §3.2 framing only by appending this addendum, not by editing the original §3.2 text, so the planning agent can review my original framing as it was first written.
