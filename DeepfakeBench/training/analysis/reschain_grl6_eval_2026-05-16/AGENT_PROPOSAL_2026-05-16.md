# Agent Proposal — Slot α + Slot β overnight interpretation (2026-05-16)

> **Status: OPINION, not RECORD.** Reviewer is invited to disagree. Each load-bearing claim cites RESULTS_FACTS_2026-05-16.md (`§N.M`) or DEEP_DIVE_FACTS_2026-05-16.md.
>
> **Authoring**: same agent who proposed the slot designs, wrote the new
> `resolution_chain_aug` module, ran the 2026-05-15 CPU probe, drafted the
> yamls, launched training, ran the CPU follow-up probe, AND wrote both
> FACTS docs above. **Single-agent session — pass-1 / pass-2 independence
> is not guaranteed.** The next agent should form their own view from the
> FACTS docs FIRST and then engage with this OPINION doc.
>
> **Required disclaimer per AGENT_GUIDE.md Rule 5**: past framings in this
> project have been demonstrably wrong. Examples from the most recent 4 weeks:
> - The "substrate-overfit" framing in the T4 2026-05-11 OPINION (partly retracted in iq_shortcut_thread 2026-05-11 update)
> - P15/P18 GRL packets predicting invariance lifts that did not materialize
> - Cyclic-λ T5-A packet hypothesized to capture attractor moments — refuted same-night
> - face_scale_jitter@0.50 "load-bearing single lever" framing — refuted on T6/T7 as a composable lever (2026-05-12)
> - LoRA-L10-L11 (Slot α/β 2026-05-14 packet) "rank ablation will reveal the chronic-FP trade-off curve" — refuted (the trade-off was hard-coupled, not tunable)
> - **THIS doc adds another candidate framing to that ledger; treat skeptically.**

---

## 1. Executive summary

Slot α + Slot β were launched 2026-05-15 22:39 UTC, both FT-from-T5C-step3500.
Both targeted the resolution-chain instability documented in the 2026-05-15
CPU probe. Scorecard verdict landed 2026-05-16 10:36 UTC.

**Headline**: P8A retains rank-1 (RESULTS §3); the most surprising
non-promotion finding is **Slot β** (RESULTS §5):

- `dev_fake_macro_recall` 0.545 — highest of any ckpt in the scorecard
- `visomaster_enhanced_macro_dev` 0.235 — first visomaster-detection lift above 0.20 on this contract since the packet sequence began
- `lockbox_real_fpr` 0.088 — 3.2× T5C's; the metric that keeps Slot β out of the top 2

**Slot α step3500** — the CPU-probe winner — FAILS the recall floor (0.226 < 0.30,
RESULTS §3). Slot α step1500 promotes (rank 3) but regresses viso 5.6% vs T5C's
13.8% (RESULTS §4). The intended mechanism worked at the CPU-probe level
(`score_range` 0.605 → 0.448) but at the cost of compressing the entire
score distribution including the fake side (DEEP_DIVE §3).

The mechanism claims I propose below are interpretations of the data;
treat with the skepticism warranted by the retraction record cited above.

## 2. High-confidence claims (paraphrases of FACTS)

These are direct paraphrases, not interpretations:

1. P8A holds contract rank 1; no Slot α or Slot β ckpt displaces it. **Citation**: RESULTS §3.
2. Slot α step3500 fails `dev_fake_macro_recall ≥ 0.30` (0.226). **Citation**: RESULTS §3.
3. Slot α step1500 passes all three contract gates with rank 3. **Citation**: RESULTS §3.
4. Slot β step3500 has the highest dev_fake_macro_recall (0.545), highest visomaster_enhanced_macro_dev (0.235), highest deeplive_enhanced_dev (0.738) of any ckpt in this scorecard. **Citation**: RESULTS §5.
5. Slot β step3500 has lockbox_real_fpr 0.088, the largest of any all-pass ckpt and 4.8× P8A's. **Citation**: RESULTS §5.
6. Among the 4 all-pass ckpts, rank ordering is consistent with ascending `lockbox_real_fpr`. **Citation**: RESULTS §3.
7. The 2026-05-15 CPU probe's mechanism close criterion (`score_range ≤ 0.40`) was approximately met by Slot α step3500 (0.448, 12% short of target). **Citation**: cpu_diagnostics_2026-05-15_resolution_chain/RESULTS_OVERNIGHT_FACTS_2026-05-16.md §2 + 8.
8. The scorecard's contract close criteria for Slot α step3500 were not met (viso 0.018, lockbox_fake 0.372 — both well below T5C step3500's). **Citation**: RESULTS §4.

## 3. Mechanism claims (caveated)

### 3.1 The resolution_chain_aug compresses the entire score distribution, not just the real-side instability (MEDIUM)

The 2026-05-15 CPU probe characterized score variance on REALS across 20
down→up variants. Slot α step3500 achieved 25% reduction in median real
`score_range` — a real measurement (RESULTS §6).

DEEP_DIVE §2 and §3 surface that the same training move also compressed
fake-side scores: Slot α step3500 mean fake scores sit in [0.65, 0.78]
across sizes, with the spread narrowed vs T5C. At the scorecard τ=0.860,
a fake-side mean of 0.65-0.78 leaves a thin tail above the threshold —
the fake-recall drop in RESULTS §4 follows.

**Mechanism caveat**: the aug couples real and fake distributions through
the encoder's shared representation. Asymmetric attention to reals (i.e.,
applying the aug only to label=0 frames) would test whether asymmetric
exposure produces real-only stabilization without fake-side compression.
This is untested; the proposed mechanism is one of several possible
explanations.

**Falsifier**: a sister-variant Slot α that applies `resolution_chain_aug`
only to real frames during training. If fake recall is preserved AND
real-side score_range stays low, the asymmetric mechanism is supported.
If fake recall drops anyway, the coupling is through the encoder and
asymmetric application is not the fix.

### 3.2 The 6-axis GRL bit a visomaster-detection axis I did not design it to bite (MEDIUM-HIGH)

I designed Slot β to extend GRL coverage to color_b + luma_mean, expecting
those axes to attack the resolution-chain mechanism (color_b is Roy_D-aligned
per memory `project_dor_drift_named_axes_2026-05-06`; luma_mean is brightness
which shifts with kernel choice).

The 2026-05-15 CPU probe verdict: 3% reduction in real `score_range`, within
noise — null on the targeted axis (RESULTS_OVERNIGHT §5).

The scorecard verdict: highest dev_fake_macro, highest visomaster_enhanced_macro_dev,
+10pp absolute viso lift vs T5C — a real and large effect on a different axis.

**Mechanism candidate**: adding 2 more adversarial axes to the GRL block
applies more pressure on the encoder to find a forgery axis that's
orthogonal to ALL the named IQ axes. With 4 axes covered (sharpness +
color_a + chronic_flag + is_dor), the encoder could "hide" forgery signal
along color_b or luma. With those covered too, the encoder is forced to
find a forgery axis that's robust to broader IQ variation — which happens
to correspond to features that visomaster_enhanced fakes carry. This is
speculative.

**Caveat**: I cannot distinguish this from "Slot β was trained for 4h 23m
while Slot α was 1h 47m, so Slot β saw more steps of the same data and
that's what produced the viso lift". Both ckpts are at step3500 in optimizer
steps so the wall-time difference is overhead (more axis heads = slower
backward); the optimizer trajectory should be comparable. But this is not
fully verified.

**Falsifier**: a Slot β variant with axes = {chronic_flag, is_dor,
sharpness_laplacian_high, color_a_approx_dev_high, color_b_dev_high} only
(5 axes — adding only color_b, not luma). If viso recall remains lifted,
the marginal contribution is in color_b. If it drops back to T5C levels,
the lift required both color_b AND luma in combination.

### 3.3 Slot β's lockbox_real_fpr penalty is plausibly identity-localized (LOW-MEDIUM)

Slot β step3500 lockbox_real_fpr = 0.088 (3.2× T5C). Prior memory
`project_chronic_offenders_partition_per_ckpt_2026-05-04` documents that
each ckpt's chronic-FP behavior partitions across identities — different
ckpts fail on different chronic identities.

**Untested**: a per-identity decomposition of Slot β step3500's lockbox FPR.
DEEP_DIVE §6 names this as the pending follow-up. If the +5.5pp lockbox_real_fpr
penalty is concentrated on 2-3 chronic identities, the per-identity rule
(memory `project_blend_unsharp_lever_2026-05-14`: `frac>0.6>0.4 AND
count>0.9≥1`) could potentially rescue. If it's distributed across new
identities, the rule won't help.

**Cost of test**: ~$0 CPU, ~1h. Highest-EV follow-up from this packet.

## 4. Three things the CPU probe should have measured but didn't

This subsection exists because the next agent should know what was missing
from the design of the 2026-05-15 probe. These are NOT criticisms of the
specific FACTS in that probe; they are gaps in the probe's CHARACTERIZATION
that, in retrospect, would have caught the recall risk.

### 4.1 Fake-vs-real AUC on the same panel

The probe measured `score_range`, which is mechanism-agnostic (DEEP_DIVE §7).
Stability via encoder-level invariance and stability via score-distribution
compression both produce low score_range. Fake-vs-real AUC distinguishes
them: invariance preserves AUC; compression reduces it.

**Action for next CPU probe author**: add an AUC measurement for each ckpt
on the panel. ~5 minutes of CPU; high information.

### 4.2 Fake-side score distribution

DEEP_DIVE §3 surfaces that Slot α compressed BOTH sides of the score
distribution. The probe reported per-cohort score_range but did not
explicitly report fake-side mean and spread by size. Had it done so,
the recall risk would have been visible: Slot α step3500 fake-side mean
sits in [0.65, 0.78] while T5C's spans a wider range and reaches higher
peaks.

**Action**: report per-cohort (real, fake) × per-size (5 sizes) score statistics
in the standard CPU probe template.

### 4.3 Mid-training canary trajectory

The yamls inherited the canary-disabled state from R13_T5C_T4_BIG_CLASSIFIER_2026-05-11.
DEEP_DIVE §8 details what the canary would have shown if enabled. The
canary infrastructure already exists; the cost of enabling it is one
yaml line.

**Action**: add a default-on canary block to the next packet template.

## 5. Implications for the open-loop program (load-bearing thread)

### 5.1 Resolution-chain stability ≠ deployment-grade lift (HIGH)

The thread `threads/iq_shortcut_deconvolution_program_2026-05-08.md` Stage 2a
(IQ GRL) and Stage 4 (drop smooth training reals) both presupposed that
attacking the IQ-shortcut axis at the data or loss layer would compose with
fake recall. Slot α tests this directly at the data layer for the size axis.
The result: stability achievable, fake recall not preserved.

This does NOT close the iq_shortcut thread — it sharpens the close criterion.
**Future iq_shortcut packets must demonstrate that their proposed lever
preserves fake-vs-real AUC on a panel comparable to the 2026-05-15 probe.**
This is the new pre-launch CPU gate for that thread.

### 5.2 The `lockbox_real_fpr` tiebreak is the persistent rank-1 mechanism (HIGH)

P8A holds rank-1 across 13+ packets because of the `lockbox_real_fpr`
tiebreak. Every R13 candidate that lifts dev_fake_macro_recall pays for
it with lockbox_real_fpr. Slot β is the latest instance.

Per D4 (2026-05-12 CPU diagnostic), `lockbox_real_fpr` at dev-cal τ is
actually LOWER than dev FPR for P8A/T5C/T3 (0.92% / 2.40% / 2.19% vs
the 5% dev calibration target). The contract is ranking on a metric that
is below the policy's own calibration target — i.e., the ranking is
within the noise band of the policy.

The thread `iq_shortcut_deconvolution_program_2026-05-08.md` Stage 3
(eval reframe) proposed adopting HDTF-conditional readouts as primary.
After 13+ packets of P8A holding rank 1 on a tiebreak that may not be
load-bearing, the case for actually executing Stage 3 strengthens.

**Concrete proposal**: re-rank the 5-ckpt scorecard above without the
`lockbox_real_fpr` tiebreak — instead, rank by `dev_fake_macro_recall` as
primary tiebreak (the metric this scorecard actually measures vs the
gate). Under that re-rank, Slot β step3500 ranks 1 (0.545 dev_macro),
T5C step3500 ranks 2 (0.459), Slot α step1500 ranks 3 (0.443), P8A ranks
4 (0.300, exactly at floor). This is not a proposal to change the policy
— it is a diagnostic showing how much of the rank ordering is the
tiebreak vs the gate.

## 6. Self-correction log

### 6.1 (RETRACTED) "Slot α is a clear win on resolution-chain stability"

In the 2026-05-16 04:24 UTC handback message (Slack-equivalent: my
response after the CPU follow-up probe completed), I described Slot α
as "a clear win on resolution-chain stability" and recommended a scorecard.
The CPU probe data supported the FACTS portion of that claim. The
INTERPRETATION ("clear win") embedded an assumption that the stability
lift would compose with contract metrics. The scorecard contradicts that
assumption.

**Retraction**: "Slot α is a clear win on resolution-chain stability" is
scoped to the CPU probe's specific metric (`score_range` on reals). On
the v3-fix contract, Slot α step3500 ranks 5 (fails recall floor) and
step1500 ranks 3 (regresses viso). The phrase "clear win" should not have
been used without qualifying the scope. I will use mechanism-specific
language (e.g., "Slot α achieves the targeted reduction in real-frame
score_range") rather than evaluative language for future probe verdicts.

### 6.2 (RETRACTED) "Slot β is a null result"

In the 2026-05-16 RESULTS_OVERNIGHT_FACTS doc §5, I labeled Slot β a
"null result" based on the resolution-chain probe finding (3% reduction
in score_range, within noise). This was scope-correct for THAT metric.
The scorecard surfaces that Slot β has the highest dev_fake_macro_recall
(0.545), highest visomaster_enhanced_macro_dev (0.235), highest
deeplive_enhanced_dev (0.738) of any scored ckpt.

**Retraction**: "Slot β is a null result" is scoped to the resolution-chain
metric only. On contract metrics, Slot β step3500 has the largest positive
deltas of any scored ckpt for three of the five dev fake suites. The label
"null result" should not have been used as a generalized verdict; future
probe writeups should explicitly scope null-result claims to the metric
measured.

### 6.3 (RETRACTED) "scorecard is the followup test of the mechanism"

In the 2026-05-16 handback, I framed the scorecard as a test of whether
the mechanism worked. The scorecard does not test the mechanism — it tests
contract metrics. The CPU probe tests the mechanism. These are different
questions. The CPU probe verdict on Slot α was "mechanism approximately
met"; the scorecard verdict was "contract not met". Both can be true;
they're testing different things.

**Retraction**: future packet retros should not conflate "mechanism worked"
with "contract met". They are independent verdicts. The DEEP_DIVE §7
mechanism-discrimination diagnostic (fake-vs-real AUC on the panel) is the
specific test that distinguishes them.

## 7. Recommended next-step (single proposal, with explicit caveats)

The highest-EV CPU follow-up is **per-identity decomposition of Slot β
step3500's lockbox_real_fpr** (DEEP_DIVE §6). Cost: $0, ~1h MPS.

**Decision criterion**:
- If lockbox FPR concentrates on 2-3 chronic identities already covered
  by your per-identity rule (memory `project_blend_unsharp_lever_2026-05-14`),
  then Slot β step3500 + the per-identity rule + G2(110) is potentially
  shippable. A subsequent integration probe ($0 CPU) verifies whether
  the rule actually rescues the new chronic-FP frames.
- If lockbox FPR distributes across new identities (not in the chronic-6
  pool), the rule doesn't help and Slot β is not a deployment path.

**Caveat**: this proposal is conditional on the contract policy's
`lockbox_real_fpr` tiebreak being load-bearing. Per §5.2, the tiebreak
may not be load-bearing — and if it isn't, Slot β step3500's raw recall
metrics (highest of any scored ckpt) make it deployment-grade independent
of the per-identity decomposition.

I am NOT proposing a packet-design lever yet. Two CPU probes determine
whether Slot β is a deployment candidate (per-identity decomposition) and
whether the lockbox_real_fpr tiebreak is load-bearing (re-eval per D4's
finding at the per-quartile cell level on the same 9-suite panel).

Both are ~$0. Total time: half a day MPS.
