# AGENT_PROPOSAL_2026-05-16 — Slot β + rule integration verdict

> **OPINIONS.** Same agent who designed both slots, ran the scorecard, and
> just authored RESULTS_FACTS_2026-05-16.md. Interpretation, not facts.

## §1. Bottom line

Slot β step3500 is a **near-shippable candidate** in combination with a small
adjustment to the per-identity Option-3 rule's threshold. The CPU work just
completed shows three concrete things:

1. **Slot β's lockbox over-fires are 100% chronic-6**, entirely on `dor_shkedi`
   (94%) and `bla_bla_chow` (6%). No new chronic identities surfaced. The
   substrate failure mode is the same one we already understand.
2. **Under any non-lockbox-FPR tiebreak, Slot β ranks 1 in the scorecard**.
   P8A is at the dev_fake_macro_recall floor (0.300 — barely passing). The
   13+-packet "P8A wins" streak is, in this run, a 0.07pp tiebreak margin on
   a 2-identity-dominated 5-identity lockbox.
3. **Trivial rule-threshold adjustments rescue Slot β's lockbox FPR to 0%**.
   `count_above_0.9 ≥ 1` → `count_above_0.92 ≥ 1` flips the verdict because
   Slot β's `dor_shkedi` max_score is 0.909. The rule sits right at its
   decision boundary on Slot β.

Combining these: Slot β + tightened rule + alt-tiebreak policy potentially
gives us the first promotion-eligible candidate that delivers the visomaster
recall lift we've been failing to break for 3 packet sequences (0.235 vs P8A
0.136, +73% relative).

## §2. The case for shipping Slot β

| metric | P8A (rank 1 by orig) | Slot β (rank 4 by orig) | delta |
|---|---:|---:|---:|
| dev_fake_macro_recall | 0.300 | 0.545 | **+0.245** (+82%) |
| visomaster_enhanced_macro_dev | 0.136 | 0.235 | **+0.099** (+73%) |
| deeplive_enhanced_dev | 0.239 | 0.738 | **+0.499** (+209%) |
| teams_fake_all_dev | 0.526 | 0.664 | **+0.138** (+26%) |
| teams_fake_all_lockbox | 0.387 | 0.541 | **+0.154** (+40%) |
| lockbox_real_fpr (raw) | 0.018 | 0.088 | +0.070 (worse) |
| lockbox_real_fpr (after rule variant A) | 0.013 | 0.000 | -0.013 (better) |
| dev_primary_real_fpr | 0.069 | 0.064 | -0.005 (better) |

After applying rule variant A (`count_above_0.92 ≥ 1`), Slot β's lockbox FPR
is 0% versus P8A's 1.3% (with current rule, since `Chikara_Takahashi` and
`PC_Generator` are not rescuable under any rule we tested). On every other
production-relevant metric (fake recall on each of viso/deeplive/teams), Slot β
beats P8A by margins not seen in 13+ packets.

## §3. The case against — and why I think it's weaker than it looks

### Caveat A: the lockbox is only 5 identities

This cuts both ways. The 5-identity lockbox is the actual eval substrate we
score against, but it means any verdict from this CPU probe is on a small
sample. The "Slot β over-fires are chronic-6-localized" finding is only as
robust as the assumption that production deployment will look like the
lockbox composition. That's a reasonable assumption given that's literally
what the lockbox is for, but it's worth stating.

### Caveat B: tightening the rule could break the 71-pool validation

The Option-3 rule was tuned/validated on a 71-pool benchmark
(`project_blend_unsharp_lever_2026-05-14`, 69/71 = 97.2%). The rule-threshold
adjustments I tested (variant A: `count_0.92 ≥ 1`) rescue Slot β on the
5-identity lockbox without changing P8A or T5C verdicts on the same 5
identities — but I have NOT re-validated those variants on the 71-pool. It is
possible variant A introduces 1-2 errors on the 71-pool, dropping rule
accuracy by 2-3pp. That's the diagnostic that should run before any deployment
decision.

### Caveat C: changing the tiebreak is a structural policy decision

Re-ranking with `dev_fake_macro_recall` (descending) flips P8A to rank 4. This
is not a small change. It says "we'd rather catch more fakes than over-fire
less on real identities the model can already identity-rescue". Whether
that's the right product call depends on the user-facing cost of FP vs FN —
a judgment I shouldn't make autonomously. But the FACTS say the existing
policy's `lockbox_real_fpr`-ascending tiebreak is the only thing keeping P8A
at rank 1 in this scorecard.

### Caveat D: Slot β at step3500 is one ckpt

We have no Slot β step1500 or step2500 readout. The full picture of how this
lever behaves over training is unknown. A short replication run is cheap
($~3-5 of GPU, since it's effectively rerunning the same 4h 23m we already
paid for and just keeping more periodic ckpts).

## §4. Recommended path forward, ranked

These are options for the user. I am not autonomously authorized to push any
of them.

**Option α: Validate rule variants on the 71-pool benchmark, then ship.**
~$0, ~2h CPU. Re-score the 71-pool with rule variant A (`count_0.92 ≥ 1`).
If it preserves 69/71 accuracy or better, ship Slot β step3500 + variant A
+ G1+G2(110) gate. This is the minimum-risk path that converts CPU work into
a shippable model.

**Option β: Run the alt-tiebreak as the new policy and re-evaluate the last
3 packets' scorecards.** ~$0, ~30 min. The 2026-05-16 tiebreak finding may
extend backwards — earlier packets we declared "rank 2+" may also rank 1
under the dev_fake_macro_recall tiebreak. Worth knowing if the rank-1 we've
been chasing was the wrong metric.

**Option γ: Short Slot β replication with periodic snapshots.** ~$5-8 GPU,
~5h. Same lever (multi_axis_grl 6-axis), seed-varied (9913), keep periodic
ckpts at 500/1000/1500/2000/2500/3000/3500. Confirms step3500 isn't a
seed-specific outlier and gives us trajectory information for future packet
design.

**Option δ: Sister-variant — 5-axis GRL (add only color_b_dev_high, not
luma_mean_high).** ~$5-8 GPU. Tests which of the 2 added axes carries the
viso lift. If color_b is the lifter, future packets can drop luma and stay
clear of the resolution-chain axis the user wants to address. This is the
"GPU packet" path; I do not recommend it before Options α/β/γ.

## §5. What I would NOT do

- Run another resolution_chain_aug variant. Slot α refuted the lever as a
  single-lever attack on the contract (failed dev_fake_macro_recall floor).
- Run another GRL-axis-extension blind. The Slot β finding tells us 6-axis
  has a viso lift but at a lockbox-FPR cost. Without sister-variant data,
  any new axis-extension is a different bet, not a refinement.
- Touch P8A. The score-compression mechanism shown by Slot α suggests any
  FT-from-P8A risks the same outcome.

## §6. Self-correction log

This document is the third one I've written this session. To maintain the
discipline established in the RESCHAIN_GRL6 OPINIONS doc (§6), explicit
retractions of framings I've made earlier in THIS conversation:

- **Earlier framing**: "Slot β is closed if over-fires distribute across new
  identities." → Retain. Over-fires ARE chronic-6-localized, so Slot β is NOT
  closed by that test.
- **Earlier framing**: "G2(110) gate + per-identity rule rescues Slot β."
  → AMENDED. The default per-identity rule does NOT rescue dor_shkedi on
  Slot β (rule says FAKE due to frac=0.505 just above 0.4 threshold). A
  trivial rule-threshold adjustment does rescue. The G2(110) gate has NOT
  been tested on Slot β scores — that's a separate diagnostic.
- **Earlier framing**: "Step 1 says identity-localized then Step 3 is a
  routine rule integration." → Retain Step 1 verdict; Step 3 required
  unpacking the rule mechanics more carefully than I expected.

## §7. Open question for the user

The single biggest unresolved question is whether the
`lockbox_real_fpr`-ascending tiebreak in the v3-fix policy represents the
intended product behavior. P8A wins by 0.07pp on a tiebreak metric that
trades against dev fake recall. If the answer is "yes the tiebreak is
intentional and we accept being conservative on real-side over-firing",
then Slot β is currently rank 4 and the path is rule-tightening + re-rank.
If the answer is "the tiebreak was chosen when we didn't have a candidate
with this much viso/deeplive recall headroom", then we may want to revise
the policy. This is a Roee judgment call I should not make autonomously
(`feedback_decision_points`).
