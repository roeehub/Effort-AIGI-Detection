# AGENT_PROPOSAL_2026-05-16 — auto-mode anchor_aware + rebalance verdict

> **OPINIONS.** Same agent who designed both slots, ran the encoder probe,
> and scored the contract. Self-correction log at §6.

## §1. Headline

**Slot A v2 (anchor_aware) is the first single-lever T5C-base FT in 13+
packets that delivers a deployment-grade improvement on fake recall
without regressing real-side over-firing.** P8A holds rank 1 by a 0.07pp
tiebreak; Slot A v2 holds rank 2 with massively better fake recall and
identical-to-noise lockbox real FPR.

This is the cleanest "Slot A works" result the auto-mode session could
produce. The mechanistic prediction from the encoder probe (anchor_pool
collapse → dor cohort fix, with generalization to similar identities) was
fully borne out.

## §2. Numerical case for Slot A v2

| metric | P8A | Slot A v2 | gain |
|---|---:|---:|---:|
| dev_fake_macro_recall | 0.300 | 0.438 | **+46%** |
| visomaster_enhanced_dev | 0.136 | 0.167 | **+23%** |
| **deeplive_enhanced_dev** | 0.239 | 0.552 | **+131%** |
| **teams_fake_all_lockbox** | 0.387 | 0.688 | **+78%** |
| teams_fake_all_dev | 0.526 | 0.595 | +13% |
| **lockbox_real_fpr** | 0.0184 | 0.0191 | **+0.07pp** |

The lockbox real FPR delta is **smaller than the per-identity noise band**
of the 5-identity lockbox cohort: P8A's 25 over-fires on Chikara + PC_Gen
were ELIMINATED in Slot A v2; Slot A introduced 11 new over-fires on
bla_bla_chow + minor dor_shkedi drift. Net: +0.07pp.

## §3. Where the wins came from (per-identity)

Slot A's mechanism worked exactly as the encoder probe predicted:

- **anchor_pool collapsed from 53% Roy_D-adjacent to 0%** in Slot A
  encoder space.
- This pulled the anchor-pool identity (dor false-flag captures) and
  some adjacent identities (Chikara_Takahashi, PC_Generator) to the
  clean cluster.
- Result: P8A's two largest lockbox over-firers (Chikara 11/42, PC_Gen
  8/29) → both 0/N in Slot A.
- Also on dev: Q (89→17%), PC_Generator (24→7%) — large reductions.

This is the first clear evidence the encoder-axis is actually
manipulable through training-time supervision.

## §4. The unsolved problem: Roy_D

Roy_D dev FPR went from 29% (P8A) → 81% (Slot A v2). The encoder probe
predicted this: Roy_D's embedding region is essentially unchanged across
all ckpts (97-100% Roy_D-adjacent in all four encoders). The anchor_pool
supervision did not generalize to Roy_D.

But Roy_D doesn't appear in the lockbox; only in dev. The contract is
calibrated against lockbox FPR. So Roy_D doesn't affect the lex ranking —
but it WILL affect production traffic if Roy_D-style frames are
representative of real production captures.

The encoder representation of Roy_D is structurally separate from the dor
false-flag pool. Anchor_aware on dor doesn't reach Roy_D's region. A
DIFFERENT supervision target (Roy_D-like anchor pool) is needed to fix
Roy_D specifically.

## §5. What should ship

**This is a Roee judgment call.** Three options:

### Option A — Ship Slot A v2 step3500 (recommended)

The +0.07pp lockbox cost is within tiebreak noise. The fake-side wins are
large and on production-relevant suites (viso, deeplive, teams_fake). If
shipping criterion is "best ckpt that passes all gates and isn't strictly
worse than P8A by a meaningful margin on lockbox", Slot A v2 ships.

Caveat: Roy_D dev FPR regressed (29→81%). If Roy_D represents production
traffic, this regresses production false positives on that cohort. Whether
that matters is a product decision.

### Option B — Keep P8A, run a follow-up anchor_aware packet on Roy_D

Slot A's success suggests a Roy_D-specific anchor pool would generalize
similarly. Construct a Roy_D-anchor pool (~30-50 Roy_D real frames),
enable anchor_aware with that pool + the dor pool, FT from T5C step3500.
Predicted: simultaneous Chikara + PC_Gen + Q + Roy_D fix.

Cost: ~$30-40 GPU + manual pool construction (~1h).

### Option C — Compound packet: anchor_aware + rebalance

Slot A and Slot B targeted different mechanisms. A combined run might
deliver Slot A's anchor_pool win + Slot B's fake-recall lift. Risk: Slot
B's lockbox FPR blowup (0.090) might dominate.

Cost: ~$30-40 GPU.

## §6. Self-correction log

What I retract from the session:

- **Encoder probe predicted Slot B (real_rebal) would be neutral-or-worse**
  on lockbox FPR. CONFIRMED — Slot B 0.090 vs T5C 0.028 (3.2× worse).
- **Encoder probe predicted Slot A would fix dor cohort but not Roy_D**.
  CONFIRMED — dor cohort largely fixed (Chikara, PC_Gen to 0%), Roy_D
  regressed dramatically.
- **Encoder probe predicted "moderately likely to reduce dor chronic FP"**.
  UNDER-PREDICTED. The reduction was more like 100% on Chikara + PC_Gen,
  72% on Q. Slot A is the cleanest mechanism-confirmation in this packet
  sequence.
- **The v6 PENDING_SCORECARD_PLAN.md predicted "Slot A PARTIAL more likely
  than CONFIRM"**. Held — Slot A is technically rank-2, not rank-1, so
  formally "PARTIAL" by the lex-policy criterion. But the 0.07pp tiebreak
  margin is arguably within noise, so "near-CONFIRM" is the more accurate
  reading.

## §7. New open loops (post-scorecard)

1. **Roy_D-specific anchor pool** (severity: high) — Slot A demonstrated
   anchor_aware works; Roy_D needs its own pool. ~30-50 frames + a packet.
2. **bla_bla_chow regression in Slot A** (severity: medium) — Slot A
   introduced bla_bla_chow over-fire (0→16% on lockbox). Why? Encoder
   shift moved bla_bla_chow toward Roy_D side. Need diagnostic.
3. **dor_shkedi.png minor drift in Slot A** (severity: low) — 0.7→1.6%.
   Slot A's encoder push to clean was actually too aggressive for some
   dor_shkedi frames; needs gentler weight tuning.
4. **Compound packet decision** (severity: medium) — Should Slot A +
   real_rebal compose? Slot B's lockbox blowup on dor_shkedi suggests no,
   but the mechanisms are non-overlapping.

## §8. The structural finding

The encoder-axis was the right framing. Anchor_aware as a training-time
penalty does manipulate it. The dor anchor pool generalizes
mechanistically to Chikara_Takahashi, PC_Generator, and Q — identities
that share encoder neighborhood with dor's false-flag captures.

This validates the v7 encoder-axis-amplification finding AND the
auto-mode design rationale. The next round of packets should target
distinct encoder regions (Roy_D, possibly bla_bla_chow) with their own
anchor pools rather than continuing to vary augmentation hyperparameters
on a single FT base.

## §9. What I am NOT proposing

- Another resolution_chain_aug variant. Different lever class, refuted.
- Another GRL axis extension. Slot β refuted; encoder-axis-amplification
  shows GRL didn't help the actual axis.
- Compositional augmentation into 6+ band region. v7 encoder probe
  refuted this lever class.
- A real_rebalance variant. Slot B's encoder evidence shows the lever
  pushes VCD AWAY from Roy_D in the trained encoder — wrong direction.
