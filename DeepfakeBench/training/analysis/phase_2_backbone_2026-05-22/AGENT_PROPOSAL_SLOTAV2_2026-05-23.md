# Phase 2 BACKBONE-SlotAv2 — agent proposal (opinion doc; numbers TBD)

Status: scoring in flight for step 3500 (PID 88774, started 2026-05-22T22:55Z).
This doc has the recommendation framework; numbers fill in post-scoring.

FACTS in `RESULTS_FACTS_SLOTAV2_2026-05-23.md`.

---

## Important context — root-cause adjustment

**Image `1.3.299` was missing the substrate-paired inventory CSV** (see
`STATE.md` 2026-05-23 00:00 UTC subsection). So this SlotAv2 full ran with:
- `SubstratePairStamper` DISABLED at runtime (CSV not found)
- `substrate_paired_inventory` loader emitting 0 samples
- GroupDRO `group_id_mapping` built from EXISTING data lanes only (R-D /
  F-B / source-based), NOT from the additional 1,826 HDTF+QCLIP
  substrate-paired rows

The intended lever was GroupDRO substrate-balanced **with** the 1,880-row
inventory. The actual lever was GroupDRO on the existing data lanes only.
**This is a partial test of the intended design.**

### Implication for the deploy/iterate/abort decision

- If this run DEPLOYS — the lever even with the bug worked. Substrate-pair
  data may have been overkill. Keep the run as deployment candidate.
- If this run is BORDERLINE (within 0.05 of any gate) — the inventory fix
  could plausibly tip it. Consider rerun on image `1.3.300`. Cost ~$50.
- If this run is CLEARLY DEGRADED vs Slot A v2 step3500 (composite up by
  > 0.10, or lockbox_real_fpr > 0.035) — the GroupDRO mechanism doesn't
  work on this base, OR the inventory data is necessary. A rerun is
  probably wasted; investigate other levers.

---

## Decision matrix (from yaml comment header)

| Outcome | Criterion |
|---|---|
| Deploy | composite λ=1.0 ≤ 0.25 AND lockbox_real_fpr ≤ 0.018 AND lockbox_fake_recall ≥ 0.70 AND viso recall ≥ 0.15 AND dev_fake_macro_recall ≥ 0.40 |
| Iterate | Lockbox kept but viso < 0.15; rerun with inventory fix |
| Abort | composite > Slot A v2 step3500 baseline (0.331) by > 0.05 |

---

## Step 3500 scorecard

| Metric | Base | face-pool inf | HEAD step 250 | **SlotAv2 step 3500** |
|---|---:|---:|---:|---:|
| composite λ=1.0 | 0.331 | 0.249 | 0.235 | **0.502** |
| lockbox_real_fpr | 0.0191 | 0.0154 | 0.0176 | **0.0118** |
| lockbox_fake_recall | 0.6877 | 0.7668 | 0.7826 | **0.510** |
| viso_enhanced_macro_dev recall | 0.1673 | 0.0673 | 0.0727 | **0.005** |
| deeplive_enhanced_dev recall | 0.420 | 0.552 | 0.7505 | **0.534** |
| dev_fake_macro_recall | 0.310 | 0.439 | 0.4921 | **0.353** |
| selected τ | — | — | 0.5587 | **0.8265** |

---

## Recommendation: **ABORT**

The composite λ=1.0 = 0.502 exceeds the yaml's literal abort threshold
(base 0.331 + 0.05 = 0.381) by 0.121. Four of the five deploy gates fail.
Only lockbox_real_fpr improves (0.012 vs 0.018 gate); the cost is severe
recall collapse (lockbox_fake -18pp, viso -99%, deeplive -22pp from HEAD,
dev_fake_macro -14pp).

The mechanism is doing what GroupDRO worst-group reweighting is designed
to do: pull the boundary toward the under-served real group (chronic
identities like dor / bla_bla_chow), which forces a high τ. But the
calibration band shifts entirely above where viso fakes score on Slot A v2,
collapsing viso recall to 0.5% (3 of 550 fakes).

### Argument for deploy

None. Composite triggers the yaml's abort criterion. Three of four fake
recall gates fail catastrophically.

### Argument for iterate (rerun with inventory fix on 1.3.300)

Considered + rejected. The inventory CSV bug meant the substrate-paired
data lanes didn't deliver the additional 1,826 HDTF+QCLIP rows. With the
fix, GroupDRO would have a richer group structure — but the SAME mechanism
(worst-group reweighting on a chronic-identity-skewed substrate-balanced
key) would still push τ up. Adding more matched-pair data would intensify
the conservative-boundary effect, not reverse it. The viso collapse to
0.5% is structural (the calibration band shifted above viso's score
distribution), not a function of how much substrate-paired data the loss
sees. Saving the $50 rerun.

If a future packet wants to test "GroupDRO + inventory data", a more
useful single-lever design would be a SMALLER `beta` (e.g., 0.05) or a
LARGER `clip_max` (e.g., 0.5) — reducing the worst-group amplification to
keep the boundary closer to base. That's a different yaml, not this one.

### Argument for abort (chosen)

The lever produced the predicted-direction effect (FPR ↓ at cost of recall
↓) but blew past the abort threshold. The yaml's literal abort criterion
fires. Per Phase 3 outcome matrix, SlotAv2 column = ABORT.

---

## Comparison to other Phase 2 arms

| Arm | composite λ=1.0 | lockbox_real_fpr | lockbox_fake_recall | viso recall | rank |
|---|---:|---:|---:|---:|---:|
| Slot A v2 step3500 + CLS-pool (base) | 0.331 | 0.0191 | 0.6877 | 0.1673 | 3 |
| Slot A v2 step3500 + face-pool inference ($0) | 0.249 | 0.0154 | 0.7668 | 0.0673 | 1 (deploy candidate) |
| HEAD step 250 (winner) | 0.235 | 0.0176 | 0.7826 | 0.0727 | 2 (iterate) |
| BACKBONE-SlotAv2 step 3500 | **0.502** | 0.0118 | 0.510 | 0.005 | **4 (abort)** |
| BACKBONE-T5C step 3500 (rerun pending image 1.3.300) | _TBD_ | | | | |

At time of writing, the $0 face-pool inference baseline holds rank 1. HEAD
step 250 (rank 2, ITERATE → Phase 4 HEAD ALT) Pareto-equivalent. SlotAv2
step 3500 (rank 4) is ABORT. If T5C also aborts/iterates, the deployment
recommendation falls back to the face-pool inference baseline ($0, no
training cost beyond the existing Slot A v2 step3500 ckpt).
