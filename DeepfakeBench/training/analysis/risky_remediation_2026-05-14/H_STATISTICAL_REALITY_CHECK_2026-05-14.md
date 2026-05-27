# Statistical Reality Check on the Combined Rule (2026-05-14)

## TL;DR — The honest revised picture

After comprehensive search (320-rule grid, 1000-perm null, 5000-iter bootstrap, leave-one-pool-out CV), **the combined bulk+tail rule's +4.22pp improvement is NOT statistically robust**:

1. **Permutation test (multi-test corrected): p = 0.526** — the same Δ arises by chance in over half of label shuffles
2. **Only 1 of 320 grid rules achieves the 69/71 optimum** — knife-edge, not stable neighborhood
3. **Holdout CV: only teams_lockbox shows held-out gain (+14.3pp);** teams_dev and dor_cross holdouts show **zero** held-out gain

The bootstrap P(Δ>0)=0.954 was misleading because it ignores multiple testing across 320 candidate rules.

**Revised recommendation: stay with the simple baseline `frac > 0.49 > 0.5` OR `frac > 0.6 > 0.4`** for the per-identity rule. The combined bulk+tail rule's improvement is at the boundary of what could arise from rule-search noise on 71 identities. The FP reduction is real on THIS pool but cannot be claimed to generalize.

## Why this matters

The comprehensive search ran THREE independent statistical checks. The bootstrap looked promising (P>0.95). But the more rigorous tests — permutation and holdout — show the apparent signal evaporates when properly controlled for multiple testing.

This is a textbook case of **multiple-testing inflation**: when you scan 320 rules looking for the best, the best rule's measured Δ is inflated by the maximum-over-rules statistic. The permutation test corrects for this.

## The three statistical tests, side-by-side

| test | what it asks | result | interpretation |
|---|---|---|---|
| Bootstrap (rule fixed) | "does THIS rule beat baseline when we resample identities?" | P(Δ>0) = 0.954 | rule looks good *if* it was the only one tested |
| Pool-stratified bootstrap | same, but within-pool resampling | P(Δ>0) = 0.966 | similar — but assumes the rule was pre-specified |
| **Permutation (multi-test corrected)** | "**does our SEARCH PROCEDURE produce rules with Δ ≥ +0.042 under random labels?**" | **p = 0.526** | **same Δ arises in 53% of label shuffles — no signal beyond noise** |

The permutation test is the honest one because **we didn't pre-specify the rule**. We searched 320 rules and reported the best. The null hypothesis must therefore account for "best of 320".

## Holdout cross-validation results

Leave-one-pool-out: train on 2 pools, find the best rule, evaluate on the held-out pool.

| held-out pool | best train rule | train Δ | **test Δ on held-out pool** |
|---|---|---|---|
| teams_dev | f>0.6>0.4 AND c>0.9≥1 | +8.1pp | **0** (same as baseline) |
| **teams_lockbox** | f>0.4>0.5 AND c>0.9≥1 | +1.6pp | **+14.3pp** (1 rescue out of 7) |
| dor_cross | f>0.5>0.6 AND c>0.9≥1 | +4.9pp | **0** (same as baseline) |

Only the lockbox holdout shows generalization. The dev and dor_cross gains observed on the full-pool fit don't replicate on held-out data.

**Interpretation**: the count>0.9 extreme check might be a real signal **specifically for the lockbox cohort** (where chronic identities like bla_bla_chow__s1 have elevated but non-extreme score distributions). For dev and dor_cross, the apparent gain was overfitting.

## Rule neighborhood — the knife-edge problem

Of 320 grid rules, only **1 achieves the optimum 69/71**:
- `bulk f>0.6>0.4 AND tail c>0.9>=1`

Tiny perturbations drop the rate:
- `f>0.6>0.35 AND c>0.9>=1` → 68/71
- `f>0.6>0.45 AND c>0.9>=1` → likely 68/71 (different rescue pattern)
- 18 other rules also achieve 68/71 (5.9% of grid)
- 42 rules strictly beat baseline (13.1% of grid)

The optimum is NOT in a stable neighborhood. A neighboring rule with marginally different parameters loses 1 identity. This is consistent with the optimum being a noise spike.

For comparison, if the rule were "really" optimal in some structural sense, we'd expect a smooth optimum surrounded by similar-performing neighbors. Instead we have a sharp peak.

## What about the +14.3pp lockbox holdout gain?

This IS a real held-out finding. Lockbox has 7 identities; baseline gets 5/7 right; trained-on-others rule gets 6/7 right. The rescued identity is consistent with the "no extreme frames" mechanism (likely bla_bla_chow__s1).

But: 7 identities is a tiny sample. One identity = 14.3pp swing. Bootstrap CI on the lockbox-only Δ would be enormous.

A useful generalization would be: "the count>0.9 extreme check helps when applied to chronic-identity FP cohorts." We'd want to validate this on a fresh held-out set of chronic-FP cohorts to confirm.

## What changed in my opinion

Earlier I leaned toward deploying the combined rule. After permutation + holdout, I'm now more skeptical:

- **Pre-revision recommendation (overweighted bootstrap)**: "deploy the combined rule, P(Δ>0)=95.4%"
- **Revised recommendation (honest)**: "the combined rule's apparent improvement does not survive multi-test correction; only the lockbox-specific count check has held-out evidence"

## Three honest options

| option | rule | identity-correct | FPs | held-out evidence |
|---|---|---|---|---|
| A: Status-quo baseline | `frac > 0.49 > 0.5` | 92.96% | 4 | n/a |
| B: Simple stricter | `frac > 0.6 > 0.4` | 95.77% | 2 | partial (CIs overlap) |
| C: Combined bulk+tail | `frac > 0.6 > 0.4 AND count > 0.9 ≥ 1` | 97.18% | 1 | **fails permutation test** |

### Recommendation

**Pick A or B.** They have similar statistical support (none has strong evidence over the other). B has a smaller FP count on our 71-identity pool. C looks better but is statistically equivocal once we control for the 320-rule search space.

If you want simplicity: A. If you want the FP reduction: B. 

**Don't deploy C as a final spec without fresh-cohort validation** — its apparent advantage is at the boundary of rule-search noise. The lockbox-specific gain is the only piece with held-out support, and that's just 1 identity rescue.

## Caveats

1. Permutation test uses 1000 shuffles — finite resolution on p-value (about ±0.015).
2. With more identities (say 200+), the permutation test would have more power and could either confirm or definitively refute the signal.
3. The lockbox cohort has only 7 identities — too few for any rule-fitting to be statistically reliable on its own.
4. We've effectively pooled three cohorts (teams_dev, teams_lockbox, dor_cross) with different sample sizes. A more sophisticated analysis would use mixed-effects models accounting for cohort heterogeneity.

## What I'd do to get stronger evidence

To upgrade from "directional finding" to "statistically supported":
1. **More identities** — score more cohorts (e.g., live_prod with new identity-level grouping). Need ~150-200 identities for permutation test to have power against this Δ.
2. **Targeted held-out cohort** — collect a fresh cohort of chronic-identity FPs, see if count>0.9 extreme rule helps them specifically.
3. **Identity-level mixed-effects** — model variance properly across cohorts. May reveal lockbox-specific effect with proper inference.

## Summary

The "combined bulk+tail" rule is **likely** an improvement, but the statistical evidence is **inconclusive** once we control for multiple testing. The honest read: stay with baseline or simple `frac > 0.6 > 0.4` until we have a fresh-cohort validation. The 320-rule search found a peak in the noise; permutation test confirms it's at chance level.

## Artifacts

- `comprehensive_rule_search.py` — main script (4D grid + permutation + bootstrap + holdout)
- `outputs/comprehensive_rule_grid.csv` — all 320 rule × pool evaluations
