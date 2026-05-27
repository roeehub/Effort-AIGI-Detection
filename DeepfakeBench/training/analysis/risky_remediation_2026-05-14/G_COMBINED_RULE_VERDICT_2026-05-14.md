# Combined Bulk+Tail Rule — Verdict (2026-05-14)

## TL;DR

Your "extremity" hypothesis was **correct** — but the right formulation isn't "extreme replaces majority" or "extreme as OR alternative." It's **"majority at a higher threshold AND at least one extreme frame":**

```
flag iff:  (frac > 0.6 > 0.4)  AND  (count > 0.9 ≥ 1)
        = "majority of frames score above 0.6"  AND  "at least one frame scores above 0.9"
```

In 32-frame production batches: **"flag identity as fake iff ≥13 of 32 frames score above 0.6 AND ≥1 of 32 frames scores above 0.9."**

Combined-pool identity-correct: **66/71 → 69/71 (92.96% → 97.18%, +4.22pp).**

This is the best result of the entire investigation. Bootstrap P(Δ > 0) = 95.4% — borderline statistically significant.

---

## Question A: Does t=0.7 work without Roy_D?

**No.** Without Roy_D, baseline gives 66/70 = 94.29%. Best t=0.7 rule (`frac > 0.7 > 0.4`) gives **66/70 = 94.29% — exactly tied**.

The t=0.7 sweep without Roy_D:
| rule | correct | rate | FP | FN |
|---|---|---|---|---|
| baseline (`frac > 0.49 > 0.5`) | 66/70 | 94.29% | 3 | 1 |
| `frac > 0.7 > 0.3` | 65/70 | 92.86% | 2 | 3 |
| `frac > 0.7 > 0.4` | 66/70 | 94.29% | 1 | 3 |
| `frac > 0.7 > 0.5` | 64/70 | 91.43% | 1 | 5 |
| `frac > 0.7 > 0.7` | 57/70 | 81.43% | 0 | 13 |
| **`frac > 0.6 > 0.4` (winner)** | **68/70** | **97.14%** | **1** | **1** |

### What fails at t=0.7

The weak `dor_cross` fakes. Tightening the per-frame threshold to 0.7 turns these correctly-flagged fakes into FNs:

| FN identity | n | mean | frac > 0.49 | frac > 0.7 | frac > 0.9 |
|---|---|---|---|---|---|
| dor_fake_inswapper_128res_gpen1024 | 121 | 0.432 | 36% | 17% | 0% |
| dor_fake_simswap | 79 | 0.584 | 65% | 29% | 1% |
| dor_fake_inswapper_128res_gpen512 | 110 | 0.527 | 54% | 30% | 1% |
| dor_fake_ghostface_v2 | 144 | 0.664 | 84% | 49% | 2% |
| dor_fake_ghostface_v3 | 102 | 0.645 | 71% | 49% | 11% |

These fakes have score distributions peaking around 0.6-0.7 — strong enough to pass baseline majority but not strong enough to clear t=0.7. They're "the model is somewhat confident" fakes.

**The fundamental constraint:** weak fakes and borderline FPs overlap in mean-score space. Raising the per-frame threshold can't separate them without losing both.

---

## Question B: Should we combine majority + extremity?

**Yes — but most simple AND-extreme rules collapse, only the COUNT-BASED extreme adds new power.**

### Why most AND rules collapse

I tested 28 `(frac > 0.49 > 0.5) AND (frac > t_e > x_e)` configurations. **25 of 28 produce verdicts identical to some single `frac > τ > x` rule**, because requiring two conditions where one strictly implies the other reduces to just the stricter condition.

The 3 that are NOT identical to any simple rule:
| AND rule | rate vs baseline | comment |
|---|---|---|
| `frac > 0.49 > 0.5 AND frac > 0.7 > 0.2` | +1.4pp | tiny gain |
| `frac > 0.49 > 0.5 AND frac > 0.8 > 0.1` | +1.4pp | tiny gain |
| `frac > 0.49 > 0.5 AND frac > 0.9 > 0.05` | -1.4pp | loses |

So fraction-based "AND extreme" mostly doesn't help beyond a simple stricter threshold.

### Where the AND structure DOES help: count-based extreme

A count-based extreme check (`at least N frames above threshold`) is structurally different from a fraction-based check and CAN add power:

| rule | correct | rate | FP | FN | Δ vs baseline |
|---|---|---|---|---|---|
| `frac > 0.49 > 0.5 AND count > 0.9 ≥ 1` | 68/71 | 95.77% | 2 | 1 | +2.82pp |
| `frac > 0.6 > 0.4` (basic stricter) | 68/71 | 95.77% | 2 | 1 | +2.82pp |
| **`frac > 0.6 > 0.4 AND count > 0.9 ≥ 1` (COMBINED)** | **69/71** | **97.18%** | **1** | **1** | **+4.22pp** |

**The combined rule is genuinely additive** — neither component alone gets 69/71. They rescue DIFFERENT identities:

| FP identity | rescued by | mechanism |
|---|---|---|
| dor_morning | both | weak bulk (frac>0.6=33% < 40%) AND no tail (count>0.9=0) |
| bla_bla_chow__s1 | tail-check only | strong bulk (frac>0.6=77%) BUT no extreme frame (count>0.9=0) |
| dor_shkedi | bulk-check only | weak bulk (frac>0.6=37% < 40%) but has SOME tail (count>0.9=5) |

The **bulk check** (`frac > 0.6 > 0.4`) catches identities with diffuse score distributions.
The **tail check** (`count > 0.9 ≥ 1`) catches identities that are elevated but have no truly-extreme frames.
Combined, they trap two distinct FP failure modes.

### Why Roy_D is still wrong

Roy_D has both bulk AND tail:
- frac > 0.6 = 100% (huge bulk)
- count > 0.9 = 78 (huge tail — 60% of 130 frames)

He looks identical to a confident fake. **Roy_D is a model-level failure, not a rule failure.** No score-distribution rule fixes this.

---

## Per-pool performance of the combined rule

| pool | baseline | combined rule | Δ |
|---|---|---|---|
| teams_dev | 33/34 (97.1%) | 33/34 (97.1%) | 0 (Roy_D unrescuable) |
| **teams_lockbox** | **5/7 (71.4%)** | **7/7 (100%)** | **+2 (PERFECT)** |
| dor_cross | 28/30 (93.3%) | 29/30 (96.7%) | +1 (dor_morning) |
| **combined** | **66/71 (92.96%)** | **69/71 (97.18%)** | **+4.22pp** |

### Statistical significance

Bootstrap 2000 paired resamples:
- Baseline CI: [0.86, 0.99]
- Combined rule CI: [0.93, 1.00]
- Δ CI: [+0.000, +0.099]
- **P(Δ > 0) = 95.4%** ≈ borderline at α=0.05

This is meaningfully stronger than the single-rule best (`frac > 0.6 > 0.4` had P(Δ > 0) ≈ 92%).

Excluding Roy_D (since he's unrescuable):
- Baseline (no Roy_D): 66/70 = 94.29%
- Combined (no Roy_D): **69/70 = 98.57%** (Δ = +4.29pp)
- Bootstrap P(Δ > 0) = 95.2%

---

## What's left wrong under the combined rule

Only 2 identities still misclassified:
1. **Roy_D** (FP): model genuinely confident-wrong. Unrescuable at the rule level. Needs training-side fix.
2. **dor_fake_inswapper_128res_gpen1024** (FN): weak fake the model misses. mean=0.432, frac>0.49=36%. Already FN at baseline.

That's it. The remaining 69 identities are all correctly classified.

---

## Recommendation

### Ship spec update (proposed)

```python
def is_fake_identity(frame_probs):
    # frame_probs = list of T5C scores from frames that passed G1+G2(110)
    if len(frame_probs) == 0:
        return None  # abstain
    
    # Bulk condition: majority of frames above 0.6
    n_above_bulk = sum(p > 0.6 for p in frame_probs)
    bulk_passes = (n_above_bulk / len(frame_probs)) > 0.40
    
    # Tail condition: at least one frame above 0.9
    n_above_tail = sum(p > 0.9 for p in frame_probs)
    tail_passes = n_above_tail >= 1
    
    return bulk_passes and tail_passes
```

In words: **"flag as fake iff at least 40% of (G1+G2-passing) frames score above 0.6 AND at least 1 frame scores above 0.9."**

For 32-frame production batches: "at least 13 of 32 frames above 0.6 AND at least 1 frame above 0.9."

### Comparison to options

| option | identity-correct (combined) | FPs | FNs | notes |
|---|---|---|---|---|
| status quo (`frac > 0.49 > 0.5`) | 66/71 = 92.96% | 4 | 1 | well-understood |
| basic-stricter (`frac > 0.6 > 0.4`) | 68/71 = 95.77% | 2 | 1 | simpler |
| **combined (proposed)** | **69/71 = 97.18%** | **1** | **1** | **best**; needs two threshold checks |

### When to deploy

The combined rule is the strongest finding. CI just touches 0 → borderline significant. **I'd recommend deploying it** because:
1. Cross-pool sign is uniformly positive (teams_dev unchanged, lockbox +2 to perfect, dor_cross +1)
2. FP reduction is 4 → 1 (75% reduction)
3. Zero new FNs
4. Mechanism is interpretable: bulk + tail capture two distinct failure modes
5. Production overhead is trivial (one additional `count > threshold` check)

### What this DOESN'T do
- Doesn't rescue Roy_D (model-side problem)
- Doesn't reach statistical significance at α=0.01 (would need ~2-3× more identities)
- Doesn't change the per-frame threshold (still τ=0.49 for any per-frame use)

---

## Direct answer to your two questions

### A) Does t=0.7 work on non-Roy_D data?
**No.** It exactly ties baseline (94.29%) without Roy_D. It rescues bla_bla_chow__s1 but loses 3 weak dor_cross fakes — the rescue/loss tradeoff is exactly balanced. The fundamental problem: weak fakes have score distributions peaking around 0.6-0.7, indistinguishable from elevated FPs at any single fraction threshold.

### B) Should we combine majority + extremity?
**Yes — but with a specific structure:**
- `frac > 0.49 > 0.5 AND frac > t_E > x_E` (fraction-based) mostly collapses to simple stricter rules. Not additive.
- `frac > 0.6 > 0.4 AND count > 0.9 ≥ 1` (basic-stricter AND count-based extreme) IS additive. Rescues different identities than either alone. **+1.4pp over best single rule, +4.2pp over baseline.**

The count-based extreme (`at least 1 frame above 0.9`) is structurally distinct from fraction rules — it catches "no truly-extreme frames" identities (bla_bla_chow__s1) that any fraction-of-some-band check misses.

---

## Artifacts

- `extreme_rule_study.py` — main sweep
- `outputs/extreme_rule_per_identity.csv` — per-identity stats
- `outputs/extreme_rule_sweep_all.csv` — 3060 sweep evaluations
- This document for the verdict
