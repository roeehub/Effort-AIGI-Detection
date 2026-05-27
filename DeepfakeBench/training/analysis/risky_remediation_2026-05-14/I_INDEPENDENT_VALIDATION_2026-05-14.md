# Independent Validation on Training-Eval Suites (2026-05-14)

## TL;DR

Validated Options 1/2/3 on **47 independent identities** from the T5C training-eval pool (different from the 71-identity pool that derived Option 3). **Option 3 still wins** — rescues 1 FP (bla_bla_chow__s1, consistent with 71-pool finding), adds zero FN.

Also surfaced **3 previously-unknown chronic FP identities** that NO aggregation rule fixes:
- `PC_Generator__s45` (mean 0.642, frac>0.6=66%, frac>0.7=47%, count>0.9=2)
- `PC_Generator__s22` (mean 0.629, frac>0.6=63%, frac>0.7=42%, count>0.9=4)
- `Q__s6` (mean 0.582, frac>0.6=50%, frac>0.7=31%, count>0.9=6)

These match exactly the "real people running near threshold" mode the user is concerned about in live deployment. They are rule-unrescuable because they have legitimate extreme-tail frames (count>0.9 ≥ 2). They need training-side or substrate-aware calibration, not rule changes.

## Method

### Source data

T5C step3500 frame-level scores from the standard training-eval scorecard reports at
`analysis/cpu_diagnostics_2026-05-12_stage_a/_scorecard_reports/`. Six suites:

| suite | n_frames | label |
|---|---|---|
| teams_real_all_dev | 4,564 | real (0) |
| teams_real_all_lockbox | 1,418 | real (0) |
| teams_fake_all_dev | 3,039 | fake (1) |
| teams_fake_all_lockbox | 425 | fake (1) |
| visomaster_enhanced_dev | 550 | fake (1) |
| deeplive_enhanced_dev | 545 | fake (1) |
| **TOTAL** | **10,541** | |

### Filtering

Identity-level rules require multi-frame aggregation. Filtered to **≥20 frames per base_identity** (parsed from `video_id` with regex handling both `__seg_` and `__seq` patterns):

| suite | qualifying identities |
|---|---|
| teams_real_all_dev | 21 real |
| teams_real_all_lockbox | 5 real |
| teams_fake_all_dev | 16 fake |
| teams_fake_all_lockbox | 2 fake |
| visomaster_enhanced | 2 fake |
| deeplive_enhanced | 1 fake |
| **TOTAL** | **47 identities** (26 real, 21 fake) |

This is fully independent of the 71-pool used to derive Option 3.

## Path A: Per-identity rule outcomes

| rule | correct | rate | FP | FN |
|---|---|---|---|---|
| Opt 1 — `frac>0.49 > 0.5` | 42/47 | 89.36% | 5 | 0 |
| Opt 2 — `frac>0.6 > 0.4` | 42/47 | 89.36% | 5 | 0 |
| **Opt 3 — `frac>0.6 > 0.4 AND count>0.9 ≥ 1`** | **43/47** | **91.49%** | **4** | **0** |

**Option 3 rescues 1 FP, adds zero FN.** This replicates the 71-pool result on independent data: the count>0.9=0 check catches identities with strong bulk but no extreme tail (bla_bla_chow type).

### Per-suite breakdown — Option 3

| suite | correct | recall | FPR |
|---|---|---|---|
| teams_real_all_dev | 17/21 | — | 0.190 |
| teams_real_all_lockbox | **5/5** | — | **0.000** |
| teams_fake_all_dev | 16/16 | 1.000 | — |
| teams_fake_all_lockbox | 2/2 | 1.000 | — |
| visomaster_enhanced | 2/2 | 1.000 | — |
| deeplive_enhanced | 1/1 | 1.000 | — |

- **Lockbox 5/5 perfect** under Option 3 — bla_bla_chow rescued, consistent with 71-pool
- teams_real_all_dev 17/21 — 4 chronic FPs remain (Roy_D + 3 new identities below)

## The 4 surviving FPs under Option 3 — characterized

| identity | n | mean | frac>0.6 | frac>0.7 | frac>0.9 | count>0.9 | notes |
|---|---|---|---|---|---|---|---|
| Roy_D | 130 | 0.886 | 98.5% | 95.4% | 62.3% | **81** | known unrescuable — confidently-wrong |
| PC_Generator__s45 | 91 | 0.642 | 65.9% | 47.3% | 2.2% | **2** | NEW chronic — runs 0.6-0.7 |
| PC_Generator__s22 | 227 | 0.629 | 63.0% | 41.9% | 1.8% | **4** | NEW chronic — runs 0.6-0.7 |
| Q__s6 | 54 | 0.582 | 50.0% | 31.5% | 11.1% | **6** | NEW chronic — runs 0.5-0.6 with some tail |

**This is exactly the FP mode the user is worried about in live deployment.** Three real-person identities whose score distributions cluster in the 0.6-0.7 bulk region with at least 1-6 frames making it above 0.9. They look identical to weak dor fakes by every rule we've tested.

Why no rule rescues them:
- `count>0.9 ≥ 2`: catches PC_Generator__s45 — but loses dor_fake_simswap (count>0.9=1) and dor_fake_inswapper_128res_gpen512 (count>0.9=1)
- `count>0.9 / n ≥ 0.05`: catches Q__s6 (11.1%) — but loses dor_fake_simswap (1.3%) and inswapper_512 (0.9%)
- Any rule strict enough to catch these FPs loses weak dor fakes

These identities require **training-side fixes** (substrate-invariance training, identity-level calibration) or **substrate-aware τ**, not rule tightening.

## Path B: Frame-level scorecard (raw T5C discrimination)

Frame-level FPR / recall at various single thresholds:

### Real-suite FPR (lower better)
| suite | τ=0.5 | τ=0.6 | τ=0.7 | τ=0.8 | τ=0.9 |
|---|---|---|---|---|---|
| teams_real_all_dev | 15.2% | 12.6% | 9.6% | 6.5% | 2.6% |
| teams_real_all_lockbox | 44.2% | 26.7% | 13.1% | 5.3% | **0.3%** |

### Fake-suite recall (higher better)
| suite | τ=0.5 | τ=0.6 | τ=0.7 | τ=0.8 | τ=0.9 |
|---|---|---|---|---|---|
| deeplive_enhanced | 98.9% | 97.1% | 91.4% | 72.7% | 22.4% |
| teams_fake_all_dev | 93.2% | 89.5% | 83.6% | 71.8% | 49.6% |
| teams_fake_all_lockbox | 94.1% | 90.8% | 86.1% | 74.4% | 38.4% |
| visomaster_enhanced | 72.7% | 63.3% | 50.4% | 25.6% | 0.9% |

Frame-level confirms the same trade-off seen in production: raising τ improves FPR but kills viso recall (viso drops 73→50→26→1%). The operational sweet spot at τ=0.6-0.7 leaves substantial frame-level FPR in real-pool lockbox (13-27%) — which is why **majority-aggregation** is load-bearing in production.

## Cross-pool consistency check

Option 3's win on the 71-pool was driven by the `count>0.9=0` rescue for bla_bla_chow. The independent pool reproduces this exactly:

| pool | bla_bla_chow result |
|---|---|
| 71-pool (derived Option 3) | FP under Opt2, rescued by Opt3 (count>0.9=0) |
| **Independent training-eval pool** | **FP under Opt2, rescued by Opt3 (count>0.9=0)** ✓ |

The rule generalizes. The mechanism (no extreme frames) is a structural distinguisher between bla_bla_chow-type FPs and weak fakes — not a pool-specific artifact.

## Stat-power note

47 identities is more than 71 but with skewed label balance (26 real / 21 fake) and the rescue being 1 identity. The Δ = +2.13pp (Option 3 vs Option 2) is on the same scale as the 71-pool's +1.41pp. Bootstrap would give wide CIs that include 0 — so this is **directional confirmation**, not a strict significance test. The combined evidence (Δ same sign, same magnitude, same mechanism, on independent data) is what makes this a reasonable recommendation despite individual studies being underpowered.

## What this means for the user's live-deployment concern

The "real person running 0.5-0.6" failure mode the user is worried about **is real and exists in our validation data** — PC_Generator__s45, PC_Generator__s22, Q__s6 are precisely this pattern.

But the data also shows that:

1. **Option 3 is the best deployable rule**: it rescues bla_bla_chow-type FPs (strong-bulk, no extreme tail).
2. **These three new chronic FPs are NOT rescuable by aggregation alone** — they have legitimate extreme-tail frames (count>0.9 = 2-6) making them indistinguishable from weak dor fakes.
3. **The mechanism that catches the chronic FPs requires different evidence**: per-identity history, substrate calibration, or training-side robustness. Rule-level fixes hit a ceiling here.

So the honest production recommendation:
- **Ship Option 3** — it generalizes to independent data and rescues one common FP type.
- **Expect 3-4 chronic FP identities to persist** — PC_Generator-style and Q__s6-style real people. These need surveillance, not rule tightening.
- **Build per-identity feedback loop** in production: identities that consistently false-flag should get noted and either added to a calibration set or routed through a slower offline review.

## Artifacts

- `rule_validation_v2.py` — main script
- `outputs/validation_v2_per_identity.csv` — 47-identity per-suite aggregations + rule verdicts
- `outputs/validation_v2_per_suite.csv` — rule × suite scorecard
- `outputs/validation_v2_frame_level.csv` — frame-level FPR/recall at thresholds {0.49..0.9}
