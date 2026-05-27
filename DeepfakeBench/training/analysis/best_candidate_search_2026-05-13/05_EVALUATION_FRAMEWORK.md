# Per-identity majority-vote evaluation framework (2026-05-13)

This document specifies the evaluation regime that matches how the production
app actually decides: many frames per identity, majority-vote aggregation.
It supersedes the per-frame FPR/recall framing used in prior R13 packets.

## Why the prior framework was misleading

Promotion-contract scorecards report **per-frame FPR/recall**. Production
deployment aggregates **per-identity** (a Teams call yields many frames from
one person; the call is flagged if >50% of frames pass tau). The two metrics
diverge when:

- A real identity has wide score variance (some frames pass tau, most don't).
  Per-frame says "some FPs"; per-identity says "no flag" if <50%.
- A model has a sharp score distribution per-identity. Per-frame underestimates
  reliability; per-identity sees a clean 100/0 cut.

Empirically: at per-frame FPR=5% on the production-realistic P2 substrate,
T5C looked weak (32% macro recall). At per-identity FPR=0% on F4, same model,
same tau region, T5C is at 100% recall. The model didn't change — the
evaluation regime did.

## The four metrics that matter

For each candidate ckpt at each per-frame threshold tau:

### M1 — Per-identity FPR
```
fpr_id = (# real identities where >50% of frames pass tau) / (# real identities)
```
On the eval substrate (F4 = chronic-cohort dropped + production gate), this is
the headline false-positive rate the user sees.

### M2 — Per-identity macro recall
```
recall_id = mean over fake suites of [(# fake identities where >50% of frames pass tau) / (# fake identities in that suite)]
```
What fraction of fake calls/sessions get flagged.

### M3 — Safety gap (the new addition)
```
gap = min(fake_id_frac_above_tau)  −  max(real_id_frac_above_tau)
```
The DISTANCE between the worst-detected fake and the worst-misbehaving real
in the fraction-above-tau space. **Larger gap = safer.**

Interpretation:
- gap > 0.40 — very safe. Even doubling per-session frame-noise wouldn't flip a verdict.
- gap = 0.20–0.40 — comfortable. Some marginal cases possible under heavy substrate drift.
- gap < 0.20 — fragile. A single drift event could flip a real or a fake.
- gap < 0 — unsound. Some fake identity has fewer frames above tau than some real identity.

### M4 — Per-frame fire rate on production drift cohorts
For each known production-drift dataset (may6 Xinhe, future cohorts):
```
fire_rate = (# frames with score > tau) / (# frames in cohort)
```
Must stay < 50% for the cohort's identity to pass majority-vote. The lower the
fire_rate, the more headroom against future drift in similar direction.

## The full evaluation protocol

1. **Apply production gate to real-identity panel:**
   - G1: face detector present
   - G2: min(W, H) ≥ 200 px
   - (G3 sharpness gate retired — not needed under majority-vote)

2. **Score every remaining frame with candidate ckpt + tau.**

3. **Aggregate per-identity:** for each base identity (extracted via
   `extract_identity_from_video_id`), compute `frac_above_tau`. Identity
   is flagged fake iff `frac_above_tau > 0.5`.

4. **Compute M1, M2, M3, M4** as defined above.

5. **Pick the tau that maximizes M3 (safety gap) subject to M1=0 and M2 hits
   the recall floor** (e.g., M2 ≥ 95% macro).

6. **Cross-validate against production-drift cohorts (M4).** Reject any
   recommendation where M4 ≥ 50% on any known drift cohort.

## Identity-level chronic handling

The "chronic-6" identities (bla_bla_chow, bla_bla_chow__s2, pc_generator__s22,
pc_generator__s45, q__s6, roy_d) are dropped via two mechanisms:

- **At inference (production):** G2 min(W,H)≥200 removes the 3 tiny-face
  chronics (pc_gen_s22/s45/q__s6). bla_bla_chow__s2 is also ~96% sub-200.
  Roy_D is the one chronic that passes G2.

- **At evaluation (F4 substrate):** explicit identity-list drop in addition
  to G2 + G3. This is the "ideal" eval condition.

For Roy_D specifically: it has median min(W,H)=270 → passes G2. Roy_D as an
identity gets flagged on T5C and P2D under majority-vote on the un-cleaned
P2 substrate. Mitigation options:

- (a) Tighten G2 from 200 to ~220–240 (Roy_D's lower-tail drops out).
- (b) Accept 1/21 false-positive identity on P2 (4.76% identity-FPR).
- (c) Add a sharpness check at higher resolutions (G3 lap_var < 80).
- (d) Train another packet that handles Roy_D-style substrate.

The safer path is (a) — a configuration change in production, no retraining.

## Per-suite recall caveat

The fake-side identity counts are uneven:
- visomaster_enhanced_macro_dev: 2 fake identities (raw + teams variants)
- deeplive_enhanced_dev: 1 fake identity (deeplive_dor)
- teams_fake_all_dev: 16 fake identities

Macro recall weighs each suite equally even though identity-counts differ.
Consider also reporting:
- **viso recall**: 2-identity binary, sensitive to either one being missed
- **deeplive recall**: single-identity binary (the entire suite passes or fails)
- **teams_fake recall**: 16-identity, the most reliable signal

A safety-conscious deployment would require ALL THREE suites to hit ≥ 95% recall.

## When to revisit this framework

If production deploys an aggregation other than majority-vote:
- 75% rule → re-derive M1/M2 with the new threshold
- mean-score-per-identity → use mean instead of fraction-above-tau
- spike-robust window with override → see `project_w48_failures` for the
  framework variant

If frame-count-per-identity in production differs significantly from the
eval data (e.g., production has 500 frames per session, eval has 100):
- Stage 9 bootstrap stability tells you whether the verdict is N-stable
- Re-run with the production frame count distribution

## Artifact paths

- `_stage6/STAGE6_PER_IDENTITY.csv` — full ckpt × tau × substrate × metrics
- `_stage7/STAGE7_SAFETY_GAP.csv` — adds the gap metric (M3)
- `_stage8/STAGE8_ENSEMBLE.csv` — 2-model ensemble
- `_stage9/STAGE9_BOOTSTRAP.csv` — frame-count stability
- `_stage10/STAGE10_PER_SUITE.csv` — per-fake-identity caught/missed
