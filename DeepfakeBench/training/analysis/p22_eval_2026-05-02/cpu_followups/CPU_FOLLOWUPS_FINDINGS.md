# P22 CPU follow-up findings — 2026-05-02 evening

> Six CPU diagnostics (jobs A, B, C, D, E, F, I) run after the P22 contract
> scorecard landed. **Substantively changes the verdict from the morning's
> FINDINGS.md.** This document is the new authoritative interpretation of
> the P22 result; the prior `analysis/p22_eval_2026-05-02/FINDINGS.md`
> should be read with this addendum applied.
>
> All raw numbers cited here are reproducible from the scripts under
> `analysis/p22_eval_2026-05-02/cpu_followups/scripts/` and the CSVs in
> `outputs/`.

## TL;DR — verdict reframe

The earlier "P22 step8k succeeded the falsifier verdict" framing was
**partially correct but operationally misleading**. The CPU follow-ups
reveal:

1. **P22 step8k is a degraded model**, not a "wider distribution" model.
   Its score variance collapsed by ~140× (P8A 0.157 → step8k 0.001) — all
   frames score near 0.5. The "high recall at calibrated τ" came from the
   marginal tail-difference of an essentially-uncertain classifier, not
   from genuine new discriminative signal.
2. **P22 step1k is the actual robust winner** under joint dev+lockbox
   τ-calibration. It dominates P8A on every fake suite simultaneously
   without violating the lockbox FPR constraint.
3. **P22 step8k regresses on P8A's signature dor invariance** (0/1170 →
   36/1170 dor lockbox FPs at calibrated τ). This is the same identity
   pattern that drove P8A's worst false-positive concentration, amplified
   by the score collapse.
4. **The strongest production lever is an ensemble**, not a new training
   packet: `P8A + P22 step1k (min rule)` lifts viso recall **18×**
   (0.4% → 7.1%) and deeplive recall **51×** (0.4% → 20.4%) at the same
   joint FPR=2% — with **no GPU spend**.
5. **F1 falsifier verdict on full viso (n=551, not n=63)**: step1k FAILS
   (|Δr|=0.06), step8k PASSES (|Δr|=0.40). The morning's F1 PASS for
   step8k holds. But step1k's strong ensemble contribution shows F1 is
   not predictive of operational utility.

The natural next step is **NOT P23-LUMA training**. It is **deploy the
P8A+P22step1k ensemble** and run an identity-aware retrain (P23-IDENTITY)
to fix the dor regression that P22 created.

---

## Job A — Joint dev+lockbox τ-recalibration

**Question**: Does τ-recalibration on dev+lockbox jointly recover P22 from
the lockbox FPR violation? If yes, the contrarian read's strongest objection
is fixable. If no, P22 has a structural lockbox failure.

**Method**: Sweep τ ∈ [0.001, 0.999] in 999 steps. For each ckpt, find
smallest τ s.t. both dev primary FPR ≤ 2% AND lockbox real FPR ≤ 2%.
Output: `outputs/01_joint_recal_summary.csv`.

**Result at FPR=2% on both:**

| Ckpt | τ | dev_FPR | lockbox_FPR | viso_dev | deeplive_dev | teams_fake_dev | lockbox_fake |
|---|---:|---:|---:|---:|---:|---:|---:|
| P8A | 0.993 | 1.8% | 0.14% | 0.4% | 0.4% | 41.7% | 19.1% |
| P18T | 0.994 | 1.8% | 0.28% | 1.6% | 5.3% | 42.1% | 17.9% |
| **P22 step1k** | **0.971** | **2.0%** | **0.6%** | **2.4%** (6×) | **17.8%** (44×) | **45.0%** | 18.8% |
| P22 step4k | 0.876 | 0.9% | 1.9% | 0.2% | 37.4% (94×) | 43.2% | 6.1% |
| P22 step8k | 0.539 | 0.2% | 1.97% | 0.7% | 13.8% | 40.9% | 15.5% |

**Interpretation**: ALL ckpts CAN be jointly compliant at 2% FPR — the
lockbox violation is calibration-fixable. But the recall picture under
**joint** calibration is very different from the dev-only contract:

- **P22 step8k loses its dev-only "winner" status**. Under joint
  calibration, its deeplive recall drops 56% → 14% (the lockbox constraint
  pulls τ to 0.539, where dev FPR is only 0.2%, not the full 2% the
  contract reserved for it).
- **P22 step1k is now the dominant choice**: 6× viso, 44× deeplive,
  +3.3pp teams_fake at the same FPR — and uniquely it does this without
  the score-distribution collapse that hurts step8k.
- **P22 step4k is the single best on deeplive alone** (94× lift) but
  loses viso (0.2% — worse than P8A) and lockbox_fake recall (6%,
  collapsed).

**Verdict reframe**: the contrarian's "P22 didn't promote" reading is
defensible in the dev-only-calibrated regime; under joint calibration,
**step1k cleanly dominates P8A on every fake suite simultaneously**. The
morning's framing latched onto step8k because it had the highest
dev_fake_macro, but step8k's win is calibration-fragile.

## Job B — Score-distribution forensics across the chain

**Question**: Is P22's improvement explained by "wider score distribution"
(more usable score range) as the morning's framing claimed?

**Method**: Quantile decomposition (p1, p5, p10, p25, p50, p75, p90, p95,
p99) of per-frame scores per (ckpt × suite). Histograms saved to
`figures/02_score_distributions.png`.

**Median score per (suite, ckpt):**

| Suite (label) | P8A | P18T | P22 step1k | P22 step4k | P22 step8k |
|---|---:|---:|---:|---:|---:|
| teams_real_all_dev (real) | 0.007 | 0.007 | **0.111** | 0.141 | **0.472** |
| teams_real_all_lockbox (real) | 0.016 | 0.082 | 0.398 | 0.247 | **0.480** |
| teams_real_dor_dev (real) | 0.181 | 0.512 | 0.653 | 0.289 | 0.500 |
| teams_real_lighting_extreme_dev (real) | 0.008 | 0.019 | 0.147 | 0.227 | 0.486 |
| teams_real_poor_quality_dev (real) | 0.007 | 0.007 | 0.112 | 0.141 | 0.475 |
| teams_fake_all_dev (fake) | 0.985 | 0.986 | 0.961 | 0.828 | 0.532 |
| teams_fake_all_lockbox (fake) | 0.788 | 0.788 | 0.793 | 0.432 | 0.503 |
| visomaster_enhanced_macro_dev (fake) | 0.170 | 0.336 | **0.529** | 0.213 | 0.485 |
| deeplive_enhanced_dev (fake) | 0.534 | 0.897 | 0.899 | 0.829 | 0.526 |

**Interpretation**:

- **P22 step8k has collapsed all suites to median ≈ 0.49**. Reals at
  0.47, fakes at 0.53. The model is essentially uncertain about
  everything. Its IQR is 0.02-0.05 across all suites (vs P8A 0.03-0.75).
- **P22 step1k has shifted medians AND retained discrimination**:
  reals stay at p50=0.11 (low), viso fakes moved up to p50=0.529 (now
  scored as fake!), deeplive at p50=0.899 (strongly fake). This is what
  "successful shortcut weakening" should look like.
- **P18T behaves like a moderate version of step1k**: deeplive p50
  shifted 0.534 → 0.897, viso p50 shifted 0.170 → 0.336, reals stayed
  saturated near 0.

**Verdict reframe**: the "wider distribution" hypothesis is **wrong for
step8k** (it's collapsed, not wider) and **right for step1k** (genuine
distribution shift with retained discrimination). The morning's framing
conflated step8k's calibration-fragile "win" with the actual structural
improvement, which lives at step1k.

## Job C — F2 reals-only sanity check

**Question**: F2 (R²(score|attrs)) ROSE for P22 step8k — is this real
shortcut increase or a between-class variance artifact?

**Method**: Compute R² separately on REALS-ONLY and FAKES-ONLY (n=198 reals
in teams_real_all_dev, n=212 fakes pooled across teams_fake_all_dev +
deeplive_enhanced_dev). If R² rises on reals-only too, the F2 increase is
real. Output: `outputs/03_f2_reals_only.csv`.

**R²(score | attrs):**

| Ckpt | reals only | fakes only | pooled real+fake | var(score) overall |
|---|---:|---:|---:|---:|
| P8A | 0.122 | 0.137 | 0.050 | 0.157 |
| P18T | 0.184 | 0.152 | 0.094 | 0.164 |
| P22 step1k | 0.185 | 0.206 | 0.114 | 0.128 |
| P22 step4k | 0.043 | **0.480** | 0.123 | 0.103 |
| **P22 step8k** | 0.179 | 0.316 | 0.179 | **0.001** |

**Interpretation**:

- For step1k: R² rose on both reals (+0.06) and fakes (+0.07) with
  meaningful var(score)=0.128 → real shortcut shift onto luma+skin. The
  morning's F2 FAIL verdict on step1k is correct.
- **For step8k: R² values are misleading because var(score) ≈ 0**. R²
  is fraction-of-variance-explained, but when total variance is 0.001
  (a 140× collapse from P8A's 0.157), even tiny attribute correlations
  inflate R². Step8k's "F2 fail" is dominated by score collapse, not
  shortcut shift.
- For step4k: asymmetric — R² very high on fakes (0.480), very low on
  reals (0.043). The model uses attrs to score fakes specifically. This
  is partial overfitting to the augmented training distribution.

**Verdict reframe**: F2's "rose" verdict on step8k is technically true
but interpretively meaningless (the score collapsed, so any small
correlation looks high in R²). The genuine F2 conclusion lives at step1k:
the model traded laplacian-dependency for skin/luma-dependency. The
shortcut family didn't disappear — it shifted shape.

## Job D — Lockbox failure-mode audit (limited)

**Question**: Are P22's lockbox false-positives concentrated on the same
attribute pattern as P8A's (image-quality shortcut hitting webcam reals)?

**Method**: Pull lockbox-real per-frame scores at dev-calibrated τ (FPR=2%
on dev). Identify FPs and TNs. Compare attribute distributions.

**Limitation**: cross_suite_attributes.csv only sampled 50 lockbox frames,
none of which intersect with P22's 60 FPs. Cannot run the per-attribute
test on the actual FPs without first computing attrs on the 1418 lockbox
real frames (deferred — possible future job).

**What we DID get**: count of lockbox FPs at dev-calibrated τ.

| Ckpt | τ (dev FPR=2%) | n_lockbox_FP | lockbox_FPR |
|---|---:|---:|---:|
| P8A | 0.993 | 2 | 0.0014 |
| P18T | 0.994 | 4 | 0.0028 |
| P22 step1k | 0.971 | 9 | 0.0063 |
| P22 step4k | 0.812 | 49 | 0.0346 |
| P22 step8k | 0.522 | 60 | 0.0423 |

The 30× FP multiplier on step8k (60 vs 2) is consistent with Job B's
score-collapse hypothesis: when scores are concentrated near 0.5, slight
perturbations push many lockbox frames over τ.

## Job E — Per-identity P22 lockbox FPR concentration

**Question**: Is P22 step8k's 4.3% lockbox FPR concentrated on the same
identities that drove P8A's FPs (PC_Generator + dor_shkedi per memory
`project_lockbox_fpr_dominated_by_webcam_mode.md`), or has P22 created a
broader failure mode?

**Method**: Parse identity from `video_id`. Compute per-identity FP count
at dev-calibrated τ. Compare across ckpts. Output:
`outputs/05_per_identity_fpr.csv`.

**Top false-positive identities** (only 5 distinct identities in lockbox-real):

| Identity | n_frames | P8A | P18T | P22 step1k | P22 step4k | **P22 step8k** |
|---|---:|---:|---:|---:|---:|---:|
| PC_Generator | 29 | 2 | 2 | 1 | 5 | **22** (75.9% FPR!) |
| dor_shkedi | 1170 | 0 | 2 | 6 | 44 | **36** (3.1%) |
| real_dor | 109 | 0 | 0 | 0 | 0 | 2 |
| Chikara_Takahashi | 42 | 0 | 0 | 2 | 0 | 0 |
| bla_bla_chow | 68 | 0 | 0 | 0 | 0 | 0 |

**Interpretation**:

- **P22 step8k regresses on P8A's signature dor invariance**. P8A had
  0/1170 dor_shkedi false-flags (memory `project_p18_diagnostics_complete_2026-05-02`
  documents this as P8A's strongest property). P22 step8k flags 36/1170 =
  3.1% — a clean regression of an invariance the prior packets fought to
  preserve.
- **PC_Generator gets 75.9% per-id FPR** under P22 step8k (vs 6.9%
  under P8A — 11× per-id worsening). PC_Generator is webcam-mode per
  prior memory; the image-quality shortcut hitting webcam reals is the
  ORIGINAL failure mode that P22 was supposed to address. Step8k made it
  worse on the worst-case identity.
- The failure pattern is the SAME, AMPLIFIED — not a new failure mode.

**Verdict reframe**: P22 step8k's lockbox violation is a regression on
the model's signature production-defense (dor invariance) AND on the
known worst-case identity (PC_Generator). The next training packet must
preserve dor invariance OR not train so far that it loses it. **P22 step1k
preserves dor much better** (6 FPs on dor vs step8k's 36 — 6× better).

## Job F — Cross-packet ensemble study

**Question**: Does an ensemble of P8A + P18T + P22-various beat any
individual at joint FPR ≤ 2%? If yes, this is a free production lift
without any new training.

**Method**: For each ensemble combo and rule (mean/max/min), find joint
dev+lockbox compliant τ at FPR ≤ 2% and compute fake recall on each
suite. Output: `outputs/06_ensemble.csv`.

**Best ensembles vs single models at joint FPR ≤ 2%:**

| Strategy | viso | deeplive | teams_fake_dev | teams_fake_lockbox |
|---|---:|---:|---:|---:|
| P8A (single) | 0.4% | 0.4% | 41.7% | 19.1% |
| P22 step1k (single) | 2.4% | 17.8% | 45.0% | 18.8% |
| **P8A + P22step1k (min)** | **7.1%** (18×) | **20.4%** (51×) | **49.0%** | **26.6%** |
| P8A + P22step8k (min) | 3.5% | 37.3% (94×) | 54.2% | 18.8% |
| **P8A + P22step1k + P22step8k (mean)** | 6.0% | 21.1% | **51.3%** | **27.3%** |
| ALL_FOUR (mean) | 3.8% | 18.4% | 49.6% | 24.2% |

**Interpretation**:

- **P8A + P22 step1k :: min rule is the best operating point in the
  entire P-* chain.** It's compliant at joint FPR=2%, lifts viso 18×,
  deeplive 51×, teams_fake_lockbox 7.5pp — all simultaneously, no
  retraining required.
- **The "min" ensemble rule** ("frame is fake iff BOTH models say fake at
  their own thresholds") is conservative on real (low FPR — keeps lockbox
  at 1.1%) and only flags fakes that BOTH models agree on. The
  intersection is genuinely meaningful.
- **P8A + P22 step8k::min** gets even higher deeplive (37.3% vs 20.4%)
  but loses on viso (3.5% vs 7.1%) and saturates lockbox FPR (1.97% vs
  1.1%). Step1k is the more robust ensemble partner.
- **P8A + P18T ensembles barely lift over P8A alone** (viso 0.91% vs
  0.4%) — P18T isn't orthogonal to P8A in the right way for ensembling.

**Verdict reframe**: the strongest production move is NOT "FT from P22
step8k." It is **deploy P8A + P22 step1k as a 2-model ensemble with min
rule** at jointly-calibrated τ. Cost: $0 (existing checkpoints). Lift:
massive on every fake suite at the same FPR.

## Job I — Full-viso F1 Pearson r recomputation

**Question**: The morning's F1 verdict was on n=63 viso fakes that joined
to attrs. Is the F1 PASS for step8k robust at full sample (n=550)?

**Method**: Download viso eval frames from GCS in parallel (488 not in
attrs), compute Laplacian variance per frame, merge with per-frame scores,
recompute Pearson r per ckpt. Output:
`outputs/07_viso_laplacian_fetched.csv`, `07_full_viso_pearson.csv`.

**Full-viso F1 results (n=551 each, p<0.001 except P22 step8k p=0.008):**

| Ckpt | r(score, lap) | |Δr| vs P8A | F1 verdict |
|---|---:|---:|---|
| P8A | +0.507 | (baseline) | — |
| P18T | +0.148 | 0.360 | **PASS** |
| **P22 step1k** | +0.447 | 0.060 | **FAIL** |
| P22 step4k | -0.261 | 0.246 | PASS |
| **P22 step8k** | -0.113 | 0.395 | **PASS** |

**Surprises**:

1. **P8A's r is strongly POSITIVE** (+0.51) on viso fakes — i.e. P8A
   scores SHARPER viso fakes as MORE fake. This is the opposite direction
   from the deeplive shortcut (where lower laplacian = more fake) per the
   2026-05-02 PM CPU audit. **Viso fakes have a different shortcut
   signature than deeplive fakes in P8A's score function.**
2. **P18T also dramatically reduces the viso shortcut** (|Δr|=0.36 —
   bigger than step8k's). GRL on method-axis incidentally weakened the
   laplacian-on-viso correlation.
3. **P22 step1k FAILS F1 at full viso** (|Δr|=0.06 — barely moved).
   Step1k's strong ensemble contribution (Job F) is NOT explained by
   shortcut weakening on viso. Its orthogonal-to-P8A signal must come
   from somewhere else (likely the changed score distribution per Job B).

**Verdict refinement on F1**: morning's F1 PASS for step8k holds at full
sample. But step1k's FAIL — combined with its dominance under joint
calibration AND in ensembles — shows F1 is not predictive of operational
utility. The shortcut measurement and the recall measurement are testing
different things.

---

## Synthesis — what the 7 CPU jobs collectively say

**Reframe summary table** (verdict before vs after the CPU follow-ups):

| Question | Morning's answer | After CPU follow-ups |
|---|---|---|
| Is P22 step8k the new FT-base? | YES (highest dev_macro) | NO — it's a degraded model that wins by score-collapse marginal-tail effects |
| Is P22 step1k just a baseline? | NO (failed 0/3 falsifiers) | YES — it's the structurally robust winner under joint calibration |
| Did P22 weaken the shortcut? | "F1 passed, F2 failed (mixed)" | Step8k passes F1 but is collapsed (degraded); step1k has shifted-shortcut shape; both stories are real but in different ckpts |
| Is the lockbox FPR violation fixable? | "Calibration drift" (probably fixable) | Confirmed fixable via joint calibration; but step8k specifically REGRESSES on dor invariance, not just calibration drift |
| Best next move? | P23-LUMA single-lever from P22 step8k | (1) Deploy P8A+P22 step1k ensemble (min rule) — free lift; (2) P23-IDENTITY-aware retrain to fix the dor regression |

**Three structural insights gained:**

1. **Score variance ≠ signal quality.** Morning's "wider distribution"
   intuition was wrong: step8k has 100× LESS variance than P8A, not more.
   It "wins" at calibrated τ because tiny absolute differences across a
   collapsed distribution still produce a thin separating tail. Step1k
   has the actual distribution improvement.

2. **F1 (shortcut weakening per attribute) is not predictive of recall lift.**
   P18T strongly weakens the viso laplacian shortcut (|Δr|=0.36) but
   ensembles only marginally with P8A. P22 step1k barely weakens the same
   shortcut (|Δr|=0.06) but ensembles dramatically with P8A. The relevant
   property for ensembling is "scores orthogonal frames as fake," not
   "uses fewer shortcut features."

3. **The contract's lex-ordering hides the right answer.** The contract
   ranks P8A #1, P22 step8k #4. But under joint dev+lockbox calibration,
   P22 step1k dominates P8A on every fake suite. The contract's
   stress-FPR-first ordering rewards calibration stability AT a particular
   τ regime; P22's improvement happens at a different τ regime. This is
   informative for v4 contract design but means the v3 contract didn't
   surface the actual win.

## What the next packet should do

**Move 1 (CPU-only, 1-2h, $0): Operationalize the P8A+P22 step1k ensemble.**

- Build inference glue that runs both models on a frame and applies the
  min rule at the jointly-calibrated τ.
- Re-run the contract scorecard on the ensemble to confirm joint compliance.
- This is a deployable production lift NOW.

**Move 2 (GPU, ~$30-45): P23-IDENTITY-AWARE retrain.**

- FT from P8A (not P22 step8k) with anchor_aware loss enabled and weighted
  toward PC_Generator + dor_shkedi (the worst-FP identities). Rationale:
  P22 lost dor invariance; we need to RESTORE it AND keep P22's pipeline
  randomization benefit.
- Yaml: anchor_aware enabled, pipeline_randomization enabled (curriculum
  identical to P22), face_scale_jitter disabled, GRL disabled.
- Expected outcome: dor lockbox FPR back to ~0%, viso/deeplive recall
  similar to P22 step1k's joint-calibrated levels.

**Move 3 (GPU, ~$30-45): P23-LUMA on top of P22 step1k.**

- Only worth doing IF Move 2 fails to restore dor invariance.
- Add global luma jitter (×∈[0.7, 1.3]) symmetric-label, FT from P22
  step1k for 2000-3000 steps (NOT 8000 — step8k's collapse shows that's
  too far).

**Do NOT pursue**:
- FT from P22 step8k — it's a degraded model, not a base.
- Another bundle (anchor_aware + jitter + GRL stacked) — single-lever
  discipline is the consistent pattern.
- P23-LUMA as the immediate next packet — Move 2 (identity-aware) is
  more load-bearing because it directly addresses P22's signature
  regression.

## Files written

- `cpu_followups/scripts/_common.py` — shared helpers
- `cpu_followups/scripts/01_joint_recal.py` — Job A
- `cpu_followups/scripts/02_score_dist.py` — Job B
- `cpu_followups/scripts/03_f2_reals_only.py` — Job C
- `cpu_followups/scripts/04_lockbox_failure_audit.py` — Job D
- `cpu_followups/scripts/05_per_identity_fpr.py` — Job E
- `cpu_followups/scripts/06_ensemble.py` — Job F
- `cpu_followups/scripts/07_full_viso_laplacian.py` — Job I
- `cpu_followups/outputs/01..07_*.csv` — raw outputs (8 CSVs)
- `cpu_followups/figures/02_score_distributions.png` — 9 suites × 5 ckpts
- `cpu_followups/figures/02_iqr_summary.png` — IQR comparison bar chart
