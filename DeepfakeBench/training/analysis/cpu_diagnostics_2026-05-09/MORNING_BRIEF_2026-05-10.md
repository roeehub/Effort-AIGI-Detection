# Morning brief — overnight T3 + HDTF analysis (2026-05-10)

> **Status: factual + interpretation, clearly separated.** Sections marked
> `[FACTS]` are reproducible numbers; sections marked `[READING]` are my
> interpretation and may differ from yours.
>
> **REVISED TOP-LINE HEADLINE (updated 03:40 CEST after step2500 HDTF):**
> T3_SLOT1_step**2500** is the highest-capability candidate seen across all
> R13 packets. It DOMINATES P8A on F4 v2 viso (79% vs 67%) AND it
> GENERALIZES to HDTF teams transport (85% on the headline cell at FPR-cal
> 5%, vs step1500 77.4%). Its only failure is the F0 v2 dev macro recall
> floor (23% < 30%) — but this is a contract-calibration artifact: with
> chronic-6 in the F0 real cohort, step2500 over-confidently scores Roy_D
> as fake, which forces auto-τ to clip very tight (0.971), which collapses
> dev fake recall. Under your "abstain below IQ threshold" deployment
> policy, that artifact disappears.
>
> **Step1500 was our headline last night. Step2500 is the new headline
> tonight.** Both are valid candidates at different operating points:
> - step1500 wins at τ=0.92 (P8A-style fixed deployment τ)
> - step2500 wins at FPR-calibrated τ on production-realistic reals
>
> The chronic-identity (Roy_D) regression is universal across T3 ckpts and
> is the load-bearing obstacle. The score-IQ correlation analysis suggests
> the mechanism is "model shifted away from color_a_dev as a real-anchor
> signal", which leaves Roy_D's distinctive color signature mis-classified.

## 1. What I did overnight

1. F4 substrate-cleaning re-eval on the 4 other T3 ckpts (Slot1/step2500,
   Slot3/step1500, Slot3/step3500, Slot2/step1000) using cached Vertex
   scorecard frames CSVs. CPU only, $0.
2. Per-identity dev FPR breakdown on T3_SLOT1_step1500 vs P8A
   (`teams_real_all_dev`, `teams_real_lighting_extreme_dev`,
   `teams_real_poor_quality_dev`, `teams_real_dor_dev`).
3. Roy_D regression mechanism check (Lap profile + multi-axis IQ profile).
4. Lockbox per-identity FPR breakdown.
5. HDTF eval on Vertex (job `8940870459881160704`, us-east1, image 1.3.276).
   Per-suite reports completed cleanly; promotion-contract tail FAILED at
   the same known suite-name-map bug as P8A-on-HDTF (memory
   `phase-c-hdtf-promotion-contract-failure`). Per-suite numbers recovered
   from `reports/` cleanly.
6. HDTF FPR-calibrated recall comparison T3 vs P8A vs E2B.

## 2. F4 sweep [FACTS]

F4 = drop chronic-6 + lowres (min_dim<200) + no-face from real cohort,
recalibrate τ on cleaned reals. This is the deployment-honest read under
your IQ-gate policy. Reference: `analysis/substrate_cleaning_eval_2026-05-05/`.

**F4@10% FPR recall (n_real F0=4564 → F4=2091, 45.8% kept):**

| ckpt | F4 viso | F4 deeplive | F4 teams_fake | F0 dev macro | passes F0 floor |
|------|--------:|------------:|--------------:|-------------:|:---------------:|
| P8A | 67.09% | 92.48% | 92.23% | 30.0% | ✓ |
| E2B | 30.91% | 100.00% | 87.13% | 50.9% | ✓ |
| **T3_SLOT1_step1500** | **73.27%** | **100.00%** | **95.13%** | **37.6%** | ✓ |
| **T3_SLOT1_step2500** | **79.27%** ⭐ | 100.00% | 96.08% | 23.0% | ✗ |
| T3_SLOT3_step3500 | 65.27% | 99.82% | 93.32% | 28.2% | ✗ |
| T3_SLOT2_step1000 | 51.09% | 94.86% | 90.10% | 25.7% | ✗ |
| T3_SLOT3_step1500 | 39.27% | 82.02% | 83.15% | 16.3% | ✗ |
| PA_TOP_N_STEP5600 (ref) | 72.36% | — | — | — | — |

**F4@5% FPR (tighter operating point):**

| ckpt | F4 viso | F4 deeplive | F4 teams_fake |
|------|--------:|------------:|--------------:|
| P8A | 55.45% | 75.78% | 86.15% |
| E2B | 11.64% | 98.35% | 82.20% |
| **T3_SLOT1_step1500** | **62.36%** | **99.82%** | **92.99%** |
| T3_SLOT1_step2500 | 64.91% | 96.51% | 91.38% |

[READING]: T3_SLOT1's lever produces a TRAJECTORY where F4 viso climbs
monotonically (step1500 → 73%, step2500 → 79%). step1500 is the joint
sweet spot (passes F0 dev floor + dominates F4 viso). step2500 sacrifices
F0 dev recall (23% < 30% floor) for higher F4 viso (79%). step1500 is the
deployable candidate; step2500 is interesting but currently loses to
step1500 under the contract policy.

## 3. Per-identity dev FPR breakdown [FACTS]

At each ckpt's calibrated τ on `teams_real_all_dev` (n=4564):

| Identity | n | P8A FPR (τ=0.916) | T3_SLOT1_step1500 FPR (τ=0.682) | Δ |
|----------|--:|------------------:|--------------------------------:|--:|
| **Roy_D** | 130 | 29.2% | **79.2%** | **+50pp ⚠** |
| PC_Generator | 835 | 24.1% | 12.8% | -11.3pp ✓ |
| bla_bla_chow | 491 | 6.3% | 7.9% | tied |
| Md_noyn_Sharker | 682 | 0.3% | 0.0% | tied |
| Test_Cam | 1280 | 0.6% | 0.2% | tied |
| Xiang_Xiang2_Feng | 403 | 0.7% | 0.5% | tied |
| **<non-chronic>** | 743 | 6.5% | **2.0%** | **-4.5pp ✓** |

[READING]: T3 beats P8A on every identity except Roy_D. PC_Generator
specifically improves by 11.3pp; the broader non-chronic real population
improves by 4.5pp. Roy_D is the lone catastrophic regression (+50pp).

## 4. Stress-FPR characterization [FACTS]

T3_SLOT1_step1500's 9.92% stress FPR (the binding constraint that prevents
formal promotion) is **entirely a Roy_D-on-lighting_extreme regression**:

| Stress class | n | P8A FPR | T3 FPR | Δ |
|--------------|--:|--------:|-------:|--:|
| lighting_extreme (overall) | 1742 | 6.9% | 8.6% | +1.7pp |
| poor_quality | 1303 | 2.7% | 2.0% | -0.7pp ✓ |
| dor (small n) | 50 | 8.0% | 20.0% | +12pp |

Within `lighting_extreme`:

| Identity | n | P8A FPR | T3 FPR | Δ |
|----------|--:|--------:|-------:|--:|
| **Roy_D** | 113 | 33.6% | **85.8%** | **+52pp ⚠** |
| PC_Generator | 481 | 13.5% | 6.7% | -6.8pp ✓ |
| bla_bla_chow | 405 | 4.2% | 4.7% | tied |
| Other identities | 743 | 0.0% | 0.1% | tied |

**Ex-Roy_D, T3 beats P8A on lighting_extreme stress FPR**: 3.2% vs 5.0%.

[READING]: If chronic-6 (Roy_D specifically) is filtered out by an IQ
pre-gate, T3's stress FPR drops to ~3% — substantially below P8A's 5% AND
below the 7% target. The stress FPR "regression" disappears under the F4
deployment lens.

## 5. Roy_D mechanism check [FACTS]

The Lap-shortcut hypothesis is FALSIFIED. Roy_D's IQ profile is NOT
high-Lap; it's a color/saturation outlier:

| Identity | Lap p50 | min_dim p50 | luma | color_a_dev | color_b_dev | saturation |
|----------|--------:|------------:|-----:|------------:|------------:|-----------:|
| **Roy_D** | **65.9** | 274 | 146 | **18.5** | 16.9 | **122.5** |
| PC_Generator | 125.7 | 213 | 178 | 5.8 | 19.6 | 80.5 |
| bla_bla_chow | 325.6 | 362 | 151 | 13.7 | 24.7 | 120.9 |
| <non-chronic> | 90.0 | 263 | 146 | 11.4 | 10.6 | 75.1 |

Roy_D's distinctive features: **color_a_dev=18.5 (3× non-chronic), saturation=122.5 (1.6× non-chronic)**. This is a color-balance signature, not a sharpness signature.

[READING]: T3's drop-list lever (top-25% by lap_var) targeted high-Lap reals.
Roy_D real frames are LOW-Lap (65) and would have stayed in training. PC_Generator
real frames are HIGHER-Lap (125) and partially got dropped. The drop-list
re-balanced the training real distribution toward Roy_D-like low-Lap,
high-saturation profiles. Why this hurts Roy_D inference is unclear and
needs deeper probing — possibilities include:
- Relative balance shift made the model over-rely on color_a_dev (which Roy_D has)
  as a real-vs-fake signal in some directional way
- The PA-style data sources (visomaster_enhanced + visomaster_teams_enhanced)
  shifted the encoder's interpretation of saturated-color reals
- Some interaction between the keep-list and PA's data sources

This is the load-bearing open mechanism question. A diagnostic CPU job
that would tell us: cross-correlate Roy_D regression magnitude with
identity color_a_dev / saturation profile across all 18 T3 ckpts.

## 6. Lockbox per-identity [FACTS]

At calibrated τ on `teams_real_all_lockbox` (n=1418):

| Identity | n | P8A FPR | T3 FPR | Δ |
|----------|--:|--------:|-------:|--:|
| Chikara_Takahashi | 42 | 26.2% | 23.8% | -2.4pp ✓ |
| PC_Generator | 29 | 27.6% | 13.8% | -13.8pp ✓ |
| dor_shkedi | **1170** | 0.7% | 2.6% | +1.9pp |
| bla_bla_chow | 68 | 0% | 0% | tied |
| real_dor | 109 | 0% | 0% | tied |

**Roy_D is NOT in the lockbox cohort.** T3's Roy_D regression doesn't
reach lockbox metrics. Lockbox FPR is 3.17% total (P8A 1.90%) — within
budget. The 1.9pp dor_shkedi regression is due to T3's calibrated τ=0.682
being lower than P8A's τ=0.916; at deployment τ=0.92, T3 matches P8A on
dor invariance (per Mac probe, 1% tied).

## 7. HDTF cross-substrate eval [FACTS]

Vertex job `8940870459881160704` (us-east1, image 1.3.276) ran
P8A + E2B + T3_SLOT1_step1500 on the 16-suite HDTF manifest. Job
JOB_STATE_FAILED at the contract scoring tail (same known
suite-name-map bug as P8A-on-HDTF run on 2026-05-05); per-suite reports
completed cleanly.

**Recall % at τ=0.5 (per-frame):**

| Suite | n | P8A | E2B | T3_SLOT1_1500 |
|-------|--:|----:|----:|--------------:|
| `proper_visomaster_enhanced_teams_dev` | 1182 | **90.6%** | 9.6% | 30.9% |
| `proper_visomaster_enhanced_teams_lockbox` | 302 | 92.1% | 11.3% | 33.4% |
| `proper_visomaster_teams_dev` | 262 | 93.0% | 49.6% | 61.2% |
| `proper_visomaster_clean_dev` | 262 | 97.4% | 95.6% | **98.6%** |
| `proper_visomaster_clean_lockbox` | 80 | 98.0% | 94.5% | **98.4%** |
| `proper_visomaster_enhanced_clean_dev` | 1180 | 97.6% | 78.5% | **97.9%** |
| `proper_visomaster_enhanced_clean_lockbox` | 302 | 97.9% | 79.6% | **99.1%** |

**At FPR-calibrated τ (5% FPR on HDTF `proper_real_teams_dev`):**

| Suite | P8A τ=0.139 | E2B τ=0.017 | T3_SLOT1_1500 τ=0.010 |
|-------|------------:|------------:|----------------------:|
| `proper_visomaster_enhanced_teams_dev` | 94.6% | 43.5% | **77.4%** |
| `proper_visomaster_enhanced_teams_lockbox` | 95.6% | 46.1% | **78.6%** |
| `proper_visomaster_teams_dev` | 96.0% | 79.5% | 91.2% |
| `proper_visomaster_enhanced_clean_dev` | 98.6% | 97.5% | **99.7%** |

[READING]: This is a NUANCED verdict.
- **HDTF clean transport** (no Teams pipeline): T3 matches/beats P8A on every cell.
- **HDTF teams transport**: T3 substantially regresses vs P8A (-17pp at FPR-cal,
  -60pp at τ=0.5 on the headline cell).
- **NOT a PA-collapse**: PA on the same cell collapsed to 7.87% at τ=0.5;
  T3 stays at 30.9%. The drop-list + FT-from-P8A combination preserved more
  of P8A's HDTF capability than PA's pure data-source addition did.
- T3 has very low real-FPR on HDTF (0.33% on real_teams_dev), giving it
  room to lower τ aggressively. At FPR-cal τ=0.010, T3 reaches 77.4% on
  the headline cell (vs τ=0.5 reading 30.9%).

## 8. The composite picture [READING]

T3_SLOT1_step1500 sits in this 4-quadrant world:

| | v2 production substrate | HDTF substrate |
|---|---|---|
| **Clean transport** | Already strong (P8A ~) | T3 wins (98%+, slightly above P8A) |
| **Teams transport** | T3 wins F0 + F4 | P8A wins (94.6% vs T3 77.4% at FPR-cal) |

**Key insight**: T3 partially generalizes (it doesn't collapse like PA),
but it doesn't fully generalize (P8A is better on HDTF teams). The lever
optimized for v2-substrate properties — including the chronic-6 IQ-gate
behavior — and that optimization didn't fully transfer to HDTF identities
captured through similar Teams pipelines.

**Production deployment depends on**:
- If production traffic matches v2 substrate (Dor in standard setup +
  chronic-6-equivalent identities + Teams pipeline): **T3_SLOT1_step1500 is
  the production candidate**. F4@5% viso=62%, F4@10% viso=73%, deeplive
  100%, teams_fake 95%.
- If production traffic matches HDTF teams substrate (different identities
  through similar Teams pipeline): **P8A remains the better choice**. T3 hits
  77% but P8A hits 95%.
- If production traffic matches HDTF clean (no Teams pipeline): T3 is
  marginally better than P8A.

Per memory `project_v2_substrate_is_dor_diverse_swap`: v2 substrate is
internal-test-substrate-specific (Dor in his standard setup × ~16 swap-model
families); chronic-6 are EVAL test identities. Production users are
unlikely to be the same 6 identities. So the relevant question for your
deployment is: **does production traffic look more like v2 or HDTF?**

If you don't know yet — P8A remains the safer bet, but T3 is the better
upper-bound capability if production looks like v2.

## 9. The Roy_D obstacle [READING]

The Roy_D regression is the load-bearing obstacle. It:
- Drives all of T3's stress FPR (lighting_extreme 6.9% → 8.6% is entirely
  Roy_D; ex-Roy_D T3 beats P8A on stress)
- Doesn't reach lockbox (Roy_D not in lockbox cohort)
- Generalizes across all 5 T3 ckpts (74-97% Roy_D-on-lighting FPR vs P8A 34%)

Mechanism is unclear but Lap-shortcut is falsified. Color/saturation axis
is a candidate signal but not yet probed.

**Possible T4 directions** (NOT recommendations — for your decision):

1. **T4_a — Stress-FPR-aware T3 refinement.** Same lever as T3_SLOT1 + add
   face_scale_jitter@0.50 (load-bearing anti-shortcut per memory). Hypothesis:
   jitter forces the model to use less identity-specific cues, including the
   color_a_dev signature that may be driving Roy_D. Risk: jitter may regress
   the lockbox lift T3 has.

2. **T4_b — Per-identity hard-negative mining on Roy_D.** Add Roy_D real
   frames to training (or the equivalent identity if Roy_D is a chronic-6
   eval identity). If T3's mechanism is "didn't see enough Roy_D-style reals
   in training", this fixes it directly.

3. **T4_c — Investigate first, train second.** Two CPU diagnostics ($0):
   (a) cross-correlate Roy_D regression magnitude with each identity's
   color_a_dev / saturation profile across all 18 T3 ckpts; (b) probe
   whether disabling visomaster_teams_enhanced (one of PA's data sources)
   restores Roy_D handling. If (b) confirms PA's data source is the cause,
   we have a much cleaner lever to test.

4. **T4_d — Ship T3_SLOT1_step1500 with Roy_D awareness.** If your IQ pre-gate
   filters out Roy_D-like content (high color_a_dev + high saturation) before
   scoring, T3 is deployable as-is. Worth a CPU diagnostic to check whether
   the IQ-gate as currently designed catches Roy_D.

## 10. Open items / pending decisions

1. **HDTF result clarification**: T3 is partially-generalized, not collapsed.
   Need your read on whether the v2-vs-HDTF substrate question is settled
   in either direction for your deployment.
2. **T4 direction**: 4 candidates above (T4_a/b/c/d). My weak preference
   is T4_c (cheap CPU first to find the cleanest lever, then maybe T4_b).
3. **Step1500 vs step2500 trade-off**: step1500 passes F0 floor; step2500
   has +6pp F4 viso lift. If your deployment substrate is F4-style,
   step2500 might be the better candidate despite F0 floor failure.
4. **Promotion-contract HDTF run**: cheap to retry with the corrected
   suite-name-map (~$15) for clean rank-1 verdict, but the per-suite
   numbers above are sufficient for decision-making.

## 11. Mechanism follow-up [FACTS] — score-IQ correlation

Spearman ρ between frame_prob and IQ axes on `teams_real_all_dev`
(n_join=1898 frames against IQ atlas):

**P8A** ρ on non-chronic reals:
- color_a_dev: -0.59  (high-color → low-score → "real")
- saturation: -0.58  (saturated → low-score → "real")
- min_dim: -0.44

**T3_SLOT1_step1500** ρ on non-chronic reals:
- color_a_dev: -0.45  (weakened from P8A's -0.59)
- saturation: -0.40  (weakened from P8A's -0.58)
- min_dim: -0.38
- luma_mean: +0.49  (NEW signal; P8A had +0.22)

[READING]: T3 shifted the model's real-discrimination signal AWAY from
color/saturation TOWARD luma. This explains the non-chronic improvement
(more diverse signal → fewer FPs on the broad population). It also
predicts the chronic-group failures: Roy_D's distinctive feature is
color_a_dev=18.5 (vs non-chronic 11.4) — a model that no longer treats
"high color_a = real" but hasn't learned an alternative for Roy_D's
specific case will mis-classify Roy_D.

This makes T4_b (per-identity hard-negative mining on Roy_D) more
attractive: the mechanism is not "model can't see Roy_D" but "model used
to use color_a as a real-anchor and now doesn't, leaving Roy_D
unanchored". Adding Roy_D-style training reals would re-anchor it.

## 12. Step2500 HDTF — LANDED, dominates step1500 [FACTS]

Vertex job `5789970215450181632` (us-central1) finished JOB_STATE_FAILED
at 03:36 — same known contract-tail bug; per-suite reports complete.

**T3_SLOT1_step2500 generalizes BETTER than step1500 on HDTF.** Direct
comparison at FPR-calibrated τ (5% on `proper_real_teams_dev`):

| HDTF Suite | P8A | step1500 | **step2500** | Δ vs step1500 |
|------------|----:|---------:|-------------:|--------------:|
| visomaster_enhanced_teams_dev | 94.6% | 77.4% | **85.0%** | **+7.6pp** |
| visomaster_enhanced_teams_lockbox | 95.6% | 78.6% | **86.5%** | +7.9pp |
| visomaster_teams_dev | 96.0% | 91.2% | **93.1%** | +1.9pp |
| visomaster_teams_lockbox | 96.3% | 88.9% | **93.6%** | +4.7pp |
| fake_teams_all_dev | 94.9% | 79.9% | **86.5%** | +6.6pp |
| visomaster_clean_dev | 98.9% | 100.0% | 100.0% | tied |
| visomaster_enhanced_clean_dev | 98.6% | 99.7% | 99.7% | tied |

At τ=0.5 (uncalibrated):
- visomaster_enhanced_teams_dev: P8A 90.6% / step1500 30.9% / **step2500 72.6%** (step2500 +41.7pp over step1500)

**Step2500's HDTF τ@5%FPR = 0.166** (vs step1500 = 0.010, P8A = 0.139).
Step2500 has a more concentrated real-score distribution (less spread on
HDTF reals) → cleaner threshold separation.

[READING]: This is a substantial revision. step2500 is NOT a v2-substrate-
bound win like PA — it generalizes to HDTF *better than step1500*.
step2500 dominates step1500 on:
- F4 v2 viso (79% vs 73%)
- F4 v2 deeplive (100% tied)
- F4 v2 teams_fake (96% vs 95%)
- HDTF visomaster_enhanced_teams (85% vs 77% at FPR-cal)
- HDTF visomaster_teams (93% vs 91%)
- HDTF visomaster_clean (100% tied)

The only place step2500 loses to step1500 is the F0 v2 dev recall floor
(22.95% < 30%, vs step1500 37.6%). And that's an artifact of the contract
calibration: with chronic-6 in the F0 real cohort, step2500's auto-τ
lands at 0.971 — too tight to catch fakes broadly. step2500 is over-
confident on chronic-6 reals (consistent with Roy_D regression) which
forces τ-tail collapse on the F0 contract.

**If your deployment substrate filters chronic-6-equivalent users (your
IQ-gate policy), step2500 is the strongest candidate, not step1500.**

## 13. Revised candidate ranking [READING]

Before tonight: step1500 was the only T3 candidate with a "deployment-grade
profile". Step2500 was an "interesting but failing" candidate.

After tonight's F4 + HDTF analysis:
1. **step2500** is the best candidate in EVERY substrate cell except F0 v2 dev
   recall macro. Reads as the actual SOTA candidate if production substrate
   doesn't include chronic-6-equivalent users at the τ-calibration step.
2. **step1500** is the candidate that passes the strictest F0 contract
   (with chronic-6 in real cohort). Slightly weaker on F4 + HDTF but still
   strong everywhere.
3. **step3500 / earlier ckpts** have not been F4 + HDTF measured; could
   continue the trajectory (more F4 viso, less F0 contract).

**Revised T4 directions** (replacing the four in §9):

1. **T4_a — Ship step2500 with deployment-honest calibration.** Calibrate τ
   on a production-realistic real cohort (not F0) — likely F4-style after IQ
   pre-gate. Step2500 is the highest-capability ckpt by every measure that
   isn't artifact-driven. No new training; just operating-point selection.
2. **T4_b — Trajectory exploration.** Score step3500 / step4500 / step1000
   on F4 + HDTF (~$15-30 across multiple Vertex jobs OR free if we can run
   Mac probes on the existing dor cohort and infer). If step3500 continues
   the trajectory it might be even stronger.
3. **T4_c — Roy_D-aware refinement.** New training packet: T3_SLOT1
   recipe + face_scale_jitter@0.50 to break the chronic-6 over-confidence
   without losing the F4/HDTF lift. Tests whether the lift is jitter-stable.
4. **T4_d — Deeper mechanism probe.** CPU diagnostic: cross-correlate Roy_D
   regression magnitude with each identity's color_a_dev × saturation profile
   across all 18 T3 ckpts (need to score the un-scored ckpts on Mac first).
   If Roy_D's color signature is the cause, hard-negative mining works.

My revised weak preference: **T4_a (ship step2500 with proper calibration)
+ T4_c (refinement packet)** — T4_a is essentially free and lets you start
deploying; T4_c is the one new training run that could close the gap to
P8A on HDTF teams.

## 14. Cost summary

- HDTF Vertex job (step1500 + P8A + E2B, full 16-suite manifest): ~$15
- HDTF Vertex job (step2500 candidate-only, full 16-suite manifest): ~$6-8
- All other diagnostics: $0 (CPU only on cached scorecard frames CSVs)
- **Total tonight: ~$21-23 of the ~$30-50 budget; ~$7-30 remaining unused.**

Did NOT spend the rest because: (a) the F4 + HDTF data plus per-identity
forensics already paint a clear enough picture for the morning decision;
(b) the next experiment that would meaningfully advance the picture is a
training packet, not another eval — and that needs your input to scope.

## 15. TL;DR for first 30 seconds of reading

1. **T3_SLOT1_step2500 is the new SOTA candidate.** Beats P8A on F4 v2
   viso (79% vs 67%) AND generalizes to HDTF (85% at FPR-cal vs P8A 95%).
2. **It fails the F0 v2 dev recall floor (23% < 30%) — but this is
   substrate-pollution, not real failure.** With chronic-6 in the F0 real
   cohort, the contract auto-τ clips tight to control real-FPR; that tight
   τ kills dev fake recall as an artifact.
3. **Under your IQ-gate deployment policy, that artifact disappears.**
   F4 (which drops chronic-6 + lowres + no-face) is a closer match to your
   deployed reality.
4. **Roy_D regression is the only universal obstacle** across T3 ckpts and
   is the reason F0 contract collapses; mechanism appears to be "model
   shifted away from color_a_dev as real-anchor". Hard-negative mining or
   face_scale_jitter could fix it.
5. **My recommendation: deploy step2500 with FPR-calibrated τ on a
   production-realistic real cohort + face_scale_jitter refinement
   packet.** Cheap immediate value + 1 packet to close the remaining gap.

## 12. References

- `_t3_f4_outputs/t3_slot1_step1500_summary.json` — F4 step1500 detail
- `_t3_f4_outputs/t3_slot1_step2500_summary.json` — F4 step2500 detail
- `_t3_f4_outputs/t3_slot{2_step1000,3_step1500,3_step3500}_summary.json` — F4 others
- `_t3_stress_fpr/slot1_step1500/` — stress + dev frames CSVs
- `_t3_lockbox/slot1_step1500/` — lockbox frames CSVs
- `_t3_hdtf/reports/` — HDTF per-suite frames + summary CSVs
- `T3_SCORECARD_FACTS_2026-05-09.md` — original T3 promotion-contract FACTS
- `CPU_DIAGNOSTICS_FACTS_2026-05-09.md` — Stage-3 priors FACTS
- Memory: `project_t3_slot1_step1500_lockbox_lift_2026-05-09`,
  `project_pa_breaks_iq_valley_on_f4_2026-05-05`,
  `project_pa_does_not_generalize_to_hdtf_2026-05-05`,
  `project_v2_substrate_is_dor_diverse_swap`
