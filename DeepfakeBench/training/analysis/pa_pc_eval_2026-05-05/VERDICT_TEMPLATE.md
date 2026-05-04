# PA + PC F0/F4 verdict (template — to be filled in when results land)

**Date authored**: 2026-05-05 (template, results pending)

---

## Status

- P8A-on-HDTF (`4232735281465262080`): _PENDING_
- PA+PC F0 contract scorecard (`7756239039929253888`): _PENDING_

## Headline (fill in)

[ ] (a) Data-axis lever IS dispositive — PA/PC materially beats E2B on viso recall under deployment policy
[ ] (b) Data-axis lever as cleanly tested still does not lift — memory amends to "pulled three times"
[ ] Mixed / other (describe)

## Numerical results (fill in from `f0_at_tau_0p5.json`, `f4_summary.json`, `psubstrate_summary.json`)

### F0 at FPR=10% target (`teams_real_all_dev`)

| Ckpt | viso recall | deeplive recall | teams_fake recall | lockbox_fake recall |
|---|---:|---:|---:|---:|
| P8A_REFERENCE_STEP5000 | _ | _ | _ | _ |
| E2B_TOP_N_STEP3200 | _ | _ | _ | _ |
| PA_TOP_N_STEP5600 | _ | _ | _ | _ |
| PA_TOP_N_STEP3800 | _ | _ | _ | _ |
| PA_PERIODIC_STEP5000 | _ | _ | _ | _ |
| PC_TOP_N_STEP7400 | _ | _ | _ | _ |
| PC_TOP_N_STEP5400 | _ | _ | _ | _ |
| PC_PERIODIC_STEP5000 | _ | _ | _ | _ |

### F4 at FPR=10% target (chronic-6-cleaned)

| Ckpt | viso recall | deeplive recall | teams_fake recall | F0→F4 lift on viso |
|---|---:|---:|---:|---:|
| P8A | 26.91% (ref) | 42.39% (ref) | 69.89% (ref) | +40.18pp |
| E2B_3200 | 8.36% (ref) | 93.94% (ref) | 79.40% (ref) | +22.55pp |
| PA_TOP_N_STEP5600 | _ | _ | _ | _ |
| PA_TOP_N_STEP3800 | _ | _ | _ | _ |
| PA_PERIODIC_STEP5000 | _ | _ | _ | _ |
| PC_TOP_N_STEP7400 | _ | _ | _ | _ |
| PC_TOP_N_STEP5400 | _ | _ | _ | _ |
| PC_PERIODIC_STEP5000 | _ | _ | _ | _ |

### Per-substrate τ (single deployable τ; worst-substrate FPR ceiling)

| Ckpt | strict (5%) viso | moderate (10%) viso | loose (20%) viso |
|---|---:|---:|---:|
| P8A (ref) | 0.18% | 1.64% | 20.91% |
| PA_TOP_N_STEP5600 | _ | _ | _ |
| PC_TOP_N_STEP7400 | _ | _ | _ |

## Comparison vs predictions (`PRE_LANDING_PREDICTIONS.md`)

For each prediction: PASS / SURPRISE / WRONG

| Prediction | Result | Match? |
|---|---|:-:|
| PA F0 viso 5-15% | _ | _ |
| PA F4 viso 25-45% | _ | _ |
| PC F0 viso ≈ PA F0 ± 3pp | _ | _ |
| PC F4 viso ≈ PA F4 ± 5pp | _ | _ |
| PA F0 deeplive 75-90% | _ | _ |
| PA inherits E2B's IQ profile | _ | _ |
| Lockbox FPR drift at dev-cal τ | _ | _ |

## Comparison vs IQ-valley framing (`IQ_VALLEY_FINDING.md`)

- Predicted: PA's caught viso fakes overlap heavily with E2B's caught set (Cohorts A, W, X, Y).
- Observed: _

If PA catches Cohort B/C/D frames where E2B doesn't → IQ-valley framing extended (data lever broadens IQ sweet spot).
If PA catches same frames as E2B → IQ-valley framing confirmed (FT-base shift doesn't change IQ sweet spot).

## Comparison vs codec_hedge prior (`PRIOR_CODEC_AUG_CONTEXT.md`)

- Prior: codec aug had slight NEGATIVE effect on viso (C1=22.7% vs C3=21.5%, FT-from-P8A).
- Observed PC vs PA on viso: _

## Hard-fact deltas

(Fill in after computing.)

- PA viso F0 vs E2B viso F0: Δ = _
- PA viso F4 vs E2B viso F4: Δ = _
- PC viso F0 vs PA viso F0: Δ = _ (codec lever effect)
- PA deeplive F0 vs E2B deeplive F0: Δ = _ (FT drift)

## Verdict justification

[Fill in: which prediction was confirmed, which was surprising, what the verdict is, why.]

## Memory updates

- [ ] If (b): amend `project_data_axis_lever_pulled_twice_no_lift` to "pulled three times"
- [ ] If verdict differs from predictions: update `project_data_axis_clean_retest_packet_a_c_2026-05-04`
- [ ] If IQ-valley framing confirmed: write new memory `project_iq_valley_v2_substrate_2026-05-05`
- [ ] If codec lever shows surprise effect: write memory + amend `project_data_axis_lever_pulled_twice_no_lift`

## Thread updates

- `viso_bucket_gap.md` — extend with "2026-05-05 morning update — PA/PC verdict" subsection
- `processing_signature_shortcut.md` — extend if IQ-valley finding is confirmed/refuted
- In-progress loop `data-axis-clean-single-lever-retest-in-progress` — flip to `resolved`

## Next-packet recommendation

(Fill in based on verdict + remaining $30 GPU budget + AGENT_GUIDE Rule 3 CPU-first.)

If (b): _
If (a): _

## Open questions for follow-up

(Fill in items for the next agent or session.)
