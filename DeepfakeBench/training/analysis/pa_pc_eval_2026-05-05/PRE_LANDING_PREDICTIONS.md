# Pre-registered predictions for PA / PC-codec F0 + F4 + per-substrate τ

**Date authored**: 2026-05-05 (before results landed)
**Purpose**: register predictions so the post-hoc reading is honest about confirmation vs surprise

---

## Theory I'm testing

Building from prior evidence (memories `project_data_axis_lever_pulled_twice_no_lift`, `project_p16_data_axis_does_not_promote_2026-04-30`, `project_image_quality_shortcut`, `project_eval_substrate_reframe_2026-05-04`, `project_e2b_breaks_deeplive_ceiling`, `project_v2_substrate_is_dor_diverse_swap`, `project_job_b_findings_universal_vs_trajectory_2026-05-04`):

**The data-axis lever (visomaster_enhanced + visomaster_teams_enhanced training) is well-known to produce high training-time recall but collapse at deployment τ on the v2 production substrate** (P16 demonstrated: 51.6% at τ=0.5 → 1% at deployment τ=2% FPR). The 550-frame v2 substrate has 364/550 unreachable frames per the 3-way union cap (33.8%) (memory `project_eval_substrate_reframe_2026-05-04`). This unreachability is most parsimoniously attributed to IQ confounds + identity skew on Dor-only substrate + chronic-6 real FPR pollution.

PA (FT-from-E2B + visomaster data lever, fw=4.0, no bundle) tests whether shifting the FT-base from P8A to E2B changes the τ-collapse pattern. PC-codec adds codec aug specifically targeting teams-transport IQ degradation — this should help robustness on viso→teams transport.

## Predictions (registered before reading data)

### F0 (full production substrate) at FPR=10% target on `teams_real_all_dev`

| Suite | E2B baseline | PA prediction | PC prediction |
|---|---|---|---|
| visomaster_enhanced_macro_dev | 8.4% | **5-15%** (likely τ-collapse pattern; lift over E2B not dispositive) | **8-20%** (codec aug helps marginally on substrate IQ) |
| deeplive_enhanced_dev | 93.9% | 75-90% (FT-from-E2B preserves much of E2B's specialization, minus drift) | 75-90% (similar) |
| teams_fake_all_dev | 79.4% | 75-85% (modest improvement from data) | 75-85% |

### F4 (chronic-6 cleaned substrate) at FPR=10% target on cleaned reals

| Suite | E2B baseline | PA prediction | PC prediction |
|---|---|---|---|
| visomaster_enhanced_macro_dev | 30.9% | **25-45%** (similar lift profile to E2B; minor data benefit) | **30-50%** (codec aug helps slightly more) |
| deeplive_enhanced_dev | 100.0% | 95-100% | 95-100% |
| teams_fake_all_dev | 87.1% | 85-90% | 85-90% |

### Per-substrate τ at strict (5% worst-substrate FPR)

| Suite | P8A baseline | PA prediction | PC prediction |
|---|---|---|---|
| viso_dev | 0.18% | 0-5% (substrate τ collapses; data lever doesn't survive) | 0-5% |
| deeplive_dev | 0.0% | 30-60% (E2B base helps) | 30-60% |
| teams_fake_dev | not in ref | 30-60% | 30-60% |

## Verdict criteria (from in-progress loop close criterion)

Per `docs/packet_retrospectives/threads/viso_bucket_gap.md` `data-axis-clean-single-lever-retest-in-progress`:

**The verdict closes EITHER as**:
- **(a) data-axis lever is dispositive**: PA/PC materially beats E2B on viso recall under deployment policy
- **(b) data-axis lever as cleanly tested still does not lift** (memory `project_data_axis_lever_pulled_twice_no_lift.md` amends to "pulled three times")

My pre-landing read: **most likely (b)**, because:
- The τ-tail collapse pattern is consistent across 3 prior P-series packets (P14_DATA_FIX, P16, S3)
- E2B's IQ shortcut is INVERTED vs P8A (memory `project_iq_gating_viability_2026-05-04`); whether this affects the τ-collapse pattern is unknown
- The 364/550 unreachable v2 frames are likely unreachable by PA/PC too (per `project_eval_substrate_reframe_2026-05-04`)

**Possible-(a) mechanism**: if PA's FT-from-E2B + fw=4.0 single-lever cleanly avoids the bundle-confound (P14_DATA_FIX) and sub-baseline weight (P16) issues, PA might survive τ-collapse where its predecessors didn't. But the prior magnitude (51% → 1% in P16) is large and any survival would have to be substantial.

**PC vs PA**: PC's codec aug is specifically calibrated to teams transport (cosine 0.87-0.99 vs actual transport per `analysis/codec_aug_verification_2026-05-05/`). This should make PC marginally better than PA on viso-via-teams transport, but probably not by much on the v2 dev substrate (which is `visomaster_enhanced_v2`, not "viso → teams transport" specifically).

## What would surprise me

1. **PA F4 viso recall > 50%**: would indicate the data lever DOES survive the τ-collapse on cleaned substrate, calling for a memory amendment.
2. **PC F0 viso recall > 30%**: would indicate codec aug is dispositive on this substrate (would update `project_data_axis_lever_pulled_twice_no_lift` to acknowledge codec lever as orthogonal).
3. **PA/PC deeplive recall < 75% F0**: would indicate FT-from-E2B drift is large enough to undo E2B's deeplive specialization.
4. **PA/PC teams_fake recall > 90% F0**: would indicate the data lever produces a strong teams-fake lift even at deployment τ — interesting if it doesn't translate to viso.

## What would NOT surprise me

1. PA looks similar to E2B on F0 (canonical τ-collapse for viso, modest lifts elsewhere).
2. PA F4 looks like E2B F4 + small lift (chronic-6 cleaning helps comparably across ckpts).
3. PC marginally better than PA on viso (codec aug as predicted).
4. Both do well on deeplive (E2B FT-base specialization preserved).
5. Lockbox real FPR drift: PA/PC at dev-calibrated τ may violate 2% lockbox FPR (canonical drift, P22 pattern).

## What this analysis should produce

Once results land:
1. Pivot table at FPR=10% (PA, PC, E2B, P8A on F0)
2. F4 pivot table at FPR=10%
3. Per-substrate τ table at strict / moderate / loose
4. Compare to predictions; flag surprises
5. Verdict on (a) vs (b)
6. If (b): what's the next move? (cheaper alternatives are F4 substrate cleaning + per-substrate τ on existing P8A; another data-axis test would be a fourth try and is hard to justify)
7. If (a): what's the mechanism? (FT-base shift? fw value? clean isolation?)

The verdict + nuance gets written to `verdict_FINDINGS.md` in this directory, plus a thread extension on `viso_bucket_gap.md`.
