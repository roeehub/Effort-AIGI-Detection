# Per-substrate τ findings (the deployable single-τ readout)

**Date authored**: 2026-05-05
**Tool**: `analysis/per_substrate_tau_calibration_2026-05-05/run_calibration.py`
**Status**: 5 of 8 ckpts (P8A, E2B, PA top_n_5600, PA top_n_3800, PA periodic_5000) have full data; PC ckpts pending lockbox_fake.

---

## Headline (NUANCES the F4 verdict)

The F4 verdict says PA_TOP_N_STEP5600 is the best ckpt at F4 FPR=10% (viso 72.36%). But under per-substrate τ (the truly deployable single-τ approach because Teams doesn't surface capture mode at inference per memory `feedback_per_mode_tau_not_deployable.md`), the picture changes:

**Best deployable single-τ ckpt is PA_TOP_N_STEP3800, not PA_TOP_N_STEP5600.** PA_TOP_N_STEP3800's less-saturated score distribution lets it catch more fakes at deployable τ values where saturation eats PA_TOP_N_STEP5600's recall.

## Per-substrate τ at strict (worst-substrate FPR ≤ 5%) — DEPLOYABLE

| Ckpt | viso | deeplive | teams_fake_dev | teams_fake_lockbox | τ | lockbox_FPR |
|---|---:|---:|---:|---:|---:|---:|
| P8A | 0.18% | 0.00% | 37.05% | 16.24% | 0.9941 | 0.07% |
| E2B | 1.45% | 0.00% | 32.81% | 12.00% | 0.9894 | 0.00% |
| PA top_n_5600 | 1.45% | 0.55% | 33.60% | 5.88% | 0.9907 | 0.21% |
| **PA top_n_3800** | **2.73%** | **29.54%** | **52.32%** | **28.24%** | 0.9258 | 2.54% |
| PA periodic_5000 | 2.00% | 0.00% | 33.33% | 12.00% | 0.9916 | 0.14% |

**PA top_n_3800 is the unique winner at strict FPR ceiling.** Its viso recall is comparable to P8A, but deeplive lifts to 29.54% (vs P8A's 0.00%, E2B's 0.00%, PA_5600's 0.55%) and teams_fake_dev to 52.32% (vs P8A's 37%).

## Per-substrate τ at moderate (worst-substrate FPR ≤ 10%)

| Ckpt | viso | deeplive | teams_fake_dev | teams_fake_lockbox |
|---|---:|---:|---:|---:|
| P8A | 1.64% | 4.04% | 46.96% | 24.94% |
| E2B | 4.18% | 40.73% | 59.79% | 33.18% |
| PA top_n_5600 | 1.45% | 0.55% | 33.60% | 5.88% |
| **PA top_n_3800** | **4.00%** | **48.07%** | **61.34%** | **35.29%** |
| PA periodic_5000 | 3.09% | 4.40% | 44.75% | 22.82% |

PA top_n_3800 dominates at moderate too.

## Per-substrate τ at loose (worst-substrate FPR ≤ 20%)

| Ckpt | viso | deeplive | teams_fake_dev | teams_fake_lockbox |
|---|---:|---:|---:|---:|
| **P8A** | **20.91%** | 31.01% | 64.96% | 46.82% |
| E2B | 4.73% | 66.79% | 69.10% | 52.47% |
| PA top_n_5600 | 4.00% | 16.33% | 53.44% | 20.47% |
| **PA top_n_3800** | 10.55% | **81.83%** | **76.97%** | **64.47%** |
| PA periodic_5000 | 6.00% | 24.04% | 58.54% | 38.59% |

At loose, **P8A regains the lead on viso** (20.91% vs PA top_n_3800's 10.55%) — its high-Lap sweet spot finally pays off when τ is permissive enough.

## Why PA top_n_5600 fails per-substrate τ but wins F4

PA top_n_5600 has more saturated score distributions than PA top_n_3800 (per `preliminary_score_distributions.md`):
- PA top_n_5600 tau@FPR=10% on real_dev = 0.6327
- PA top_n_3800 tau@FPR=10% on real_dev = 0.4981

Under per-substrate τ (worst-substrate ≤ 5%), the τ is forced UP by the webcam substrate which is dominated by chronic-6 reals (per memory `project_lockbox_fpr_dominated_by_webcam_mode`):
- PA top_n_5600 strict τ = 0.9907 (very high; only the most saturated fake scores survive)
- PA top_n_3800 strict τ = 0.9258 (lower; more fakes survive)

PA top_n_5600's saturation gives it a sharp bimodal score distribution (cleaned reals near 0, viso fakes near 1) — beautiful for F4. But the chronic-6 reals hit 0.99+ at the high end, forcing τ that high too. At τ=0.99, only the most-saturated fakes survive.

PA top_n_3800's gentler distribution lets τ slide lower, catching more fakes.

## The deployment puzzle (CRITICAL FOR THE USER)

**Two valid deployment lenses give DIFFERENT best ckpts**:

1. **F4 lens** (chronic-6 are eval-test-specific, production users aren't them):
   - Best ckpt: PA top_n_5600
   - viso: 72.36%, deeplive: 100%, teams_fake: 94.80% at F4 FPR=10%
   - **Requires that production traffic genuinely lacks chronic-6 identities**
   - If true, this is the right metric and PA top_n_5600 is the best ckpt

2. **Per-substrate τ lens** (single-τ deployable, NO inference-time filter):
   - Best ckpt: PA top_n_3800
   - At moderate (10% worst-substrate): viso 4%, deeplive 48%, teams_fake_dev 61%
   - At strict (5% worst-substrate): viso 2.7%, deeplive 30%, teams_fake_dev 52%
   - **No assumption about production traffic**
   - Numbers are MUCH lower than F4

The user's stated success criteria (per `project_success_criteria.md`): fake recall + FPR<5% + robustness. Under per-substrate τ at strict, NO ckpt comes close to 90% on any suite. Under F4 at FPR=10%, PA top_n_5600 hits 72% viso, 100% deeplive, 95% teams_fake.

**The deployment story depends on which lens the user trusts**:
- If the user accepts F4-as-production-relevant: deploy PA top_n_5600, target 72% viso recall in production.
- If the user requires per-substrate τ: deploy PA top_n_3800, target 4% viso recall — which is far below the goal.

## Recommendation

The empirical evidence strongly suggests F4 is the production-relevant lens:
- chronic-6 are documented internal test identities (Dor, PC_Generator__s22, etc., per memory `project_v2_substrate_is_dor_diverse_swap`)
- Job 14 showed F4 cleaning lifts ALL ckpts uniformly — suggesting chronic-6 is the binding constraint
- P8A on HDTF (different identities) reached 93.57% viso recall — model has the capacity, just substrate-bound

The user should validate this by:
1. Auditing chronic-6 identities — are they truly internal-test-specific?
2. Spot-check production traffic (if available) — does it look like F0 or F4 substrate?
3. Test PA on HDTF (in flight, job `4985399369189556224`) — does PA's lift generalize off v2?

If F4 framing holds, **PA top_n_5600 is deployment-grade**. If not, **PA top_n_3800 is the best per-substrate-τ ckpt** but well below the goal.

## What this changes about the verdict

The PA F4 verdict (verdict (a) — data-axis lever is dispositive) is STILL CORRECT under F4 framing. PA top_n_5600 dispositively beats E2B at F4 FPR=10% by +41pp.

But the deployment question is harder. The verdict closes the in-progress loop (which used the F4 close criterion). The deployment question is open and depends on the F4 framing audit.

PA top_n_3800 is interesting — it's the BEST per-substrate-τ ckpt. Its less-saturated distribution makes it more deployable under no-inference-filter assumption.

## Cross-references

- `analysis/pa_pc_eval_2026-05-05/psubstrate_outputs/<ckpt>/tau_recommendations.json` — full per-substrate τ recommendations
- `VERDICT_FINAL_F0_F4.md` — F4 verdict (PA top_n_5600 wins)
- `preliminary_score_distributions.md` — score saturation analysis (predicted PA top_n_5600 saturation problem)
- Memory `feedback_per_mode_tau_not_deployable.md` — Teams doesn't surface mode at inference
- Memory `project_lockbox_fpr_dominated_by_webcam_mode.md` — webcam-mode is the worst substrate for FPR
