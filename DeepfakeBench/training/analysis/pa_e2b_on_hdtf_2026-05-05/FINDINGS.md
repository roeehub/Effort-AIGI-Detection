# PA + E2B on HDTF substrate — full 16-suite findings

**Date authored**: 2026-05-05 ~03:55 UTC
**Vertex job**: `4985399369189556224` (us-east1, image 1.3.258), JOB_STATE_RUNNING (validation phase complete; will end FAILED at contract-scoring tail like Job B + P8A-on-HDTF)
**Output GCS**: `gs://training-job-outputs/test_results/pa_e2b_on_hdtf_2026-05-05/pa-e2b-on-hdtf-2026-05-05/`
**Refines**: the CRITICAL_WALKBACK doc — adds nuance that PA's failure on HDTF is SPECIFIC to teams-transported fakes; PA does fine on clean variants.

---

## Headline (NUANCED walkback)

**PA's HDTF failure is restricted to teams-transported fakes.** On clean variants, PA performs comparably to or BETTER than P8A.

The earlier "PA collapses on HDTF" framing was correct on the headline cell (`proper_visomaster_enhanced_teams_dev` 7.87%) but missed that:
- PA on viso clean variants: 98%+ (matches P8A's 98%+)
- PA on fake clean all (mixed methods, no teams transport): 98%+ (vs E2B's 82-84%)
- PA on viso teams (no enhancement): 55% (P8A 94%, big gap)
- PA on viso enhanced+teams: 7.87% (P8A 93.57%, biggest gap)

PA learned to handle ENHANCEMENT, but not in combination with HDTF-style TEAMS TRANSPORT.

## Full table

| Suite | n | P8A on HDTF | **PA top_n_5600 on HDTF** | E2B top_n_3200 on HDTF |
|---|---:|---:|---:|---:|
| proper_real_clean_dev FPR | 1443 | 0.42% | 0.69% | 0.28% |
| proper_real_clean_lockbox FPR | 382 | 0.00% | 0.52% | 0.00% |
| proper_real_teams_dev FPR | 1444 | 0.97% | 0.07% | 0.21% |
| proper_real_teams_lockbox FPR | 382 | 1.31% | 0.00% | 0.00% |
| proper_visomaster_clean_dev | 262 | 98.47% | **98.85%** | 96.56% |
| proper_visomaster_clean_lockbox | 80 | 98.75% | **98.75%** | 92.50% |
| proper_visomaster_enhanced_clean_dev | 1180 | 98.22% | **97.88%** | 79.24% |
| proper_visomaster_enhanced_clean_lockbox | 302 | 98.01% | **98.68%** | 81.79% |
| proper_visomaster_teams_dev | 262 | **94.27%** | 54.96% | 50.38% |
| proper_visomaster_teams_lockbox | 80 | **95.00%** | 37.50% | 36.25% |
| **proper_visomaster_enhanced_teams_dev** | 1182 | **93.57%** | **7.87%** | 6.94% |
| proper_visomaster_enhanced_teams_lockbox | 302 | **95.03%** | 6.95% | 9.27% |
| proper_fake_clean_all_dev | 1442 | 98.27% | **98.06%** | 82.39% |
| proper_fake_clean_all_lockbox | 382 | 98.17% | **98.69%** | 84.03% |
| proper_fake_teams_all_dev | 1444 | **93.70%** | 16.41% | 14.82% |
| proper_fake_teams_all_lockbox | 382 | **95.03%** | 13.35% | 14.92% |

## Pattern decomposition: where does PA fail?

PA's FAILURE PROFILE on HDTF (relative to P8A):

- **Viso clean variants** (no teams transport): PA matches/beats P8A (98%+).
- **Viso teams (no enhancement)**: PA 55% vs P8A 94% — significant regression.
- **Viso enhanced+teams**: PA 7.87% vs P8A 93.57% — catastrophic regression.
- **Other-method clean fakes**: PA matches P8A (98%+).
- **Other-method teams fakes**: PA 16% vs P8A 94% — catastrophic regression.

The pattern: **PA fails specifically on TEAMS-TRANSPORTED fakes when the identity isn't Dor**.

PA's training enabled `visomaster_teams_enhanced` (the conjunction source). Per `data/sources/visomaster.py`, this source samples from `visomaster-enhanced-face-cropped-v2-companion` style buckets — which are Dor-dominant. PA learned "Dor-after-teams-transport = fake." For non-Dor identities going through teams transport, PA's learned pattern doesn't fire.

E2B has the SAME failure mode on HDTF teams suites (all teams-transported fakes <50% recall). This is a property of FT-from-CLIP-scratch + heavy aug, NOT specific to PA's data lever. **Both PA + E2B inherit this teams-on-HDTF blindspot from the E2B base or from CLIP-scratch training dynamics.**

## What this changes about PA's deployment story

**PA's apparent F4 v2 lift (72% viso) was on a substrate dominated by Dor-style teams-transported viso fakes.** PA's training included Dor-style data extensively. The lift is real WITHIN that distribution but doesn't generalize to non-Dor identities going through teams transport.

For production deployment:
- If production traffic = Teams calls from various users (NOT specifically Dor): PA will fail on teams-transported fakes from non-Dor users. PA UNSAFE.
- If production traffic = mostly clean uploads (no teams transport): PA performs well. PA OK.
- Mixed production scenario: P8A is the safer default since it generalizes both clean and teams-transported.

## Real-side FPR comparison on HDTF (lowercase = better)

| Suite | P8A | PA top_n_5600 | E2B |
|---|---:|---:|---:|
| proper_real_clean_dev | 0.42% | 0.69% | 0.28% |
| proper_real_clean_lockbox | 0.00% | 0.52% | 0.00% |
| proper_real_teams_dev | 0.97% | 0.07% | 0.21% |
| proper_real_teams_lockbox | 1.31% | 0.00% | 0.00% |

PA's real-side FPR on HDTF is ULTRA-LOW (0% on lockbox teams reals). This is because PA isn't recognizing HDTF teams content as fake AT ALL — it's defaulting to "real" for all HDTF teams content. The low FPR is misleading; it's not high precision, it's blanket-conservatism on teams-transported HDTF.

## Implications

1. **The "PA dispositive" verdict on F4 v2 is real BUT v2-distribution-bound.** On HDTF teams-transported, PA fails worse than the original E2B baseline did on v2.

2. **The walkback in `CRITICAL_WALKBACK_PA_DOES_NOT_GENERALIZE.md` is CORRECT** — PA's lift doesn't generalize. The nuance: PA does generalize on clean variants (98%+). Failure is specifically on teams-transported fakes.

3. **Production deployment**: P8A is the safer default. If production has any teams-transported fakes from non-Dor users, PA will fail.

4. **For future packets**: cross-substrate validation (HDTF or similar) MUST be part of close criterion. F4 v2 alone is insufficient.

## Comparison to original v2 numbers

| Suite/cell | P8A on v2 F4@10% | PA on v2 F4@10% | P8A on HDTF | PA on HDTF |
|---|---:|---:|---:|---:|
| viso enhanced+teams | 67.09% | **72.36%** | **93.57%** | **7.87%** |
| Across-substrate consistency | both substrates: P8A 67-94% | v2 high, HDTF crashed | both substrates: P8A ~90%+ | inverted: v2 high, HDTF low |

P8A is consistent across substrates (67% v2 F4 → 94% HDTF, going UP because HDTF is structurally easier). PA is INVERTED: 72% v2 F4 → 7.87% HDTF, going DOWN dramatically.

**P8A's substrate-robust behavior is the deployment-relevant property.** PA's substrate-volatile behavior is risky.

## Cross-references

- `analysis/p8a_on_hdtf_2026-05-05/FINDINGS.md` — P8A on HDTF (the comparison anchor)
- `analysis/pa_pc_eval_2026-05-05/CRITICAL_WALKBACK_PA_DOES_NOT_GENERALIZE.md` — the walkback (now refined by this doc)
- `docs/relaunch_handoffs/HANDOFF_FINAL_SYNTHESIS_2026-05-05.md` — top-level synthesis
- Memory `project_pa_does_not_generalize_to_hdtf_2026-05-05.md` — the new memory
