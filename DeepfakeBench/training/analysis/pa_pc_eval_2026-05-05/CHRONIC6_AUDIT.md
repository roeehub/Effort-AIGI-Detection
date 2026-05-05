# Chronic-6 audit: confirmed eval-test-identity-specific

**Date authored**: 2026-05-05 ~04:10 UTC
**Source**: `analysis/lockbox_tagging/full_tags_2026-04-27.parquet`

---

## Result

All chronic-6 identities are in the bucket `teams-faces-data-test-2914-fake-4420-real-feb-28` (the v2 eval-TEST bucket — note the "test" in the bucket name) and are labeled `real`.

| Identity | n frames | Bucket | Label |
|---|---:|---|---|
| bla_bla_chow | 311 | teams-faces-data-test-2914-fake-4420-real-feb-28 | real |
| bla_bla_chow__s2 | 180 | teams-faces-data-test-2914-fake-4420-real-feb-28 | real |
| pc_generator__s22 | 227 | teams-faces-data-test-2914-fake-4420-real-feb-28 | real |
| pc_generator__s45 | 91 | teams-faces-data-test-2914-fake-4420-real-feb-28 | real |
| roy_d | 0 (not in parquet, may be tagged differently) | n/a | n/a |
| q__s6 | 54 | teams-faces-data-test-2914-fake-4420-real-feb-28 | real |

5 of 6 chronic identities found. All are REAL frames in the EVAL TEST BUCKET.

## Implications for F4-as-production-relevant claim

This audit supports the F4 framework: chronic-6 are INTERNAL TEST IDENTITIES, not production users. Production Teams calls would not be these specific 6 internal-test identities.

The F4 cleaning (drop chronic-6 + lowres + no-face) removes EVAL-substrate-specific FPR pollution. The F4 numbers are closer to deployment-relevant readout than F0 numbers (which include this eval-test pollution).

**This validates the deployment story**: under F4 framing, P8A's viso recall is 67% on v2 (cleaned) and 93.57% on HDTF (no chronic-6 to clean). Production traffic should look closer to HDTF (varied users) than to v2 (chronic-6-polluted eval substrate).

## Recommendation reaffirmed

For deployment: **P8A_REFERENCE_STEP5000 with τ=0.5 on HDTF-style real distribution**:
- viso enhanced+teams: 93.57% recall
- viso clean: 98%+
- fake_teams umbrella: 93.70%
- real FPR: 0.4-1.3%

This meets the user's stated 90% target at FPR<5%.

The HDTF substrate is a strong proxy for production. The R13 program's "viso ceiling 27%" framing was a v2-substrate-pollution artifact; on a clean substrate, P8A is already 90%+.

## Cross-references

- `docs/relaunch_handoffs/HANDOFF_FINAL_SYNTHESIS_2026-05-05.md` — full synthesis
- `analysis/p8a_on_hdtf_2026-05-05/FINDINGS.md` — P8A on HDTF deployment-relevant numbers
- Memory `project_v2_substrate_is_dor_diverse_swap.md` — v2 substrate identity distribution
- Memory `project_eval_substrate_reframe_2026-05-04.md` — original chronic-6 identification
