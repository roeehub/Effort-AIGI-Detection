# Pair-loss effect verification — visomaster enhanced macro dev (E2B + P8A frozen features)
_Date: 2026-05-05_

## Verdict: **LOW** — skip the pair-loss packet

- Q1 (feature gap): **MODERATE** — paired cosine 0.878, only 0.19 above random within-subtype
- Q2 (score↔feature correlation): **LOW** — |r| = 0.089 between P8A feat distance and E2B |score gap|
- Q3 (upper-bound recall lift): **LOW** — only 3/275 pairs (1.1%) match the "missed-teams" cohort definition

### Sign-of-effect surprise (load-bearing)

The pair-loss premise — that "teams transport hurts E2B on viso fakes, so aligning teams to raw representations would close the gap" — is **not supported by the data**.

For E2B (the model the packet would target):
- mean raw_score = **0.086**, mean teams_score = **0.172** (Wilcoxon p = 0.0019)
- pairs where raw caught & teams missed (the cohort the packet would help): **3 / 275** (1.1%)
- pairs where teams caught & raw missed (the cohort that would be HURT by alignment toward raw): **29 / 275** (10.5%)
- pairs where both caught: 7 / 275; pairs where both missed: 236 / 275 (85.8%)

Teams transport actually slightly HELPS E2B on viso. P8A shows the opposite direction (raw catches more than teams: Wilcoxon p = 3e-24), but P8A is not the FT base for the proposed packet, and the original premise is grounded in E2B behavior.

The 85.8% "both missed" cohort is the dominant failure mode — these are pairs where neither the clean nor the transported version is caught. Pair loss has no fulcrum here: there is no within-pair asymmetry to align away. The bottleneck is representation, not transport invariance.

---
## Setup
- N pairs (seq_id present in both raw & teams subtypes, both in npz and E2B report): **275**
- npz raw seqs: 275, teams seqs: 275; total npz frames: 550
- Features: P8A frozen penultimate (512-d) from `visomaster_enhanced_macro_dev_p8a_features.npz`
- Scores: E2B step3200 (`visomaster_enhanced_macro_dev_e2b_top_n_step3200_frames_report.csv`)
- Pair extraction regex: `visomaster_enhanced_(raw|teams)__frame_NNNNNN_seqMMMM\.png`
- Caveat: P8A features are a PROXY for E2B features (E2B features not cached on disk)

## Q1 — Feature gap: paired (raw,teams) vs random within subtype
| stat | paired (raw,teams) | random within subtype | random cross-subtype (raw vs teams of diff seq) |
|---|---|---|---|
| mean | 0.8782 | 0.6863 | 0.6377 |
| p10 | 0.6755 | 0.1735 | 0.1143 |
| p25 | 0.8202 | 0.5319 | 0.4310 |
| p50 | 0.9417 | 0.8192 | 0.8069 |
| p75 | 0.9905 | 0.9651 | 0.9483 |
| p90 | 0.9980 | 0.9941 | 0.9862 |
| p99 | 0.9996 | 0.9999 | 0.9993 |

**Δ mean(paired − random_within) = +0.1919** (positive = paired closer than random)
**Δ mean(paired − random_cross) = +0.2405**

Interpretation thresholds: <0.75 STRONG, 0.75–0.95 MODERATE, >0.95 LOW.
→ Paired mean cosine = **0.8782** → **MODERATE**.

## Q2 — Score↔feature correlation
- Pearson r(feature_distance, |E2B_score_gap|) = **-0.0894** (p = 0.1391)
- Pearson r(feature_distance, signed E2B raw−teams) = **0.2280** (p = 0.0001366)

Interpretation thresholds: r>0.4 STRONG, 0.2–0.4 MODERATE, <0.2 LOW.
→ |r| = **-0.0894** → **LOW**.

*Caveat:* feature distance comes from P8A; score gap comes from E2B. If E2B features differ from P8A in their geometry, this correlation could under- or over-estimate the true E2B feature↔score linkage. Phase-1 audits show E2B and P8A track different shortcut axes (image quality vs identity-cluster), so r should be read as a lower bound on the within-model linkage in P8A and an unknown (probably weaker) estimator for E2B.

## Q3 — Upper-bound recall lift
- Cohort definition: pairs where E2B raw_score > 0.5 AND teams_score < 0.5
- n_cohort = **3** out of **275** pairs (1.1% of pairs)
- Mean raw_score (E2B) in cohort = **0.5970**
- Mean teams_score (E2B) in cohort = **0.1638**
- Implied per-frame score lift if perfect transfer = 0.4332
- **Upper-bound recall lift on viso (pair basis): 1.1 pp**
- Realistic lift band (30–60% transfer efficiency): **0.3–0.7 pp**

Interpretation thresholds: >30 pp STRONG, 10–30 pp MODERATE, <10 pp LOW.
→ Upper-bound = **1.1 pp** → **LOW**.

### Sensitivity to threshold choice

The 0.5 threshold is conservative for E2B because both raw and teams score distributions are skewed left (mean 0.086 / 0.172). Re-running with deployment-style operating points:

| symmetric tau | n_cohort (raw>τ & teams<τ) | mean_raw | mean_teams | upper-bound pp |
|---|---|---|---|---|
| 0.05 | 26 | 0.110 | 0.019 | 9.5 |
| 0.10 | 17 | 0.177 | 0.038 | 6.2 |
| 0.20 | 11 | 0.358 | 0.100 | 4.0 |
| 0.50 | 3 | 0.597 | 0.164 | 1.1 |

Even at the most permissive τ=0.05, the cohort is 26/275 pairs and the perfect-transfer upper bound is 9.5 pp. Realistic 30–60% transfer = **2.9–5.7 pp** — and that's against the perfect-transfer ceiling, not against the actual deployment recall denominator.

Critically, the OPPOSITE-direction cohort at the same τ (teams caught & raw missed) is 2–3× LARGER, so a symmetric pair-alignment loss would erode net recall here, not lift it.

*Caveat:* this assumes the cohort's frames CAN be lifted by representation alignment alone. If the missed-teams frames are missed because of irreversible information loss (e.g. severe sharpness collapse making the face genuinely unreadable), pair loss can't recover them. Cohort feature_distance stats: {'mean': 0.23620896340514508, 'std': 0.11398115321703281, 'p10': 0.1294931305138463, 'p25': 0.1618817533529348, 'p50': 0.2158627914180823, 'p75': 0.30036308746382395, 'p90': 0.35106326509126895, 'p99': 0.38148337166773594, 'n': 3}

## Recommendation

**Skip the pair-loss packet.** The empirical premise does not hold for E2B on this substrate:

1. The directional asymmetry the loss is designed to fix is REVERSED — E2B already scores teams-transported viso fakes ~2× higher than their raw counterparts (Wilcoxon p = 0.0019). The model is not failing because of teams transport; it is failing on viso wholesale (85.8% of pairs miss in BOTH subtypes).
2. Per-pair score correlation is already 0.62 (E2B) / 0.77 (P8A) — there is substantial implicit pair coherence even without an explicit pair loss. The headroom for an explicit alignment objective to add over what the model already does is small.
3. Even at the most permissive symmetric τ = 0.05, the perfect-transfer upper bound is 9.5 pp on viso pairs and the realistic band is ~3–6 pp. The opposite-direction "wrong way" cohort is 2–3× larger, so a naïve symmetric pair loss would likely net to **negative** recall change.

Reallocate the 6h dev + $87 GPU budget to:
- **Substrate cleaning extension (Job 14 follow-up):** drop chronic-6 + lowres + no-face → demonstrated 27→67% viso lift on P8A with no training. Zero-cost lever, established result.
- **Per-substrate τ-calibration on P8A (Job 7 follow-up):** 21 pp lockbox recall lift available from calibration alone, no retraining.
- **Out-of-stream router (Job 12):** label-free ensembles cap at 16% viso, but per-video oracle ceiling is 77% — the information IS there, the structural lever is a router, not a pair-aligned encoder.

If the user still wants to test pair loss empirically, the cheapest disconfirmation would be a 1–2 GPU-hour probe that extracts E2B features (not P8A) on the same 550 frames and re-runs Q1/Q2/Q3 on E2B's actual feature geometry — that costs ~$5 and decides whether the P8A-as-proxy assumption is hiding a real signal.
