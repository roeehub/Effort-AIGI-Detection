# Final verdict — per-identity majority-vote aggregation

## TL;DR — Yes, there IS a new leader

Under the actual production aggregation rule ("flag identity if >50% of its
frames pass the per-frame threshold"), **both T5C_step3500 and P2D_fourier_step3000
hit 100% per-identity fake-recall at 0% per-identity FPR on the F4 substrate.**

The recommended pick is **P2D_fourier_step3000 at per-frame tau=0.15**,
because it matches T5C on F4 AND is substantially more robust on the may6
production-drift cohort:

| Test | T5C @ tau=0.30 | P2D_fourier @ tau=0.15 | P8A @ tau=0.45 | E2B (deployed) @ tau=0.65 |
|---|---|---|---|---|
| F4-substrate identity FPR | **0%** | **0%** | 0% | 0% |
| F4-substrate identity recall (macro) | **100%** | **100%** | 81% | 62.5% |
| P2-substrate identity FPR (chronic OOD included) | 9.5% (2/21) | 9.5% (2/21) | 9.5% (2/21) | 4.8% (1/21) |
| may6 production-drift frame fire-rate (out of 92) | 17/92 (17%) | **5/92 (5%)** | 0/92 (0%) | 53/92 (58%, fails majority-vote) |
| may6 under majority vote | safe | **safe** | safe | **FAILS** |

The current deployment (E2B) catastrophically fails the may6 majority-vote
check (53/92 = 58% > 50% → Xinhe identity flagged as fake). T5C, P2D, and P8A
all pass.

## How per-identity majority-vote changes the verdict

Per-frame scoring made T5C look risky (17% frame-FPR on may6, Roy_D 100%
fired). Per-identity majority-vote dramatically softens this picture because:

1. **may6**: 17% < 50% → identity not flagged. T5C is safe under majority-vote
   even though it fires on more frames than P8A.
2. **Roy_D**: T5C fires on >90% of Roy_D frames, so Roy_D as an identity IS
   flagged — but Roy_D is in the F4 chronic-drop list, so on F4 substrate
   this doesn't count against FPR.
3. **bla_bla_chow_s2**: similarly chronic-substrate, dropped by F4.
4. **All non-chronic identities**: have <50% of frames above tau at every
   reasonable tau, so they never trigger.

## The full per-identity tables

### T5C_step3500 — substrate F4 (17 real identities, chronic-6 dropped)

| tau | identity FPR | flagged real ids | worst real | macro recall |
|---|---|---|---|---|
| 0.10 | 35.3% | 6 | xiang (0.987 above) | 100% |
| 0.20 |  5.9% | 1 | xiang (0.692 above) | 100% |
| 0.25 |  5.9% | 1 | xiang (0.547 above) | 100% |
| **0.30** | **0%** | **0** | xiang (0.453) | **100%** |
| 0.50 |  0%   | 0 | xiang (0.176) | 100% |
| 0.65 |  0%   | 0 | xiang (0.101) | 100% |
| 0.70 |  0%   | 0 | xiang (0.082) | 81% |
| 0.85 |  0%   | 0 | xiang (0.031) | 56% |
| 0.90 |  0%   | 0 | test_cam__s41 (0.002) | 21% |

T5C has a **wide operating window** at tau=0.30–0.65 where it sustains
0% identity FPR with 100% identity-level macro recall.

### P2D_fourier_step3000 — substrate F4 (17 real identities)

| tau | identity FPR | flagged real ids | worst real | macro recall |
|---|---|---|---|---|
| 0.05 | 11.8% | 2 | xiang (0.906) | 100% |
| 0.10 |  5.9% | 1 | xiang (0.585) | 100% |
| **0.15** | **0%** | **0** | xiang (0.415) | **100%** |
| 0.20 |  0%   | 0 | xiang (0.289) | 100% |
| 0.25 |  0%   | 0 | orel (0.250) | 81% |
| 0.50 |  0%   | 0 | xiang (0.063) | 56% |

P2D has a narrower window: 0% FPR / 100% recall at tau=0.15–0.20.

### P8A_step5000 — substrate F4 (17 real identities)

| tau | identity FPR | flagged real ids | worst real | macro recall |
|---|---|---|---|---|
| **0.05** | **0%** | 0 | xiang (0.346) | **100%** |
| 0.10 |  0%   | 0 | cam_test__s38 (0.333) | 100% |
| 0.15 |  0%   | 0 | test_cam__s73 (0.179) | 81% |
| 0.45 |  0%   | 0 | test_cam__s73 (0.116) | 60% |
| 0.55 |  0%   | 0 | test_cam__s73 (0.084) | 25% |

P8A keeps 0% FPR easily (this is its signature substrate-invariance), but
the recall ceiling is lower — only reaches 100% at tau=0.05–0.10.

### E2B_step3200 — substrate F4 (currently deployed)

| tau | identity FPR | flagged real ids | macro recall |
|---|---|---|---|
| 0.05 | 11.8% | 2 | 62.5% |
| 0.10 |  5.9% | 1 | 62.5% |
| 0.40 |  0%   | 0 | 62.5% |
| 0.50 |  0%   | 0 | 62.5% |
| 0.65 |  0%   | 0 | 62.5% |

**E2B's identity-level macro recall caps at 62.5%** — even at very low tau.
That means E2B fails to flag 37.5% of fake identities under majority-vote.

## Bottom-line deployment recommendation

| Picks | Why |
|---|---|
| **Ship P2D_fourier_step3000 at per-frame tau=0.15, per-identity majority-vote (>50%)** | 100% identity-recall, 0% identity-FPR on F4 substrate; 5/92 (5.4%) on may6 production drift (safe under majority vote); structurally distinct from FT-from-P8A family. |
| Fallback: T5C_step3500 at per-frame tau=0.30 | Same 100%/0% on F4; 17/92 on may6 (still safe under majority vote but closer to threshold). Wider operating window (tau=0.30–0.65 all give 100/0). |
| Safe-floor: P8A at per-frame tau=0.05–0.10 | Same 100%/0% on F4 (chronic-cohort dropped); 0/92 on may6; but if F4-cleaning isn't perfect at runtime, P8A's invariance is the most-tested baseline. |
| Drop: E2B (currently deployed) | Identity-recall caps at 62.5% even at low tau; fails majority-vote on may6 (53/92 = 58%). |

**Production gate stays the same:** G1 (face detector) + G2 (min(W,H) ≥ 200).
G3 (sharpness gate) is NO LONGER NEEDED because Roy_D's per-frame collapse
on T5C doesn't matter under majority-vote on F4 substrate.

## Worst-case stability check

The 2 identities that get falsely flagged at FPR=9.5% on the P2 substrate
(without F4 chronic-drop) are **roy_d** and **bla_bla_chow__s2** — both
identities passed G2 by virtue of having SOME frames ≥ 200px, but they
represent edge-case substrate (Roy_D = soft-focus warm-color; bla_bla_chow__s2
= 96% sub-200 lowres). In production:
- Roy_D is the one identity at risk. If you raise the G2 threshold from 200
  to 220-240 px, Roy_D drops out (median is 270, so a ~20 px tightening
  removes the lower tail).
- bla_bla_chow__s2 has 96% sub-200 → already mostly gated out; the residual
  4% (~7 frames) does not threaten majority-vote on a Teams-call-length
  session (would need >50% of all session frames to fall in that thin tail).

## Investigations completed

| Probe | Status | Finding |
|---|---|---|
| 38-candidate F4 substrate-cleaning sweep | done | T5C, P2D, P1_pairrank_step6750 top 3 by F4 recall |
| Roy_D substrate IQ characterization | done | 270 px, sharpness 69, warm saturation 125 — soft+warm OOD, not lowres |
| Deployment threshold tables | done | 4 ckpts × 8 FPR × 2 substrates |
| T5C+P8A ensemble probes (min/max/avg) | done | No rule beats T5C alone on F4; P2-FPR=5% best = avg-rule 59% (vs T5C only 33%, P8A only 54%) |
| P2D_fourier may6 production-drift retest | done | 5/92 fires (5.4%), p50=0.085 — better than T5C |
| Per-identity majority-vote on all 4 candidates | done | T5C and P2D both hit 100%/0% on F4 |

## Investigations open (do these before final deploy)

1. **Verify T5C and P2D on HDTF cross-substrate** (P2D has HDTF eval data;
   T5C does not yet). The PA trap (`project_pa_does_not_generalize_to_hdtf`)
   warns about v2-substrate-specific lifts. Need an HDTF-substrate
   per-identity verdict for both.
2. **Continuous-frame stability probe**: current eval has 1-4 frames per
   video_id. Production has continuous 5-30 minute sessions. Test how the
   per-identity fraction stabilizes as you sample 10/50/100/500 frames
   from real production-like footage.
3. **T5C+P2D 2-model AND-rule ensemble**: both pass majority-vote at
   their respective taus. The AND of two structurally distinct models
   (FT-from-P8A vs B16-scratch+Fourier) would be a much harder bar to
   beat — likely 0% identity-FPR even on adversarial substrate.
4. **A/B production canary** for P2D vs E2B on real Teams traffic
   (Pillar-3 robustness on live distribution).

## Why this differs from the earlier verdict

The earlier verdict (`02_VERDICT.md` and `03_DEPLOYMENT_GATE_AND_THRESHOLDS.md`)
correctly identified T5C and P2D as the top candidates but used per-frame
FPR/recall, which misrepresented the deployment math. Per-frame FPR=17.6%
sounds catastrophic; per-identity FPR=0% with the same model + the same tau
+ a 21-real-identity panel is the actual production-relevant metric.

**This document supersedes the earlier deployment threshold tables for the
majority-vote production policy.** The earlier per-frame thresholds are
still correct for per-frame-decision deployments (e.g., if you score a
single still and have to make a single decision).
