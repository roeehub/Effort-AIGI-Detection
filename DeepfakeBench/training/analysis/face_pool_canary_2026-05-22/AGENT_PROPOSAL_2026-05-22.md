# Face-Pool Canary Follow-Up — Agent Proposal (OPINION-ONLY) — 2026-05-22

> Opinion-only doc. Numbers + tables in `RESULTS_FACTS_2026-05-22.md`.

## Bottom line

The face-region pool is a real lever class on the substrate-pair direction — but as a drop-in inference-side replacement for the CLS pool on the existing trained head, it gives a mixed verdict, not a clean deployment improvement.

- **FPR=10 % calibration on the 800-frame canary: +6pp lockbox recall** (0.61 → 0.67). Meaningful.
- **FPR=5 % calibration on the same canary: −4pp lockbox recall** (0.59 → 0.55). Goes the other way.
- **Chronic-identity story is split:** Roy_D (the most-cited chronic offender) drops 0.86 → 0.76 (−0.10, better), but `Q__s6` rises 0.25 → 0.62 (+0.37, worse) and `bla_bla_chow` rises 0.41 → 0.64 (+0.23, worse). 5 of 6 chronic identities show higher mean real-frame scores under face-pool — the face-pool is moving SOME substrate-conditioned over-fires down (Roy_D) but creating others.
- **Score distribution shifts dramatically:** real-frame score median goes 0.18 → 0.57 (+0.39); std collapses 0.31 → 0.16. The trained classifier head was calibrated on CLS-pool input statistics; face-pool gives uncalibrated probabilities. The contract metric's τ-recalibration on the canary's own reals partially compensates, but the overall score-rank quality is what determines recall at calibrated FPR.

## What I think the numbers mean

1. **The face-pool substrate-invariance shown in Probe 2 IS real, but it's not concentrated.** Probe 2 measured cos_pair (clean ↔ teams of same identity) and that did rise 0.87 → 0.96 under face-pool — that's a 4× tightening of the substrate gap. But that tightening is a population-average measurement. The per-chronic-identity table here shows the tightening is uneven: Roy_D's substrate fingerprint (the one Probe 2 was likely capturing for Roy_D-adjacent identities in the HDTF/quickclips inventory) responds well; the chronic identities NOT in the substrate-pair inventory (Q__s6, bla_bla_chow) don't respond the same way and may even get worse.

2. **The FPR=5 % regression is interpretable as score-rank degradation in the tail.** Face-pool's real-frame score distribution is tightly clustered (std 0.16 vs CLS's 0.31). The contract τ at FPR=5 % therefore sits in a region of the score distribution where small score perturbations move many frames across the boundary. The 5pp gap between FPR=5 % and FPR=10 % calibrations should swing only ~6 lockbox-fake recall in a well-calibrated head; here it swings 12pp (0.55 ↔ 0.67), which suggests calibration noise dominates the FPR=5 % readout.

3. **The +6pp at FPR=10 % is the strongest signal here.** This is the FPR target most R13 packets have been tuned toward (per `project_lockbox_fpr_dominated_by_webcam_mode.md` the 5 %→10 % τ relaxation has historically unlocked 4–77 % recall lifts on past ckpts; here it unlocks 12pp on face-pool alone). If the 800-frame canary generalizes to the full lockbox real-suite (~50 videos) and full lockbox fake-suite (~100 videos), this is a candidate inference-side lever worth taking to a full scorecard rerun.

4. **The chronic_max delta (−0.10) is genuine and surprising.** Roy_D has been the chronic offender across many R13 packets (memory `project_band_shortcut_ood_hypothesis_2026-05-16.md`). A 10pp drop on Roy_D's mean real-frame score from a pure inference-side swap is the kind of structural improvement that's worth verifying on the full eval suite. The chronic_max is the worst-case chronic-identity over-fire and is the metric directly mapped to lockbox real FPR.

5. **Probe 1's "trained-encoder substrate axis" finding is consistent with this result.** Each trained encoder constructs a substrate-discriminating axis at L11; face-pool projects features such that the projection onto that axis weakens (per Probe 2 frozen-KLIEP-μ also weakens). The trained encoder's substrate signal lives in non-face patches that face-pool excludes. The improvement on Roy_D specifically is consistent with Roy_D's substrate being non-face (e.g., body framing / capture-mode signature) — the kind of signal CLS attends to globally.

## What I am NOT claiming

- I am NOT claiming face-pool is a deployment lever. The 800-frame canary is too small to make that call; the per-identity table shows non-trivial regressions on 5 of 6 chronic identities; and the FPR=5 % regression is concerning.
- I am NOT claiming face-pool improves Slot A v2's underlying model quality. The classifier head was trained on CLS-pool input. The face-pool result is best read as "different feature pool feeding the same head produces a different score distribution" — the actual real/fake separation has shifted in a way that's not strictly comparable.
- I am NOT claiming face-pool obviates the substrate-pair contrastive lever. The two questions are separate: the substrate-pair contrastive (Phase 1) attacks the encoder's representation directly; face-pool is an inference-time replacement of the readout. Phase 1's verdict was "defer" because the contrastive axis was orthogonal to the FPR axis — that orthogonality hasn't moved.

## Recommended next phase

### Decision matrix

| Hypothesis | Test | Cost |
|---|---|---|
| Face-pool +6pp at FPR=10 % generalizes to full eval suite | Full lockbox eval-suite rerun with face-pool head | ~4–8 hours CPU, $0 GPU |
| Face-pool Roy_D improvement is identity-class generalizable | Per-substrate / per-identity scorecard on full eval (uses existing teams_real_dor_dev + lockbox readout-only suites) | ~2 hours CPU |
| Face-pool benefits are population-conditional on the canary's substrate mix | Re-run face-pool canary on a webcam-only subset of lockbox reals | ~10 min CPU |
| Face-pool benefit goes away with a face-pool-trained head | Quick LoRA-on-face-pool 200-step smoke | ~$10 GPU |

### What I'd actually queue next

1. **Webcam-only canary subset rerun** (~10 min). The lockbox FPR is dominated by webcam captures (memory `project_lockbox_fpr_dominated_by_webcam_mode.md`: webcam = 65.7 % FPR vs modern_v2 0.71 %). If the +6pp at FPR=10 % is webcam-specific, the deployment lever story is much stronger. If it's uniform across capture modes, the canary signal is more general. Either way the answer is fast and changes the next-step recommendation.

2. **Full eval-suite scorecard rerun with face-pool head**, but ONLY if the webcam canary check shows the +6pp is concentrated in the production-relevant capture modes. This is a ~4–8 hour CPU job; can be left running overnight. Output: a `analysis/face_pool_scorecard_2026-05-22/` folder with face-pool versions of the 9-suite scorecard, comparable to the 2026-05-20 panel.

3. **Track B and the deployment-candidate change to Slot A v2 step3500 (λ=1.0) remain independent and stand as the immediate operational recommendation.** None of the face-pool follow-ups touch Track B.

### What I'd NOT do

- DO NOT swap face-pool into production based on the 800-frame canary alone. The split per-identity story and FPR=5 % regression argue strongly against it.
- DO NOT use this result as evidence to revive Phase 1 substrate-pair contrastive on the original axis. The face-pool result is a separate intervention class — it does not change the orthogonality finding for the contrastive direction.

## What I am most uncertain about

- Whether the canary's 6-chronic-identity sample is representative of the full lockbox real distribution. If 5/6 chronic identities regress on the canary but the broader lockbox real distribution responds like Roy_D, face-pool could still net out positive on the full lockbox FPR. The webcam-only canary check addresses this partially.
- Whether the FPR=5 % regression is purely calibration noise (face-pool's narrower score distribution) or a real signal that face-pool loses real-vs-fake discrimination power. The same metric at FPR=10 % moves the other way, which is more consistent with calibration noise than with discrimination loss. But the lockbox_fake recall_at_tau=0.5 jumped 0.91 → 1.00 (face-pool catches all 100 lockbox fakes at the un-calibrated naive threshold), which is also consistent with face-pool's score-distribution shift pushing fakes above τ=0.5 by default. Either reading would resolve under the full eval-suite rerun.

## Decision arrow

→ **Cheap follow-up first**: webcam-only canary subset (~10 min, $0).
→ Conditional on that: full eval-suite face-pool scorecard rerun (~4–8 h, $0).
→ Independent: Track B deployment-candidate update to Slot A v2 step3500 + composite tiebreak λ=1.0 stands as the immediate operational recommendation.
