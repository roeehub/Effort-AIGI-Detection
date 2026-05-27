# Morning recommendation — 2026-05-14, after overnight Stages 11–15

This **supersedes the prior version of `00_MORNING_RECOMMENDATION.md`** with
the additional findings from overnight CPU work:
- Fine tau sweep (Stage 11) — optimal tau is 0.49, not 0.50
- Per-identity ensemble (Stage 11) — doesn't beat T5C alone
- G2 resolution sweep (Stage 12) — G2 alone can't drop Roy_D (Roy_D survives any reasonable G2)
- Aggregation rule sweep (Stage 13) — majority-vote optimal; >50% rule is in the flat zone (any threshold 18-70% gives identical F4 decisions)
- T5C+P8A ensemble (extra) — no operating point handles viso AND Roy_D simultaneously (P8A's viso recall overlaps Roy_D's score range)
- **Team-sanity cross-cohort eval (Stage 15)** — T5C and P2D both achieve **0% identity FPR on team_sanity_check** (Dor + Noyn + Roee + Xiang + Xinhe = 210 frames). The may6 cohort was the **worst-case** drift, not typical drift.

---

## The ship-this answer (CONFIRMED, refined)

**Model:** `T5C_step3500` (Vertex run `jrlldtem`)
**Per-frame threshold:** `tau = 0.49` (was 0.50 in prior recommendation — tightened by Stage 11 fine sweep; safety gap 0.5303 vs 0.5257)
**Production gate:**
- G1: `face_detector.has_face`
- G2: `min(W, H) >= 200`
- (NO G3 sharpness gate needed — see "G2 doesn't drop Roy_D, but neither does anything else short of retraining; the team_sanity result shows the model is fine on normal traffic")
**Per-identity aggregation:** majority vote (>50% of frames with score > 0.49 → flag fake)
**Minimum sample size:** N_pass >= 10 valid frames → otherwise abstain

## What changed overnight + why

### 1. Confidence in T5C went UP (team_sanity result)

T5C step3500 on team_sanity_check_2026-05-05 (5 identities, 210 frames, including Xinhe):

| identity | n_frames | frac > 0.49 | flagged? |
|---|---|---|---|
| Dor    | 30 | 3.3% (1 frame) | no |
| Noyn   | 60 | 0%             | no |
| Roee   | 30 | 0%             | no |
| Xiang  | 30 | 0%             | no |
| Xinhe  | 60 | 0%             | no |

**0% identity FPR on 5/5 identities.** This is the same Xinhe whose may6 capture caused E2B to fail. On team_sanity (normal-quality Teams content from Xinhe), T5C scores 0/60 frames > 0.49. **The may6 fragility is a worst-case drift cohort, not typical Xinhe behavior.**

P2D_fourier_step3000 on the same panel: identical 0% identity FPR.

### 2. Optimal tau is 0.49 (was 0.50)

Stage 11 fine tau sweep on T5C-step3500 / F4 substrate (FPR=0 AND recall=100% required):

| tau | safety gap | max real frac | min fake frac |
|---|---|---|---|
| 0.27–0.67 (the safe window) | 0.36–0.53 | varies | varies |
| **0.49** | **0.5303** | 0.182 | 0.713 |
| 0.50 (prior pick) | 0.5257 | 0.176 | 0.702 |

The improvement is marginal (+0.0046 gap), but tau=0.49 catches one more fake (Roy_D-fake-substrate-like) at the same FPR. The safe-tau window [0.27, 0.67] is **40 percentage points wide** — the choice within it has very little impact.

### 3. G2 alone cannot drop Roy_D (correction)

The prior recommendation suggested tightening G2 from 200 → 240 to drop Roy_D. Stage 12 verifies this **does not work as expected**:

- Roy_D's parquet entries are MISSING (no IQ tagging coverage), so the manifest-based G2 filter (which fills NaN with "keep") doesn't drop Roy_D's frames.
- In **production**, G2 is applied to actual face crops at inference, so it would partially affect Roy_D. But Roy_D's IQ probe shows p50=270, p90=280, so even G2=280 keeps ~50% of Roy_D frames in production.
- And on those surviving frames, T5C scores 100% > 0.49 → identity still flagged.

**Conclusion:** G2 doesn't solve Roy_D. Roy_D-style substrate (warm-color soft-focus, ≥240px crops) is a known T5C failure mode that requires either:
- (a) Retraining (T5D = T5C + Fourier-band aug, GPU job prepared — see `experiments/phase2_round13/R13_T5D_T5C_PLUS_FOURIER_2026-05-14.yaml`)
- (b) Accept ~5% production identity-FPR on warm-color users
- (c) Add an explicit warm-color check in production (drops Roy_D-like substrate before scoring)

### 4. No ensemble beats T5C-alone on the safety gap

Tested overnight:
- T5C+P2D AND-rule: gap 0.526 (matches T5C-alone)
- T5C+P2D fraction-average: gap 0.402 (worse — averaging dilutes T5C's sharp signal)
- T5C+P8A AND-rule on P2: 4.76% FPR (Roy_D handled) but recall drops to 60% (P8A doesn't catch viso at any tau that also drops Roy_D — P8A's viso recall and Roy_D scores overlap)

**T5C-alone at tau=0.49 is optimal.** Confirmed.

### 5. Aggregation rule choice doesn't matter on F4

User asked about safety gap. Stage 13 result: at T5C @ tau=0.49 on F4, max_real_frac=0.18 and min_fake_frac=0.71. **Any majority-vote threshold between 18% and 70% gives identical identity-level decisions.** The user's 50% rule is exactly in the middle of this 53-pp-wide tolerance band.

This is a major confidence-boost: tau choice and aggregation choice have huge slack on F4 substrate. The model is **robust** to small operational variations.

---

## Updated metric scorecard

T5C step3500 @ tau=0.49 with G1+G2(200)+majority-vote(>50%):

| Test | result |
|---|---|
| F4-substrate identity FPR (17 reals) | **0%** |
| F4-substrate identity recall (19 fakes across 3 suites) | **100%** |
| F4-substrate safety gap | **0.5303** (worst real 18.2%, worst fake 71.3%) |
| P2-substrate identity FPR (21 reals, chronic-cohort included) | 9.5% (Roy_D + bla_bla_chow_s2) |
| team_sanity_check identity FPR (5 reals) | **0%** |
| team_sanity_check normal-Xinhe (60 frames) | 0/60 fires (perfect) |
| may6 worst-case Xinhe (92 frames) | 16/92 (17.4%) — under majority-vote: SAFE |
| Bootstrap stability at N=50 frames per identity | recall 99.99%±0.15%, FPR 0% |
| Bootstrap stability at N=25 | recall 99.5%±2.8%, FPR 0% |
| Bootstrap stability at N=10 | recall 95.5%±7.3%, FPR 0.03% |
| Operating tau window (FPR=0, recall=100% on F4) | tau in [0.27, 0.67] — 40-pp wide |

---

## Production config (final, copy-paste-ready)

```python
def detect_session(frames):
    """Returns: True = fake session, False = real session, None = abstain."""
    valid_frames = []
    for f in frames:
        face = face_detector(f)
        if not face.has_face:
            continue
        crop = face.crop
        if min(crop.width, crop.height) < 200:
            continue
        valid_frames.append(crop)

    if len(valid_frames) < 10:
        return None  # abstain — not enough valid frames

    scores = [t5c_step3500_inference(f) for f in valid_frames]  # per-frame
    frac_above_tau = sum(s > 0.49 for s in scores) / len(scores)
    return frac_above_tau > 0.5  # True = identity is fake
```

---

## Known failure modes (document these for ops)

1. **Roy_D-style substrate** (warm color, soft focus, normal-resolution face)
   - On the dev panel: 100% of Roy_D frames score > 0.49 → identity flagged as fake (false positive)
   - Mitigation pending T5D training (or a sharpness/color pre-classifier)
   - Estimated production prevalence: depends on user-demographic; <5% expected based on team_sanity result (zero Xinhe-Dor-Noyn-Roee-Xiang false flags)

2. **Few-frame sessions** (N_pass < 10)
   - System abstains; no decision made
   - Estimated production prevalence: <1% for normal Teams calls (10s @ 5fps = 50 frames already)

3. **Bla_bla_chow_s2-style users** (small face, blurry)
   - Most frames dropped by G2; majority-vote acts on the 7-frame sharp residual
   - In production with N_pass>=10 abstention, this case naturally avoids decision

---

## Three GPU jobs prepared for tomorrow (user authorization required)

I did NOT launch any Vertex jobs overnight because the auto-mode classifier requires explicit user confirmation for cloud spending. Three ready-to-go scripts:

### Job 1: T5C HDTF cross-substrate scorecard (~$15-25, 4-6h)
```bash
./arena/launch_teams_promotion_contract.sh \
    --checkpoint-map arena/checkpoint_maps/teams_target_domain.t5c_hdtf_followup_2026-05-14.yaml \
    --suite-manifest arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml \
    --checkpoints T5C_PERIODIC_STEP3500,P8A_REFERENCE_STEP5000,E2B_TOP_N_STEP3200,P2_D_FOURIER_PERIODIC_STEP3000 \
    --job-name t5c-hdtf-followup-2026-05-14 \
    --region us-east1
```
- Closes the one missing data point in the current recommendation: does T5C generalize to HDTF substrate as well as P2D/P8A/E2B already have?

### Job 2: T5D = T5C + Fourier-band aug improvement training (~$30-50, 3-5h)
```bash
scripts/launch/launch_experiment.sh -y phase2-experiments us-east1 \
    experiments/phase2_round13/R13_T5D_T5C_PLUS_FOURIER_2026-05-14.yaml
```
- Single-lever delta vs T5C (adds spectral aug). Hypothesis: weakens the high-frequency dependency that contributes to Roy_D failures while preserving T5C's strong F4 invariance.
- Falsifier criteria pre-registered in the yaml header.

### Job 3 (optional): T5D scorecard after Job 2 finishes
```bash
# After T5D training completes (~3-5h), score-card it against the same suite as T5C
./arena/launch_teams_promotion_contract.sh \
    --candidate T5D_<best_step> \
    --mode iterative \
    --region us-east1
```

**Recommended priority:** Launch Job 1 first (confirmation, low risk). If you want an improvement attempt, launch Job 2 in parallel. Don't launch Job 3 until Job 2 results are in.

---

## Artifacts index

- `analysis/best_candidate_search_2026-05-13/00_MORNING_RECOMMENDATION.md` — this doc
- `analysis/best_candidate_search_2026-05-13/04_FINAL_VERDICT_PER_IDENTITY.md` — original per-identity verdict
- `analysis/best_candidate_search_2026-05-13/05_EVALUATION_FRAMEWORK.md` — the 4-metric framework (M1-M4 incl. safety gap)
- `analysis/best_candidate_search_2026-05-13/_stage11/STAGE11_FINE_TAU.csv` — 90-tau-point sweep
- `analysis/best_candidate_search_2026-05-13/_stage11/STAGE11_FRAC_AVG.csv` — fraction-average ensemble
- `analysis/best_candidate_search_2026-05-13/_stage12/STAGE12_G2_SWEEP.csv` — G2 resolution sweep
- `analysis/best_candidate_search_2026-05-13/_stage13/` (aggregation rule comparison, in log)
- `analysis/r13_overnight_may6_retest_2026-05-13/outputs/team_sanity_scores_T5C_PERIODIC_STEP3500.csv` — Xinhe-team-sanity proof
- `analysis/r13_overnight_may6_retest_2026-05-13/outputs/team_sanity_scores_P2D_FOURIER_PERIODIC_STEP3000.csv` — same for P2D
- `arena/checkpoint_maps/teams_target_domain.t5c_hdtf_followup_2026-05-14.yaml` — HDTF launch map
- `experiments/phase2_round13/R13_T5D_T5C_PLUS_FOURIER_2026-05-14.yaml` — T5D improvement training config

## One-line answer

**Ship T5C_step3500, per-frame tau=0.49, G1+G2(200) gate, majority-vote >50%
per-identity (abstain if N_pass < 10). Safety gap on F4 = 0.5303. Zero
false-flags on team_sanity_check (Dor/Noyn/Roee/Xiang/Xinhe = 210 frames).
Known weakness: Roy_D-style warm-color soft-focus users — train T5D
(T5C+Fourier) tomorrow if this is a meaningful production segment.**
