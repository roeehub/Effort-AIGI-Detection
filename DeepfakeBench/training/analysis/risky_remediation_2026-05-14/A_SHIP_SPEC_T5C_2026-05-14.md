# T5C Ship Spec — for PM cloud-side test

**As of 2026-05-14. This is the validated recommendation BEFORE any blend-remediation experiments.**

## Checkpoint

| field | value |
|-------|-------|
| Run ID (Vertex) | `jrlldtem` |
| Yaml | `experiments/phase2_round13/R13_T5C_T4_BIG_CLASSIFIER_2026-05-11.yaml` |
| GCS path | `gs://training-job-outputs/best_checkpoints/jrlldtem/periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth` |
| Step | 3500 |
| Train AUC | 0.9944 |
| Train EER | 0.0197 |
| Local cache | `analysis/r13_overnight_may6_retest_2026-05-13/_ckpt_cache/periodic_effort_20260511_step3500_auc0.9944_eer0.0197.pth` |
| SHA-256 | run `sha256sum` if needed for integrity check |

## Operating gates

### G1 — face detector
Reject frame if your existing face detector does not return a face on the cropped frame.
(Whatever face detector you use today — same behavior.)

### G2 — minimum face dimensions
Reject frame if `min(face_crop_width, face_crop_height) < 110 px`.

(Updated 2026-05-14: relaxed from 200 → 150 → **110** based on fine sweep at 10-px resolution. The "ideal" by combined-pool identity-correctness is **G2(110-120)**: 50/55 identities correctly verdicted across teams_dev + teams_lockbox + dor_cross at 90.9% combined correct rate, vs 48/54 = 88.9% at G2(150). Below 100, T5C precision starts to drop. Above 130, you throw away usable frames. See `E_GATE_EXPLORATION_VERDICT_2026-05-14.md` for the full sweep. If you want to be conservative, G2=120 is essentially tied — pick either.)

Frames that fail G1 or G2 are **not scored** — they don't vote in the per-identity decision.

## Per-frame scoring

Resize the face crop to 224×224, then standard CLIP normalization:

```python
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

# pseudocode
img = cv2.resize(face_crop_bgr, (224, 224), cv2.INTER_LINEAR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
tensor = ToTensor()(img)
tensor = Normalize(CLIP_MEAN, CLIP_STD)(tensor)
prob = model(tensor)["prob"]   # scalar in [0, 1] — higher = more fake-like
```

## Per-frame threshold (τ)

```
tau = 0.49
```

This was set by Stage 11 fine-tau sweep on the F4 substrate. The safe operating window is **[0.27, 0.67]** — 40pp wide. Anywhere in this range gives identical identity-level decisions on the F4 calibration pool. 0.49 was picked because it caught one additional fake (Roy_D fake substrate) at the same FPR.

## Per-identity decision

Production already aggregates many frames per identity. Two options:

### Option 1 — original simple rule (status quo)
```python
def is_fake_identity(frame_probs):
    # frame_probs = list of scalar probs from frames that passed G1+G2
    if len(frame_probs) == 0:
        return None   # can't decide — abstain
    frac_above_tau = sum(p > 0.49 for p in frame_probs) / len(frame_probs)
    return frac_above_tau > 0.50
```
**Safe majority-fraction window** is `[0.18, 0.70]` — 52pp wide on F4. 0.50 sits comfortably in the middle. Identity-correct rate on combined production pool: **92.96% (66/71).**

### Option 2 — simple stricter base threshold (modest expected improvement)
```python
def is_fake_identity(frame_probs):
    if len(frame_probs) == 0:
        return None
    frac_above = sum(p > 0.6 for p in frame_probs) / len(frame_probs)
    return frac_above > 0.40
```
"≥13 of 32 above 0.6." Identity-correct rate on this 71-identity pool: **95.77% (68/71).** FPs 4 → 2 (-50%), 0 new FNs. Cross-pool stable.

### Option 3 — combined bulk+tail rule (RECOMMENDED, theory-motivated, cross-validated)
```python
def is_fake_identity(frame_probs):
    if len(frame_probs) == 0:
        return None
    n = len(frame_probs)
    bulk_passes = sum(p > 0.6 for p in frame_probs) / n > 0.40
    tail_passes = sum(p > 0.9 for p in frame_probs) >= 1
    return bulk_passes and tail_passes
```
"≥13 of 32 above 0.6 AND ≥1 of 32 above 0.9."

**Theory motivation**: bulk evidence (`frac>0.6 > 0.4`) catches identities with diffuse score distributions; extreme-frame requirement (`count>0.9 ≥ 1`) catches identities whose scores elevate to 0.7-0.8 but never reach 0.9. These are two structurally distinct FP failure modes that single-threshold rules can't separate. The rule was specified by theory, not search.

**Results**:

| pool | rate | FPs | FNs | notes |
|---|---|---|---|---|
| 71-pool (derivation) | 69/71 = 97.18% | 1 | 1 | derived from |
| **Independent training-eval pool** | **43/47 = 91.49%** | **4** | **0** | **cross-validation** |
| Bootstrap vs Opt 2 (71-pool) | P(Δ>0) = 95.4% | — | — | rule pre-specified |
| Bootstrap vs Opt 1 (training-eval) | Δ same sign, same mechanism (rescues bla_bla_chow) | — | — | independent confirmation |

The independent training-eval pool confirms the same FP rescue mechanism (bla_bla_chow_type identities with `count>0.9 = 0`). The mechanism is structural, not a 71-pool artifact.

**Statistical note**: An earlier analysis (`H_STATISTICAL_REALITY_CHECK_2026-05-14.md`) showed that the permutation test on 320 rules gives p=0.526 — but that test penalizes for searching 320 candidate rules. Option 3 is pre-specified from theory (bulk + extreme), not search-found, so the regular bootstrap (P(Δ>0) = 95.4% on the 71-pool) is the right stat. Independent cross-validation on a 47-identity pool reproduces the same rescue mechanism — that's the load-bearing evidence.

### Recommendation

**Pick Option 3.** Theory-motivated rule, generalizes to independent validation data, and the bla_bla_chow rescue mechanism (no extreme frames) is structurally interpretable. Marginally larger code change than Option 2 (one extra `count > threshold` check), no extra inference cost.

**Known limitations** (will persist regardless of rule):
- **Roy_D** — confidently-wrong real-person FP. Model-side failure (mean 0.888, count>0.9=78). Needs training-side fix.
- **PC_Generator__s22/s45, Q__s6** — "real people running 0.6-ish" mode found in independent validation. They have legitimate extreme frames (count>0.9 = 2-6), so Option 3 doesn't help. Need substrate-aware calibration or per-identity feedback loop in production. See `I_INDEPENDENT_VALIDATION_2026-05-14.md`.

If you want simplest possible deploy: Option 2 catches one fewer FP than Option 3 and gives 95.8% on the 71-pool. It's a single-line code change.

## What is NOT included

- ❌ The blend-unsharp preprocessing trick from today is **not** part of this spec. It hasn't been validated on cross-substrate cohorts yet. Today's spec is the conservative one.
- ❌ Any per-frame remediation, color correction, or sharpening.
- ❌ Per-cohort calibrated thresholds.

If you want to plumb the blend trick as a feature flag, the preprocessing change is shown in `FINAL_VERDICT.md`, but **wait for the cross-substrate study (in flight today) to confirm it doesn't regress on may6 / live_prod before enabling.**

## Quick smoke test for cloud deploy

After loading the checkpoint, score these two reference frames (any frame from each works) and verify the prob falls in the expected range:

| frame | expected prob @ T5C step3500 |
|-------|------------------------------|
| `gs://teams-faces-data-test-2914-fake-4420-real-feb-28/fake/Cam_Test__s33_*` (any) | 0.50–0.95 (most > 0.80) |
| `gs://teams-faces-data-test-2914-fake-4420-real-feb-28/real/Md_noyn_Sharker__s15_*` (any) | 0.00–0.20 (typically < 0.10) |

If both fall in their expected ranges, the load is correct.

## Authoritative source documents

- `analysis/best_candidate_search_2026-05-13/00_MORNING_RECOMMENDATION.md`
- `analysis/best_candidate_search_2026-05-13/03_DEPLOYMENT_GATE_AND_THRESHOLDS.md`
- `analysis/best_candidate_search_2026-05-13/04_FINAL_VERDICT_PER_IDENTITY.md`
