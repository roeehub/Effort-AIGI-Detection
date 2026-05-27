# AGENT_PROPOSAL — KLIEP substrate-matching check

Date: 2026-05-23. Interpretive doc accompanying `RESULTS_FACTS_2026-05-23.md`.

Task #4 in the CPU-first sequence from `TRAINING_DIRECTIONS_REVIEW_2026-05-23.md`. Establishes the policy basis for any external-dataset-acquisition decision (B.III.3).

---

## 1. Headline

**OPTB and team-identity are essentially disjoint distributions in frozen-CLIP space.** 99.9% CV accuracy + ESS_on_OPTB=3% confirms the OPTB-negative-result hypothesis at the geometry level.

Even more striking: **every team-human's real cohort is 100% CV-separable from every other team-human's** in CLIP space. The team-identity "population" is actually 5 distinct sub-distributions; treating it as homogeneous masks the per-human variance baked into the data.

---

## 2. What this means for B.III.3 (7000-webcam acquisition)

The plan recommends "pursue in parallel" for dataset acquisition; my review recommended "acquire but gate training on a substrate-match check." This readout makes the gate concrete:

| Action | CV-accuracy (new_data ↔ team-id) threshold |
|---|---|
| Train on it freely | ≤ 65% |
| Train on KLIEP-weighted subset | 65 - 80% |
| Train on ESS-selected subset only | 80 - 95% |
| Do NOT train on it without per-frame filtering | > 95% |

For reference: OPTB sits at 99.9% — solidly in "do not train" territory. The frozen-CLIP baseline §1.5.6 confirmed harm at this distance (min recall 0.037 vs 0.205 baseline, Δ −0.168).

**Operational implication**: when the 7000-webcam dataset arrives, the first thing to do is the same LR-discriminator measurement. Until that's done, do not commit GPU to training on it.

---

## 3. What this means for the structural-reframe direction

The intra-team-identity finding is decision-relevant: **each team-human is its own substrate**. This has implications for the IRM environment partition discussed in `TRAINING_DIRECTIONS_OPTIONS §3.II B.II.1`:

- The plan's per-method 12-environment partition uses fake-method as the environment axis
- An alternative partition the plan doesn't consider: **per-human 5-environment partition** (Noyn, Roee_W, Xiang, Xinhe, dor)
- The team-identity-real cohort is naturally partitioned this way (the labels exist; the geometric separation is 100%)
- An IRM penalty on per-human-invariance would directly target "the classifier should behave the same across humans" — which IS the binding gate

**Two IRM partition candidates** worth considering for Week 2+:
- 12-env per-method (plan's current spec)
- 5-env per-team-human (new from this readout) — directly targets the binding gate

A 5-env partition might be too narrow (DomainBed shows IRM brittle below ~5 envs); 12 is safer. But the 5-env partition is the direct mechanism for the team-identity gate, which is what we're optimizing for.

A practical middle-ground: **17-env partition** (12 per-method + 5 per-human as additional environments). The IRM penalty would force the classifier to be invariant across BOTH method-axis and human-axis, which is the joint constraint the deploy actually faces.

---

## 4. Bonus mechanism: the "ESS on OPTB" decomposition

ESS_on_OPTB varies 0.010 (Xinhe) to 0.041 (dor) across team humans. dor real frames are slightly less far from OPTB than Xinhe real frames. Speculation (not measured): OPTB has webcam frames (AVSpeech) that share some characteristics with dor-recorded sessions; Xinhe-recorded sessions are even further afield. This is consistent with the empirical fact that:
- Xinhe-recapture frames are the hardest to recover (lowest fake recall across all ckpts)
- dor-recapture frames are catchable by most ckpts

There may be a *very small* signal (~4% ESS) where filtering OPTB to the dor-similar subset would help dor specifically. But ~3% ESS is marginal at best.

---

## 5. Recommendations

### 5.1 Immediate ($0)

1. **Add the LR-discriminator + ESS substrate-match check as a wiki-protocol step** before any future dataset acquisition. The check is ~30 min on a new 6K-frame sample.
2. **Use the per-team-human partition as an IRM environment axis** in Week 2 if/when IRM-only smoke passes its abort criteria.

### 5.2 For Week 1

This readout doesn't change Week 1 GPU recommendations directly. But it adds two pieces of context:
- The 5%/50% bar's per-human variance is mechanically baked in by the 100% CV-separable subdistribution structure. Even a great encoder might not perfectly equalize the per-human FPR/recall.
- Confirms B.III.3 (dataset acquisition) needs the substrate-match gate before any GPU spend.

### 5.3 For the review file (TRAINING_DIRECTIONS_REVIEW)

Update the §3.2 OPTB section to cite this readout's CV accuracy numbers as concrete evidence ("99.9% CV accuracy, ESS=3%"). Strengthens the case for the dataset-acquisition guard rail.

---

## 6. Self-correction log

- **Initial framing**: I expected OPTB vs team-identity to be ~85-90% CV accuracy (mismatched but with some overlap). The 99.9%-on-both-sides was higher than I'd estimated.
- **Intra-team finding**: This was NOT what I was looking for; it was a bonus from running the comparison. Reframed §3 to highlight it as the structural-IRM implication.
- **Did not run OPTB ↔ lockbox**: The D8 cache lacks labels; reconstructing the dev/lockbox split adds an hour. I decided the OPTB ↔ team-identity comparison was sufficient for the policy question. If the user wants OPTB ↔ lockbox, it's a 1-hr follow-up.

---

## 7. Followups (TODOs for user-decided application)

### Memory updates

- **NEW**: `project_kliep_substrate_match_optb_99pct_2026-05-23.md` — "Frozen-CLIP LR discriminator OPTB-real vs team-identity-real CV accuracy 0.999 (AUC 1.000); ESS_on_OPTB=0.031. OPTB-fake vs team-id-fake similarly 1.000 CV accuracy, ESS=0.027. Strongly confirms training-corpus is geometric disjoint from deploy distribution. ALSO each team-human's real cohort 100% CV-separable from every other (intra-team substrate fragmentation). Policy: use this discriminator as gate before any external-dataset acquisition; threshold table at analysis/kliep_substrate_match_2026-05-23/RESULTS_FACTS_2026-05-23.md §4."

### Threads to amend

- `docs/packet_retrospectives/threads/viso_bucket_gap.md` and `iq_shortcut_deconvolution_program_2026-05-08.md` should cross-reference this finding.

### OPEN_LOOPS

- Open: "OPTB ↔ lockbox substrate-match measurement" — 1-hr CPU follow-up; needed for full policy coverage.
- Open: "Per-team-human IRM environment partition as candidate for B.II.1" — input to the Week 2 IRM spec.
- Open: "Substrate-match gate policy doc" — codify §4 of the FACTS doc into the AGENTS.md or wiki protocol.

### TIMELINE

- Append: `2026-05-23 PM — KLIEP substrate-match check (analysis/kliep_substrate_match_2026-05-23/) — OPTB ↔ team-identity CV accuracy 0.999 (ESS_on_OPTB=0.031); OPTB ↔ each team-human 0.999-1.000; intra-team between-human reals 0.999-1.000. Disjoint distributions in CLIP space. Policy: no new external data trained on without substrate-match gate (threshold table at §4). Adds 'per-team-human (5-env)' to IRM partition candidates.`

---

## 8. Gaps and blockers

- **D8 lockbox not measured.** OPTB ↔ lockbox and team-id ↔ lockbox would round out the substrate-distance map. 1-hr CPU follow-up.
- **Permutation test not run** on the 100% CV-accuracy results. Likely robust but worth verifying.
- **Substrate-distance not measured at non-L11 layers.** Could be different at L8 or L4; not measured.
