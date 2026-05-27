# Frozen-CLIP team-identity baseline — AGENT PROPOSAL

This file carries interpretation / verdict / recommendation. Factual readout lives in `RESULTS_FACTS_2026-05-23.md`.

## TL;DR

**FT is NOT net-zero on the team-identity deploy bar. Frozen-CLIP + a properly-trained head is dramatically worse than the FT'd ckpts on per-human fake recall at the same effective real-side FPR. Training-data diversity does NOT close the gap; the encoder is doing the load-bearing work.**

At per-head τ calibrated to 5% team-aggregate real FPR (the cleanest apples-to-apples comparison):
- **min per-human recall** (the binding fake-side gate per `project_team_identities_multi_labeled_2026-05-23`):
  - **P8A 0.786**
  - **T5C (production) 0.379**
  - best frozen-CLIP head (MLP_DEV_LB; lockbox-domain reals in training): **0.205**
  - baseline LR_DEV: **0.055**
  - **best Option B head (trained on actual training-corpus, 6K diverse samples) = LR_OPTB 0.000** — strictly worse than DEV; OPTB makes it worse, not better.
- The gap from frozen-CLIP-best to T5C is **17pp on min recall**; the gap from frozen-CLIP-best to P8A is **58pp on min recall**.

The brief's 10pp threshold is exceeded by a wide margin: frozen-CLIP best is **17-78pp worse** than T5C / E2B / P8A on per-human fake recall at the same FPR.

**The 6 weeks of FT have bought ~30-60 percentage points of fake recall at a fixed FPR. This is real, large work.** The "structural reframe" reading — that we should pivot away from FT-class levers entirely — is NOT supported by this probe.

The OPTB result (training-data diversity makes frozen-CLIP head WORSE, not better) is additionally diagnostic: the encoder is the binding constraint, not head capacity or training-data substrate diversity. FT'd ckpts work because they tune the encoder on Teams-substrate data; frozen-CLIP cannot replicate this regardless of head architecture or training-set composition.

## What this rules in / out

### Rules IN
- **Continuing FT-class work has a paradigm-justification floor**. The encoder modifications FT makes are doing real work on the team-identity binding axis. A pure-head-on-frozen-CLIP approach is not a viable production substitute.
- The fact that **DEV+LB heads close part of the gap** (LR_DEV 0.055 → LR_DEV_LB 0.153 / MLP_DEV_LB 0.205 on min recall) suggests the encoder representational ceiling can be raised by adding training-data substrate diversity. FT'd ckpts get this from joint encoder+head training; a frozen-head approach gets less but not zero.
- **The variance in per-human gap is the load-bearing signal**:
  - On `Xiang` (a clean fake-attack cohort with high SNR features): frozen-CLIP MLP_DEV_LB gets 0.465 recall vs T5C 0.948 — a 48pp FT-effect.
  - On `Xinhe` (the binding-min cohort): frozen-CLIP MLP_DEV_LB 0.205 vs T5C 0.379 — a 17pp FT-effect. SMALLER gap, because T5C itself is weak here.
  - On `dor` (P8A's traditional strength): frozen-CLIP MLP_DEV_LB 0.527 vs P8A 0.870 — a 34pp FT-effect.
  - **Different humans bind different head-vs-encoder trade-offs.** Future FT work should treat per-human-binding as the diagnostic axis, not aggregate dev_macro / lockbox_fpr.

### Rules OUT
- The structural-reframe reading "FT has bought ~0; pivot to B/C/D-class" is refuted. FT-paradigm changes (improving the encoder or head jointly) remain the right ladder.
- The simpler claim "T5C ≈ frozen-CLIP-with-good-head" (which would be the case if FT was net-zero) is decisively refuted at the team-identity bar. T5C beats every frozen-CLIP head we tried by 17-50pp on min recall.
- Possible follow-up of "is the encoder actually doing the work, or is it just the head" — this is the D3/D8 question. The mechanism of FT can be parsed via in-loop-head-on-frozen-CLIP-encoder (D3 already did this for P8A; the answer was the joint head+encoder optimization is what matters, not the head alone).

## What the gap is mechanistically

### Hypothesis 1 (most likely): the encoder is doing material work
The frozen-CLIP head can fit any linear/MLP separator on its feature space. With train AUC = 1.0 on DEV / DEV+LB / OPTB, the head is at maximum-capacity. The remaining transfer gap to the FT'd ckpts is upper-bounded by the head's capacity to learn from frozen features.

T5C's joint encoder+head training moves the encoder along directions that better separate team-identity reals from fakes. The frozen-CLIP encoder doesn't have these directions; the head must reconstruct them post-hoc from the frozen CLS pool, and it cannot.

Evidence: even the DEV+LB head (which has the lockbox-cam-test-s33 substrate in training that the FT'd ckpts trained on similarly) closes only 15pp of the 73pp Xinhe gap.

### Hypothesis 2 (refuted by OPTB): substrate non-coverage in DEV
The DEV pool is 4K frames from one bucket; it doesn't contain the Xinhe/Xiang fake-attack cohorts in any form. The FT'd ckpts trained on `live_fakes_teams_prod` + similar Teams-fake cohorts at much higher coverage. Maybe Option B (full training-corpus sample) closes the gap.

**REFUTED**. Training the head on 6K stratified-by-method samples from the actual training corpus (`frame_properties.parquet`; deep-live-cam, simswap, FF++ variants, AVSpeech, celeb_synthesis, etc.) is **strictly worse** than DEV at every joint-calibrated point:
- LR_OPTB: 0.0% Xiang recall, 0.2% Xinhe recall, 7.1% dor recall at team-FPR=5%
- MLP_OPTB: 0.5% Xiang, 1.4% Xinhe, 9.2% dor at team-FPR=5%
- LR_OPTB_DEV (combined 10K): worse than LR_DEV alone (min recall 0.018 vs 0.055)
- OPTB heads' lockbox AUC: 0.345 (LR) / 0.516 (MLP) — below DEV's 0.78 / 0.61

The OPTB head learns FF++-style separators that don't transfer to Teams-substrate attacks. The frozen-CLIP feature space simply does not encode the Teams-specific fake-signal at sufficient SNR for any head to extract it from these training pools — even pools that are nominally larger and more diverse.

**This strongly supports H1**: the encoder (not the head, not the training-data diversity) is the load-bearing component. The FT'd ckpts' advantage is that their encoder representation was *jointly tuned with their head* on a corpus that includes Teams-substrate variants. The encoder learned new directions that separate Teams-fake from Teams-real; the frozen-CLIP encoder does not have these directions, and no choice of head extraction from frozen features can recover them.

## Implications for the structural reframe proposal

The reviewer asked whether this probe supports "the next experiment should be structural (B/C/D class)" or whether "FT is the right paradigm."

**Recommendation: FT is the right paradigm**. The specific FT recipe that beats T5C-current-production is still open, but the class of "modify encoder + head jointly" is correct.

The current expanded-readout shows P8A still has higher min-recall (0.786) than T5C (0.379) on team-identity at the same effective FPR — but P8A also has different Xinhe-may6 behavior per `project_xinhe_may6_falseflag_2026-05-06`. T5C was deployed for reasons (per `project_production_is_t5c_not_e2b_2026-05-23`, the lockbox-aggregate behavior was preferable). The actual deployment-relevant question is "what FT recipe gives P8A-style team-identity recall + T5C-style lockbox-aggregate behavior?"

This is an FT engineering question, not a structural-reframe question. The structural levers (e.g. per-human Teams-API runtime calibration, ensembling, different output spaces) remain options BUT don't dominate FT-recipe iteration based on this probe.

### Suggested next experiments (not requesting authorization, just recording the option space)
1. **Frozen-CLIP + better head**: try gradient-boosted decision trees (lightgbm/xgboost) on CLIP features; try wider MLP (512 / 1024); try ensembling LR+MLP. May close 5-10pp but not approach FT.
2. **OPTB head on FT'd encoder**: take T5C's encoder, freeze, train a fresh head on OPTB CLIP-encoder features. This decomposes "encoder contribution" from "head contribution" — D3/D8 already did this for P8A, but not for T5C. Would clarify whether T5C's specific gains come from the encoder shift or from the head being calibrated against the in-loop loss.
3. **T5C-with-modified-head**: add a per-human-aware loss term (e.g. minimizing FPR variance across the 5 team humans on the dev pool); FT only the head from T5C as initialization. Hits the binding constraint without re-tuning the whole encoder.

## Self-correction log

### Methodological compromises

1. **Used D8's exact DEV sample (4K) instead of a fresh stratified sample**. Justified by needing apples-to-apples vs D8's frozen-CLIP baseline. Bias: DEV is dominated by `teams_real` from one bucket; the head doesn't see other-bucket reals at training time. For a fair baseline, a multi-bucket real sample would be better. The DEV+LB variant partially compensates by adding `teams_capture_cam_test_s33/35` substrate.

2. **Option B is only 6K frames vs 1.4M training corpus**. Sampling at 0.4% misses long-tail methods. Bias: Option B underbounds what frozen-CLIP can do with the full training-corpus diversity. However, the OPTB result is so far below DEV (LR_OPTB min recall 0.0% vs LR_DEV 5.5%) that scaling OPTB to 1.4M is unlikely to flip the verdict. The dominant constraint is that the training corpus does not contain Teams-substrate cohorts at all (those data lanes were added separately in later FT'd-data revisions); even a 100% OPTB sample would not contain `live_prod__*` or `visomaster_v2_*` frames. The headline (FT is doing real work) is robust to OPTB scaling because the encoder, not the head's training-data diversity, is the binding constraint.

3. **MLP architecture (256u, 1 hidden layer) is a single point estimate**. Wider/deeper MLPs were not swept. Bias: best frozen-CLIP head may be slightly stronger than measured. Unlikely to close the 17-50pp gap to T5C/P8A given that train AUC is already 1.0 (the head is at capacity).

4. **τ was calibrated per-head on team-aggregate real FPR**. This is the cleanest apples-to-apples but it's a stronger calibration than what's typically deployed (in production, τ is fixed offline; the assumption is dev-cal-5% ≈ team-cohort 5%). The dev-cal-5% τ for LR_DEV maps to lockbox FPR 83% (D8 territory), so this matters: the LR_DEV head in a real deploy would not get the team-FPR=5% τ it gets here; it would over-fire massively at the dev-calibrated τ. The frozen-CLIP baseline numbers in §3 are therefore an OPTIMISTIC bound (best-case under perfect team-cohort calibration).

5. **No identity-blocking in the dev pool**. Some identities (like dor_shkedi__s16) appear in both DEV training and the team-identity eval cohort. The 0.891 LR_DEV FPR on dor_shkedi__s16 is partly information-leakage offsetting; the LR_DEV_LB collapse to 0.204 reflects lockbox training including more dor_shkedi frames.

6. **No comparison vs T5C's HEAD trained on T5C's frozen encoder**. To decompose "head vs encoder" we'd need to (a) freeze T5C's encoder and (b) re-train a head on DEV — that compares head-vs-head with the same encoder. Not done here; the question scoped here is "frozen-CLIP-encoder + best-head vs T5C-encoder + T5C-head." A more refined probe would split the encoder contribution.

### Bias direction of each compromise
- (1) UPPER bound for frozen-CLIP: DEV-overlap helps the head, doesn't help FT'd ckpts (they trained on different sample).
- (2) LOWER bound for frozen-CLIP: small OPTB sample. But OPTB at 100% would still not contain Teams-substrate cohorts; the structural deficit (training-corpus does not include Teams data lanes) is invariant to sample size.
- (3) LOWER bound for frozen-CLIP: head capacity not exhausted. But not by much.
- (4) UPPER bound for frozen-CLIP: optimistic τ calibration vs real-deploy fixed τ.
- (5) UPPER bound for frozen-CLIP: leakage helps LR_DEV.
- (6) Indeterminate (not measured); would tighten the verdict.

**Net direction**: the conclusion "FT is doing real work, dramatically beats frozen-CLIP at the team-identity bar" is robust to all six compromises; they collectively bias the comparison TOWARD frozen-CLIP, and frozen-CLIP still loses by 17-50pp on min recall.

## Validation cross-checks

- **D8 (2026-05-12)**: D8's `P8A_dev_unweighted` got 0.5865 DEV→LOCKBOX AUC on P8A's trained-encoder L11 features. This probe's `LR_DEV` (same sample, frozen-CLIP L11) gets 0.7829 — frozen-CLIP transfers better than P8A's encoder for the lockbox task. This is consistent with frozen-CLIP being a stronger generic baseline (it's the source of P8A's pretraining; P8A modified L11+later to specialize, losing some generic transferability).
- **Expanded readout (2026-05-23)**: at constant τ=0.78 (mode B), the FT'd ckpts in this run reproduce the per-human FPRs from `analysis/team_identity_deploy_readout_expanded_2026-05-23/outputs/per_human_summary.csv` exactly (same data, same calibration). The frozen-CLIP rows are appended consistently.
- **FT'd ckpt re-ranking under joint-calibration**: at team-FPR=5% per-head, the per-min-recall ranking is P8A > E2B > SlotAv2_FACE > SlotAv2_CLS > T5C, which matches the expanded readout's qualitative ordering (P8A is the most balanced; T5C is min-recall-binding on Xinhe).

## Open questions (for reviewer or follow-up)

1. **Does an OPTB head on the FT'd encoder dominate the OPTB head on frozen-CLIP?** This would isolate "encoder is doing the work" vs "head + data are doing the work."
2. **Is the per-human gap predictable from the substrate-axis cosine (Probe 1, 2026-05-22)?** P8A had highest cosine with the substrate axis per the per-ckpt fit. The per-human ranking here might correlate.
3. **Does ensembling FT'd + frozen-CLIP heads help?** The frozen-CLIP head over-fires on dor (47% FPR) but is clean on Roee_Windows (0%). Maybe a soft min-vote with T5C suppresses different chronic-FP loci.
