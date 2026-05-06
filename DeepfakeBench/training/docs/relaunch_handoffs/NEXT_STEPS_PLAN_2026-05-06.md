

> **Status:** living document. Sections 1-11 are the plan as of 2026-05-06 evening. Section 12 is an append-only activity log; results from CPU probes and GPU packets land there as they complete, and Sections 1-11 are revised from the log when results materially change priors. **Do not edit the log retroactively** — append only, and update sections 1-11 to reflect new state when warranted.

---

## 1. Circumstances that produced this document

This plan is the output of a deliberate three-way independent-review process held on 2026-05-06, in response to a structural blocker in the R13 sequence: 30+ packets attacking shortcut learning have each produced bounded weakening, none has cleared the 4/4 deployment-grade close criterion, and a fresh same-day-same-camera production false-flag (92 Xinhe real frames at deployment scores 0.83–0.93 captured 2026-05-06 11:06 UTC at `gs://live-fakes-teams-prod/real/session_20260506_125113/xinhe-may6-real-false-flag-1/`) confirms the pattern is operationally active.

**The three views being synthesised:**

1. **Team's existing ranked list** — `docs/relaunch_handoffs/ANTI_SHORTCUT_TECHNIQUES_RANKED_2026-05-06.md`. 1–9 anti-shortcut technique catalogue with merged ranking informed by Probes 1–7 and 30+ prior R13 packets. Authored by the team after the 4 afternoon CPU probes + Probes 6 & 7.

2. **In-house second-opinion (this agent, Round 1 + Round 2)** — given the bias-stripped facts pack at `docs/relaunch_handoffs/FACTS_FOR_SECOND_OPINION_2026-05-06.md` and asked to form an independent recommendation before being shown the team's synthesis. Round 1 = independent reading; Round 2 = comparison with team's list.

3. **External advisor (separate system, no repo access for the analysis pack itself)** — given the isolated pack at `docs/external_advisor_pack_2026-05-06/` (architecture doc + facts pack + raw CSVs/JSONs only), asked the same question with no exposure to team's synthesis. They had access to read the actual codebase, which surfaced a code-level finding neither the team nor the in-house second opinion identified.

**Two timing facts:**

- **PD verdict in flight.** `pd-corr-penalty-scorecard-2026-05-06` is running in us-east1 against the 8-entry ckpt map (`teams_target_domain.deeplive_viso_corr_2026-05-06.yaml`); verdict expected within hours of writeup. The verdict is the central pending factual update.
- **Production false-flag is acute.** Deployment ≡ E2B (Pearson r=+1.000 between deployment scores and local CPU E2B inference on may6 substrate, 92 frames; `analysis/xinhe_cross_camera_audit_2026-05-06/outputs/MODEL_SCORES_FINDINGS.md`). The user's "same person, different camera, opposite results" lived experience is E2B's behaviour, not P8A's. P8A is robust on may6 (FPR 0% vs E2B 57.6%).

The plan that follows synthesises all three views, with explicit acknowledgement of where they agree (high confidence to act on) and where they disagree (flag for explicit user decision). **Pair-ranking loss as a primary lever — surfaced only by the external advisor — is the highest-leverage missed item across all three views** and is gated on a $0 audit before any GPU spend.

---

## 2. The goal — three pillars

From `memory/project_success_criteria.md` and verbatim user statements:

> A single deployed model, single τ at inference, that holds simultaneously:
>
> 1. **Fake recall ≥ 90% on target methods** (deeplive, visomaster, teams)
> 2. **Real-side FPR ≤ 5%**
> 3. **Robustness across capture conditions** (lighting / camera / codec / colour)

All three are load-bearing.

**Hard constraints:**
- **Single deployed model** — ensemble not deployable.
- **Single τ at inference** — per-mode τ not deployable (Teams does not surface capture mode at inference; memory `feedback_per_mode_tau_not_deployable.md`).
- **No hard-negative mining** — user explicit; classified as patch, not structural fix.

---

## 3. Current state — quantitative snapshot (2026-05-06)

### 3.1 Per-suite Real FPR @ τ=0.5 (full pool, no chronic-6 exclusion)

| suite | n | P8A | E2B | PA_3800 |
|---|---:|---:|---:|---:|
| `teams_real_all_dev` | 4,295 | 12.9% | 10.7% | 13.5% |
| `teams_real_all_lockbox` | 1,418 | 6.8% | 6.3% | 18.1% |
| `teams_real_dor_dev` | 50 | 30.0% | 24.0% | 76.0% |
| `dor_morning` | 244 | 10.7% | 3.3% | 22.1% |
| `dor_evening` | 324 | 0.0% | 0.0% | 1.5% |
| `live_reals_teams_prod` | 677 | 0.4% | 1.2% | 1.3% |
| `team_sanity_may5` | 210 | 1.0% | 1.0% | 0.5% |
| `extra` (real subset) | 479 | 7.3% | 14.0% | 17.5% |
| **`xinhe_may6_falseflag` (NEW)** | **92** | **0.0%** | **57.6%** | **16.3%** |

### 3.2 Per-suite Fake recall @ τ=0.5

| suite | n | P8A | E2B | PA_3800 |
|---|---:|---:|---:|---:|
| `teams_fake_all_dev` | 1,620 | 93.9% | 94.8% | 96.4% |
| `teams_fake_all_lockbox` | 425 | 66.4% | 83.3% | 79.3% |
| `visomaster_v2_dor` | 2,073 | 73.8% | 56.0% | 25.2% |
| `live_fakes_teams_prod` | 1,675 | 75.3% | 81.1% | 50.4% |
| `dor_fake_local` | 605 | 92.6% | 97.0% | 98.7% |
| `extra` (fake subset) | 439 | 89.7% | 94.8% | 57.4% |

### 3.3 Production-distribution AUC (matched-domain, `live_*_teams_prod` only)

1,675 prod fakes vs 677 prod reals:

| ckpt | AUC | recall@FPR=5% | recall@FPR=10% |
|---|---:|---:|---:|
| P8A | 0.994 | 98.1% | 99.5% |
| E2B | 0.988 | 91.3% | 96.6% |
| PA_3800 | 0.952 | 67.8% | 83.0% |

### 3.4 Per-identity recall on xinhe-fake-* variants in `live_fakes_teams_prod`

| variant | n | P8A | E2B | PA |
|---|---:|---:|---:|---:|
| xinhe-fake-1 (no glasses) | 111 | 28.8% | 47.7% | 9.0% |
| xinhe-fake-2 (no glasses) | 86 | 16.3% | 51.2% | 1.2% |
| xinhe-fake-3 (no glasses) | 133 | 42.9% | 42.9% | 18.0% |
| xinhe-fake-4 | 104 | 67.3% | 94.2% | 51.9% |
| xinhe-fake-5 | 88 | 60.2% | 88.6% | 33.0% |
| xinhe-fake-6 | 82 | 81.7% | 67.1% | 15.9% |
| xinhe-fake-7 | 66 | 100.0% | 100.0% | 57.6% |
| xinhe-fake-8 | 107 | 85.0% | 98.1% | 29.9% |
| xinhe-fake-8-glasses | 123 | 80.5% | 91.1% | 29.3% |
| xinhe-fake-9-glasses | 60 | 61.7% | 96.7% | 36.7% |
| xinhe-fake-10-glasses | 73 | 76.7% | 97.3% | 37.0% |
| xinhe-fake-11-glasses | 44 | 65.9% | 95.5% | 59.1% |

(xiang-fake-* variants are uniformly ≥96% recall on all ckpts.)

### 3.5 Three-pillar status — no current ckpt meets all three simultaneously

- **P8A**: closest on substrate-invariance (0% may6 FPR, 0.4% live-prod FPR) but **lockbox fake recall 66.4% (need ≥90%)**.
- **E2B**: closest on lockbox fake recall (83.3%) but **57.6% may6 FPR**, and per-xinhe-fake-1/2/3 recall 16-48% (sub-pillar-1).
- **PA_3800**: F4 v2 winner (substrate-bound) but **HDTF cross-substrate collapse** (memory `project_pa_does_not_generalize_to_hdtf_2026-05-05.md`); 76% FPR on `teams_real_dor_dev`.

### 3.6 Probes 1–7 — load-bearing structural findings

- **Probe 1 (`xinhe_cross_camera_audit_2026-05-06/`)**: may6 vs may5 reals separable on raw IQ axes at 5-fold CV AUC=1.000. Top discriminators: `sat_std` (Cohen's d=−5.95, single-axis AUC=1.0; max(may6)=44.98 < min(may5)=55.85), `sobel_mean_face` (d=−4.33), `face_area` (d=+4.31), `lap_var_face` (d=−2.65). The shortcut is at the input level and chroma-loaded.
- **Probe 2 (`dor_drift_mechanism_2026-05-06/`)**: P8A score on Dor real moves `dor_evening` 0.015 → `teams_real_dor_dev` 0.331 (22.4× drift). Endpoint-union ridge attribution via `drift_attribution.csv` shows __PREDICTED_DRIFT__/__OBSERVED_DRIFT__ ≈ 0.064/0.316 (~20%); regression R² (`regression_p8a.json`) = 0.142 P8A, 0.336 E2B. **Note:** thread `processing_signature_shortcut.md` cites "90% recovered" and "R²_union=0.42-0.62" — these are different metrics from what the on-disk regression files show. **Open question — needs metric clarification (see §9).**
- **Probe 3 (`amp_vs_phase_probe_2026-05-06/`)**: 5-fold mean AUC amplitude=0.946, phase=0.861, pixel-baseline=0.671, shuffle=0.497. Both spectra carry separable manipulation signal; verdict MIXED-with-amplitude-bias.
- **Probe 4 (`paired_feature_consistency_2026-05-06/`)**: 275 (raw, teams) viso pair P8A cosine-distance distribution is bimodal — 47% <0.05 tight invariant, 11% >0.30 catastrophic-shift (max 0.92). CLIP-B16 baseline has 0% in the catastrophic tail. r(P8A pair-feat-distance, P8A pair-score-delta) = +0.697, OLS R²=0.485. r(P8A pair-feat-dist, E2B score-delta) = −0.089 (null). **E2B mean score raw=0.086 vs teams=0.172 (sign reversed from P8A).** Verdict: ENCODER_PARTIAL_INVARIANCE — ~47% head-fixable, ~33% encoder-bound, 11% catastrophic.
- **Probe 5 (deployment ≡ E2B)**: Pearson r=+1.000 between deployment scores and local CPU E2B inference on 92 may6 frames. P8A robust (0% FPR), E2B 57.6% FPR.
- **Probe 6 (`fourier_band_overlap_2026-05-06/`)** — bands 12-13 are CLEAN shortcut-only (shortcut AUC 0.97, signal AUC 0.46-0.52); bands 5-6 are signal-carrying with negative shortcut Δ; bands 8-10 are mostly safe. **Caveat: probe is on grayscale FFT.** May6 shortcut is chroma-loaded (sat_std, b_std, r_std are top discriminators), so grayscale-band findings may not transfer to a chroma-aware Fourier-aug recipe. **Open: chroma-band extension probe needed.**

  Per-band AUCs (full):

  | band | shortcut AUC | signal AUC | Δ | category |
  |---:|---:|---:|---:|---|
  | 0 | 0.66 | 0.63 | +0.03 | mixed |
  | 1 | 0.69 | 0.47 | +0.22 | mixed |
  | 2 | 0.85 | 0.69 | +0.15 | both moderate |
  | 3 | 0.91 | 0.73 | +0.18 | both moderate |
  | 4 | 0.85 | 0.74 | +0.11 | both moderate |
  | **5** | 0.63 | **0.72** | -0.09 | **PRESERVE — signal-carrying** |
  | **6** | 0.61 | **0.70** | -0.08 | **PRESERVE — signal-carrying** |
  | 7 | 0.89 | 0.68 | +0.21 | shortcut-leaning |
  | **8** | **0.98** | 0.66 | +0.32 | **mostly safe to randomise** |
  | **9** | **0.98** | 0.64 | +0.34 | **mostly safe** |
  | 10 | 0.90 | 0.62 | +0.29 | safe |
  | 11 | 0.83 | 0.59 | +0.24 | safe |
  | **12** | **0.97** | **0.52** | **+0.44** | **CLEAN — randomise** |
  | **13** | **0.97** | **0.46** | **+0.52** | **CLEAN — randomise** |
  | 14 | 0.83 | 0.54 | +0.28 | safe |
  | 15 | 0.63 | 0.55 | +0.09 | weak |

- **Probe 7 (`per_layer_p8a_e2b_pa_2026-05-06/`)**: shortcut AUC=1.000 at every encoder layer 0-11 across all 3 ckpts; manipulation signal climbs 0 → 0.998 by layer 8-9. **cos(P8A, E2B) per-layer p50: 0.998 at layer 0 → 0.954 at layer 6 → 0.855 at layer 10 → 0.320 at layer 11.** P8A and E2B share encoder representation through layer 9; catastrophic divergence is at layers 10-11 + head. PA tracks E2B at layer 11 (cos 0.77) but is as far from P8A as E2B is (cos 0.35). **Caveat:** AUC=1.000 at layer 0 is partly because the substrate is so cleanly separable on raw IQ that any feature extractor preserves it; this is substrate separability, not necessarily encoder shortcut-usage.

### 3.7 Prior R13 attempts ledger (factual, no interpretation)

| packet | base | single-lever delta | bottom-line outcome |
|---|---|---|---|
| RLP1-RLP6 | various | hint lanes, proper-data, arcface margin, gate alignment | RLP6_04 wins value_composite=0.9006 → 2-camera test reveals camera-signature shortcut |
| P8A | RLP7_02 | unfreeze visual.proj + ln_post + apply MLP-SVD | breaks anchor ceiling Δ=−0.188; regresses fake recall −13.6pp aggregate |
| P9-P12 | P8A | recipe tuning (soften, codec_hedge, HEAVY aug) | C3 codec_hedge: NOT a Phase D promotion candidate |
| P13 | scratch CLIP | anchor-aware loss + pipeline-random + face_scale_jitter@0.25 | γ verdict: cross-domain capability collapsed; modern_v2 FPR 30.2% |
| P14 (3 sister variants) | P8A_step5000 | jitter@0.50 ALONE wins trainer composite 0.661; bundle and DATA_FIX both <0.130 | leader `mclioexb`: zero τ in 5549-pt grid clears contract |
| P15 | bundle | DANN/GRL on quality-domain head, λ=0.20 static | bundle drag; weaker than jitter-isolated |
| P16 | E2B base | data-axis at fw=2.0 | does NOT promote, ranks 2-9 below P8A |
| P17 | various | layer-3 readout heads (ArcFace + LINEAR + others) | trained head systematically destroys substrate-invariance |
| P18 | E2B | 12-class method-conditional GRL | GRL preserves FT-induced regression, doesn't add invariance; P8A still wins |
| P22 | E2B | pipeline_randomization aug curriculum | step1k robust winner; step8k score variance collapsed 140× |
| S1/S2/S3 | P22 step1k | training-cap, earlier base, viso fw=8.0 | viso ceiling unbroken; S2 step600 wins teams_fake_lockbox at 91.5% |
| E1/E2B/E3 | scratch + CE | B16 (E2B), L14 (E3) | E2B breaks deeplive ceiling but viso REGRESSES 27%→7%; L14 doesn't break viso |
| Job 7 | P8A frozen features | head-only retrain (6 head variants) | all 6 over-fire 80-92% on lockbox reals; refuted |
| PA | E2B | visomaster_enhanced + visomaster_teams_enhanced data fw=4.0 | F4 v2 viso 72.4% (best in R13); HDTF 7.87% (collapse) — substrate-bound |
| PC | E2B | data + Teams codec aug | codec aug HURTS viso recall 35-50pp on F4 vs PA |
| **PD (in flight)** | E2B | correlation_penalty (Pearson) sharpness/luma/face_area, λ=1.0 | scorecard pending verdict 2026-05-06; close criterion 4/4 |

### 3.8 Cross-cutting bug + caveat list

- **`INTER_AREA → INTER_LINEAR` preprocessing parity bug** (WS-P0 fix at commit `855871e`); pre-fix retro-scores have silent kernel drift.
- **`apply_svd_to_in_proj` silent zero-gradient bug** pre-`2feea58` (2026-04-26); memory `project_in_proj_svd_gradient_bug.md`. **P8A_REFERENCE_STEP5000 training date relative to this commit is open question** (see §9 — external advisor flagged this).
- **`quality_enhancement` family-routing bug**: pre-2026-05-05 R13 packets trained on ~5,120 / ~18,880 deeplive_enhanced_fake frames (~27%) mislabeled. **PD is the first post-fix packet.** Every prior §3.7 verdict is potentially confounded.
- **Contract-policy v3 fix (recall floor)** is in working tree but uncommitted (open loop `contract-policy-bug-fix-not-committed`).

---

## 4. The three views — summary

### 4.1 Team's existing ranked list (1–9)

From `docs/relaunch_handoffs/ANTI_SHORTCUT_TECHNIQUES_RANKED_2026-05-06.md`:

1. **Fourier amplitude perturbation + consistency** (after mandatory amp-vs-phase pre-validation; bands 12-13 + 8-9 randomise, preserve 5-6).
2. **Customised + generic AugMix consistency (JSD)** — 60% custom-substrate branches + 40% generic ops; β=10-12 sweep.
3. **Self-Blended Images (SBI) pseudo-fake lane** — 40/40/20 real/known-fake/pseudo-fake mix; 3d eng.
4. **Test-Time Augmentation at inference** — free, deployable today.
5. **GroupDRO over (identity × capture-condition) groups** — outcome-equalisation.
6. **HSIC nonlinear penalty** (PD-shifted successor) — only if PD shows shortcut-shifting.
7. **ViT patch-aware masking + consistency** — fold into #2, not standalone.
8. **Stylized training (texture-bias breaking)** — deep bench.
9. **Lightweight adversarial augmentation on substrate axes** — last resort, 3-5× training cost.

Branching:
- *PD passes 4/4*: PE = AugMix on PD base; PF = SBI as follow-up.
- *PD muddles or shifts*: skip AugMix → PE = SBI on E2B base; PF = HSIC parallel; PG = AugMix on whichever wins.

### 4.2 In-house second-opinion (this agent, Round 2)

After comparison with team's list and reading FINDINGS.md / thread synthesis:

1. **TTA at inference + P8A-vs-E2B substrate-class disposition** — free, today, addresses may6 production pain immediately.
2. **PD verdict + chroma-band Probe 8 in parallel** — gates branching.
3. **Branch on PD outcome**: PD passes → AugMix on PD base, then SBI; PD muddles → skip AugMix to SBI on E2B, HSIC parallel.
4. **Band-limited Fourier-aug** as PG/PH — only after chroma extension and only single-lever.
5. **GroupDRO over (identity × capture-condition)** on bench.
6. **Withdrew Round-1 P8A→E2B layer-10/11 distillation** in favour of input-level + training-distribution levers, on the team's reasoning that an unconstrained head will find the shortcut at any layer.

Flagged additionally:
- **Chroma-band caveat** on Probe 6.
- **Apparent metric conflation in dor-drift writeup** (90% magnitude attribution vs ~14-34% R²).
- **`deployment-vs-p8a-substrate-tradeoff-not-quantified` open loop** deserves immediate disposition.

### 4.3 External advisor's recommendation

From `docs/external_advisor_pack_2026-05-06/` review with code access:

- **PE_PAIR_RANK_DRO** — **pair-ranking loss + multi-axis GroupDRO**, FT-from-P8A (post-`2feea58` codepath). Primary next packet.
- **PF_SBI** — target-domain self-blended pseudo-fakes (10-25% auxiliary mix).
- **PG_AUGMIX/JSD** — after pair-rank fixes objective geometry.
- **PH_FOURIER_BAND** — last; only after band-limited validation.

Six pre-launch CPU audits (mandatory, $0):
1. `PAIR_COVERAGE_AUDIT_2026-05-06`
2. `PAIR_GAP_AUDIT_P8A_E2B_PA_2026-05-06` — go/no-go for PE
3. `GROUP_ID_DESIGN_AUDIT_2026-05-06`
4. `CURRENT_CHECKPOINT_COHORT_DIAGNOSIS_2026-05-06`
5. `FROZEN_PAIR_HEAD_PROBE_2026-05-06` (optional, cheapest objective-signal test)
6. `SBI_SMOKE_2026-05-06`

Code-level finding: `combined_paired.py:3503` collate groups by `sample_id+label`; `effort_detector.py:1073` standard CE — loader is paired, **loss is not pair-aware**. Pair-rank formulation:

```
real_score = real_logit_fake - real_logit_real
fake_score = fake_logit_fake - fake_logit_real
pair_rank_loss = softplus(margin - fake_score + real_score)
```

Initial: `pair_rank_lambda` = 0.1 or 0.2; `margin` = 0.5 logit units.

Multi-axis GroupDRO:
- Fake-side: `label=fake × method_family × enhancer_family × transport × quality_band`.
- Real-side: `label=real × source × transport × quality_band × chronic_flag`.

Explicit "do not" list:
- Do not collect new data immediately.
- Do not start with naive raw/Teams consistency (prior refutation applies to that specific premise).
- Do not run another single-axis Pearson/HSIC nuisance penalty as main packet.
- Do not run global Fourier amp aug without band-limited validation.
- Do not do one-sided method/enhancer data boosts unless pair-balanced and cross-substrate gated.

Explicit P8A FT base bias: P8A is least bad under strict real constraints; historical P8A may not have benefited from the post-`2feea58` in-proj SVD fix path.

---

## 5. Convergence map

✓ = endorsed/recommended, ✗ = explicitly rejected, ~ = mentioned but neutral, blank = absent.

| Direction | Team | Me (R2) | External |
|---|:---:|:---:|:---:|
| **Real/fake pair-ranking loss (same-source)** | | | **✓ #PE primary** |
| Multi-axis GroupDRO (asymmetric fake/real keys, transport+quality+chronic) | ✓ #5 (id×capture, less granular) | ~ on bench | **✓ #PE primary** |
| Code-level: loader paired, loss not pair-aware | | | **✓ explicit** |
| `PAIR_GAP_AUDIT` ($0 go/no-go) | | | **✓ mandatory** |
| `PAIR_COVERAGE` / `GROUP_ID_DESIGN` / `COHORT_DIAGNOSIS` | partial (Job 11 done) | | **✓ mandatory** |
| `FROZEN_PAIR_HEAD_PROBE` | | | **✓ optional** |
| Self-Blended Images (target-domain, auxiliary) | ✓ #3 | ✓ adopted | ✓ #PF |
| Fourier band-limited amp aug | ✓ #1 | ✓ #4 deferred | ~ #PH last |
| AugMix/JSD consistency (60/40 hybrid) | ✓ #2 | ✓ adopted | ~ #PG later |
| TTA at inference | ✓ #4 free | ✓ #1 free | absent |
| Deployment-swap to P8A on may6 substrate | open loop, flagged | ✓ flagged | absent |
| HSIC nonlinear penalty | ✓ #6 conditional | ~ deferred | **✗ rejected as main packet** |
| Chroma-band Fourier extension probe | absent | ✓ flagged | absent |
| dor-drift R² metric clarification | absent | ✓ flagged | absent |
| FT base bias for next packet | E2B implicit | E2B implicit | **✓ P8A explicit** (in-proj SVD timing) |
| ViT patch-aware masking | ✓ #7 (folded) | absent | absent |
| Adversarial aug on substrate axes | ✓ #9 last | absent | absent |
| Stylized training | ✓ #8 deep bench | absent | absent |

---

## 6. Novel-to-external-advisor-only items (highest-value)

### 6.1 Real/fake pair-ranking loss as a primary structural lever

**Highest-leverage missed item.** Neither the team nor the in-house second-opinion proposed this. The team's `pair_loss_premise_refuted` memory was applied broadly; the advisor correctly narrows the refutation to **raw/Teams symmetric consistency on viso enhanced** (the actual prior probe's framing), and points out that **real-vs-fake-from-same-source ranking is a structurally different loss**.

Within a matched real/fake pair: identity, source video, pose, lighting, crop family, capture environment, frame index are held approximately constant. The residual is closer to "manipulation absent vs present." The current loss does not enforce this local ordering.

**Caveat (added by R2 synthesis):** pair-rank doesn't *suppress* the substrate shortcut, it just enforces local ordering. The model can still use substrate features for absolute scoring as long as pair members share substrate. So pair-rank is necessary-not-sufficient — it complements substrate-invariance levers (AugMix/Fourier/SBI/GRL) rather than replacing them.

### 6.2 Code-level diagnosis: loader paired, loss not pair-aware

The advisor traced the actual code path:
- `data/sources/combined_paired.py::CombinedPairedIterableDataset` — yields paired real/fake from same sample.
- `data/sources/combined_paired.py:3503` collate function — groups by `sample_id + label` but does not preserve explicit real↔fake relationship as a first-class field.
- `detectors/effort_detector.py:1073` — loss is standard CE; no pair-aware term.
- `trainer/mixins/group_dro.py` — current GroupDRO upweights method-level groups via `method_id`; too coarse for observed multi-axis failures.

This explains why pair-rank is a cheap *structural* change: the loader infrastructure exists and isn't being used.

### 6.3 `PAIR_GAP_AUDIT` as $0 go/no-go probe

For each matched real/fake pair on cached P8A/E2B/PA scores, compute `score(fake_same_source) − score(real_same_source)`, condition on missed-fakes vs caught-fakes vs FP-reals.

**Decision rule:**
- Missed fakes have pair_gap ≤ 0 on >25% → pair-rank has signal → run PE.
- Pair_gap > 0 already on missed fakes → pair-rank won't help → skip to SBI.
- Pair_gap excellent everywhere → skip PE entirely.

This is a $0 probe that decisively gates the next GPU-spend packet.

### 6.4 Multi-axis GroupDRO grouping richer than team's `identity × capture-condition`

Team's #5 is "GroupDRO over (identity × capture-condition)." Advisor proposes asymmetric fake/real keys:
- Fake = `method_family × enhancer_family × transport × quality_band`.
- Real = `source × transport × quality_band × chronic_flag`.

More directly captures failure modes documented in §3.4 (xinhe-fake-1/2/3 under-recall by enhancer/quality), §3.1 (chronic-6 over-fire by source/chronic_flag), Probe 2 (dor-cross-session drift by transport/quality_band).

### 6.5 P8A FT base argument from in-proj SVD bug timing

Memory `project_in_proj_svd_gradient_bug.md`: `apply_svd_to_in_proj` had silent zero-gradient pre-`2feea58` (2026-04-26). **P8A_REFERENCE_STEP5000 training date relative to this commit is unverified.** If pre-fix, retraining FT-from-P8A on the post-fix codepath has unmeasured headroom. **Cheap audit:** check the W&B run timestamp on the run that produced `gs://training-job-outputs/best_checkpoints/<P8A_run_id>/`.

### 6.6 Explicit "do not" list

Sharper than team's implicit version:
- Do not start with naive raw/Teams consistency.
- Do not run another single-axis nuisance penalty as main packet.
- Do not run global Fourier amp aug without band-limited validation.
- Do not do one-sided method/enhancer data boosts.
- Do not collect new data immediately.

---

## 7. Convergent vs divergent recommendations

### 7.1 Convergent (all three views endorse — highest confidence to act on)

1. **SBI / target-domain self-blended pseudo-fakes** as a near-term packet (team #3, R2 #3, external #PF). Strongest known cross-method generalisation lever; directly addresses the PA-on-HDTF substrate-bound walkback.
2. **Some form of GroupDRO-class richer-grouping intervention** (team #5, R2 bench, external #PE primary). All three see worst-group risk as the right framing for the same-person-drift observation.
3. **Single-lever discipline + cross-substrate validation in close criterion** — all three explicit. HDTF F4 gate must be in every packet's close criterion (PA-on-HDTF walkback cost a packet).
4. **PD verdict is the central in-flight gating event** — team explicit, R2 explicit, advisor acknowledges.
5. **No new data collection needed; existing data has the structure to test next hypotheses** — team implicit, R2 implicit, advisor explicit.

### 7.2 Divergent (flag for explicit user decision)

| Question | Team | R2 | External | Stronger evidence basis |
|---|---|---|---|---|
| What's the highest-priority lever after PD? | Fourier band-limited amp aug | TTA + audits + PD branch | **Pair-rank + multi-axis GroupDRO** | **External** — pair-rank fixes a structural objective gap not previously tested; team's Fourier is verified-but-incremental. The `PAIR_GAP_AUDIT` is a $0 disambiguator. |
| FT base for next packet | E2B implicit | E2B implicit | **P8A explicit** (in-proj SVD bug timing) | **External** — IF P8A was trained pre-`2feea58`. Verify timing first. |
| HSIC nonlinear penalty | ✓ conditional on PD shifting | ~ deferred | **✗ rejected** as main packet | **External** — §3.7 ledger shows nuisance-axis interventions consistently shift the shortcut; HSIC catches nonlinear dependence on the *same axes*, doesn't fix axis-shifting. Use as diagnostic, not main packet. |
| TTA + deployment swap to P8A | ✓ #4 free | ✓ #1 free | absent | **Team & R2** — addresses live production pain at $0; advisor missed because focused on training-side. |
| Chroma-band FFT extension before Fourier packet | absent | ✓ flagged | absent | **R2** — Probe 6 grayscale; may6 shortcut chroma-loaded (sat_std d=−5.95). Verify before any Fourier packet. |
| Fourier first vs pair-rank first | Fourier #1 | Fourier #4 deferred | Pair-rank first, Fourier last | **External + R2** — "consistency can stabilise the wrong shortcut unless pair/objective geometry is fixed first" matches team's bundle discipline applied to ordering. |
| dor-drift R² metric | "90% recovered" cited | flagged discrepancy | not engaged | **R2** — `regression_p8a.json` R²=0.142, `drift_attribution.csv` __PREDICTED/OBSERVED__ ≈ 0.20 vs thread cite "90%" / "R²_union 0.42-0.62". Three different numbers. Reconcile before priors flow to PD-successor design. |

---

## 8. Concrete plan

### 8.1 Phase 0 — TODAY, all CPU, $0 (high confidence in value: ~90%)

> **Phase 0 status as of 2026-05-06 evening:** 6 of 8 jobs dispatched in parallel sub-agent batch and complete; 2 deferred (TTA, SBI smoke); 1 added post-hoc (0g, load-bearing go/no-go for P1). Verdict summary inline below; full results in §12 log entries and `analysis/<job>_2026-05-06/outputs/FINDINGS.md`.

These are constructive (designing the next packet from what data contains) rather than reactive (gating already-named recipes). All are independent and can run in parallel.

**0a. TTA at inference + deployment-swap disposition for may6 substrate-class.** ✅ COMPLETE (TTA POC). **Verdict: NOT VIABLE.** E2B may6 FPR 57.6% → 56.5% (Δ −1.1pp; net rescue ≈ 1 frame). may6 misclassifications are confident (median fake-prob 0.58, view-std 0.10) — TTA has no leverage. P8A may6 invariance preserved (0% → 0%). Drift-named-axis-ridge framing explains the failure: pixel-jitter operates on the wrong axis. Deployment-vs-P8A-substrate-tradeoff loop's remaining levers are deployment-side P8A swap on may6-class substrates + IQ gating (P8A-specific). See `analysis/tta_poc_may6_2026-05-06/outputs/FINDINGS.md` for full data.
- Implement N-view TTA (e.g., flip + 3 light geometric jitters, score-average) on both P8A and E2B.
- Re-score `xinhe_may6_falseflag` (n=92) and `live_*_teams_prod`.
- Write disposition for open loop `deployment-vs-p8a-substrate-tradeoff-not-quantified`: which substrate-classes get P8A, which get E2B (or single-model with TTA).
- Damps may6 production pain in hours.
- Owner: TBD. Output: `analysis/tta_inference_2026-05-06/outputs/`.

**0b. External-advisor audit suite** (priority order; ✅ 4 of 6 complete, 2 deferred):
- ✅ `PAIR_GAP_AUDIT_P8A_E2B_PA_2026-05-06` — RAN. **Verdict: RED on cross-product eval pairs (P8A 9.88% / E2B 5.63% / PA 17.79% inversion rate among missed fakes; threshold 25%). BUT verdict is partial** — the audit used loose cross-product within identity, not the tight same-source frame pairs the training loop emits. **The decisive go/no-go is now Phase 0g** (`SAME_SOURCE_PAIR_GAP_AUDIT`). Per-subject heterogeneity severe: dor_local 27.6% (GREEN), viso_v2_inswapper raw 30-49% (GREEN), all teams identities 0-8% (dead).
- ✅ `PAIR_COVERAGE_AUDIT_2026-05-06` — RAN. **Verdict: ~91-97% of training is paired uniformly across all 6 lanes; same pair key `(sample_id, frame_idx)` opposite-label.** Recommendation: **apply pair-rank UNIFORMLY** across all 6 paired lanes; only `external_vcd_real` (1,200 frames) needs `is_unpaired_real=True` skip. The 2026-05-04 `pair_loss_premise_refuted` memory does NOT transfer (different premise).
- ⏸ `GROUP_ID_DESIGN_AUDIT_2026-05-06` — DEFERRED to second wave (depends on PAIR_COVERAGE outputs which are now in).
- ✅ `CURRENT_CHECKPOINT_COHORT_DIAGNOSIS_2026-05-06` — RAN. **Verdict: FT base for PE_PAIR_RANK_DRO = P8A. HIGH confidence.** P8A wins 7/9 cohort axes; `real_FP_fake_missed` (cell pair-rank targets) at τ=0.5: P8A 0.0% / E2B 3.2% / PA 26.3%. Production-decisive cohorts favour P8A by 0.4-0.7 absolute failure-rate margin. E2B's narrow wins are real-FPR-only and reclaimable via multi-axis GroupDRO real-side key (`source × transport × quality_band × chronic_flag`).
- ⏸ `FROZEN_PAIR_HEAD_PROBE_2026-05-06` — DEFERRED to second wave (needs feature extraction; superseded in importance by 0g).
- ⏸ `SBI_SMOKE_2026-05-06` — DEFERRED to second wave (needs face-landmark + blending pipeline setup).

Outputs: `analysis/{pair_gap_audit,pair_coverage_audit,checkpoint_cohort_diagnosis}_2026-05-06/outputs/{*.csv, summary.json, FINDINGS.md}`. Deferred jobs will land in `analysis/{group_id_design_audit,frozen_pair_head_probe,sbi_smoke}_2026-05-06/`.

**0c. P8A training-date verification.** ✅ COMPLETE. **Verdict: PRE-FIX, HIGH confidence.**
- P8A_REFERENCE_STEP5000 (run `9lmvb5b4`) trained 2026-04-24 21:41 UTC → 23:48 UTC, 43h 25m BEFORE commit `2feea58` (2026-04-26 19:13 UTC).
- Three independent in-repo sources confirm pre-fix classification.
- **Implication:** `apply_svd_to_in_proj=true` was zero-gradient throughout P8A's original training. FT-from-P8A on the post-fix codepath has unmeasured headroom — argued by external advisor as a structural reason to choose P8A as FT base.
- **Caveat:** post-fix C-ablation tied within noise on P10_SYM-on-P8A recipe — under THAT recipe the in-proj-SVD lever doesn't materially move. Under PE_PAIR_RANK_DRO's loss it's plausible-but-unverified; **include grad-audit at step 100/500/1000** in PE design.

**0d. Chroma-band Fourier extension probe.** ✅ COMPLETE. **Verdict: CHROMA_DOMINATES_DIFFERENT_BANDS** — chroma extension is REQUIRED.
- R/G/B/L (luminance-like): cleanest at bands 9-13 (matches grayscale Probe 6).
- **lab_a:** cleanest at bands **1, 6, 7, 8, 9, 10**; argmax shortcut band 7 (AUC 0.935); **NO signal-carrying bands** (max signal AUC 0.58).
- **lab_b:** cleanest at bands **6, 7, 8**; argmax shortcut band 7 (AUC 0.921); **NO signal-carrying bands** (max signal AUC 0.70).
- **Grayscale-only band masks miss the chroma shortcut at bands 1, 6, 7.** The grayscale-implicit recipe sketched in original §8.2 P3 is insufficient.
- Probe 1's named axes (sat_std d=−5.95, b_std d=−2.61) align directly with lab_a/lab_b bands.
- **Recipe update for `PE_FOURIER_BAND` (P3) — see §8.2.**

**0e. dor-drift R² metric reconciliation.** ✅ COMPLETE. **Verdict: no contradiction in existing docs** — three metrics answer three different questions.
- R²=0.142 = within-frame score variance, NOT cross-session drift.
- 0.064/0.316 ≈ 20% = all-sessions-fit projection (under-estimate due to omitted-fixed-effect regime).
- **0.285/0.316 = 90%, R²_union 0.42-0.62** = endpoint-pair-only ridge — **this is the right number for "X% of drift explained by named axes."**
- Top single drivers on P8A by univariate Pearson r: **`min_dim`** (resolution, |r|=0.57), **`edge_mag`** (|r|=0.45), **`color_b_dev`** (B-channel cast, |r|=0.43).
- **Important new finding:** PD currently targets only `sharpness_lap` + `luma_mean` + `face_area_fraction`. The dor-drift dominant axes `min_dim` and `color_b_dev` are UN-targeted by PD — these are the high-impact axes likely to absorb the shifted gradient when PD's penalty bites. Agent 5 (chroma-band Fourier) independently confirmed lab_b is a load-bearing chroma shortcut channel; the two findings converge. **Recommendation:** PD-class successors (HSIC, AugMix-with-corr-pen, etc.) should include `min_dim` and `color_b_dev` in penalty axes.
- **Caveat:** 90% means "drift lives in a low-dim named-IQ subspace," NOT "named IQ is the causal driver." Necessary not sufficient for PD-class success.

**0f. Wait for PD scorecard verdict.** ✅ Scorecard artifacts AVAILABLE LOCALLY at `analysis/pd_scorecard_artifacts_2026-05-06/` (~42 MB). 232 cells = 8 ckpts × 29 suites complete via two-phase run (original disk-leak failed; resume on 500GB boot disk succeeded). Headline data: `unified_scorecard_simple.csv` (232 rows, τ=0.5 diagnostic-only metrics: real_fpr_at_0p5, fake_recall_at_0p5, accuracy_at_0p5). **Conclusions NOT yet drawn** per user instruction; user will direct when to analyse. **Scorecard scope caveat:** answers F1 for Teams suites only — does NOT include F1-lockbox for `visomaster_enhanced_macro` / `deeplive_enhanced` / `teams_real_dor` lockbox; does NOT include F2 (shortcut weakening), F3 (no-axis-amplify), F4 (HDTF cross-substrate). All four close-criterion gates require separate audits beyond this scorecard. See `analysis/pd_scorecard_artifacts_2026-05-06/README.md`.

**0g. SAME_SOURCE_PAIR_GAP_AUDIT_2026-05-06.** ✅ COMPLETE. **Verdict: MIXED, leaning LANE_RESTRICTED_LAUNCH on FT-from-P8A.**
- Cache-first audit: 0 tight `(sample_id, frame_idx)` pairs in any local score cache (training-only buckets are not cached).
- Cross-product proxy mapped to paired-lane semantics: 2 of 6 paired lanes measurable.
  - **viso_enhanced GREEN (28.1%, n_pairs=2,869).**
  - **deeplive AMBER (23.5%, n_pairs=847, n_missed=68).**
  - df40, viso_v1, viso_teams_enh, deeplive_teams: INSUFFICIENT_DATA.
- E2B RED on both measurable lanes (6.9% / 8.6%) — independently confirms FT-base = P8A.
- **Critical finding:** the cross-product Agent 1 RED (9.88% on P8A) was driven by `teams_fake_all_dev` (eval-only substrate, NOT a paired training lane). On training-substrate proxies, P8A flips RED → GREEN/AMBER. **The cross-product RED was misleading.**
- §8.2 strict gate ("≥2 GREEN paired lanes") not met from cache, but ZERO RED among measured.

**0h+0j. COMBINED PAIRED-FEATURE EXTRACTION + TIGHT-PAIR AUDIT.** ✅ Vertex extraction COMPLETE. Job `3077166152858730496` SUCCEEDED in **3 minutes** (16:47:25 → 16:50:27 UTC; A100 much faster than the ~40 min estimate). All 4 outputs verified in `gs://training-job-outputs/analysis_outputs/frozen_pair_features_2026-05-07/`:
- `p8a_paired_features.npz` (6.87 MiB)
- `e2b_paired_features.npz` (7.09 MiB)
- `clip_b16_raw_paired_features.npz` (6.82 MiB)
- `frame_manifest.csv` (901 KiB)

Total 21.66 MiB. Schema: 512-d post-`visual.proj` [CLS] features. **Ready for the post-extraction processing agent** (per `analysis/path_a_launch_2026-05-07/RESULTS_PROCESSING_PLAN.md` 7-step runbook): pull NPZs locally → re-run 0j tight-pair audit on df40+deeplive → re-run 0h frozen_pair_head_probe → decide P1 path with promotion gates.

**0h. PAIRED_FEATURE_EXTRACTION_2026-05-07.** 🆕 NEEDED to unblock FROZEN_PAIR_HEAD_PROBE. (NOW EXECUTING via combined 0h+0j Vertex job above.)
- Cost: ~$5-10, ~1-2 A100-hours on us-east1 (or local A100).
- Reuse `analysis/feature_space_2026-04-23/extract_features.py` template; path-list from `analysis/pair_gap_audit_2026-05-06/outputs/pair_gaps.csv` (5,311 unique frame paths).
- Output: `analysis/frozen_pair_features_2026-05-07/{p8a, e2b, clip_b16_raw}_paired_features.npz` (~32 MB per ckpt).
- Then re-run `analysis/frozen_pair_head_probe_2026-05-06/run_probe.py` with the new caches → tells us if pair-rank signal is head-side (cheap retrain) or encoder-side (full FT).
- Promotion gate: Head B ≥3pp lift over Head A → demote P1 to head-only retrain; no lift → P1 justified iff 0g GREEN.

**0i. SBI_SCORE_COLLECTION_2026-05-07.** ✅ COMPLETE. **Verdict: GREEN-by-direct-measurement.** All 3 ckpts strictly bimodal (Sarle BC > 0.555); P8A sharpest with 27% in the (0.1, 0.9) training sweet spot. 98%/98.5%/98.5% of pairs shift toward fake. Score-side Cohen's d on P8A = 0.81 vs IQ-side 0.047 (17× larger) — encoder picks up manipulation cues pixel-IQ axes miss (the SBI thesis). **Design note:** oversample SBI > 0.5 tail in PE_SBI training. Local CPU inference, all ckpts found locally cached. See `analysis/sbi_score_collection_2026-05-07/outputs/FINDINGS.md`.
- Cost: ~10 min CPU + ~90 MB ckpt downloads, near-$0.
- Re-uses `analysis/xinhe_cross_camera_audit_2026-05-06/run_local_inference.py` with `RAW_DIR` pointing at `analysis/sbi_smoke_2026-05-06/outputs/pseudofakes/`.
- Score the 200 SBI pseudofakes on P8A/E2B/PA_3800 to upgrade SBI_SMOKE GREEN-by-headroom → GREEN-by-direct-measurement.
- Confirms SBI scores land in (0, 1) intermediate range with bimodality, validating P2 launch readiness.

**0j. FORWARD_PASS_TIGHT_PAIR_AUDIT_2026-05-07.** 🆕 follow-up to 0g (Option A in 0g recommendation).
- Cost: ~$3, ~40 min on us-east1 (or local A100).
- Frame-list extraction from `data/sources/df40_paired.py` and `data/sources/deeplive_paired.py` loaders.
- Output: `analysis/forward_pass_pair_audit_2026-05-07/{p8a, e2b}_paired_scores.npz` with `(sample_id, frame_idx, label, score)` per row.
- Re-run `analysis/same_source_pair_gap_audit_2026-05-06/run_probe.py --tight-pairs <new_csv>` for definitive 4/6 lane coverage.
- Resolves df40 (largest unmeasured lane, 4,698 base samples) and direct deeplive measurement.
- **0h and 0j can be combined into a single forward-pass extraction step** since they share infrastructure (same loaders + ckpts; only output schema differs).

### 8.2 Phase 1 — Conditional packet selection (1 of 3, GPU)

#### P1: PE_PAIR_RANK_DRO

> **Status as of 2026-05-06 evening (post-Phase-0-second-wave): LANE_RESTRICTED_LAUNCH VIABLE on FT-from-P8A** with the choice of two paths to commit. The cross-product RED was misleading — driven by `teams_fake_all_dev` eval substrate (not a paired training lane). On training-substrate proxies: **viso_enhanced GREEN (28.1%), deeplive AMBER (23.5%), 4 lanes INSUFFICIENT_DATA, ZERO RED among measured.** FROZEN_PAIR_HEAD_PROBE returned INSUFFICIENT_COVERAGE — head-vs-encoder localisation requires a small extraction.
>
> **Two paths to launch:**
>
> 1. **Path A — Resolve Phase 0h + 0j first (~$8-13 / ~1 hr on us-east1 or local A100), then launch P1 with full info.** Phase 0h extracts paired features at training-pair indices (~$5-10) and re-runs FROZEN_PAIR_HEAD_PROBE → tells us if pair-rank signal is head-side (cheaper retrain) or encoder-side (full FT). Phase 0j runs the forward-pass tight-pair audit on df40 + deeplive (~$3) → converts 2/6 → 4/6 lane coverage with definitive verdict. Both can run in a single combined extraction pass.
> 2. **Path B — Lane-restricted P1 launch now** with pair-rank applied uniformly (per Agent 2 coverage finding: loss=0 on RED-lane batches is harmless). Run **P2 (PE_SBI) in parallel** as the no-regret structural alternative. Both packets FT-from-P8A on the post-`2feea58` codepath. df40 (largest unmeasured lane) carries unknown-direction risk.
>
> **Recommendation:** if time-constrained → Path B (P1 + P2 in parallel). If not time-constrained → Path A first, then Path B with confidence.

- **Trigger:** `SAME_SOURCE_PAIR_GAP_AUDIT_2026-05-06` (Phase 0g) shows missed-fakes have pair_gap ≤ 0 on >25% per-lane on at least 2 of {DF40, DeepLive, VisoMaster_v1, VisoMaster_enhanced, VisoMaster_teams_enhanced, DeepLive_teams}. If only viso/deeplive raw lanes show signal (consistent with cross-product audit's heterogeneity finding), P1 is still viable but with restricted scope.
- **FT base: existing `P8A_REFERENCE_STEP5000`** (run `9lmvb5b4`, step 5000) — HIGH confidence per `CHECKPOINT_COHORT_DIAGNOSIS` Agent 3 (P8A wins 7/9 cohort axes; `real_FP_fake_missed` cell at τ=0.5: P8A 0.0% / E2B 3.2%; production-decisive cohorts favour P8A by 0.4-0.7 absolute failure-rate margin). Training runs on the **post-`2feea58` codepath**, which activates `in_proj_svd` q/k/v classification gradient for the first time on this lineage. **No prerequisite clean retrain of P8A** — FT directly from the existing checkpoint.
- **Recipe:**
  - Pair-ranking loss: `pair_rank_loss = softplus(margin − (fake_score − real_score))` at λ=0.1-0.2, margin=0.5 logit units. Apply on pair-aligned items per the new `pair_id = (sample_id, frame_idx)` collate field.
  - **Apply UNIFORMLY across all 6 paired lanes** (per `PAIR_COVERAGE_AUDIT` Agent 2; pair-rank loss=0 on RED-lane batches is harmless per Agent 2 finding). The `external_vcd_real` lane gets a one-line skip rule (`is_unpaired_real=True` → CE only, no pair-rank contribution).
  - **Multi-axis GroupDRO with the empirically-validated (F-B, R-D) keys** (per Agent `GROUP_ID_DESIGN_AUDIT`):
    - **Fake-side F-B:** `label=fake | method_family | enhancer_family`. (Advisor's full proposal `method × enhancer × transport × quality_band` was DEMOTED — quality_band degenerates because 67% of fake rows have `quality=unknown`; transport adds only 1pp spread for too-fragmented groups.)
    - **Real-side R-D:** `label=real | source | transport | quality_band | chronic_flag`. (Advisor's full proposal — `chronic_flag` is the load-bearing axis, lifts real-side spread from 0.11 to 0.61.)
    - 27 groups total. Min group=68, median=381, max-share=0.184. Passes all DRO-stability thresholds.
    - **Chronic-6 list:** `bla_bla_chow`, `bla_bla_chow__s2`, `PC_Generator__s22`, `PC_Generator__s45`, `roy_d`, `Q__s6` (1,297 real rows match; zero fake rows).
    - Implementation: `trainer/mixins/group_dro.py` `method_mapping` → `group_id_mapping`. Default `ema_alpha=0.1`, `beta=3.0`, 100-step EMA warmup. Ready-to-paste python snippet at `analysis/group_id_design_audit_2026-05-06/outputs/group_id_python_snippet.py`.
  - **Add `pair_id` as first-class field** in `combined_paired_collate_fn` (currently groups by `sample_id+label`; need explicit pair link).
  - **Grad-audit hook at step 100/500/1000 — plumbing-check only, NOT a promotion gate.** Confirms `apply_svd_to_in_proj=true` delivers nonzero classification gradient on q/k/v residuals under the pair-rank+DRO loss. **Binary semantics: zero gradient → abort the run** (the `2feea58` fix has regressed silently); nonzero → continue, regardless of magnitude. Per `packets/P8A.md:108`, the post-`2feea58` C-ablation tied within noise vs C.1 on the P10_SYM-on-P8A recipe — under that one FT recipe `in_proj_svd` does not materially move the canonical P8A configuration. So if grad-audit shows nonzero gradient but downstream F1-F5 metrics don't lift, that's consistent with the lever being unhelpful for this recipe (rather than a regression of the fix). Promotion is gated on F1-F5 metrics only.
  - All other new levers OFF (no Fourier, no AugMix, no SBI, no codec aug, no HEAVY aug). Single-lever discipline.
- **Close criterion (revised per cohort-diagnosis findings):**
  - **F1:** lockbox fake recall ≥90% at FPR ≤10%.
  - **F2:** pair-ranking metric — fraction `fake_score > real_score` on previously-missed fakes lifts ≥30% on at least 2 of the 6 paired lanes; worst-group recall lift ≥20% on chronic / dor-drift cohorts identified by Agent 3 (`gpen` enhancer 46.7% missed, `inswapper` 33.1% missed, `dor_evening_morning` 17%, `live_fakes_teams_prod` 24.7%, `visomaster_v2_dor` 26.2%).
  - **F3:** no untargeted axis (is_webcam, face_area_fraction, **min_dim, color_b_dev**) amplifies +50%. Note: `min_dim` and `color_b_dev` added to the F3 list explicitly because Agent 6 identified them as the un-targeted dor-drift dominant axes.
  - **F4:** HDTF cross-substrate FPR ≤5%.
  - **F5:** the chronic-FP `pc_generator` cluster failure rate (P8A baseline 0.520) drops by ≥0.10 absolute via the GroupDRO `chronic_flag` term.
- **Confidence in single-packet three-pillar closure: 20-30%** if 0g greenlights. **Probability drops to ~5-10%** if 0g returns RED at training-pair indices (would mean pair-rank has no signal even on the tight pairs).

#### P2: PE_SBI

> **Status as of 2026-05-06 evening: ELEVATED to co-primary candidate.** Given Agent 1's cross-product RED verdict and the latency on Phase 0g, SBI is the no-regret structurally-novel packet that doesn't depend on pair-structure existing in training data. Run alongside or before P1 depending on 0g timing.

- **Trigger:** any of:
  - PD muddles OR PD shifts onto un-targeted axes (`min_dim`, `color_b_dev`).
  - 0g `SAME_SOURCE_PAIR_GAP_AUDIT` returns RED → P1 is dead → P2 takes over as primary.
  - P1 caps without three-pillar closure → P2 follows.
  - User chooses to run no-regret structural intervention first.
- **FT base: P8A** (per Agent 3 cohort-diagnosis recommendation — same reasoning as P1: P8A wins 7/9 cohorts; chronic real-FPR cohorts where E2B currently wins are reclaimable via training-time intervention).
- **Recipe:** target-domain self-blended pseudo-fakes at 10-25% auxiliary mix; face landmark detection via MediaPipe (already in repo); blending boundary as the universal artifact signal. Starting mix ratio 40/40/20 (real / known-fake / pseudo-fake) with sweep at 30/30/40 and 50/35/15 endpoints.
- **Single-lever:** pseudo-fake lane only; no AugMix/Fourier/pair-rank in same packet.
- **Pre-validation:** run `SBI_SMOKE_2026-05-06` (deferred Phase 0 task) on 100-300 target-domain real crops to confirm visual plausibility and IQ-axis separability before committing 3d engineering.
- **Close criterion:**
  - **F1:** lockbox fake recall ≥90% at FPR ≤10%.
  - **F4:** HDTF cross-substrate FPR ≤5% (load-bearing — SBI's claim is method-agnostic generalisation; this is the gate that PA failed and led to the substrate-bound walkback).
  - **F5:** per-method-family recall on held-out fake families lifts ≥15% on at least 2 families.
  - **F6:** chroma-band shortcut weakening as a side effect — IQ-axis correlations on `color_b_dev` and `min_dim` (the un-PD-targeted dominant axes) drop ≥20%.
- **Confidence: 20-30%.** Strongest literature evidence for cross-method generalisation; novel to this codebase. SBI's blending-boundary signal is conceptually orthogonal to the chroma-domain shortcut, so it may also independently weaken the may6-style failure mode.

#### P3: PE_FOURIER_BAND

> **Status as of 2026-05-06 evening: recipe revised per Agent 5 chroma-band findings.** Grayscale-only band randomisation would leave the chroma shortcut at bands 1, 6, 7 intact and shrink the ratchet. Per-channel chroma extension is REQUIRED.

- **Trigger:** PD passes 4/4 AND P1/P2 not yet pursued (or have capped). NOT the highest priority packet — P1 (if 0g greenlights) and P2 (SBI) should run first.
- **FT base:** P8A (per cohort diagnosis) post-`2feea58` codepath; or PD-base if PD passes 4/4.
- **Recipe (revised per Agent 5):** per-channel band-limited FFT amplitude same-label randomisation. BGR → LAB at training time (per-frame, in the augmentation transform); apply per-channel:
  - **R / G / B / L:** randomise bands {8, 9, 10, 12, 13}, preserve bands {5, 6}.
  - **lab_a:** randomise bands {1, 6, 7, 8, 9, 10}, no preserve required (no signal-carrying chroma_a bands).
  - **lab_b:** randomise bands {6, 7, 8}, no preserve required (no signal-carrying chroma_b bands).
  - LAB → BGR back. Phase preserved universally on all channels.
  - Optional JSD consistency between original and band-randomised views (β=10-12 sweep if used).
- **Single-lever:** Fourier-aug only; no AugMix/SBI/pair-rank in the same packet.
- **Close criterion:**
  - **F1:** lockbox fake recall ≥90% at FPR ≤10%.
  - **F2:** shortcut weakening — IQ-axis Pearson correlations down ≥30% on ≥2 of 5 axes. **Add `min_dim` and `color_b_dev` to the F2 axis list explicitly** (per Agent 6 — these are the un-PD-targeted dor-drift dominant axes).
  - **F3:** no untargeted axis amplifies +50%.
  - **F4:** HDTF cross-substrate FPR ≤5%.
- **Confidence: 20-25%.** Probe 6 + chroma extension are real evidence but univariate; manipulation-signal preservation in bands 8-11 is a real risk (signal AUC 0.59-0.66 on grayscale, not negligible). Chroma channels have NO signal-carrying bands so chroma randomisation is safer per-channel than luminance.

### 8.3 Default sequencing (REVISED 2026-05-06 evening per Phase 0 results)

```
Phase 0 — 6 of 8 jobs complete; 2 deferred + Phase 0g added
  ✅ 0c P8A pre-fix verdict
  ✅ 0d Chroma-band cleanest-cell bands per channel
  ✅ 0e dor-drift R² metric reconciliation
  ✅ Cross-product PAIR_GAP (RED on aggregate, AMBER on subsets)
  ✅ PAIR_COVERAGE (~91-97% paired, uniform application)
  ✅ COHORT_DIAGNOSIS (FT base = P8A, HIGH confidence)
  ⏸ 0a TTA + deployment-swap disposition (still TODO)
  ⏸ Deferred: GROUP_ID_DESIGN, FROZEN_PAIR_HEAD_PROBE, SBI_SMOKE
  🆕 0g SAME_SOURCE_PAIR_GAP_AUDIT (load-bearing for P1)
  ⏳ 0f PD scorecard (in-flight, hours)

Branch on:
  (1) Does 0g greenlight P1?
    - YES (≥25% pair-gap inversions on missed-fakes per-lane on ≥2 lanes):
        Run P1 PE_PAIR_RANK_DRO on FT-from-P8A-post-`2feea58`.
        Then P2 PE_SBI as follow-up if P1 caps.
    - NO (cross-product RED corroborated at training-pair indices):
        Run P2 PE_SBI on FT-from-P8A as primary.
        Then P3 PE_FOURIER_BAND-CHROMA only if F4-cross-substrate not met.
  (2) Does PD pass 4/4?
    - YES: PD becomes next deployment candidate; P1/P2 follow as additive single-lever extensions on PD base.
    - NO (muddles or shifts): no change to (1) — PD is supplemented or replaced.
  (3) Does the user prefer the no-regret SBI path over waiting for 0g?
    - YES: P2 first; P1 deferred until 0g returns.
```

**Sequencing principle:** objective fix (pair-rank) before training-distribution change (SBI) before frequency-domain regularisation (Fourier). External advisor's argument — "consistency can stabilise the wrong shortcut unless objective geometry is fixed first" — applies team's bundle discipline to ordering of single-lever packets. **Caveat from Phase 0:** if pair-rank's signal at training-pair indices is RED (Phase 0g), the objective-fix premise is dead and SBI becomes primary.

**Scratch-as-parallel-ablation-arm option (user's question, deferred):** with FT-from-P8A now strongly anchored by Agent 3 cohort diagnosis (P8A wins 7/9 cohort axes), the empirical case for FT is solid. Scratch-on-clean-data + pair-rank + multi-axis GroupDRO + SBI auxiliary remains theoretically possible and is not refuted, but the §3.7 ledger 0/3 batting average on scratch + the strong P8A cohort lead make it lower-priority. **Recommendation:** keep scratch on the bench as a possible parallel arm to whichever P1/P2 lands first; do not prioritise as primary.

### 8.4 Realistic three-pillar prognosis

- **Single-packet probability of meeting all three pillars at deployment: 15-25%** — unchanged from R2 estimate. The advisor's pair-rank lever adds a credible structural attempt that wasn't on the table, but the breadth of the three-pillar gap means single-packet closure remains improbable.
- **Multi-packet sequenced probability over 2026-05-06 to 2026-05-15: 40-55%** if Phase 0 + at least P1 and P2 are run.

---

## 9. Open questions and loose ends

**Resolved 2026-05-06 evening (kept here as reference, see corresponding §12 log entries):**

- ✅ ~~**`deployment-vs-p8a-substrate-tradeoff-not-quantified`**~~ — partially addressed by Agent 3 `CHECKPOINT_COHORT_DIAGNOSIS`: P8A wins 7/9 cohort axes; E2B's narrow wins are real-FPR-only. Full disposition still requires running TTA (0a) for the deployment-swap option. **Status: CLOSED on training-side question; OPEN on deployment-side TTA question.**
- ✅ ~~**dor-drift R² metric reconciliation.**~~ — Agent 6 confirmed all three numbers (14% / 20% / 90%) are correct, answering different questions. The headline "90% of cross-session drift explained by named axes" via union-fit is the right cite. **Closed.**
- ✅ ~~**P8A_REFERENCE_STEP5000 training date relative to `2feea58`**~~ — Agent 4 verdict: PRE-FIX, HIGH confidence. P8A trained 43h before fix. **Closed.**
- ✅ ~~**Chroma-band Fourier structure**~~ — Agent 5 found per-channel structure differs from grayscale; lab_a / lab_b have shortcut at bands 1, 6-10 with NO signal-carrying preserve bands. P3 recipe revised. **Closed.**

**Newly opened (2026-05-06 evening):**

- 🆕 **`same-source-pair-gap-audit-pending`** (load-bearing for P1, severity HIGH). Cross-product PAIR_GAP_AUDIT returned RED on the eval substrate, but the verdict is partial — the actual training loop emits tight same-source frame pairs (`(sample_id, frame_idx)` opposite-label) that were NOT measured. P1 is AMBER-conditional until Phase 0g returns. Per-subject heterogeneity (dor_local 27.6%, viso_v2_inswapper raw 30-49% on P8A) suggests there may be lane-specific signal even if aggregate is dead.
- 🆕 **`pd-class-undertargeted-axes`** (medium). PD targets `sharpness_lap` + `luma_mean` + `face_area_fraction`. Agent 6 found `min_dim` (resolution) and `color_b_dev` (B-channel cast) are the dor-drift dominant axes, both UN-targeted. PD-class successors should include them; this also ties to Agent 5's chroma-band finding (lab_b is a load-bearing shortcut channel).
- 🆕 **`scratch-deferred-pending-FT-from-P8A-plateau`** (deferred, user-confirmed 2026-05-06 evening). User's standing preference is scratch over FT-from-massively-FT'd checkpoint, BUT given Agent 3 cohort diagnosis (P8A wins 7/9 cohorts) and §3.7 ledger 0/3 batting average on prior scratch attempts, the agreed approach is: **try to make gains on FT-from-P8A first; if FT-from-P8A reaches a plateau, then revisit scratch as the next axis.** Documented here so a future agent picking up the work knows scratch is on the deferred bench, not refuted. **Trigger to revisit:** a P1/P2/P-* packet hits the 4/4 close criterion with marginal lift, OR FT-from-P8A explores all single-lever sub-axes without clearing the three-pillar gap.

**Standing open loops (unchanged from this morning's facts pack):**

- **`shortcut-deployment-block`** (critical, in-progress) — `dor-real-webcam-false-flag-no-virtual-bg ≤ 0.30` AND `lockbox_fake_recall ≥ 0.60` at single τ holding `teams_ood_real` FPR ≤ 5%. Not met by any candidate to date.
- **`face-size-label-leak`** (high) — flip rate from face-size-targeted intervention not yet ≤ 10%.
- **`eval-production-crop-tightness-mismatch`** (high) — eval substrate has looser crop than production; unquantified delta.
- **`sharpness-metric-computed-on-full-image-not-face`** (high) — full-image Laplacian, not face-crop; downstream FPR-by-quartile reports partially confounded.
- **`frame-level-vs-clip-level-scorer-mismatch`** (medium) — clip-level recall vs frame-level AUC reconciliation under corrected contract policy.
- **`shortcut-deployment-block`** (critical, in-progress) — `dor-real-webcam-false-flag-no-virtual-bg ≤ 0.30` AND `lockbox_fake_recall ≥ 0.60` at single τ holding `teams_ood_real` FPR ≤ 5%. Not met by any candidate to date.
- **`face-size-label-leak`** (high) — flip rate from face-size-targeted intervention not yet ≤ 10%.
- **`eval-production-crop-tightness-mismatch`** (high) — eval substrate has looser crop than production; unquantified delta.
- **`sharpness-metric-computed-on-full-image-not-face`** (high) — full-image Laplacian, not face-crop; downstream FPR-by-quartile reports partially confounded.
- **`frame-level-vs-clip-level-scorer-mismatch`** (medium) — clip-level recall vs frame-level AUC reconciliation under corrected contract policy.
- **Whether xinhe-fake-1/2/3 (no-glasses) miss because of mask quality, codec, lighting, or interaction.** Worth a focused audit on her variants alone — same person, different masks, same setup. Compare frame-level features across her catchable vs uncatchable variants.
- **Whether the chronic-6 reals are addressable via training or only via deployment-side filtering.** Job 11 showed chronic-6 partition by ckpt; some ckpts handle some chronic identities cleanly. Substrate-aware τ may be the right answer (but user said no per-substrate τ at deployment).
- **Whether the ceiling is architectural** (saturate this CLIP-B16 family) **or method-class** (saturate "FT-from-CLIP" as a paradigm). E2B, P8A, E2B-scratch, L14-scratch all hit the same v2 cap. SBI is a training-distribution change, not architecture; if SBI doesn't crack it, architectural change becomes the next axis.

---

## 10. Trigger conditions for plan revision

This plan is not static. The following events MUST trigger a revision (edit sections 1-11; do NOT delete log entries):

1. **PD scorecard verdict lands.** Update §4.1 (team's branching), §8.2/8.3 (P1/P2/P3 triggers), §3.7 (ledger row).
2. **`PAIR_GAP_AUDIT` returns.** If positive (>25% missed-fakes have pair_gap ≤ 0): elevate P1 to next packet. If saturated: demote P1, elevate P2. Revise §8.2.
3. **`FROZEN_PAIR_HEAD_PROBE` returns.** If frozen pair-rank improves pair metrics: green-light P1. If no signal: demote P1.
4. **Chroma-band probe (0d) returns.** If chroma cleanest-cell bands differ from grayscale 12-13: revise P3 recipe. If chroma shows no clean cell: demote P3.
5. **TTA + deployment disposition (0a) returns.** If P8A swap on may6 substrate is operationally clean: write disposition into open loop, may6 production pain resolved without training packet.
6. **Any GPU packet completes.** Update §3 metrics, §3.7 ledger, §8.2 P1/P2/P3 priors.
7. **Cross-cutting bug discovery / data fix lands.** Add to §3.8; review whether ledger entries (§3.7) need re-running on clean codepath.

Do not edit the activity log retroactively — append only. Use the log entries as the audit trail for why §1-11 changed.

---

## 11. Cross-references

- **This plan:** `docs/relaunch_handoffs/NEXT_STEPS_PLAN_2026-05-06.md`
- **Team's ranked list:** `docs/relaunch_handoffs/ANTI_SHORTCUT_TECHNIQUES_RANKED_2026-05-06.md`
- **Bias-stripped facts pack:** `docs/relaunch_handoffs/FACTS_FOR_SECOND_OPINION_2026-05-06.md`
- **Second-opinion protocol:** `docs/relaunch_handoffs/SECOND_OPINION_PROMPT_2026-05-06.md`
- **External advisor pack:** `docs/external_advisor_pack_2026-05-06/`
- **Probe outputs:** `analysis/{xinhe_cross_camera_audit,dor_drift_mechanism,amp_vs_phase_probe,paired_feature_consistency,fourier_band_overlap,per_layer_p8a_e2b_pa}_2026-05-06/outputs/`
- **PD packet retro:** `docs/packet_retrospectives/packets/PD.md`
- **Correlation-penalty thread:** `docs/packet_retrospectives/threads/correlation_penalty_loss.md`
- **Shortcut taxonomy thread:** `docs/packet_retrospectives/threads/processing_signature_shortcut.md`
- **Bundle discipline thread:** `docs/packet_retrospectives/threads/anti_shortcut_bundle_decomposition.md`
- **Pair-loss prior refutation:** `analysis/pair_loss_effect_verification_2026-05-05/FINDINGS.md`
- **Identity browser dataset:** `analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv`
- **Open loops:** `docs/packet_retrospectives/OPEN_LOOPS.md` — `corr-penalty-deployment-grade-verdict-pending`, `deployment-vs-p8a-substrate-tradeoff-not-quantified`, `shortcut-deployment-block`
- **Code paths cited by external advisor:**
  - `data/sources/combined_paired.py::CombinedPairedIterableDataset` (paired loader)
  - `data/sources/combined_paired.py:3503` (collate function)
  - `detectors/effort_detector.py:1073` (loss path)
  - `trainer/mixins/group_dro.py` (current method-id GroupDRO)
  - `train_sweep.py` → `create_data_pipeline(...)` → trainer entrypoint
- **Key memories:**
  - `project_success_criteria.md` (three pillars)
  - `feedback_per_mode_tau_not_deployable.md` (single τ constraint)
  - `project_deployment_is_e2b_2026-05-06.md` (deployment ≡ E2B)
  - `project_pa_does_not_generalize_to_hdtf_2026-05-05.md` (PA-on-HDTF walkback)
  - `project_in_proj_svd_gradient_bug.md` (the `2feea58` fix)
  - `project_quality_enhancement_routing_2026-05-05.md` (~27% mislabeled deeplive_enhanced pre-2026-05-05)
  - `project_pair_loss_premise_refuted_2026-05-04.md` (raw/Teams premise; does NOT generalise to real/fake same-source per advisor)
  - `project_chronic_offenders_partition_per_ckpt_2026-05-04.md` (chronic-6 not uniform)
  - `project_p8a_breakthrough.md` (P8A recipe)

---

## 12. Activity log (append-only)

> **Format:** each entry is a self-contained record. Do not edit prior entries. When a job's outcome materially changes the plan above, append a new log entry stating the result AND a separate "Plan revision" entry pointing to which section(s) of §1-11 were edited.

> **Entry template:**
> ```
> ### YYYY-MM-DD HH:MM TZ — <job_type>: <job_name>
> - Owner: <agent or human>
> - Status: queued | running | complete | failed | cancelled
> - Trigger: <which §10 condition or other reason>
> - Output: <path or "n/a">
> - Result summary: <1-3 sentences>
> - Plan update needed: yes (sections X, Y) | no
> ```
>
> ```
> ### YYYY-MM-DD HH:MM TZ — Plan revision
> - Triggered by: <previous log entry>
> - Sections edited: <list>
> - Summary of change: <1-3 sentences>
> ```

---

### 2026-05-06 evening — Document created
- Owner: Claude (Opus 4.7), in-house second-opinion agent
- Status: complete
- Trigger: user request to capture three-view synthesis as a standing document before launching jobs
- Output: this file
- Result summary: documented circumstances, three-view synthesis, convergence map, novel-to-external items, convergent/divergent recommendations, Phase 0 ($0 today) + Phase 1 (conditional GPU packets) plan, three-pillar prognosis 15-25% single-packet / 40-55% multi-packet sequenced.
- Plan update needed: no (this entry IS the plan creation)

### 2026-05-06 evening — User question on FT-base vs scratch
- Owner: user
- Status: noted
- Trigger: user asked "how strongly does the plan push for FT vs scratch?"
- Result summary: §8.2 P1/P2/P3 all default to FT-from-P8A or FT-from-E2B; scratch not actively prioritised. Empirical ledger (§3.7) shows 3 arch-distinct scratch attempts have failed viso recall (E2B viso 27→7%, E3 viso 11.6%, P13 modern_v2 FPR 30.2%). HOWEVER — every prior scratch attempt was on pre-`quality_enhancement`-routing-fix data (~27% mislabeled deeplive_enhanced). "Scratch + clean data + pair-rank + multi-axis GroupDRO + SBI auxiliary" has not been tested. Plan does not actively rule out scratch; just unprioritised because no cheap pre-validation probe exists.
- Plan update needed: yes — adding a flagged "scratch-as-parallel-ablation-arm" option to be revisited after Phase 0 results.

### 2026-05-06 evening — Phase 0 CPU jobs dispatched (6 parallel sub-agents)
- Owner: Claude (Opus 4.7), parallel sub-agent dispatch on Mac (strong CPU)
- Status: **complete — all 6 first-wave agents returned**
- Trigger: §10 condition — user authorised parallel dispatch of Phase 0 CPU tasks
- Jobs dispatched:
  1. `PAIR_GAP_AUDIT_P8A_E2B_PA_2026-05-06` → `analysis/pair_gap_audit_2026-05-06/outputs/`
  2. `PAIR_COVERAGE_AUDIT_2026-05-06` → `analysis/pair_coverage_audit_2026-05-06/outputs/`
  3. `CURRENT_CHECKPOINT_COHORT_DIAGNOSIS_2026-05-06` → `analysis/checkpoint_cohort_diagnosis_2026-05-06/outputs/`
  4. `P8A_TRAINING_DATE_CHECK_2026-05-06` → `analysis/p8a_training_date_check_2026-05-06/outputs/`
  5. `CHROMA_BAND_FOURIER_2026-05-06` → `analysis/chroma_band_fourier_2026-05-06/outputs/`
  6. `DOR_DRIFT_R2_RECONCILIATION_2026-05-06` → `analysis/dor_drift_reconciliation_2026-05-06/outputs/`
- Deferred to second wave (heavier setup or hard dependencies):
  - `GROUP_ID_DESIGN_AUDIT_2026-05-06` (depends on PAIR_COVERAGE outputs)
  - `FROZEN_PAIR_HEAD_PROBE_2026-05-06` (needs feature extraction)
  - `SBI_SMOKE_2026-05-06` (needs face-landmark + blending pipeline setup)
  - `TTA_INFERENCE_2026-05-06` (needs model load; better after PD verdict)
- Result summary: pending — individual entries per job will be appended on completion.
- Plan update needed: TBD per job

### 2026-05-06 evening — Result: PAIR_COVERAGE_AUDIT_2026-05-06
- Owner: Claude sub-agent (general-purpose)
- Status: complete
- Trigger: Phase 0b-2 (in plan §8.1) — pair-rank applicability breadth
- Output: `analysis/pair_coverage_audit_2026-05-06/{run_probe.py, pairing_semantics_notes.md, outputs/{coverage_by_method.csv, coverage_by_enhancer.csv, coverage_by_transport.csv, coverage_by_identity.csv, coverage_summary.json, FINDINGS.md}}` (FINDINGS by parent — sub-agent wrote `pairing_semantics_notes.md` directly)
- Result summary: **~91-97% of training is paired** (base samples 91.5%, frames per epoch 96.6%, family-weight share 93.3%). All 6 paired lanes share the same pair key `(sample_id, frame_idx)` with opposite `label` — **same-source frame pairing is tight**, not the loose cross-product Agent 1 measured. Lanes: `df40` (4,698), `deeplive` (4,067), `visomaster_v1_base` (342), `visomaster_enhanced` (1,484), `visomaster_teams_enhanced` (997 — only 5.4% with true teams_v2 companion), `deeplive_teams` (~1,300, family weight 7.0 — heaviest, closest-to-production). Only `external_vcd_real` (1,200) is unpaired; needs `is_unpaired_real=True` skip rule.
- **Critical recommendation: apply pair-rank UNIFORMLY across all 6 paired lanes** — do NOT restrict to "DF40+VisoMaster+DeepLive only, drop Teams passthrough" because that would punt on the highest-weight production-relevant lane (`deeplive_teams`).
- **Reconciliation with PAIR_GAP_AUDIT (Agent 1):** both correct, different metrics. Agent 1's RED on cross-product pairs is loose-pairing-specific; the actual training loader emits TIGHT same-source pairs. The cross-product audit over-counts comparisons; the training loss only fires on tight pairs. **`SAME_SOURCE_PAIR_GAP_AUDIT_2026-05-06` (Phase 0g) is the load-bearing go/no-go**, not the cross-product audit.
- **Confirms external advisor's framing:** the 2026-05-04 `pair_loss_premise_refuted` memory targets raw-vs-teams symmetric consistency, NOT real-vs-fake same-source ranking. The two are orthogonal premises.
- Pair-rank loss design implication: collate at `combined_paired.py:3503` should add `pair_id = (sample_id, frame_idx)` as a first-class field so the loss finds pairs in O(N).
- Plan update needed: yes
  - §8.2 P1 recipe — UNDO the previous "restrict to raw lanes" revision from Agent 1's entry. Apply uniformly across all 6 paired lanes; only `external_vcd_real` needs `is_unpaired_real=True` skip.
  - §8.1 Phase 0g — elevate to "load-bearing go/no-go" status (was "follow-up"). Until 0g runs, P1's verdict is open.
  - §6.2 (code-level pair-aware finding) — update with `pair_id` field design recommendation.

### 2026-05-06 evening — Result: PAIR_GAP_AUDIT_P8A_E2B_PA_2026-05-06
- Owner: Claude sub-agent (general-purpose)
- Status: complete
- Trigger: Phase 0b-1 (in plan §8.1) — load-bearing go/no-go for `PE_PAIR_RANK_DRO`
- Output: `analysis/pair_gap_audit_2026-05-06/outputs/{run_probe.py, pair_gaps.csv (37,327 rows × 31 cols), summary.json, missed_fake_audit.csv (177 rows), fp_real_audit.csv (120 rows), FINDINGS.md (160 lines)}`
- Pairs found: **37,327** across 11 canonical subjects with both labels (out of 65 canonical subjects total). 54 subjects were label-monochromatic. **Pairing methodology: cross-product within `canonical_subject` capped at 4,000 pairs/subject** — frame-level same-source pairing is NOT present in this eval manifest (teams fake/real are different sessions; visomaster_v2_dor has no co-bucketed real). Lanes covered: `dor_local` (568×2871, capped 4k), `xiang_xiang2_feng`, 6 teams identities, `extra_xiang`, `extra_xinghe`.
- **Headline verdict: RED on eval substrate.** Pair-weighted P(pair_gap ≤ 0 | missed_fake):
  - P8A 9.88% → **RED** (decision rule threshold = 25%)
  - E2B 5.63% → **RED**
  - PA_3800 17.79% → **AMBER**
- **Critical caveat — verdict is partial:** The audit ran on cross-product eval pairs, NOT on the tighter same-source training pairs that `PE_PAIR_RANK_DRO`'s loss would actually fire on. The training loader (`combined_paired.py`, `df40_paired.py`) emits **~5,379 frame-level same-source pairs on DF40 + DeepLive + VisoMaster** lanes — these were not measured. A separate `SAME_SOURCE_PAIR_GAP_AUDIT` extracting cached scores at training-time pair indices is required for a definitive verdict.
- **Per-subject heterogeneity is severe — the lever is GREEN in some lanes, dead in others:**
  - `dor_local` alone: P8A 27.6% / PA 42.5% — **GREEN if isolated**.
  - All six teams identities: 0–8% — dead.
  - viso_v2_inswapper raw: 30–49% inversions on P8A — **GREEN**.
  - teams_passthrough: 2.8% — dead.
- **By transport (P8A):** raw=21.8% vs teams=2.8%. Pair-rank inversions concentrate exactly where Pearson-r + HDTF probes already showed structural fragility — i.e., the lever's potential signal is on raw viso/deeplive lanes.
- Plan update needed: yes
  - §8.2 P1 — **DEMOTE from primary to AMBER-conditional.** Pair-rank verdict on cross-product eval pairs is RED, but same-source training-pair audit might flip it on raw-viso/deeplive subset. Restrict potential P1 scope to {DF40, DeepLive, VisoMaster raw} lanes only — explicitly NOT teams_passthrough.
  - §8.2 P2 — **ELEVATE PE_SBI to primary candidate.** SBI doesn't depend on existing pair structure in training data; it's the structurally novel intervention; cohort-diagnosis showed P8A has direct leverage on the viso/deeplive cohorts SBI also targets.
  - §8.3 sequencing — invert: SBI BEFORE pair-rank, with a contingency that pair-rank may follow if same-source training-pair audit comes back GREEN on the raw-lane subset.
  - §8.1 — add Phase 0g: `SAME_SOURCE_PAIR_GAP_AUDIT_2026-05-06` at training-time pair indices on the ~5,379 paired DF40/DeepLive/VisoMaster frames.
  - §8.4 prognosis — single-packet three-pillar probability essentially unchanged (15-25%); the ranking shifts but the absolute prognosis doesn't.
  - §9 — add new open question: does same-source pair-rank loss on raw-viso/deeplive-only lanes (excluding teams_passthrough) recover signal that the cross-product audit missed?

### 2026-05-06 evening — Result: DOR_DRIFT_R2_RECONCILIATION_2026-05-06
- Owner: Claude sub-agent (general-purpose)
- Status: complete
- Trigger: Phase 0e (in plan §8.1) — reconcile 14% / 20% / 90% R² discrepancy
- Output: `analysis/dor_drift_reconciliation_2026-05-06/outputs/{run_probe.py, metric_reconciliation.json, reconciliation_recompute.json, verification_log.md, FINDINGS.md}` (FINDINGS by parent — sub-agent harness blocked .md writes)
- Result summary: **No contradiction in existing docs.** All three numbers reproduced exactly from `per_frame_features.csv`; they answer three different questions on the same data:
  - **R²=0.142** = within-frame score variance explained by IQ across full 819-frame dor real population. NOT cross-session drift.
  - **0.064/0.316 ≈ 20%** = all-sessions-fit's projection onto endpoint difference (under-estimate; coefficients pulled toward within-session signal).
  - **0.285/0.316 = 90%, R²_union 0.42-0.62** = endpoint-pair-only ridge on 250 frames in (dor_evening + teams_real_dor_dev). **This is the right number for "X% of drift explained by named axes."**
- Right statement to use: *"~90% of the cross-session drift between dor_evening and teams_real_dor_dev (Δmean P8A=0.316) is recoverable as a linear projection of named pixel-domain IQ axes (R²_union = 0.45 P8A, 0.42-0.62 across all ckpts). Top drivers: min_dim (|r|=0.57), edge_mag (|r|=0.45), color_b_dev (|r|=0.43)."*
- **Important new finding for PD-class successors:** PD currently targets sharpness_lap + luma_mean + face_area_fraction; the dor-drift DOMINANT axes are `min_dim` (resolution) and `color_b_dev` (B-channel cast) — both UN-TARGETED. These are the high-impact axes most likely to absorb the shifted gradient when PD's penalty bites; Agent 5 (chroma-band Fourier) independently confirmed lab_b as a load-bearing chroma shortcut channel. **Recommendation:** PD-class successors (HSIC, AugMix-with-corr-pen, etc.) should include `min_dim` and `color_b_dev` in the penalty axes.
- Caveat: 90% is "drift lives in a low-dim named-IQ subspace," NOT "named IQ is the causal driver" — necessary but not sufficient for PD-class success.
- Plan update needed: yes
  - §3.6 — annotate the three-metric structure of dor-drift attribution; cite the union-fit 90% as the headline; flag PD's under-targeted axes (`min_dim`, `color_b_dev`).
  - §9 — close 0e entry (no error, was a three-question situation).
  - §4.2 (in-house Round-2 view) — note the metric-conflation flag was over-conservative.

### 2026-05-06 evening — Result: CHECKPOINT_COHORT_DIAGNOSIS_2026-05-06
- Owner: Claude sub-agent (general-purpose)
- Status: complete
- Trigger: Phase 0b (in plan §8.1) — FT-base recommendation P8A vs E2B for `PE_PAIR_RANK_DRO`
- Output: `analysis/checkpoint_cohort_diagnosis_2026-05-06/outputs/{run_probe.py, cohort_by_*.csv (10 axes), summary.json, ft_base_recommendation.json, paired_frames_with_outcomes.csv (10,289 rows), FINDINGS.md}` (FINDINGS written by parent agent — sub-agent harness blocked .md writes)
- Result summary: **VERDICT — FT base for `PE_PAIR_RANK_DRO` = P8A. HIGH confidence.**
  - n=10,289 paired frames across 9 person-clusters (70% of manifest).
  - Aggregate at τ=0.5 — `real_FP_fake_missed` (the cell pair-rank specifically targets): P8A **0.0%** vs E2B 3.2% vs PA 26.3%.
  - P8A wins 7/9 cohort axes weighted by frames.
  - Production-decisive cohorts favour P8A by huge margins: `live_fakes_teams_prod` (P8A 0.247 vs E2B 0.928 failure rate), `xinhe_may6_falseflag` (0.000 vs 0.576), `xinhe` cluster (0.280 vs 0.928), `xiang` cluster (0.064 vs 0.474), `transport=teams_live` (0.234 vs 0.910).
  - E2B's narrow wins are all real-FPR-dominated (no fake-recall cohort unique to E2B): `pc_generator` chronic identity, `test_cam__s73` outlier, `face_size=far`, pure-real `teams_real_all_dev`. **Multi-axis GroupDRO real-side key `source × transport × quality_band × chronic_flag` directly reclaims these.**
  - PA_3800 ruled out (26% double-fail).
  - PE headroom on P8A lives in: `gpen` enhancer (46.7% missed), `inswapper` method (33.1%), `dor_evening_morning` cluster (17.0%), `live_fakes_teams_prod` (24.7%), `visomaster_v2_dor` (26.2%) — paired-rich substrates with structural pair-rank leverage.
- Plan update needed: yes
  - §8.2 P1 — FT base = P8A explicit (not "P8A or E2B based on cohort diagnosis"). Multi-axis GroupDRO real-side key MUST include `chronic_flag` and `source` (to reclaim E2B's narrow wins). Grad-audit at step 100/500/1000 (per 0c verdict on P8A pre-fix).
  - §6.5 — advisor's P8A FT-base argument now FULLY validated (pre-fix timing from 0c + cohort coverage from this job).
  - §9 — close "FT base for next packet" ambiguity.

### 2026-05-06 evening — Result: CHROMA_BAND_FOURIER_2026-05-06
- Owner: Claude sub-agent (general-purpose)
- Status: complete
- Trigger: Phase 0d (in plan §8.1) — extend Probe 6 to per-channel; gates `PE_FOURIER_BAND` recipe (§8.2 P3)
- Output: `analysis/chroma_band_fourier_2026-05-06/outputs/{run_probe.py, per_band_aucs_{R,G,B,L,lab_a,lab_b}.csv, cleanest_cells_table.csv, summary.json, FINDINGS.md}` (FINDINGS written by parent agent — sub-agent harness blocked .md writes)
- Result summary: **VERDICT — CHROMA_DOMINATES_DIFFERENT_BANDS.** R/G/B/L (luminance-like) channels reproduce grayscale Probe 6 almost exactly (cleanest at bands 9-13). Chroma channels lab_a / lab_b do NOT — shortcut peaks at mid-frequency bands 6-8 (lab_a additionally at band 1 and 9-10). Both chroma channels have ZERO signal-carrying bands (max signal AUC lab_a=0.58, lab_b=0.70). lab_a uniquely has 6 cleanest cells across the low-mid frequency axis. **Grayscale-only band masks miss the chroma shortcut at bands 1, 6, 7.** Probe 1's named axes (sat_std d=−5.95, b_std d=−2.61) align directly with lab_a/lab_b bands as expected. Chroma-targeted randomization is the right mechanism — hits the correct axis with negligible signal loss.
- Recipe update for `PE_FOURIER_BAND`:
  - R/G/B/L: randomize {8, 9, 10, 12, 13}, preserve {5, 6}.
  - lab_a: randomize {1, 6, 7, 8, 9, 10}, no preserve required.
  - lab_b: randomize {6, 7, 8}, no preserve required.
  - Implementation: BGR → LAB, per-channel FFT-amp same-label randomization, LAB → BGR. Phase preserved universally.
- Plan update needed: yes
  - §3.6 — annotate Probe 6 caveat is now resolved; add chroma findings.
  - §8.2 P3 — replace grayscale-implicit recipe with per-channel masks above. Note explicitly that grayscale-only would leave chroma shortcut at bands 1, 6, 7 intact and shrink the ratchet vs budget.
  - §9 — close "chroma-band Fourier structure" open question.

### 2026-05-06 evening — Result: P8A_TRAINING_DATE_CHECK_2026-05-06
- Owner: Claude sub-agent (general-purpose)
- Status: complete
- Trigger: Phase 0c (in plan §8.1) — verify FT-from-P8A is on post-fix codepath
- Output: `analysis/p8a_training_date_check_2026-05-06/outputs/{verdict.json, evidence_log.md, FINDINGS.md}` (FINDINGS written by parent agent — sub-agent harness blocked .md writes)
- Result summary: **VERDICT — PRE-FIX, HIGH confidence.** P8A_REFERENCE_STEP5000 (run `9lmvb5b4`, "smooth-haze-250") trained 2026-04-24 21:41 UTC → 23:48 UTC (Vertex job `1205078281779412992`, us-east1). Fix commit `2feea58` ("Fix silent zero-gradient bug in apply_svd_to_in_proj path") landed 2026-04-26 19:13 UTC, **43h 25m AFTER P8A finished**. Three independent sources confirm pre-fix classification: P8A retro (`packets/P8A.md:29,108`), master-plan log (`april-26-training-master-plan-v2.LOG.md:214`), in-proj-SVD bug memory. Caveat: post-fix C-ablation slot on P10_SYM-on-P8A recipe tied within noise — in_proj-SVD lever doesn't materially move canonical P8A config, but that's under one specific FT recipe, not PE_PAIR_RANK_DRO's loss.
- Plan update needed: yes
  - §6.5 — advisor's "FT-from-P8A on post-fix codepath has unmeasured headroom" argument is now CONFIRMED (was previously "unverified, check in 0c"); update wording.
  - §8.2 P1 — when documenting FT base options, add explicit note: "if FT-from-P8A chosen, include a cheap grad-audit at step 100/500/1000 to confirm in-proj-SVD lever is live under the new pair-rank+DRO loss; this lever was zero-gradient during P8A's original training."
  - §9 — close the "P8A_REFERENCE_STEP5000 training date relative to `2feea58`" open question with the pre-fix verdict.

### 2026-05-06 evening — Plan revision (consolidated, post-Phase-0)
- Triggered by: all 6 dispatched Phase 0 sub-agents complete
- Sections edited: §8.1 (status markers + new 0g), §8.2 P1 (full recipe revision), §8.2 P2 (elevated to co-primary, FT base = P8A explicit), §8.2 P3 (per-channel chroma recipe), §8.3 (default sequencing branch tree replaced), §9 (closed 4 questions, opened 3 new)
- Summary of changes:
  1. **FT base = P8A** with grad-audit hook (Agent 3 cohort + Agent 4 pre-fix verdicts, both HIGH confidence). E2B's narrow real-FPR wins reclaimable via multi-axis GroupDRO real-side key with `chronic_flag` and `source`.
  2. **P1 PE_PAIR_RANK_DRO is AMBER-conditional, not green.** Cross-product PAIR_GAP_AUDIT was RED on aggregate but verdict is partial — same-source training-pair audit (Phase 0g) is the load-bearing go/no-go. Recipe applies pair-rank uniformly across all 6 paired lanes (Agent 2 finding); `external_vcd_real` gets `is_unpaired_real=True` skip; collate adds `pair_id = (sample_id, frame_idx)` first-class field.
  3. **P2 PE_SBI elevated to co-primary candidate** — no-regret structurally novel intervention; doesn't depend on pair structure existing in training; FT base = P8A.
  4. **P3 PE_FOURIER_BAND recipe revised to per-channel chroma** (Agent 5 finding): R/G/B/L randomise {8,9,10,12,13} preserve {5,6}; lab_a randomise {1,6,7,8,9,10} no preserve; lab_b randomise {6,7,8} no preserve. BGR↔LAB conversion in transform.
  5. **PD-class successors should add `min_dim` and `color_b_dev`** to penalty axes (Agent 6 finding; corroborated by Agent 5's chroma-band lab_b finding). Currently un-targeted by PD; high probability of absorbing shifted gradient.
  6. **F2 close criterion across all P1/P2/P3** updated to include `min_dim` and `color_b_dev` axes explicitly.
  7. **Sequencing inverted contingent on Phase 0g**: SBI before pair-rank if 0g returns RED at training-pair indices; pair-rank first if 0g greens.
  8. **Scratch-as-parallel-ablation-arm option** flagged in §8.3 and §9 — not primary, but kept on the bench for explicit user decision.
- Three-pillar prognosis unchanged: 15-25% single-packet, 40-55% multi-packet sequenced.
- Outstanding actions for the user / next agent: (a) authorise / dispatch Phase 0g `SAME_SOURCE_PAIR_GAP_AUDIT_2026-05-06`; (b) optionally dispatch deferred 0a TTA + 0b SBI smoke + 0b GROUP_ID_DESIGN; (c) wait for PD scorecard verdict; (d) decide whether to add a parallel scratch arm.

### 2026-05-06 evening — Phase 0 second-wave dispatched (4 parallel sub-agents)
- Owner: Claude (Opus 4.7), parallel sub-agent dispatch
- Status: **complete — all 4 second-wave agents returned**
- Trigger: §10 condition — user authorised continuing CPU jobs as fit; first-wave verdicts identify load-bearing follow-ups
- Jobs dispatched:
  1. `SAME_SOURCE_PAIR_GAP_AUDIT_2026-05-06` (0g) → `analysis/same_source_pair_gap_audit_2026-05-06/outputs/` — **load-bearing go/no-go for P1**
  2. `GROUP_ID_DESIGN_AUDIT_2026-05-06` → `analysis/group_id_design_audit_2026-05-06/outputs/` — multi-axis DRO key design
  3. `FROZEN_PAIR_HEAD_PROBE_2026-05-06` → `analysis/frozen_pair_head_probe_2026-05-06/outputs/` — corroborates 0g; head-vs-encoder localisation
  4. `SBI_SMOKE_2026-05-06` → `analysis/sbi_smoke_2026-05-06/outputs/` — P2 foundation validation
- Skipped this batch: TTA (orthogonal production-side concern; will dispatch separately if user prioritises may6 mitigation).
- Result summary: pending — individual entries per job will be appended on completion.

### 2026-05-06 evening — Result: SAME_SOURCE_PAIR_GAP_AUDIT_2026-05-06 (Phase 0g)
- Owner: Claude sub-agent (general-purpose)
- Status: complete (verdict: MIXED, leaning LANE_RESTRICTED_LAUNCH)
- Trigger: second-wave Phase 0 dispatch — load-bearing go/no-go for P1 (replaces cross-product audit's role)
- Output: `analysis/same_source_pair_gap_audit_2026-05-06/{run_probe.py, outputs/{same_source_pairs.csv (4,000 rows), summary.json, verdict.json, coverage_blueprint.md, FINDINGS.md}}` (FINDINGS by parent — sub-agent harness blocked .md writes)
- Result summary: **VERDICT — MIXED, leaning LANE_RESTRICTED_LAUNCH on FT-from-P8A.**
  - Step-1 cache-first audit: **0 tight `(sample_id, frame_idx)` pairs in any local cache.** Training-only buckets (df40, deeplive, visomaster_v1, viso_teams_enh, deeplive_teams) are not in any local score cache.
  - Cross-product proxy mapped to paired-lane semantics covers **2 of 6 paired training lanes** with verdicts on P8A:
    - viso_enhanced: 28.1% (n_pairs=2869) — **GREEN**.
    - deeplive: 23.5% (n_pairs=847) — AMBER (small n_missed).
    - df40, viso_v1, viso_teams_enh, deeplive_teams: INSUFFICIENT_DATA.
  - E2B RED on both measurable lanes (6.9% / 8.6%) — independently confirms FT-base = P8A.
  - PA_3800 GREEN (43.5%) — informational only (HDTF non-generalisation).
- **Critical reconciliation with cross-product Agent 1 RED:** the original aggregate 9.88% on P8A was driven almost entirely by `teams_fake_all_dev` (2.8% inversion, n_missed=2,286). `teams_fake_all_dev` is **eval-only substrate, NOT one of the 6 paired training lanes** — pair-rank loss never fires there. Re-aggregating by training-lane flips P8A from RED → GREEN/AMBER on lanes the loss can actually fire on. **The cross-product RED was misleading; the on-training-substrate verdict is favorable.**
- §8.2 strict gate ("≥2 paired lanes GREEN") **NOT met from cache** (have 1 GREEN + 1 AMBER + 4 INSUFFICIENT) **but ZERO RED among measured**.
- **User decision point (per `feedback_decision_points.md`):**
  - **Option A:** authorize forward-pass tight-pair audit on df40 + deeplive (us-east1, ~$3, ~40 min) → 4/6 lane coverage, definitive verdict.
  - **Option B:** lane-restricted P1 launch on FT-from-P8A with uniform pair-rank application + P2 (SBI) in parallel. df40 (largest unmeasured lane) carries unknown-direction risk; pair-rank loss=0 on RED batches is harmless per coverage agent.
  - Agent recommendation: B if time-constrained, A then B if not.
- Plan update needed: yes
  - §8.2 P1 — UPGRADE from "DEFERRED" to "LANE_RESTRICTED_LAUNCH VIABLE" (Option B) OR "AWAITING OPTION A FORWARD-PASS." Revise prognosis accordingly.
  - §8.1 — add Phase 0j: forward-pass tight-pair audit on df40 + deeplive (Option A; ~$3 / 40 min).
  - §8.4 prognosis — slight upward revision possible: single-packet P1 prognosis with lane-restricted launch is not lower than the deferred-status read.
  - §6.1 (advisor's pair-rank lever, novel-to-external-only) — note that the cross-product RED was misleading; on-training-substrate aggregate is favorable; advisor's argument substantially holds.

### 2026-05-06 evening — Result: SBI_SMOKE_2026-05-06
- Owner: Claude sub-agent (general-purpose)
- Status: complete
- Trigger: second-wave Phase 0 dispatch — validates P2 (PE_SBI) foundation
- Output: `analysis/sbi_smoke_2026-05-06/{run_probe.py, outputs/{summary.json, verdict.json, iq_comparison.csv, iq_per_frame.csv, sbi_records.csv, score_distributions.csv, pseudofakes/_visual_panel_real_vs_sbi.png, pseudofakes/sbi_*.png (60 images), run.log, FINDINGS.md}}` (FINDINGS by parent — sub-agent harness blocked .md writes)
- Result summary: **VERDICT — GREEN on all three criteria.** P2 launch-ready foundation.
  - 200 pseudofakes, **97.5% MediaPipe landmark success** (5 fallbacks to elliptical mask).
  - IQ similarity: median |Cohen's d| **real-vs-SBI = 0.047** (virtually indistinguishable on luma/sat/RGB). SBI-vs-fake d=0.68 > real-vs-fake d=0.60 — confirms SBI carries method-agnostic blending signal, not method-specific fingerprints (the whole premise).
  - Score headroom on P8A: real_median 0.0058 → fake_median 0.884 (0.88 gap). E2B and PA show >0.45 gap each.
- **Watch-item (mild but real):** face Laplacian variance drops 30% (real 483 → SBI 339, d=0.75). Same direction as known-fake softening. Risk: model latches on "slightly less sharp than canonical real" → passes F1 (lockbox recall) but fails F4 (HDTF cross-substrate) — identical failure mode to PA's substrate-bound walkback. **F4 is the differential gate for P2.**
- **Recipe parameter recommendations for P2:**
  - `feather_px` uniform {5,7,9,11,13,15} — KEEP.
  - `geom_strength` 1.0 → **SWEEP 0.5-1.5**.
  - `noise_std` 0.01 → **SWEEP {0.005, 0.01, 0.02}**.
  - same-image self-blend → KEEP for v1.
  - Optional: matching small Laplacian sharpen post-blend to overlap real more tightly.
  - Log `lap_var_face` on SBI samples during training; monitor d-vs-real online (target d<0.30).
- **What's deferred:** direct SBI score collection on P8A/E2B/PA (no gs:// + no local ckpts). Blueprint: ~10 min CPU + ~90 MB ckpt downloads; re-use `analysis/xinhe_cross_camera_audit_2026-05-06/run_local_inference.py`. Upgrades GREEN-by-headroom → GREEN-by-direct-measurement.
- Plan update needed: yes
  - §8.2 P2 recipe — add `geom_strength` and `noise_std` sweeps; emphasise **F4 as differential gate**; add `lap_var_face` monitoring; document the sharpness watch-item.
  - §8.1 — add Phase 0i: deferred SBI score collection on P8A/E2B/PA (~10 min, near-$0).

### 2026-05-06 evening — Result: FROZEN_PAIR_HEAD_PROBE_2026-05-06
- Owner: Claude sub-agent (general-purpose)
- Status: complete (verdict: INSUFFICIENT_COVERAGE; produces blueprint for follow-up)
- Trigger: second-wave Phase 0 dispatch — corroborates 0g; localises pair-rank signal to head-side vs encoder-side
- Output: `analysis/frozen_pair_head_probe_2026-05-06/outputs/{run_probe.py, coverage_report.md, summary.json, verdict.json, FINDINGS.md}` (FINDINGS by parent — sub-agent harness blocked .md writes)
- Result summary: **VERDICT — INSUFFICIENT_COVERAGE.** No on-disk feature cache contains same-source opposite-label pairs.
  - `clip_vs_p8a_viso_2026-05-03` cache: 131/1812 paired reals overlap, **0/3499 paired fakes overlap** (its 550 fakes are visomaster_enhanced without co-bucketed real partners).
  - 31 `_features_cache_2026-04-30/*.npz` files: positional `valid_idx` only, no `frame_path`, unmappable without missing `sampling_manifest.json`.
  - Net: 0 same-source paired features available.
- **Implication for P1:** **DEFERRED, not AMBER-conditional.** GPU-cost asymmetry — frozen-feature extraction is ~$5-10 / 1-2 A100-hours, encoder-FT is GPU-weeks. Run the extraction first; the probe's outcome strictly improves P1's prior either way (head-side signal → demote P1 to head-only retrain; no signal → P1 justified iff 0g GREEN, else dead).
- **Blueprint:**
  - Reuse `analysis/feature_space_2026-04-23/extract_features.py` as template.
  - Path-list from `analysis/pair_gap_audit_2026-05-06/outputs/pair_gaps.csv` (5,311 unique frame paths).
  - Output: `analysis/frozen_pair_features_2026-05-07/{p8a, e2b, clip_b16_raw}_paired_features.npz` (~32 MB per ckpt).
  - Then re-run `run_probe.py` (heads-training path activates).
- **Promotion gate for P1 after extraction:** Head B (CE+pair-rank) ≥3pp lift over Head A (CE-only) → defer P1 in favour of head-only retrain (cheaper); no lift → P1 justified iff 0g GREEN.
- **Adjacent deployable lever (parallel, no extraction):** per-substrate τ-calibration gives **21pp lockbox recall lift on P8A today** (Job 7 result). Operating-point lever, not weights. Worth capturing in parallel — non-interactive with P1/P2/P3.
- Plan update needed: yes
  - §8.1 — add Phase 0h: paired-feature extraction at training-pair indices (~$5-10, 1-2 A100-hours). Gates the head-vs-encoder localisation.
  - §8.2 P1 — DEFER (not AMBER) until either 0h+probe runs OR user accepts encoder-FT risk without head-side localisation. Add cost note: ~$5-10 to disambiguate vs ~GPU-weeks to commit blind.
  - §9 — add open question: is the 21pp per-substrate τ-calibration lever worth deploying TODAY as a parallel patch (alongside training-side fix)?

### 2026-05-06 evening — Result: GROUP_ID_DESIGN_AUDIT_2026-05-06
- Owner: Claude sub-agent (general-purpose)
- Status: complete
- Trigger: second-wave Phase 0 dispatch — designs the multi-axis GroupDRO `group_id` for P1/P2
- Output: `analysis/group_id_design_audit_2026-05-06/outputs/{run_probe.py, candidate_groups.csv, pairwise_evaluation.csv, chronic_flag_definition.json, quality_band_thresholds.json, recommendation.json, group_id_python_snippet.py, FINDINGS.md}` (FINDINGS by parent — sub-agent harness blocked .md writes)
- Result summary: **VERDICT — recommended pair (F-B, R-D).**
  - **Fake-side F-B:** `label=fake | method_family | enhancer_family`. (Advisor's full proposal `method × enhancer × transport × quality_band` was DEMOTED — quality_band degenerates because 67% of fake rows have `quality=unknown`; transport adds only 1pp spread for too-fragmented groups.)
  - **Real-side R-D:** `label=real | source | transport | quality_band | chronic_flag`. (Advisor's full proposal — `chronic_flag` is the load-bearing axis, lifts real-side spread from 0.11 to 0.61.)
  - Stats: 27 groups, min=68, median=381, max-share 0.184, spread 0.915. Passes all three DRO-stability thresholds.
- **Chronic-6 list (explicit):** `bla_bla_chow`, `bla_bla_chow__s2`, `PC_Generator__s22`, `PC_Generator__s45`, `roy_d`, `Q__s6`. 1,297 real rows match. Zero fake rows match (chronic_flag is correctly omitted from F-B).
- **External lane handling:** `external_vcd_real` (1,200 unpaired reals) gets `is_unpaired_real=True` → CE only, but still contributes to GroupDRO via its own real group.
- **Implementation:** `trainer/mixins/group_dro.py` needs `method_mapping` → `group_id_mapping` (built at config-load). Tiny EMA buffer (27 groups). Default `ema_alpha=0.1`, `beta=3.0`, 100-step warmup. Ready-to-paste python snippet at `group_id_python_snippet.py`.
- **Train/eval stability caveat:** `is_lockbox` proxy is structurally weak (cosine sim ~0.10 across all schemes — substrate-disjointness fact, not grouping flaw). R-D's `eval_train_recall=0.33` exercises 5 of 15 real-side groups including the chronic-flag groups.
- Plan update needed: yes
  - §8.2 P1 GroupDRO recipe — replace generic spec with explicit (F-B, R-D) keys and reference `group_id_python_snippet.py` as the implementation source.
  - §6.4 (multi-axis GroupDRO grouping) — note the advisor's transport+quality_band on fake side was empirically demoted; quality unknown on 67% of fakes is the binding constraint.
  - §8.2 P2 (PE_SBI recipe) — same (F-B, R-D) GroupDRO key applies.

### 2026-05-06 evening — Plan revision (consolidated, post-second-wave)
- Triggered by: all 4 second-wave Phase 0 sub-agents complete (PAIR_GAP_AUDIT_TIGHT, GROUP_ID_DESIGN, FROZEN_PAIR_HEAD, SBI_SMOKE)
- Sections edited: §8.1 (added 0h/0i/0j; updated 0g status), §8.2 P1 (LANE_RESTRICTED_LAUNCH viable; explicit GroupDRO (F-B, R-D) keys + chronic-6 list), §8.2 P2 (sweep params + F4 differential gate), §6.4 (advisor proposal demoted on fake side; data-driven F-B chosen)
- Summary of key shifts:
  1. **P1 status moved from DEFERRED → LANE_RESTRICTED_LAUNCH VIABLE** on FT-from-P8A. Cross-product RED was misleading (driven by eval-only `teams_fake_all_dev`); on training-substrate the verdicts are GREEN (viso_enhanced 28.1%) + AMBER (deeplive 23.5%) + 4 INSUFFICIENT_DATA + ZERO RED.
  2. **GroupDRO recipe is now empirically validated: (F-B, R-D)** — fake-side `method_family × enhancer_family`, real-side `source × transport × quality_band × chronic_flag`. Advisor's full fake-side proposal (transport+quality_band) was demoted because 67% of fakes have `quality=unknown`. Real-side full proposal kept; chronic_flag is the load-bearing axis. 27 groups, all DRO-stability thresholds passed.
  3. **P2 SBI is GREEN on smoke** with one watch-item (face Laplacian variance drops 30%; same direction as known-fake softening; risks F4 HDTF cross-substrate failure mode = PA-style walkback). Mitigations: shrink `geom_strength` 1.0 → 0.5, lower `noise_std`, optional post-blend Laplacian sharpen, log `lap_var_face` online.
  4. **Two cheap follow-ups newly required:** 0h (~$5-10 paired-feature extraction → unblocks head-vs-encoder localisation) + 0j (~$3 forward-pass tight-pair audit → 4/6 lane coverage); can combine into a single ~$8-13 extraction step. 0i (~$0 SBI score collection) follows when network egress authorised.
  5. **Two paths to launch:** Path A (do 0h+0j first, ~$8-13/~1hr, then commit P1 with full info) or Path B (lane-restricted P1 launch + P2 in parallel, accept df40 unknown-direction risk).
- Three-pillar prognosis: single-packet probability mostly unchanged 15-25%; multi-packet sequenced 40-55%. Slight upward shift possible if 0h verdict gives head-side signal (P1 becomes head-only retrain, much cheaper).
- Outstanding actions for the user: (a) **decide Path A vs Path B for P1**; (b) optionally dispatch 0a TTA + 21pp τ-calibration deployable lever; (c) wait for PD scorecard verdict; (d) parallel scratch-arm decision still open.

### 2026-05-06 evening — Phase 0 third-wave dispatched (2 parallel sub-agents)
- Owner: Claude (Opus 4.7), parallel sub-agent dispatch
- Status: **complete — both third-wave agents returned**
- Trigger: auto-mode authorisation; all Phase 0 audit jobs done; remaining CPU-only value-adds
- Jobs dispatched:
  1. `TTA_POC_MAY6_2026-05-06` (Phase 0a tight-scope) → `analysis/tta_poc_may6_2026-05-06/outputs/` — 4-view TTA on 92 may6 frames + 60 may5 control + 50 dor_morning + 50 dor_evening; P8A only. Tests whether TTA damps the may6 drift before scaling.
  2. `SBI_SCORE_COLLECTION_2026-05-07` (Phase 0i) → `analysis/sbi_score_collection_2026-05-07/outputs/` — local CPU inference of P8A/E2B/PA_3800 on the 200 SBI pseudofakes. Upgrades SBI_SMOKE GREEN-by-headroom → GREEN-by-direct-measurement.
- Both authorised to download ckpts (~90 MB total) if not locally cached.
- Result summary: pending — individual entries per job will be appended on completion.

### 2026-05-06 evening — Result: SBI_SCORE_COLLECTION_2026-05-07 (Phase 0i)
- Owner: Claude sub-agent (general-purpose)
- Status: complete
- Trigger: third-wave Phase 0 dispatch — upgrades SBI_SMOKE GREEN-by-headroom → GREEN-by-direct-measurement
- Output: `analysis/sbi_score_collection_2026-05-07/{run_probe.py, outputs/{sbi_scores.csv (200 rows), score_distributions.csv, score_d_table.csv, pair_correlation.csv, summary.json, verdict.json, run.log, FINDINGS.md}}` (FINDINGS by parent — sub-agent harness blocked .md writes)
- Result summary: **VERDICT — GREEN-by-direct-measurement.** SBI_SMOKE's GREEN-by-headroom verdict held.
  - **All 3 ckpts strictly bimodal** (Sarle BC > 0.555). P8A sharpest: 10.5% > 0.9, 62.5% < 0.1, **27% in (0.1, 0.9) training sweet spot**.
  - **98% / 98.5% / 98.5% of pairs have SBI > real-source** (P8A / E2B / PA) — directional shift toward fake confirmed.
  - Score-side Cohen's d on P8A: real-vs-SBI **0.81**, SBI-vs-fake **1.36**. SBI is closer to real but firmly shifted to fake.
  - **Score-side d (0.81) is ~17× larger than IQ-side d (0.047 from SBI_SMOKE)** — encoder picks up manipulation cues that pixel-IQ axes miss. Exactly the SBI thesis: blending-boundary signal exists in model representation, not in raw pixel statistics.
  - Per-pair Pearson r(SBI, real-source): PA 0.562, P8A 0.388, E2B 0.380 — partly content-anchored, ~70% residual is the SBI-specific training signal.
  - Local inference exactly reproduces cached scores (4-decimal match) — preprocessing parity confirmed.
- **Design note for P2:** 50-60% of vanilla SBI is "too easy" (<0.1 score) — **oversample the >0.5 SBI tail** in PE_SBI training (or use curriculum/hard-mining) rather than uniform sampling. The 27% in (0.1, 0.9) on P8A is the load-bearing slice.
- All ckpts found **locally cached** from `xinhe_cross_camera_audit_2026-05-06`'s prior local inference run — no gs:// pulls needed. ~3min wall-clock total.
- Plan update needed: yes
  - §8.1 0i — mark complete; verdict GREEN-by-direct-measurement.
  - §8.2 P2 recipe — add **tail-oversampling recommendation** (`p > 0.5` slice; ckpt-conditional ~37% on P8A).

### 2026-05-06 evening — Result: TTA_POC_MAY6_2026-05-06 (Phase 0a)
- Owner: Claude sub-agent (general-purpose)
- Status: complete
- Trigger: third-wave Phase 0 dispatch — addresses live may6 production false-flag
- Output: `analysis/tta_poc_may6_2026-05-06/{run_probe.py, outputs/{tta_scores.csv (504 rows), tta_scores_{E2B,P8A}.csv, fpr_comparison.csv, per_view_breakdown.csv, summary.json, verdict.json, probe_run.log, FINDINGS.md}}` (FINDINGS by parent — sub-agent harness blocked .md writes)
- Result summary: **VERDICT — TTA NOT VIABLE for may6 production false-flag.** E2B may6 FPR 57.6% → 56.5% with 4-view TTA (Δ = −1.1pp; net rescue ≈ 1 frame). Far short of <20% target.
- **Mechanism of failure:** may6 false-flags are CONFIDENT (median V1 fake-prob 0.58, q90=0.95, mean view-std 0.10). Only 4/92 frames sit in the V1 ∈ [0.45, 0.55] band where TTA could flip a τ=0.5 decision. q25 actually rises +0.15 with TTA (pushes some clean frames up) — TTA's effect is symmetric, not a damping mechanism. Consistent with `project_dor_drift_named_axes_2026-05-06`: drift lives on a named-axis ridge (`min_dim`, `color_b_dev`, `edge_mag`) at ~90% explained variance — **pixel-level local jitter operates on the wrong axis.**
- P8A may6 invariance preserved (0% → 0%). V1 baseline reproduces morning's audit (0.576) exactly — pipeline parity confirmed.
- Per-view independence: r(V1, V2_flip)=0.85, r(V1, V3_shift+scale)=0.81, r(V1, V4_shift+scale+flip)=0.73. No single augmentation drives substantive correction.
- **Implications for `deployment-vs-p8a-substrate-tradeoff-not-quantified` open loop:** TTA option REMOVED. Remaining levers:
  1. Substrate-aware τ-raise — OFFLINE-only (per per-mode-τ-not-deployable constraint).
  2. **Deployment-side switch E2B → P8A on may6-class substrates** — now the primary production-pain mitigation option. Concrete deployment-side action; trades P8A's higher live-prod FPR or lower fake recall on some cohorts.
  3. IQ gating per `project_image_quality_shortcut.md` — P8A-specific lever (P8A recall increases monotonically with sharpness; E2B inverted), non-starter for E2B.
- Plan update needed: yes
  - §8.1 0a — close as NOT_VIABLE_FOR_MAY6.
  - §9 deployment-vs-P8A-substrate-tradeoff — TTA option removed; deployment-side P8A swap + IQ gating remain.

### 2026-05-06 evening — Plan revision (consolidated, post-third-wave)
- Triggered by: all third-wave Phase 0 sub-agents complete (TTA_POC_MAY6, SBI_SCORE_COLLECTION)
- Sections edited: §8.1 0a (TTA closed NOT_VIABLE_FOR_MAY6), §8.1 0i (SBI score collection complete GREEN-by-direct-measurement), §8.2 P2 (tail-oversampling design note), §9 deployment-vs-P8A loop (TTA option removed)
- Summary of key shifts:
  1. **TTA is dead as production patch for may6.** E2B may6 FPR drops only −1.1pp (57.6 → 56.5). Mechanism: may6 misclassifications are confident (median fake-prob 0.58, view-std 0.10); TTA has no leverage on confident wrong predictions. Drift lives on a named-axis ridge (`min_dim`, `color_b_dev`, `edge_mag`); pixel-jitter operates on the wrong axis.
  2. **Deployment-side P8A swap on may6-class substrates is now THE primary production-pain mitigation option** (TTA dead, IQ gating P8A-specific only). Concrete deployment-side action; trade documented per `CHECKPOINT_COHORT_DIAGNOSIS`.
  3. **P2 SBI is GREEN-by-direct-measurement** (upgrades from GREEN-by-headroom). All 3 ckpts strictly bimodal; encoder picks up manipulation cues 17× stronger than pixel-IQ d (0.81 vs 0.047) — the SBI thesis empirically validated.
  4. **PE_SBI design refinement:** oversample SBI > 0.5 tail (~37% on P8A) rather than uniform sampling. The 27% in (0.1, 0.9) on P8A is the load-bearing training slice.
- Three-pillar prognosis unchanged: 15-25% single-packet, 40-55% multi-packet sequenced.
- Outstanding actions for the user (final state of Phase 0 work): (a) **decide Path A vs Path B for P1**; (b) **decide whether to swap deployment to P8A on may6-class substrates** as immediate production fix; (c) wait for PD scorecard verdict; (d) decide whether to add a parallel scratch arm; (e) authorise 0h+0j combined extraction (~$8-13, ~1hr GPU) IF Path A chosen.

## End of Phase 0

All eight Phase 0 audit-class jobs have completed (one with INSUFFICIENT_COVERAGE blocker requiring 0h follow-up). 13 FINDINGS.md files in `analysis/*_2026-05-06/outputs/` and `analysis/sbi_score_collection_2026-05-07/outputs/`. Plan ~970 lines. Activity log §12 above is the full audit trail.

**Three durable verdicts that constrain the next packet decision:**
1. FT base = P8A (cohort + pre-fix verdicts).
2. Pair-rank lever has GREEN/AMBER on training-substrate lanes (cross-product RED was eval-substrate-driven; tight-pair audit pending Phase 0h+0j).
3. SBI is GREEN-by-direct-measurement — P2 is launch-ready.

**Two newly required cheap follow-ups before P1 launches:**
1. Phase 0h paired-feature extraction (~$5-10, 1-2 A100-hours) — unblocks head-vs-encoder localisation.
2. Phase 0j forward-pass tight-pair audit on df40 + deeplive (~$3, 40 min) — converts 2/6 → 4/6 lane coverage.

Both can combine into a single ~$8-13, ~1hr forward-pass extraction.

**One closed deployment-side option:** TTA is not viable for may6.

**One emergent deployment-side option:** P8A-on-may6-substrate swap is now the primary production-pain mitigation lever.

### 2026-05-06 evening — User decisions on standing items
- Owner: user (Roee)
- Status: noted
- Decisions:
  1. **Path A authorised** for P1 — green light to launch the 0h+0j combined Vertex GPU extraction. Sub-agent dispatched to prepare scripts + submit Vertex job in us-east1.
  2. **CHECKPOINT_COHORT_DIAGNOSIS** — user acknowledged finding the file at `analysis/checkpoint_cohort_diagnosis_2026-05-06/outputs/FINDINGS.md`; no extra documentation work needed.
  3. **Scratch** — agreed deferred until FT-from-P8A reaches a plateau. Documented in §9 as `scratch-deferred-pending-FT-from-P8A-plateau`.
  4. **PD verdict** — expected in ~30 min; user will signal.
  5. **TRAINING_DATA_QUALITY_AUDIT** — user authorised CPU job to investigate whether training data contains samples too low-quality to be representative of deployment-time inputs.

### 2026-05-06 evening — Phase 1 launch + Phase 0 quality-audit dispatched (2 parallel sub-agents)
- Owner: Claude (Opus 4.7), parallel sub-agent dispatch
- Status: **complete — both launch agents returned; Vertex job `3077166152858730496` now executing remotely in us-east1**
- Trigger: user authorisation (Path A green light + TRAINING_DATA_QUALITY_AUDIT yes)
- Jobs dispatched:
  1. `PATH_A_LAUNCH_2026-05-07` → `analysis/path_a_launch_2026-05-07/` — prepares + submits the 0h+0j combined Vertex GPU extraction (us-east1, ~$8-13, ~40 min Vertex runtime). Sub-agent EXITS after submission with job_id; results processing follows in a separate subsequent agent. Output: extraction script, launch YAML, `launch_log.json` with job_id and expected GCS output paths, `RESULTS_PROCESSING_PLAN.md` for the post-extraction analyses.
  2. `TRAINING_DATA_QUALITY_AUDIT_2026-05-06` → `analysis/training_data_quality_audit_2026-05-06/outputs/` — historical-evidence-grounded analysis of training-data IQ distribution per lane (Probe 1 axes: lap_var_face, min_dim, color_b_dev, sat_std, edge_mag); estimates deployment-aligned quality floor by cross-referencing with production substrates; projects what fraction of training would be filtered at various thresholds.
- Result summary: pending — individual entries per job will be appended on completion. PD scorecard verdict expected in same window (~30 min).

### 2026-05-06 evening — Result: TRAINING_DATA_QUALITY_AUDIT_2026-05-06
- Owner: Claude sub-agent (general-purpose)
- Status: complete
- Trigger: user-flagged hypothesis on whether training contains samples too low-quality to be deployment-representative
- Output: `analysis/training_data_quality_audit_2026-05-06/{run_probe.py, outputs/{training_iq_distribution.csv, training_iq_distribution_all_lanes.csv, production_iq_distribution.csv, filter_impact_table.csv, filter_impact_table_all_lanes.csv, low_quality_score_distribution.csv, summary.json, verdict.json, FINDINGS.md}}` (FINDINGS by parent — sub-agent harness blocked .md writes)
- Result summary: **VERDICT — CONDITIONAL, defer.** Do NOT pull data-cleanup as a primary packet.
  - **Direction-flip on visomaster (load-bearing lane):** training viso lap_var p50 = 421 (224×224 cache) vs production p50 = 78 (~5.4× sharper). Only 0.3% of viso training falls below prod_p05. **User's hypothesis "training has degraded long tail" is REFUTED on viso.** The successful prior intervention (P22 aug curriculum, +3× macro recall) went the OPPOSITE direction (make training softer). A QUALITY_FLOOR packet going the other way is structurally inconsistent with the proven lever.
  - **Confirmed on DF40 lanes:** 47.9% below prod_p10 lap_var; 5.7% below prod_p05. DF40 has the long tail. But DF40 is down-weighted (`fw=0.2`); recipe-weighted impact muted.
  - **IQ shortcut is alive in score data:** bottom-5% lap_var EVAL samples have P8A median score 0.32 vs 0.01 (32× inflation). Real shortcut signal — but lever to weaken it has structural problems.
  - **Data-axis lever has failed twice** (P14_DATA_FIX, P16_DATA_AXIS). PE_QUALITY_FLOOR's "removal" framing is weak distinction from "re-weighting at fw=0". Memory `project_data_axis_lever_pulled_twice_no_lift.md` explicitly warns against this.
  - Recipe-weighted: ~5-15% of training removed at recommended threshold; bulk concentrated in down-weighted df40.
- **Critical caveat — resolution asymmetry:** viso training cache is 224×224 post-resize; production is native ~219-373px. lap_var at lower resolution is INFLATED. Qualitative finding (viso train ≥ production sharpness) is robust; quantitative 5.4× is soft. **A targeted ~5,000-sample native-resolution GCS scan (~$5) would resolve and either harden CONDITIONAL → NO or upgrade to YES.**
- **Recipe sketch (smoke test only):** PE_QUALITY_FLOOR with `min_dim_min=200` AND `lap_var_min_native=15` joint filter; only worth running AFTER P22 substrate-generalisation verified AND resolution-asymmetry caveat resolved. Estimated lift 0 to +5pp viso recall, bounded above by P22's already-realised 3× lift.
- Plan update needed: yes
  - §9 — add `pe-quality-floor-deferred-after-resolution-caveat-and-p22-verification` as deferred sub-component.
  - §8.X — note that QUALITY_FLOOR is NOT a primary candidate; flagged for hypothetical future packet if (a) P22 generalisation verified AND (b) native-res GCS scan upgrades the verdict.

### 2026-05-06 evening — Result: PATH_A_LAUNCH_2026-05-07
- Owner: Claude sub-agent (general-purpose); Vertex job runs remotely after agent exit
- Status: agent complete; **Vertex job `3077166152858730496` RUNNING in us-east1**
- Trigger: user authorisation of Path A
- Output: `analysis/path_a_launch_2026-05-07/{extract_paired_features.py, launch.sh, launch_yaml/extraction_job.yaml.template, launch_yaml/extraction_job.20260506-184624.rendered.yaml, launch_log.json, RESULTS_PROCESSING_PLAN.md, run.log}`
- Result summary: **Vertex job submitted successfully on third attempt.** Two prior submission attempts failed and are documented in `launch_log.json` under `previous_attempts`:
  1. Image tag `1.3.268` from `VERSION` file didn't exist in registry (only `1.3.267` was built); fixed by pinning version explicitly.
  2. Container bootstrap used `gsutil cp` which falls back to anonymous on Vertex AI (gsutil doesn't pick up the service-account ADC the way the Python SDK does); fixed by replacing `gsutil cp` with `python -c "from google.cloud import storage; ..."` matching the pattern in `entrypoint.sh`.
- Job details:
  - **job_id:** `3077166152858730496`
  - **display name:** `path-a-extract-20260506-184624`
  - **region:** us-east1
  - **state at agent exit:** JOB_STATE_RUNNING (transitioned from PENDING ~1 min after submit, 2026-05-06 16:47:25 UTC)
  - **image:** `us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:1.3.267`
  - **expected runtime:** ~40 min, $8-13
  - **monitor:** `gcloud ai custom-jobs describe 3077166152858730496 --region=us-east1 --project=train-cvit2`
  - **stream logs:** `gcloud ai custom-jobs stream-logs 3077166152858730496 --region=us-east1 --project=train-cvit2`
- **Schema correction (load-bearing):** features are **512-d post-projection** (Effort detector applies `visual.proj` 768→512), NOT 768-d as in the original brief. This matches existing `analysis/clip_vs_p8a_viso_2026-05-03` schema and what `frozen_pair_head_probe_2026-05-06/run_probe.py` expects.
- pair_gaps.csv input: 5,311 unique frames (1,812 real + 3,499 fake).
- Expected outputs (post-completion):
  - `gs://training-job-outputs/analysis_outputs/frozen_pair_features_2026-05-07/p8a_paired_features.npz`
  - `gs://training-job-outputs/analysis_outputs/frozen_pair_features_2026-05-07/e2b_paired_features.npz`
  - `gs://training-job-outputs/analysis_outputs/frozen_pair_features_2026-05-07/clip_b16_raw_paired_features.npz`
  - `gs://training-job-outputs/analysis_outputs/frozen_pair_features_2026-05-07/frame_manifest.csv`
- Inputs uploaded to GCS for reproducibility: `gs://training-job-outputs/path_a_inputs_2026-05-07/{extract_paired_features.py, pair_gaps.csv}`.
- **Post-completion runbook:** `analysis/path_a_launch_2026-05-07/RESULTS_PROCESSING_PLAN.md` — 7-step plan: (1) verify success → (2) pull NPZs → (3) re-run Phase 0j same-source-pair-gap audit on tight pairs → (4) re-run Phase 0h frozen_pair_head_probe with new feature caches → (5-7) decide P1 path with explicit promotion gates (Head B ≥3pp lift over Head A → demote P1 to head-only retrain; no lift → P1 justified iff lane-restricted launch acceptable).
- Plan update needed: no immediate plan-section changes; will append result-processing entry once job completes (~40 min).

### 2026-05-06 evening — Result: Vertex job 3077166152858730496 SUCCEEDED (Path A extraction)
- Owner: Vertex AI us-east1 (job submitted by sub-agent)
- Status: **JOB_STATE_SUCCEEDED**
- Trigger: PATH_A_LAUNCH submission
- Runtime: **3 minutes** (16:47:25 → 16:50:27 UTC). Much faster than the 40-min estimate; A100 spot was very capable.
- Outputs verified in GCS at `gs://training-job-outputs/analysis_outputs/frozen_pair_features_2026-05-07/`:
  - `p8a_paired_features.npz` (6.87 MiB) — 5,311 frames × 512-d post-`visual.proj` [CLS] features + scores.
  - `e2b_paired_features.npz` (7.09 MiB) — same schema.
  - `clip_b16_raw_paired_features.npz` (6.82 MiB) — same schema (CLIP-B16 raw baseline).
  - `frame_manifest.csv` (901 KiB) — frame_path / sample_id / frame_idx / label / pair_id mapping.
  - Total 21.66 MiB.
- Cost realised: ~$1-2 (3 min × A100 spot).
- **Next action (when user authorises):** dispatch results-processing agent per `analysis/path_a_launch_2026-05-07/RESULTS_PROCESSING_PLAN.md` 7-step runbook — pull NPZs locally, re-run 0j tight-pair audit, re-run 0h frozen_pair_head_probe, decide P1 path with promotion gates (Head B ≥3pp lift → demote P1 to head-only retrain; no lift → P1 justified iff lane-restricted launch acceptable).
- Plan update needed: yes — §8.1 0h+0j status updated to "Vertex extraction complete, results-processing pending"; results entry will follow when processing agent runs.

### 2026-05-06 evening — PD scorecard artifacts available
- Owner: user (delivered to local repo)
- Status: artifacts present at `analysis/pd_scorecard_artifacts_2026-05-06/` (~42 MiB)
- Trigger: user signal — PD scorecard ready
- Per the directory's `README.md` (10,994 bytes; read in full):
  - **232 cells** = 8 ckpts × 29 contract suites, complete coverage. Two-phase run:
    - Original Vertex job `8207399447131324416` (image 1.3.267, us-east1, launched 2026-05-06T11:09:32Z) FAILED at 16:27:59Z due to **boot-disk exhaustion** (per-cell ckpt cache filenames unique per-suite; 157 × ~896 MB filled the default 200 GB boot disk). Wrote 157 cells across 19 fully-covered + 1 partial suite before crashing.
    - Resume Vertex job `4910131201198522368` (image 1.3.268 with new in-image suite YAML, us-east1, launched 17:04:31Z) SUCCEEDED at 18:10:38Z (1h6m). Boot disk bumped to 500 GB via `--yaml-template /tmp/vertex_job_template_500gb.yaml` override. Re-ran 1 partial suite + 9 missing per-capture-mode mini-slices = 80 cells across 10 suites.
- **8 ckpts in scope:**
  - `p8a_reference_step5000` — production anchor (run `9lmvb5b4` step5000)
  - `e2b_top_n_step3200` — FT base for both PD arms; deployment-equivalent (run `rmat8lwx`)
  - `deeplive_corr_top_n_step4800`, `deeplive_corr_top_n_step1800`, `deeplive_corr_periodic_step2000` — PD deeplive arm (run `8jgyw1am`)
  - `viso_corr_top_n_step600`, `viso_corr_periodic_step2000`, `viso_corr_periodic_step1000` — PD viso arm (run `7u3zc5zt`)
- **PD recipe:** both arms FT-from-E2B with `correlation_penalty(λ=1.0)` on three axes — `sharpness_laplacian`, `luma_mean`, `face_area_fraction`. Single-lever discipline (anchor_aware + face_scale_jitter + arcface_head all OFF). Arms differ only in whether visomaster training data is enabled (deeplive arm: disabled; viso arm: enabled at fw=4.0 mirroring PA).
- **29 suites:** 6 dev (covering all method families) + 4 Teams lockbox mirrors + 1 Dor diagnostic + 18 per-capture-mode dev mini-slices. From `arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml`.
- **File structure:**
  - `unified_scorecard_simple.csv` — 232 rows; merged rollup; HEADLINE READ. Schema: `suite, ckpt, run, n_videos, n_real, n_fake, accuracy_at_0p5, real_fpr_at_0p5, fake_recall_at_0p5`.
  - `coverage_manifest.csv` — per-cell provenance + artifact completeness.
  - `scorecards_resume_only/{scorecard.csv, scorecard.wide.csv, scorecard.int8_delta.csv, scorecard.json}` — cloud-side rollups for the 10 resume suites only.
  - `reports_original/` — 628 files / 37 MiB (157 cells × 4 artifacts each).
  - `reports_resume/` — 320 files / 2.4 MiB (80 cells × 4 artifacts; 3 cells lack `videos_report.csv` but flow into unified via `scorecard.csv` confusion-matrix columns).
  - Per-cell artifacts: `<suite>_<ckpt>_summary_report.txt`, `_frames_report.csv`, `_group_metrics.csv`, `_videos_report.csv`.
- **Caveats (from README §Caveats):**
  1. **Threshold τ=0.5 is DIAGNOSTIC, not deployment-grade.** For deployment-style numbers, the per-substrate τ-calibration via `arena/launch_teams_promotion_contract.sh` is authoritative.
  2. 7 cells partial (have <4 of 4 artifact types); 3 lack `videos_report.csv` but ARE represented in resume `scorecard.csv` and flow into `unified_scorecard_simple.csv` correctly.
  3. `teams_flat_xiang_xiang2_feng_dev` appears in both report dirs; resume's 8 cells overwrite original's 5; unified manifest reflects this de-dup (resume wins).
  4. Resume `scorecard.csv` has richer columns (checkpoint_path, label_mode, split_hint) for 10 resume suites only.
- **What this scorecard ANSWERS** (from these files, no further work):
  - F1-style **Teams headline numbers** (real_FPR / fake_recall at τ=0.5) for each (ckpt × suite).
  - Per-video breakdowns via `reports_*/<suite>_<ckpt>_videos_report.csv`.
  - Cross-ckpt deltas for the 8 ckpts on each of the 29 contract suites.
- **What this scorecard does NOT ANSWER** (require separate audits):
  - **F1 lockbox for `visomaster_enhanced_macro` / `deeplive_enhanced` / `teams_real_dor`** — those lockbox suites are NOT in this scorecard's manifest. Only the Teams real/fake/lighting/poor-quality lockbox mirrors are present.
  - **F2 (shortcut weakening ≥30% on ≥2 of 5 axes)** — needs the 5-axis correlation audit. Phase 1 baseline at `../deeplive_viso_corr_eval_2026-05-06/abs_pearson_summary.csv`; Phase 2 (PD ckpts vs baseline) NOT yet run.
  - **F3 (no untargeted axis +50%)** — same data needed as F2.
  - **F4 (HDTF cross-substrate FPR ≤5%)** — HDTF substrate is in `arena/target_domain_suites.proper_data_future.provisional_2026-04-19.yaml`, not this scorecard.
- **Conclusions NOT drawn** per user instruction. Headline numbers in `unified_scorecard_simple.csv` will be analysed when user signals readiness.
- Plan update needed: §8.1 0f status updated to "scorecard artifacts available locally; analysis pending user signal"; full plan revision will follow once analysis runs.

### 2026-05-07 — Plan revision: §8.2 P1 FT-base + grad-audit framing tightened
- Owner: parent agent (Opus 4.7), under explicit user direction
- Trigger: user request to tighten §8.2 P1 wording on the FT base and the grad-audit's role
- Sections edited: §8.2 P1 — FT-base bullet (formerly "retrain FT-from-P8A"), grad-audit hook bullet
- **What changed:**
  - FT-base bullet now states explicitly: FT base = **existing** `P8A_REFERENCE_STEP5000` (run `9lmvb5b4`, step 5000); training runs on the post-`2feea58` codepath which activates `in_proj_svd` q/k/v classification gradient for the first time on this lineage; **no prerequisite clean retrain of P8A**. Removes ambiguity about whether a fresh P8A baseline was a prerequisite. The cohort-diagnosis evidence and post-fix codepath rationale are preserved verbatim.
  - Grad-audit hook bullet now framed as **plumbing-check, NOT a promotion gate.** Binary semantics: zero gradient → abort the run (the `2feea58` fix has regressed silently); nonzero → continue regardless of magnitude. Plus the explicit cite to `packets/P8A.md:108` — under the P10_SYM-on-P8A recipe, post-fix `in_proj_svd` ties within noise vs C.1, so a "nonzero gradient + flat metrics" outcome is consistent with the lever being unhelpful for this recipe rather than a regression of the bug fix. Promotion gated on F1-F5 metrics only.
- **Why this matters:** prior wording could have been read as "the lever WILL unlock and that's the FT-base rationale." The new wording separates the plumbing-check (was the bug fix preserved?) from the value question (does the lever actually move metrics under PE_PAIR_RANK_DRO's loss class?). The P10_SYM C-ablation null is a directional prior on the value question, not a binding prediction.
- **No change** to P1's close criteria (F1-F5), recipe, single-lever discipline, or confidence range. The trigger conditions in §8.1 0g remain authoritative.
- **No change** to §6.5 (in-proj SVD bug timing argument). User flagged §8.2 specifically; §6.5 stays as-is unless a follow-up tightening is requested.

### 2026-05-07 — Path A processed + P1 wiring landed + Slots 1 & 2 launched
- Owner: parent agent (Opus 4.7), under explicit user direction (auto mode)
- Trigger: user authorised launching Slots 1 & 2 + 30-min Option C job

**Path A subagent verdict** (FINDINGS at `analysis/path_a_launch_2026-05-07/FINDINGS.md`):
- Vertex extraction job `3077166152858730496` ran 3.03 min (~$0.10–0.23 vs $8–13 estimate). Outputs sit at `gs://training-job-outputs/analysis_outputs/frozen_pair_features_2026-05-07/`.
- Phase 0j on cached scores: **P8A meets ≥2-GREEN bar** — `deeplive_v1` 0.255 (GREEN) + `viso_enhanced` 0.355 (GREEN) + `deeplive_v2` 0.154 (AMBER). E2B is RED on both measurable lanes (reinforces FT-base = P8A).
- Phase 0h head-vs-encoder probe: **head-only retrain NOT viable.** P8A B−A lift = −0.13pp; E2B = −0.38pp; CLIP_B16_raw = 0.00pp. Multi-seed sensitivity (5 seeds × 50 epochs): paired B−A AUC mean = −0.028pp ± 0.039pp. HP sensitivity (3 hp × 3 seeds): all ≈ 0pp. Lift is robustly zero or slightly negative.
- **Structural caveat that downgrades the verdict's reach**: Phase 0h substrate is 100% `teams_passthrough` because 32% of `pair_gaps.csv` rows have `gs://local/...` paths (Dor + extra + dor-fake-local lanes). The runbook's "six paired training lanes" (df40, viso_v1, viso_enhanced, viso_teams_enhanced, deeplive_v1, deeplive_v2) **literally never appear** in `pair_gaps.csv` — it was built from eval-substrate cross-products, not training tight pairs. Phase 0h cannot answer the head-vs-encoder question on the load-bearing training lanes from this extraction.

**Slot 3 (Option C) — SKIPPED.** Subagent recommended re-extracting on a non-`gs://local` filtered manifest. Verified empirically: filter drops 12,000 rows (32.2%), all from non-runbook lanes (extra, dor_evening, dor_morning, visomaster_v2_dor, dor_fake_local, live_fakes_teams_prod). Surviving 25,327 rows are 100% teams_passthrough (`teams_real_all_dev/lockbox` × `teams_fake_all_dev/lockbox`) — the same substrate Phase 0h already tested. Filtering can't unlock viso/deeplive lanes because they were never in the CSV. To actually answer the head-vs-encoder question on training tight pairs requires instrumenting the loader to dump `(sample_id, frame_idx)` tuples at training time — multi-hour task, not 30 min.

**P1 wiring committed in `b50f245`** (12 files, +3,668 LOC):
- Loss-side: `_compute_pair_rank_loss` in `detectors/effort_detector.py` (softplus margin in logit space; skips empty pair_ids and pair_ids missing one label class). `pair_rank_loss.{lambda, margin}` config block with re-apply allowlist entry in `train_sweep.py` (W&B nested-dict flattening fix per `project_wandb_flattens_nested_dicts.md` memory).
- Loader-side: `combined_paired_collate_fn` emits per-video `pair_id` (= sample_id) and `group_id` (passthrough or derived in-collate from method/source/identity via embedded `make_group_id` parity with the audit snippet). Build pre-pass in the data pipeline computes `group_id_mapping` from `all_samples × {label=0, label=1}` at config-load.
- Trainer-side: `trainer/mixins/group_dro.py` rewritten to support `group_id_mapping` (str→int, R-D / F-B keying) alongside legacy `method_mapping`. Adds `warmup_steps` (default 100). Unknown group_ids bucket to 0 with one-shot warning. `train_sweep.py` wires `group_id_mapping` from `data_split_stats` → `data_params`.
- Tests: 36 new in `tests/test_pair_rank_and_group_dro.py` (margin behaviour, gradient flow, collate emission, mixin path resolution, snippet parity, derive helpers). All 67 tests in the touched suites green.
- YAMLs: `experiments/phase2_round13/R13_P1_BUNDLE_FT_FROM_P8A.yaml` (Slot 1; pair-rank + GroupDRO bundle) + `R13_P1_PAIRRANK_ONLY_FT_FROM_P8A.yaml` (Slot 2; pair-rank only single-lever ablation).

**Image release**: `1.3.269` (`38beaf6`). Cloud Build `93c90398-62c2-40f9-8ab4-cf23a4e556c9`, 20m7s, SUCCESS. Digest `sha256:ab725ae6681fd030c41016f60f318a7f7b103c818c991cbf15487b24e4af97a1`.

**Vertex jobs launched** (us-east1):
- Slot 1 (BUNDLE): job `6083600379105247232`, RUNNING at t+4m, expected ~5–7h, ~$50–80.
- Slot 2 (PAIRRANK_ONLY): job `6401104152834867200`, RUNNING at t+4m, expected ~5–7h, ~$50–80.

**Pending for tomorrow morning**:
- Day-1 verdict: did Slot 1 vs Slot 2 ablation reveal the load-bearing lever (pair-rank alone vs pair-rank+GroupDRO)? Compare via promotion-contract scorecard against P8A reference.
- Grad-audit at periodic checkpoints step 100/500/1000: zero gradient → fix regressed silently (abort and investigate); nonzero + flat metrics → lever inert under this loss class (consistent with `packets/P8A.md:108` P10_SYM C-ablation prior; not a bug). Promotion gated on F1-F5 metrics only.
- F2/F3 audit on PD ckpts (still in `analysis/pd_scorecard_artifacts_2026-05-06/` scope, awaiting user signal per "Conclusions NOT drawn yet" instruction from 2026-05-06 evening).
- If Slots 1 + 2 close any of F1–F5: deployment swap consideration. If both fail: P2 (PE_SBI) becomes the next move; SBI loader still needs implementation (target-domain pseudo-fake generator + auxiliary mix at 15-25%).
- Loader instrumentation for training-tight-pair extraction: would let Phase 0h actually answer the head-vs-encoder question on viso/deeplive/df40 lanes. Multi-hour task; gated on Slot-1/2 verdict.
