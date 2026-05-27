# Handoff — P18 corrective diagnostics A/B/C/E/F/G complete; D launching

**Date generated**: 2026-05-02 morning CEST
**Branch**: `teams-relaunch-root-2026-04-17`
**Predecessor**: [`HANDOFF_2026-05-02_P18_CORRECTIVE.md`](HANDOFF_2026-05-02_P18_CORRECTIVE.md) (the same-day handoff with the inconclusive verdict)

> **What changed**: I ran diagnostics A, B, C, E from the corrective handoff plus the two foundational sanity checks (F, G). The picture is now substantially less ambiguous than the corrective handoff suggested, but **also substantially less interesting**. P18T does NOT add new invariance over P8A — it preserves P8A's invariance while preventing FT-induced regression. Diagnostic D (Vertex contract scorecard) is launching to confirm whether this preservation translates to deployment-grade promotion.

---

## TL;DR

The previous corrective probe ("β bite, n_dor=9, inconclusive") had no P8A baseline. With P8A in the same pipeline:

| Operating point | Metric | **P8A** | P18T (treatment) | P18C (no-GRL control) |
|---|---|---:|---:|---:|
| τ=0.5 (raw) | dor lockbox real FPR | **0.12** | 0.40 | 0.92 |
| τ=0.92 (deployment, **s-scaled**) | dor lockbox real FPR | **0.00** | **0.00** | **0.52** |
| τ=0.92 (deployment, s-scaled) | nondor Teams real FPR @ lockbox | 0.33 | **0.20** | 0.20 |
| τ=0.974 (deployment, s-scaled) | dor lockbox real FPR | 0.00 | 0.00 | 0.16 |
| τ=0.974 (deployment, s-scaled) | nondor Teams real FPR @ lockbox | 0.13 | **0.07** | 0.20 |
| τ=0.92 (deployment) | aggregate lockbox real FPR (n=40) | 0.12 | **0.07** | 0.40 |
| τ=0.92 (deployment) | aggregate lockbox fake recall (n=47) | **0.47** | 0.38 | 0.43 |

**Three clean conclusions from CPU diagnostics:**

1. **P18T preserves P8A's deployment-τ behavior on dor reals (both 0% FPR at τ=0.92).**
   It does NOT improve over P8A on dor reals. The "GRL bites the dor shortcut" framing
   from the corrective handoff is wrong in direction — P8A already has 0% FPR there;
   GRL can't go below zero. What GRL DOES do is prevent the FT-only control from
   regressing to 52%.

2. **P18C catastrophically regresses from P8A on dor reals (12% → 92% at τ=0.5;
   0% → 52% at τ=0.92).** FT-only on the new training data introduces a strong
   dor_shkedi shortcut. GRL prevents this regression.

3. **At τ=0.974 P18T improves over P8A on nondor Teams reals (13% → 7% FPR)** with
   a small recall trade (47 → 38% on lockbox fakes). Net deployment value depends
   on the contract floor — answered by D.

The local-substrate "lockbox" subset is small (n=40 reals, n=47 fakes) and dominated by
Teams content. **D (Vertex contract scorecard, 3 ckpts × full lockbox)** is launching
in image 1.3.241, region us-east1, to test whether these patterns hold at the
contract-relevant scale and operating points.

---

## Diagnostics — what was run, what was found

All scripts under `analysis/p18_probe_2026-05-01/`. Outputs under `outputs/`.

### G — L3 hook fidelity (foundational)

**Script**: `verify_l3_hook.py`
**Question**: Does the forward hook on `transformer.resblocks[3]` in `corrective_probes.py`
capture the same tensor as the cached features at `intermediate__P8A__layer03__n800.npz`?

**Result**: PASS. Per-frame max abs diff = 1.9e-6 (float32 noise floor); cosine = 1.0 on all
64 frames sampled. The L3 hook in the corrective probe pipeline is correct.

### F — P8A baseline through the same corrective pipeline

**Script**: `corrective_probes.py --label P8A_step5000_FINAL --arm baseline`
**Output**: `outputs/corrective_probes__P8A_step5000_FINAL.json`
**Question**: What do the corrective-probe metrics look like on P8A?

**Result** (with the original corrective_probes.py code — sigmoid(margin), no s-scaling):

| Metric | **P8A** | P18T | P18C |
|---|---:|---:|---:|
| Within-bucket-3 dor-vs-other LR @ FINAL [CLS] AUC | **0.9826** | 0.9536 | 0.9965 |
| Within-bucket-3 dor-vs-other LR @ L3 AUC | 0.8194 | 0.8194 | 0.8194 |
| Fresh-LR L3 dev→lockbox AUC | 0.9495 | 0.9404 | 0.9527 |
| Lockbox dor real mean prob_fake (no s-scaling) | **0.439** | 0.494 | 0.549 |
| Lockbox dor real FPR @ τ=0.5 (no s-scaling) | **0.12** | 0.40 | 0.92 |
| Lockbox other-Teams real FPR @ τ=0.5 | 0.53 | 0.60 | 0.80 |

**Crucial reread of corrective handoff**: the corrective handoff's "β bite" framing read T's
0.40 < C's 0.92 as evidence GRL biting the shortcut. With P8A baseline at 0.12, this is more
honestly read as **C catastrophically regresses from P8A; T regresses less than C; neither
matches P8A**. GRL's actual role is defensive (prevent FT regression), not additive.

### A — Bootstrap CIs on within-bucket-3 LR AUC

**Script**: `run_diagnostics_abce.py` (Diagnostic A section)
**Output**: `outputs/diagnostics_abce_2026-05-02.json` (key `A_*`)
**Question**: Are the within-bucket-3 LR AUC differences statistically robust at n_dor=9?

**Result**: NO at the AUC bootstrap level. All paired delta CIs include 0:

| Pair | Δ AUC | 95% CI |
|---|---:|---:|
| P18T - P18C | -0.032 | [-0.104, +0.000] |
| P18T - P8A | -0.026 | [-0.080, +0.000] |
| P18C - P8A | +0.006 | [-0.002, +0.022] |

This is a power problem (n_dor=9 inside bucket-3) — not a "no-effect" finding.
Diagnostic B (paired Wilcoxon on actual model scores) does have power and finds
strongly significant differences. Don't read A as "no signal"; read it as
"AUC bootstrap can't see signal at this n; use B for the per-frame test instead".

### B — Per-frame paired Wilcoxon on dor lockbox reals (n=25)

**Script**: `run_diagnostics_abce.py` (Diagnostic B section)
**Output**: `outputs/diagnostics_abce_2026-05-02.json` (key `B_*`)
**Question**: For each of the 25 dor_shkedi lockbox real frames, does T systematically
score them differently from C / P8A?

**Result**: HIGHLY SIGNIFICANT for all comparisons on dor reals:

Unscaled scores (sigmoid(margin)):

| Comparison | median Δ p_fake | Wilcoxon W | p (two-sided) |
|---|---:|---:|---:|
| P18T - P18C (dor) | -0.058 | 0.0 | <0.0001 |
| P18T - P8A  (dor) | +0.059 | 1.0 | <0.0001 |
| P18C - P8A  (dor) | +0.114 | 0.0 | <0.0001 |
| P18T - P18C (nondor Teams, n=15) | -0.021 | 15.0 | 0.0084 |
| P18T - P8A  (nondor Teams) | -0.002 | 43.0 | 0.36 (NS) |
| P18C - P8A  (nondor Teams) | +0.032 | 31.0 | 0.11 (NS) |

s-scaled scores (sigmoid(s × margin), s=9.75 for P8A, 12 for P18):

| Comparison | median Δ p_fake | Wilcoxon W | p (two-sided) |
|---|---:|---:|---:|
| P18T - P18C (dor) | -0.387 | 0.0 | <0.0001 |
| P18T - P8A  (dor) | +0.310 | 0.0 | <0.0001 |
| P18C - P8A  (dor) | +0.762 | 0.0 | <0.0001 |
| P18T - P18C (nondor Teams) | -0.041 | 22.0 | 0.030 |

Key reads:
- **T scores dor reals less fake than C** (Δ -0.058 unscaled, -0.387 scaled; both p<0.0001).
  GRL has a real, systematic effect on the dor identities.
- **T scores dor reals MORE fake than P8A** (Δ +0.059 unscaled; +0.310 scaled). T does
  not match P8A on dor — it's a smaller regression than C, not an improvement.
- **C scores dor reals MUCH more fake than P8A** (Δ +0.114 unscaled; +0.762 scaled).
  This is the catastrophic FT regression GRL is counteracting.
- **All nondor-Teams T-vs-P8A comparisons are NS at unscaled / non-significant or weak at scaled.**
  GRL's effect is concentrated on dor (the targeted axis), not generalized.

### C — ArcFace s-scaling and deployment-τ FPR

**Script**: `run_diagnostics_abce.py` (Diagnostic C section)
**Output**: `outputs/diagnostics_abce_2026-05-02.json` (key `C_*`)
**Question**: At deployment τ ∈ {0.92, 0.974}, with proper sigmoid(s × margin) scaring, what
are the FPR / recall numbers?

**Key result** (lockbox reals, by identity class):

| Class \ τ | scaled τ=0.92 P8A | P18T | P18C | scaled τ=0.974 P8A | P18T | P18C |
|---|---:|---:|---:|---:|---:|---:|
| dor lockbox real (n=25) | **0.00** | **0.00** | **0.52** | 0.00 | 0.00 | 0.16 |
| nondor Teams lockbox real (n=15) | 0.33 | **0.20** | 0.20 | 0.13 | **0.07** | 0.20 |

**The 52% FPR for P18C on dor reals at deployment τ=0.92 is the headline finding.**
P8A and P18T both hold 0% FPR there. This 52pp gap explains why GRL matters for production
deployment — without it, FT-from-P8A on the new training data introduces a shortcut that
fires fakes at deployment τ on a large fraction of dor real frames.

P18T also slightly improves over P8A on nondor Teams reals at τ=0.92 (33% → 20%) and
τ=0.974 (13% → 7%).

### E — Within-bucket-3 LR AUC at intermediate layers

**Script**: `run_diagnostics_abce.py` (Diagnostic E section)
**Output**: `outputs/diagnostics_abce_2026-05-02.json` (key `E_layer*`)
**Question**: Where in the encoder stack is the GRL effect concentrated?

**Result**: with l2norm-row normalization (per the original corrective_probes' L3 convention):

| Layer | P8A AUC | P18T AUC | P18C AUC | Notes |
|---|---:|---:|---:|---|
| L3 | 0.819 | 0.819 | 0.819 | identical across arms — frozen by FT |
| L6 | 1.000 | 1.000 | 1.000 | overdetermined LR; not informative at this n_dor=9 |
| L9 | 1.000 | 1.000 | 0.995 | tiny separation, CIs overlap |
| L11 | 0.964 | 0.944 | 0.993 | T < P8A < C, but CI 95% overlap |
| FINAL CLS | 0.994 | 0.968 | 1.000 | (different normalization; from A's bootstrap) |

The GRL's discriminator effect lives at the **final pooled CLS**, not at intermediate
layers. L3 features are unchanged by training (consistent with the prior intermediate-layer
probe). L6-L11 are not robustly distinguishable across arms with n_dor=9.

### Mini-scorecard on full 800-frame substrate (n=40 lockbox reals, n=47 lockbox fakes)

**Script**: `run_minicard.py`
**Output**: `outputs/mini_scorecard_2026-05-02.json`

Aggregate (cross-method) lockbox numbers:

| Metric | τ | P8A | P18T | P18C |
|---|---:|---:|---:|---:|
| FPR (reals) | 0.5 | 0.28 | 0.47 | 0.88 |
| FPR (reals) | 0.92 | 0.12 | **0.07** | 0.40 |
| FPR (reals) | 0.974 | 0.05 | **0.03** | 0.17 |
| Recall (fakes) | 0.5 | 0.68 | 0.72 | 0.81 |
| Recall (fakes) | 0.92 | **0.47** | 0.38 | 0.43 |
| Recall (fakes) | 0.974 | **0.34** | 0.30 | 0.28 |

**P18T trade vs P8A at τ=0.92**: −9pp recall for −5pp FPR. Whether this is net positive
depends on the contract weighting. The mini-scorecard says P18T is roughly equivalent or
slightly worse than P8A in deployment-grade aggregate terms, with a meaningful improvement
on the FPR distribution under tail crops (webcam: 36% → 21% at τ=0.92).

**Note the substrate's lockbox fakes**: only n=47 (36 from `teams_capture_cam_test_s33` +
11 from `teams_capture_pc_generator_s15`); 0 deeplive_enhanced. The deployment fake-recall
matters most on deeplive_enhanced — the contract scorecard tests this.

### Lockbox AUC decomposed by dor / nondor reals (`decompose_lockbox_auc.py`)

Splitting the 40 lockbox reals into 25 dor + 15 nondor:

| Subset | n_real / n_fake | P8A | P18T | P18C |
|---|---|---:|---:|---:|
| lockbox all | 40 / 47 | 0.793 | 0.673 | 0.497 |
| lockbox **minus dor reals** | 15 / 47 | 0.651 | 0.593 | 0.556 |
| lockbox **only-dor reals + all fakes** | 25 / 47 | **0.877** | 0.722 | 0.462 |
| lockbox **only-nondor reals + all fakes** | 15 / 47 | 0.651 | 0.593 | 0.556 |

**The 12pp lockbox-AUC gap (P8A 0.79 vs P18T 0.67) is almost entirely concentrated on the
25 dor lockbox reals**. Without dor, P8A and P18T are within 6pp (0.65 vs 0.59). With dor
isolated, P8A surges to 0.88 — P8A is uniquely good at recognizing dor identities as real.

This sharpens the strategic picture:
- **P8A's lockbox advantage is a dor-specific strength**, possibly because P8A's training
  data composition gave it sufficient dor invariance.
- **The FT data with heavy `deeplive_teams_*` weights destroys this dor invariance**: P18C
  (no GRL) drops to AUC 0.46 on the dor view (worse than chance because the model
  systematically over-flags dor reals as fake). Even P18T (with GRL) only recovers to 0.72,
  not 0.88.
- **A successful next packet must restore P8A's dor-specific strength while gaining on
  other axes**. Move 4 (paired same-identity contrastive) is the cleanest structural lever
  for this — same-identity pairs by construction prevent identity-as-shortcut.

### ROC analysis on the 87-frame lockbox subset (`run_roc_curves.py`)

Lockbox AUC (single-number summary on n=87 lockbox: 40 reals + 47 fakes):

| Arm | lockbox AUC |
|---|---:|
| **P8A** | **0.7926** |
| P18T | 0.6734 |
| P18C | **0.4973** (essentially random) |

P18C lost almost all lockbox discrimination — the FT-only run drove AUC to chance level.
P18T is +0.18 better than P18C (GRL's defensive role is worth ~18pp lockbox AUC) but still
−0.12 below P8A. **This sets a hard prior for D**: hard to imagine P18T beats P8A on the
full lockbox if it's already worse on the 87-frame subset. The full contract numbers will
tell.

Recall at fixed FPR floors (lockbox aggregate; n=47 fakes):

| Arm | FPR≤0.01 | FPR≤0.05 | FPR≤0.10 | FPR≤0.20 | FPR≤0.30 |
|---|---:|---:|---:|---:|---:|
| P8A | 0.298 | 0.383 | **0.447** | **0.532** | **0.745** |
| P18T | 0.298 | 0.319 | 0.383 | 0.468 | 0.660 |
| P18C | 0.149 | 0.170 | 0.191 | 0.277 | 0.362 |

**P8A dominates at every FPR floor.** P18T is uniformly inferior to P8A on lockbox aggregate.
This pre-empts the most optimistic reading of D — the GRL's defensive-against-FT-regression
role is *real and large* (preventing AUC collapse from 0.79 → 0.50) but does not deliver an
absolute model better than P8A.

---

## Diagnostic D — running

**Goal**: full-lockbox v3 contract scorecard for P8A + P18T + P18C, using the
2026-04-23-with-dor suite manifest (which adds `teams_real_dor_dev`).

**Setup**:
- Image: `1.3.241` (rebuilt 2026-05-02 to bake in the new checkpoint map yaml)
- Region: `us-east1` (per CLAUDE.md US-region preference for GCS-locality cost / throughput)
- Suite: `arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml`
- Checkpoint map: `arena/checkpoint_maps/teams_target_domain.p18_corrective_2026-05-02.yaml`
- Three checkpoints:
  - `P8A_REFERENCE_STEP5000` — `9lmvb5b4` step 5000 baseline
  - `P18T_GRL_TREATMENT_STEP4000` — `xpbvc1e4` periodic step 4000
  - `P18C_NO_GRL_CONTROL_STEP4000` — `rgt4kw2u` periodic step 4000

**Expected runtime / cost**: ~3h on a single A100, ~$12-20.

**The contract output to read**:
- `gs://training-job-outputs/test_results/teams_promotion_contract/<job-name>/promotion_contract/promotion_contract.json`
- `selected_threshold_scorecard.csv`
- `checkpoint_summary.csv`
- `promotion_winner.json`

The v3 default policy `target_fake_recall_min=0.70` is engaged (commit `974e033`,
verified by Vertex `5427016015462531072`). For each checkpoint, contract picks the τ
that achieves dev_fake_macro_recall ≥ 0.70 if such τ exists; else flags as "no τ
qualifies". P8A v3 verify already showed P8A's dev_fake_macro_recall caps at 0.136
under v3 default — likely no τ qualifies for P8A. Question for D is whether P18T
clears the floor that P8A doesn't.

**Decision tree on D's outcome**:
- If **P18T promotes** (clears 0.70 floor with reasonable lockbox FPR): P18T is a
  promotable model. Recommend memory entry + handoff for promotion. Possibly P19
  with stronger λ to push further.
- If **P18T does not promote but is closer than P8A**: GRL is doing useful defensive
  work. P19 is justified.
- If **P18T does not promote and is no better than P8A on contract metrics**: GRL's
  effect is real but production-irrelevant. Pivot to Move 4 (paired same-identity
  contrastive) is the next correct lever.
- If **P18T regresses materially from P8A on contract metrics**: confirms the recall
  trade is net-negative. Same Move 4 pivot, with confidence.

---

## Why the GRL doesn't remove the dor shortcut more aggressively

P18's training data (`R13_P18_METHOD_DOMAIN_GRL.yaml` `combined_paired.family_weights`):

```
visomaster_fake: 4.0
deeplive_non_enhanced_fake: 2.5
deeplive_enhanced_fake: 3.0
deeplive_teams_fake: 7.0           ← heaviest weight
deeplive_teams_real: 5.0           ← second heaviest
realpool_real: 1.5
external_real: 1.0
df40_fake: 0.2
df40_real: 0.5
```

`deeplive_teams_fake` carries **7.0** weight (highest in the family table); `deeplive_teams_real`
carries **5.0**. So bucket 3 (`deeplive_teams`, where `dor_shkedi`-cluster lives) is
strongly over-represented in the data, both real and fake. The GRL is fighting against
a 7× data-signal pressure on this bucket.

Implication: if a future packet wants to push the GRL further on the dor axis, **lowering
`deeplive_teams_fake` weight** (currently 7.0 → maybe 1.0 or 2.0) is a more direct lever
than raising λ. The data composition is overpowering the discriminator's gradient.

## What this corrects / supersedes from the predecessor handoff

The HANDOFF_2026-05-02_P18_CORRECTIVE.md said:

> Either of these could be true: β actually-bit interpretation; noise interpretation;
> α-but-tiny interpretation. The next agent should not proceed on either of my verdicts
> without running the diagnostics below.

The diagnostics now narrow this:

- **β-actually-bit was wrong-framed**. T does not "bite" P8A's invariance — P8A is
  already invariant on dor reals at deployment τ. T preserves that invariance against
  the FT-only regression that C exhibits.
- **noise interpretation is partially right** for the AUC bootstrap (Diagnostic A) but
  refuted for the per-frame Wilcoxon (Diagnostic B). The effect is real and large at
  the score level; the bootstrap on AUC at n_dor=9 has too little power to see it.
- **α-but-tiny is also half-right** for the dor → "α partial bite" reading: GRL has a
  real defensive effect, but at deployment τ this defensive effect is large (52pp FPR
  prevented), not tiny.

The **load-bearing reframe**: GRL's role isn't *additive shortcut removal*; it's
*defensive against FT-induced shortcut amplification*. This changes downstream packet
design — P19's job (if D justifies it) is to test whether stronger GRL or a different
loss form gives additive improvement, OR whether the right next move is to remove the
shortcut from the training data itself (Move 4).

---

## File inventory (this session)

```
NEW (uncommitted):
A  analysis/p18_probe_2026-05-01/verify_l3_hook.py             (Diagnostic G)
A  analysis/p18_probe_2026-05-01/extract_all_arms_layers.py    (cache T+C+P8A multi-layer features)
A  analysis/p18_probe_2026-05-01/run_diagnostics_abce.py       (Diagnostics A, B, C, E)
A  analysis/p18_probe_2026-05-01/run_minicard.py               (full-substrate mini-scorecard)
A  analysis/p18_probe_2026-05-01/outputs/corrective_probes__P8A_step5000_FINAL.json  (Diagnostic F output)
A  analysis/p18_probe_2026-05-01/outputs/diagnostics_abce_2026-05-02.json
A  analysis/p18_probe_2026-05-01/outputs/mini_scorecard_2026-05-02.json
A  analysis/_features_cache_2026-04-30/final_cls__{P8A,P18T,P18C}__n800.npz
A  analysis/_features_cache_2026-04-30/intermediate__P18T__layer{03,06,09,11}__n800.npz
A  analysis/_features_cache_2026-04-30/intermediate__P18C__layer{03,06,09,11}__n800.npz
A  arena/checkpoint_maps/teams_target_domain.p18_corrective_2026-05-02.yaml  (D launch map)
A  docs/relaunch_handoffs/HANDOFF_2026-05-02_P18_DIAGNOSTICS_COMPLETE.md     (this file)

VERSION bumped: 1.3.240 → 1.3.241 (image build to bake in new yaml)
```

## Operational state

- **D launching**: image 1.3.241 building via Cloud Build; submitting to us-east1
  once image is ready. Expected ~3h Vertex + ~10min build + queue.
- **Local CPU diagnostics**: complete and saved.
- **Working tree**: post-`aaf4d04` (the predecessor's commit). Not yet committed.
  Per project pattern, do not commit without explicit user OK.
- **No active wakeups / cron**: none.

## Memory entries to write after D returns

- Replace `project_p18_corrective_probe_2026-05-02.md` with a calibrated entry
  summarizing the load-bearing reframe (defensive-not-additive) and D's outcome.
- Possibly add `project_p8a_dor_invariance_baseline.md` capturing that P8A is already
  0% FPR on dor lockbox reals at deployment τ.

## Next-agent checklist

1. Read this doc.
2. Wait for D to finish (~3h after launch). Pull `promotion_contract.json` and
   `checkpoint_summary.csv`.
3. Check if any arm clears v3 floor (target_fake_recall_min=0.70).
4. If P18T clears: this is a promotable result. Write up.
5. If no arm clears: read the threshold_grid.csv and look at the closest-to-floor
   operating point per arm. Compare lockbox FPR / per-bucket breakdowns. Recommend
   next packet (P19 with stronger λ or Move 4 paired contrastive).
6. Update memory entries.
