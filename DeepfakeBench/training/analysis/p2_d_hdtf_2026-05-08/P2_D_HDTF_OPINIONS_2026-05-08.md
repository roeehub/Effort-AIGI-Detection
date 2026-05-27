# P2-D HDTF Phase C — OPINIONS / Stage 2 reading (2026-05-08)

> **Status: OPINION + SUGGESTION.** Past framings on this codebase have been
> demonstrably wrong (P14 bundle 5.7× weaker than isolated lever; P18 GRL
> "biting" verdict reversed; viso ceiling reframed twice). Read the FACTS
> doc (`P2_D_HDTF_FACTS_2026-05-08.md`) for numbers. This doc reasons about
> what the convergent dataset (P2 verdict + J1-J5 + IQ atlas + Stage 1
> R² + check (a) per-layer + check (c) Dor encoder + check (b) HDTF
> Phase C) jointly indicates for Stage 2.
>
> **Audience**: the user, who reserves the GPU spend authorization.
>
> **Author**: agent picking up `HANDOFF_HDTF_AND_STAGE2_DECISION_2026-05-08.md`.
> Read all 6 FACTS docs before any OPINIONS doc, per the FRESH-AGENT
> GUIDANCE in `STATE.md`. Where this doc disagrees with prior agents'
> OPINIONS docs, the disagreement is explicit (§6).

---

## 1. Headline reading

**The HDTF result reframes the Stage 2 decision space.** Three load-bearing
findings:

1. **P8A is the only ckpt with substrate-invariant Teams-transport
   robustness.** On HDTF, P8A's macro_fake_recall = 0.896 (worst-case 0.83
   on viso_enhanced_teams). E2B and P2D collapse to 0.48 / 0.54 macro,
   with worst-case 0.06 / 0.09. The clean–teams transport gap per ckpt is
   +0.10 (P8A) / +0.62 (E2B) / +0.81 (P2D).
2. **P2D's Phase A v2 viso-ceiling break is partial — clean-transport-only.**
   P2D matches or exceeds P8A on HDTF clean-transport visomaster_enhanced
   (0.96/0.98 vs 0.95/0.95) but reverses on the Teams-transport variants
   (P2D 0.09/0.09 vs P8A 0.83/0.86). The Phase A v2 substrate's
   `visomaster_enhanced_macro_dev` is a Teams-transport substrate; the
   v2 lift (P2D 16.5% > P8A 13.5%) does NOT generalize.
3. **P2D's Phase A v2 lockbox_real_fpr regression is substrate-bound.**
   P2D's v2 `lockbox_real_fpr=0.164` (9× E2B) does not appear on
   `proper_real_teams_lockbox` HDTF (P2D 0.001 vs E2B 0.001). Same axis,
   different substrate, different rate. The v2 lockbox is the only place
   where P2D's IQ-shortcut amplification (Stage 1 LOCKBOX R² 0.54 vs P8A
   0.40) translates to a Pillar-2 regression.

The convergent reading is: **P2D moved the encoder along an axis that
benefited the v2 lockbox substrate's IQ-shortcut layer at a cost to
P8A's Teams-transport substrate-invariance.** The cost is far larger
than the benefit on Pillar-3-relevant substrates.

---

## 2. What the data jointly indicates about the binding constraint

The strategic question (`HANDOFF` §6): "What is the binding constraint
on Pillar 3 (robustness across capture conditions) and what's the
highest-leverage intervention to attack it?"

My reading: **The binding constraint is Teams-transport substrate-invariance,
NOT IQ-shortcut on lockbox.** Evidence:

| evidence line | reading |
|---|---|
| HDTF teams-transport recall: P8A 0.85 vs E2B/P2D ~0.10-0.17 | substrate-invariance is rare; only P8A has it |
| HDTF clean-transport recall: all 3 ckpts at 0.79-0.98 | content channel is fine on clean; problem is transport-induced |
| Stage 1 R² on HDTF: 0.02-0.09 across all cells | IQ shortcut does NOT bind on HDTF (the substrate where Pillar 3 matters) |
| Stage 1 R² on LOCKBOX_TEAMS: 0.40-0.54 | IQ shortcut binds on v2 lockbox specifically |
| P8A LOCKBOX R²=0.40 + HDTF teams recall 0.85 | the IQ shortcut and substrate-invariance are independent axes |
| Memory `pa_does_not_generalize_to_hdtf_2026-05-05` | E2B family doesn't transport-generalize; only P8A does |
| Memory `per_layer_divergence_2026-05-06` | P8A vs E2B diverge at L10-11 (cosine p50 0.998→0.32) |

The v2 lockbox IQ-shortcut is real and measurable, but it's not where the
production deployment lives — production lives on Teams-transport frames,
where HDTF is the cleanest measurement and where E2B (the deployed model)
is at 0.48 macro recall. **A lockbox-IQ-targeted Stage 2a (IQ GRL) would
optimize the wrong substrate.**

This refines the program proposal §3.1's three-hypothesis frame:
- Hypothesis A (mostly IQ classifier) is dead per Stage 1.
- Hypothesis B (IQ + content compose, substrate-conditional) is right —
  but the substrate-conditional structure means the "where-it-binds"
  question dominates.
- A new reading: **the binding-constraint substrate is HDTF teams-transport,
  not v2 lockbox.** The Stage 2 close criterion needs to be about HDTF
  teams-transport recall, not about LOCKBOX R².

---

## 3. What changes my view from the prior OPINIONS docs

Where I diverge (in priority order):

### 3.1 Stage 1 OPINIONS' substrate-conditional close criterion is right but the WEIGHTING is wrong

`IQ_DECOMP_OPINIONS_2026-05-08.md` §4.3 lists:
- LOCKBOX R² drops ≥ 0.10 (the load-bearing one)
- DEV residual AUC stays ≥ 0.85
- HDTF residual AUC stays ≥ 0.95
- DEV_DEEPLIVE raw AUC doesn't regress > 0.05

The HDTF data adds an ADDITIONAL metric the prior reading didn't have:
**HDTF teams-transport recall must not regress vs P8A by > 5pp**
(per `MODEL_GOALS.md` promotion bar #4). At Phase A τ:
- P8A HDTF macro_fake_recall = 0.896
- E2B HDTF macro_fake_recall = 0.484 → **already 41pp behind P8A**
- P2D HDTF macro_fake_recall = 0.540 → **also 36pp behind P8A**

So **E2B and P2D both fail this criterion right now**. Any Stage 2 packet
that uses E2B or P2D as FT-base inherits this gap. The Stage 1 doc's
recommendation (§4.1: "GRL > sampler") and Path B (Stage 2c identity-axis
GRL) both presume one of E2B/P2D is the FT-base. **The data argues
P8A is the only structurally-correct FT-base** for any packet that
matters for Pillar 3.

### 3.2 The Dor OPINIONS doc's Path C ($45-75) is premature

`DOR_ENCODER_AXIS_OPINIONS_2026-05-08.md` §7 proposes Path C: 3 parallel
GPU packets (IQ-GRL ∥ identity-GRL ∥ both). Without (b) HDTF data, this
was reasonable. With (b) data, it's the wrong direction:
- Both IQ-GRL and identity-GRL are loss-axis interventions on the v2
  substrate. The data says the v2 substrate is not where the binding
  constraint is.
- $45-75 GPU spend BEFORE confirming the FT-base question is exactly the
  CPU-first-then-GPU violation `AGENT_GUIDE.md` Rule 3 warns about.

### 3.3 The Synthesis OPINIONS doc's Stage 2a slate is right axis, wrong slots

`P2_DEEPER_ANALYSIS_OPINIONS_2026-05-08.md` §6 proposes:
- Slot 1: STAGE2A_BASE (FT-from-best-base + Fourier, no GRL)
- Slot 2: STAGE2A_IQ_GRL (Slot 1 + IQ GRL)
- Slot 3: STAGE2A_IQ_BALANCED_SAMPLER

The doc's "FT-from-best-base" was contingent on (b) result. The data now
says: best-base is **P8A**, not D step3000 or E2B. (P2D = E2B family by
training lineage; P8A is the only ckpt with the property we want to
preserve.)

This reverses the slate's center of gravity. With P8A as FT-base:
- "Fourier aug" (originally a P2-D recipe element) is no longer load-bearing
  — Fourier aug was the perturbation that broke P8A's L11 IQ representation
  in the FT-from-CLIP-scratch family. Adding it on top of P8A FT might or
  might not work; not yet measured.
- "IQ GRL on top of Fourier" (Slot 2) addresses an axis that doesn't bind
  on HDTF. Lower priority.

---

## 4. My Stage 2 proposal

**Recommendation: ONE GPU packet from P8A FT-base, single-lever Fourier
aug, no GRL. CPU-first first.**

### 4.1 Pre-launch CPU diagnostic (Stage 2-pre, ~$0, ~2-4 hours)

The single CPU diagnostic that would update my belief: **P2D HDTF cells
added to the IQ R² decomposition (Stage 1)**. The current
`IQ_DECOMP_FACTS_2026-05-08.md` §2.3 has P2D's HDTF cells absent. Now that
HDTF per-frame scores for P2D are cached locally, computing R² for the
4 P2D HDTF cells is a CSV-join + sklearn fit — same machinery as the
existing decompose.py. ~30 min.

The decision criterion: if P2D's HDTF R² is also 0.02-0.09 (matching
P8A and E2B HDTF), then the Fourier-aug intervention preserved the
substrate-conditional R² structure on HDTF — content-channel-real on
HDTF, IQ-bound on lockbox. If P2D's HDTF R² is materially higher than
P8A's HDTF R², the encoder became more IQ-bound on HDTF too (which
would change the FT-base argument).

### 4.2 Stage 2 packet — STAGE2_P8A_FT_FOURIER_BANDLIMITED

**Single-lever single-slot.** From P8A_REFERENCE_STEP5000 FT init
(NOT scratch+CE; not E2B family). Same Fourier band-amp aug as P2-D
slot D (`fourier_band_aug.py`, mid-band 5-6 preserve, high-band 12-13
randomize per `project_fourier_band_overlap_2026-05-06`). No GRL. No
sampler change.

**Why this is structurally novel**: "Fourier band-amp aug + FT-from-P8A"
has not been tested. P2-D used Fourier from CLIP-scratch (R13_P2_SCRATCH_FOURIER.yaml,
slot D). P8A's substrate-invariance was developed during R12G FT trajectory.
The combination — keep the trajectory, perturb the IQ axis at training
— is structurally different from both. AGENT_GUIDE Rule 1 satisfied.

**Cost**: ~2-3h training (P8A FT is a shorter trajectory than scratch).
~$25-40 Vertex spend. Single-lever per Rule 1.

**Close criterion**:
| metric | requirement |
|---|---|
| P8A HDTF macro_fake_recall (current 0.896) | does NOT regress > 5pp |
| P8A HDTF teams-transport mean (current 0.846) | does NOT regress > 5pp |
| Phase A `lockbox_fake_recall` (P8A current 0.387) | improves to ≥ 0.50 |
| Phase A `teams_real_dor_dev real_fpr` (P8A current 0.080) | does NOT exceed 0.15 |

The packet either preserves P8A's substrate-invariance + lifts lockbox
recall (a clear improvement), or it regresses HDTF teams-transport
(unequivocal stop). Either outcome is informative.

### 4.3 Branches based on Stage 2 outcome

**α — packet meets close criterion** (P8A HDTF preserved + lockbox lifts):
new FT-base candidate. Run full promotion contract scorecard. If
contract verdict promotes, this becomes the new production candidate
without touching E2B.

**β — packet preserves HDTF but fails lockbox lift** (Fourier aug doesn't
bite from P8A FT base): Stage 2-followup is IQ-balanced sampler from
P8A FT (Stage 2b in the program proposal §4.2). Different intervention
class, same FT-base.

**γ — packet regresses HDTF**: Fourier aug is incompatible with P8A's
substrate-invariance trajectory. **Stop the IQ-deconvolution program**;
pivot to data-side curation (Stage 4: drop celeb_real + youtube_real
smooth training reals per program §4.4) OR to fixing the suite-name-map
bug + measuring more candidates' HDTF behavior natively.

### 4.4 What I am NOT proposing

- **NOT a 2-3 packet sister-variant ablation**: $30 packet first.
  Pillar 3 readout decides next packet.
- **NOT GRL of any axis**: GRL targets the wrong substrate per §2.
  P15/P18 history — GRL on the wrong axis "doesn't bite". Save GRL for
  if/when we have direct evidence the chosen axis is the binding one.
- **NOT FT-from-D-step3000**: D step3000 is in the E2B family and inherits
  the Teams-transport collapse on HDTF.
- **NOT FT-from-E2B**: same family, same problem.
- **NOT a mixed bundle (Fourier + GRL + sampler)**: AGENT_GUIDE Rule 1
  + memory `project_face_scale_jitter_load_bearing.md` (P14 bundle 5.7×
  weaker than the isolated lever).

---

## 5. Risk inventory

(Honest. Each carries a CPU-cheap diagnostic that would update before GPU.)

| risk | how it manifests | diagnostic that catches it |
|---|---|---|
| P8A FT trajectory has weight delta saturated already | Stage 2 encoder doesn't shift; metrics ≈ P8A | P8A weight-delta probe vs CLIP base (small CPU) — confirms FT room exists |
| Fourier aug weakens P8A's substrate-invariance | Stage 2 HDTF teams-transport recall drops | the close criterion in §4.2 catches this directly |
| The "v2 viso ceiling" is structural, not a P8A FT problem | Stage 2 lockbox_fake_recall stays at P8A's 0.387 | β branch: pivot to sampler-class intervention |
| HDTF Phase C bug recurs | next Phase C scorecard fails on suite-name-map again | fix `score_teams_promotion_contract.py` suite-name map BEFORE the next launch |
| The user's deployment ≡ E2B framing matters operationally | even if P8A wins offline, E2B is in production | distinct decision; addressed by promotion contract verdict |

The bug-fix is independently valuable: every future Phase C launch hits
the same wall. That's a separate task that should be opened as a code
change.

---

## 6. The bigger frame

P8A's `0/1170 real_lockbox_dor_FPR` (memory
`project_p18_diagnostics_complete_2026-05-02.md`) and 0.83-0.85 HDTF
teams-transport recall (this doc) are **the same property**: P8A's
encoder learned a substrate-invariant content channel that survives both
the v2 lockbox's IQ shortcut AND the Teams pipeline applied to HDTF.
This property is not in E2B (its HDTF teams-transport recall is 0.06-0.41).
P2D's encoder shifted further from P8A on the same axis (HDTF teams-transport
0.06-0.27). The IQ-deconvolution program's Stage 2 was designed assuming
"shift the encoder away from IQ on the v2 substrate". The data now
says: **the encoder shift away from the v2 IQ axis is correlated with
loss of HDTF teams-transport substrate-invariance**.

Whether they are causally linked or coincidentally co-occurring is not
determined from this data. The §4.1 CPU diagnostic (P2D HDTF R²) starts
to test it: if P2D HDTF R² ≈ P8A HDTF R², the loss is despite preserved
HDTF IQ structure — the encoder didn't go more-IQ on HDTF; it just lost
the content channel there. If P2D HDTF R² > P8A HDTF R², the loss is
through becoming more-IQ — which is the program's original expectation
on the wrong substrate.

Either reading argues for the same Stage 2 packet (FT-from-P8A) but
informs how we read the result.

---

## 7. Open loops surfaced by this analysis

1. **`phase-c-hdtf-promotion-contract-failure` root cause known**:
   `score_teams_promotion_contract.py` suite-name map keyed on Phase A
   names. Fix is a small code change. Should be made before any
   future Phase C launch.
2. **The IQ_DECOMP_FACTS_2026-05-08.md §2.3 has P2D HDTF cells absent.**
   Now reconstructable from local artifacts; should be added.
3. **MODEL_GOALS.md promotion bar #4** ("no >5pp HDTF regression vs E2B")
   is structurally weak: E2B itself is 41pp behind P8A on HDTF
   macro_fake_recall. Either the bar should be vs P8A (not E2B), or
   the deployment-model framing in §Deployment needs to acknowledge
   that E2B is a deployment compromise, not an offline-eval bar.
4. **The "Where we stand" reframe**: Pillar 3's binding constraint is
   not "IQ shortcut on lockbox" but "Teams-transport substrate-invariance
   gap". This deserves a thread of its own.

---

## 8. Cross-references

- FACTS: `P2_D_HDTF_FACTS_2026-05-08.md` (this directory).
- Convergent dataset: see all 6 FACTS docs listed in `STATE.md`
  FRESH-AGENT GUIDANCE block.
- Memory entries:
  - `project_p2_d_hdtf_p8a_dominates_2026-05-08.md` (just added).
  - `project_pa_does_not_generalize_to_hdtf_2026-05-05.md`.
  - `project_per_layer_divergence_2026-05-06.md`.
  - `project_p8a_breakthrough.md`.
  - `project_deployment_is_e2b_2026-05-06.md`.
  - `project_face_scale_jitter_load_bearing.md`.
- Predecessor frames superseded:
  - `IQ_DECOMP_OPINIONS_2026-05-08.md` §4.3 close criterion — extended
    here with HDTF teams-transport recall as the binding criterion.
  - `DOR_ENCODER_AXIS_OPINIONS_2026-05-08.md` §7 Path C — argued
    against here as premature pre-(b)-result.
  - `P2_DEEPER_ANALYSIS_OPINIONS_2026-05-08.md` §6 slate — FT-base
    selection reversed (P8A, not D step3000 or E2B).
  - The IQ-deconvolution program proposal's implicit framing that Stage
    2a is the right packet — this doc argues the binding-constraint
    substrate is HDTF teams-transport (not v2 lockbox), so Stage 2a's
    hook + axis miss the load-bearing target.
