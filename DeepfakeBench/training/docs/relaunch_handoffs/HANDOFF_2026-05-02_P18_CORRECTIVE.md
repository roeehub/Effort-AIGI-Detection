# Handoff — P18 verdict needs verification before next packet

**Date generated**: 2026-05-02 00:50 CEST
**Branch**: `teams-relaunch-root-2026-04-17`
**Latest commit**: `30e044a` (the P18 verdict commit; the corrective probes invalidate part of that doc — see §"My errors and corrections" below).

> **DO NOT trust the prior agent's two-line verdict.** The first P18 verdict ("γ no bite") was based on a probe that couldn't have detected biting (wrong axis + wrong granularity). The follow-up corrective probe suggests "β partial bite" — but at n=9 dor_shkedi frames inside the eval substrate, the corrective probe is statistically fragile too. The honest answer is **inconclusive pending more diagnostics**. Read the data yourself; don't rely on my interpretations.

---

## What you should read, in order

To understand the full arc, read these in order. Total ~1 hour for a careful pass.

### Tier 1 — Strategic frame (~15 min)
1. **This document** — calibrated state + open questions + diagnostics-needed list.
2. [`docs/relaunch_handoffs/PHASE1_2_COMPLETE_STATUS_2026-05-01.md`](PHASE1_2_COMPLETE_STATUS_2026-05-01.md) — the comprehensive Phase 1+2 status from earlier in this session. Establishes the diagnostic findings that motivated P18.
3. [`docs/relaunch_handoffs/P17_FINAL_VERDICT_2026-05-01.md`](P17_FINAL_VERDICT_2026-05-01.md) — what falsified the layer-3 readout idea. Methodology baseline for the substrate-direction probes.
4. **`CLAUDE.md`** (root) — operational rules.
5. **`~/.claude/projects/.../memory/MEMORY.md`** — pointers to all `project_*` memory entries.

### Tier 2 — The Phase 1 diagnostics (the load-bearing findings)
6. [`PHASE1A_SUBSTRATE_DIRECTION_FINDING_2026-05-01.md`](PHASE1A_SUBSTRATE_DIRECTION_FINDING_2026-05-01.md) — the canonical Phase 1A finding. **The shortcut is intra-bucket-3 (within deeplive_teams: dor_shkedi-style identities vs other Teams identities).** This single sentence is the most important thing to internalize before reading anything else.
7. [`PHASE1_SYNTHESIS_2026-05-01.md`](PHASE1_SYNTHESIS_2026-05-01.md) — the three-probe synthesis (1A + Move 1 + Move 1.5 + 2B retag).
8. `analysis/intermediate_layer_probe_2026-04-30/substrate_classifier_direction_2026-05-01.py` — the canonical substrate-direction probe. Was run on FROZEN P8A backbone, on heads trained at L3 (P17 ckpts).
9. `analysis/eval_substrate_v3_retag_2026-05-01/REPORT.md` — production-tight crop retag at full lockbox scale (839 frames).
10. `analysis/move1_5_production_recrop_2026-05-01/REPORT.md` — original Move 1.5.

### Tier 3 — The P18 packet itself
11. **`docs/relaunch_handoffs/P18_VERDICT_2026-05-01.md`** — original verdict, now partially-invalidated. Read with the correction in §"My errors and corrections" below.
12. `experiments/phase2_round13/R13_P18_METHOD_DOMAIN_GRL.yaml` (treatment) and `R13_P18_NO_GRL_CONTROL.yaml` (control) — the launched yamls.
13. `analysis/method_class_audit_2026-05-01/proposed_method_domain_map.py` — the 12-bucket map's audit + design rationale. Bucket 3 = deeplive_teams (where dor_shkedi lives).
14. `data/sources/method_domain_map.py` — the canonical home of the 12-bucket map (committed as part of P18 prep at `10bc265`).
15. `analysis/p18_probe_2026-05-01/probe_p18_ckpt.py` — head-direction probe (in P8A's frozen feature space).
16. `analysis/p18_probe_2026-05-01/extract_p18_features.py` — first encoder-side probe (the WRONG-GRANULARITY one that produced the original "γ" verdict).
17. **`analysis/p18_probe_2026-05-01/corrective_probes.py` — corrective probes that test the actual shortcut axis. Read this carefully.**
18. **`analysis/p18_probe_2026-05-01/outputs/corrective_probes__rgt4kw2u__step4000_FINAL.json`** and **`...__xpbvc1e4__step4000_FINAL.json`** — the raw numbers underlying the β finding. **Don't read my summary; read the JSON.**

### Tier 4 — Earlier verdicts (chronological context)
- `P16_DATA_AXIS_VERDICT_2026-04-30.md`, `P13_DAY4_VERDICT_2026-04-29.md`, `P15_GRL_READINESS_NOTE_2026-04-29.md` — earlier packets in the chain.

---

## The cumulative experimental arc

In chronological order, with the loadbearing claim each packet either confirmed or refuted. Don't take my summaries at face value — verify against the cited docs.

### R12g / RLP series — pre-P8A architectural exploration
- Found the FT-only ceiling (~−0.10 anchor on RLP6_04). Reading was that head + attention-only SVD reach + frozen visual.proj/ln_post created a ceiling. Memory: `project_shortcut_is_upstream.md`.
- **Caveat (2026-04-26)**: `apply_svd_to_in_proj=true` had a silent zero-gradient bug pre-fix (commit `2feea58`). Some attribution from this era is wrong. Memory: `project_in_proj_svd_gradient_bug.md`.

### P8A breakthrough (2026-04-24, run `9lmvb5b4`)
- `unfreeze_final_proj=true` + `unfreeze_final_ln=true` + `apply_svd_to_mlp=true` broke the FT-only ceiling. anchor Δ −0.188 vs RLP6_04. Memory: `project_p8a_breakthrough.md`.
- **P8A is the strongest base model.** All subsequent FT-from-X packets use `9lmvb5b4 step5000` as their `gcs_base_checkpoint`.

### Camera-signature shortcut surfaces (multiple measurements 2026-04-24+)
- 2-camera test: same person, swap camera → score flips. Memory: `project_signature_shortcut_finding.md`.
- Source-bucket linear probe: P8A 10-class probe = 0.461 (4.6× chance) → P8A is NOT shortcut-clean.
- modern_v2 hygiene filter cuts headline lockbox FPR 4.6% → 0.71%. Memory: `project_lockbox_fpr_dominated_by_webcam_mode.md`.

### Face-size label leak (2026-04-27)
- Each fake method clusters at a tight face-size band. 47-frame × 5-tightness sweep flips 53% of frames. Memory: `project_face_size_label_leak.md`.

### Eval-vs-production crop tightness gap (2026-04-29 audit)
- Eval substrate has looser crops than production. Structurally upstream of camera-signature + face-size + webcam FPR. Memory: `project_eval_production_crop_tightness_gap.md`.

### P13 from-scratch + 3 anti-shortcut interventions (2026-04-28)
- anchor_aware loss + pipeline_randomization + face_scale_jitter@0.25.
- Moved Axis 3 directionally (P13_step18000 = 0.520 vs P8A 0.659) but cross-domain collapsed (viso 5.5%, deeplive 48.8% at τ=0.5). γ verdict.

### P14 FT-from-P8A + same bundle (2026-04-29)
- Run `xan4dfto`. value_composite=0.116. β verdict (cross-domain preserved via FT init, but bundle was net-negative).

### Sister ablation: face_scale_jitter@0.50 ALONE (`mclioexb`, 2026-04-29 night)
- Trainer-side `value_composite=0.661` — 5.7× the bundle's 0.116. **The mclioexb lesson**: bundle was net-negative against single-axis. Memory: `project_face_scale_jitter_load_bearing.md`.
- BUT: 2026-04-30 evening promotion-contract scorecard showed `mclioexb` does NOT promote. Even relaxed v3 contract — zero τ in 5549-pt grid qualifies. Memory: `project_mclioexb_does_not_promote_2026-04-30.md`.

### P15 GRL @ static λ=0.20 on capture-mode quality-domain head (`w5tky6ss`, 2026-04-30)
- domain_confusion_probe at the [CLS] manifold: P8A 0.9999 vs SLOT2_GRL 0.9994 — **GRL didn't bite the capture-mode axis at all**. Per `analysis/domain_confusion_probe_2026-04-30/outputs/probe_p8a_slot2_slot3/summary.json`.
- value_composite=0.516 (β). Memory: `project_p15_etc.md` (if not yet, in P15 retro doc).

### P16 data-axis tweaks (2026-04-30)
- 8 ckpts, none promote. P8A wins rank 1. Memory: `project_p16_data_axis_does_not_promote_2026-04-30.md`.

### P17 layer-3 readout (2026-05-01 morning)
- Trained heads on FROZEN P8A's L3 [CLS] features. Both ArcFace and LINEAR collapse to anti-correlated lockbox AUC by step ~1000. Recipe-independent + trajectory-driven failure. Memory: `project_p17_trained_head_destroys_substrate_invariance.md`.
- **The substrate-invariant signal IS in L3 features** (fresh LR achieves 0.95 lockbox transfer). The trainer refuses to use it.

### Phase 1A — substrate-classifier direction probe (2026-05-01 afternoon)
- This is the **load-bearing diagnostic** that motivated P18.
- The trained P17 heads modally align with `is_dor_shkedi` direction (cos +0.07→+0.14 monotonic across training). Orthogonal to capture-mode axes.
- **Crucial nuance often missed**: this was measured on heads trained with FROZEN P8A backbone, on L3 features. The "shortcut axis" identified is in P8A's frozen L3 feature space.
- Memory: `project_phase1a_method_cluster_axis_2026-05-01.md`.

### Move 1 (2026-05-01)
- Frozen-feature linear probe with grouped split. Bucket-LR AUC 0.916, but identity-only control AUC 0.987. AMBIGUOUS verdict — bucket gap is identity-confounded. Outputs: `analysis/move1_frozen_probe_2026-05-01/outputs/probe_results.json`.

### Move 1.5 + Phase 2B retag (2026-05-01)
- Production-tight crops (RFA=0.85) inflate FPR by +13.8 pp pooled, +40 pp on webcam reals. `PC_Generator__s15` reals 21% → 90% under tight crops.
- modern_v2 filter absorbs most of it (+2.14 pp).
- Memory: not yet a separate entry; see `eval_substrate_v3_retag_2026-05-01/REPORT.md`.

### Phase 3.7 P8A v3-substrate scorecard analog (2026-05-01)
- P8A on production-honest crops is NOT tier-0 under either 0.70 or 0.30 floor. ~5× FPR multiplier (0.036 asis → 0.179 prod) at recall=0.30. `teams_capture_cam_test_s33` prod-arm recall collapses to 0.0 at the contract-selected τ.
- **Sets the deployment-grade baseline floor for any subsequent packet.**

### Contract policy v3 (`974e033`, 2026-04-29; verified 2026-05-01)
- Default `target_fake_recall_min` flipped to 0.70. CI test guard. P8A verify scorecard (Vertex `5427016015462531072`) confirms v3 is engaged in image 1.3.240.

### P18 — 12-class method-conditional GRL with ramped λ (2026-05-01 evening)
**Treatment** (`xpbvc1e4`, Vertex `820959496569356288`):
- 12-bucket method-domain map (Phase 2C audit) so domain 2 = `deeplive_enhanced` and domain 3 = `deeplive_teams` (where dor_shkedi lives) get their own GRL targets.
- λ ramped 0 → 0.95 (per W&B `train/quality_grl_lambda` history).
- Best val AUC 0.9926 (step8000 cumulative-from-P8A = step3000 FT).
- Final periodic ckpt: step4000 FT, val AUC 0.9907, EER 0.0279.

**Control** (`rgt4kw2u`, Vertex `456167926752346112`):
- Identical recipe minus `use_quality_domain_head`.
- Best val AUC 0.9915 (step6500 = step1500 FT).
- Final periodic ckpt: step4000 FT, val AUC 0.9871, EER 0.0330.

**Both jobs SUCCEEDED 2026-05-01 evening.**

---

## My errors and corrections

I gave **two verdicts** on P18 in this session. Both should be treated with skepticism.

### Verdict 1 (initial, in `P18_VERDICT_2026-05-01.md` — partially invalid)

I ran `analysis/p18_probe_2026-05-01/extract_p18_features.py` which extracts FINAL [CLS] features from each arm's encoder and fits a 5-fold-CV multinomial LR predicting "12-class method-domain bucket". I reported **macro-OVR AUC: treatment 0.9954 vs control 0.9955 vs P8A 0.998** and concluded **γ (no bite)**.

**Why this was wrong:**

1. The eval substrate (800 frames sampled for analysis) has frames from ONLY 3 of 12 buckets:
   - bucket 2 (deeplive_enhanced): 59 frames
   - bucket 3 (deeplive_teams): 265 frames (ALL teams_capture content; `is_dor_shkedi` is a tiny subset of this)
   - bucket 11 (realpool_real): 476 frames
   The other 9 buckets have ZERO frames. So my "12-class probe" was actually a **3-class probe**.

2. The 3 buckets present are coarse content categories: "academic-source enhanced deeplive" vs "Teams-passthrough" vs "raw real". An untrained CLIP would score near-perfect on this distinction. The probe doesn't probe anything load-bearing.

3. **The Phase 1A shortcut is intra-bucket-3, not inter-bucket.** It's "dor_shkedi-style identities vs other Teams identities, both inside `deeplive_teams`". My probe averaged over inter-bucket separation — orthogonal to the question we cared about.

So the first verdict's evidence was incompetent for the question. I declared "no bite" based on a probe that couldn't have detected biting.

### Verdict 2 (provisional — written above this handoff in conversation, NOT the verdict doc)

I then ran `analysis/p18_probe_2026-05-01/corrective_probes.py` which actually tests the within-bucket-3 dor-vs-other axis. Results (the headline):

| Probe | Treatment | Control | Δ (T−C) |
|---|---:|---:|---:|
| Within-bucket-3 dor-vs-other LR @ FINAL [CLS] AUC | **0.9536** | **0.9965** | **−0.043** |
| Within-bucket-3 dor-vs-other LR @ L3 AUC | 0.8194 | 0.8194 | 0.000 |
| Fresh-LR L3 dev→lockbox AUC | 0.9404 | 0.9527 | −0.012 |
| Lockbox dor_shkedi real mean prob_fake (raw, no s-scaling) | 0.494 | 0.549 | −0.055 |
| Lockbox dor_shkedi real FPR @ τ=0.5 (raw) | **0.400** | **0.920** | **−0.520** |
| Lockbox other-Teams real FPR @ τ=0.5 (raw) | 0.600 | 0.800 | −0.200 |
| Lockbox dor_shkedi real FPR @ τ=0.92 | 0.000 | 0.000 | 0.000 |
| Lockbox dor_shkedi real FPR @ τ=0.974 | 0.000 | 0.000 | 0.000 |

I called this **β (partial bite)**. **DO NOT take this verdict at face value either.** Caveats:

1. **Within-bucket-3 LR has n_dor=9, n_other=256.** That's a very small positive class. The 0.9965 vs 0.9536 difference might be within bootstrap noise on n=9. Bootstrap CIs would tell us. Not yet computed.

2. **Lockbox dor_shkedi sample is n=25 reals.** A 92% → 40% FPR difference is qualitatively striking, but on n=25 frames the binomial CI on each FPR estimate is wide (~±20pp at 95%).

3. **The probe doesn't apply ArcFace `s` scaling.** At end of training, `arcface_s = 12`. The deployed scoring multiplies the cosine-margin by 12 before softmax. My `prob_fake` values are computed without that scaling, so they cluster around 0.5 even on confidently-fake frames. The relative T-vs-C delta might still be meaningful, but the absolute FPR@τ=0.5 has no deployment correspondence (deployment τ ≥ 0.92, where both arms read 0.0 here).

4. **At τ=0.92 and τ=0.974, both arms have FPR=0.0 on the dor sample.** So in the deployment-τ regime, we have ZERO information from this probe about which arm is better. We'd need (a) more dor frames, OR (b) the actual production-grade lockbox FPR distribution from a contract scorecard.

5. **The L3 AUC is identical (0.8194) in both arms.** The treatment-vs-control gap is concentrated at FINAL [CLS]. That's plausible (the GRL gradient hits the final pooler features) but it could also be an artifact of how the LR fits given different head normalization conventions. Need a sanity check probe.

6. **Other-Teams reals also showed reduced FPR@0.5 (80% → 60%, Δ=−0.20).** GRL's effect is *more* concentrated on dor (Δ=−0.52) than on other Teams (Δ=−0.20), which is consistent with selectivity. But it's also consistent with "GRL just generally pushes the encoder toward less-confident scoring on Teams content," which would be a less interesting non-selective effect.

### So what's the actual verdict?

**Honestly: inconclusive.** The corrective probe shows differences that are directionally consistent with GRL biting the targeted shortcut, but at sample sizes where I cannot rule out noise. Either of these could be true:

- **β actually-bit interpretation**: GRL did selectively reduce the dor_shkedi shortcut. At deployment τ this could translate to materially lower FPR on dor-style real captures. Would justify a P19 with stronger λ + production-tight crops.
- **noise interpretation**: the within-bucket dor-vs-other delta is sampling noise; the FPR delta @ τ=0.5 doesn't translate to deployment τ. Treatment and control are essentially equivalent in deployment, the macro-OVR result was correct in spirit even if measured wrong.
- **α-but-tiny interpretation**: GRL bit but only weakly; would need much stronger pressure to clear the shortcut.

The next agent should not proceed on either of my verdicts without running the diagnostics below.

---

## Diagnostics needed before declaring β/α/γ

Listed by cost. All CPU-only except (D).

### A. Bootstrap CIs on the within-bucket-3 LR AUC (~10 min CPU, $0)

Re-run `corrective_probes.py` for both arms but with bootstrap resampling (1000 iterations) on the within-bucket-3 LR. Report the 95% CI for treatment (n_dor=9) and control (n_dor=9). If the CIs overlap heavily, the within-bucket signal is noise. If treatment's upper CI is below control's lower CI, it's robust.

### B. Per-frame paired comparison on dor_shkedi lockbox (~10 min CPU, $0)

For each of the 25 dor_shkedi lockbox real frames: compute prob_fake under treatment, control, and P8A. Use a paired test (Wilcoxon signed-rank or sign test) on (treatment − control) margins. Paired tests are much more powerful than unpaired at small n. Output: signed-rank statistic + p-value.

### C. Apply the ArcFace s-scaling correctly (~5 min refactor, $0)

`corrective_probes.py:score_with_head` should multiply the cos-similarity logits by `arcface_s` (read from ckpt; should be 12 at end of training) BEFORE softmax. Then re-evaluate FPR@τ=0.92 and τ=0.974 — those are the deployment-relevant operating points where my current probe shows 0.0 in both arms (likely a result of the missing scaling).

### D. P18 promotion-contract scorecard (~$10-20 / ~3h Vertex)

The actually-deployment-relevant test. Re-uses `arena/launch_teams_promotion_contract.sh` with a P18-specific checkpoint map (treatment + control + P8A baseline for comparison). The v3 contract policy default (`target_fake_recall_min=0.70`) is now in image 1.3.240. Three checkpoints × ~3h = one Vertex job; ~$10-20.

This is the test that matters for the strategic question "should we promote P18 or its descendants?". My earlier argument for skipping it ("P8A doesn't pass the contract floor anyway") doesn't hold given the corrective-probe finding of selective FPR reduction on dor-real frames. If P18 lifts visomaster_enhanced_macro_dev recall by 5+ pp vs P8A, that's promotable. If it doesn't, it's a noise-level finding regardless of probe outcome.

### E. Within-bucket probe at intermediate layers (~30 min CPU, $0)

We have L3 (where AUC didn't budge) and final [CLS] (where AUC dropped from 0.9965→0.9536). The shortcut might be concentrated at one specific layer. Extending `corrective_probes.py` to also extract layers 6, 9, 11 would tell us where GRL's effect is concentrated and informs P19's design (multi-layer GRL? layer-targeted?).

### F. Re-extract P8A's same probes for clean baseline comparison (~30 min CPU, $0)

Run `corrective_probes.py` on P8A's `value_composite_effort_20260424_step5000_auc0.9926_eer0.0270.pth`. Currently the corrective probe outputs only have control + treatment numbers. Without the P8A baseline computed via the EXACT SAME pipeline, we can't say "control moved relative to P8A" with confidence.

### G. Sanity check: re-extract features through a different code path (~hour CPU, $0)

The `extract_l3_features` function in `corrective_probes.py` uses a forward hook on `transformer.resblocks[3]`. Verify by independent extraction (e.g., monkey-patching the resblock's forward or comparing to the cached L3 features in `analysis/_features_cache_2026-04-30/intermediate__P8A__layer03__n800.npz`) that we're capturing the right tensor at the right spot. If the L3 numbers differ between the corrective probe and the cached features for P8A, my corrective-probe pipeline has a bug.

---

## What we know with reasonable confidence

Things I'd defend even after the corrective probe:

- **P8A's [CLS] strongly discriminates the 12-bucket method-domain map** (Phase 3.5 smoke gate; macro-OVR 0.998 across the 3 active eval-substrate buckets).
- **Phase 1A's finding stands**: trained heads on FROZEN P8A's L3 align with `is_dor_shkedi` direction. This was measured directly; not affected by P18's outcome.
- **mclioexb does not promote** (`project_mclioexb_does_not_promote_2026-04-30.md`).
- **Production-tight crops materially inflate eval FPR** (Phase 2B retag, n=839; +13.8 pp pooled, +40 pp on webcam).
- **P8A on production-tight crops doesn't pass the v3 contract floor** (Phase 3.7).
- **Contract v3 (target_fake_recall_min=0.70 default) is engaged in image 1.3.240** (Vertex job `5427016015462531072` SUCCEEDED 2026-05-01 19:49).
- **P18 GRL was firing**: W&B `train/quality_grl_lambda` ramped 0.245 → 0.948; `train/loss/quality_domain_loss` was non-zero throughout. The GRL machinery did not silently fail.

## What we don't know

- Whether the within-bucket-3 corrective-probe finding (treatment 0.9536 vs control 0.9965 LR AUC) is statistically robust at n_dor=9.
- Whether the corrective-probe FPR@τ=0.5 difference (40% vs 92% on n=25 dor reals) translates to deployment τ.
- Whether P18 produces a different contract scorecard than P8A.
- Whether the L3 vs [CLS] localization of the GRL effect is a real layer-specific phenomenon or a probe artifact.
- Whether there's a methodological flaw in the corrective probe (the L3 hook, the s-scaling absence, the head.weight extraction).

---

## Strategic options for the next move

The right next move depends on the diagnostics. Don't pick one before running A-D above.

### If A and B confirm the within-bucket-3 effect is robust + D shows scorecard improvement
→ **P19: stronger GRL on the right axis.** Stack of: (a) higher λ_target (1.0 or 1.5), (b) longer training, (c) production-tight training crops (deterministic RFA=0.85 — needs new bbox-aware transform), (d) maybe multi-layer GRL heads if E says the effect is layer-localized. ~$120-200 / 1-2 days Vertex.

### If A or B shows the effect is noise OR D shows no scorecard improvement
→ **Move 4 paired same-identity contrastive** (Option II from `PHASE1_SYNTHESIS_2026-05-01.md`). 705 paired identities cached. ~2-3 days new code + ~$60-120 Vertex. By construction forbids identity-as-shortcut.

### If A is robust but D is null
→ Could be the deployment-grade τ regime is too sharp for the corrective-probe-visible effect to show. Try **P19 with calibration-aware loss** (focal / margin / label smoothing) + GRL. Or accept that GRL alone isn't sufficient and pivot to Move 4.

### If everything is null (A noise, B noise, D no improvement)
→ My second verdict was wrong too. P18 = γ truly. Pivot to Move 4 paired contrastive.

---

## Operational state summary

- **Jobs**: no active Vertex jobs.
- **Image**: `1.3.240` includes contract v3 (default `target_fake_recall_min=0.70`) AND the P18 method-conditional GRL infrastructure.
- **Working tree**: post-30e044a + e320633 + 3130334. The corrective-probe scripts are NEW (uncommitted) and the corrective verdict is currently only in chat (not in any committed doc until this handoff lands).
- **W&B**: `xpbvc1e4` (treatment) and `rgt4kw2u` (control) both finished. `phase2-experiments` project.
- **Local ckpts**: `/tmp/p18_ckpts/` has both finals (~900MB each).

## File inventory (this session's artifacts, all uncommitted unless noted)

```
COMMITTED in 30e044a (the partially-invalid verdict):
A  analysis/p18_probe_2026-05-01/probe_p18_ckpt.py
A  analysis/p18_probe_2026-05-01/extract_p18_features.py
A  analysis/p18_probe_2026-05-01/synthesize_verdict.py
A  analysis/p18_probe_2026-05-01/auto_probe_latest.sh
A  docs/relaunch_handoffs/P18_VERDICT_2026-05-01.md  ← partially-invalid; corrected
                                                        below in this handoff

UNCOMMITTED — corrective probe + this handoff:
A  analysis/p18_probe_2026-05-01/corrective_probes.py
A  analysis/p18_probe_2026-05-01/outputs/corrective_probes__rgt4kw2u__step4000_FINAL.json
A  analysis/p18_probe_2026-05-01/outputs/corrective_probes__xpbvc1e4__step4000_FINAL.json
A  docs/relaunch_handoffs/HANDOFF_2026-05-02_P18_CORRECTIVE.md  ← THIS DOCUMENT

Memory:
M  ~/.claude/.../memory/project_p18_method_grl_does_not_bite_2026-05-01.md  ← needs revision
A  ~/.claude/.../memory/project_p18_corrective_probe_partial_bite_2026-05-02.md  ← will be added
```

---

## Concrete next-agent checklist

1. Read this doc. Read it twice.
2. Read `analysis/p18_probe_2026-05-01/outputs/corrective_probes__*.json` directly. Don't take my interpretations.
3. Run **A. Bootstrap CIs** (~10 min). If overlapping, the within-bucket-3 finding is noise; act accordingly.
4. Run **B. Paired test** (~10 min). If non-significant, the FPR finding is noise; act accordingly.
5. Run **C. ArcFace s-scaling fix** (~5 min refactor + 5 min re-eval). If FPR@0.92 numbers materialize and treatment beats control there, argument for β strengthens.
6. Decide whether to spend D ($10-20 Vertex). Default: yes if A and B confirm robust.
7. **Don't launch P19 or Move 4 yet.** Wait for A, B, C, D results to inform the choice.

Be modest about claims. Surface raw numbers in any new doc, not just summaries. The pattern in this session — me declaring a verdict on the wrong probe, then having to walk it back — should not repeat.
