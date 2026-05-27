# Strategic Documentation Trail — Sub-Report 06

**Author:** Audit agent (Claude Opus 4.7), 2026-04-26
**Scope:** Trace the chronological strategic narrative of the Effort/Teams deepfake detector from January 2026 through 2026-04-26 (Packet-9 mid-flight). Surface where conclusions warrant skepticism. Annotate with FLAG markers throughout.
**Working dir:** `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/`
**Sources:** `docs/`, `docs/relaunch_handoffs/`, `docs/packet_retrospectives/`, `docs/research_2026-04-15*/`, git log on `main` and `teams-relaunch-root-2026-04-17`, user memory at `~/.claude/projects/.../memory/`.

---

## 0. TL;DR for the receiving agent

- **The codebase is on a single working branch (`teams-relaunch-root-2026-04-17`); `main` has not advanced since 2026-04-08.** The R13 relaunch arc — RLP1 through Packet-9 — exists entirely on the relaunch branch. There is no merge-back to `main` to interrogate.
- **The strategic narrative changed three times between 2026-04-17 and 2026-04-25.** It started as "fix VisoMaster bad data + add proper_data lane," pivoted to "fix the in-trainer composite + ArcFace margin + augmentation matrix," then re-pivoted to "the detector learned a camera/ISP processing signature and only backbone unfreeze can break the ceiling." Each pivot leaves load-bearing assumptions from the previous phase un-retracted in the docs.
- **Two strong claims warrant deep audit:**
  1. **"P8A broke the camera-signature ceiling"** — the R13_FULL_STORY 2026-04-25 doc and the user-memory `project_p8a_breakthrough.md` both assert this, but the same docs report a **−13.6 pp regression in fake recall** on the methods we deploy against, and the FPR breakthrough number (`lockbox_real_fpr 0.147%` vs RLP6_04's `0.441%`) was computed at τ ≈ 0.991 — a regime where the known **`project_contract_policy_bug.md`** drives τ to ~0.995 and crushes recall. **FLAG: the headline win has not been audited against this contract bug.** See §5.1.
  2. **"FT-from-RLP6_04 step 23500 is the right base"** — every R13 packet from RLP3 onward forks from this single checkpoint; no doc justifies *that specific step*. It appears to have been chosen because it had the highest `value_composite` at training-time selection — but `value_composite` is the same metric the team has explicitly demoted as not-deployment-grade. **FLAG: the foundational checkpoint of the program was selected by a metric the program has subsequently rejected.** See §6.
- **The "lockbox" holdout is treated as authoritative throughout** but I found no doc that audits its end-to-end identity-leakage cleanliness against the pre-RLP6 split-mode change. **FLAG: between RLP1 (`identity_split_mode: shuffle`) and RLP2+ (`hash_stable`), identities migrated between train/val/test without an explicit audit of whether any current "lockbox" identity ever appeared in pre-relaunch training.** See §5.4.

The remainder of this report is dense by necessity. Read §3 for the doc-by-doc index, §4 for the chronological narrative, §5 for the skepticism flags, §6 for decision provenance, and §7 for commit-driven story.

---

## 1. The current branch and what makes it "the relaunch root"

### 1.1. Branch topology

- **Current branch**: `teams-relaunch-root-2026-04-17`
- **Branched from**: commit `9ebdfc8` on `main` (2026-04-08, message "8 april") — but the actual point at which "relaunch work" began is `6c0d1fb` (2026-04-17, message "chore: snapshot training-speedup-pass1 before teams relaunch").
- **`main` has not advanced** since 2026-04-08 (`9ebdfc8`). All R13/RLP/P-packet work happened on the relaunch branch and never round-tripped back. There are 75 commits unique to the relaunch branch since divergence.
- **Other branches present** (now mostly dead): `refactor-training`, `training-speedup-pass1`, plus six worktree branches `wt-{a,b,c,d,e,f}-*-2026-04-17` that were merged into the relaunch root on 2026-04-17 / 2026-04-19.

### 1.2. What "relaunch root" means

Per `docs/TEAMS_EXPERIMENT_RELAUNCH_REPORT_2026-04-17.md` (April 17, 2026): on 2026-04-17 a **VisoMaster bad-data correction** landed. The team determined that 4,904 historical `visomaster_*` rows had been mis-trained as method-faithful supervision; the correction reduced the retained pool to:
- 480 sample IDs as `visomaster_hints`
- 202 sample IDs as `visomaster_hints_teams`
- 4,904 ignored, 3 delete-only

This was treated as a "policy reset" significant enough to fork a new working branch, then sub-fork six parallel worktrees (WT-A through WT-F) to land coordinated changes:

| Track | Owner concern | Merge date |
|---|---|---|
| WT-A | Data-truth / lane semantics | 2026-04-17 19:05 CEST |
| WT-E | Promotion-contract scorer | 2026-04-17 19:05 CEST |
| WT-D | Decision-system tooling | 2026-04-17 19:09 CEST |
| WT-C | GammaUp / nuisance augmentation truth | 2026-04-17 19:07 CEST |
| WT-F | Proper-data schema | 2026-04-17 19:06 CEST |
| WT-B | Weak-signal hint ablations | 2026-04-19 12:31 CEST |

**FLAG**: The "relaunch" framing implies a reset, but the codebase did NOT reset checkpoints, evaluation slices, or the trained weights. **Every R13 packet still forks from `R12_G_FP32` (RLP1–RLP3) or from `RLP6_04 step 23500` (RLP7+).** The "relaunch" was a *data-policy* and *evaluation-machinery* reset, not a model reset. Subsequent docs sometimes treat this as if model state was clean from 2026-04-17 onwards. It was not.

---

## 2. Reading order recommendation

If you only have time for the load-bearing docs, read in this order (all paths absolute):

1. `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/PACKET_9_MID_FLIGHT_HANDOFF_2026-04-26.md` — the most current strategic state, written today.
2. `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/R13_FULL_STORY_PRE_PACKET_9_2026-04-25.md` — full self-contained narrative as of yesterday.
3. `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/packet_retrospectives/README.md` — Phase-2 retrospective synthesis across 11 packet drafts.
4. `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/packet_retrospectives/threads/processing_signature_shortcut.md` — central technical thread.
5. `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_TRAINING_SOURCE_OF_TRUTH_2026-04-21.md` — navigation index as of mid-relaunch.
6. `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/TEAMS_EXPERIMENT_RELAUNCH_REPORT_2026-04-17.md` — strategic root for the relaunch.
7. `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md` — original April 6 strategic intent.

---

## 3. Doc-by-doc index

### 3.1. `docs/relaunch_handoffs/` (33 files)

#### 3.1.1. PACKET_9_MID_FLIGHT_HANDOFF_2026-04-26.md
- **Date**: 2026-04-26 (today, 14k bytes).
- **Strategic claim**: P8A's lockbox-FPR breakthrough is real (lockbox_real_fpr 0.147% vs RLP6_04's 0.441%), but **per-method fake recall regressed catastrophically** on visomaster_enhanced_macro and deeplive_enhanced. P9 is built on the assumption that "P8A is the right base, but trained too aggressively" — three hypotheses to test (magnitude, topology, data).
- **Recommended action**: Listen first to the user's new evidence; do NOT push the P9 plan as the directive. Halt running US relaunches if user evidence invalidates the P8A premise.
- **Reliance**: P8A scorecard, the failure-mode analysis at `analysis/p8a_fake_failure_analysis_2026-04-25/`, the running P9 jobs.
- **FLAG: This doc explicitly hedges ("Roee is less confident. He may be right.") that the P8A premise — and therefore all of P9 — may be wrong.** This is the most important meta-flag for the entire R13 program: the agent who wrote this handoff (which would be the prior session me) had already lost confidence in the P8A breakthrough by the time it was committed mid-Packet-9.
- **FLAG: The handoff acknowledges that 80% of P8A's per-method-recall regressions are NOT τ-recoverable** (separability loss, not threshold drift), but P9_01 (`backbone_lr_mult: 0.3`) is still the lead recommendation. If separability loss is the failure mode, softening LR may not recover separability — it may just produce a slower-trained P8A with the same regression. The handoff doc does not explicitly close this gap.

#### 3.1.2. R13_FULL_STORY_PRE_PACKET_9_2026-04-25.md
- **Date**: 2026-04-25 (29k bytes), written explicitly as a "second-opinion document for an external reviewer."
- **Strategic claim** (TL;DR §1): "P8A breaks the camera-shortcut: cuts false-flag rate on the anchor pool by ~half with no regression on training-time fake recall. Today (2026-04-25) we discovered P8A's broader fake-recall has regressed substantially on harder out-of-pool methods — a 13.6 pp aggregate drop at default τ. Failure-mode analysis indicates this is **separability loss**, not a recoverable threshold shift."
- **Reliance**: P8A run `9lmvb5b4`, the P8B comparison run `n8yk2hox`, the P7 run matrix, anchor-pool rescores, the calibration probe (WS-P1).
- **Recommendation**: One of (a) softened P8A, (b) return to RLP7_05 territory + augmentation pressure, (c) different intervention class. Open question: explicitly asks reviewer to challenge the "softened P8A is right" framing.
- **Was it followed?**: Partially. P9 launched (a) — the softened P8A path — but the handoff above (PACKET_9_MID_FLIGHT_HANDOFF) shows the team is now uncertain and waiting for new evidence before promoting any P9 candidate.
- **Notable §s for skepticism**:
  - **§5.4** "Anchor pool has temporal structure — pinned/escaped frames cluster in time within a single ~6-sec clip; the model has the capacity to escape, just doesn't on most frames." **FLAG: this is suggestive but n=1 anchor pool of 30 frames; the structural claim is fragile.** The doc has its own ⚠ flag.
  - **§7.2** Per-method recall table shows visomaster_enhanced_macro dropping from 57.3% → 35.6% (−21.6 pp). The doc treats RLP6_04's 57.3% as the baseline. **FLAG: the doc never asks whether 57.3% itself was inflated by the camera-signature shortcut on the matched-pipeline visomaster pool**; if RLP6_04 was getting visomaster recall partly because visomaster has a Teams-codec signature and RLP6_04 had learned that signature → "fake," then the P8A drop might be re-revealing the true unsupervised recall under the corrected representation, not a regression.
  - **§9 Open questions**: includes "Is the visomaster_enhanced_macro at ~57%-base recall acceptable to begin with?" — explicitly noting "the method may be a hard-truth limit of CLIP-DataComp-XL representational capacity."

#### 3.1.3. R13_RELAUNCH_STATE_OF_THE_TEAMS_DETECTOR_2026-04-25.md
- **Date**: 2026-04-25 (17k bytes).
- **Strategic claim**: P8A is the leader; first run to break the P7 anchor ceiling. Anchor Δ −0.188 vs best P7 −0.097.
- **Reliance**: anchor-pool rescores; W&B run summaries.
- **Recommendation**: Pre-Packet-9 checklist of 4 items (lockbox FPR, arena scorecard, OOD eval, sanity rescore RLP7_05 + RLP6_04). **§6 lists 4 Packet-9 candidates** in order: (1) RLP9_01 stack P8A unfreeze + RLP7_05 spatial+codec aug, (2) P8A + RLP7_07 CCT/lighting aug, (3) P8A schedule extended, (4) deeper unfreeze.
- **Was it followed?**: The actual P9 yamls launched a different slate (P9_01 backbone_lr_mult, P9_03 no MLP-SVD, P9_05 real_codec_uplift, P9_R reseed, P9_freeze) — the framing pivoted between 2026-04-25 morning and afternoon as the per-method regression evidence came in. **FLAG: the strategic recommendations in this doc were superseded within hours.** The fact that two docs from the same day make different recommendations (this doc → "stack P7 spatial+codec," R13_FULL_STORY → "soften P8A magnitude/topology/data") indicates the strategic theory was unstable on 2026-04-25.
- **Notable**:
  - §2.4 explicitly says: "Slices we DO NOT have for P8A yet: Lockbox FPR (production-policy)." The doc knew the headline anchor numbers but had not yet seen the lockbox readout. The handoff a day later (R13_FULL_STORY) shows the lockbox readout came in and reversed parts of the read.

#### 3.1.4. R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md
- **Date**: 2026-04-24 (6k bytes).
- **Strategic claim**: 2-camera controlled tests on 2026-04-24 confirm camera-signature shortcut (Dor laptop 0.02 vs Dor webcam 0.94, same scene). Cross-subject replication (Roee Mac vs Windows). NOT identity-specific. RLP6_04 (`value_composite=0.9006`, the "leader") false-flags real participants on different webcams.
- **Reliance**: 2 controlled tests, 6 pools × 30 frames each.
- **Recommendation**: Launch RLP7_04, _05, _02 first; skip lighting-aggressive _01 and combined _03. Per-camera calibration probe (WS-P1) closed only 30.1% of the cross-pool FPR gap → training-aug is dominant lever, calibration complementary.
- **What landed**: RLP7_01 through _08 launched; eventually superseded by Packet 8 (P8A/P8B).
- **FLAG: the "camera-signature shortcut" framing crystallized here on 2026-04-24, but the underlying observation (slot-07 dor_shkedi vs real_dor flip) was made during RLP5 in late 2026-04-23.** The framing took ~24h to consolidate. Earlier packet writeups in RLP3/RLP4/RLP5 use "lighting / stress / robustness" framing — those framings were never explicitly retracted.

#### 3.1.5. R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md
- **Date**: 2026-04-23 (~22:00 UTC revision).
- **Strategic claim**: Packet 6 is gate-alignment, not training-recipe. RLP5 E3 closed the headline VisoMaster-Teams gap (67% → 94%, 18% → 96% per-feature-space) but composite stuck at 0.7736 because **`external_youtube_avspeech_real`** (54% accuracy = 46% FPR) drives the worst_pool_fpr gate which forces τ to ~0.995. Packet-6 corrects: drop avspeech from gate, hint-clean teams_ood_fake, drop wma_failure_fake.
- **Reliance**: Reading of `trainer/trainer.py:161-216` showing `worst_pool_fpr` is **real-pool-only** by construction (an earlier framing had assumed it included fake pools — that framing was wrong).
- **Recommendation**: 8 RLP6 slots; canary slot 1 ("RLP6_01 gate_align_canary") expected to jump composite from 0.77 to ≥0.88.
- **Was it followed?**: Yes. RLP6_04 became the new leader at `value_composite=0.9006` — and is now the FT base for every subsequent packet. **It is also the checkpoint that immediately failed the 2-camera controlled test that triggered RLP7.**
- **FLAG: the same packet that produced the "best-ever" composite is the packet that triggered the realization the metric was lying.** RLP6_04 with `value_composite=0.9006` got crowned on 2026-04-23 and was demonstrated to be camera-signature-shortcut-bound on 2026-04-24 (a single day later). The doc's own §"Corrected gate-block analysis" notes that the prior packet's framing of `worst_pool_fpr` was wrong. **The implicit credibility of every "X improves the composite by Y" claim across RLP1-RLP5 is downstream of a metric that was understood to be deployment-misaligned by the time RLP6_04 was crowned.**

#### 3.1.6. R13_RELAUNCH_PACKET3_EXPERIMENT_PLAN_2026-04-21.md
- **Date**: 2026-04-21 (final v3, 59k bytes).
- **Strategic claim**: After packets 1 and 2 there are 9 instrumentation gaps (A1–A10) that distort interpretation. Land all of them before packet 3. Section §1.4 introduces **the "deployment value hierarchy"**: (1) min FPR across all real pools equally weighted; (2) max recall on Teams fakes both enhanced and unenhanced; (3) max recall non-Teams fakes; (4) the rest; (5) robustness as a multiplier.
- **Reliance**: Packet 1/2 W&B observations.
- **Recommendation**: 8 packet-3 experiments tied to A1-A10 instrumentation. **Introduces `value_composite` (A9) as a readout-only metric** — checkpoint selection stays on `best_ood_composite` for packet 3 (packet-2 comparability), to be re-evaluated for packet 4. This metric becomes the dominant story for RLP3 → RLP6.
- **Was it followed?**: Yes for the instrumentation. The "readout-only" caveat for `value_composite` was forgotten — by RLP3.5 / RLP4 it had become the de facto selection metric.
- **FLAG: §1.4's "deployment value hierarchy" lists 6 real pools to be equally weighted (df40, avspeech, vcd, teams_ood, proper_clean, proper_teams).** **Packet 6 then drops avspeech and vcd from the gate**, which is the opposite of "equally weighted." The April 21 promise of equal weighting was structurally violated by the April 23 gate-alignment correction. There is no doc retracting the April 21 hierarchy.

#### 3.1.7. R13_RELAUNCH_PACKET3_5_EXPERIMENT_PLAN_2026-04-22.md
- **Date**: 2026-04-22.
- **Strategic claim**: Packet 3.5 is a parallel wave, not successor. Four disabled training knobs to turn on cautiously: `arcface_m`, `stability_lambda`, `label_smoothing`, fix `anneal_steps`. Two metric definition changes globally: (a) `target_mean_fpr 0.02 → 0.03`, (b) `max_pool_fpr 0.04 → 0.05`, (c) `stability_jitter_stat: max → p95`. Retro-score packet 3 under new gates.
- **Reliance**: Packet 3 mid-run diagnostics, in-code inspection.
- **Recommendation**: 6 single-lever slots + 1 contingent stacked.
- **Was it followed?**: Yes. Result: ArcFace m=0.15 was the only single lever that beat retro-P3_02 by ≥+0.03. Other levers (stability_lambda, label_smoothing, family_rebalance) regressed. m=0.15 became the program default.
- **FLAG: this is the packet where the gate definition itself moved (mean_fpr 0.02→0.03, p95 instead of max).** The retro-score doc explicitly notes the new gate moves the absolute composite by ~0.10 with NO retraining. So the "ArcFace m=0.15 wins +0.04 vs P3_02" comparison is computed under apples-to-apples NEW gates — but the apples-to-apples comparison only exists *after* the gate definition was relaxed. The team chose a more permissive gate AND a winning lever in the same packet. **The published win could include some of the gate-relaxation effect masked as ArcFace effect, though the docs claim systematic biases cancel.**

#### 3.1.8. R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md and R13_RELAUNCH_PACKET4_PLANNING_HANDOFF_2026-04-22.md
- **Date**: 2026-04-22 evening.
- **Strategic claim**: Packet 4 = ArcFace margin scan {0.125, 0.175} on both control and spatial backbones; seed-B replication of m=0.15; Teams-family fake reweight 1.0 → 3.0.
- **Status**: All 8 jobs failed (yamls not in image — image-rebuild discipline lesson). Per packet retrospective, "RLP4 failed wholesale; forced rebuild discipline."
- **Was it followed?**: Failed. Triggered VERSION/image rebuild rules.
- **FLAG: the retrospective claims RLP4 forced "rebuild discipline" — but the packet retrospective README explicitly says "RLP4 m=0.15 lever was 'confirmed by carry-forward, not direct evidence.'"** So m=0.15 was promoted to program default partly because of an experiment (RLP4 m=0.15 seedB) that never completed.

#### 3.1.9. R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN_2026-04-21.md
- **Date**: 2026-04-21 (10k bytes).
- **Strategic claim**: 6 FT slots. Hints out, unenhanced proper_data is the main bet. Test moderate enhanced proper-data (not the full RLP1_05 jump). Two stability probes on the strongest data packet.
- **Reliance**: Packet 1 results.
- **Recommendation**: RLP2_02 = no_hints + unenhanced_proper main packet-2 bet.
- **Was it followed?**: Yes. RLP2_02 was top composite at 0.98937. But the conclusion "enhanced hurts" was later flagged as muddled because `val_holdout` for RLP2_02 contained zero enhanced-proper content.

#### 3.1.10. R13_RELAUNCH_PACKET1_*.md (5 files: experiment plan, evaluator handoff, monitoring handoff, live monitoring, results)
- **Dates**: 2026-04-19 through 2026-04-21.
- **Strategic claim**: Packet 1 is data-dominant: 5 data composition runs + 2 sidecars + 1 scratch hedge. Tests: (1) best honest data packet under relaunch contract, (2) hints helpfulness, (3) light WT-C sidecar utility, (4) FT vs scratch.
- **Result**: RLP1_01 (no hints, no proper) was leader at composite 0.98915. RLP1_04 was second, only new direction worth pursuing.
- **Reliance**: WT-A through WT-F merged work.
- **Was it followed?**: Yes — RLP2 built on RLP1's "no hints + unenhanced proper" finding.

#### 3.1.11. R13_RELAUNCH_VISOMASTER_TEAMS_POOL_DIAGNOSTIC_FINDINGS_2026-04-23.md
- **Date**: 2026-04-23.
- **Strategic claim**: A diagnostic looked alarming (~70% per-frame misclass on `proper_visomaster_teams_fake`) but was inflated by lane contamination — the GCS bucket commingles `proper_visomaster_teams` (in training) and `proper_visomaster_enhanced_teams` (excluded). True training-eligible per-frame accuracy ~67% vs `teams_ood_fake` 91.4% — a 22pp gap that motivated RLP5 E3.
- **Reliance**: Lane-aware rescoring of slot-05 checkpoint at 150 × 8 frames.
- **Recommendation**: E1 (bump weight 1.0→2.5), E2 (add enhanced-teams lane to training), E3 (combined).
- **Was it followed?**: Yes. RLP5 E3 was launched and achieved the headline 67%/18% → 94%/96% improvement.
- **FLAG: §6 E1 hypothesis explicitly says "if E1 moves it barely (<70%), the bottleneck is likely architectural / data-quality, not weight."** RLP5 E3 moved both lanes much further than expected. The packet retrospective for RLP5 then notes that slot-07 (the leader) turned out to be carrying the camera-signature shortcut, **so the E3 "breakthrough" is partly a shortcut breakthrough on an in-distribution-correlated pool**. This cycle — diagnostic → lever → improvement → discovery that improvement was illusion — repeats across packets.

#### 3.1.12. WT-A through WT-F handoffs (6 files dated 2026-04-17 to 2026-04-19)
- **Strategic claim**: Six parallel coordinated tracks. WT-A lane-semantics freeze, WT-B explicit hint-lane runtime, WT-C nuisance-augmentation truth, WT-D decision-system tooling, WT-E promotion-contract, WT-F proper-data schema.
- **Notable**: WT-F is the new proper-data schema (largest April 17-19 extension); WT-E added `arena/score_teams_promotion_contract.py` which is the authoritative promotion path.

#### 3.1.13. RELAUNCH_UPGRADE_REVIEW_PACKET_2026-04-19.md
- **Date**: 2026-04-19 (19k bytes).
- **Strategic claim**: April 6 plan is no longer "just a plan" — most implementation has landed. All WT-* tracks merged. Proper-data path on `combined_paired.proper_data`. Bottom line: "past the plumbing phase; experiment planning with the new data can begin now."
- **What it relied on**: Listed merge IDs and remote-smoke job IDs as proof.
- **Recommendation**: First combined `WTB3 + proper_data` startup smoke, finalize first experiment matrix, optionally run WT-E promotion contract.
- **Notable**: §"Post-Launch Critical Findings (2026-04-20)" added retroactively documents 3 issues: (a) proper-data counts in handoffs were stale, (b) shared holdout comparability drifted across packet arms (the `identity_split_mode: shuffle` → `hash_stable` change), (c) `unknown_fake` reporting bug.

#### 3.1.14. R13_RELAUNCH_TRAINING_SOURCE_OF_TRUTH_2026-04-21.md
- **Date**: 2026-04-21 (9k bytes).
- **Strategic claim**: "There is not one older file that is sufficient on its own anymore. Use this file as the top-level navigation index."
- **Notable**: §"What Is Settled" lists 9 items including "the strongest new data idea is not 'more hints' but 'explicit proper-data'" and "`stability_lambda` is not the preferred stability lever anymore." The latter foreshadows RLP3.5's negative result on stability_lambda.

#### 3.1.15. NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF_2026-04-19.md
- **Date**: 2026-04-19 (43k bytes).
- **Strategic claim**: New VisoMaster buckets are `proper_data`, NOT legacy `visomaster_hints` or a `combined_paired` stopgap. Teams snapshot ~30% short of complete.
- **Notable**: "the runtime loader still keeps anchor-index intersection as a defensive fallback if a future proper-data manifest admits ragged rows, but that is no longer the intended first-packet contract."

#### 3.1.16. WT_B_AND_NEW_DATA_READINESS_2026-04-19.md
- **Date**: 2026-04-19. WT-B runtime + new-data integration readiness check. Mostly operational.

#### 3.1.17. TASK_BOARD_2026-04-17.md
- **Date**: 2026-04-17. Worktree coordination board with claim/merge log.

#### 3.1.18. README.md (relaunch_handoffs)
- **Date**: 2026-04-21 (1k bytes). Pointer to source-of-truth doc.

### 3.2. `docs/` (root, 28 strategic docs, several R13-era and several legacy)

#### 3.2.1. TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md
- **Date**: 2026-04-06 (54k bytes).
- **Strategic claim**: R12 strong on holdout (AUC 0.9942) but production Teams deployment shows false positives + score instability. Domain gap. Shift project goal from generic holdout to Teams-deployment usefulness. Track A (resolver-driven Teams-enhanced), Track B (Teams characterization), Track C (target-domain scorecard), Track D (later TBA), Track E (later TBA).
- **Reliance**: Production W&B observations, DeepLive bucket sourcing.
- **Recommendation**: 8-step plan S0-S2 + Tracks A-D.
- **Notable progress in §0.1**: by April 11 explicit conclusion that Track A improved fake recall (89% vs 85%) but regressed real-Teams FPR (24% vs 20%; 73% vs 63% on lockbox).
- **FLAG: this doc records `teams_real_all_lockbox FPR 0.7252` for the candidate Track A and 0.6334 for R12_G — both well above 5% target.** The R13 docs from 2026-04-25 record `lockbox_real_fpr` numbers like 0.441% and 0.147% — orders of magnitude smaller. **The unit basis of the "lockbox" in 2026-04-06 is a different number from the "lockbox" in 2026-04-25.** It is not clear from the docs that the same evaluation surface is being used; it is plausible that the suite or the metric definition silently changed. This deserves explicit reconciliation.

#### 3.2.2. R12_POST_LAUNCH_PLAN.md
- **Date**: 2026-03-08 (18k bytes).
- **Strategic claim**: R12 achieves AUC=0.9942 on R9_D holdout but production Teams deploys with false positives. 5 structural fixes: (1) OOD-inclusive checkpointing (now `ood_composite`), (2) wire calibrator into production inference, (3) fix GRL quality-domain head label propagation (claimed broken in R6 — actually already fixed since R6 per update note), (4) Test-Time Augmentation (TTA), (5) expand Teams OOD holdout.
- **Reliance**: Production lighting analysis, R6_EXPERIMENT_REPORT.md (legacy).
- **Notable**: §"R12 Already Addresses" lists OneOf removal and ColorTemperatureShift as the augmentation fixes. This is where the `quality_domain_loss` was re-enabled — now in P9 it's being proposed at higher weight.
- **FLAG: §3 GRL fix update note says "On investigation, `quality_domain` label propagation was already fixed since R6"** — but other docs (R6_EXPERIMENT_REPORT, original §3 framing) had treated it as broken. **One of these is wrong; both have lived in the docs for ~2 months with no audit.**

#### 3.2.3. EFFORT_RANK_MISMATCH_INVESTIGATION_JAN_12_2026.md (Jan 12, 2026)
- **Date** (per filename): January 12, 2026.
- **Strategic claim**: Investigation of an Effort SVD-rank mismatch issue between B16 and L14 backbones.
- **Notable**: This is part of a January-era set including ARCFACE_SCALE_ABLATION (Jan 10), ARCFACE_SCALE_COLLAPSE_INVESTIGATION (Jan 11), B16_ARCFACE_INVESTIGATION_REPORT, B16_BREAKTHROUGH_RESULTS, B16_LAION_ARCFACE_INVESTIGATION, JAN_13_L14_vs_B16_LAION_Investigation, JAN_14_chat_about_B16_ArcFace.txt. **This is the era when the team chose `ViT-B-16-DataComp-XL` (LAION) as the backbone over CLIP-L14 (OpenAI).**
- **FLAG: The B16 backbone choice was made on the basis of ArcFace stability investigations in early January 2026.** The current rank=736 is inherited from this era. The 2026-04-25 R13_FULL_STORY explicitly flags this: "⚠ The choice of rank 736 is inherited from earlier R12g work; we have not re-validated it under the current data mix." **Decision provenance** for the architecture is in these January docs but has not been audited under the corrected R13 data mix.

#### 3.2.4. ARCFACE_SCALE_ABLATION_JAN_10_2026.md, ARCFACE_SCALE_COLLAPSE_INVESTIGATION_JAN_11_2026.md
- **Dates**: January 10-11, 2026.
- **Strategic claim**: ArcFace scale parameter `s` collapses on B16-LAION variants — collapse mode investigated.
- **Notable**: This is what motivated the eventual `s 6 → 12` schedule used in R13. The scale anneal completion bug discovered in RLP3.5 (`anneal_steps: 15000 > total_training_steps: 10000`) traces back to this era's reasoning being inherited without re-checking that the schedule actually completes within training steps.

#### 3.2.5. R9_R95_DATA_REPORT.md, EXPERIMENT_PLAN_27DEC2025.md, TASK_B_PLAN_29DEC2025.md, TRAINING_PIPELINE_FIXES_JAN_9_2026.md
- **Dates**: Dec 2025 - Jan 9, 2026.
- **Strategic claim**: R9-era data composition, training pipeline fixes. Pre-relaunch.
- **Notable**: The "checkpoint frenzy" era. Less directly load-bearing for current strategy; provides historical context.

#### 3.2.6. TEAMS_AUGMENTATION_SIDE_TRACK_HANDOFF_2026-04-06.md
- **Date**: 2026-04-06.
- **Strategic claim**: Old single-mode `TeamsCodecSimulation` should stay off the main line. Track B sidecar work.
- **Was it followed?**: Yes. Track B remained sidecar; never promoted to main line.

#### 3.2.7. SPATIAL_AUGMENTATION_PROPOSAL_2026-04-14.md
- **Date**: 2026-04-14.
- **Strategic claim**: Strongest concrete recommendation for crop / face-geometry robustness — proposes a spatial-jitter pipeline.
- **Was it followed?**: RLP7_04 / RLP7_05 implemented spatial augmentation; eventually overtaken by P8A's backbone unfreeze.

#### 3.2.8. POLICY_AWARE_SOURCE_REDESIGN_2026-04-17.md, VISOMASTER_BAD_DATA_POLICY_PLAN_IMPACT_2026-04-17.md, PROPER_DATA_CAPTURE_INVENTORY_AND_PROVENANCE_2026-04-17.md, PROPER_DATA_SCHEMA_AND_FUTURE_MANIFESTS_2026-04-17.md
- **Date**: 2026-04-17.
- **Strategic claim**: April 17 policy reset machinery — bad-data inventory, policy-aware loader redesign, proper-data schema.

#### 3.2.9. R13_WTB_WEAK_SIGNAL_DRAFT_RUNBOOK_2026-04-17.md, R13_WTB_WEAK_SIGNAL_RUNBOOK_2026-04-18.md
- **Dates**: 2026-04-17 / 2026-04-18.
- **Strategic claim**: WT-B weak-signal runbook — explicit hint-lane runtime.

#### 3.2.10. TEAMS_DECISION_SYSTEM_REINVESTIGATION_2026-04-17.md, WT_D_MINIMAL_BASELINE_PACKET_2026-04-17.md
- **Date**: 2026-04-17.
- **Strategic claim**: WT-D decision-system tooling for threshold sweeps, abstain bands, hysteresis.

#### 3.2.11. TRACK_C_PROMOTION_CONTRACT_RUNBOOK_2026-04-17.md, TRACK_C_SCORECARD_RUNBOOK_2026-04-07.md
- **Dates**: 2026-04-07 / 2026-04-17.
- **Strategic claim**: Promotion contract path. The April 17 version has the authoritative scorer.

#### 3.2.12. SMOKE_RUNNING_AGENT_HANDOFF_2026-04-06.md, TEAMS_AUGMENTATION_SIDE_TRACK_HANDOFF_2026-04-06.md, TRACK_A_HANDOFF_2026-04-06.md, TRACK_A_RUNTIME_RUNBOOK_2026-04-06.md, TRACK_B_MATCHED_PAIR_REPORT_2026-04-07.md
- **Dates**: 2026-04-06 / 2026-04-07.
- **Strategic claim**: Track A/B execution handoffs.

#### 3.2.13. TRAINING_SPEEDUP_PLAN.md, TRAINING_SPEEDUP_VALIDATION_RUNBOOK.md
- **Dates**: 2026-04-08 / 2026-04-09.
- **Strategic claim**: Training-speedup work. The branch `training-speedup-pass1` was the immediate predecessor to the relaunch branch.

#### 3.2.14. ENHANCED_TEAMS_GAP_REPORT.md
- **Date**: pre-March (from R6/R12 era based on filename context).
- **Strategic claim**: Enhanced Teams gap report — early framing of the deployment problem.

#### 3.2.15. LIGHTING_ROBUSTNESS_IMPL_LOG.md, LIGHTING_ROBUSTNESS_REPORT.md, "Simulating Varied Indoor Lighting...pdf"
- **Strategic claim**: Lighting robustness investigation. Underpins the "lighting is the dominant axis" framing that survived through RLP3 / RLP3.5 — and was then partially superseded by the camera-signature framing in RLP7.
- **FLAG: the lighting framing was never explicitly retracted; the camera-signature framing was added on top.** In R13_FULL_STORY the doc notes "our previously-flagged 'lighting weakness' is largely a synthetic-stress phenomenon, not a natural-distribution one" — but neither LIGHTING_ROBUSTNESS_REPORT nor any earlier doc has been amended to reflect this.

#### 3.2.16. checklist.md, EXPERIMENT_CONFIGURATION.md, REFACTORING_STATUS.md, JOB_TRAINING_README.md, KANIKO_CACHING_TROUBLESHOOTING.md, LOCAL_DEVELOPMENT.md, METHOD_WEIGHTING_GUIDE.md
- Operational/reference docs. Less directly strategic.

#### 3.2.17. NEW_DATA_DOC.md, DEEPLIVE_PIPELINE_DATA_BUCKET.md, VISOMASTER_ENHANCED_INTEGRATION.md
- Earlier data integration docs.

### 3.3. `docs/packet_retrospectives/` (synthesized 2026-04-24)

#### 3.3.1. README.md
- **Date**: 2026-04-24.
- **Purpose**: Phase-2 synthesis across 11 packet drafts. Treats itself as the executive index and reconciliation log.
- **Notable §s**:
  - "**Confirmed good**" — value_composite, proper-data unenhanced, ArcFace m=0.15, E3 dose-matched recipe, spatial backbone stacking, WT-A bookkeeping, WS-P0 preprocessing fix, WS-P2.b reducer, real-pool-only `worst_pool_fpr`.
  - "**Confirmed bad / abandoned**" — hints, "enhanced hurts" as clean conclusion, scratch training, avspeech in real-pool gate, visomaster_* bad rows, wma_failure_fake as gate driver, pre-fix retro-score absolute numbers, lockbox 90/90 on RLP6_04 as threshold-reachable.
  - "**Open mysteries**" — RLP5 slot-01 vs slot-07 shortcut probing asymmetry, residual 70% cross-pool FPR gap, no pre-fix leader ranking re-scored except RLP6_04, RLP3 retro-score determinism drift, FaceDancer blind spot, RLP6B never launched.
- **FLAG: the retrospective explicitly notes "**Pre-fix retro-score absolute numbers** (all packets through RLP6 except 6_04; see preprocessing_parity_bug) — suspect for absolute-value comparisons."** This means most of the headline numbers across R13 (RLP1 0.989 composites, RLP2/3 numbers, RLP5 67%/18% → 94%/96%) are **suspect for absolute comparison** because they were measured with `cv2.INTER_AREA` while the model was trained with `cv2.INTER_LINEAR` — a preprocessing parity bug that wasn't fixed until 2026-04-24 (commit `855871e`). **Only RLP6_04 has been rescored post-fix.**

#### 3.3.2. packets/RLP1.md through RLP7.md, RLP6B.md, WS_probes.md, WT_infrastructure.md (11 files)
- Detailed per-packet retrospectives.

#### 3.3.3. threads/processing_signature_shortcut.md (read in full above)
- **Notable**: Explicit timeline of when each piece of evidence pushed the thinking. Confirms slot-07 dor_shkedi vs real_dor flip on 2026-04-23 was the moment the shortcut went from "suspicion" to "fact."

#### 3.3.4. threads/promotion_contract_evolution.md
- **Notable**: Documents the move from `best_ood_composite` (RLP1/RLP2) → `value_composite` (RLP3 introduction, RLP3.5 re-gate, RLP6 real-pool-only correction) → "lockbox-anchored contract via WT-E machinery" (current).
- **FLAG: documents that `value_composite=0.9006` on RLP6_04 was contract-legal under the trainer but still failed the 2-camera controlled test.** This is the explicit moment where the team understood the in-trainer metric was insufficient.

#### 3.3.5. threads/preprocessing_parity_bug.md
- **Notable**: `cv2.INTER_AREA → cv2.INTER_LINEAR` fix landed 2026-04-24 (commit `855871e`). Only RLP6_04 has been re-scored under the fix. **Up to 7 packets have not been re-validated.**

#### 3.3.6. threads/calibration_vs_training_aug.md
- **Notable**: WS-P1 30.1% gap closure used pre-fix inputs. Verdict "training-aug dominant, calibration complementary" should be re-checked under post-fix inputs.

#### 3.3.7. threads/gate_alignment_story.md
- **Notable**: Real-pool-only `worst_pool_fpr` correction; permanent contract invariant.

### 3.4. `docs/research_2026-04-15*` (4 dated subdirs)

These are external research docs created 2026-04-15 by an agent doing parallel review. Three rounds + an empty round 4.
- **Round 1** (12 docs): repo-system-map, whitepaper-to-repo gap analysis, experiment-memory forensics, data-composition-and-curriculum-opportunities, target-domain-gap-and-teams-enhanced findings, literature reviews, synthesis.
- **Round 2** (10 docs): evaluation-contract-and-shortlist (this introduces the "calibrated low-FP promotion contract" framing that becomes WT-E), target-domain-data-truth (proves repo loader doesn't emit enhanced-through-Teams), sampler/curriculum-leverage, nuisance-invariance-augmentation-truth, decision-system-low-fp-analysis, literature-deepening, ranked-actions, next-round plan, next-agent prompt.
- **Round 3** (6 docs): remote-runtime-realboost-truth, shortlist-implications, internal-instability-mitigation summary, R13_E megaval backfill handoff.

These docs heavily influenced the WT-* worktree carve-up on 2026-04-17. They are particularly load-bearing on:
- The "calibrated low-FP promotion contract" framing.
- The structural finding that the loader can NOT emit `Teams real + Teams enhanced fake` (the central unresolved target condition).
- The augmentation runtime truth (GammaUp key mismatch documented here, fixed by WT-C).

### 3.5. `docs/superpowers/` (2026-04-22)

#### 3.5.1. specs/2026-04-22-compare-teams-pools-design.md and plans/2026-04-22-compare-teams-pools.md
- **Date**: 2026-04-22.
- **Purpose**: Design + plan for `analysis/compare_teams_pools.py`. The diagnostic that produced the lane-contamination-inflated alarming headline that motivated RLP5.

### 3.6. `docs/shortcut_learning_audit_2026-04-26/`

The current audit directory. Empty except for a sub_reports subdirectory.

---

## 4. Chronological narrative

### 4.1. Timeline of strategic theory shifts

| Date | Event | Strategic theory state |
|---|---|---|
| **Dec 2025 – Jan 2026** | R6, R8, R9, R12 base era. Effort detector trained on FaceForensics++ then DF40. Production Teams deployment shows FPR + score instability. | "Domain gap between training and Teams production." |
| **Jan 10–14, 2026** | ArcFace scale collapse investigation series. Choice of ViT-B-16-DataComp-XL (LAION) over L14. Rank 736 chosen. | "Backbone choice + ArcFace stability is the core issue." |
| **2026-03-08** | R12_POST_LAUNCH_PLAN.md proposes 5 structural fixes: OOD-inclusive checkpointing, calibrator wiring, GRL fix, TTA, expanded Teams OOD holdout. | "Production gap is plumbing — checkpointing metric, calibration, augmentation runtime." |
| **2026-04-06** | TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md. Track A/B/C/D split. Track A (enhanced-Teams data), Track B (Teams aug sidecar), Track C (target-domain scorecard). Project goal explicitly shifted from generic holdout to Teams-deployment usefulness. | "We need to optimize for Teams deployment. The R12 holdout AUC is misleading." |
| **2026-04-07 to 2026-04-11** | Track A baseline run completes. April 11 readout: Track A improves fake recall but regresses real-Teams FPR (24% vs 20%; 73% vs 63% on lockbox). | "Track A has the wrong shape — fixing fake recall comes at FPR cost." |
| **2026-04-15** | Three rounds of external research write up the gap analysis, evaluation-contract framing, etc. | "We need a calibrated low-FP promotion contract; current loader can't emit Teams-enhanced-fake; lighting/spatial robustness is open." |
| **2026-04-17** | **VisoMaster bad-data correction lands.** 4,904 rows ignored; 480+202 retained as hints. Six parallel worktrees (WT-A through WT-F) carve up the relaunch work. Branch `teams-relaunch-root-2026-04-17` is forked. | "Old VisoMaster data was bad supervision. We must re-frame everything under corrected lane semantics. Hints are auxiliary; proper_data is the new clean target-domain path." |
| **2026-04-19** | Proper-data runtime integration lands (WT-F + April 19 follow-up). Smoke succeeds. | "Implementation is done; experiment planning can begin." |
| **2026-04-19 to 2026-04-21** | **RLP1 (8 slots).** Hints out, unenhanced proper data is the only promising new signal. RLP1_01 (no hints, no proper) = leader. | "Proper_data unenhanced is real. Hints are dead. Scratch is dead. Sidecars are dead." |
| **2026-04-20 to 2026-04-21** | **RLP2 (6 slots).** Proper unenhanced confirmed as main bet. "Enhanced hurts" conclusion later flagged as muddled. | "Stick with unenhanced proper. The full-proper jump was too aggressive." |
| **2026-04-21 to 2026-04-22** | **RLP3 (8 slots).** A1-A10 instrumentation lands. New `value_composite` metric introduced (readout-only initially). Deployment value hierarchy declared in §1.4. | "We have the right machinery now. The metric we'll grade against is the deployment-aligned value_composite." |
| **2026-04-22** | **RLP3.5 (6 single-lever + 1 contingent).** ArcFace m, stability_lambda, label_smoothing tested. Gate definition relaxed (mean_fpr 0.02→0.03, p95 instead of max). Only m=0.15 wins. | "ArcFace m=0.15 is the only single training lever that works. Stability_lambda is dead (again)." |
| **2026-04-22 to 2026-04-23** | **RLP4 (8 slots).** All 8 jobs failed (yamls not in image). Forced rebuild discipline. m=0.15 + spatial backbone slate. | "Process failure; redo. Carry forward the m=0.15 conclusion." |
| **2026-04-23** | **RLP5 (8 slots).** E3 recipe (enhanced-teams lane + family weight 1.0→2.5) closes the 67%/18% per-feature-space gap on proper-VisoMaster-Teams to 94%/96%. Slot-07 becomes the leader at value_composite 0.7736. | "We've cracked the proper-VisoMaster-Teams training gap. This is the breakthrough." |
| **2026-04-23, ~01:00 UTC** | The diagnostic at `analysis/compare_teams_pools.py` first runs. **Slot-07 of RLP5 shows the dor_shkedi vs real_dor flip — same person, different identity_key in manifest, opposite scores. The processing-signature shortcut is recognized as a fact, not just a suspicion.** | "**The detector might be camera-pipeline-bound, not face-content-bound.** This contradicts the proper-VisoMaster-Teams 'breakthrough' interpretation we just had." |
| **2026-04-23 to 2026-04-24** | **RLP6 (8 slots).** Gate-alignment correction. avspeech and vcd dropped from real_pool gate (because they're not deployment-distribution). Slot RLP6_04 becomes leader at value_composite=0.9006. **2-camera controlled test on 2026-04-24 morning shows RLP6_04 false-flags real Dor on his Logitech webcam.** | "**The shortcut is a deployment blocker, not just a curiosity.** We need camera-signature-aware augmentation. RLP7 framework." |
| **2026-04-24** | **WS-P0 fix lands** (`cv2.INTER_AREA → cv2.INTER_LINEAR`). Train/inference preprocessing parity restored. RLP6_04 post-fix rescore: anchor δ moved only −0.008. Verdict `fully_structural__launch_full_subset`. WS-P1 calibration probe: 30.1% gap closure. | "The shortcut is a learned representation, not an inference-path artifact. Training-aug is the dominant lever." |
| **2026-04-24** | **RLP7 (5 yamls + later 3 more = 8 total).** Spatial-aug, codec-aug, CCT-aug variants. RLP7_05 (spatial+codec) best balanced (anchor Δ −0.089). All P7 hit anchor ceiling at ~−0.10. | "Augmentation alone hits a ceiling. The shortcut survives." |
| **2026-04-24 to 2026-04-25** | **Packet 8: P8A (unfreeze visual.proj + visual.ln_post + apply_svd_to_mlp). P8B (scratch on plain CLIP).** | "**Light FT was reach-limited.** Backbone unfreeze breaks the ceiling. The shortcut is in the data mix, not in the R12g/R13 weight chain. P8B confirms scratch is worse." |
| **2026-04-25** | **R13_FULL_STORY_PRE_PACKET_9 doc written.** Same morning, P8A lockbox scorecard runs and reveals **−13.6 pp aggregate fake-recall regression**. Failure-mode analysis shows **80% of regressions are NOT τ-recoverable (separability loss).** | "**P8A's anchor breakthrough comes at a real fake-recall cost on the methods we deploy against.** Soften P8A is the right next step?" |
| **2026-04-25 to 2026-04-26** | **Packet 9: 5 original variants (P9_01 backbone_lr_mult, P9_03 no_mlp_svd, P9_05 real_codec_uplift, P9_R reseed, P9_freeze) + 5 longshots.** | "Disentangle whether P8A's regression is from magnitude, topology, or data-asymmetry. Maybe forking from R12g (skipping RLP6_04) helps." |
| **2026-04-26** | **PACKET_9_MID_FLIGHT_HANDOFF.** Roee has new findings that may invalidate the entire P8A premise. The handoff explicitly tells the next agent to listen first, not push the P9 plan. **P10 anti-shortcut packet (commit `2c9778b`) lands later that day** — symmetric router + GRL slate. | "**Maybe even softened P8A is the wrong direction. Maybe we need an entirely different anti-shortcut intervention class — symmetric data routing, GRL.**" |

### 4.2. The framing evolution in one paragraph

**The detector started as a generic deepfake classifier (Effort + ArcFace + CLIP-L14), got migrated to ViT-B-16-DataComp-XL in January 2026 for ArcFace stability, hit the production-Teams domain gap (R12 era), pivoted to "Teams target-domain optimization" on April 6, 2026, then on April 17 the team realized the VisoMaster training data they had been treating as supervision was actually mostly bad — triggering a full "relaunch" that sub-divided into six parallel worktrees. The relaunch produced 11 packet drafts (RLP1 through Packet-9) over 8 days, with the strategic framing pivoting at least three times: from "data composition is the lever" (RLP1-RLP2) to "in-trainer metric definition is the lever" (RLP3-RLP3.5) to "augmentation matrix is the lever" (RLP5 E3, RLP7) to "backbone unfreeze is the lever" (P8A) to "we don't actually know what the lever is" (Packet-9 mid-flight handoff). Throughout, the central unresolved technical issue is whether the model has learned a camera/ISP processing signature instead of face-artifact features — a question first raised in the 2026-04-15 round-2 research docs, made undeniable by the 2026-04-23 dor_shkedi flip and 2026-04-24 2-camera test, and still not actually solved as of 2026-04-26.**

### 4.3. When the team realized there was a shortcut problem

The shortcut problem was discovered in stages:

| Stage | Date | Evidence | Framing |
|---|---|---|---|
| Suspicion | 2026-04-15 | Round-2 research doc identifies the loader cannot emit Teams-enhanced-fake; structural training gap. | "Domain gap" |
| First hint | RLP3 slot-07 | A 0.40-scale lighting probe on the slot-07 lineage starts looking like a symptom of something larger. | "Lighting weakness" |
| Hardened observation | 2026-04-23 ~01:00 UTC | RLP5 slot-07 dor_shkedi vs real_dor manifest A/B: same face, different `identity_key`, opposite outputs. | "Pipeline signature, not face content" — `project_signature_shortcut_finding.md` filed. |
| Deployment-blocker promotion | 2026-04-24 morning | 2-camera controlled test — Dor laptop 0.02 vs Dor webcam 0.94, Roee Mac VBG 0.90 vs Roee Windows 0.01. Camera/OS swap flips score. | "Camera/ISP signature is a deployment blocker" — `R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md` written. |
| Structural confirmation | 2026-04-24 evening | WS-P0 INTER_AREA → INTER_LINEAR fix. RLP6_04 post-fix rescore moves anchor only −0.008. Preprocessing ruled out. Verdict `fully_structural__launch_full_subset`. | "It's in the learned representation" |
| Reach diagnosis | 2026-04-25 | P8A unfreeze breaks ceiling (anchor Δ −0.188); P8B scratch worse (Δ +0.066). | "Light FT was reach-limited; backbone unfreeze breaks it" |
| Cost discovery | 2026-04-25 afternoon | P8A lockbox readout: −13.6 pp aggregate fake-recall regression. Failure-mode analysis: 80% NOT τ-recoverable. | "Backbone unfreeze trades anchor for separability" |
| Re-questioning | 2026-04-26 | PACKET_9_MID_FLIGHT_HANDOFF: "Roee has new findings that may invalidate the entire direction of Packet-9." | "Maybe we don't actually know what the right intervention is" |

---

## 5. Conclusions to challenge — FLAG list for skeptical audit

### 5.1. "P8A broke the camera-signature ceiling"

- **Where claimed**: `R13_FULL_STORY_PRE_PACKET_9_2026-04-25.md` §1, `R13_RELAUNCH_STATE_OF_THE_TEAMS_DETECTOR_2026-04-25.md` §1, `~/.claude/projects/.../memory/project_p8a_breakthrough.md`, `PACKET_9_MID_FLIGHT_HANDOFF_2026-04-26.md` "Packet 8: unfreezing the backbone (the breakthrough)" section.
- **The headline number**: lockbox_real_fpr 0.147% (P8A) vs 0.441% (RLP6_04). Anchor Δ −0.188 vs best P7 −0.097.
- **FLAG 1: The lockbox FPR was computed at τ ≈ 0.991 per the PACKET_9 handoff.** The known `project_contract_policy_bug.md` is "FPR-minimization no-budget drives τ to ~0.995, crushes fake recall, scorecard still 'passes' because no recall floor enforced." **τ = 0.991 is in the regime where this bug operates.** The doc explicitly says "Always check `selected_threshold` and the τ value before trusting a scorecard." Has anyone audited whether the 0.147% FPR is a real win vs whether it's the contract-policy bug masking the regression as a τ-inflation artifact? The PACKET_9_MID_FLIGHT handoff hints at this (§"When the scorecard lands" item 4: "Critical: check `selected_threshold` for each row. If τ is ≥0.99, the contract-policy bug may be inflating real_fpr metrics."). **It does not actually verify the P8A 0.147% number under this lens.**
- **FLAG 2: The "anchor breakthrough" is measured on 30 frames of one webcam clip (Dor's Logitech).** The doc's own §5.4 says the model has "the capacity to escape, just doesn't on most frames." 13/30 anchor frames still flip > 0.9. P8A's anchor is bimodal — the mean moved but the high-confidence false-flag rate (43%) is still catastrophic for any τ that catches reasonable fake recall. **Per `project_signature_shortcut_finding.md`: "The 90/90 goal on lockbox is currently NOT threshold-reachable on slot-07 at any τ (at τ catching 90% fakes, real FPR is 63.6%). This is a representation problem, not a calibration problem."** The same shape may still hold for P8A.
- **FLAG 3: P8A's headline FPR win on Teams real dev slices (12.11% vs RLP6_04 15.92% on `teams_real_all_dev`) is at default τ.** In R13_FULL_STORY §7.4 the same authors note: "P8A's regression analysis identified per-layer LR control as the missing knob" — i.e., they themselves identified that even the "win" might be conservative-shift-driven (the model became calibrated more "real-leaning" because the codec-aggressive aug stresses the real pass hard). **The "win" and the "regression" may be two faces of the same calibration shift.**
- **Bottom line**: This needs a hard audit. The breakthrough may be partly real, partly contract-policy-bug, partly calibration-shift. **It should not be inherited into Packet-10 reasoning without explicit re-validation.**

### 5.2. "Quality is the shortcut" — but is it the only one?

- **Where claimed**: `project_signature_shortcut_finding.md`, `processing_signature_shortcut.md` thread, `R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md`.
- **Evidence**: dct_hf_ratio (max_sep 3.46), wb_rb_ratio (2.32), mean_cb (2.23), mean_cr (1.97). Two codec-aligned (DCT-HF, bits-per-pixel), two chroma/lighting.
- **FLAG 4: The fingerprint analysis identifies the strongest pool-separating signals — but it does NOT prove these signals are the ONLY shortcuts.** Same-person-different-camera is the test that flagged the issue, but the model could be using multiple signatures at once (camera ISP + identity-codec correlation + manipulation-pipeline correlation). The "Slot-07 dor_shkedi vs real_dor" test specifically held identity fixed but pipeline different — it does NOT rule out that a second test (different identity, same pipeline) might also flip the model. **Has anyone run that complementary test?** The docs do not show evidence of it.
- **FLAG 5: `processing_signature_shortcut.md` §"Open loops" item: "FaceDancer blind spot remains open. No RLP has attacked FaceDancer pipelines directly; whether the shortcut generalizes across manipulation families is untested."** The team has explicitly noted this gap and explicitly chosen not to investigate (per `R13_RELAUNCH_PACKET4_PLANNING_HANDOFF_2026-04-22.md` §"User preferences": "AVOID: targeted facedancer augmentation").
- **Bottom line**: "Camera/ISP signature" may be the dominant shortcut, but the audit has not ruled out other concurrent shortcuts. Subsequent interventions (P8A, P9) target ONE failure mode and may be regressing on others.

### 5.3. "FT-from-RLP6_04 step 23500 is the right base"

- **Where**: Every R13 packet from RLP7 onwards. `experiments/phase2_round13/R13_RLP7_*.yaml`, `R13_RLP8_01_unfreeze_clip_codec.yaml`, all P9 yamls.
- **The choice**: `gs://training-job-outputs/phase2r13_experiments/h2pdu6i5/value_composite_effort_20260424_step23500_auc0.9942_eer0.0169.pth`.
- **FLAG 6: This checkpoint was selected by `value_composite=0.9006` AT THE TIME** — using the metric the team has subsequently demoted as "not deployment-grade." See `promotion_contract_evolution.md` thread: "RLP6 slot-04 shortcut exposure. A checkpoint with `value_composite=0.9006` — contract-legal — false-flagged real Teams participants on different cameras."
- **FLAG 7: RLP7_08 explicitly tested forking from RLP6_04 step 4500 (early) instead of step 23500 (late), to rule out "late-RLP6_04 consolidation."** Same anchor ceiling. Per `project_shortcut_is_upstream.md`: "Light FT from any RLP6_04 checkpoint (step 4500 or step 23500) hits the same ~0.84-0.89 anchor ceiling." So step 23500 is no worse than 4500 — but **neither doc explicitly tests whether a non-RLP6_04 R12g checkpoint, or a fresh checkpoint trained without the avspeech-aware gate alignment that RLP6_04 used, would have a different ceiling.** P8B was the only "fresh" experiment and it failed (Δ +0.066), but P8B was scratch on plain CLIP — a much more drastic departure than "FT from a different post-relaunch checkpoint."
- **FLAG 8: There is no doc that justifies WHY RLP6_04 step 23500 was chosen over step 23000 or step 24000 or RLP6_03 etc.** It appears to have been the highest-`value_composite` checkpoint in the RLP6 packet at the moment of selection. RLP7's pre-launch gate work (`analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json`) ratified the choice but was conducted on the candidate already chosen.
- **Bottom line**: The foundational checkpoint was selected by a now-discredited metric, has not been compared to peer checkpoints from the same packet under the post-fix rescore, and may be carrying packet-6-era gate-alignment quirks that the subsequent packets inherit silently.

### 5.4. "Lockbox is held out"

- **Where**: Treated as authoritative throughout R13. `arena/score_teams_promotion_contract.py`, `project_promotion_contract.md`.
- **The claim**: "Lockbox: held-out partition the promotion-contract uses for *final* FPR + recall calibration. Never used for selection during training."
- **FLAG 9: Between RLP1 (`identity_split_mode: shuffle`) and RLP2+ (`hash_stable`), identities re-partitioned across train/val/test/lockbox boundaries.** Per `R13_RELAUNCH_PACKET1_RESULTS_AND_PLANNING_CONTEXT_2026-04-21.md` "Split-mode caveat" pinned at top: "Packet 1 used `identity_split_mode: shuffle` (legacy default). Packet 2 and packet 3 use `identity_split_mode: hash_stable`. The switch re-partitions identities across train / val / test." **This is acknowledged for train/val/test fairness, but the docs do not explicitly audit whether any current "lockbox" identity ever appeared in pre-relaunch (R8/R9/R12) training.**
- **FLAG 10: The "lockbox" used by the WT-E promotion contract is defined relative to the suite manifest at `arena/target_domain_suites.teams_promotion_contract_2026-04-17.yaml`.** The suite snapshot was frozen on 2026-04-17 — but the *trained models* (R12_G_FP32 baseline, RLP6_04, P8A) inherit weights from training that may have seen R12-era data with different identity-split conventions. **No doc audits the end-to-end identity-leakage cleanliness from R12-era training data through the current "lockbox" partition.**
- **FLAG 11: The lockbox numbers in TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md are 0.6334–0.7252 (63.3–72.5% FPR).** The lockbox numbers in PACKET_9_MID_FLIGHT_HANDOFF_2026-04-26.md are 0.147–0.441% (0.15–0.44% FPR). **These differ by three orders of magnitude.** Either the underlying data slice changed dramatically (per April-19 reweighting / proper-data integration), or the metric definition changed, or both. **No reconciliation doc exists.** The R12 lockbox 73% FPR was the original reason the program existed; the R13 0.44% FPR is now treated as the bar to beat. These cannot both be true on the same evaluation surface.

### 5.5. "AUC ≥ 0.99 means strong model"

- **Where**: RLP1/RLP2/RLP3 era headline numbers. `R13_RLP1_01 best composite 0.98915` etc.
- **FLAG 12: The packet retrospective explicitly says these absolute numbers are SUSPECT due to the preprocessing-parity bug** that wasn't fixed until 2026-04-24. Per the retrospective: "Pre-fix retro-score absolute numbers (all packets through RLP6 except 6_04) — suspect for absolute-value comparisons. Intra-packet rankings often survive; cross-packet numeric comparisons do not."
- **FLAG 13: AUC measured on holdout has been the explicitly-demoted metric since 2026-04-06.** TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md §0 "Status Snapshot" says R12 "achieves AUC=0.9942 on the holdout validation set yet production deployment on Microsoft Teams video calls suffers from false positives." **Yet most R13 packet writeups continue to celebrate AUC ≥ 0.99 as if it indicates anything.** R13_FULL_STORY §2 "Training-time metrics (step 5000): AUC 0.9926, EER 0.0270 ... fake recall is intact." The very next-day analysis showed −13.6 pp regression on per-method recall.
- **Bottom line**: AUC numbers are inertial in the docs but have been formally demoted twice (April 6 deployment-target shift, April 21 deployment value hierarchy). They should not be trusted as evidence of model strength.

### 5.6. Other meta-FLAGs

- **FLAG 14**: The `value_composite` metric was promised as "readout-only for selection in packet 3" — to be re-evaluated for packet 4. Within the same packet 3.5 it became the de facto selection metric (RLP3.5 §4 "ranks slots 01-06 by Δvalue_composite"). This bait-and-switch is documented in `promotion_contract_evolution.md` thread but never explicitly retracted in the planning docs.
- **FLAG 15**: The `train_sweep.py` config-propagation bug (commit `872502c`, fixed 2026-04-22) means **all R13 launches before image 1.3.193 may have silently used legacy gate values even when the YAML specified new ones.** This affects RLP3 and RLP3.5 retro-scores. The retrospective README mentions this but does not enumerate which results are affected.
- **FLAG 16**: The 2026-04-25 R13_FULL_STORY ⚠ flag "we have not run a 'what's the maximum achievable visomaster_macro recall under this architecture, regardless of trade?' probe" is a major epistemic gap. The team may be chasing recall on a method that the architecture cannot represent at any operating point — and treating its 57% recall as the "baseline to maintain."
- **FLAG 17**: The "deployment value hierarchy" at `R13_RELAUNCH_PACKET3_EXPERIMENT_PLAN_2026-04-21.md` §1.4 lists "robustness" as item 5 — a multiplier on the others. By Packet-7 the team is anchored on a single Dor-on-webcam clip. **Per `feedback_small_sample_guidance.md`: "Don't bog down in per-pool details with small samples (<200 frames)."** The anchor pool is 30 frames. The entire RLP7-RLP8 framing is partly anchored on a 30-frame clip. There is tension between the user-memory rule and the operational reality.

---

## 6. Decision provenance audit

### 6.1. Why RLP6_04 step 23500 as the FT base?

**Question**: When was this checkpoint chosen as THE base for everything that follows? Was it an informed pick or a default?

**Search result**: Looking at:
- `experiments/phase2_round13/R13_RLP7_*.yaml` files all reference this checkpoint
- `R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md` does NOT explicitly justify the choice
- `R13_RELAUNCH_STATE_OF_THE_TEAMS_DETECTOR_2026-04-25.md` §3 says "Run: `h2pdu6i5` (Vertex), checkpoint step 23500, AUC 0.9942, EER 0.0169. value_composite 0.9006 — promoted leader entering Packet-7. This is the baseline against which every P7 / P8 anchor Δ is computed."
- `R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md` discusses how RLP6_04 was constructed but does not justify step 23500 specifically
- `analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json` (referenced by the WS-P0 verdict) ratifies the choice post-fix

**Provenance**: **The decision was implicit.** RLP6_04 was the highest-`value_composite` checkpoint at training-time selection in the RLP6 packet. Step 23500 is the step at which the trainer's `best_value_composite` checkpoint state was written. **There is no doc that compares step 23500 to other steps within the same RLP6_04 run, nor to other RLP6_* slots' checkpoints, under the post-fix rescore.** The choice looks defaulted — "use the value_composite leader" — using a metric the team has subsequently demoted.

**FLAG 18**: This is a foundational choice for ~5 packets of work and merits explicit reconciliation. The team should rescore RLP6_01 through RLP6_08 leaders under the post-fix preprocessing pipeline and compare. They have only done this for RLP6_04 and only post-hoc to ratify the existing choice.

### 6.2. Why ViT-B-16-DataComp-XL backbone?

**Search result**: Decision is in the January 2026 era docs (ARCFACE_SCALE_*, B16_*, JAN_13_L14_vs_B16_LAION_Investigation, JAN_14_chat_about_B16_ArcFace.txt). The choice was made on the basis of ArcFace stability investigations.

**FLAG 19**: The R13_FULL_STORY explicitly flags "the choice of rank 736 is inherited from earlier R12g work; we have not re-validated it under the current data mix." **The B16 backbone is similarly inherited** and the relaunch never explicitly re-validated it. CLIP-L14 might or might not behave differently under the current data mix.

### 6.3. Why ArcFace m=0.15?

**Search result**: RLP3.5 single-lever experiments showed m=0.15 won by +0.0415 vs retro-P3_02 baseline; m=0.10 was tied at +0.0408; m=0.20 blocked on max_fpr. The pick was (a) one of two effectively-tied winners at m=0.10/0.15, (b) below the m=0.20 collapse, (c) carried forward through RLP4 (which never completed) and into RLP5/RLP6.

**FLAG 20**: m=0.15 won under a relaxed gate (mean_fpr 0.02→0.03, max_fpr 0.04→0.05) that was changed in the same packet. The win-margin would have been smaller under the strict gate. This is acknowledged in the RLP3 retro-score doc but not explicitly resolved.

### 6.4. Why the avspeech-and-vcd drop from the real-pool gate?

**Search result**: `R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md` §"Corrected gate-block analysis" says these pools are "not deployment-distribution" — they are not Teams-passed reals. Earlier framing (in RLP3 §1.4 deployment value hierarchy) listed avspeech and vcd as among the 6 equally-weighted real pools.

**FLAG 21**: The April 21 deployment value hierarchy explicitly listed avspeech and vcd as load-bearing real pools. The April 23 packet drops them entirely from the gate. **The change is justified as deployment-alignment, but it is also the direct mechanism by which `value_composite` jumps from 0.7736 to 0.9006 with no retraining.** The team made a metric-relaxation move and a model-promotion move in the same step, then declared the model "the new leader." The retrospective `gate_alignment_story.md` thread documents this but does not retract the win.

---

## 7. Recent commit narrative (since 2026-01-01, on relaunch branch)

Sorted reverse-chronological from `git log teams-relaunch-root-2026-04-17 --since=2026-01-01`:

```
2026-04-26  d609e39  Bump VERSION to 1.3.214 (wandb artifact name fix image)
2026-04-26  f366368  Fix wandb artifact name >128 chars on long log_prefix scorecards
2026-04-26  2c9778b  Add P10 anti-shortcut packet: symmetric router + GRL slate         ← TODAY
2026-04-25  cbfac08  Bump VERSION to 1.3.211 (Packet-9 +5 longshot pack image)
2026-04-25  94d80d4  Add Packet-9 +5 longshot pack: stack + R12G fork + schedule probe
2026-04-25  c9997c0  Bump VERSION to 1.3.210 (Packet-9 launch image)
2026-04-25  009ae58  Bump VERSION to 1.3.208 (Packet-9 prod image)
2026-04-25  926af00  Add P9_freeze yaml — magnitude/topology disentangle ablation
2026-04-25  5af88c7  Add P9_R yaml — P8A replication under fresh seeds
2026-04-25  c0a91f7  Add P9_05 yaml — real-side codec uplift
2026-04-25  d509188  Add P9_03 yaml — no MLP-SVD topology
2026-04-25  dc0478f  Add P9_01 yaml — softened P8A (backbone_lr_mult: 0.3)
2026-04-25  ebce585  Add real_codec_uplift flag to family-aware aug router
2026-04-25  f766ccd  Add R13 full-story pre-Packet-9 doc for second-opinion review     ← strategic doc
2026-04-25  0f2f342  Add optional backbone_lr_mult to choose_optimizer
2026-04-25  3426bdf  Add P8A step 2500 (ood_composite) anchor rescore
2026-04-25  e326b81  Extend state-of-detector with W&B per-method + OOD stress slices
2026-04-25  8a9ae90  Bump VERSION to 1.3.207 (P8A scorecard image)
2026-04-25  856ea08  Add P8A promotion-contract scorecard inputs
2026-04-25  1573c59  Add Packet-7/8 anchor rescores + overnight readout + state-of-detector doc
2026-04-25  d5be7ce  Bump VERSION to 1.3.206 (Packet-8 image)
2026-04-25  d047da9  Draft RLP8_01 (P8A unfreeze CLIP) + RLP8_02 (P8B scratch plain CLIP) yamls   ← P8A starts
2026-04-24  7c3f6d0  Bump VERSION to 1.3.205 (RLP7_08 image)
2026-04-24  ed607b9  Draft RLP7_08 (codec-aggressive from RLP6_04 step 4500 base)
2026-04-24  edd6f50  Draft RLP7_06 (Teams CCT-only) + RLP7_07 (spatial + codec + CCT) yamls
2026-04-24  0f797a1  Track Packet-7 RLP7_01/02/03 yamls before image rebuild
2026-04-24  5798b3d  Refresh HANDOFF.md: Packet-7 pre-launch gate state
2026-04-24  584118e  Add Packet-7 camera-signature handoff: current state + launch candidates    ← shortcut framing
2026-04-24  c4affa1  Draft RLP7_04 (Teams spatial-only) + RLP7_05 (spatial + moderate codec) yamls
2026-04-24  deac44e  Add calibration probe (WS-P1) + per-identity reducer (WS-P2.b)
2026-04-24  855871e  Fix train/inference preprocessing drift: INTER_AREA → INTER_LINEAR    ← WS-P0 fix
2026-04-22  c7ac78c  Prepare handoff for next agent — flag open W&B anomaly investigation
2026-04-22  bcf4c61  Relocate slot 06 from europe-west4 (stale PENDING) to us-central1
2026-04-22  2cdb804  Refresh HANDOFF.md: post-relaunch state with train_sweep.py bug + fix recorded
2026-04-22  872502c  Fix value_composite config propagation in train_sweep.py + bump to 1.3.193   ← train_sweep bug
2026-04-22  5262e8a  Refresh HANDOFF.md with complete packet 3.5 state ...
2026-04-22  179c9e6  Update HANDOFF.md for packet 3.5 launch (13 runs in flight across 2 regions)
2026-04-22  e1cd23e  Bump VERSION to 1.3.192
2026-04-22  477b00b  Land R13 Packet 3.5 — value_composite config plumbing + cautious ArcFace wave
2026-04-22  1014b9b  Bump VERSION to 1.3.191
2026-04-22  c6f1034  Land A1-A10 OOD instrumentation and R13 Packet 3 yamls    ← RLP3 instrumentation lands
2026-04-19  77facfc  Add proper-data runtime and relaunch review packet
2026-04-19  7cde78c  Add fast WT-B startup smoke and cache priming
2026-04-19  f9303eb  Land WT-B runtime and launcher smoke readiness
2026-04-17  5fe095d  Document WT-D minimal baseline packet
2026-04-17  cf8e114  Document WT-C test order
2026-04-17  7c01467  Merge wt-c-nuisance-2026-04-17 follow-up
... [WT-* worktree commits 2026-04-17] ...
2026-04-17  6c0d1fb  chore: snapshot training-speedup-pass1 before teams relaunch     ← RELAUNCH ROOT
2026-04-08  48cef77  On branch training-speedup-pass1 -> pass2
2026-04-08  9ef022b  On branch training-speedup-pass1 -> pass1
2026-04-08  9ebdfc8  8 april
2026-04-06  6d89bd4  6 april
2026-03-19  94cc392  cleanup: finalize refactor file moves and housekeeping
2026-03-19  eed5f69  docs: add experiment reports and analysis
2026-03-19  ba36067  tests: add coverage for trainer, config, and pipelines
2026-03-19  43bdb22  scripts: add launch and run pipelines
2026-03-19  d856618  infra: docker + cloudbuild setup
2026-03-19  38558ee  refactor: modular trainer + dataset + config system
2026-02-10  d1e4337  first code dump for branch I think, app3 + stuff wip
2026-02-10  37e2378  first code dump for branch I think, app3 + stuff wip
2026-02-10  9834000  first code dump for branch I think, app3 + stuff
```

### 7.1. The story the commits tell

- **Feb 10, 2026**: First code dumps. The repo started on this branch as a "code dump."
- **March 19**: Modular refactor in 6 commits (refactor, infra, scripts, tests, docs, cleanup). This is the post-refactor state of the codebase.
- **April 6 + 8**: Dated commits "6 april" / "8 april" — minimal commit messages. These are the days the Teams target-domain plan landed.
- **April 8**: `training-speedup-pass1`/`pass2` branch work — performance optimization predecessors to the relaunch.
- **April 17**: 19 commits in one day. Six worktrees claim/merge. The relaunch infrastructure lands.
- **April 19**: Proper-data runtime lands (`77facfc`) — the "implementation" milestone for the WT-F path.
- **April 22**: 8 commits. Packet 3 instrumentation (`c6f1034`), Packet 3.5 ArcFace wave (`477b00b`), train_sweep bug fix (`872502c`). Three things landed in one day.
- **April 24**: 9 commits. Preprocessing-parity fix (`855871e` — the WS-P0 retro-affecting fix), calibration probe (`deac44e`), camera-signature handoff (`584118e`), RLP7 yamls (`c4affa1`, `0f797a1`, `edd6f50`, `ed607b9`).
- **April 25**: 18 commits. P8A/P8B yamls (`d047da9`), P8A scorecard inputs (`856ea08`), R13 full-story doc (`f766ccd`), 5 P9 yamls (`dc0478f` through `926af00`), `backbone_lr_mult` optimizer support (`0f2f342`), real_codec_uplift flag (`ebce585`), longshot pack (`94d80d4`).
- **April 26**: 3 commits. P10 anti-shortcut packet (`2c9778b`) — the latest reactive pivot.

**FLAG 22**: The pace from 2026-04-17 to 2026-04-26 is roughly 10 commits/day with 3 strategic-framing pivots in the same window. **This is sprint pace, not steady-state pace.** The risk is high that decisions made under sprint conditions are inheriting unaudited context from days earlier.

**FLAG 23**: The commit `2c9778b` from 2026-04-26 (today) is "Add P10 anti-shortcut packet: symmetric router + GRL slate" — the team is adding a NEW packet (P10) while Packet-9 is still mid-flight. The narrative is now: P9 may be wrong → P10 is being designed in parallel as a pivot option. **This means by the time this audit completes, the strategic framing will have moved AGAIN.**

---

## 8. Summary of FLAGs for the receiving agent

I have used 23 FLAGs in this report. The most load-bearing for an independent skeptical audit are:

1. **FLAG 1 (P8A FPR)**: The 0.147% lockbox FPR may be τ-inflation from the `project_contract_policy_bug.md`, not a real win. Audit the τ values explicitly.
2. **FLAG 4 (single shortcut)**: "Camera/ISP signature" may not be the only shortcut. The complementary test (different identity, same pipeline) has not been documented.
3. **FLAG 6 (FT base)**: RLP6_04 step 23500 was selected by the metric the team subsequently demoted. The choice has not been re-validated.
4. **FLAG 9-11 (lockbox audit)**: The lockbox numbers across 2026-04-06 (73% FPR) and 2026-04-26 (0.44% FPR) are not reconciled. Identity-leakage cleanliness from R12-era training has not been audited end-to-end.
5. **FLAG 13 (AUC inertia)**: AUC ≥ 0.99 has been formally demoted twice but continues to be celebrated in headline numbers.
6. **FLAG 15 (train_sweep bug)**: All R13 launches before image 1.3.193 may have silently used legacy gate values. Affected results have not been enumerated.
7. **FLAG 16 (visomaster ceiling)**: The team may be chasing recall on a method the architecture cannot represent.
8. **FLAG 18 (RLP6_04 peer comparison)**: RLP6_01 through RLP6_08 leaders have not been rescored under post-fix preprocessing pipeline.
9. **FLAG 21 (gate-relaxation move)**: avspeech/vcd drop from gate is the direct mechanism by which `value_composite` jumped 0.7736 → 0.9006. Conflated with model improvement.
10. **FLAG 23 (sprint pivot)**: Three strategic framings in 9 days. P10 already being designed before P9 readout. Decisions are inheriting unaudited context.

---

## 9. Files referenced (absolute paths)

### docs/relaunch_handoffs/
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/PACKET_9_MID_FLIGHT_HANDOFF_2026-04-26.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/R13_FULL_STORY_PRE_PACKET_9_2026-04-25.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_STATE_OF_THE_TEAMS_DETECTOR_2026-04-25.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_VISOMASTER_TEAMS_POOL_DIAGNOSTIC_FINDINGS_2026-04-23.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_PACKET4_PLANNING_HANDOFF_2026-04-22.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_5_EXPERIMENT_PLAN_2026-04-22.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_RETRO_SCORE_RESULTS_2026-04-22.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_EXPERIMENT_PLAN_2026-04-21.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN_2026-04-21.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_RESULTS_AND_PLANNING_CONTEXT_2026-04-21.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_TRAINING_SOURCE_OF_TRUTH_2026-04-21.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_LIVE_MONITORING_2026-04-20.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_MONITORING_HANDOFF_2026-04-20.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/RELAUNCH_UPGRADE_REVIEW_PACKET_2026-04-19.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF_2026-04-19.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/WT_B_AND_NEW_DATA_READINESS_2026-04-19.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_EXPERIMENT_PLAN_2026-04-19.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_EVALUATOR_HANDOFF_2026-04-19.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/WT-B_2026-04-18.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/TASK_BOARD_2026-04-17.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/WT-A_2026-04-17.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/WT-A_policy_truth_artifact_2026-04-17.json`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/WT-B_2026-04-17.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/WT-C_2026-04-17.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/WT-D_2026-04-17.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/WT-E_2026-04-17.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/WT-F_2026-04-17.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/relaunch_handoffs/README.md`

### docs/ (root)
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/TEAMS_EXPERIMENT_RELAUNCH_REPORT_2026-04-17.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/R12_POST_LAUNCH_PLAN.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/POLICY_AWARE_SOURCE_REDESIGN_2026-04-17.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/VISOMASTER_BAD_DATA_POLICY_PLAN_IMPACT_2026-04-17.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/SPATIAL_AUGMENTATION_PROPOSAL_2026-04-14.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/TEAMS_DECISION_SYSTEM_REINVESTIGATION_2026-04-17.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/WT_D_MINIMAL_BASELINE_PACKET_2026-04-17.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/PROPER_DATA_CAPTURE_INVENTORY_AND_PROVENANCE_2026-04-17.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/PROPER_DATA_SCHEMA_AND_FUTURE_MANIFESTS_2026-04-17.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/R13_WTB_WEAK_SIGNAL_DRAFT_RUNBOOK_2026-04-17.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/R13_WTB_WEAK_SIGNAL_RUNBOOK_2026-04-18.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/SMOKE_RUNNING_AGENT_HANDOFF_2026-04-06.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/TEAMS_AUGMENTATION_SIDE_TRACK_HANDOFF_2026-04-06.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/TRACK_A_HANDOFF_2026-04-06.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/TRACK_A_RUNTIME_RUNBOOK_2026-04-06.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/TRACK_B_MATCHED_PAIR_REPORT_2026-04-07.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/TRACK_C_PROMOTION_CONTRACT_RUNBOOK_2026-04-17.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/TRACK_C_SCORECARD_RUNBOOK_2026-04-07.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/TRAINING_SPEEDUP_PLAN.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/TRAINING_SPEEDUP_VALIDATION_RUNBOOK.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/ARCFACE_SCALE_ABLATION_JAN_10_2026.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/ARCFACE_SCALE_COLLAPSE_INVESTIGATION_JAN_11_2026.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/B16_ARCFACE_INVESTIGATION_REPORT.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/B16_BREAKTHROUGH_RESULTS.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/B16_LAION_ARCFACE_INVESTIGATION.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/B16_viability_checklist.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/EFFORT_RANK_MISMATCH_INVESTIGATION_JAN_12_2026.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/JAN_13_L14_vs_B16_LAION_Investigation.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/JAN_14_chat_about_B16_ArcFace.txt`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/Investigating ArcFace Instability on ViT-B_16 (LAION) vs ViT-L_14 (OpenAI).txt`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/EXPERIMENT_PLAN_27DEC2025.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/EXPERIMENT_CONFIGURATION.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/TASK_B_PLAN_29DEC2025.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/TRAINING_PIPELINE_FIXES_JAN_9_2026.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/R9_R95_DATA_REPORT.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/REFACTORING_STATUS.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/ENHANCED_TEAMS_GAP_REPORT.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/LIGHTING_ROBUSTNESS_IMPL_LOG.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/LIGHTING_ROBUSTNESS_REPORT.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/Simulating Varied Indoor Lighting for Robust Real‑vs‑Fake Face Classification.pdf`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/DEEPLIVE_PIPELINE_DATA_BUCKET.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/VISOMASTER_ENHANCED_INTEGRATION.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/NEW_DATA_DOC.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/JOB_TRAINING_README.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/KANIKO_CACHING_TROUBLESHOOTING.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/LOCAL_DEVELOPMENT.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/METHOD_WEIGHTING_GUIDE.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/checklist.md`

### docs/packet_retrospectives/
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/packet_retrospectives/README.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/packet_retrospectives/packet_template.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/packet_retrospectives/packets/RLP1.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/packet_retrospectives/packets/RLP2.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/packet_retrospectives/packets/RLP3.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/packet_retrospectives/packets/RLP3_5.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/packet_retrospectives/packets/RLP4.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/packet_retrospectives/packets/RLP5.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/packet_retrospectives/packets/RLP6.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/packet_retrospectives/packets/RLP6B.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/packet_retrospectives/packets/RLP7.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/packet_retrospectives/packets/WS_probes.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/packet_retrospectives/packets/WT_infrastructure.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/packet_retrospectives/threads/processing_signature_shortcut.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/packet_retrospectives/threads/promotion_contract_evolution.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/packet_retrospectives/threads/preprocessing_parity_bug.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/packet_retrospectives/threads/calibration_vs_training_aug.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/packet_retrospectives/threads/gate_alignment_story.md`

### docs/research_2026-04-15{,_round2,_round3,_round4}/
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/research_2026-04-15/00_research_index.md` and 9 numbered companions
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/research_2026-04-15_round2/00_research_index.md` and 9 numbered companions
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/research_2026-04-15_round3/00_research_index.md` and 5 numbered companions
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/research_2026-04-15_round4/` (empty)

### docs/superpowers/
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/superpowers/specs/2026-04-22-compare-teams-pools-design.md`
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/superpowers/plans/2026-04-22-compare-teams-pools.md`

### Memory (referenced repeatedly)
- `/Users/roeedar/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/MEMORY.md`
- `/Users/roeedar/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/project_p8a_breakthrough.md`
- `/Users/roeedar/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/project_signature_shortcut_finding.md`
- `/Users/roeedar/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/project_shortcut_is_upstream.md`
- `/Users/roeedar/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/project_promotion_contract.md`
- `/Users/roeedar/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/project_contract_policy_bug.md`
- `/Users/roeedar/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/project_success_criteria.md`
- `/Users/roeedar/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/feedback_promotion_contract_launch.md`

### Top-level
- `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/README.md` (the original Effort paper README, not a strategic doc but provides architectural context)

---

*End of sub-report 06.*
