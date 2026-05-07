# BUILD_SCAFFOLD — Chronological Map for Slice Agents

> **🟢 Build complete: 2026-04-29.** All 8 slices (0 through 7) are `done`. The wiki is ready for working-agent use under the `AGENTS.md` read/update protocol. The temporal-edge orienting surface for any future agent is the rolling `STATE.md` (the build-time snapshot now lives at `archive/STATE_2026-04-29.md`).
>
> **Purpose** (build-time): Build-time chronological enumeration of source material, partitioned into 7 substantive slices + Slice 0 scaffold. Each slice agent picked its slice, read the listed sources, and authored thread/timeline content per `AGENTS.md`.
>
> **No synthesis in this file.** Listing only — actual synthesis lives in `threads/*.md`, `packets/*.md`, and the rolling `STATE.md` (build-time snapshot at `archive/STATE_2026-04-29.md`).
>
> **Slice 0 author**: Claude Opus 4.7 (1M context), 2026-04-29.
>
> **Per the plan (`/Users/roeedar/.claude/plans/eager-weaving-canyon.md`)**: the user reviews this scaffold after Slice 0 finishes, before Slice 1 begins. Adjust slice boundaries here if needed before launching Slice 1. *(Build is now complete; this note is preserved as historical anchor.)*

---

## Slice partition rationale (briefly)

The plan suggested 5 slices spanning 04-17 → 04-29. Source-density inspection shows three high-density stretches that warrant their own slices:

1. **04-17** alone has 30 commits (WT-A through WT-F worktrees converging).
2. **04-25 → 04-26** has 23 commits (P8A scorecard work, Packet-9 setup, in_proj-SVD fix, Phase A/C overnight slate).
3. **04-27 → 04-28** has 5 commits but ~10 separate session entries in the master plan log (the contract-policy whack-a-mole, P11 launch, P12 dud, periodic_saves bug, wandb-flattening bug).

Splitting these prevents any one slice agent from drowning. The result is **7 slices**. Slice 1 covers 04-17 alone because the foundation density is exceptional. Slice 2 runs 04-18 → 04-21 because RLP1 monitoring spans days and RLP2 chains directly off RLP1 results.

Each slice ends at a natural plot point (a packet boundary, a handoff arrival, or a verdict).

---

## Slice 1 — Foundation: April 17 worktree convergence

**Range**: 2026-04-17 (start of branch) → 2026-04-17 EOD
**Status**: done
**Done by**: 2026-04-29, Slice 1 agent (Opus 4.7)

### Source files

**Handoffs (read first):**
- `docs/relaunch_handoffs/TASK_BOARD_2026-04-17.md` — coordination board
- `docs/relaunch_handoffs/WT-A_2026-04-17.md` — data policy / lane semantics freeze
- `docs/relaunch_handoffs/WT-A_policy_truth_artifact_2026-04-17.json` — JSON artifact from WT-A
- `docs/relaunch_handoffs/WT-B_2026-04-17.md` — weak-signal track (later superseded by WT-B 04-18)
- `docs/relaunch_handoffs/WT-C_2026-04-17.md` — augmentation / nuisance invariance
- `docs/relaunch_handoffs/WT-D_2026-04-17.md` — decision-system / threshold sweeps
- `docs/relaunch_handoffs/WT-E_2026-04-17.md` — eight-suite promotion contract
- `docs/relaunch_handoffs/WT-F_2026-04-17.md` — proper-data schema

**Existing packet retros (extend, don't duplicate):**
- `docs/packet_retrospectives/packets/WT_infrastructure.md`

**Existing threads (extend, don't duplicate):**
- `docs/packet_retrospectives/threads/promotion_contract_evolution.md` (initial-belief paragraph anchors here)
- `docs/packet_retrospectives/threads/gate_alignment_story.md`

**Master plan LOG**: not yet started (LOG begins 2026-04-26).

**Commits (chronological)**:
- `6c0d1fb` chore: snapshot training-speedup-pass1 before teams relaunch
- `4085ab0` docs: add relaunch worktree agent prompts
- `4f616dc` docs: add relaunch task board and dispatcher prompt
- `a6baedd` Claim WT-A task board track
- `fdbe76a` claim WT-E task board slot
- `e422220` Claim WT-D on relaunch task board
- `609071d` Claim WT-F proper-data schema track
- `213df81` add authoritative teams promotion contract path
- `3dabc13` merge WT-E promotion contract tooling
- `c1556fe` mark WT-E merged on task board
- `b208d17` Freeze WT-A policy truth and lane semantics
- `521d231` Merge WT-A policy truth freeze
- `22b6819` docs: add future proper-data schema and manifest builder
- `d07f06d` Merge WT-F proper-data schema track
- `829d0ba` Close WT-A merge and unblock WT-B
- `4b2b0c5` Update WT-F relaunch task board status
- `6661bbb` WT-C truthful gamma-up sidecars
- `e918397` Merge wt-c-nuisance-2026-04-17
- `3df2430` Mark WT-C merged on relaunch board
- `e30becb` Add WT-D decision-system analysis tooling
- `2c7fc04` Merge wt-d-decision-system-2026-04-17
- `05e5767` Mark WT-D merged on relaunch board
- `733b343` Claim WT-B weak-signal ablation track
- `44743be` Add WT-B weak-signal draft-only package
- `5533985` Merge WT-B weak-signal draft-only package
- `1fa0360` Mark WT-B blocked on source integration
- `47cea27` WT-C add opt-in Teams special augmentation
- `7c01467` Merge wt-c-nuisance-2026-04-17 follow-up
- `cf8e114` Document WT-C test order
- `5fe095d` Document WT-D minimal baseline packet

### Likely topics (filename-level only — no claims)

- April-17 policy reset / data bookkeeping (4,904 ignored, 480+202 retained as hints) — WT-A
- Hint-vs-clean loader split — WT-B (initial draft)
- Augmentation-key truth audit — WT-C
- Decision-layer / threshold sweep tooling — WT-D
- Eight-suite promotion contract scaffold — WT-E
- Proper-data schema introduction — WT-F

---

## Slice 2 — Packet 1: hints test + RLP1 monitoring

**Range**: 2026-04-18 → 2026-04-21
**Status**: done
**Done by**: 2026-04-29, Slice 2 agent (Opus 4.7)

### Source files

**Handoffs:**
- `docs/relaunch_handoffs/WT-B_2026-04-18.md` — WT-B finalization
- `docs/relaunch_handoffs/WT_B_AND_NEW_DATA_READINESS_2026-04-19.md`
- `docs/relaunch_handoffs/RELAUNCH_UPGRADE_REVIEW_PACKET_2026-04-19.md`
- `docs/relaunch_handoffs/NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF_2026-04-19.md`
- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_EXPERIMENT_PLAN_2026-04-19.md`
- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_EVALUATOR_HANDOFF_2026-04-19.md`
- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_LIVE_MONITORING_2026-04-20.md`
- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_MONITORING_HANDOFF_2026-04-20.md`
- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_RESULTS_AND_PLANNING_CONTEXT_2026-04-21.md`
- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN_2026-04-21.md`
- `docs/relaunch_handoffs/R13_RELAUNCH_TRAINING_SOURCE_OF_TRUTH_2026-04-21.md`

**Existing packet retros (extend):**
- `docs/packet_retrospectives/packets/RLP1.md`
- `docs/packet_retrospectives/packets/RLP2.md`

**Master plan LOG**: not yet started.

**Commits (chronological):**
- `f9303eb` 2026-04-19 Land WT-B runtime and launcher smoke readiness
- `7cde78c` 2026-04-19 Add fast WT-B startup smoke and cache priming
- `77facfc` 2026-04-19 Add proper-data runtime and relaunch review packet

### Likely topics

- WT-B runtime + smoke completion
- RLP1 8-slot launch (hints ladder, proper-data integration, FT-vs-scratch)
- Identity-split-mode caveat (`shuffle` → `hash_stable` between RLP1 and RLP2)
- RLP2 plan: unenhanced proper-data as main bet, "enhanced hurts" question
- "Source of truth" handoff anchoring training thread

---

## Slice 3 — Packets 3 / 3.5 / 4: instrumentation + value_composite + first failure

**Range**: 2026-04-21 → 2026-04-23 morning
**Status**: done
**Done by**: 2026-04-29, Slice 3 agent (Opus 4.7)

### Source files

**Handoffs:**
- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_EXPERIMENT_PLAN_2026-04-21.md`
- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_5_EXPERIMENT_PLAN_2026-04-22.md`
- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_RETRO_SCORE_RESULTS_2026-04-22.md`
- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET4_PLANNING_HANDOFF_2026-04-22.md`
- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET4_EXPERIMENT_PLAN_2026-04-22.md`

**Existing packet retros (extend):**
- `docs/packet_retrospectives/packets/RLP3.md`
- `docs/packet_retrospectives/packets/RLP3_5.md`
- `docs/packet_retrospectives/packets/RLP4.md`

**Existing threads (extend):**
- `docs/packet_retrospectives/threads/promotion_contract_evolution.md` (RLP3 A9 `value_composite` introduced)
- `docs/packet_retrospectives/threads/gate_alignment_story.md`

**Master plan LOG**: not yet started.

**Commits (chronological):**
- `c6f1034` 2026-04-22 Land A1-A10 OOD instrumentation and R13 Packet 3 yamls
- `1014b9b` 2026-04-22 Bump VERSION to 1.3.191
- `477b00b` 2026-04-22 Land R13 Packet 3.5 — value_composite config plumbing + cautious ArcFace wave
- `e1cd23e` 2026-04-22 Bump VERSION to 1.3.192
- `179c9e6` 2026-04-22 Update HANDOFF.md for packet 3.5 launch (13 runs in flight across 2 regions)
- `5262e8a` 2026-04-22 Refresh HANDOFF.md with complete packet 3.5 state + next-session status-update recipe
- `872502c` 2026-04-22 Fix value_composite config propagation in train_sweep.py + bump to 1.3.193
- `2cdb804` 2026-04-22 Refresh HANDOFF.md: post-relaunch state with train_sweep.py bug + fix recorded
- `bcf4c61` 2026-04-22 Relocate slot 06 from europe-west4 (stale PENDING) to us-central1
- `c7ac78c` 2026-04-22 Prepare handoff for next agent — flag open W&B anomaly investigation

### Likely topics

- A1–A10 OOD instrumentation stack (dual checkpoints, A6 carve-out, A10 lockbox)
- `value_composite` introduced as RLP3 A9; flips RLP3 rankings
- ArcFace-margin canary (m=0.05/0.10/0.15/0.20); RLP3.5 settles m=0.15
- Gate-definition shift (mean_fpr 0.02→0.03, max_fpr 0.04→0.05, jitter max→p95)
- RLP3 retro-score determinism drift (−0.0067 vs ≤1e-3 spec)
- RLP4 8-slot launch fully fails (yamls not in image) — forces image-rebuild discipline
- `train_sweep.py` value_composite config propagation bug + fix
- Slot-06 region relocation europe-west4 → us-central1

---

## Slice 4 — Packets 5, 6, 7: shortcut surfaces, gate corrected, preprocessing fix

**Range**: 2026-04-23 → 2026-04-24
**Status**: done
**Done by**: 2026-04-29, Slice 4 agent (Opus 4.7)

### Source files

**Handoffs:**
- `docs/relaunch_handoffs/R13_RELAUNCH_VISOMASTER_TEAMS_POOL_DIAGNOSTIC_FINDINGS_2026-04-23.md`
- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET6_EXPERIMENT_PLAN_2026-04-23.md`
- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md`

**Existing packet retros (extend):**
- `docs/packet_retrospectives/packets/RLP5.md`
- `docs/packet_retrospectives/packets/RLP6.md`
- `docs/packet_retrospectives/packets/RLP6B.md`
- `docs/packet_retrospectives/packets/RLP7.md`
- `docs/packet_retrospectives/packets/WS_probes.md`

**Existing threads (extend or create):**
- `docs/packet_retrospectives/threads/processing_signature_shortcut.md`
- `docs/packet_retrospectives/threads/preprocessing_parity_bug.md`
- `docs/packet_retrospectives/threads/calibration_vs_training_aug.md`
- `docs/packet_retrospectives/threads/gate_alignment_story.md`
- `docs/packet_retrospectives/threads/promotion_contract_evolution.md`

**Master plan LOG**: not yet started.

**Commits (chronological):**
- `855871e` 2026-04-24 Fix train/inference preprocessing drift: INTER_AREA → INTER_LINEAR
- `deac44e` 2026-04-24 Add calibration probe (WS-P1) + per-identity reducer (WS-P2.b)
- `c4affa1` 2026-04-24 Draft RLP7_04 (Teams spatial-only) + RLP7_05 (spatial + moderate codec) yamls
- `584118e` 2026-04-24 Add Packet-7 camera-signature handoff: current state + launch candidates
- `5798b3d` 2026-04-24 Refresh HANDOFF.md: Packet-7 pre-launch gate state
- `0f797a1` 2026-04-24 Track Packet-7 RLP7_01/02/03 yamls before image rebuild
- `edd6f50` 2026-04-24 Draft RLP7_06 (Teams CCT-only) + RLP7_07 (spatial + codec + CCT) yamls
- `ed607b9` 2026-04-24 Draft RLP7_08 (codec-aggressive from RLP6_04 step 4500 base)
- `7c3f6d0` 2026-04-24 Bump VERSION to 1.3.205 (RLP7_08 image)

### Likely topics

- VisoMaster-Teams pool diagnostic — 67%/18% TPR gap → E3 recipe path
- RLP5 E3 recipe: enable enhanced_teams lane + family weight 1.0→2.5 (training-side breakthrough)
- RLP5 slot-07 `dor_shkedi` vs `real_dor` flip — shortcut goes from suspicion to fact
- RLP6 gate correction: `worst_pool_fpr` is real-pool-only; avspeech/VCD dropped; `value_composite=0.9006`
- RLP6_04 2-camera test (Dor laptop 0.02 vs Dor webcam 0.94) — shortcut promoted to deployment blocker
- WS-P0 preprocessing fix: `INTER_AREA → INTER_LINEAR` (commit `855871e`)
- WS-P1 calibration probe (30.1% gap closure)
- WS-P2.b per-identity reducer
- RLP7 yamls drafted (5 spatial/codec variants); pre-launch gate cleared

---

## Slice 5 — Packets 8/9 + state-of-detector review

**Range**: 2026-04-24 EOD → 2026-04-26 EOD
**Status**: done
**Done by**: 2026-04-29, Slice 5 agent (Opus 4.7 1M)

### Source files

**Handoffs:**
- `docs/relaunch_handoffs/R13_FULL_STORY_PRE_PACKET_9_2026-04-25.md`
- `docs/relaunch_handoffs/R13_RELAUNCH_STATE_OF_THE_TEAMS_DETECTOR_2026-04-25.md`
- `docs/relaunch_handoffs/PACKET_9_MID_FLIGHT_HANDOFF_2026-04-26.md`

**Existing packet retros: none yet for P8A / P9 / P10** — slice 5 agent likely creates new packet retro(s) under `packets/`.

**Existing threads (extend or create):**
- `threads/processing_signature_shortcut.md` (P8A breaks the camera-signature ceiling)
- `threads/in_proj_svd_gradient_bug.md` (**create** — discovered 2026-04-26)
- `threads/value_composite_semantics.md` (**create** — referenced in plan as backfilled)

**Master plan LOG**: starts 2026-04-26. Read entries:
- `april-26-training-master-plan-v2.LOG.md:44-65` — 2026-04-26 pre-existing state
- `:66-87` — 2026-04-26 plan-headlines + scaffolding
- `:88-125` — 2026-04-26 phase-0-execution
- `:126-175` — 2026-04-26 phase-0-finish + C.1-launch
- `:176-205` — 2026-04-26 A.2-recovery
- `:206-276` — 2026-04-26 phase-c-overnight-slate
- `:277-317` — 2026-04-26 pcp-cleanup
- `:318-353` — 2026-04-26 overnight-monitor-and-slate-readout

**Commits (chronological):**
- `d047da9` 2026-04-25 Draft RLP8_01 (P8A unfreeze CLIP) + RLP8_02 (P8B scratch plain CLIP) yamls
- `d5be7ce` 2026-04-25 Bump VERSION to 1.3.206 (Packet-8 image)
- `1573c59` 2026-04-25 Add Packet-7/8 anchor rescores + overnight readout + state-of-detector doc
- `856ea08` 2026-04-25 Add P8A promotion-contract scorecard inputs
- `8a9ae90` 2026-04-25 Bump VERSION to 1.3.207 (P8A scorecard image)
- `e326b81` 2026-04-25 Extend state-of-detector with W&B per-method + OOD stress slices
- `3426bdf` 2026-04-25 Add P8A step 2500 (ood_composite) anchor rescore
- `0f2f342` 2026-04-25 Add optional backbone_lr_mult to choose_optimizer
- `f766ccd` 2026-04-25 Add R13 full-story pre-Packet-9 doc for second-opinion review
- `ebce585` 2026-04-25 Add real_codec_uplift flag to family-aware aug router
- `dc0478f` 2026-04-25 Add P9_01 yaml — softened P8A (backbone_lr_mult: 0.3)
- `d509188` 2026-04-25 Add P9_03 yaml — no MLP-SVD topology
- `c0a91f7` 2026-04-25 Add P9_05 yaml — real-side codec uplift
- `5af88c7` 2026-04-25 Add P9_R yaml — P8A replication under fresh seeds
- `926af00` 2026-04-25 Add P9_freeze yaml — magnitude/topology disentangle ablation
- `009ae58` 2026-04-25 Bump VERSION to 1.3.208 (Packet-9 prod image)
- `c9997c0` 2026-04-25 Bump VERSION to 1.3.210 (Packet-9 launch image)
- `94d80d4` 2026-04-25 Add Packet-9 +5 longshot pack: stack + R12G fork + schedule probe
- `cbfac08` 2026-04-25 Bump VERSION to 1.3.211 (Packet-9 +5 longshot pack image)
- `2c9778b` 2026-04-26 Add P10 anti-shortcut packet: symmetric router + GRL slate
- `f366368` 2026-04-26 Fix wandb artifact name >128 chars on long log_prefix scorecards
- `d609e39` 2026-04-26 Bump VERSION to 1.3.214 (wandb artifact name fix image)
- `2feea58` 2026-04-26 Fix silent zero-gradient bug in apply_svd_to_in_proj path
- `7af72b1` 2026-04-26 Add anchor-pool monitor + in_proj-fix verification scaffolding
- `ac83ba1` 2026-04-26 Bump VERSION to 1.3.216 (in_proj fix + anchor monitor image)
- `f9ddf3a` 2026-04-26 Bump VERSION to 1.3.217 (Phase A.2 checkpoint-map image)
- `6665910` 2026-04-26 Add Plan-v2 Phase C overnight slate (ablation, codec hedge, RLP6_04 forks)
- `04710ab` 2026-04-26 Bump VERSION to 1.3.218 (Phase C overnight slate image)
- `ad76cd8` 2026-04-26 Add pre-launch image-currency guard and close out PCPs (plan-v2 LOG)

### Likely topics

- Packet-8: P8A (unfreeze CLIP) + P8B (scratch plain CLIP) — P8A is single-variable winner
- P8A breaks the camera-signature ceiling (memory `project_p8a_breakthrough.md`)
- Packet-9: 5 + 5 longshot variants probing P8A's structure
- Memory `project_in_proj_svd_gradient_bug.md` — silent zero-gradient bug discovered, fixed `2feea58`; affects every R12/RLP/P-* run with `apply_svd_to_in_proj=true`
- Phase A.2 Vertex job sequence; pre-launch image-currency guard adopted as discipline
- `value_composite` as semantics question (deployment-grade vs training-side directional)

---

## Slice 6 — P11/P12 + contract policy whack-a-mole + bug audit

**Range**: 2026-04-27 → 2026-04-28
**Status**: done
**Done by**: 2026-04-29, Slice 6 agent (Opus 4.7 1M)

### Source files

**Handoffs:**
- `docs/relaunch_handoffs/WANDB_FLATTENING_BUG_HANDOFF_2026-04-28.md`

**Existing packet retros: none yet for P11 / P12** — slice 6 agent likely creates new packet retro(s).

**Existing threads (extend or create):**
- `threads/contract_policy_bug.md` (**create** — see plan §"Backfilled threads")
- `threads/wandb_flattening.md` (**create** — see plan §"Backfilled threads")
- `threads/face_size_label_leak.md` (**create** — discovered during this stretch per memory)
- `threads/webcam_fpr_dominance.md` (**create**)
- `threads/promotion_contract_evolution.md` (extend with policy-bug fix attempts)

**Master plan LOG entries (read in this slice):**
- `:354-389` — 2026-04-27 07:18 UTC probe-write-and-codec-hedge-validation
- `:390-424` — 2026-04-27 09:50 UTC codec-hedge-readout
- `:425-495` — 2026-04-27 14:55 UTC forks-ii-iii-execution
- `:496-577` — 2026-04-27 17:26 CEST autonomous diagnostic
- `:578-680` — 2026-04-27 18:06 CEST autonomous diagnostic
- `:681-749` — 2026-04-27 18:24 CEST P8A batch inference launched
- `:750-819` — 2026-04-27 21:30 CEST post-compaction
- `:820-900` — 2026-04-27 22:40 CEST continuation
- `:901-972` — 2026-04-27 23:30 CEST STRATEGIC CHECK BEFORE LAUNCH
- `:973-1032` — 2026-04-27 23:55 CEST WEBCAM-HARDEN PRIMITIVE BUILT
- `:1033-1087` — 2026-04-27 23:15 CEST SMOKE PASSED + 4 OVERNIGHT LAUNCHED
- `:1088-1155` — 2026-04-27 23:25 CEST ALL-4-RUNNING + MODERN-LOCKBOX-V2 BUILT
- `:1156-1212` — 2026-04-28 09:00 Morning P11 status + Day-2 verdict setup
- `:1213-1274` — 2026-04-28 09:25 Day-2 pivot: P12_HEAVY_LONG launch + trainer patch
- `:1275-1336` — 2026-04-28 11:53 P11 verdict β + P12 dud + periodic_saves bug + Option A pivot
- `:1337-1386` — 2026-04-28 13:30 Option A subagent verdict + Plan v4 authored

**Commits (chronological):**
- `ed6f53b` 2026-04-27 Fix promotion contract crash on missing readout-only suite reports
- `c7dc828` 2026-04-28 Add periodic step-based checkpoint saves with isinstance(dict) defense
- `cab2909` 2026-04-28 Add P13 anti-shortcut interventions: anchor-aware loss, pipeline-random aug, face scale-jitter
- `e748d78` 2026-04-28 Bump VERSION to 1.3.224 (P13 anti-shortcut image) + handoff for launch
- `c366026` 2026-04-28 Fix wandb-flattening: re-apply anchor_aware/face_scale_jitter/periodic_saves
- `6c9a320` 2026-04-28 Bump VERSION to 1.3.226 (post-wandb-flattening-fix image)

### Likely topics

- **Contract policy bug whack-a-mole**: 04-23 first noticed, 04-27 second attempt, 04-29 third (recall floor) — see memory `project_contract_policy_bug.md`. Three "fix" attempts in 6 days, none committed.
- Codec-hedge validation (~$3, 32 reports, 5 contract artifacts)
- Webcam-harden primitive built + 4-run overnight (P11)
- Modern-lockbox-v2 (memory `project_lockbox_fpr_dominated_by_webcam_mode.md`)
- P11 day-2 verdict β; P12_HEAVY_LONG launch + dud
- `periodic_saves` patch silently failed (LOAD-BEARING bug)
- Plan v4 authored
- Wandb-flattening bug: nested yaml blocks silently dropped (`project_wandb_flattens_nested_dicts.md`)
- P13 anti-shortcut interventions yaml drafted (anchor-aware loss, pipeline-random aug, face scale-jitter)
- Face-pixel-area label leak discovered (memory `project_face_size_label_leak.md`)

---

## Slice 7 — P13 verdict, bucket gap, contract policy fix v3, P15 GRL note

**Range**: 2026-04-28 EOD → 2026-04-29
**Status**: done
**Done by**: 2026-04-29, Slice 7 agent (Opus 4.7 1M)

### Source files

**Handoffs:**
- `docs/relaunch_handoffs/P13_DAY4_VERDICT_2026-04-29.md`
- `docs/relaunch_handoffs/TRAINING_EVAL_VISO_BUCKET_GAP_2026-04-29.md`
- `docs/relaunch_handoffs/P15_GRL_READINESS_NOTE_2026-04-29.md`

**Existing packet retros: none yet for P13 / P14 / P15** — slice 7 agent likely creates new packet retro(s).

**Existing threads (extend or create):**
- `threads/viso_bucket_gap.md` (**create** — see plan §"Backfilled threads")
- `threads/contract_policy_bug.md` (extend with v3 / recall-floor fix; this is the canonical thread for the whack-a-mole)
- `threads/identity_audit.md` (**create** — see plan §"Backfilled threads")
- `threads/clean_teams_identity_pairing.md` (**create**)
- `threads/processing_signature_shortcut.md` (extend with frame-level AUC)

**Master plan LOG entries (read in this slice):**
- `:1387-1454` — 2026-04-29 Bucket-gap diagnosis (post-P13_FROM_SCRATCH γ verdict)

**Commits**: none on 2026-04-29 yet (verified via `git log --since=2026-04-29`).

### Likely topics

- P13_FROM_SCRATCH Day-4 verdict (γ — Axis 3 Δ = 0.520 vs gate 0.15; cross-domain recall collapse)
- Identity-diversity refutation: 429 viso + 1916 deeplive in active training, +2,300 unused (memory `project_data_inventory_identity_diversity.md`)
- **Bucket gap**: training viso (no enhancers, no Teams) ≠ eval viso (`visomaster_enhanced_macro` Teams-recapture); ~3-5× recall gap explained
- P14_DATA_FIX retrain proposal (~$70, 1-2 days)
- Contract policy bug fix v3 (recall_floor=0.30 + recall_min=0.30 + stress_fpr=0.10) — in working tree, uncommitted, with test coverage (memory `project_contract_policy_bug.md`)
- P15 GRL readiness note (drafted, not launched, smoke-test required)
- P8A frame-level AUCs (memory `project_p8a_frame_level_auc_2026-04-29.md`): viso 0.75 / deeplive 0.86 / teams_fake 0.91 — load-bearing evidence "low recall" was scorer artifact, not model failure
- Clean and Teams buckets share identities — paired transport (memory `project_clean_teams_same_identity.md`)

---

## Source-file index by date (cross-reference)

| Date | Handoffs added | Packet retros existing | LOG entries |
|---|---|---|---|
| 2026-04-17 | 7 (TASK_BOARD, WT-A/B/C/D/E/F + json) | WT_infrastructure | — |
| 2026-04-18 | 1 (WT-B updated) | — | — |
| 2026-04-19 | 4 (WT-B/data readiness + RELAUNCH_UPGRADE + NEW_DATA_LOADER + RLP1_PLAN + RLP1_EVALUATOR) | RLP1 | — |
| 2026-04-20 | 2 (RLP1_LIVE_MONITORING + RLP1_MONITORING_HANDOFF) | — | — |
| 2026-04-21 | 4 (RLP1_RESULTS, RLP2_PLAN, RLP3_PLAN, TRAINING_SOURCE_OF_TRUTH) | RLP2 | — |
| 2026-04-22 | 4 (RLP3_5_PLAN, RLP3_RETRO_SCORE, RLP4_PLANNING, RLP4_PLAN) | RLP3, RLP3_5, RLP4 | — |
| 2026-04-23 | 2 (VISOMASTER_TEAMS_DIAGNOSTIC, RLP6_PLAN) | RLP5, RLP6, RLP6B | — |
| 2026-04-24 | 1 (RLP7_CAMERA_SIGNATURE) | RLP7, WS_probes | — |
| 2026-04-25 | 2 (R13_FULL_STORY, STATE_OF_THE_TEAMS_DETECTOR) | — | — |
| 2026-04-26 | 1 (PACKET_9_MID_FLIGHT) | — | 7 sessions |
| 2026-04-27 | 0 | — | 12 sessions |
| 2026-04-28 | 1 (WANDB_FLATTENING_BUG) | — | 4 sessions |
| 2026-04-29 | 3 (P13_DAY4_VERDICT, TRAINING_EVAL_VISO_BUCKET_GAP, P15_GRL_READINESS_NOTE) | — | 1 session |

Total handoffs in scope: 32 markdown files (READMEs and the JSON artifact excluded from "in scope" but listed in slice 1).
Total existing packet retros: 11 under `packets/` (including `WT_infrastructure.md`).
Total existing threads: 5 (`promotion_contract_evolution`, `processing_signature_shortcut`, `preprocessing_parity_bug`, `calibration_vs_training_aug`, `gate_alignment_story`).
Master plan log entries: 24 sessions across 04-26 → 04-29 (1454 lines total).
Commits in scope: 87 commits 04-17 → 04-28.

---

## Threads to backfill during build (per plan)

This is a non-exhaustive list from the plan; slice agents should create more as topics surface.

- `threads/contract_policy_bug.md` — canonical thread for the whack-a-mole (Slice 6 or 7).
- `threads/viso_bucket_gap.md` — bucket gap diagnosis (Slice 7).
- `threads/in_proj_svd_gradient_bug.md` — silent zero-gradient bug (Slice 5).
- `threads/identity_audit.md` — identity-diversity refutation (Slice 7).
- `threads/value_composite_semantics.md` — what `value_composite` does and does not say (Slice 5; consolidate with `promotion_contract_evolution.md` if redundant).
- `threads/face_size_label_leak.md` — face-pixel-area label leak (Slice 6).
- `threads/webcam_fpr_dominance.md` — webcam-style captures dominate lockbox FPR (Slice 6).
- `threads/clean_teams_identity_pairing.md` — clean and teams buckets share identities (Slice 7).
- `threads/wandb_flattening.md` — nested-yaml-block flattening bug (Slice 6).

---

## Memory entries by topic (for slice agents updating frontmatter)

Memory entries that slice agents will likely touch (add `status:`, `wiki_ref:`, `last_verified:` frontmatter):

| Memory entry | Likely owning thread | Likely slice |
|---|---|---|
| `project_promotion_contract.md` | `promotion_contract_evolution` | 1 (initial), 3 (refined) |
| `project_contract_policy_bug.md` | `contract_policy_bug` | 6 / 7 (smoke-test by Slice 0) |
| `feedback_promotion_contract_launch.md` | `promotion_contract_evolution` | 5 |
| `project_signature_shortcut_finding.md` | `processing_signature_shortcut` | 4 |
| `project_p8a_breakthrough.md` | `processing_signature_shortcut` | 5 |
| `project_p8a_frame_level_auc_2026-04-29.md` | `processing_signature_shortcut` | 7 |
| `project_in_proj_svd_gradient_bug.md` | `in_proj_svd_gradient_bug` | 5 |
| `project_shortcut_is_upstream.md` | `processing_signature_shortcut` | 4 / 5 |
| `project_lockbox_fpr_dominated_by_webcam_mode.md` | `webcam_fpr_dominance` | 6 |
| `project_face_size_label_leak.md` | `face_size_label_leak` | 6 |
| `project_viso_train_eval_bucket_gap.md` | `viso_bucket_gap` | 7 |
| `project_wandb_flattens_nested_dicts.md` | `wandb_flattening` | 6 |
| `project_data_inventory_identity_diversity.md` | `identity_audit` | 7 |
| `project_clean_teams_same_identity.md` | `clean_teams_identity_pairing` | 7 |
| `project_success_criteria.md` | (cross-cutting; cite from many threads) | any |
| `project_gcs_region_locality.md` | (operational, cite from CLAUDE.md) | any |
| `feedback_no_cancelling_vertex_jobs.md` | (operational) | any |
| `feedback_sklearn_njobs.md` | (operational) | any |
| `feedback_decision_points.md` | (operational) | any |
| `feedback_small_sample_guidance.md` | (operational) | any |
| `reference_image_rebuild.md` | (operational) | any |
| `user_role.md` | (operational) | any |
| `feedback_promotion_contract_launch.md` | `promotion_contract_evolution` | 5 |

---

## Slice agent contract (reminder; full contract in `AGENTS.md`)

When a slice agent finishes:

1. Mark its slice's status as `done` (replace `not started` or `ready` with `done`).
2. Fill in the `Done by:` line with date + agent identifier.
3. Mark the next slice's status as `ready`.
4. Run `python tools/regenerate_open_loops.py` from `docs/packet_retrospectives/`.
5. Append TIMELINE entries.
6. Stop. Do not start the next slice.
