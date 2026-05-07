# R13 Relaunch — Packet Retrospectives

> **⚠ Critical-reading note (added 2026-05-04 night, updated 2026-05-05, 2026-05-07)**: This index is approximately current through 2026-04-30, but several entries below have been nuanced or refuted by post-2026-04-30 work that did NOT all back-fill into this README. **2026-05-07 update**: rolling [`STATE.md`](STATE.md) is now the canonical current-state snapshot (formerly date-stamped `STATE_2026-04-30.md`, archived to `archive/`). Per-session `HANDOFF_*.md` files in `docs/relaunch_handoffs/` are now a frozen archive — new findings flow into STATE.md + threads + packet retros. The eval-folder authoring contract (FACTS / OPINIONS file-level split) is documented in [`eval_folder_template.md`](eval_folder_template.md). New agents: see "Reading order for forming an independent view" in [`AGENTS.md`](AGENTS.md). Specifically: (1) the "Confirmed good" entry for `face_scale_jitter@0.50 isolated` was refuted on its own design-intent mechanism the same afternoon (TIMELINE 2026-04-30 afternoon — face-size flip rate 43.9% on the leader vs P8A 39.4%); (2) the `mclioexb` leader did NOT promote on the contract scorecard the same evening; (3) the bucket-fix-as-deployed framing in [`viso_bucket_gap`](threads/viso_bucket_gap.md) Current stance is being retested in flight on 2026-05-04 evening (Packet A + Packet C-codec); (4) the contrastive-loss inference in [`clean_teams_identity_pairing`](threads/clean_teams_identity_pairing.md) has a DEBATE block (2026-05-04) flagging the symmetric-KL pair-loss premise as empirically refuted on E2B for viso; (5) per-mode τ recall lifts cited in [`webcam_fpr_dominance`](threads/webcam_fpr_dominance.md) and [`calibration_vs_training_aug`](threads/calibration_vs_training_aug.md) are OFFLINE-ONLY because Teams does not surface capture mode at inference (`feedback_per_mode_tau_not_deployable.md`). Read individual thread "Current stance" sections AND any post-stance dated update subsections; do not treat README endorsements as final.
>
> **⚠ Quality-enhancement routing bug — affects ALL R13 packets (added 2026-05-05)**: every R13 P-series, S-series, E-series, and A/B/C-series packet trained between 2026-03-19 and 2026-05-05 was trained with `quality_enhancement` deeplive frames mislabeled as `deeplive_enhanced_fake` family (~5,120 frames / ~27% of the enhanced-fake training volume). Confirmed via manifest evidence (`quality_enhancement_*` folders carry no `enhancement: GFPGAN_sample` tag) AND visual inspection. Fix landed 2026-05-05. **Relative comparisons between R13 ckpts remain valid** (uniform contamination across packets); **absolute claims about model handling of GFPGAN-enhanced fakes need revisiting after the first post-fix training run**. See [`quality_enhancement_strategy_misrouting`](threads/quality_enhancement_strategy_misrouting.md) for full evidence and resolution.

> **Agents start here**: read [`AGENTS.md`](AGENTS.md) before doing anything else in this tree. It is the read/update protocol that prevents the rediscovery whack-a-mole the wiki was built to fix. Working agents follow that protocol every session; this README is the executive index, not the entry point.
>
> **Quick map of this tree** (full description in [`AGENTS.md`](AGENTS.md)):
>
> - [`STATE.md`](STATE.md) — rolling current-state snapshot. Always-current; pass-1 step 1 in the read protocol.
> - [`OPEN_LOOPS.md`](OPEN_LOOPS.md) — mechanically-generated open-issue inventory. Read every session.
> - [`TIMELINE.md`](TIMELINE.md) — chronological master index, append-only.
> - [`AGENT_GUIDE.md`](AGENT_GUIDE.md) — rolling guide; validate-before-suggest checklist for next-packet proposals.
> - [`thread_template.md`](thread_template.md) / [`packet_template.md`](packet_template.md) / [`eval_folder_template.md`](eval_folder_template.md) — canonical formats (per-thread / per-packet / per-eval-folder).
> - [`threads/`](threads/) — cross-cutting topic docs.
> - [`packets/`](packets/) — per-packet retros.
> - [`archive/`](archive/) — superseded dated snapshots (e.g., older STATE files).
> - [`plans/`](plans/) — frozen multi-page plan documents authored at specific points in time.
> - [`tools/regenerate_open_loops.py`](tools/regenerate_open_loops.py) — run after editing any `### Open loop:` block.
> - [`BUILD_SCAFFOLD.md`](BUILD_SCAFFOLD.md) — chronological build map for slice agents (build complete 2026-04-29).

> Originally written 2026-04-24 as the Phase-2 synthesis across 11 packet drafts; rolled forward 2026-04-30 with the P8A → P15 substrate-redesign + anti-shortcut sequence. Each packet owns its own file under `packets/`; this README is the executive index and the reconciliation log.

## Arc in one paragraph

R13 began from a legacy baseline with mixed VisoMaster lane semantics that a late-March audit had begun to sour. On 2026-04-17 a full policy reset landed as six parallel worktrees ([WT-A](packets/WT_infrastructure.md) through WT-F): bad-data bookkeeping (4,904 ignored, 480+202 retained as hints), a hint-vs-clean loader split, augmentation-truth audits, a decision layer, the eight-suite promotion contract, and the new `proper_data` schema. On that foundation [RLP1](packets/RLP1.md) ran an 8-slot probe (hints ladder, proper-data integration, FT-vs-scratch) — hints failed, unenhanced proper-data was the only promising new signal. [RLP2](packets/RLP2.md) made unenhanced proper-data the main bet and tried to read "enhanced hurts" (muddled in hindsight — the validation pool was unenhanced-only). [RLP3](packets/RLP3.md) landed the A1–A10 instrumentation stack and introduced the deployment-aligned `value_composite` metric, which immediately flipped the packet's rankings and surfaced the "~5× smaller deltas" pattern. [RLP3.5](packets/RLP3_5.md) ran the arcface-margin canary and settled `m=0.15` as control; [RLP4](packets/RLP4.md) failed wholesale (all 8 jobs died on a yaml-not-in-image error) but forced the image-rebuild discipline. [RLP5](packets/RLP5.md)'s E3 recipe moved the two proper-VisoMaster-Teams fake lanes from 67%/18% to 94%/96% — a real training-side breakthrough, but slot-07 later revealed the detector had learned a camera/ISP processing signature, not a manipulation-content boundary. [RLP6](packets/RLP6.md) reframed the ceiling as gate-bound (real-pool-only `worst_pool_fpr`, avspeech out, hint-clean `teams_ood_fake`) and unlocked `value_composite=0.9006`, then the same leader failed a 2-camera controlled test (Dor laptop 0.02 vs Dor webcam 0.94) — making the shortcut a deployment blocker. [RLP7](packets/RLP7.md) is the camera-signature response (Teams spatial + codec training-aug, per-camera calibration, per-identity reducer); pre-launch gate cleared with verdict `fully_structural__launch_full_subset` and launch is pending a user trigger.

## Arc continuation (P8A → P15, the substrate-redesign track)

[P8A](packets/P8A.md) (2026-04-25 RLP8_01) broke the camera-signature ceiling: unfreezing CLIP `visual.proj` + `ln_post` + applying SVD to the MLP cut anchor `dor-real-webcam-no-VBG` Δ by ~2× vs RLP6_04 — but the in_proj-SVD lever was silently zero-gradient pre-`2feea58` (memory `project_in_proj_svd_gradient_bug.md`), so P8A's attribution is split between MLP-SVD + visual.proj/ln_post unfreeze. [P9](packets/P9.md) → [P12](packets/P12.md) ran recipe-tuning portfolios on the P8A substrate; the bundle of findings — the buggy contract policy crowning τ ≈ 0.99 candidates, the wandb-flattening recurrence that silenced anchor_aware + face_scale_jitter on P13's first launch, the slot-1000 checkpoint policy artifact — accumulated faster than the recipe-tuning could absorb. [P13_FROM_SCRATCH](packets/P13.md) (2026-04-29) was the substrate-redesign track of Plan v4 with the three-lever anti-shortcut bundle (anchor_aware loss + pipeline_random aug + face_scale_jitter@0.25): Day-4 verdict γ — Axis 3 shortcut Δ moved directionally (P8A 0.66 → P13 0.52, ~14pp improvement) but cross-domain capability collapsed. The Day-4 audit also surfaced the **viso bucket gap** ([`viso_bucket_gap`](threads/viso_bucket_gap.md)) and the **face-pixel-area label leak** ([`face_size_label_leak`](threads/face_size_label_leak.md)) as parallel structural findings. [P14](packets/P14.md) launched FT-from-P8A_step5000 + the P13 bundle (Day-5 fallback); the Slice-7 wiki build authored its plan retro before the verdict. [P15](packets/P15.md) added GRL on a quality-domain head (DANN, Ganin 2015) — the only structurally-different anti-shortcut lever the codebase already supported — drafted but not launched at end of Slice 7. **Overnight 2026-04-29 → 2026-04-30** the slate ran in parallel with a sister-variant ablation: jitter@0.50 ALONE (anchor_aware + pipeline_random DISABLED) won the packet at trainer-side `value_composite=0.661`, beating both the bundle (0.116) and DATA_FIX (0.126) by 5.7×, with P15 GRL second at 0.516. The bundle was net-negative against its single load-bearing component; new thread [`anti_shortcut_bundle_decomposition`](threads/anti_shortcut_bundle_decomposition.md) captures the discipline rule. Promotion-contract scorecard for the leader (`mclioexb`) is in flight 2026-04-30 morning on us-east1 (image 1.3.233).

## Confirmed good

- **`value_composite` deployment-aligned metric** ([RLP3](packets/RLP3.md), still in use through [RLP7](packets/RLP7.md)) — dual-checkpoint saving, A6 carve-out, A10 held-out partition still operational. **Caveat post-2026-04-25**: trainer-side composite is *not* deployment-grade ([`promotion_contract_evolution`](threads/promotion_contract_evolution.md)); promotion goes through lockbox-anchored contract scorecard.
- **Proper-data unenhanced recipe** ([RLP1](packets/RLP1.md) slot-04, [RLP2](packets/RLP2.md) slot-02) — the anchor that every subsequent packet inherits.
- **arcface `m=0.15`** ([RLP3.5](packets/RLP3_5.md), carried through [RLP7](packets/RLP7.md)) — the only large single-lever win in the margin scan.
- **E3 dose-matched training recipe** ([RLP5](packets/RLP5.md)) — enabling `proper_visomaster_enhanced_teams` lane + family weight 1.0→2.5 closes the 67%/18% diagnostic gap. Training-side win is real (separate from shortcut question).
- **Spatial backbone stacking** ([RLP5](packets/RLP5.md) on top of [RLP3](packets/RLP3.md)) — `context_variation_shift: 0.08, context_variation_individual_p: 0.40` is the operating default.
- **WT-A data-policy bookkeeping** ([WT-infrastructure](packets/WT_infrastructure.md)) — the root artifact every gate-alignment claim resolves against.
- **WS-P0 preprocessing fix** ([WS probes](packets/WS_probes.md), commit `855871e`) — `cv2.INTER_AREA → cv2.INTER_LINEAR` at `batch_inference_gcs.py:407` and `arena/model_arena.py:472`, guarded by parity test.
- **WS-P2.b per-identity reducer** ([WS probes](packets/WS_probes.md)) — `arena/postprocess_per_identity.py`; enables "narrowest per-identity FPR spread" as the RLP7 leader-pick metric.
- **Real-pool-only `worst_pool_fpr` correction** ([RLP6](packets/RLP6.md)) — permanent contract invariant; fake pools cannot drive it by construction.
- **CLIP visual.proj + ln_post + MLP-SVD unfreeze (P8A recipe)** ([P8A](packets/P8A.md)) — broke the camera-signature ceiling on the anchor pool. P8A_step5000 is the FT init for P14/P15 and the leader's parent.
- **face_scale_jitter@0.50 isolated** ([P14](packets/P14.md), 2026-04-30 leader on trainer composite) — sister-variant ablation winning trainer-side `value_composite=0.661` with anchor_aware + pipeline_random DISABLED. **Status updated 2026-04-30 evening + afternoon (see TIMELINE)**: the leader (`mclioexb`) did NOT promote on the contract scorecard, AND the design-intent mechanism (close the face-size leak) was directly probed and the leader did not close the leak (43.9% flip rate vs P8A 39.4% on 180-frame face-size invariance probe). Trainer-side composite win survives as a measurement; production-grade win + design-intent close are both negative. The "use scale_limit ≥ 0.50 by default" framing reads as a weak "strictly less bad than the alternatives in that bake-off," not a recommendation; see [`face_size_label_leak`](threads/face_size_label_leak.md) and [`jitter_winner_mechanism_unknown`](threads/jitter_winner_mechanism_unknown.md) for the refutation evidence.
- **GRL infrastructure on quality-domain head** ([P15](packets/P15.md), 2026-04-30 first-ever production exercise) — DANN λ=0.20 static; the head fired live, cross-domain head room was preserved (other_fakes_tpr=0.659). Structurally working; the lever as deployed (with bundle drag) is materially weaker than jitter@0.50 isolated on trainer composite. Neither lever has produced a contract-grade win as of 2026-05-04.

## New 2026-04-30 disciplines

- **Anti-shortcut bundle decomposition** ([`anti_shortcut_bundle_decomposition`](threads/anti_shortcut_bundle_decomposition.md)) — when stacking ≥ 2 anti-shortcut interventions, at least one slot must be the strongest single lever alone holding FT init + data + LR fixed. The P14 bundle was net-negative against jitter alone by 5.7×; the discipline prevents future packets from shipping bundles whose components mask each other.

## Confirmed bad / abandoned

- **Hints as supervision** ([RLP1](packets/RLP1.md)/[RLP2](packets/RLP2.md)) — both RLP1_02/03 and RLP2_01/03 underperformed their non-hint controls across matched holdouts. Closed.
- **"Enhanced hurts" as a clean conclusion** ([RLP2](packets/RLP2.md), revised muddled) — the val_holdout was unenhanced-only; the conclusion couldn't be read. [RLP3](packets/RLP3.md) added A6 OOD lanes to fix the measurement.
- **Scratch training** ([RLP1](packets/RLP1.md) slot-08) — non-competitive on balance despite a late OOD spike. FT-from-R12_G_FP32 is default.
- **avspeech in real-pool OOD gate** ([RLP6](packets/RLP6.md)) — 54.2%/45.8% FPR at natural τ, not deployment-distribution; force-pushed τ to ~0.995. Out permanently.
- **`visomaster_*` bad rows as clean supervision** ([WT-A](packets/WT_infrastructure.md)) — 4,904 rows ignored; `path_exclude_contains: ["/visomaster_"]` on `teams_ood_fake` is standard.
- **`wma_failure_fake` as a gate driver** ([RLP6](packets/RLP6.md) correction) — real-pool-only `worst_pool_fpr` makes the prior "wma drives the gate" framing a category error. Demoted to readout-only.
- **Pre-fix retro-score absolute numbers** (all packets through [RLP6](packets/RLP6.md) except 6_04; see [preprocessing_parity_bug](threads/preprocessing_parity_bug.md)) — suspect for absolute-value comparisons. Intra-packet rankings often survive; cross-packet numeric comparisons do not.
- **Lockbox 90/90 on RLP6_04 as threshold-reachable** (memory `project_signature_shortcut_finding.md`) — at τ catching 90% fakes, real FPR is 63.6%. Representation ceiling, not calibration gap.
- **The P14 anti-shortcut bundle (anchor_aware + pipeline_random + face_scale_jitter@0.25) as a stack** ([P14](packets/P14.md), 2026-04-30) — net-negative against jitter@0.50 alone by 5.7× on trainer-side value_composite. Bundle composition was wrong; one single lever (jitter strengthened) carries the lift. Discipline rule in [`anti_shortcut_bundle_decomposition`](threads/anti_shortcut_bundle_decomposition.md). Future packets must include single-lever ablation when stacking.
- **`R13_P14_DATA_FIX.yaml` as a bucket-gap-as-headline-driver lever** ([P14](packets/P14.md) variant 2, 2026-04-30) — empirically refuted Move-1 prediction (memory `project_move1_bucket_gap_refuted.md`). Adding visomaster_teams_enhanced at fw=8.0 collapsed cross-method generalization (`other_fakes_tpr=0.047`). Bucket-gap finding remains valid at the frame-level AUC layer; the lever as deployed (DATA_FIX yaml on top of bundle) is not viable.
- **`visomaster_hints` and `visomaster_hints_teams` as training signal** (memory `project_visomaster_hints_lanes_bad_data.md`) — bad data lane (April-17 face_parser_enabled bug residue); RLP1 ablation showed net drag; the Teams variant has co-encoded codec×partial-swap shortcut confound. Policy: never enable.

## Open mysteries

- **RLP5 slot-01 was never probed separately for the shortcut.** Slot-01 and slot-07 are seed replicates on the same E3 recipe (Δ +0.0045, inside seed noise). If slot-01 carries the same shortcut, the E3 "breakthrough" is more shortcut-driven than currently recorded.
- **~70% of cross-pool FPR gap remains** after per-camera calibration (WS-P1 closed 30.1%). Training-aug is the dominant lever but nobody knows whether RLP7 closes the residual 70%, or where the next lever past RLP7 comes from.
- **No pre-fix leader ranking has been re-scored** except RLP6_04. Whether any rank shuffles materially under INTER_LINEAR is unknown; RLP5_07 post-fix re-score is the obvious next check.
- **RLP3 retro-score determinism drift** — `retro 0.6030 vs training 0.6097 (−0.0067)` on sanity. Larger than the ≤1e-3 spec asked for. Root cause not traced.
- **FaceDancer blind spot** (per convmem mining) — no RLP has attacked FaceDancer pipelines directly; whether the shortcut generalizes across manipulation families is untested.
- **RLP6B: never launched.** The quality_lambda / feat_norm_reg sweep targets a low-norm shortcut framing that [RLP7](packets/RLP7.md) partly supersedes with camera/ISP framing. Decision pending on merge vs standalone; see [RLP6B](packets/RLP6B.md) recommendation to fold into [RLP6](packets/RLP6.md).

## Current status (2026-04-24)

[RLP7](packets/RLP7.md) is authored and pre-launch gate is **cleared** — `analysis/rlp6_04_postfix_rescore_2026-04-24.summary.json` carries verdict `fully_structural__launch_full_subset`, confirming the camera/ISP shortcut survives the preprocessing fix and is structural. Five yamls are on disk (`RLP7_01..05`); handoff recommendation is to launch the subset `{RLP7_05, RLP7_02, RLP7_04}` first, with `RLP7_01/_03` (lighting-aggressive, combined) held in reserve because controlled-pool evidence does not point at lighting as dominant. Launch is blocked on a user-triggered run; `WANDB_*` environment variables must be exported (memory `feedback_promotion_contract_launch.md`), and new yamls require a `./dev.sh build-prod -y` image rebuild before container launch (memory `reference_image_rebuild.md`). Next probe in the implicit queue is a WS-P1 re-run on post-fix inputs.

## Current status (2026-05-07)

> **Canonical current state lives in [`STATE.md`](STATE.md)** (rolling). This dated section is a stable snapshot of the headline at the time of writing; STATE.md has the always-current version.

The most recently completed packet is **[P1](packets/P1.md) (PE_PAIR_RANK_DRO)** — pair_rank λ=0.2 + multi-axis GroupDRO with `chronic_flag` (BUNDLE) vs pair_rank-only (PAIRRANK), both FT-from-P8A_step5000 on the post-`2feea58` codepath. **Phase A (29-suite contract scorecard) SUCCEEDED**; Phase C (16-suite HDTF cross-substrate) FAILED at the `promotion_contract/` step but the F4 verdict was recovered locally. **F4 PASS + F5 PASS BIG for BUNDLE** (PC_Generator chronic FPR drops 62.9% → 0%); **F1 fails at contract τ but is reachable at non-contract τ** (`BUNDLE_step500` hits 96.5% lockbox recall at FPR ≤ 10% τ=0.989); **F3 partial** (3 of 4 untargeted IQ axes decouple; face_area_fraction amplifies +106-266%); **F2(a) not testable** on Phase A substrate (only 1 of 6 paired training lanes represented; baseline saturated). Critically, **P1 introduced a Roy_D regression** (29% → 78-93% FPR on 130 single-frame video_ids); the regression is `color_b_dev`-aligned (Δr=+0.71 mirror of P8A's r=−0.71) AND **shared between BUNDLE and PAIRRANK arms** (Wilcoxon p=0.875), which refuted the agent's initial GroupDRO-balloon hypothesis and shifted the suspect lever to `pair_rank_loss` itself. **Deployment remains E2B** (memory `project_deployment_is_e2b_2026-05-06.md`); E2B beats every P1 ckpt on `dev_fake_macro_recall`. Two bug fixes landed mid-session and are uncommitted: `trainer/trainer.py:1718-1727` (W&B logging gap when `use_group_dro=true`) and `analysis/p1_pe_eval_2026-05-07/phase_d/run_chronic_filter.py` (regex matcher silently dropped PC_Generator + Q chronic identities). The wiki contract was also upgraded this session: rolling [`STATE.md`](STATE.md), FACTS/OPINIONS file-level split via [`eval_folder_template.md`](eval_folder_template.md), [`AGENT_GUIDE.md`](AGENT_GUIDE.md) moved into the wiki, [`AGENTS.md`](AGENTS.md) extended with the FACTS-first / OPINIONS-second reading order, hard stop on per-session HANDOFF authoring (`docs/relaunch_handoffs/` is now a frozen archive). New thread [`pair_rank_collateral`](threads/pair_rank_collateral.md) tracks the Roy_D mechanism hypothesis.

---

## Current status (2026-04-30)

The **2026-04-29 → 2026-04-30 overnight slate** ran three FT-from-P8A_step5000 variants in parallel: P14_FT bundle (anchor + pipeline + jitter@0.25), P14_DATA_FIX (bundle + visomaster_teams_enhanced fw=8.0), P14_FACE_SCALE_JITTER_ISOLATED (jitter@0.50 alone, anchor + pipeline DISABLED), and P15_GRL (bundle + GRL λ=0.20). Trainer-side `value_composite` ranking: **`mclioexb` (jitter@0.50 isolated) 0.661 > `w5tky6ss` (P15 GRL) 0.516 ≫ `xan4dfto` (DATA_FIX) 0.126 ≈ P14_FT bundle 0.116**. The single-lever ablation beat the full bundle by 5.7× — see [`anti_shortcut_bundle_decomposition`](threads/anti_shortcut_bundle_decomposition.md). DATA_FIX confirmed Move-1's pre-launch prediction (memory `project_move1_bucket_gap_refuted.md`); P15 GRL's first-ever production exercise of the DANN infrastructure preserved cross-domain head room (other_fakes_tpr=0.659) but is materially weaker than jitter alone. **Promotion-contract scorecard on `mclioexb` is in flight** on us-east1 (image 1.3.233 via Cloud Build `46ef62da`); ETA ~3h. Two CPU-only validation analyses are also queued: face-size invariance probe + domain-confusion linear probe (`analysis/face_size_invariance_2026-04-30/` + `analysis/domain_confusion_probe_2026-04-30/`). See [STATE.md](STATE.md) for the rolling current-state snapshot, including pending decisions and next-action recommendations. (Older dated snapshots — `STATE_2026-04-29.md`, `STATE_2026-04-30.md` — are preserved in `archive/`.)

## How to read this tree

### Packets (chronological)

| Packet | Dates | Verdict | Headline |
|---|---|---|---|
| [WT-infrastructure](packets/WT_infrastructure.md) | 2026-04-17 → 2026-04-19 | ✅ | Six parallel worktrees; April-17 policy reset |
| [RLP1](packets/RLP1.md) | 2026-04-19 → 2026-04-21 | ✅ | Hints fail; unenhanced proper-data is the only new signal |
| [RLP2](packets/RLP2.md) | 2026-04-20 → 2026-04-21 | ⚠️ | Proper-unenhanced confirmed; "enhanced hurts" muddled |
| [RLP3](packets/RLP3.md) | 2026-04-21 → 2026-04-22 | ✅ | A1–A10 instrumentation + `value_composite` metric |
| [RLP3.5](packets/RLP3_5.md) | 2026-04-22 | ✅ | Arcface `m=0.15` adopted; stability/smoothing dropped |
| [RLP4](packets/RLP4.md) | 2026-04-22 → 2026-04-23 | ❌ | All 8 jobs failed (yamls not in image); forced rebuild discipline |
| [RLP5](packets/RLP5.md) | 2026-04-23 | 🔬 | E3 training-side breakthrough; shortcut finding pending |
| [RLP6](packets/RLP6.md) | 2026-04-23 → 2026-04-24 | ✅/🔬 | Gate alignment (+0.127); slot-04 fails on camera shortcut |
| [RLP6B](packets/RLP6B.md) | 2026-04-23 → 2026-04-24 | 🟡 | Authored, never fired; hypothesis partly superseded |
| [RLP7](packets/RLP7.md) | 2026-04-24 | 🟡 | Camera/ISP countermeasure; pre-launch gate cleared |
| [WS probes](packets/WS_probes.md) | 2026-04-24 | ✅ | WS-P0 preprocessing fix; WS-P1 calibration 30.1%; WS-P2.b reducer |
| [P8A](packets/P8A.md) | 2026-04-25 | ✅ | RLP8_01 unfreeze CLIP visual.proj + ln_post + MLP-SVD; broke camera-signature ceiling on anchor |
| [P9](packets/P9.md) | 2026-04-25 → 2026-04-26 | 🔬 | Recipe-tuning P8A variants; soften lever sweep; muddled by contract policy bug |
| [P10](packets/P10.md) | 2026-04-26 | 🔬 | Anti-shortcut packet (symmetric router + GRL slate drafted); GRL half deferred |
| [P11](packets/P11.md) | 2026-04-27 → 2026-04-28 | 🟡 | First face-size axis disruption (ctx_scale 0.30/0.50) + WEBCAM_HARDEN; HEAVY direction confirmed; step-1000 checkpoint policy artifact |
| [P12](packets/P12.md) | 2026-04-28 | ❌ | HEAVY_LONG dud; periodic_saves silently failed (wandb-flattening recurrence) |
| [P13](packets/P13.md) | 2026-04-28 → 2026-04-29 | 🔬 | FROM_SCRATCH + 3-lever anti-shortcut bundle; γ verdict (cross-domain collapsed); triggered viso bucket-gap audit |
| [P14](packets/P14.md) | 2026-04-29 → 2026-04-30 | 🟢 | **Sister variant `face_scale_jitter@0.50 ISOLATED` wins packet at value_composite=0.661**; bundle (P14_FT) + DATA_FIX both far below; bundle was net-negative |
| [P15](packets/P15.md) | 2026-04-29 → 2026-04-30 | 🟡 | First-ever GRL infrastructure run; β verdict; structurally working but weaker than jitter-isolated; GRL+jitter@0.50 ramped-λ is the natural follow-up |
| [PA](packets/PA.md) | 2026-05-04 → 2026-05-05 | ⚠️ | Single-lever data-axis test (visomaster_enhanced + teams_enhanced at fw=4.0); F4 v2 viso lift 72.36% beat E2B + P8A; **walkback 2026-05-05**: does NOT generalize to HDTF (7.87% vs P8A 93.57%) — v2-substrate-bound |
| [PC](packets/PC.md) | 2026-05-04 → 2026-05-05 | ❌ | Codec aug + data lever stack on E2B; codec aug HURTS viso recall 35-50pp on F4 vs PA; codec lever empirically refuted |
| [PD](packets/PD.md) | 2026-05-05 → 2026-05-06 | 🟡 | First explicit-form anti-shortcut loss (`correlation_penalty` λ=1.0 on 3 pixel-derivable axes); two arms (deeplive ship + viso ship) FT-from-E2B + single-lever discipline; training complete; scorecard pending |
| [P1](packets/P1.md) | 2026-05-06 → 2026-05-07 | 🟡 | PE_PAIR_RANK_DRO: pair_rank λ=0.2 + multi-axis GroupDRO with chronic_flag (BUNDLE) vs pair_rank-only (PAIRRANK). F4 PASS + F5 PASS for BUNDLE arm (PC_Generator FPR 62.9% → 0%); F1 fails at contract τ but is reachable at non-contract τ (BUNDLE_step500 96.5% lockbox recall @ FPR≤10%). Roy_D regression (29% → 78-93%) shared between both arms; refutes initial GroupDRO-balloon hypothesis. First eval folder under FACTS/OPINIONS contract. |

### Cross-cutting threads

- [Processing-signature shortcut](threads/processing_signature_shortcut.md) — the same-person-different-camera flip (slot-07 → 2-camera test → RLP7)
- [Promotion contract evolution](threads/promotion_contract_evolution.md) — `value_composite` is not deployment-grade; dev-calibrated τ + lockbox is
- [Preprocessing parity bug](threads/preprocessing_parity_bug.md) — commit `855871e`; only RLP6_04 rescored post-fix so far
- [Calibration vs training-aug](threads/calibration_vs_training_aug.md) — WS-P1 30.1% → training-aug dominant, calibration complementary
- [Gate alignment story](threads/gate_alignment_story.md) — real-pool-only `worst_pool_fpr`; avspeech out; WT-A root artifact
- [Face-pixel-area label leak](threads/face_size_label_leak.md) — each fake method clusters at a tight face-size band; dual-role training-data leak + deployment-domain question; **2026-04-30 update: jitter@0.50 alone licensed empirical lift; close criterion direct measurement queued**
- [Viso train/eval bucket gap](threads/viso_bucket_gap.md) — visomaster_enhanced_macro_dev resolves to a Teams-recaptured GAN-enhanced bucket absent from training; **2026-04-30 update: DATA_FIX as deployed empirically failed; loop closed-as-superseded**
- [Identity audit](threads/identity_audit.md) — refutes "more identity diversity" hypothesis; 429 viso + 1916 deeplive + 2300 unused; corollary actions live in viso_bucket_gap thread
- [Clean ↔ Teams identity pairing](threads/clean_teams_identity_pairing.md) — paired-transport structure that makes Layer-2 closure structurally tractable; opens contrastive-loss design lever
- [Contract policy bug](threads/contract_policy_bug.md) — flagship thread for the wiki's whack-a-mole prevention; v3 fix is uncommitted in working tree
- [In_proj-SVD silent zero-gradient](threads/in_proj_svd_gradient_bug.md) — pre-`2feea58` every R12g/RLP/P-* with apply_svd_to_in_proj=true had q/k/v residuals receiving zero classification gradient; affects P8A breakthrough attribution
- [Anti-shortcut bundle decomposition](threads/anti_shortcut_bundle_decomposition.md) — **NEW 2026-04-30** — discipline rule that drops out of P14: future stacked-lever packets must include a single-lever ablation slot
- [Correlation-penalty loss](threads/correlation_penalty_loss.md) — **NEW 2026-05-06** — first explicit-form anti-shortcut loss class on R13 (`lambda * sum_axes |Pearson_batch(score, axis)|` over pixel-derivable nuisance axes); home thread for [PD](packets/PD.md); training complete on both arms (deeplive ship + viso ship), scorecard pending
- [Eval-vs-production crop-tightness gap](threads/eval_production_crop_tightness_gap.md) — 2026-04-29 audit; structurally upstream of multiple shortcut/FPR investigations
- [Sharpness metric bug](threads/sharpness_metric_bug.md) — `sharpness_laplacian` computed on full image not face crop; downstream FPR-by-quartile reports partially confounded
- [Eval substrate data hygiene](threads/eval_substrate_data_hygiene.md) — is_no_face slice (n=219) data degeneracy + 99×110-pixel source crops below any reasonable production resolution floor
- [Webcam FPR dominance](threads/webcam_fpr_dominance.md) — modern_v2 filter takes lockbox FPR 4.59% → 0.71%; dor_shkedi-skewed (75% of v2 reals)
- [Wandb-flattening](threads/wandb_flattening.md) — recurring "added-but-not-firing" bug class; structural fix in commit `c366026` + train_sweep allowlist + regression test
- [Value composite semantics](threads/value_composite_semantics.md) — trainer-side directional metric; not deployment-grade per memory `project_promotion_contract.md`
- [Identity split mode](threads/identity_split_mode.md) — `shuffle → hash_stable` migration; quantification of the pre-RLP2 split-mode delta still open

### Templates

- [`packet_template.md`](packet_template.md) — the structure every per-packet retro under `packets/` follows. Updated 2026-05-07 with required `### Factual evidence` + `### In-session opinion` subsections under `## Conclusions drawn in-session`.
- [`thread_template.md`](thread_template.md) — the structure every cross-cutting thread under `threads/` follows.
- [`eval_folder_template.md`](eval_folder_template.md) — **NEW 2026-05-07** — canonical layout for `analysis/<packet>_eval_<date>/`. Codifies the FACTS/OPINIONS file-level split: factual docs (`*_FACTS_<date>.md`) carry numbers, the single `AGENT_PROPOSAL_<date>.md` carries opinion. Mid-session self-correction protocol (rename buggy CSVs, append to Self-correction log) is non-negotiable. The first eval folder following this contract is `analysis/p1_pe_eval_2026-05-07/`.

### Eval-folder pattern (2026-05-07+)

Each packet's evaluation produces a dedicated `analysis/<packet>_eval_<date>/` folder. The contract (see `eval_folder_template.md`) enforces a file-level split:

- **`*_FACTS_<date>.md`** — factual docs. Numbers, tables, no interpretation. New agents read these first.
- **`AGENT_PROPOSAL_<date>.md`** — the single OPINION doc. Stakes a position. Includes a "Self-correction log" naming any retracted framings.
- **Sub-investigation subdirs** — per-topic (e.g., `f3_color_b_dev/`) with their own `RESULTS_<topic>_FACTS_<date>.md` + scripts + CSVs.

A new agent forming an independent view reads only the FACTS docs (and the packet retro's `### Factual evidence` subsection). The AGENT_PROPOSAL is read second, after the user authorizes — see "Reading order for forming an independent view" in [`AGENTS.md`](AGENTS.md).

## Glossary

Short, for cold readers. Longer definitions live in the threads they link.

- **RLP / RLP*N*** — R13-relaunch packet *N*. Training slates launched sequentially (RLP1 → RLP7) since 2026-04-19. RLP3.5 and RLP6B are side-branches.
- **WT-A…F** — April-17 parallel-**w**ork**t**ree policy-reset work (data policy, split, augmentation, decision layer, promotion contract, `proper_data` schema). See [WT-infrastructure](packets/WT_infrastructure.md).
- **WS-P0/P1/P2.b/P4.a** — 2026-04-24 **w**ork**s**tream **p**robes: preprocessing-parity fix, per-camera calibration, per-identity reducer, Teams spatial-aug wiring. See [WS probes](packets/WS_probes.md).
- **τ (tau)** — decision threshold on `prob_fake`. "Lexicographic τ" = τ picked by the promotion contract to satisfy a worst-pool FPR budget, then sorted on a ranked tuple of metrics.
- **FPR / TPR** — false / true positive rate on a pool (FPR = real classified as fake; TPR = fake correctly classified). "Real-pool FPR" and "fake-pool TPR" are the two halves the contract balances.
- **`value_composite`** — RLP3 deployment-aligned metric: `0.6 * teams_fakes_tpr + 0.3 * other_fakes_tpr + 0.1 * stability`, evaluated at the τ meeting `mean_FPR ≤ 0.02, max_FPR ≤ 0.04` across 6 real pools. In-trainer readout; **not** deployment-grade by itself ([promotion contract evolution](threads/promotion_contract_evolution.md)).
- **`worst_pool_fpr`** — max FPR across real pools in the gate. **Real-pool-only** by construction (`trainer/trainer.py:161-216`); fake pools cannot drive it. Discovering this reframed the RLP5 "representation ceiling" as a gate-alignment artifact ([gate alignment story](threads/gate_alignment_story.md)).
- **Lockbox** — held-out partition (blake2b video-id 90/10 per OOD pool; RLP3 A10) used as the deployment-readiness readout. Authority of record per user memory `project_promotion_contract.md`.
- **`teams_ood_fake` / `teams_ood_real`** — deployment-distribution pools for Microsoft Teams. "Deployment target" in this repo = Teams-passed reals + Teams-method fakes.
- **Proper-data** — the WT-F HDTF/quickclips schema with explicit `include_lanes` control. "Unenhanced proper" = baseline lane; "enhanced teams" / "enhanced clean" = VisoMaster-enhanced sub-lanes.
- **Shortcut (processing / camera / ISP signature)** — the detector learned a correlation between low-level processing artifacts (codec, chroma, DCT-HF geometry) and the "fake" label. Same person, different camera → score flips. See [processing signature shortcut](threads/processing_signature_shortcut.md).

## Reconciliation needed

Cross-packet contradictions and open loops noticed during synthesis.

- **RLP5 slot-07 vs slot-01 shortcut probing asymmetry.** Slot-07 beats slot-01 by +0.0045 (inside RLP3.5 ±0.005 seed-noise envelope). Only slot-07 became the retro-score / `dor_shkedi` subject. If slot-01 carries the same shortcut, the "breakthrough" interpretation weakens further, and the E3 recipe's composite lift is more shortcut-driven than currently recorded. Flagged explicitly in [RLP5.md](packets/RLP5.md) Retrospective; no slot-01 probe scheduled.
- **RLP6B slots 01 and 02 have identical `quality_domain_loss_weight: 0.4`** despite being named `quality_lambda_2x` and `quality_lambda_4x` respectively. Slot-01 is a genuine 2× against the RLP5 baseline (0.2 → 0.4); slot-02's numeric value should be 0.8 but is 0.4. Either the names are wrong or the yaml values are. RLP6B.md flags this; yamls should be corrected before any launch, or the branch folded into [RLP6.md](packets/RLP6.md) per that file's recommendation.
- **RLP7 seed collision.** All three of `RLP7_01/_02/_03` use seed 742; `RLP7_04=744`, `RLP7_05=745`. Seed 742 was also used in `RLP5_07` (the shortcut slot) — three RLP7 slots running on the seed most likely to replicate the slot-07 shortcut geometry. Intentional vs accidental is a user decision point; flagged in [RLP7.md](packets/RLP7.md) Configuration.
- **WS-P1 inputs are pre-fix.** The 30.1% calibration result was computed through the pre-INTER_LINEAR retro-score path. The [calibration_vs_training_aug](threads/calibration_vs_training_aug.md) thread uses this number as the founding datum. The anchor-pool Δ being −0.008 makes a full verdict flip unlikely, but a re-run on post-fix inputs is the obvious next probe.
- **RLP3 retro-score determinism drift (−0.0067).** Spec called for ≤1e-3 drift on sanity; actual is `retro 0.6030 vs training 0.6097`. Documented in `R13_RELAUNCH_PACKET3_RETRO_SCORE_RESULTS_2026-04-22.md:30-58` but never root-caused. Changes the error bars on any retro-scored cross-packet comparison; keep in open-loops until traced or declared acceptable.

### Resolved during synthesis

- **RLP4 "confirmed m=0.15" language** — audited across [RLP5.md](packets/RLP5.md), [RLP6.md](packets/RLP6.md), [RLP4.md](packets/RLP4.md); all three already correctly attribute the m=0.15 lever to RLP3.5 evidence, with RLP4 marked "confirmed by carry-forward, not direct evidence." No edits needed.
