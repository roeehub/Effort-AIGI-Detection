# State — current rolling snapshot

> **Last refreshed**: 2026-05-22 (Phase 1 CPU probes closed). All 3 Phase 1 close criteria resolved at $0 / ~14 min wall total (CPU-3 0.4s, CPU-2 5 min, CPU-1 8.4 min). CPU-1 gamma (viso saliency 72.6% in non-face patches but non-face recall = 0% at CLS τ → HEAD ALT deferred to Phase 4). CPU-2 beta (Slot A v2 mean(raw)=0.503 < mean(teams)=0.542, Δ=+0.039, Wilcoxon p=0.002, same direction as E2B 2026-05-04 → BACKBONE pivots to GroupDRO substrate-balanced). CPU-3 gamma (|cos(axis, false-flag-normal)| = 0.04-0.08 ≪ 0.3 alpha threshold met, but cos(P8A, SlotAv2) = 0.692 < 0.7 alpha threshold by 0.008 → both BACKBONE runs proceed). Net Phase 2 implications: HEAD experiment unchanged; BACKBONE-SlotAv2 + BACKBONE-T5C BOTH pivot from substrate-pair-orthogonal loss to GroupDRO substrate-balanced; HEAD ALT (dual-readout) deferred. Earlier 2026-05-22 face-pool work block retained below.
>
> #### 2026-05-22 — Phase 1 CPU probes closed (FACTS only)
>
> All 3 probes documented in `analysis/{viso_fake_signature_localization,pair_loss_slot_a_v2,per_ckpt_axis_decomposition}_2026-05-23/RESULTS_FACTS_2026-05-23.md` + sentinels `_cpu{1,2,3}_complete.json`. Reproducible scripts: `run_cpu_{1,2,3}_*.py` in each folder. Phase 1 master sentinel: `analysis/phase_1_cpu_probes_complete_2026-05-23.json`.
>
> **CPU-1 (Viso-fake signature localization, 8.4 min MPS)** — gamma. On 550 viso fakes + 545 deeplive fakes (control), Slot A v2 step3500 at the calibrated CLS τ=0.788 gives viso recall 0.184 / face-pool recall 0.067 / non-face-pool recall 0.000. 14×14 per-patch saliency map: viso non-face-region mass 72.6% (alpha condition 2 met; >70%), but viso non-face recall = 0.000 < face recall 0.067 + 0.05 (alpha condition 1 not met). Top-3 viso patches all at row 2 (top of frame), OUTSIDE the centered 7×7 face region. Deeplive control: row 6 col 11, row 4 col 2 (also non-face). HEAD ALT (dual-readout face ⊕ non-face → 1024-dim head) deferred to Phase 4 per gamma rule.
>
> **CPU-2 (Pair-loss re-verification on Slot A v2, 5.2 min CPU)** — beta. 275/275 paired viso fakes from the 2026-05-04 inventory. Slot A v2 step3500 CLS-pool: mean(raw)=0.5029 < mean(teams)=0.5419, Δ=+0.0390, Wilcoxon stat=14893 p=0.0020. Same direction as E2B step3200 (mean(raw)=0.086 < mean(teams)=0.172, p=0.002). At τ=0.20: target cohort (teams>τ AND raw≤τ) = 62 vs wrong_way cohort = 1 — strong asymmetry pointing toward teams→raw alignment, BUT decision rule keys on mean(raw) > mean(teams) which is not the observed sign. Per literal rule: pair-loss premise does not hold on Slot A v2 either. BACKBONE-SlotAv2 + BACKBONE-T5C pivot from substrate-pair-orthogonal to GroupDRO substrate-balanced (Phase 4 Fallback B).
>
> **CPU-3 (Per-ckpt axis vs anchor decomposition, 0.4s)** — gamma. Pairwise cosines between the 3 per-ckpt substrate axes: cos(P8A, SlotAv2) = +0.692, cos(P8A, T5C) = +0.726, cos(SlotAv2, T5C) = +0.924. Cross-encoder |cos(per-ckpt axis, false-flag-normal-frozen)| using 15 dor real dev normal_photo + 38 dor real lockbox webcam frames from D8 frozen-CLIP-L11 cache: P8A=0.081, SlotAv2=0.063, T5C=0.039 (alpha condition 1 met; <0.3). cos(P8A, SlotAv2) = 0.692 misses alpha condition 2 (>0.7) by 0.008. No beta trigger (max |cos| = 0.081 << 0.6). Per literal rule: gamma → both BACKBONE runs proceed.
>
> **Net Phase 2 design changes**:
>
> 1. HEAD experiment (face-pool head-only retrain): UNCHANGED. CPU-1 gamma means HEAD ALT (dual-readout) is conditionally deferred but the primary HEAD run proceeds as scheduled.
> 2. BACKBONE-SlotAv2 + BACKBONE-T5C: pivot loss family from substrate-pair-orthogonal (projection-MSE on per-ckpt axis) to GroupDRO substrate-balanced. Same 1,880 paired identity-frames; same yaml flag `combined_paired.substrate_pair_sampling`; replace `loss/substrate_pair_orthogonal.py` with GroupDRO accumulator. CPU-2 + CPU-3 both indicate the projection-MSE objective is not the binding lever: CPU-2 says the per-pair raw/teams asymmetry has the wrong sign for the originally-conceived pair-loss premise; CPU-3 says SlotAv2 and T5C substrate axes are 0.924 cosine-aligned (single axis problem; GroupDRO worst-group reweighting is the appropriate framing).
> 3. Master plan `Phase 2 — Experiment HEAD` decision gate: unchanged (`lockbox_real_fpr ≤ 0.016` etc.).
> 4. Master plan `Experiment BACKBONE-SlotAv2/T5C` decision gate: unchanged numerically but the loss family changes — re-derive smoke pass criteria for GroupDRO.
>
> ---
>
> #### 2026-05-22 — Phase 0 documentation pass + face-pool inference readout (FACTS only)
>
> Today's 7 commits on `teams-relaunch-root-2026-04-17`: `66f8f39` (A0.1 inventory: 1,880 pairs = HDTF 1,094 + quickclips 732 + viso_teams_enhanced 54; 1,826 fully-paired), `d338bfe` (Track B composite λ-tiebreak scorer code + 12 tests pass; opt-in CLI flag, default remains lex), `6021358` (A0.2 12-cell L0/L4/L8/L11 multi-layer geometry; gate verdict AMBIGUOUS at SlotAv2/L11 Δ=−0.0774, |kliep_proj_mean|=0.0135), `52c3c77` (operator picks λ=1.0; Slot A v2 step3500 rank-1 under composite at 0.331), `7833d03` (Fallback 1 Probes 1+2: trained encoders re-fit substrate axis at ~0.98 acc, cos(per-ckpt, frozen)≈0.04–0.10 ≈ 85–88° angular separation, per-pair direction projects 9–23× stronger on per-ckpt axis; Probe 2 face-pool raises cos_pair 0.87→0.96, shrinks |Δ| ~4×), `5f7c8c5` (face-pool 800-frame canary: lockbox recall @ FPR=10% +0.06, @ FPR=5% −0.04; Roy_D mean real-frame prob_fake −0.10pp; 5/6 other chronics rise), `fad721a` (face-pool 9-suite full scorecard on Slot A v2 step3500: Pareto-improves lockbox + deeplive_enhanced; visomaster_enhanced regresses −0.100; composite λ=1.0 0.331 → 0.249).
>
> **Deployment-candidate update**: Slot A v2 step3500 + face-pool inference readout + composite λ=1.0 tiebreak is the new ranked-1 deployment candidate on the standing 4-ckpt panel under the composite policy. Slot A v2 step3500 with CLS-pool inference remains ranked 1 by the same policy on the 4-ckpt panel (composite 0.331 < P8A 0.631 < T5C 0.368 < Slot A v2 step1500 tier-1 demoted). The face-pool readout improves Slot A v2's own composite score to 0.249. The muddled verdict caveat: visomaster_enhanced_macro_dev recall regresses 0.167 → 0.067 (−0.100) under face-pool inference; CPU-1 in Phase 1 will determine whether the viso fake signature is in non-face patches (structurally lost by face-pool) or in face patches with a head-calibration mismatch (recoverable by head retraining).
>
> **Phase 0 documentation deliverables landed today**:
>
> - Packet retro: [`packets/FACE_POOL_2026-05-22.md`](packets/FACE_POOL_2026-05-22.md) (this section's anchor).
> - 4 thread updates: `threads/clean_teams_identity_pairing.md` (DEBATE 2026-05-22 + new open loop `per-base-substrate-pair-cohort-math-untested`), `threads/viso_bucket_gap.md` (2026-05-22 update + new open loop `viso-fake-signature-non-face-vs-face-localization`), `threads/face_size_label_leak.md` (2026-05-22 update, no new loop), `threads/iq_shortcut_deconvolution_program_2026-05-08.md` (2026-05-22 update, no new loop).
> - 4 new memories at `~/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/`: `project_substrate_pair_geometry_2026-05-22.md`, `project_per_ckpt_substrate_axis_2026-05-22.md`, `project_face_pool_scorecard_pareto_2026-05-22.md`, `project_face_pool_viso_regression_mechanism_open_2026-05-22.md`.
> - 3 revised memories (dated 2026-05-22 append): `project_pair_loss_premise_refuted_2026-05-04.md` (reopen-trigger #2 fires for Slot A v2), `project_p17_trained_head_destroys_substrate_invariance.md` (per-ckpt substrate axis re-fit rule), `project_deployment_three_modes_slot_a_v2_2026-05-21.md` (face-pool $0 baseline note).
> - `OPEN_LOOPS.md` regenerated; 2 new structured open loops surfaced (`per-base-substrate-pair-cohort-math-untested`, `viso-fake-signature-non-face-vs-face-localization`).
> - Eval folders: `analysis/substrate_pair_geometry_2026-05-22/`, `analysis/face_pool_canary_2026-05-22/`, `analysis/face_pool_scorecard_2026-05-22/`, `analysis/contract_reframe_2026-05-22/` — all with `RESULTS_FACTS_<date>.md` + `AGENT_PROPOSAL_<date>.md` split.
>
> **Probe 1 load-bearing rule** (memorialized in `project_per_ckpt_substrate_axis_2026-05-22.md`): the frozen-CLIP-L11 KLIEP axis is the WRONG axis for substrate-pair work on a trained encoder. Always re-fit per ckpt. Held-out accuracy on per-ckpt re-fit was 0.9804 (P8A), 0.9836 (SlotAv2), 0.9836 (T5C). Angular separation cos = 0.04–0.10 from frozen-CLIP axis. The per-pair direction projects 9.0× (T5C), 10.1× (SlotAv2), 22.9× (P8A) more strongly on the per-ckpt axis. Any substrate-pair contrastive loss aligned to the frozen-CLIP direction targets a near-orthogonal axis on the trained encoder.
>
> **Open loops touched / opened this session**:
>
> - `per-base-substrate-pair-cohort-math-untested` (HIGH, NEW) — Slot A v2 became deployment candidate; reopen-trigger #2 from `pair-loss-asymmetric-variant-untested` (2026-05-04) fires. CPU-2 in Phase 1 re-runs Wilcoxon + cohort partition on Slot A v2's scores on the 275 paired viso fakes.
> - `viso-fake-signature-non-face-vs-face-localization` (HIGH, NEW) — face-pool regresses visomaster_enhanced_macro_dev recall by 10pp while lifting deeplive_enhanced_dev by 13pp on the same Teams-recapture eval bucket. CPU-1 closes via 14×14 per-patch ablation saliency map.
> - `cpu-probe-mechanism-discrimination` (HIGH, EXISTING) — adds a third data point to the compression-vs-invariance dichotomy: face-pool produces real-score compression (canary real-std 0.305 → 0.159) AND lockbox-fake recall lift (+0.079) on the same ckpt, distinct from Slot α's compression-with-recall-loss and from pure substrate invariance.
> - `pair-loss-asymmetric-variant-untested` (LOW, RESOLVED 2026-05-04) — remains resolved for E2B; the new `per-base-substrate-pair-cohort-math-untested` loop is the Slot A v2-specific successor.
>
> **Pending decisions (user-gated)**:
>
> 1. Phase 1 CPU-1/CPU-2/CPU-3 launches (all $0; ~4h wall total per master plan).
> 2. Phase 2 GPU budget authorization ($120–171; 3 parallel experiments HEAD + 2× BACKBONE).
> 3. Face-pool readout productionization (move the monkey-patch from `analysis/face_pool_canary_2026-05-22/score_canary_face_pool.py` into `detectors/effort_detector.py:198-222` behind a yaml flag `face_pool_readout`). Currently untracked working-tree code change.
>
> ---
>
> #### 2026-05-21 auto-mode — codec restoration triple launched (FACTS only)
>
> **Audit findings (Step 0, $0)**:
> - `teams_codec_sim_p=0.40 + teams_codec_sim_quality=[20, 65]` are P8A's exact augmentation keys (R13_RLP8_01_unfreeze_clip_codec.yaml lines 93-94). The keys are FLAT under augmentation block (not nested) and propagate via train_sweep.py's full-block reapply (line 207); no allowlist amendment required.
> - The actual transform fired by `teams_codec_sim_p>0` is **`VideoCodecSimulation`** (`data/augmentations/transforms.py:1216`) via `_build_teams_passthrough_pipeline` (`pipelines.py:1562-1570`), NOT `TeamsCodecSimulation` despite the yaml key name.
> - P22 (2026-05-02) yaml `R13_T3_SLOT1_DROP_HIGH_IQ_TEAMS_REALS_2026-05-09.yaml` line 10 comment confirms: "All other knobs identical: pipeline_randomization (P22 winner)" — the swap was deliberate, evaluated on dev fake_macro_recall at the time (3× lift). T3 → T4 → T5C → Slot A v2 all inherit P22's swap.
> - `loss/anchor_aware_penalty.py:77` shows `pool_names` is plural-capable (`pools = self.pool_names or [ANCHOR_POOL]`), but only 6 named pools exist in `analysis/teams_pool_rescore.py:44-51` POOLS list — none are Roy_D-specific. Roy_D anchor pool extension requires bucket-prefix + frames-upload + registry edit (multi-step, deferred).
>
> **Pre-launch CPU gates on Roy_D natural-experiment crop** (`analysis/codec_restoration_gates_2026-05-21/gates_summary.json`):
>
> | ckpt | baseline Roy_D | aug median (12 seeds, q=20-65) | guest ref | Gate 1 (validity ≤0.10) | Gate 3 (saturation swing ≤0.30) |
> |---|---:|---:|---:|---|---|
> | T5C step3500 | 0.7952 | **0.7239** | 0.6275 | **PASS** (gap 0.0964) | PASS (swing 0.0995) |
> | Slot A v2 step3500 | 0.7668 | 0.7481 | 0.5987 | FAIL (gap 0.1494) | PASS (swing 0.1435) |
>
> Gate 1 verdict on Slot A v2 FAILS by 5pp — but direction is correct on both (aug median drifts downward toward Guest). T5C reproduces the natural-experiment Δ nearly exactly with the same aug (median drift −0.071, vs natural Δ −0.168 = 42% of the way). The single-frame gate is a strict test; training applies aug to ~110K augmented frames over 3500 steps and the encoder learns invariance rather than score-matching individual frames. **Launch proceeded.**
>
> **3 packets launched 2026-05-21 ~00:23 UTC under image `1.3.295`** (Cloud Build `564298ac-c0ab-4ce9-aef8-08ecc5660722`, 1m 49s):
>
> | # | Slot | YAML | Region | Vertex Job ID | Image | Anchor | codec_p | FT base |
> |---|---|---|---|---|---|---|---:|---|
> | 1 | R13_T5C_ANCHOR_AWARE_PLUS_CODEC | `2026-05-21` | us-west4    | `5839421592722997248` | 1.3.295 | ON  | 0.40 | Slot A v2 step3500 |
> | 2 | R13_T5C_ANCHOR_AWARE_PLUS_CODEC_LIGHT | `2026-05-21` | us-east1    | `1506065191836581888` | 1.3.295 | ON  | 0.20 | Slot A v2 step3500 |
> | 3 (failed) | R13_T5C_CODEC_ONLY_NO_ANCHOR | `2026-05-21` | us-central1 | `5637646623317164032` | 1.3.295 | OFF | 0.40 | T5C step3500 |
> | 3 (relaunched) | R13_T5C_CODEC_ONLY_NO_ANCHOR | `2026-05-21` | us-central1 | `7516210617884082176` | 1.3.296 | OFF | 0.40 | T5C step3500 |
>
> Job submission times 00:23:47 / 00:24:02 / 00:24:15 UTC (sequential ~15s apart). Slot 3 original FAILED at 22:30 UTC with `FileNotFoundError: '/workspace/experiments/phase2_round13/R13_T5C_CODEC_ONLY_NO_ANCHOR_2026-05-21.yaml'` — image-currency race condition (yaml mtime fell between Cloud Build tarball capture at ~22:17:17 UTC and image push at ~22:19:50 UTC; check_image_currency.sh passed but yaml not in source tarball). New thread `threads/image_currency_check_race_condition_2026-05-21.md` documents close criteria. Rebuilt to `1.3.296` (Cloud Build `7e3c49a8-3e69-481a-b74b-83aeaebb3096`, 2m 28s) and relaunched Slot 3 as `7516210617884082176`. Monitor armed at `tasks/blbxkk5u3` (replacing prior `ba1ur5e2z` which carried the failed Slot 3 ID). W&B dashboard: https://wandb.ai/dtect-vision/phase2r13-experiments.
>
> **VERDICT 2026-05-21 ~03:30 UTC** (full FACTS at `analysis/codec_restoration_gates_2026-05-21/VERDICT_FACTS_2026-05-21.md`): All 3 SUCCEEDED. Slot 1 4h 16m, Slot 2 2h 35m, Slot 3v2 2h 54m. Codec aug bites the natural-experiment transport axis (37-60% Δ reduction) but NO packet meets the deployment-grade closure criterion (|Δ| ≤ 0.05; best closure Slot 2 at -0.067). All 3 regress may6 production-drift vs Slot A v2 base (Slot A v2 0.043 → packets 0.065-0.120; Slot 3 worst at 12% over the 10% criterion). Mechanism is asymmetric: Guest's score moves UP toward Roy_D's (encoder loses transport discriminator), not Roy_D moving DOWN. Decision-tree branch: NUANCED D (mechanism activity confirmed, single-lever insufficient). **Slot A v2 step3500 remains the best deployment ckpt.** 5 OPINION-only next-packet paths documented in VERDICT_FACTS §recommendations — user decision points reserved.
>
>
> 2×2 design (anchor × codec): rows below are Slot A v2 step3500 (anchor=ON, codec=0.0) and T5C step3500 (anchor=OFF, codec=0.0) as references; Slot 2 sits between Slot 1 and Slot A v2.
>
> **Hypothesis under test**: codec aug + anchor compose additively. Each addresses a structurally distinct failure mode (anchor → chronic-identity FP; codec → per-Teams-account transport shortcut). If both compose: Slot 1 ≥ Slot A v2 lockbox_fake_recall AND Slot 1 reduces 2026-05-19 natural-experiment Δ to ≤0.05. Otherwise: levers don't compose (similar to RESCHAIN_GRL6's anchor+GRL non-composition).
>
> **Falsifiers (close criterion)**: dev_fake_macro_recall regresses >−0.05 absolute vs Slot A v2 step3500 at any sampled ckpt → compression trap from heavy aug stack (pipeline_rand + codec layered); lockbox_real_fpr >+0.01 absolute → aug+anchor interaction breaks invariance; 2026-05-19 natural-experiment Δ on trained ckpt >0.30 → aug doesn't desensitize the targeted transport axis.
>
> **Confirmation (deployment grade)**: lockbox_fake_recall ≥0.688 (matches Slot A v2 step3500) at some ckpt AND lockbox_real_fpr ≤0.025 AND natural-experiment Δ ≤0.05 on trained ckpt.
>
> **Open loops touched this session**:
> - `lockbox-real-fpr-tiebreak-is-load-bearing` — NOT RESOLVED but DOWNWEIGHTED by Job B 95% CI evidence (covers 0); the program's directional drift is no longer constrained by the 0.07pp tiebreak. New direction: address natural-experiment Δ first; tiebreak fight resolves itself if a single ckpt beats both metrics simultaneously.
> - `silent-feature-failures-pattern` — REINFORCED by audit finding that P22's lever swap (`teams_codec_sim` → `pipeline_randomization`) was evaluated on dev fake_macro_recall only; the deployment-pipeline-shortcut cost was invisible for 19 days until the 2026-05-19 natural experiment. Pattern: load-bearing-replacement-evaluated-on-the-wrong-metric.
>
> ---
>
> #### 2026-05-20 evening — 7-job CPU evidence batch for Slot A v2 deployment (FACTS only)
>
> #### 2026-05-20 evening — 7-job CPU evidence batch for Slot A v2 deployment (FACTS only)
>
> Eval folder `analysis/cpu_jobs_slot_a_v2_deployment_2026-05-20/` with top-level `RESULTS_FACTS_2026-05-20.md` (navigator) + per-job FACTS docs + single `AGENT_PROPOSAL_2026-05-20.md` OPINIONS doc.
>
> | Job | Question | Headline | Verdict |
> |---|---|---|---|
> | A | Slot A v2 on Roy_D/Guest crops | Δ = **−0.168** identical to T5C; flip window WIDER ([0.60, 0.70)) | Anchor mechanism vertical-shift only; transport-axis unchanged |
> | B | Paired bootstrap CI on lockbox FPR gap | 95% CI on Δ = **[−0.008, +0.010]**; P=0.519 | Tiebreak inside sampling noise; literal 1-video gap (25 vs 26 FPs / 1361) |
> | C | Tiebreak policy rerank | Lex policies: P8A wins; composite λ ≥ 5: SlotAv2 wins; Pareto: SlotAv2 | Verdict is policy-fragile |
> | D | Per-identity FPR at calibrated τ | Roy_D **0.30 → 0.84 (+0.54)**; 4 chronics fixed (Chikara/PCGen/Q/dor) | Largest regression is dev-only (Roy_D NOT in contract lockbox) |
> | E | τ-sweep operating frontier | At dev_fpr ≤ 0.10: SlotAv2 lockbox_recall **0.60** vs T5C 0.40; P8A unreachable on chronic-heavy panel | Anchor mechanism enables lower-τ operating point |
> | F | may6 production-drift | Slot A v2 **4/92** (P8A 0/92 sanity-preserved within MPS noise; E2B 53/92) | Bar 1 (P8A parity ≤ 5) MET; bar 4 (p50 ≤ 0.05) NOT MET at 0.097 |
> | G | Per-axis perturbation | G_scale swing 0.698 (≈ T5C 0.678); saturation swing **4.93× T5C** | Bar 3 amplification TRIGGERED on saturation |
> | Move 3 | Canary + LoRA-load test files | **16/16 pass** in 5.05s | Both fixes ready for commit |
>
> **Open loops touched**:
> - `lockbox-real-fpr-tiebreak-is-load-bearing` (HIGH, from RESCHAIN_GRL6 retro) — RESOLVABLE per Job B: 95% CI on the 0.07pp gap covers 0 with CI width 25× the observed gap; P(SlotAv2 truly higher) = 0.519. The tiebreak is statistically indistinguishable from "tied"; the contract's current verdict is not load-bearing on this delta.
> - `canary-silence-when-multi-axis-grl-active` (HIGH, opened earlier 2026-05-20) — RESOLVABLE per Move 3 test verification (3/3 tests pass).
> - `lora-enabled-not-propagated-by-load-model` (HIGH, opened earlier 2026-05-20) — RESOLVABLE per Move 3 test verification (3/3 ckpt-roundtrip + 10/10 adapter tests pass).
> - `roy-d-specific-anchor-pool-packet` (HIGH, from 2026-05-16 auto-mode verdict) — REMAINS OPEN; Job D confirms the regression on the contract substrate (Roy_D 0.30 → 0.84 at τ_cal); the predicted Roy_D-anchor-pool extension is still the proposed next intervention.
>
> **Direction synthesis (OPINION at `AGENT_PROPOSAL_2026-05-20.md` §6-8)**: Slot A v2 step3500 is a strict improvement on chronic-identity FP suppression at the contract surface; it does NOT address the Teams-transport per-frame shortcut (the 2026-05-19 natural-experiment failure mode persists with identical Δ); it introduces a +0.54 Roy_D dev FPR regression that is invisible to the contract's `lockbox_real_fpr` tiebreak. Deployment decision rests on 3 user-gated questions:
> 1. Does Roy_D generalize to a production user class? (→ multi-account capture sweep, $0 + user time)
> 2. Is the within-session Teams-account flip operationally tolerable? (→ SLA question)
> 3. What's the operational FP/FN cost ratio λ? (→ if λ ≥ 5, composite tiebreak picks SlotAv2)
>
> **Pending decisions (user-gated)**:
> 1. Tiebreak amendment in `arena/score_teams_promotion_contract.py` (composite vs lex; $0 yaml change, reversible).
> 2. Roy_D-anchor-pool packet authorization (medium GPU, $30-50).
> 3. Multi-account capture sweep (real-world capture task on user's side; CPU analysis takes ~30 min after capture).
> 4. Commit policy for the 2 Move 3 fixes (single bundled PR vs split).
>
> ---
>
> #### 2026-05-20 — T5C_TRIPLE 3-packet batch + Slot A v2 29-suite validation (FACTS only)
>
> Three forks on T5C step3500 base launched 2026-05-20 ~08:50 UTC under image `1.3.294` (Cloud Build `735984aa-db5a-4372-8425-dcfbbd807bf7`, 2m 3s). All 3 SUCCEEDED by 13:03 UTC; companion 4-ckpt Slot A v2 29-suite scorecard SUCCEEDED 12:55 UTC.
>
> | # | Slot | W&B run | Vertex job | Region | Runtime | Terminal step |
> |---|---|---|---|---|---:|---|
> | 1 | R13_T5C_6AXIS_PLUS_ANCHOR (Slot β 6-axis GRL + anchor_aware) | `afydiase` | `2752204048160522240` | us-west4 | 4h 13m | `periodic_step3500` |
> | 2 | R13_LORA_T5C_L8_L9_R8 (LoRA r=8, layers [8,9]) | `3yo9d1f3` | `5302124688686710784` | us-east1 | 2h 30m | `periodic_step5000` |
> | 3 | R13_T5C_5AXIS_NOLUMA (Slot β 6-axis minus luma) | `9p1zo42l` | `7849248291890921472` | us-central1 | 1h 59m | `periodic_step3500` |
>
> **29-suite scorecard verdict** (job `146629015254335488`, 4 ckpts × 29 suites, output GCS prefix `slot-a-v2-validation-2026-05-20/`):
>
> | rank | ckpt | dev_real_fpr | dev_stress_fpr | dev_fake_macro | lockbox_real_fpr | lockbox_fake_recall | viso_enh | deeplive_enh |
> |---:|---|---:|---:|---:|---:|---:|---:|---:|
> | 1 | P8A_REFERENCE_STEP5000 | 0.070 | 0.069 | 0.300 | **0.0184** | 0.387 | 0.136 | 0.239 |
> | 2 | SLOT_A_ANCHOR_AWARE_STEP3500 | 0.066 | 0.099 | 0.438 | 0.0191 | **0.688** | 0.167 | **0.552** |
> | 3 | T5C_PERIODIC_STEP3500 | 0.065 | 0.099 | 0.459 | 0.0279 | 0.660 | 0.138 | 0.626 |
> | 4 | SLOT_A_ANCHOR_AWARE_STEP1500 | 0.065 | 0.099 | **0.207** ❌ | 0.0044 | 0.640 | 0.029 | 0.134 |
>
> The 29-suite reproduces the 9-suite auto-mode-scorecard 2026-05-16 ckpt-level metrics for Slot A v2 step3500 IDENTICALLY (lockbox_real_fpr 0.0191, lockbox_fake_recall 0.6878, dev_fake_macro_recall 0.4381). P8A's rank-1 position over Slot A v2 step3500 remains decided by the 0.07pp `lockbox_real_fpr` tiebreak; +30pp lockbox_fake_recall gain is preserved.
>
> **Tonight's 3 new ckpts (Slot 1, 2, 3) are NOT in this scorecard**. They are evaluated only via the off-line manual canary (see below) because the in-training canary probe silently failed on Slots 1+3.
>
> **Manual canary on 8 ckpts** (eval folder `analysis/manual_canary_2026-05-20/`, 281 LOC script replicating `trainer/mixins/canary_probe.py:_aggregate_metrics`, 800-frame parquet, Apple MPS local, ~37 sec per ckpt):
>
> | metric (↑ unless ↓ noted) | P8A | T5C | **SlotA.3500** | S1.1500 | S1.2500 | **S1.3500** | S3.3500 |
> |---|---:|---:|---:|---:|---:|---:|---:|
> | lockbox_R @ FPR=5% | 0.23 | 0.19 | **0.59** | 0.18 | 0.23 | **0.17** | 0.17 |
> | lockbox_R @ FPR=10% | 0.25 | 0.50 | **0.61** | 0.53 | 0.39 | **0.30** | 0.31 |
> | viso_R @ τ=0.5 | 0.34 | 0.86 | 0.66 | 0.80 | 0.74 | **0.56** | 0.82 |
> | mean_per_identity (↓) | 0.64 | 0.61 | **0.53** | 0.56 | 0.58 | 0.56 | 0.63 |
>
> Slot 1 trajectory: `lockbox_R @ FPR=10%` 0.53 → 0.39 → 0.30 over steps 1500/2500/3500 (monotonic DOWN). Slot 1 step 3500 vs Slot A v2 step 3500: 3.5× worse on lockbox_R @ FPR=5% (0.17 vs 0.59). Slot 3 ≈ Slot 1 on lockbox metric (luma axis drop did not change the outcome). Slot 2 manual canary numbers are NOT faithful — `load_model` doesn't propagate `lora.enabled` from ckpt's `model_config`; refer to W&B in-training canary `lockbox_recall_at_FPR_5pct=0.25` instead.
>
> #### Open-loop status changes this session
>
> Two HIGH-severity open loops added:
>
> 1. `canary-silence-when-multi-axis-grl-active` (`threads/in_training_canary_signal.md`) — in-training canary probe silently disabled when `multi_axis_grl.enabled: true`; lost 4h 13m + 1h 59m of mid-training deployment-shaped visibility on Slots 1+3. Workaround: off-line replication via `analysis/manual_canary_2026-05-20/score_canary.py`.
> 2. `lora-enabled-not-propagated-by-load-model` (`threads/wandb_yaml_propagation_bugs.md`) — fifth instance of the wandb-side-surface-hygiene bug class. `batch_inference_gcs.load_model` does not propagate `lora.enabled` from the ckpt's `model_config`, so LoRA-trained ckpts run as degraded non-LoRA models in off-line scorecards/analyses.
>
> #### Direction decision (user, 2026-05-20)
>
> User is **pivoting away from the stacking-on-top-of-anchor_aware direction**. The 3-packet batch refutes the additive-composition hypothesis (Slot 1: GRL pressure costs more than anchor wins) and the luma-axis isolation hypothesis (Slot 3 ≈ Slot 1 on lockbox metric). Slot A v2 step3500 remains the operative ckpt; next packet should be structurally distinct from "stack levers on top of anchor_aware." Candidates documented in [`AGENT_PROPOSAL_2026-05-20.md §4`](../../analysis/manual_canary_2026-05-20/AGENT_PROPOSAL_2026-05-20.md) ranked by structural distinctness: Roy_D-targeted anchor pool extension; output-preservation aux loss for LoRA; tiebreak amendment + ship Slot A v2; distillation-from-anchor-aware student; per-substrate τ-calibration at deployment.
>
> #### Pending decisions (user-gated)
>
> 1. Direction selection from OPINION §4 candidate list.
> 2. Canary-silence fix (HIGH-severity infra debt; blocks all future GRL-stacked packets from in-flight monitoring).
> 3. LoRA-load fix (HIGH-severity infra debt; blocks Slot 2-class re-evaluations).
> 4. Tiebreak amendment decision (yaml change in `arena/score_teams_promotion_contract.py`) — same decision as carried from RESCHAIN_GRL6 retro.
>
> ---
>
> #### 2026-05-13 overnight batch — training side (FACTS only)
>
> Four single-lever slots launched 2026-05-13 ~00:30 UTC, all `JOB_STATE_SUCCEEDED` by 07:37 UTC:
>
> | # | Slot | W&B run | Vertex job | Region | Runtime | Terminal step |
> |---|---|---|---|---|---:|---|
> | 1 | R13_LORA_L10_L11 (LoRA r=16 α=32 @ resblocks 10-11, frozen P8A_step5000 base) | `gf6l06rf` | `3547787915872436224` | us-east1 (after us-west4 reroute) | 1h 58m | `periodic_step5000` (12 ckpts) |
> | 2 | R13_LORA_T5C_L10_L11 (same LoRA recipe, T5C step3500 base) | `912kd88q` | `2103821285346770944` | us-east1 | 2h 04m | `periodic_step3500` (11 ckpts; early-stopped) |
> | 3 | R13_SLOT3_T5C_PLUS_JITTER030 (T5C step3500 base + `face_scale_jitter scale_limit=0.30`; multi-axis-L11-GRL `hidden_dim=1024` inherited) | `502dcznh` | `5691671127147937792` | us-central1 | 3h 03m | `periodic_step5000` (12 ckpts) |
> | 4 | R13_SLOT4_B16_SCRATCH_FOURIER (B16-scratch + `fourier_aug` bands 8-13 randomize, 5-6 preserve, phase preserved, `p_apply=0.5`) | `qrpf5dtr` | `5178096134941310976` | us-west4 | 6h 45m | `top_n_step10000` (17 ckpts; ran past nominal cap) |
>
> Region reroute: original Slot 1 us-west4 submission `3052397110822436864` cancelled 2026-05-13T01:30:03Z after PENDING >30 min, per CLAUDE.md region-capacity rule.
>
> Combined scorecard `r13-overnight-scorecard-2026-05-13` (Vertex `1806724447428673536`, us-east1, image `1.3.285`): 12 candidate ckpts (3 per slot, picked at early / mid / peak operating points per [`arena/checkpoint_maps/teams_target_domain.r13_overnight_2026-05-13.yaml`](../../arena/checkpoint_maps/teams_target_domain.r13_overnight_2026-05-13.yaml)) + 3 anchors (P8A_REFERENCE_STEP5000, E2B_TOP_N_STEP3200, T5C_PERIODIC_STEP3500). Suite manifest `arena/target_domain_suites.teams_promotion_contract_2026-04-23_with_dor.yaml` (29-suite); standing v3-fix policy (target_real_fpr=0.07, target_stress_fpr=0.10, target_fake_recall_min=0.30). Output GCS prefix: `gs://training-job-outputs/test_results/teams_promotion_contract/r13-overnight-scorecard-2026-05-13/`. Verdict pending. Packet retro [`packets/U_SLOTS_2026-05-13.md`](packets/U_SLOTS_2026-05-13.md).
>
> ---
>
> #### 2026-05-12 late evening (10 CPU diagnostics complete: D1-D6 + train-overlap + GCS audit + D7-D10)
>
> **FACTS docs authored 2026-05-12** (numerical findings only; interpretations live in OPINIONS docs — see [§OPINIONS pointer](#opinions-docs-this-session) below):
>
> | File | Headline numbers |
> |---|---|
> | [`d1_p8a_iq_alignment/D1_FACTS_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_d1_p8a_iq_alignment/D1_FACTS_2026-05-12.md) | P8A chronic-6 frame-level R²=0.350 vs healthy 0.163; chronic dominant axis `saturation_mean` β=+4.27; per-identity within chronic-6 R² range 0.18-0.76; saturation_mean β sign-flips Roy_D −6.15 vs dor_shkedi +3.33 |
> | [`d2_encoder_separation/D2_FACTS_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_d2_encoder_separation/D2_FACTS_2026-05-12.md) | CLIP-frozen L11 chronic-6 5-fold CV AUC=1.000 (n=282, 41 fakes); D2 single-split chronic-6 IQ-PC1 angles: CLIP_FROZEN 83.68°, P8A 78.54°, E2B 74.95°, T5C 75.19°, T3 77.09°; D2 §7 caveats theoretical 768-d random baseline 87.93° as not empirically validated; smallest per-axis chronic-6 angles on `color_a_dev` / `saturation_mean` for FT'd ckpts |
> | [`d3_cross_substrate_probe/D3_FACTS_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_d3_cross_substrate_probe/D3_FACTS_2026-05-12.md) | DEV→LOCKBOX transfer AUC: P8A 0.594, E2B 0.667, T5C 0.747, T3 0.668; LOCKBOX→DEV: P8A 0.893, E2B 0.951, T5C 0.825, T3 0.948; in-sample 5-fold CV ≥ 0.999 every cell; 0/4 ckpts pass AUC>0.95 both directions; lockbox slice n=87 |
> | [`d4_iq_quartile_alignment/D4_FACTS_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_d4_iq_quartile_alignment/D4_FACTS_2026-05-12.md) | Per-IQ-quartile dev↔lockbox FPR alignment r (mean across 6 axes): P8A +0.716 \|Δ\|=0.026; T5C +0.508 \|Δ\|=0.050; T3 −0.032 \|Δ\|=0.046. At dev-cal 5% τ: lockbox FPR = 0.92% (P8A) / 2.40% (T5C) / 2.19% (T3) |
> | [`d5_per_identity_iq/D5_FACTS_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_d5_per_identity_iq/D5_FACTS_2026-05-12.md) | Per-identity output mean-logit vs mean-IQ joint R² (n=11 identities ≥30 frames; dof=4): P8A 0.222, T3 0.490, T5C 0.598. Residual-sign: 5/5 chronic_6 above and 5/6 non-chronic below predicted line for all 3 ckpts; `real_dor` / `Cam_Test` / `PC_Generator` recur top-residual across ckpts |
> | [`d6_empirical_orthogonality/D6_FACTS_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_d6_empirical_orthogonality/D6_FACTS_2026-05-12.md) | **B=50 final** (run 2026-05-12 18:08–19:12; 3736 s). Bootstrap-mean chronic-6 angles vs CLIP-frozen empirical baseline 79.53°: P8A 74.47° (Δ 5.06°), T3 71.98° (Δ 7.55°), T5C 69.99° (Δ 9.54°), E2B 69.08° (Δ 10.45°). D2's single-split chronic-6 cells biased +2.12σ to +3.16σ above bootstrap-mean uniformly. Full-cohort + non-chronic bootstrap-means within ±2° of 87.93° random baseline (no drift outside chronic-6). |
> | [`train_overlap_audit_2026-05-12/TRAIN_OVERLAP_FACTS_2026-05-12.md`](../../analysis/train_overlap_audit_2026-05-12/TRAIN_OVERLAP_FACTS_2026-05-12.md) | 3 D5 residual identities (`real_dor`, `Cam_Test`, `PC_Generator`) have eval frames in bucket `teams-faces-data-test-2914-fake-4420-real-feb-28`; 0 references in 158 R13 training yamls; P8A's Teams training source is bucket `live-deepfake-methods-real-and-fake-frames-cropped-teams-v2` |
> | [`gcs_identity_audit/GCS_IDENTITY_AUDIT_FACTS_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_gcs_identity_audit/GCS_IDENTITY_AUDIT_FACTS_2026-05-12.md) | 1646 sample listings + 250 manifests in `teams-v2`; 0 matches for 23 identity-name patterns at either listing or manifest level; manifest schema lacks identity-style fields; source-video tokens are YouTube-ID-anonymized (e.g., `lm0hNQmOdFg`) |
> | [`d7_fpr_decomposition/D7_FACTS_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_d7_fpr_decomposition/D7_FACTS_2026-05-12.md) | Panel n=3,416 eval reals. P8A FPR 3.31%, T5C 5.77% at dev-cal 5% τ. Block-drop ΔR² same rank order for both ckpts: IQ (6 axes) > chronic_6 > substrate_distance. P8A: IQ +0.2359 / chronic +0.0137 / substrate +0.0146. T5C: IQ +0.3143 / chronic +0.0625 / substrate +0.0001. Substrate-only AUC 0.575 (P8A) / 0.527 (T5C). Chronic vs non-chronic FPR ratio P8A 4.68× / T5C 8.66×. P8A `substrate_distance` β = −0.41; T5C β = +0.06 |
> | [`d8_head_retrain_substrate_balanced/D8_FACTS_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_d8_head_retrain_substrate_balanced/D8_FACTS_2026-05-12.md) | DEV→LOCKBOX probe transfer AUC on P8A frozen-encoder features: unweighted baseline 0.587 [0.550, 0.620]; PCA32-KDE 0.561; PCA8-KDE 0.667 [disjoint CI]; **classifier-KLIEP 0.679 [disjoint CI, Δ +0.093]**. KLIEP discriminator achieves 99.09% accuracy distinguishing dev-real from lockbox-real in CLIP-frozen feature space. Offline-LR-head lockbox FPR at dev-cal 5% τ is 66-72% across all 4 heads (P8A's in-loop production head is 0.92% per D4 — head-construction is structurally different) |
> | [`d9_source_substrate_inventory/D9_FACTS_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_d9_source_substrate_inventory/D9_FACTS_2026-05-12.md) | 195 R13 yamls; 191/195 (97.9%) enable teams lane; 102/195 (52.3%) enable proper_data. Cross-tab on real frames: train.teams-v2 100% youtube_origin / 0% direct_teams; train.proper_data 0% direct_teams (HDTF 59.9% + QCLIPS 40.1%); dev 100% direct_teams; lockbox 100% direct_teams. teams-v2 `source` field uniformly `"teams_capture"` (describes pipeline applied, not origin) — coexisting with YouTube-ID `original_trimmed_video_name`. Zero shared `source_kind` values between teams-v2 manifest and eval manifest |
> | [`d10_training_pool_position/D10_FACTS_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_d10_training_pool_position/D10_FACTS_2026-05-12.md) | KLIEP discriminator re-fit (balanced acc 0.9909 matches D8). Mean projection: dev_real (n=2000) −3.162, train_real (n=371) −2.401, lockbox_real (n=414) +3.169. Decision-boundary split: 87.3% train_real proj<0 (dev side), 99.4% dev_real proj<0, 2.4% lockbox_real proj<0. Wasserstein-1: W(train, dev) 0.769; W(train, lockbox) 5.570 (7.25×); W(dev, lockbox) 6.332. KLIEP discriminator axis vs IQ-PC1 angle = 89.71° (cos 0.0051); robust 89.26°-89.71° across 4 normalization variants. Bootstrap-stable train_mean: −2.401 ± 0.036 (CV 1.49%), sign preserved 20/20 |
>
> #### OPINIONS docs this session
>
> Read these AFTER forming a Pass-1 view (per `AGENTS.md` Reading order for forming an independent view). The user has flagged past framings as retracted; this session retracted 2 more (see `D1_D5_OPINIONS_2026-05-12.md` §4 + `D1_D5_CRITIC_REVIEW_2026-05-12.md` §3.2).
>
> - [`d1_d5_synthesis/D1_D5_OPINIONS_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_d1_d5_synthesis/D1_D5_OPINIONS_2026-05-12.md) — planning-agent synthesis of D1-D5.
> - [`d1_d5_synthesis/D1_D5_CRITIC_REVIEW_2026-05-12.md`](../../analysis/cpu_diagnostics_2026-05-12_d1_d5_synthesis/D1_D5_CRITIC_REVIEW_2026-05-12.md) — independent critic review (disagrees on 3 of 7 tagged claims; restored L11 anchor / 5-identity cohort; flagged data-ingestion slot as planning-agent omission).
>
> #### LoRA infrastructure 2026-05-12 (in-repo, NOT LAUNCHED)
>
> Separate agent delivered per [`LORA_LAYERS_10_11_TASK_2026-05-12.md`](archive/LORA_LAYERS_10_11_TASK_2026-05-12.md). Module `detectors/lora_adapter.py` (wraps `attn.{in_proj,out_proj}` + `mlp.{c_fc,c_proj}` on selected resblocks; stacks on frozen SVD). New `lora` param group in `choose_optimizer` (`utils/setup.py`) with `lora_lr_mult` knob. Yaml `experiments/phase2_round13/R13_LORA_L10_L11_2026-05-13.yaml` (P8A_step5000 base, rank=16, alpha=32, layers=[10,11], head LR 1e-5, LoRA LR 1e-4). Verification: `tests/test_lora_adapter.py` 10/10 pass; `scripts/smoke_lora_wiring_2026-05-12.py` trainable encoder fraction 0.17%, optimizer `lora` group at LR=1e-4 with 393,216 params.
>
> #### Open-loop status changes this session
>
> - `dev-to-lockbox-substrate-transfer-gap` (HIGH) — close criterion (b) is MET by D3 (0/4 ckpts pass substrate-agnostic AUC>0.95 both directions). Status moved to `resolvable`. Resolution proposed in the IQ-shortcut thread; pending user gate.
> - `chronic-6-encoder-iq-angle-drift-during-ft` (MEDIUM) — D6 B=50 final numbers are loaded; remains open per its own close criterion (which requires a counterfactual training experiment, not a measurement).
> - `clip-frozen-chronic-6-auc-robustness` (LOW) — D6 B=50 measures angle stability at n=282; AUC robustness at larger N remains untested.
> - `t5c-classifier-capacity-mechanism` (MEDIUM) — D5+D2+D6 add per-identity / encoder-direction measurements but do not directly probe inv_mean recomputation. Remains open.
> - `train-bucket-identity-overlap-gcs-audit` (LOW) — CLOSED 2026-05-12. GCS audit verdict: training-data anonymization scheme has no shared join key with eval-side identity names; person-level overlap is structurally undetectable from manifests.
>
> Two open loops added this session:
> - `chronic-6-encoder-iq-angle-drift-during-ft` (MEDIUM, see above)
> - `clip-frozen-chronic-6-auc-robustness` (LOW, see above)
>
> #### Pending decisions (user-gated)
>
> 1. GPU slot authorization (count + which slots). Slot menu + tradeoffs in OPINIONS docs.
> 2. LoRA launch authorization (yaml ready).
> 3. Formal closure of `dev-to-lockbox-substrate-transfer-gap`.
>
> ---
>
> ⚠ **HISTORICAL CONTEXT below — agent thinking 2026-05-09 → 2026-05-12 morning**
>
> The entries below mix numerical findings with interpretive framings, some of which were later partially retracted (e.g., the "binding constraint is FT itself" reading in the 2026-05-09 entry; the "memorization" framing addressed in this file's top section). Dated FACTS docs under `analysis/*_eval_*/` and `analysis/cpu_diagnostics_*/` are authoritative for numerical claims. **2026-05-12 cleanup pass**: the most egregious forbidden-word usages (`REFUTED`, `SUPPORTED`, etc. — see `AGENT_GUIDE.md` Rule 5) have been replaced with neutral phrasings; other interpretive language is preserved in place because rewriting it would lose the historical record of how the agents were reasoning. Read this section as agent thinking at the time, not as current ground truth.
>
> ---
>
> **2026-05-12 morning update**: The combined **T6 / T7 / T5C scorecard** SUCCEEDED 2026-05-12 03:13 UTC (Vertex `6455763074874867712`, runtime 9h 54m). 12 ckpts on the standing v3-fix policy. **P8A retains contract rank-1**; **T5C step3500 is the highest-ranking new ckpt at rank 3**, above T3_SLOT1 step1500 at rank 4. 6 of 12 ckpts pass all three contract gates (P8A, E2B, T5C step3500, T3_SLOT1 step1500, T5C step3750, T5C step1500). **All 6 jitter ckpts (T6 ×3 + T7 ×3) regress dev_fake_macro_recall below the 0.30 floor** — jitter@0.50 does NOT compose as a free additive lever on T3 or T4 bases. Among all-pass ckpts, rank ordering tracks ascending `lockbox_real_fpr` strictly (P8A 0.0184 → E2B 0.0235 → T5C step3500 0.0279 → T3_SLOT1 0.0309 = T5C step3750 0.0309 → T5C step1500 0.3204). This `lockbox_real_fpr`-ascending pattern has been observed at the top of the contract ranking across 13+ packets; whether it functions as a binding operational constraint or as an artifact of the contract's tiebreak structure is interpretive (see D4 FACTS doc for the per-quartile measurement). T5C step3500 vs T4 step10500 (its base): dev_macro +0.04, lockbox_real_fpr halved (0.028 vs 0.054), lockbox_fake_recall +0.29 (0.66 vs 0.37), dor_dev (n=50) FPR 0.06 vs 0.18. Interpretation flagged at time of writing: the dor_dev delta is consistent with the "classifier capacity → uniform encoder invariance" hypothesis from A3 2026-05-11, but the mechanism has not been verified — CPU L11 atlas recompute on T5C is the relevant probe. Eval folder authored at [`analysis/t6_t7_t5c_scorecard_eval_2026-05-12/`](../../analysis/t6_t7_t5c_scorecard_eval_2026-05-12/) with FACTS + AGENT_PROPOSAL split. Packet retro at [`packets/T6_T7_T5C.md`](packets/T6_T7_T5C.md). Memory `project_face_scale_jitter_load_bearing.md` has been amended (per MEMORY.md index description "load-bearing AS BUNDLE REPLACEMENT, refuted as composable lever") — the prior "load-bearing single lever" framing was scoped to P14 bundle replacement; this session's data on the 6 T6/T7 jitter ckpts is not consistent with a composable-lever reading.
>
> **Deployment Q for the user**: T5C step3500 catches 1.7× more lockbox fakes than P8A (0.66 vs 0.39) at +0.0095 absolute `lockbox_real_fpr` (0.0279 vs 0.0184). The v3-fix policy ranks P8A first via the lockbox-FPR tiebreak — but whether to ship T5C alongside or in place of P8A depends on the FP/FN cost ratio (which the policy doesn't make explicit). User decision pending.
>
> **Next-step plan**: CPU-first L11 atlas inv_mean recompute on T5C step1500 + step3500 + step3750 (~3h MPS, $0) — tests whether T5C's chronic_6 invariance recovered vs T4's −0.0315 regression. If yes, authorizes a T5C × hidden_dim sweep ($70-100 GPU); if no, the rank-3 placement is via a different mechanism and needs different investigation.
>
> ---
>
> **Earlier 2026-05-10 PM**: Post-T3 robustness diagnostics + may6 production-frame retest landed at `analysis/cpu_diagnostics_2026-05-10/`. 5 CPU jobs (multi-axis robustness matrix, catastrophic-tail count, IQ-gate filterability, post-gate FPR, cross-ckpt disagreement) + may6 retest of T3 ckpts on 152 production-drift frames. Per-frame data covers 4 ckpts (P8A, E2B, T3_S1_step1500, T3_S1_step2500) on `teams_real_all_dev` plus may6/may5. Headline numbers: P8A scores 0/92 may6 frames > 0.5; E2B 53/92 (57.6%); step1500 6/92 (6.5%); step2500 71/92 (77.2%). Per-cohort score statistics + per-axis bin FPR + IQ-gate retention by chronic identity all in `ROBUSTNESS_FACTS_2026-05-10.md`. The agent who ran the diagnostics produced an OPINION doc at `AGENT_PROPOSAL_2026-05-10.md` arguing for step1500 as primary deployment candidate; per AGENTS.md "Reading order for forming an independent view," read the FACTS docs first and form your own view before reading the OPINION doc. Memory `project_xinhe_may6_falseflag_2026-05-06` was amended this session to make the E2B attribution explicit (the may6 false-flagging recorded there was the deployed model, which is E2B per `project_deployment_is_e2b_2026-05-06`, not P8A).
>
> ---
>
> **2026-05-10 ~07:00 (prior agent's headline, preserved for context)**: T3 (Stage 3) ran 2026-05-09 → 2026-05-10. **T3_SLOT1_PERIODIC_STEP2500** numerical readout: 79.27% F4 v2 viso recall (vs P8A 67.09%, PA 72.36%); 84.97% HDTF `proper_visomaster_enhanced_teams_dev` at FPR-cal 5%; step2500 HDTF teams cells uniformly above step1500's. Step2500 dev recall on the strict F0 v2 floor: 22.95% (< 30%). Per the prior agent's reading (preserved): the shortfall is a contract-calibration artifact — with chronic-6 in F0 reals, step2500 scores Roy_D high → auto-τ clips at 0.971 → dev fakes collapse at that τ. T3_SLOT1_PERIODIC_STEP1500 dev recall on the strict F0 floor: 37.6% (meets floor).**
>
> **Prior agent's reading (preserved for context, not re-verified)**: under the user's "abstain below IQ threshold" deployment policy (= F4 substrate), step2500 was identified as the candidate of interest. Under strict F0 contract policy (chronic-6 included), step1500 was the only T3 ckpt above the floor; P8A holds lex-policy rank 1 because all T3 ckpts hit ~9.92-9.99% `dev_worst_real_stress_fpr` (target ≤10%; P8A 6.85%). The 9.92% T3 stress FPR is concentrated on Roy_D-on-lighting_extreme — ex-Roy_D, T3 stress FPR is 3.2% vs P8A 5.0%; the F4 deployment lens masks this distribution.
>
> **Prior agent's note to the next agent (interpretive; preserved)**: a deployment path was sketched — ship step2500 with FPR-calibrated τ on a production-realistic real cohort. Open question (at time of writing): whether to launch a refinement packet (T4 — face_scale_jitter or Roy_D hard-negative mining) before shipping. The deployment decision remains user-gated and has not been re-verified this session. T4 directions documented in `analysis/cpu_diagnostics_2026-05-09/MORNING_BRIEF_2026-05-10.md` §13. Memory `project_t3_slot1_step1500_lockbox_lift_2026-05-09.md` should be amended (top-line headline now points at step2500, not step1500). Memory `project_data_axis_lever_pulled_twice_no_lift.md`'s original "no lift" claim is no longer consistent with PA + T3_SLOT1 outcomes — the memory entry has been amended.
>
> **Reading order for the next agent**: (1) `MORNING_BRIEF_2026-05-10.md` §15 TL;DR (30-second read), then (2) `packets/T3.md` (full packet doc), then (3) `T3_SCORECARD_FACTS_2026-05-09.md` + `_t3_f4_outputs/*.json` for the raw numbers, then (4) `_t3_hdtf/reports/` + `_t3_hdtf_step2500/reports/` for HDTF per-suite reports.
>
> ---
>
> **Earlier 2026-05-09 13:00 snapshot (preserved for context)**: Stage 2 ALL 3 SUCCEEDED. CPU analytics: 3 probes × 9 ckpts on Dor cohort + Roy_D + L11 features. **Headline: ALL Stage 2 step4500 ckpts regress P8A's chronic-identity Pillar 2 — Roy_D real-FPR at τ=0.5 is S1 99.2% / S2 96.9% / S3 100% vs P8A 45.4%.** L11 cos-distance vs P8A on Roy_D: S2_step500 0.21 → S3_step4500 0.82; Spearman r=0.91 with score regression across n=9 sibling Stage 2 ckpts.
> 
> **Reading (provisional, 2026-05-09 audit-corrected):** Across the levers tested (data-aug-toggle / LR / pair-rank-λ / Fourier-aug / pair-rank+GroupDRO bundle), every FT-from-P8A regime drifts the dor cluster within 500-2500 steps. The strongest reading from this convergent pattern is "FT-from-P8A under the tested regimes drifts L10-L11 dor invariance"; this was previously phrased as "FT itself is the binding constraint" — that universal framing is overstated and has been corrected (see `analysis/stage2_cpu_2026-05-09/STAGE2_SCORE_PROBE_OPINIONS_2026-05-09.md` 2026-05-09 caveats and `project_stage2_all_levers_regress_p8a_2026-05-09.md` Why-section). Untested lever classes: LoRA / parameter-efficient adapter FT (preserves base by construction), hard-output-preservation losses on a reference cohort, multi-objective training with Pillar-2 explicit constraint, frozen-encoder regimes other than head-only retrain.
>
> **L11 anchor-loss FT (option 3a)** remains a candidate next intervention — its targeting is supported by the L11-distance ↔ regression correlation, but the correlation alone is not dispositive (training-step amount is a confounder; alternative anchor layers like L9/L10 / output-side anchor have not been compared). Under the user's 2026-05-09 reframe (production target genuinely unknown; Pillar 3 robustness must come from forgery signal), preserving L11 also preserves the IQ-shortcut representation co-located there — so the anchor lever is not unambiguously net-positive without first establishing that L11 carries a forgery channel separable from the IQ + identity shortcuts. The forgery-signal atlas CPU diagnostic at `analysis/cpu_diagnostics_2026-05-09/` is the cheap test of that.
>
> Promotion contract scorecard NOT recommended for any Stage 2 ckpt — saves $45-60 GPU. FACTS + OPINIONS at `analysis/stage2_cpu_2026-05-09/`.
>
> **2026-05-09 audit-pass artifacts at `analysis/cpu_diagnostics_2026-05-09/`** — under user reframe (production target unknown; robustness must come from forgery signal not other-property shortcuts), four CPU diagnostics ran on the 800-frame triptych for all 12 ckpts (3 reference + 9 Stage 2):
>  - **Forgery-signal atlas** (per-layer × per-signal AUC matrix): all signals (real/fake, is_dor, is_chronic_6, lap_var-high, min_dim-high, face_size-high) extractable at AUC 0.95+ from L11 across all ckpts; L0 already gives 0.72 AUC for real/fake from frozen patch+pos embed alone.
>  - **F4 substrate filter recompute** on the triptych: dropping chronic-6 + is_no_face + min_dim<200 cuts Stage 2 step4500 real-FPR from 0.149-0.163 (F0_full) to 0.042-0.049 (F4) while preserving fake-recall within ≤0.02.
>  - **IQ R² + residual AUC decomp**: LOCKBOX R² varies 12× across Stage 2 ckpts (0.07-0.84); DEV_NO_CHRONIC residual AUC is 0.90-0.95 across all Stage 2 ckpts.
>  - **Per-identity score table**: same-root-identity-different-session sub-identities span 100×-280× score range under same ckpt (`dor_shkedi` 0.298-0.859 vs `dor_shkedi__s16` 0.003-0.080).
> 
> FACTS doc: `analysis/cpu_diagnostics_2026-05-09/CPU_DIAGNOSTICS_FACTS_2026-05-09.md`. No OPINIONS doc written this pass — by user request, conclusions are user-gated.)
>
> **Purpose**: single-page current-state snapshot. Always-current; rolling. Older dated snapshots archived in [`archive/`](archive/) for historical reference.
>
> **How to use this page**: a new agent reads this AFTER `AGENTS.md` and BEFORE the user's specific task. It tells you what's running, what just landed, what's open, and what decision the user might want next. If the wall-clock is materially after the "Last refreshed" date, verify against `git status`, the tail of [`TIMELINE.md`](TIMELINE.md), and the latest [`packets/`](packets/) entry — state moves quickly.
>
> **FRESH-AGENT GUIDANCE (2026-05-10 — START HERE)**: the user has explicitly asked for an independent fresh-agent take on the T3 packet results. Read [`HANDOFF_2026-05-10.md`](archive/HANDOFF_2026-05-10.md) FIRST — it contains the user's framing constraint (production robustness across cameras/lighting/rooms/people), a load-bearing reading order (FACTS docs first, OPINIONS / `packets/T3.md` interpretation only AFTER you form your own view), and the specific deliverable (state your own reading + propose a next step before opening any interpretive doc). Do not skip the production-robustness constraint section — it changes how F4 substrate-cleaning numbers should be interpreted.
>
> ---
>
> **PRIOR FRESH-AGENT GUIDANCE (2026-05-08, preserved for context)**: the IQ-shortcut deconvolution program is mid-flight. The user's explicit instruction was to organize the artifacts so a new agent can form their own view of the next-step decision **without being biased by the prior agents' OPINIONS docs**. Reading order for an unbiased fresh take:
>
> 1. Read the **FACTS docs FIRST** (factual-only, forbidden-words enforced):
>    - [`analysis/p2_eval_2026-05-08/P2_PHASE_A_VERDICT_FACTS_2026-05-08.md`](../../analysis/p2_eval_2026-05-08/P2_PHASE_A_VERDICT_FACTS_2026-05-08.md) — promotion contract scorecard
>    - [`analysis/p2_eval_2026-05-08/P2_DEEPER_ANALYSIS_FACTS_2026-05-08.md`](../../analysis/p2_eval_2026-05-08/P2_DEEPER_ANALYSIS_FACTS_2026-05-08.md) — J1-J5 characterization
>    - [`analysis/iq_data_atlas_2026-05-08/IQ_ATLAS_FACTS_2026-05-08.md`](../../analysis/iq_data_atlas_2026-05-08/IQ_ATLAS_FACTS_2026-05-08.md) — cross-pool IQ measurement
>    - [`analysis/iq_shortcut_decomp_2026-05-08/IQ_DECOMP_FACTS_2026-05-08.md`](../../analysis/iq_shortcut_decomp_2026-05-08/IQ_DECOMP_FACTS_2026-05-08.md) — 23-cell R² decomposition (Stage 1 of IQ-deconvolution program)
>    - [`analysis/iq_perlayer_probe_2026-05-08/IQ_PERLAYER_PROBE_FACTS_2026-05-08.md`](../../analysis/iq_perlayer_probe_2026-05-08/IQ_PERLAYER_PROBE_FACTS_2026-05-08.md) — check (a) per-layer IQ probe (NEW)
>    - [`analysis/dor_encoder_axis_2026-05-08/DOR_ENCODER_AXIS_FACTS_2026-05-08.md`](../../analysis/dor_encoder_axis_2026-05-08/DOR_ENCODER_AXIS_FACTS_2026-05-08.md) — check (c) Dor encoder-axis (NEW)
>    - [`analysis/p2_d_hdtf_2026-05-08/P2_D_HDTF_FACTS_2026-05-08.md`](../../analysis/p2_d_hdtf_2026-05-08/P2_D_HDTF_FACTS_2026-05-08.md) — check (b) D step3000 HDTF Phase C (NEW)
>    - [`docs/packet_retrospectives/MODEL_GOALS.md`](MODEL_GOALS.md) — three pillars + promotion bar
>    - [`docs/packet_retrospectives/SCORECARD_GUIDE.md`](SCORECARD_GUIDE.md) — when to use which scorecard mode
> 2. Form your own reading of what the data shows.
> 3. THEN, optionally, read the OPINIONS docs to compare your reading to the prior agents'. **You are explicitly free to disagree.**
>    - [`analysis/p2_eval_2026-05-08/P2_DEEPER_ANALYSIS_OPINIONS_2026-05-08.md`](../../analysis/p2_eval_2026-05-08/P2_DEEPER_ANALYSIS_OPINIONS_2026-05-08.md) — synthesis + proposed pre-Stage-2a sequence
>    - [`analysis/iq_shortcut_decomp_2026-05-08/IQ_DECOMP_OPINIONS_2026-05-08.md`](../../analysis/iq_shortcut_decomp_2026-05-08/IQ_DECOMP_OPINIONS_2026-05-08.md) — sister-agent's Stage 2 reading
>    - [`docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md`](threads/iq_shortcut_deconvolution_program_2026-05-08.md) — the program proposal thread
> 4. (No pending check artifacts as of 2026-05-08 evening — all three pre-Stage-2a checks landed; Stage 2 decision is now user-gated.)
>
> The fresh agent's task is to look at the convergent FACTS, optionally challenge the existing OPINIONS, and make their own call on Stage 2 (or whatever direction they propose). The user reserves the GPU spend authorization regardless.

---

## Eval-substrate caveats (carried over; still apply uniformly)

These caveats apply to every FPR / recall number on this page and the eval folder:

1. **Eval-vs-production crop-tightness mismatch** ([`eval_production_crop_tightness_gap`](threads/eval_production_crop_tightness_gap.md)) — eval frames carry more background context than production crops; structurally upstream of face-size leak + camera-signature shortcut + webcam FPR investigations.
2. **`sharpness_laplacian` is computed on full image, not face crop** ([`sharpness_metric_bug`](threads/sharpness_metric_bug.md)) — per-quartile FPR analyses indexed on this column are partially confounded.
3. **Eval substrate has known data-quality issues** ([`eval_substrate_data_hygiene`](threads/eval_substrate_data_hygiene.md)) — is_no_face slice (n=219) data degeneracy + 99×110-pixel source crops below any reasonable production resolution floor.
4. **Quality-enhancement routing bug** affects ALL R13 packets pre-2026-05-05 (memory `project_quality_enhancement_routing_2026-05-05.md`). Relative comparisons remain valid (uniform contamination); absolute claims about GFPGAN-enhanced fakes need post-fix retraining to revisit.

---

## Cross-references

- [`AGENTS.md`](AGENTS.md) — read-first protocol; "Reading order for forming an independent view" is the canonical pass-1 / pass-2 split.
- [`TIMELINE.md`](TIMELINE.md) — append-only chronological master index; tail tells you what just landed.
- [`OPEN_LOOPS.md`](OPEN_LOOPS.md) — mechanically-generated open-issue inventory.
- [`packets/P1.md`](packets/P1.md) — most recent packet retro.
- [`AGENT_GUIDE.md`](AGENT_GUIDE.md) — the 6-rule contract for proposing the next packet.
- [`eval_folder_template.md`](eval_folder_template.md) — the FACTS/OPINIONS file-level split for `analysis/<packet>_eval_<date>/` folders.
- [`packet_template.md`](packet_template.md) — the structure every per-packet retro under `packets/` follows.
- [`archive/`](archive/) — older dated state snapshots.
