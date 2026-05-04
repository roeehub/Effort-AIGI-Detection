# RLP2  ·  R13 Relaunch Packet 2 — "enhanced hurts?" + hints rematch, on an unenhanced-only val pool

> **Template contract**: fill every section. If a section truly has no content, write `*(none — reason)*` rather than deleting the heading. Keep the status-card table intact.

## Status card

| Field | Value |
|---|---|
| Dates | 2026-04-20 → 2026-04-21 |
| Slots | 6 (RLP2_01, RLP2_02, RLP2_03, RLP2_04, RLP2_05, RLP2_06) |
| Headline lever | no-hints + unenhanced proper-data, probing moderate enhanced-proper add-ons and two prediction-stability hedges |
| Leader slot | RLP2_02 |
| Leader metric | best_ood_composite = 0.98937 (packet-2 top, pre-`value_composite`) |
| Verdict | ⚠️ muddled (partially superseded) |
| Next-packet decision | RLP3 accepts unenhanced-proper as the main bet; re-opens enhanced with dose-matched slot + enhanced-proper OOD lanes for every run |
| Themes touched | [`../threads/promotion_contract_evolution.md`](../threads/promotion_contract_evolution.md), [`../threads/gate_alignment_story.md`](../threads/gate_alignment_story.md), [`../threads/preprocessing_parity_bug.md`](../threads/preprocessing_parity_bug.md) |

## Configuration

What changed vs RLP1:

- **Hints fully out** of every slot (`visomaster_hints.enabled: false` — `experiments/phase2_round13/R13_RLP2_06_FT_WTB1_plus_proper_unenhanced_low_arcface_live.yaml:136`). RLP2_01 and RLP2_03 specifically answer "does no-hints still hold with proper-data?"
- **Split mode flipped** `shuffle` → `hash_stable` (same yaml, line 79). See `docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_RESULTS_AND_PLANNING_CONTEXT_2026-04-21.md:5` — re-partitions identities and introduces an unquantified AUC delta vs RLP1.
- **FT-only**, no scratch (RLP1_08 still live), no WT-C sidecars, no `stability_lambda` (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN_2026-04-21.md:67-74`).
- Everything else inherited from the RLP1 FT scaffold (ViT-B-16-DataComp-XL, rank 736, 10k steps, LR 3e-5, ArcFace s 6→12, base ckpt R12_G_FP32).

Control slot: **RLP2_01** — fresh no-hints control on the current mutable-source + proper-data snapshot. Explicitly not comparing against RLP1_01 (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN_2026-04-21.md:77-83`).

Variants:

- `RLP2_01` → refreshed no-hints control (`R13_RLP2_01_FT_WTB1_no_hints_refresh_live.yaml`)
- `RLP2_02` → main bet: `01` + unenhanced proper-data (`R13_RLP2_02_FT_WTB1_plus_proper_unenhanced_live.yaml`)
- `RLP2_03` → `02` + `proper_visomaster_enhanced_teams` moderate enhanced step (`R13_RLP2_03_FT_WTB1_plus_proper_unenhanced_plus_teams_enhanced_live.yaml:180`)
- `RLP2_04` → `02` + `proper_visomaster_enhanced_clean` (clean-enhanced diagnostic)
- `RLP2_05` → `02` + stronger spatial augmentation (jitter probe)
- `RLP2_06` → `02` + lower ArcFace endpoint `s_end: 10.0` (calibration probe; `R13_RLP2_06_*.yaml:265`)

Yamls: `experiments/phase2_round13/R13_RLP2_01_*.yaml` through `R13_RLP2_06_*.yaml`.

## Results at the time

Leading readouts (`best_ood_composite`, AUC-based, ~= mean of holdout + OOD AUC; metric was `ood_composite` — `value_composite` did not yet exist in RLP2):

| Slot | best composite | Δ vs RLP2_01 | Notes |
|---|---:|---:|---|
| RLP2_02 | **0.98937** | +0.00275 | packet leader; clean win on shared snapshot |
| RLP2_06 (low-s ArcFace) | ~tied with 02 | ~0 | packet's top holdout AUC 0.99161 |
| RLP2_05 (spatial) | ~tied with 02 | ~0 | live jitter did not cleanly separate from 02 |
| RLP2_03 (teams_enhanced) | under 02 | < 02 | above 01 but below 02; teams_ood_real jitter 0.0579 vs 0.0382-0.0477 on non-enhanced arms |
| RLP2_04 (clean_enhanced) | under 01 (barely) | +0.00058 | basically indistinguishable from the fresh control |
| RLP2_01 | 0.98662 | — | fresh control |

Cross-packet comparison (caveated): `RLP2_01 = 0.98662` vs `RLP1_01 = 0.98915` = Δ `−0.00253`. See `docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_RESULTS_AND_PLANNING_CONTEXT_2026-04-21.md:8` — this delta carries the split-mode component and cannot be attributed solely to mutable-source drift.

Data shape at launch (from `docs/relaunch_handoffs/R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN_2026-04-21.md:146-156`):

- RLP2_02 unenhanced proper fake rows: ~684 (inherited the RLP1_04 dose)
- RLP2_03/04 enhanced add: ~+1484 rows each → enhanced arms ran at **2.2× the unenhanced dose**, not dose-matched

Artifacts:

- W&B project: `dtect-vision/phase2r13-rlp2-overnight-20260421`
- Live-monitoring read captured in session `3f5b35f6` (2026-04-21 evening)

## Conclusions drawn in-session

Drawn before the RLP3 reframe, from session `3f5b35f6` (2026-04-21):

- **Main hypothesis validated against the refreshed control.** Quote: *"`RLP2_02` (no hints + unenhanced proper) beats `RLP2_01` by +0.00275 composite … a clean win on a shared-snapshot A/B."*
- **Hints verdict solidified.** RLP2_01 (no hints) ≤ RLP1_02/03 hint-bearing controls; combined with RLP1, hints settled failed (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_EXPERIMENT_PLAN_2026-04-21.md:20`).
- **"Enhanced hurts" called in-session.** Quote: *"two independent packets saying the enhanced proper-data lanes are not where the target-domain value lives."*
- **Stability probes inconclusive on live metrics.** RLP2_05 (spatial) and RLP2_06 (low-ArcFace) tied RLP2_02 on composite; live jitter did not separate them.
- Fresh-control drop left open: mutable-source drift **or** split-mode artifact.
- **Session IDs**: `3f5b35f6` (live read), `e65d2f50`, `eabf9ffb` (RLP3 plan v3), `1e871e7e` (dose-match math: `unenhanced_teams=342, enhanced_teams=1484`).

## Retrospective (as of 2026-04-24)

**Verdict: ⚠️ muddled (partially superseded).**

What held up:

- **Hints are out, permanently.** RLP2_01 and RLP2_03 both carried no hints and did not lose to hint-bearing controls; combined with RLP1, this closed the hint family. See `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_EXPERIMENT_PLAN_2026-04-21.md:20`: *"Hints are out. RLP1_02, RLP1_03, RLP2_01 (no hints) all underperform the corresponding hint-bearing controls — hints have failed in two packets."*
- **Unenhanced proper-data is the best new signal.** RLP2_02 at 0.98937 became the anchor recipe that RLP3 and later packets reconfirmed (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_EXPERIMENT_PLAN_2026-04-21.md:21`).

What was explicitly revised:

- **"Enhanced hurts" was downgraded to muddled** in RLP3's planning doc, section 1.2 (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_EXPERIMENT_PLAN_2026-04-21.md:27`): *"`val_holdout` for `RLP2_02` contains zero enhanced-proper content, so we are measuring against an unenhanced-only validation slice. We cannot separate 'enhanced training hurts the enhanced slice we don't measure' from 'enhanced training hurts the unenhanced slice we do measure.'"* The evaluation pool composition guaranteed the conclusion could not be read cleanly. Compounded by the 2.2× enhanced-to-unenhanced dose ratio — the RLP2 enhanced arms were simultaneously overdosed and measured against an unenhanced-only pool.
- **Packet-2 control drop is partly split-mode artifact, not just drift.** `docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_RESULTS_AND_PLANNING_CONTEXT_2026-04-21.md:5-9` pins the split-mode caveat and explicitly blocks attribution of the `−0.00253` RLP2_01-vs-RLP1_01 gap to mutable-source drift alone. Do not compare RLP2 numbers to RLP1 numbers without bounding the split-mode component first.
- **Stability probes (05/06) remained inconclusive.** RLP3 carries them forward as separate slots to test under seed variance (`R13_RLP3_03_FT_proper_low_arcface`, `R13_RLP3_04_FT_proper_spatial`, `R13_RLP3_05_FT_proper_low_arcface_spatial`).

How RLP3 addressed the muddle:

- **A6 enhanced-proper OOD lanes** for every run + enhanced/unenhanced slicing of OOD fake metrics (`docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_EXPERIMENT_PLAN_2026-04-21.md:156-180`) — finally made the enhanced-vs-unenhanced question measurable.
- **Slot 08 dose-matched enhanced** (~342 enhanced_teams = 342 unenhanced_teams) to separate dose from content (`…PACKET3_EXPERIMENT_PLAN_2026-04-21.md:348`).
- **A9 `value_composite`** introduced as a readout-only deployment-aligned metric. RLP2 numbers predate it entirely; any cross-packet comparison requires A2b retroactive re-evaluation.

**Preprocessing-parity status — PRE-FIX.** All RLP2 best_ood_composite numbers were produced before commit `855871e` (2026-04-24) changed `cv2.INTER_AREA → cv2.INTER_LINEAR` in `batch_inference_gcs.py:407` and `arena/model_arena.py:472`. Any retro-score or arena readout of RLP2 checkpoints taken before that commit carries silent preprocessing drift vs training-time preprocessing. Training-time composite numbers are internally consistent; retro-score comparisons that mix RLP2 pre-fix readouts with later post-fix readouts are not. See [`../threads/preprocessing_parity_bug.md`](../threads/preprocessing_parity_bug.md).

**Thread continuations:**

- [`../threads/promotion_contract_evolution.md`](../threads/promotion_contract_evolution.md) — RLP2's muddled verdict was partly a validation-pool composition issue (val_holdout had zero enhanced-proper). RLP3's A6 + A10 + `value_composite` are direct responses.
- [`../threads/gate_alignment_story.md`](../threads/gate_alignment_story.md) — hints-as-supervision was the surviving test here; hints closing in RLP2 is one of the steps that fed the broader gate-alignment rewrite later.

## Source files

- **Handoffs**:
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET2_EXPERIMENT_PLAN_2026-04-21.md` — 6-run design, launch order, data-shape approximations (lines 84-94, 146-156)
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_RESULTS_AND_PLANNING_CONTEXT_2026-04-21.md:3-10` — split-mode caveat (pinned)
  - `docs/relaunch_handoffs/R13_RELAUNCH_PACKET3_EXPERIMENT_PLAN_2026-04-21.md:20-29` — revised "enhanced hurts" reading as muddled; RLP2_02 leader metric cited at line 21
  - `docs/relaunch_handoffs/R13_RELAUNCH_TRAINING_SOURCE_OF_TRUTH_2026-04-21.md:78-113` — canonical placement of RLP2 in the doc stack
- **Yamls**: `experiments/phase2_round13/R13_RLP2_01_*.yaml` through `R13_RLP2_06_*.yaml` (6 files).
- **Scorecards / analysis**: no frozen RLP2 scorecard predated the `final_eval/` namespace introduced in RLP3 (A2); retroactive re-evaluation via `tools/retroactive_final_eval.py` (A2b) is the only way to produce comparable numbers.
- **Memory pointers**: contract-policy bug (`project_contract_policy_bug.md`) does not apply to RLP2 readouts — the contract scorer was not used here; RLP2 was selected on `best_ood_composite`.
