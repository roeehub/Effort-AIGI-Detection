# State — current rolling snapshot

> **Last refreshed**: 2026-05-08 09:00 local (all 4 P2 slots `JOB_STATE_SUCCEEDED` overnight; A 21:38→23:49 UTC = 2h 11m, B 21:42→00:39 = 2h 57m, C 21:36→00:53 = 3h 17m, D 22:48→05:31 = 6h 43m; eval folder skeleton populated with FACTS docs; D1-D4 CPU diagnostics in flight).
>
> **Purpose**: single-page current-state snapshot. Always-current; rolling. Older dated snapshots archived in [`archive/`](archive/) for historical reference.
>
> **How to use this page**: a new agent reads this AFTER `AGENTS.md` and BEFORE the user's specific task. It tells you what's running, what just landed, what's open, and what decision the user might want next. If the wall-clock is materially after the "Last refreshed" date, verify against `git status`, the tail of [`TIMELINE.md`](TIMELINE.md), and the latest [`packets/`](packets/) entry — state moves quickly.

---

## Where we stand right now (one paragraph)

The most recently completed packet is **[P1](packets/P1.md) (PE_PAIR_RANK_DRO)** — pair_rank λ=0.2 + multi-axis GroupDRO with `chronic_flag` (BUNDLE) vs pair_rank-only (PAIRRANK), both FT-from-P8A_step5000 on the post-`2feea58` codepath. Phase A (29-suite contract scorecard) **SUCCEEDED**; Phase C (16-suite HDTF cross-substrate) **FAILED** at the promotion_contract step (per-suite diagnostic completed; verdict bundle never written; root cause not yet investigated). At the contract-selected τ, **no P1 ckpt clears F1 (90% lockbox recall)**; the contract's rank-1 winner is `P1_PAIRRANK_PERIODIC_STEP500` with 70.8% lockbox recall at τ=0.768. **F1 is reachable at non-contract τ** — `BUNDLE_step500` hits 96.5% lockbox recall at FPR ≤ 10% (τ=0.989); `PAIRRANK_step500` hits 91.5% at τ=0.50. **F5 PASSES big** for the BUNDLE arm: PC_Generator chronic FPR drops 62.9% → 0-3%. **F4 PASSES** universally at calibrated τ (max HDTF real FPR 1.4%). **F3 is partial**: 3 of 4 untargeted IQ axes show decoupling (sharpness, min_dim, color_b_dev) but face_area_fraction amplifies (+106-266%). **P1 introduced a Roy_D regression**: 130 frames flip from 29% FPR (P8A) to 78-93% FPR (P1); the regression is `color_b_dev`-aligned (Δr=+0.71 mirror of P8A's r=−0.71) AND **shared between BUNDLE and PAIRRANK arms** (Wilcoxon p=0.875), refuting the initial GroupDRO-balloon hypothesis. **Deployment is still E2B** per memory `project_deployment_is_e2b_2026-05-06.md`; E2B beats every P1 ckpt on `dev_fake_macro_recall` (0.508 vs 0.354). This session also delivered two bug fixes (trainer.py:1727 W&B logging gap + phase_d/run_chronic_filter.py regex) and a wiki contract upgrade (rolling STATE, FACTS/OPINIONS file convention, hard stop on per-session HANDOFF docs).

---

## In flight / running right now

**P2 packet** (PREVENT_NOT_UNLEARN, 4 slots) — **all 4 slots `JOB_STATE_SUCCEEDED` 2026-05-07/08**. Eval folder at [`analysis/p2_eval_2026-05-08/`](../../analysis/p2_eval_2026-05-08/) carries the populated FACTS docs (`RESULTS_FACTS_2026-05-08.md`, `CANARY_TRAJECTORY_FACTS_2026-05-08.md`). Per-slot summary:

| slot | yaml | wandb run | wall clock | terminal | final `_step` | canary fires | training-AUC peak |
|---|---|---|---|---|---:|---:|---|
| A — BUNDLE | `R13_P2_SCRATCH_BUNDLE.yaml` | `u22wz1vf` | 2h 11m | 23:49:26 UTC | 7508 | 1 | 0.6802 (step500) → declines to 0.5100 (step7000) |
| B — CORR_ONLY | `R13_P2_SCRATCH_CORR_ONLY.yaml` | `mlo5vfe8` | 2h 57m | 00:39:28 UTC | 6508 | 1 | 0.6800 (step500) → drops below chance to 0.3815 (step3500) |
| C — PAIRRANK_ONLY | `R13_P2_SCRATCH_PAIRRANK_ONLY.yaml` | `oaur8odo` | 3h 17m | 00:53:42 UTC | 13013 | 4 | 0.8840 → 0.9862 (step7000) monotone |
| D — FOURIER | `R13_P2_SCRATCH_FOURIER.yaml` | `89tt9xyz` | 6h 43m | 05:31:20 UTC | 25013 | 8 | 0.8743 → 0.9908 (top_n step19000) monotone |

Slot D's wall-clock is ~3× the other slots; consistent with the per-frame FFT overhead from `data/augmentations/fourier_band_aug.py` (~3-channel FFT2 + IFFT2 per fired frame at p_apply=0.5).

Canary trajectory headlines (from [`CANARY_TRAJECTORY_FACTS_2026-05-08.md`](../../analysis/p2_eval_2026-05-08/CANARY_TRAJECTORY_FACTS_2026-05-08.md)):

- Slot A canary single-fire: `score_p95_on_reals=0.4994`, all `recall_at_tau05` = 0 — uniform-output state.
- Slot B canary single-fire: `score_p95_on_reals=0.5005`, `lockbox_recall@FPR_10pct=0.01` but `recall_at_tau05/{viso,deeplive}_fake = 1.00` — every frame scored ~0.500, fakes barely above τ=0.5.
- Slot C 4 fires: `score_p95_on_reals` saturates to 0.998+ at first fire; chronic-6 means rise from 0.07-0.55 (step3000) to 0.66-0.97 (step12000); `lockbox_recall@FPR_10pct` peaks at 0.26 (step12000).
- Slot D 8 fires: `score_p95_on_reals` lower (0.88-0.99 oscillating); **non-monotonic chronic dip at `_step=18000`** — Roy_D 0.94→0.68, bla_bla_chow 0.91→0.41, max_per_id 0.94→0.68 — coinciding with `lockbox_recall@FPR_10pct=0.41` peak (top among all 4 slots).

**Open observations** flagged in the FACTS docs (no interpretation, no resolution at this time):
- Canary cadence anomaly: yaml configured `frequency_steps: 1000` but observed cadence is every 3000 W&B `_step`s for C/D, only 1 fire each for A/B. Cause unverified.
- All 4 slots terminated before `nEpochs: 16` cap; early_stopping_patience=12 likely fired but stop-condition not verified at this level.
- Slot A and Slot B both early-stopped at the SAME `_step` band (6500-7500) at epoch 3 with similar final score distributions.

**D1-D4 CPU diagnostics** at [`analysis/p2_eval_2026-05-08/d1_d4_cpu/`](../../analysis/p2_eval_2026-05-08/d1_d4_cpu/) — in flight as of writing. Compares 5 selected ckpts (A/top_n_step500, B/top_n_step500, C/top_n_step7000, D/top_n_step6000, D/top_n_step19000) against P8A reference on the canary 800-frame substrate. D4 weight-delta probe completed: Slot A and Slot B are nearly parameter-identical (cos ≈ 0.997 across all layers — both early-stopped at step500); P2 ckpts at `visual_proj` are nearly orthogonal (cos < 0.01) to A/B and to P8A; Slot D step19000 last-layer residuals are uniquely orthogonal to D step6000 (cos 0.034 at resblock_11). Inference + D1-D3 analyses pending.

**Bug fixes committed (2026-05-07, commits `2e3c26b` + `fb6c38f`)**:
- `trainer/trainer.py:1727` W&B logging gap when `use_group_dro=true` — RESOLVED.
- `analysis/p1_pe_eval_2026-05-07/phase_d/run_chronic_filter.py` regex bug — RESOLVED.
- Wiki contract upgrade (`608dbe2`) — RESOLVED.

**Canary probe infrastructure (NEW 2026-05-07)**:
- [`trainer/mixins/canary_probe.py`](../../trainer/mixins/canary_probe.py) — bulletproof in-training deployment-quality monitor; logs ~15 deployment-relevant scalars to W&B every N steps.
- [`arena/canaries/teams_chronic_diverse_800_2026-05-07.parquet`](../../arena/canaries/teams_chronic_diverse_800_2026-05-07.parquet) — 800-frame canary (600 reals + 200 fakes) covering 6 chronic + 5 healthy + HDTF reals + lockbox + viso + deeplive fakes.
- See thread [`in_training_canary_signal`](threads/in_training_canary_signal.md) for the design rationale.

---

## Most recent eval folder

[`analysis/p2_eval_2026-05-08/`](../../analysis/p2_eval_2026-05-08/) — P2 packet outcomes (4 slots SUCCEEDED). Read in order:

1. `RESULTS_FACTS_2026-05-08.md` — per-slot training outcome + ckpt inventory
2. `CANARY_TRAJECTORY_FACTS_2026-05-08.md` — full per-slot canary trajectory tables
3. `SLOT_D_SMOKE_FAIL.md` — Slot D Fourier-aug 3-tier smoke (Tier-3 floor relaxation rationale)
4. `d1_d4_cpu/` — CPU diagnostics (D1 A/B degeneracy, D2 Slot D step18000 deep-dive, D3 4-slot histogram, D4 weight-delta) — in flight
5. `slot_{A,B,C,D}_canary_history.csv` — raw W&B history pulls

Phase A scorecard for the P2 ckpts has NOT been authorized yet; the user reviews the canary trajectories + D1-D4 CPU diagnostics before deciding GPU spend on the P2 ckpt pool.

[`analysis/p1_pe_eval_2026-05-07/`](../../analysis/p1_pe_eval_2026-05-07/) — prior packet (P1) eval. Read in order:

1. `RESULTS_FACTS_2026-05-07.md` — raw scorecard
2. `RESULTS_F1_F5_FACTS_2026-05-07.md` — close-criterion verdicts
3. `DEEP_DIVE_FACTS_2026-05-07.md` — consolidating record (includes mid-session Self-correction log §16)
4. `FOLLOWUPS_FACTS_2026-05-07.md` — 7 additional CPU jobs (color_b_dev, lockbox τ curves, roy_d axis attribution, counter-theory probe, etc.)
5. Sub-investigation FACTS docs in `f2_pair_rank/`, `f3_color_b_dev/`, `roc_degeneracy/`, `roy_d_regression/`

The single OPINION doc is `AGENT_PROPOSAL_2026-05-07.md` — read AFTER user authorizes per `AGENTS.md` "Reading order for forming an independent view". The agent retracted one mechanism claim mid-session (GroupDRO balloon → refuted by counter-theory probe); §4 of the proposal documents this for reviewer skepticism calibration.

---

## High-severity / load-bearing open loops (snapshot — see `OPEN_LOOPS.md` for full list)

(Regenerate `OPEN_LOOPS.md` via `python tools/regenerate_open_loops.py` after thread edits.)

- **`trainer-py-1727-wandb-logging-gap`** (resolved 2026-05-07; commit pending) — fix applied locally, needs commit + a future GroupDRO training run to verify scalars flow.
- **`phase-c-hdtf-promotion-contract-failure`** — Phase C 2026-05-07 failed at the contract scoring step; per-suite diagnostic completed but `promotion_contract/` directory never written. Root cause not yet investigated. F4 verdict for P1 was recovered locally by applying Phase A τ to Phase C per-frame reports.
- **`f2-not-testable-from-phase-a`** — F2(a) close criterion ("≥30% relative lift on ≥2 of 6 paired lanes among previously-missed fakes") cannot be evaluated from Phase A substrate: only 1 of 6 yaml-named paired lanes (`deeplive_teams`) has a Phase A proxy + P8A baseline saturated at 0.971-1.000 → 30% lift bar arithmetically unreachable. Tracked in `threads/eval_substrate_layering.md`.
- **`pair-rank-non-paired-lane-collateral`** (NEW 2026-05-07) — Roy_D regression mechanism: pair_rank loss appears to lift fake-class scores on real frames belonging to identities NOT in any paired training lane. Δr=+0.71 mirror of P8A's color_b_dev signal; shared between BUNDLE and PAIRRANK arms. Owning thread: [`threads/pair_rank_collateral.md`](threads/pair_rank_collateral.md).
- **`contract-policy-bug-fix-not-committed` component 3** (resolvable 2026-05-07) — recall-floor tier mechanism worked correctly under P1; τ=0.999x for BUNDLE_step3750/4000 was ROC degeneracy (0 grid points hit recall ≥ 0.30 within FPR ≤ 0.07) not a contract bug. Component 3 of this loop closes; thread `threads/contract_policy_bug.md` to update.
- **`grouped-manifest-v2-stale-paths`** — 5 of 9 Phase A.5 diagnostic substrates have stale `gs://local/...` placeholder paths; cannot be re-evaluated until manifest is regenerated.

---

## Decisions pending (mapped to user)

1. **Authorize commits** for the trainer.py:1727 fix + phase_d regex fix. Both are low-risk; trainer fix is a logging-only change; phase_d fix matches the `_FIXED.csv` outputs the agent already verified.
2. **Authorize next packet** direction. Three candidate frames live in `AGENT_PROPOSAL_2026-05-07.md` §6:
   - **Loss-function**: per-identity score-distribution preservation auxiliary loss (catches Roy_D-class regressions before training ends).
   - **Training-base**: multi-FT-base study (P8A vs E2B vs CLIP-direct) — tests whether FT-from-FT chain depth matters.
   - **Investigation-only**: P8A train-set membership audit for chronic-6 + Roy_D — cheap; refines the §3 mechanism claim.
3. **Authorize Phase C re-run** (or local F4-from-cached-reports) — current F4 verdict was recovered without the cloud `promotion_contract/` step. A re-run would close the loop with proper artifacts; or accept the local verdict and close the open loop.
4. **Authorize wiki contract finalization** — this session's restructure (rolling STATE, FACTS/OPINIONS, eval_folder_template, AGENT_GUIDE move, AGENTS.md extension, hard stop on HANDOFFs). The user pre-authorized the plan; commit at end-of-session if everything works.

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
