# State — current rolling snapshot

> **Last refreshed**: 2026-05-08 00:50 local (P2 Slot D launched on image 1.3.272 — `exp-R13_P2_SCRATCH_FOURIER-20260508-004331`, Vertex `8486657808400384000`, us-east1, transitioned to JOB_STATE_RUNNING at 22:49:06 UTC, 5 min after submit; A/B/C also RUNNING; all 4 P2 slots in flight).
>
> **Purpose**: single-page current-state snapshot. Always-current; rolling. Older dated snapshots archived in [`archive/`](archive/) for historical reference.
>
> **How to use this page**: a new agent reads this AFTER `AGENTS.md` and BEFORE the user's specific task. It tells you what's running, what just landed, what's open, and what decision the user might want next. If the wall-clock is materially after the "Last refreshed" date, verify against `git status`, the tail of [`TIMELINE.md`](TIMELINE.md), and the latest [`packets/`](packets/) entry — state moves quickly.

---

## Where we stand right now (one paragraph)

The most recently completed packet is **[P1](packets/P1.md) (PE_PAIR_RANK_DRO)** — pair_rank λ=0.2 + multi-axis GroupDRO with `chronic_flag` (BUNDLE) vs pair_rank-only (PAIRRANK), both FT-from-P8A_step5000 on the post-`2feea58` codepath. Phase A (29-suite contract scorecard) **SUCCEEDED**; Phase C (16-suite HDTF cross-substrate) **FAILED** at the promotion_contract step (per-suite diagnostic completed; verdict bundle never written; root cause not yet investigated). At the contract-selected τ, **no P1 ckpt clears F1 (90% lockbox recall)**; the contract's rank-1 winner is `P1_PAIRRANK_PERIODIC_STEP500` with 70.8% lockbox recall at τ=0.768. **F1 is reachable at non-contract τ** — `BUNDLE_step500` hits 96.5% lockbox recall at FPR ≤ 10% (τ=0.989); `PAIRRANK_step500` hits 91.5% at τ=0.50. **F5 PASSES big** for the BUNDLE arm: PC_Generator chronic FPR drops 62.9% → 0-3%. **F4 PASSES** universally at calibrated τ (max HDTF real FPR 1.4%). **F3 is partial**: 3 of 4 untargeted IQ axes show decoupling (sharpness, min_dim, color_b_dev) but face_area_fraction amplifies (+106-266%). **P1 introduced a Roy_D regression**: 130 frames flip from 29% FPR (P8A) to 78-93% FPR (P1); the regression is `color_b_dev`-aligned (Δr=+0.71 mirror of P8A's r=−0.71) AND **shared between BUNDLE and PAIRRANK arms** (Wilcoxon p=0.875), refuting the initial GroupDRO-balloon hypothesis. **Deployment is still E2B** per memory `project_deployment_is_e2b_2026-05-06.md`; E2B beats every P1 ckpt on `dev_fake_macro_recall` (0.508 vs 0.354). This session also delivered two bug fixes (trainer.py:1727 W&B logging gap + phase_d/run_chronic_filter.py regex) and a wiki contract upgrade (rolling STATE, FACTS/OPINIONS file convention, hard stop on per-session HANDOFF docs).

---

## In flight / running right now

**P2 packet** (PREVENT_NOT_UNLEARN, 4 slots) — **Slots A/B/C launched 2026-05-07 23:33 local on image 1.3.271; Slot D launched 2026-05-08 00:43 local on image 1.3.272** (Slot D Cloud Build `f66bedcb-6dc6-4bca-b6ff-9335c8244d3a`, 16m25s, commit `27e95b8`). All 4 slots are FROM-SCRATCH (CLIP-B16 init, no FT base) on the post-quality_enhancement-fix data composition with the canary probe enabled @ frequency_steps=1000:

- **Slot A** [`R13_P2_SCRATCH_BUNDLE`](../../experiments/phase2_round13/R13_P2_SCRATCH_BUNDLE.yaml) — 4-axis corr_penalty (sharpness + luma + face_area + color_b_dev) + pair_rank_loss + face_scale_jitter@0.50. Vertex `8395459915946131456` (us-east1, `JOB_STATE_RUNNING` since 21:38:26 UTC). W&B `u22wz1vf` (`R13_P2_SCRATCH_BUNDLE_0507-2138`).
- **Slot B** [`R13_P2_SCRATCH_CORR_ONLY`](../../experiments/phase2_round13/R13_P2_SCRATCH_CORR_ONLY.yaml) — 4-axis corr_penalty only (single-lever ablation). Vertex `5580112770428305408` (us-west4, `JOB_STATE_RUNNING` since 21:42:11 UTC). W&B `mlo5vfe8` (`R13_P2_SCRATCH_CORR_ONLY_0507-2142`).
- **Slot C** [`R13_P2_SCRATCH_PAIRRANK_ONLY`](../../experiments/phase2_round13/R13_P2_SCRATCH_PAIRRANK_ONLY.yaml) — pair_rank only (matched against P1 PAIRRANK_ONLY but from-scratch). Vertex `3126673777023254528` (us-central1, `JOB_STATE_RUNNING` since 21:36:36 UTC). W&B `oaur8odo` (`R13_P2_SCRATCH_PAIRRANK_ONLY_0507-2137`).
- **Slot D** [`R13_P2_SCRATCH_FOURIER`](../../experiments/phase2_round13/R13_P2_SCRATCH_FOURIER.yaml) — band-limited Fourier amplitude aug (bands 8-13 randomize, 5-6 preserve, log_range [-0.3,+0.3]) + face_scale_jitter@0.50; no corr_penalty, no pair_rank. Vertex `8486657808400384000` (us-east1, `JOB_STATE_RUNNING` since 22:49:06 UTC; 5 min PENDING). Display name `exp-R13_P2_SCRATCH_FOURIER-20260508-004331`. W&B run id pending first-step log emission.

W&B project URL: https://wandb.ai/dtect-vision/phase2-round13. Step counts at 22:05 UTC for the original 3 slots: A 501, C 314, B 0 (in dataset discovery — VisoMaster Enhanced loading). Canary first-fire is at step 1000; A closest. No canary readouts yet.

Per CLAUDE.md region-distribution: A/D on us-east1, B on us-west4, C on us-central1. Slot D's 30-min PENDING-threshold check is closed: D transitioned to RUNNING within 5 min of submit; no region-failover needed for any slot. Per packet retro [`packets/P2.md`](packets/P2.md). Cost ~$60-80 each, $240-320 total.

**Slot D task spec** at [`SLOT_D_FOURIER_TASK_2026-05-08.md`](SLOT_D_FOURIER_TASK_2026-05-08.md) — was the operational handoff for this slot. Smoke notes at [`analysis/p2_eval_2026-05-08/SLOT_D_SMOKE_FAIL.md`](../../analysis/p2_eval_2026-05-08/SLOT_D_SMOKE_FAIL.md) (3-tier smoke; Tier-3 mean_abs floor relaxed 1.0→0.4 per user authorization 2026-05-08 because faces have low FFT amplitude in bands 8-13 — the aug produces a measurable, bounded effect on every cohort but averages 0.50-0.76 per pixel rather than the spec's 1.0 floor calibrated to uniform-noise synthetic). Image 1.3.272 = image 1.3.271 + commit `27e95b8` (fourier_band_aug primitive + Slot D yaml + collate-fn wiring + train_sweep allowlist).

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

[`analysis/p1_pe_eval_2026-05-07/`](../../analysis/p1_pe_eval_2026-05-07/) — first folder following the [`eval_folder_template.md`](eval_folder_template.md) FACTS/OPINIONS contract. New agents read in order:

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
