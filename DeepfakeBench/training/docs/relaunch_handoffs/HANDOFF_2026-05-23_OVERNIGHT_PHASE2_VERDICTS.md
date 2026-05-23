# HANDOFF 2026-05-23 (overnight) — Phase 2 verdicts + T5C rerun in flight

## Tl;dr for the morning

Phase 2 results (COMPLETE, Phase 3 verdict = deploy $0 face-pool inference baseline):

- **HEAD** (face-pool head-only retrain): **ITERATE → Phase 4 HEAD ALT.** Plateaued at step 250 (1,026 trainable params), composite 0.235 vs ≤ 0.20 gate, viso stuck at 7%. Pareto-equivalent to the $0 face-pool inference baseline. Doesn't deploy.
- **BACKBONE-SlotAv2** (GroupDRO substrate-balanced): **ABORT.** Composite 0.502 vs base 0.331 (abort threshold 0.381 → triggered). Viso recall 0.5%. Conservative-boundary collapse. No rerun.
- **BACKBONE-T5C** (asymmetric pair-loss): **BLOCKED.** Three smokes (images 1.3.299 / 1.3.300 / 1.3.301) all read substrate_pair_asymmetric_loss=0. Three root causes debugged: (1) .gcloudignore CSV missing → fixed; (2) DataLoader workers missing global stamper → fixed via worker_init_fn (`c2ed748`); (3) STRUCTURAL: sampler doesn't pair-group wrappers, matched (clean, teams) pairs effectively never co-occur in same batch → NOT fixable overnight. Auto-launcher saved ~$100 of GPU on blocked T5C fulls.

**Phase 3 verdict**: all Phase 2 arms fail. **Deployment recommendation = Slot A v2 step3500 + face-pool inference ($0 baseline)**, composite λ=1.0 = 0.249. Phase 4 HEAD ALT (dual-readout face+non_face → 1024-dim head) is the highest-EV next experiment.

Phase 3 ranking (composite λ=1.0 ascending = better):

| Rank | Arm | composite | Status |
|---:|---|---:|---|
| 1 | HEAD step 250 | 0.235 | ITERATE → Phase 4 |
| 2 | **Slot A v2 step3500 + face-pool inference ($0)** | **0.249** | **DEPLOY (recommendation)** |
| 3 | Slot A v2 step3500 CLS-pool (training base) | 0.331 | — |
| 4 | BACKBONE-SlotAv2 step 3500 | 0.502 | ABORT |
| — | BACKBONE-T5C step 3500 | BLOCKED | sampler-design issue |

HEAD step 250 is marginally better on composite (-0.014) but Pareto-equivalent on lockbox metrics; the $0 baseline is preferred because it's the same ckpt with no training cost and no introduction of new head weights. HEAD ALT (Phase 4 Fallback A) is the highest-EV next experiment.

## T5C debug chain landed at sentinel `_t5c_relaunch_v3_blocked.json`

Three smoke attempts, all loss=0:
- **v1** (image 1.3.299, W&B `4ruwa8by`) — .gcloudignore CSV missing
- **v2** (image 1.3.300, W&B `2g6nlncj`) — DataLoader workers missing stamper
- **v3** (image 1.3.301, W&B `ytmi23qn`) — sampler doesn't pair-group wrappers

Auto-launchers correctly blocked T5C full launches on v1 + v3 (saved ~$100). Root-cause analysis written to `~/.claude/projects/.../memory/project_backbone_t5c_blocked_pair_sampling_design_2026-05-23.md`.

Unblocking T5C requires sampler redesign (pair-grouped sampling) — out of overnight scope.

## Files that landed overnight (commits)

- `e632767` Phase 2 HEAD verdict ITERATE + .gcloudignore inventory CSV bug fix + new yamls + new analysis tooling
- `ddaf156` BACKBONE-SlotAv2 ABORT + Phase 3 partial verdict
- `9676536` TIMELINE.md append

New memories:
- `feedback_gcloudignore_analysis_csv_pattern.md` — rule: every yaml referencing `analysis/<sidecar>` needs an explicit allow entry in `.gcloudignore` to survive Cloud Build
- `project_head_face_pool_verdict_iterate_2026-05-23.md` — HEAD ITERATE rationale + Phase 4 path
- `project_backbone_slotav2_groupdro_abort_2026-05-23.md` — SlotAv2 mechanism + abort rationale + future GroupDRO recommendations (smaller β, clip_max < 1.0)

## Open loops the morning user should look at

1. **T5C v2 smoke loss check** — if smoke v2 also reads loss = 0 despite the inventory CSV being in image 1.3.300, there's a deeper bug (maybe in `SubstratePairStamper` registration or in the collate's substrate_pair_id passthrough). Check `analysis/phase_2_backbone_2026-05-22/_t5c_relaunch.log` for the loss-check verdict line.
2. **HEAD ALT** (Phase 4 Fallback A) — the live next lever. Requires: detector code change (dual face_pool + non_face_pool → 1024 concat → 1024-dim head), new yaml, smoke + full. Cost ~$33. Not started overnight (significant dev work + user review desirable).
3. **GroupDRO at lower β** — IF the user wants to retry GroupDRO on Slot A v2 with `beta=0.05` + `clip_max=0.3` + inventory CSV fix, that's a new packet. Cost ~$55.

## Live state at handoff

- Cumulative spend: ~$120 of $150 budget ($13 HEAD + $3 SlotAv2 smoke + $3 T5C smoke v1 + $50 SlotAv2 full + ~$50 projected T5C full v2)
- 3rd GPU slot UNUSED
- Image current: `1.3.300` (has substrate_pair_geometry CSV)
- `.gcloudignore` has explicit allows for HDTF + QCLIP inventory CSV path
- Background polls: PID 90931 (T5C chain), bgvm9rrlf (Claude task watching sentinel)
- `find_wandb_run_id.py` + `check_t5c_smoke_loss.py` + `complete_phase_2_backbone.sh` + `cls_pool_scorer_2026-05-22/` in tree; reusable for future arms

## Recommended next packet (operator pick)

In order of expected-value:

1. **Phase 4 HEAD ALT (dual-readout)** — most directly addresses viso ceiling found in CPU-1 + HEAD verdicts. ~$33, ~4h wall. (Requires detector dev.)
2. **GroupDRO retry with dampening** — IF you want to revisit substrate-balancing. ~$55, ~4h. (Lower priority given the conservative-boundary mechanism reading.)
3. **Roy_D-specific anchor pool** — Phase 4 Fallback C from the original plan. Doesn't address viso but addresses the chronic-identity FP tail. ~$30, ~4h.

If T5C v2 lands a deploy result before morning, all of the above become lower priority and the operator decision is "ship T5C".

Cumulative confidence (subjective): HEAD ALT 35-50% to clear all 4 gates; GroupDRO retry 10-20%; Roy_D anchor 25-35% to materially reduce Roy_D's chronic FP without breaking elsewhere.
