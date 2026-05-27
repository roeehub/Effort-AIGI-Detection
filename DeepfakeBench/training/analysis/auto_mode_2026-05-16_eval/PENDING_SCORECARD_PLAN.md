# PENDING_SCORECARD_PLAN — auto-mode 2026-05-16

> Pre-scorecard plan written 2026-05-16 evening while GPU jobs Slot A + Slot B
> are training. This document captures the analysis pipeline to execute once
> Vertex jobs complete and periodic checkpoints are saved to GCS.

## §1. Execution sequence after both jobs SUCCEEDED

1. **Fill ckpt map placeholders** (~30s):
   ```bash
   cd analysis/slot_b_property_shortcut_2026-05-16
   python3 scripts/fill_ckpt_map.py --auto
   ```
   Verifies that step1500 + step3500 periodic ckpts exist for both runs and updates `arena/checkpoint_maps/teams_target_domain.auto_mode_2026-05-16.yaml`.

2. **Launch the scorecard** (~2.5h Vertex):
   ```bash
   cd /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training
   SUITE_MANIFEST=arena/target_domain_suites.teams_promotion_contract_minimal_9suite_2026-05-14.yaml \
   CHECKPOINT_MAP=arena/checkpoint_maps/teams_target_domain.auto_mode_2026-05-16.yaml \
   REGION=us-east1 \
   ./arena/launch_teams_promotion_contract.sh -y
   ```
   6 ckpts × 9 suites = 54 cells. ETA ~2.5h.

3. **While scorecard runs, pre-stage post-scorecard analysis**:
   - Download the scored periodic_step3500 ckpts to local
   - Pre-build the encoder re-probe scripts for Roy_D / ilan / orel against trained encoders (vanilla openclip + state_dict overlay since SVD-aware loader was deferred)

4. **After scorecard SUCCEEDED**:
   - Download `promotion_contract/checkpoint_summary.csv` + `selected_threshold_scorecard.csv` + per-frame reports
   - Run `analysis/slot_b_property_shortcut_2026-05-16/scripts/tiebreak_reranking.py` adapted to the new csv
   - Run per-identity decomposition on the new ckpts using the existing scripts (per-identity Slot A/B over-fires on lockbox + dev PNG)
   - Compute the v6 §4 pre-defined falsifiers/confirmers

## §2. Pre-defined verdict rubric (set BEFORE scorecard)

### Slot A — anchor_aware

| outcome | criteria | next action |
|---|---|---|
| **CONFIRM** (deployment-grade) | dev_fake_macro_recall ≥ 0.40 AND lockbox_real_fpr ≤ 0.03 AND viso ≥ 0.15 AND dor chronic_FP ≤ 0.40 | Document as first single-lever from T5C that promotes. Ship. |
| **PARTIAL** (mechanism confirmed but not deployable) | dor chronic_FP ≤ 0.40 (mechanism works) AND any of: dev_fake_macro_recall < 0.40, viso < 0.15 | Document mechanism support; consider multi-pool / higher-weight follow-up |
| **NO_TRANSFER** (predicted) | dor chronic_FP improves but Roy_D dev FPR ≈ unchanged from T5C | Confirms bounded-scope prediction; close as documented |
| **NULL** | dev_fake_macro_recall < 0.30 OR no improvement vs T5C on any chronic FP cohort | anchor_aware as single lever does not work on T5C base |

### Slot B — real_rebalance

| outcome | criteria | next action |
|---|---|---|
| **CONFIRM** | dev_fake_macro_recall ≥ 0.40 AND lockbox_real_fpr ≤ 0.03 AND viso ≥ 0.15 AND Roy_D dev FPR ≤ 0.40 | Document as data-side lever that addresses encoder-axis chronic FP. Ship. |
| **PARTIAL** | Roy_D dev FPR ≤ 0.40 (Roy_D mechanism works) AND any of: dev_fake_macro_recall < 0.40, viso < 0.15 | Validates VCD-as-Roy_D-adjacent hypothesis; consider larger VCD boost or compound w/ anchor_aware |
| **NULL** | Roy_D dev FPR unchanged AND no scorecard movement | encoder-axis hypothesis refuted at training-data composition layer |

### Compound (both slots)

If both PARTIAL with disjoint mechanisms, the next packet is the compound (anchor_aware + real_rebalance). If both NULL, encoder-axis intervention is required (different backbone, contrastive loss, etc.).

## §3. Post-scorecard CPU follow-ups (no GPU needed)

Each one is ~30 min to 1h CPU.

1. **Re-embed Roy_D / ilan / orel through vanilla openclip + per-ckpt diff** to see whether FT shifted the PC1 axis. The trained ckpts will have backbone state_dict; we can selectively load the `backbone.visual.transformer.resblocks.*.attn.in_proj_weight` keys (the un-SVD original weights still present) into a vanilla openclip model. This is a simpler version of the SVD-aware loader.

2. **Per-identity lockbox FPR breakdown** for Slot A and Slot B ckpts (analogous to the 2026-05-16 morning per-identity probe).

3. **Band-shortcut readout on new ckpts** — score the 1198 PNG pool on Slot A/B trained ckpts, recompute over-fire by n_bands. If Slot B reduces 6+ band FPR to ≤ 30% (vs Slot β 65-100%), encoder-axis hypothesis confirmed.

## §4. Decision tree on outcomes

```
                        ┌─ Slot A CONFIRM → ship anchor_aware
                        ├─ Slot A PARTIAL → multi-pool follow-up packet
After scorecard ────────┼─ Slot A NO_TRANSFER → close (documented as bounded)
                        └─ Slot A NULL → close (lever refuted)
                        ┌─ Slot B CONFIRM → ship rebalance
                        ├─ Slot B PARTIAL → larger boost or compound packet
                        └─ Slot B NULL → encoder-axis intervention needed
```

## §5. Output structure (planned)

```
analysis/auto_mode_2026-05-16_eval/
├── RESULTS_FACTS_2026-05-16.md         # scorecard tables, mechanical pass/fail
├── DEEP_DIVE_FACTS_2026-05-16.md       # per-identity + band-shortcut readout
└── AGENT_PROPOSAL_2026-05-16.md        # opinion + retraction log + next step

docs/packet_retrospectives/packets/AUTO_MODE_ANCHOR_REBALANCE.md  # retro
```

## §6. Limits / known unknowns

- The encoder probe used vanilla openclip; the trained encoder may shift PC1. The SVD-aware loader was deferred. If scorecard verdict is ambiguous, building the loader becomes higher priority.
- Slot B uses identity-based sampling — boosting family weight does not directly target chronic-6 identities specifically. Memory `project_data_inventory_identity_diversity` notes ~2300 identities exist in unused buckets — VCD has 158 identities per the GCS listing. The boost surfaces more of those into the per-step batch.
- anchor_aware's 30-frame pool is small; supervision signal is bounded.
- Both packets inherit T5C's 4-axis GRL. If GRL itself caused the v3 band-shortcut amplification, both A and B are weak interventions.
