# Phase 2 BACKBONE-SlotAv2 — Results FACTS (2026-05-22)

> **Status: factual-only, scope-limited.** Banned-word check applies: succeeds, fails, wins, loses, promotes, deployment-grade, ship, kill, best, worst, confirmed, refuted, gap-is-wide, gap-is-narrow.
>
> **Scope.** This document records the code-edit phase of BACKBONE-SlotAv2 (GroupDRO substrate-balanced on Slot A v2 step3500). Vertex smoke + full training + scorecard + Probe 1 re-fit are PENDING. No GPU compute has been spent.

---

## 1. Code edits committed to the working tree

| Path | Change |
|---|---|
| `data/sample/__init__.py` | NEW. Package init for `data.sample`. |
| `data/sample/substrate_paired.py` | NEW. `SubstratePairStamper` + module-level singleton stamper. Reads `analysis/substrate_pair_geometry_2026-05-22/inventory_manifest.csv` (1,880 rows; 759 unique identities). |
| `loss/substrate_pair_asymmetric.py` | NEW. `SubstratePairAsymmetricLoss` registered under `module_name='substrate_pair_asymmetric'`. Used by BACKBONE-T5C (BACKBONE-SlotAv2 keeps it disabled). |
| `loss/__init__.py` | Import of `SubstratePairAsymmetricLoss` for side-effect registration. |
| `trainer/trainer.py` | Init: instantiates `SubstratePairAsymmetricLoss` from `config['substrate_pair_asymmetric_loss']` (None when disabled) AND installs the `SubstratePairStamper` as the process-global stamper. Loss accumulation site (line ~1854): adds `substrate_pair_asymmetric` term to `losses['overall']` when the loss object is set. |
| `data/sources/combined_paired.py` | Collate (`combined_paired_collate_fn`) now reads the active stamper and emits `substrate_pair_id` (LongTensor[B]) and `substrate_transport` (LongTensor[B]) per video. Default is (-1, -1) — no-op when stamper is disabled. |
| `train_sweep.py` | W&B-flattening re-apply allowlist updated for `substrate_pair_asymmetric_loss` and `combined_paired.substrate_pair_sampling`. |
| `experiments/phase2_round13/R13_GROUPDRO_SUBSTRATE_SLOTAV2_2026-05-26.yaml` | NEW. BACKBONE-SlotAv2 packet yaml. |
| `tests/test_substrate_paired_and_asymmetric_loss.py` | NEW. 20 unit tests covering stamper + loss + collate end-to-end. |
| `tests/test_train_sweep_reapply_allowlist.py` | `substrate_pair_asymmetric_loss` added to `TRAINER_NESTED_KEYS`. |

---

## 2. Yaml configuration (BACKBONE-SlotAv2)

`experiments/phase2_round13/R13_GROUPDRO_SUBSTRATE_SLOTAV2_2026-05-26.yaml`:

- `name`: R13_GROUPDRO_SUBSTRATE_SLOTAV2_2026-05-26
- `seed`: 9915
- `load_base_checkpoint`: true
- `gcs_base_checkpoint`: gs://training-job-outputs/best_checkpoints/hp35c51p/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth (Slot A v2 step3500)
- `total_training_steps`: 3500
- `periodic_saves.step_list`: [100, 500, 1000, 1500, 2500, 3500]
- `use_group_dro`: true; `group_dro_params`: {beta=0.1, ema_alpha=0.9, warmup_steps=200, clip=[0.05, 1.0]}
- `substrate_pair_asymmetric_loss.enabled`: false (this packet does NOT use the pair-hinge — it uses GroupDRO; the pair-hinge is for BACKBONE-T5C)
- `combined_paired.substrate_pair_sampling.enabled`: true (stamper active for downstream observability)
- `anchor_aware.enabled`: true (inherited from Slot A v2 base; weight=5.0, target=0.10, samples=16)
- `multi_axis_grl.enabled`: true (inherited from Slot A v2 base)

GroupDRO grouping note. The yaml does NOT specify a manual `data_params.group_id_mapping` because the existing combined_paired pipeline auto-builds the asymmetric R-D / F-B group_id at config-load time (`data/sources/combined_paired.py::build_group_id_mapping_for_samples`) and transfers it into `config['data_params']` via `train_sweep.py:609-615`. The asymmetric key encodes substrate transport (`teams_capture` vs `raw_capture` vs `visomaster` vs `external`) AND source AND chronic-vs-regular AND quality, so the substrate axis is already balanced.

---

## 3. Unit test results

`tests/test_substrate_paired_and_asymmetric_loss.py` — 20 tests, all passing on the host CPU at 2026-05-22:

```
tests/test_substrate_paired_and_asymmetric_loss.py ....................  [100%]
20 passed in 3.46s
```

Broader regression sweep (138 tests across `tests/test_train_sweep_reapply_allowlist.py`, `tests/test_pipeline_randomization.py`, `tests/test_unpaired_reals_and_grl.py`, `tests/test_anchor_aware_penalty.py`, `tests/test_pair_rank_and_group_dro.py`, `tests/test_face_scale_jitter.py`, `tests/test_resolution_chain_aug.py`, plus the new file):

```
138 passed in 15.51s
```

Notable invariants verified by the unit tests:

1. Inventory load. The stamper reads 1,880 rows and assigns one `pair_id` per row; 759 unique identities surface; per-source totals match `INVENTORY_FACTS_2026-05-22.md` Table 3.1 (hdtf=1094, quickclips=732, viso_teams_enhanced=54).
2. Identity-only matching. Same identity from clean source + teams source emits the same `pair_id` and differing `substrate_transport` (0 vs 1).
3. `companion_domain == 'teams_v2'` forces `substrate_transport = 1` even when the source label looks clean.
4. Fake-side rows (label=1) and unknown identities both emit (-1, -1).

---

## 4. Inventory coverage caveat (documented gap)

The inventory CSV enumerates 1,880 paired identity-rows across three sources: `hdtf_visomaster_teams`, `quickclips_visomaster_teams`, `visomaster_teams_enhanced`. Of these:

- `visomaster_teams_enhanced` (54 rows) is already wired into the training data pipeline via the resolver manifest (`visomaster_teams_enhanced` lane in combined_paired.py). Frames from this lane will carry the stamped `substrate_pair_id` and `substrate_transport`.
- `hdtf_visomaster_teams` (1,094 rows) and `quickclips_visomaster_teams` (732 rows) are NOT currently loaded by any existing iterator. Their GCS buckets (`hdtf_visomaster_cropped_frames` and `quickclips_visomaster_cropped_frames`, with `_teams` companions) sit unread by combined_paired.py.

Therefore the practical pair-fraction at training time is bounded by the `visomaster_teams_enhanced` lane's contribution (~3% of inventory rows; per-batch substrate-paired-real fraction is small).

For BACKBONE-SlotAv2 (GroupDRO substrate-balanced), this is acceptable: GroupDRO does not require matched pairs — it requires accurate group_id assignment, which the existing asymmetric R-D / F-B key already provides via the transport axis. The substrate-pair stamping is observability-only for SlotAv2.

A follow-on packet would add `hdtf_visomaster_teams` and `quickclips_visomaster_teams` as new data-source lanes in combined_paired.py to lift the practical pair-fraction. Estimated effort: ~3-4h CPU dev + smoke; not in this packet's scope.

---

## 5. Vertex submission (PENDING)

No Vertex job has been submitted. The yaml + production code are ready for `dev.sh build-prod -y` + `arena/launch_*.sh` invocation when the user authorizes.

Expected smoke cost: ~$3 (us-west4, A100×1, 200 steps with periodic_saves at [100]).
Expected full-run cost: ~$45-65 (us-west4, A100×1, 3500 steps, ~3.5h).

---

## 6. Scorecard (PENDING)

The standard contract scorer (`arena/score_teams_promotion_contract.py`) is unmodified. Once Vertex produces checkpoints at `gs://training-job-outputs/best_checkpoints/<run_id>/periodic_effort_*_step{100,500,1000,1500,2500,3500}_*.pth`, the scorer can be invoked under both lex and composite λ=1.0.

For BACKBONE-SlotAv2, the additional face-pool scorer at `analysis/face_pool_scorecard_2026-05-22/score_face_pool_suites.py` will measure the CLS-pool variant alongside the face-pool variant per the operator decision in the plan.

---

## 7. Post-hoc Probe 1 KLIEP re-fit (PENDING)

`analysis/substrate_pair_geometry_2026-05-22/run_probe1_kliep_refit.py` is unmodified and can be re-invoked on the trained BACKBONE-SlotAv2 ckpt L11 features once available. Close criterion per the task: per-ckpt classifier accuracy on the substrate axis ≤ 0.85 (down from 0.98) is the mechanism-worked signal.

---

## 8. Sentinel file

`_phase_2_backbone_slotav2_complete.json` — not written. Will be created on Vertex completion + scorecard finalization.
