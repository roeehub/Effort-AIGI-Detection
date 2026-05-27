# S1/S2 post-training probes — how to run

Three probes to run after S1 (or S2) training completes. All CPU; expected
combined wall time ~5-10 min per packet.

## Prerequisites (before running probes)

1. Vertex training job has reached `JOB_STATE_SUCCEEDED`
2. Periodic save checkpoints visible in
   `gs://training-job-outputs/phase2r13_experiments/<run_id>/`
3. A promotion contract scorecard has been launched on the chosen ckpts
   (use `arena/launch_teams_promotion_contract.sh` with the
   `teams_promotion_contract_2026-04-23_with_dor.yaml` suite manifest)
4. Scorecard outputs pulled to local at
   `analysis/s1_s2_eval/<run_id>/scorecard/promotion_contract/threshold_grid.csv`
   and `analysis/s1_s2_eval/<run_id>/scorecard/reports/`

## Probe 1 — pick best ckpt by class_separation

```bash
python analysis/s1_s2_2026-05-02_planning/probes/01_select_best_ckpt_by_class_sep.py \
    --run-id <wandb_run_id_for_s1_or_s2>
```

Output: `outputs/01_select_best_ckpt_<run_id>.json` with the recommended
checkpoint path (peak class_separation). USE THIS for the scorecard
checkpoint map, not the val-AUC peak.

## Probe 2 — compute pre-registered falsifiers

```bash
# After scorecard completes
python analysis/s1_s2_2026-05-02_planning/probes/02_compute_falsifiers_s1_s2.py \
    --probe1-json analysis/s1_s2_2026-05-02_planning/probes/outputs/01_select_best_ckpt_<run_id>.json \
    --grid analysis/s1_s2_eval/<run_id>/scorecard/promotion_contract/threshold_grid.csv \
    --frames-dir analysis/s1_s2_eval/<run_id>/scorecard/reports/ \
    --packet S1 \
    --ckpt-key S1_REDUX_BEST_BY_CLASS_SEP \
    --ckpt-lower-pat s1_redux_best
```

For S2 add `--s1-best-class-sep <value-from-S1-probe2-output>`.

Output: `outputs/02_verdict_<packet>_<ckpt_key>.json` with PASS/FAIL on
each of F-A/B/C and the overall verdict.

## Probe 3 — direct comparison vs P22 step1k

```bash
python analysis/s1_s2_2026-05-02_planning/probes/03_p22_redux_diff.py \
    --packet S1 \
    --new-grid analysis/s1_s2_eval/<run_id>/scorecard/promotion_contract/threshold_grid.csv \
    --new-ckpt-key S1_REDUX_BEST_BY_CLASS_SEP
```

Output: figure `figures/03_diff_<packet>_vs_p22_step1k.png` with
recall-vs-FPR curves for viso/deeplive/teams_fake_dev compared
side-by-side; CSV with deltas at each FPR floor.

## Mid-training monitoring (no probe — just wandb dashboard)

While S1/S2 are running, watch these W&B metrics in real-time:

- **`train/collapse/class_separation`** — TARGET ≥ 2.0 throughout. If it drops
  below 1.5 before step 800, the training has collapsed early. Cancel
  the run and inspect.
- **`train/collapse/logit_std`** — TARGET ≥ 1.5. If it drops below 1.0,
  same as above.
- **`train/collapse/prob_entropy`** — TARGET ≤ 0.55. If it rises above 0.60,
  same as above.
- **`train/loss/overall`** — TARGET in [0.4, 1.2]. If it goes above 1.5,
  inspect for divergence.

These metrics are why we trained the trainer to log `train/collapse/*` —
P22's val_AUC was a lagging indicator.

## Cost summary

- Probe 1: ~30 sec (W&B API)
- Probe 2: ~1 min (CSV joins, regression on ~500 viso frames)
- Probe 3: ~30 sec (CSV joins + matplotlib)
- Total CPU: ~2 min per packet
- GPU: 1 promotion-contract scorecard per packet (~$3-5 each)
