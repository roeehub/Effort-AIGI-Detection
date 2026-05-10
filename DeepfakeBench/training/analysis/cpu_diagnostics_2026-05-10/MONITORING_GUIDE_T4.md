# T4 Live Monitoring Guide (auto-mode reference)

Once the two T4 Vertex jobs are RUNNING, these are the W&B metrics that
matter and the early-warning patterns to watch for. This is for me (Claude)
to use as a checklist while monitoring; user can also use it to make sense
of the W&B dashboard if they check in.

W&B project: https://wandb.ai/dtect-vision/phase2-round13
Filter: tag `t4-multi-axis-grl`

---

## Critical metrics (per training step, every 50 steps)

### λ schedule sanity
- `train/multi_axis_grl_lambda` — should ramp linearly 0.0 → λ_max over the first 500 steps, then flat. λ_max=1.0 for slot 1; λ_max=2.0 for slot 2.
- If this stays at 0.0, the trainer's `_update_multi_axis_grl_lambda` isn't firing → experiment is broken.

### GRL bite signal (the load-bearing question)
- `train/multi_axis_grl_acc_chronic_flag`
- `train/multi_axis_grl_acc_is_dor`
- `train/multi_axis_grl_acc_sharpness_laplacian_high`
- `train/multi_axis_grl_acc_color_a_approx_dev_high`

**Reading**: at step 0 (with random classifier heads on encoder features), accuracy ≈ 0.5 + epsilon for binary axes. As the classifier heads learn to read shortcut info from L11 features they'll RISE above 0.5. As the GRL forces the encoder to STOP encoding shortcut info, accuracy DROPS back toward 0.5.

- **Healthy bite**: accuracy peaks early (0.7-0.9 range by step 500-1000), then drops toward 0.55-0.65 as the encoder removes shortcut.
- **Failed bite**: accuracy stays high (0.85-0.95) throughout — the encoder can encode shortcut faster than the classifier can learn under GRL pressure.
- **No information**: accuracy stays at 0.5 throughout — likely a label issue (axis isn't distinguishable in this batch composition).

### Forgery preservation
- `train/cls_loss` — main forgery CE loss. Should drop normally.
- `train/auc` (or `train/raw_logits`-derived) — should reach 0.95+ by step 2500.
- `train/multi_axis_grl_loss` — should INITIALLY rise (CE on classifier heads), then plateau or slightly decline.
- `train/multi_axis_grl_loss_raw` — without the 1.0 weight; same pattern.

### Eval (every 250 steps)
- Various dev / target_domain metrics emitted by the standard eval scaffolding.
- Most informative early: `dev/auc` cell, `dev/teams_fake_*`, `dev/visomaster_enhanced_*_recall`.

---

## Early-warning patterns (should trigger user notification)

| Pattern | What it means | Action |
|---|---|---|
| `cls_loss` spike at step >500, AUC drops below 0.85 | λ too aggressive (especially likely for slot 2 λ=2.0) | Note + continue; if AUC still <0.85 by step 1500, recommend cancel |
| All shortcut accs at 0.5 throughout | GRL is biting too hard OR labels are degenerate | Check label distribution in W&B histograms |
| Shortcut accs at 0.85+ throughout, no drop | GRL not biting at this attachment | T4 mainline missed; pivot signal |
| `multi_axis_grl_loss` exactly 0 | Loss not getting added (config didn't propagate) | Check `train/multi_axis_grl_n_axes_active` — if 0 → wiring issue |
| Job FAILED before step 500 | Likely yaml/data path issue | Check Vertex logs |

---

## Periodic ckpt evaluation (post-training)

After training completes, the 7 periodic ckpts at {100, 500, 1000, 1500, 2500, 3500, 5000} land in:
`gs://training-job-outputs/phase2r13_experiments/<wandb_run_id>/`

For each ckpt, run the existing scoring scaffold:
1. Score on F4 substrate (drop chronic-6 + lowres + no-face) — viso recall headline.
2. Score on may6 frames (92 Xinhe frames) — false-flag count.
3. Re-extract L11 atlas features → measure `inv_mean` (should be ≥ 0.05 if mechanism worked).
4. Per-axis: chronic_6 AUC at L11 (should drop from P8A's 0.909 toward 0.55-0.65).

Best ckpt = max over (forgery_AUC ≥ 0.97 AND viso ≥ 70% AND may6 ≤ 6/92 AND inv_mean ≥ 0.05).

---

## Cancellation policy

Per CLAUDE.md memory `feedback_no_cancelling_vertex_jobs.md`: I will NOT cancel any Vertex job without explicit user authorization. If a job is clearly failed (FAILED state) I'll just note it and stop monitoring it. If a job is running but trending badly, I'll surface the concern + recommended action and wait for user.

Region rotation rule: if PENDING > 30 min, queue same job in another US region; cancel original after replacement reaches RUNNING (per CLAUDE.md). This rotation IS authorized in the standing instructions.
