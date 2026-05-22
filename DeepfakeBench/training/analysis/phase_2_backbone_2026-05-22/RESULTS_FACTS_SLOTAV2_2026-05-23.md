# Phase 2 BACKBONE-SlotAv2 — FACTS (auto-generated draft, will fill numbers after scoring)

> Banned-word policy: this FACTS doc uses no opinion verbs about outcomes
> (no "succeeds", "fails", "wins", "loses", "promotes", "deployment-grade",
> "kill", "best", "worst", "confirmed", "refuted"). Interpretation lives in
> the companion AGENT_PROPOSAL doc.

---

## Lever

GroupDRO substrate-balanced on Slot A v2 step3500 base (anchor_aware ON,
multi_axis_grl ON, all other levers inherited unchanged from
`R13_T5C_ANCHOR_AWARE_2026-05-16.yaml`). Single-lever delta:

```yaml
use_group_dro: true
group_dro_params:
  beta: 0.1
  ema_alpha: 0.9
  warmup_steps: 200
  clip_min: 0.05
  clip_max: 1.0
```

Substrate-pair stamping `substrate_pair_sampling.enabled: true` and
HDTF+QCLIP inventory lane `substrate_paired_inventory.enabled: true` were
both enabled in the yaml. **NOTE (root-cause finding 2026-05-23)**: image
`1.3.299` was missing the inventory CSV
(`analysis/substrate_pair_geometry_2026-05-22/inventory_manifest.csv`) from
the Cloud Build tarball — `SubstratePairStamper` and the inventory loader
both disabled themselves at runtime with a soft warning. The GroupDRO
group_id_mapping was therefore built from the existing data lanes' groups
(R-D / F-B / source-based) but did NOT include the 1,826 additional
HDTF+QCLIP substrate-paired rows. **This run tested GroupDRO on the
existing data lanes only — NOT the full substrate-balanced design.**

---

## Vertex run

| Field | Value |
|---|---|
| Vertex job id | `6793621763772121088` |
| W&B run id | `6ypu1ds3` |
| W&B URL | `https://wandb.ai/dtect-vision/effort-r13-phase2/runs/6ypu1ds3` |
| Image | `1.3.299` (missing inventory CSV — see Lever note above) |
| Region | us-west4 |
| GPU | NVIDIA_TESLA_A100 x 1 |
| Submitted | 2026-05-22T17:51:02Z |
| RUNNING | 2026-05-22T17:52:44Z (~1m 42s PENDING) |
| SUCCEEDED | 2026-05-22T21:54:13Z (4h 01m wall) |
| GPU cost estimate | ~$50 (A100 us-west4 × ~4h at $12.50/h) |

### Periodic checkpoints (gs://training-job-outputs/best_checkpoints/6ypu1ds3/)

| Step | Checkpoint | Holdout AUC | Holdout EER |
|---:|---|---:|---:|
| 250 (top_n) | `top_n_effort_20260522_step250_auc0.9968_eer0.0220.pth` | 0.9968 | 0.0220 |
| 500 (periodic) | `periodic_effort_20260522_step500_auc0.9969_eer0.0160.pth` | 0.9969 | 0.0160 |
| 1000 (periodic) | `periodic_effort_20260522_step1000_auc0.9953_eer0.0220.pth` | 0.9953 | 0.0220 |
| 1500 (periodic) | `periodic_effort_20260522_step1500_auc0.9802_eer0.0760.pth` | 0.9802 | 0.0760 |
| 2500 (periodic) | `periodic_effort_20260522_step2500_auc0.9942_eer0.0340.pth` | 0.9942 | 0.0340 |
| 2750 (top_n) | `top_n_effort_20260522_step2750_auc0.9972_eer0.0160.pth` | 0.9972 | 0.0160 |
| 3500 (periodic) | `periodic_effort_20260522_step3500_auc0.9968_eer0.0220.pth` | 0.9968 | 0.0220 |
| first_best ep1 | `first_best_effort_20260522_ep1_auc0.9968_eer0.0220.pth` | 0.9968 | 0.0220 |

Holdout AUC trajectory dipped to 0.9802 at step 1500 then recovered to
0.9968 by step 3500. Slot A v2 base (hp35c51p step3500) had auc 0.9952; the
GroupDRO run ended slightly above base on holdout but holdout AUC is NOT
the deployment-grade signal (see HEAD packet's similar caveat).

---

## Scorecard (per-ckpt) — step 3500 first, additional steps conditional

Scored with the **CLS-pool** scorer
(`analysis/cls_pool_scorer_2026-05-22/score_cls_pool_suites.py`) — same I/O
contract as the face-pool scorer but with no monkey-patch (standard pooling
since the BACKBONE ckpt was trained without face_pool_readout).

| ckpt | step | τ | lockbox_real_fpr | lockbox_fake_recall | viso_enhanced_dev recall | deeplive_enhanced_dev recall | dev_fake_macro_recall | composite λ=1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Slot A v2 step3500 (CLS, base) | — | — | 0.0191 | 0.6877 | 0.1673 | 0.420 | 0.310 | 0.331 |
| Slot A v2 step3500 (face-pool, $0 baseline) | — | — | 0.0154 | 0.7668 | 0.0673 | 0.552 | 0.439 | 0.249 |
| BACKBONE-SlotAv2 6ypu1ds3 step 3500 | 3500 | 0.8265 | 0.0118 | 0.5099 | 0.0055 | 0.5340 | 0.3529 | 0.5019 |

Per-suite metrics at BACKBONE-SlotAv2 step 3500 (composite winner, calibrated τ=0.8265):
- `teams_real_all_dev`: dev_primary_real_fpr 0.0676 (220/3253 reals above τ)
- `teams_real_poor_quality_dev`: real_fpr per stress threshold
- `teams_real_lighting_extreme_dev`: real_fpr (worst stress) 0.0999
- `teams_real_all_lockbox`: real_fpr 0.0118 (16/1361)
- `teams_fake_all_dev`: fake_recall 0.5193 (1251/2409)
- `teams_fake_all_lockbox`: fake_recall 0.5099 (129/253)
- `visomaster_enhanced_macro_dev`: fake_recall 0.0055 (3/550)
- `deeplive_enhanced_dev`: fake_recall 0.5340 (291/545)
- dev_fake_macro_recall: 0.3529

Gate readout vs yaml header's four-gate decision criterion:
- composite ≤ 0.25: FAIL (0.502, over by 0.252)
- lockbox_real_fpr ≤ 0.018: PASS (0.0118)
- lockbox_fake_recall ≥ 0.70: FAIL (0.510, under by 0.190)
- viso recall ≥ 0.15: FAIL (0.005, under by 0.145)
- dev_fake_macro_recall ≥ 0.40: FAIL (0.353, under by 0.047)

Abort criterion (composite > base 0.331 + 0.05 = 0.381): TRIGGERED (0.502 > 0.381).

Directional pattern. SlotAv2 step 3500 raised the threshold to 0.8265 (vs HEAD's 0.559 and base's implied 0.535). The model became more conservative everywhere: lockbox FPR improved 0.019→0.012 but fake recalls collapsed by 13-99% across all dev + lockbox suites. The GroupDRO worst-group reweighting amplified the boundary trade-off observed at base; the lever pulled toward the under-served real groups (likely chronic identities) and pushed the entire fake distribution away from the calibration band. Viso recall collapsed to 0.5% (3 of 550 fakes).

The decision is taken in `AGENT_PROPOSAL_SLOTAV2_2026-05-23.md`.

Reproducibility:
- Code: `loss/substrate_pair_asymmetric.py` (NOT used here), `data/sample/substrate_paired.py` (stamper disabled at runtime — see Lever note), `trainer/mixins/group_dro.py` (the actual lever)
- Yaml: `experiments/phase2_round13/R13_GROUPDRO_SUBSTRATE_SLOTAV2_2026-05-26.yaml`
- Base ckpt: `gs://training-job-outputs/best_checkpoints/hp35c51p/periodic_effort_20260516_step3500_auc0.9952_eer0.0123.pth`
- Image: `us-docker.pkg.dev/train-cvit2/effort-detector/effort-detector:1.3.299` (missing inventory CSV — see Lever note)
- Launcher: `./scripts/launch/launch_experiment.sh -y effort-r13-phase2 us-west4 experiments/phase2_round13/R13_GROUPDRO_SUBSTRATE_SLOTAV2_2026-05-26.yaml`
- Scoring driver: `analysis/phase_2_backbone_2026-05-22/complete_phase_2_backbone.sh 6ypu1ds3 slotav2 3500`
