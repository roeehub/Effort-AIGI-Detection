# E3 (L14 SCRATCH) Final Verdict — 2026-05-04

**Status:** Trim4 contract scorecard complete. Job `1895705638541983744` SUCCEEDED 03:34 Paris.
11-ckpt scorecard cancelled mid-flight at user OK (saved ~$30, freed us-east1 GPU). Trim4 covered the load-bearing question with 4 ckpts: P8A baseline + 3 best L14 ckpts.

## TL;DR

**L14 does NOT break the viso ceiling.** Best L14 ckpt gives viso 11.6% at FPR=10%, vs P8A's 27.1%. Confirms viso ceiling is structural — not encoder-capacity-bound.

**L14 ties B16-scratch on deeplive lift.** Both architectures independently lift deeplive recall to ~85-88% at FPR=10% (vs P8A's 43%). At 4× inference cost, L14 adds no operational value over B16.

## Multi-FPR-floor table (L14 trim4 contract scorecard)

```
ckpt                     FPR=2%        FPR=5%         FPR=10%
                         viso/dl/tf    viso/dl/tf     viso/dl/tf

P8A_REFERENCE_5000       1.1/2.4/37    7.8/15.0/47    27.1/42.9/64
E3_TOP_N_STEP4800        0.5/32.3/43   2.4/58.5/54    6.0/71.4/61
E3_TOP_N_STEP6600        3.8/2.4/4.8   6.7/71.6/65    11.6/85.5/73
E3_TOP_N_STEP7200        3.5/0.9/3.3   4.7/60.6/62    8.7/81.1/71
```

(viso = visomaster_enhanced_macro_dev, dl = deeplive_enhanced_dev, tf = teams_fake_all_dev)

## Training stats (L14 SCRATCH from CLIP)

L14 trained the cleanest val_AUC chain of any R13 packet:

| Step | val_AUC | val_EER | Notes |
|------|---------|---------|-------|
| 1000 | 0.9702 | 0.0561 | early |
| 4000 | 0.9956 | 0.0093 | top_n start |
| 4800 | 0.9961 | 0.0047 | best EER |
| 6600 | 0.9972 | 0.0093 | joint best |
| **7200** | **0.9976** | 0.0140 | **best AUC of entire R13 chain** |
| 8000 | 0.9972 | 0.0140 | end |

Per `project_train_auc_not_valid_promotion_signal`: train AUC of 0.9976 did NOT translate to operational performance — STEP7200 ranks below STEP6600 on contract.

## Comparison vs B16 scratch (E2B_3200 in E2b run)

```
                     viso @ FPR=10%   deeplive @ FPR=10%   teams_fake @ FPR=10%
B16 scratch          7.1%             87.5%                72.0%
(E2B_TOP_N_STEP3200)
L14 scratch          11.6%            85.5%                72.7%
(E3_TOP_N_STEP6600)
P8A FT chain         27.1%            42.9%                64.1%
```

L14 marginally beats B16-scratch on viso (11.6 vs 7.1 — both well below P8A 27.1%) and is essentially identical on deeplive (85.5 vs 87.5) and teams_fake (72.7 vs 72.0).

**L14 is not worth deploying.** It costs ~4× more inference (304M params vs 86M) for marginal viso lift over B16-scratch and zero meaningful improvement on the other suites.

## What this refutes

- "Encoder capacity (B16's 86M params) is the viso bottleneck" — REFUTED. 4× larger encoder doesn't fix it.
- "Original Effort paper's L14 backbone is required for production performance" — REFUTED in our context. We can match Effort's published numbers on standard benchmarks but viso/Teams target domain has a structural bottleneck that L14 doesn't address.
- "More training capacity = closer to the production goal" — REFUTED. Adding params adds cost, not capability, on the binding constraint.

## What this supports

- **Viso ceiling is structural** — combined with E1 (B16-FT) and E2b (B16-scratch) refutations, 3 architecturally-distinct packets all fail viso → bottleneck is upstream of training.
- **B16 is sufficient backbone** — can match L14 operationally at ¼ cost. User's preference to "keep current B16 backbone" is correct.
- Per `project_p8a_breakthrough` and `project_p8a_frame_level_auc_2026-04-29`: P8A's viso strength is uniquely tied to its specific FT trajectory (R12g→RLP6_04→RLP7_02→P8A), not its architecture.

## Cost

- L14 training (8000 steps): ~$25 us-central1 A100 × 8h
- L14 11-ckpt scorecard (cancelled at ~5h in): ~$10
- L14 trim4 scorecard (4 ckpts × 7.5h us-central1 A100): ~$15
- **L14 packet total**: ~$50

## Files

- `/tmp/e3_trim4_scorecard/` — full contract artifacts
- `arena/checkpoint_maps/teams_target_domain.e3_2026-05-03.yaml` — 11-ckpt map (kept for trajectory reference)
- `experiments/phase2_round13/R13_E3_L14_SCRATCH_EVAL_AUG.yaml` — training config
- `analysis/e2b_ensemble_2026-05-04/run_3way_ensemble.py` — ensemble script
- W&B run: `dtect-vision/phase2r13-experiments/jzroefab`

## Memory

- `project_l14_does_not_break_viso_ceiling.md` — created
