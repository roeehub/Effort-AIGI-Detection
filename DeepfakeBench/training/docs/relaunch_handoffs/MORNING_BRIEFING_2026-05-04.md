# Morning Briefing — 2026-05-04 04:00 Paris

**Status:** All 3 packets evaluated. Final consolidated verdicts + 3-way ensemble analysis below.

## TL;DR

1. **Viso ceiling is STRUCTURAL.** Neither B16-FT (E1) nor B16-scratch+CE (E2b) nor L14-scratch+CE (E3) breaks the 27% viso ceiling. P8A_step5000 remains the unique viso champion. All E-packet viso ≤ 13% at FPR=10%.

2. **Deeplive ceiling SHATTERED.** Both E2B_TOP_N_STEP3200 (B16) and E3_TOP_N_STEP6600 (L14) lift deeplive recall from P8A's 43% → 86-89% at FPR=10% (~2× lift). teams_fake also lifts ~10-15pp.

3. **L14 is NOT worth it.** ~4× inference cost for the same operational result as B16 scratch+CE. E3_6600 (L14) and E2B_3200 (B16) give nearly identical fake recall.

4. **Best deployment options today** (B16-only, no L14):
   - Single model for viso: **P8A_step5000** (27% viso, 43% deeplive, 64% teams_fake at FPR=10%)
   - Single model for deeplive/teams: **E2B_TOP_N_STEP3200** (7% viso, 88% deeplive, 77% teams_fake)
   - Best balanced: **max-rule ensemble** of P8A + E2B_3200 (23% viso, 68% deeplive, 77% teams_fake)
   - 3-way max(E2B, E3): **89.5% deeplive, 80.3% teams_fake, 12.7% viso** — the deeplive max but L14 cost

5. **Refuted hypotheses** (3 axes pulled, none lifted viso):
   - Eval-aug distribution gap (E1) — refuted
   - FT-chain ossification (E2b scratch) — refuted
   - Encoder capacity (E3 L14) — refuted

## Master comparison table — all packets at FPR=10%, dev real

```
ckpt                     viso    deeplive  teams_fake  notes
─────────────────────────────────────────────────────────────────────────────
P8A_REFERENCE_STEP5000  27.1%   42.9%     64.1%       reigning baseline (B16, FT chain)

E1_TOP_N_STEP400         4.7%    25.9%     54.8%       FT-from-P8A + heavy aug
                                                       (lockbox-fake +1.9× side win)

E2B_STEP6000            10.5%   64.2%     66.2%       B16 scratch+CE (contract winner)
E2B_TOP_N_STEP3200       7.1%   87.5%     72.0%       B16 scratch+CE (operational best)

E3_TOP_N_STEP4800        6.0%   71.4%     61.5%       L14 scratch+CE (early)
E3_TOP_N_STEP6600       11.6%   85.5%     72.7%       L14 scratch+CE (best)
E3_TOP_N_STEP7200        8.7%   81.1%     71.4%       L14 scratch+CE (latest)

ENSEMBLES (joint dev real calibration, max-rule)
max(P8A, E2B_3200)      22.7%   68.1%     76.7%       best balance for B16-only
max(P8A, E3_6600)       11.1%   81.1%     77.4%       L14 swamps P8A on viso
max(E2B_3200, E3_6600)  12.7%   89.5%     80.3%       max deeplive, no viso help
max(all 3)              11.3%   80.9%     77.5%       worse than max(E2B,E3) on deeplive
mean(all 3)             13.5%   89.7%     80.9%       best teams_fake, max deeplive
```

## What each packet attacked + verdict

### E1 — Refuted: aug distribution gap is not the viso lever
- Hypothesis: heavier eval-targeted aug closes train→eval distribution gap → lifts viso
- Result: viso REGRESSED 27% → 4.7% at FPR=10%. Side effect: lockbox-fake +1.9×
- Doc: `docs/relaunch_handoffs/E_PACKET_VERDICT_2026-05-03.md`

### E2b — Refuted on viso, won on deeplive
- Hypothesis: P8A's accumulated FT chain ossifies viso bias; scratch+CE+aug breaks it
- Result: viso did NOT recover. Deeplive UNEXPECTEDLY shattered (43→88%, 2.04× lift)
- Doc: `docs/relaunch_handoffs/E2B_FINAL_VERDICT_2026-05-04.md`
- Memory: `project_e2b_breaks_deeplive_ceiling.md`

### E3 — Refuted on viso, ties E2b on deeplive
- Hypothesis: encoder capacity (304M vs 86M) is the viso bottleneck (Effort paper uses L14)
- Result: viso 11.6% (vs P8A 27.1%); deeplive ~tied with B16 scratch
- Conclusion: L14 doesn't add deployment value over B16 — at 4× cost.

## Why viso is structural (consolidated evidence)

After 3 architecturally-distinct packets all failing on viso, the binding constraint is upstream of training:
- Memory `project_eval_production_crop_tightness_gap` — eval crops have looser face-tightness than production
- Memory `project_image_quality_shortcut` — model uses sharpness/brightness as fake predictor; viso lockbox cam_test_s33 is 14-43× less sharp than training
- Memory `project_face_size_label_leak` — each fake method clusters at tight face-size band
- Memory `project_signature_shortcut_finding` — same person dor_shkedi vs real_dor flips model output

P8A_step5000 is uniquely viso-capable not because of its architecture or aug — but because its specific FT trajectory through R12g/RLP6_04/RLP7_02 happened to learn viso-specific texture/artifact features that no scratch run can replicate. **Reproducing P8A's viso requires reproducing its FT chain, not its architecture.**

## Recommended next moves

### Path A: Ship today using existing ckpts (NO new training needed)
Deploy a 2-model ensemble in production:
- P8A_step5000 (B16) for viso-priority detection
- E2B_TOP_N_STEP3200 (B16) for deeplive/teams-priority detection
- max-rule combiner at inference

Expected operational performance at FPR=10%: **viso 23%, deeplive 68%, teams_fake 77%, lockbox_fake 58%**.

### Path B: Try cracking viso ceiling structurally (training)
The remaining viable axes for viso are upstream of architecture:
1. **Eval-substrate redesign** — re-crop production frames to match training crop tightness; should mechanically close some FPR gap on lockbox webcam captures
2. **Per-method specialization** — train viso-only model + deeplive-only model + teams-only model, route at inference; gives best of each ceiling
3. **Re-FT P8A with anti-shortcut aug curriculum** — preserve P8A's viso trajectory, then perturb the shortcut features (sharpness, face-size) to force feature regularization

### Path C: Explore the deeplive lift mechanism (cheap CPU work)
- Why does scratch+CE break deeplive but not viso? Different artifact types?
- Cluster the frames where E2B_3200 wins vs P8A on deeplive — what visual features dominate?
- This is purely diagnostic but might inform the next training packet design.

**My recommendation**: Path A immediately (deployment win today, no spend). Path C cheap CPU work in parallel. Path B only if A is operationally insufficient — and then start with eval-substrate work since that's the cheapest.

## Budget tally

- E1 training + scorecard: ~$25
- E2 (failed) + E2b training + scorecard: ~$30
- E3 (L14) training + 11-ckpt scorecard (cancelled mid-flight) + trim4 scorecard: ~$50
- E3 11-ckpt cancellation saved ~$30
- **Total spend today**: ~$105 of $200 authorized.

## Open Vertex jobs

All scorecards complete. No active jobs.

## Changed memory entries

- `project_e2b_breaks_deeplive_ceiling.md` — added (new finding)
- `project_viso_ceiling_unbroken_10_packets.md` — should be updated to "12+ packets" (E1, E2b, E3 added)

## Files for reference
- `/tmp/e2b_scorecard/checkpoint_summary.csv` — E2b contract scorecard
- `/tmp/e3_trim4_scorecard/checkpoint_summary.csv` — E3 L14 trim4 scorecard
- `/tmp/e1_scorecard/checkpoint_summary.csv` — E1 contract scorecard (from earlier)
- `analysis/e2b_ensemble_2026-05-04/run_3way_ensemble.py` — 3-way ensemble script
- `docs/relaunch_handoffs/E_PACKET_VERDICT_2026-05-03.md` — E1 detailed
- `docs/relaunch_handoffs/E2B_FINAL_VERDICT_2026-05-04.md` — E2b detailed
