# Slot β property shortcut investigation — index

> **Index** of the v3-v7 CPU diagnostic chain that ran 2026-05-16. Auto-mode session
> opened after the RESCHAIN_GRL6 scorecard verdict and produced this analysis +
> two GPU packets (anchor_aware, real_rebalance) launched same evening.

## Reading order

1. **v3 — band shortcuts found** → `RESULTS_FACTS_v3_BANDS_2026-05-16.md`
   9 image-property bands each have 4-10× over-fire rate inside band.
   Compositional dose-response (0 bands → 7% over-fire, 9 bands → 100% on Slot β).
   Decision tree on 11 properties → over-fire AUC 0.89 (Slot β) / 0.99 (P8A).

2. **v4 — OOD-band hypothesis** → `RESULTS_FACTS_v4_OOD_BAND_2026-05-16.md`
   Training reals have 0 frames at n_bands ≥ 6.
   Roy_D (chronic-FP target) averages 7 bands hit.
   Encoder defaults "fake" in unanchored region.

3. **v5 — encoder probe (vanilla)** → `RESULTS_FACTS_v5_ENCODER_PROBE_2026-05-16.md`
   Vanilla OpenCLIP separates Roy_D from clean at AUC 1.0.
   The axis is inherited from pretraining, not introduced by FT.
   Compositional augmentation lever refuted at encoder level.

4. **v6 — auto-mode launch + sanity** → `RESULTS_FACTS_v6_AUTO_MODE_2026-05-16.md`
   Two GPU packets launched (Slot A: anchor_aware; Slot B: real_rebalance).
   Slot A v1 failed at trainer init (ModuleNotFoundError); fixed via
   .gcloudignore whitelist; relaunched as Slot A v2 in us-central1.

5. **v7 — SVD-aware probe of trained encoders** → `RESULTS_FACTS_v7_TRAINED_ENCODER_2026-05-16.md`
   FT amplifies the Roy_D vs clean axis. Vanilla cosine +0.74, Slot β cosine -0.93.
   In T5C encoder: anchor pool 53% Roy_D-adjacent, VCD 27%, dor_shkedi 60%.

## What's running

Two single-lever T5C-base FT packets:

| slot | yaml | mechanism | region | Vertex | W&B | status |
|---|---|---|---|---:|---|---|
| A v2 | R13_T5C_ANCHOR_AWARE_2026-05-16.yaml | training-time penalty on dor false-flag pool | us-central1 | 8471062799328477184 | hp35c51p | RUNNING |
| B | R13_T5C_REAL_REBALANCE_2026-05-16.yaml | family weight rebalance | us-west4 | 3559896493831749632 | iw2kk1h0 | RUNNING |

## Predictions (before scorecard)

- Slot A: moderately likely to reduce dor chronic FP. Roy_D generalization conditional on encoder dynamics.
- Slot B: weaker prediction. VCD's 27% Roy_D-adjacency in T5C is modest.
- Encoder-axis amplification (v7) is the dominant structural force.

## Outputs

```
analysis/slot_b_property_shortcut_2026-05-16/
├── README.md                                         # this file
├── RESULTS_FACTS_v3_BANDS_2026-05-16.md             # band shortcuts
├── RESULTS_FACTS_v4_OOD_BAND_2026-05-16.md          # OOD-band hypothesis
├── RESULTS_FACTS_v5_ENCODER_PROBE_2026-05-16.md     # vanilla openclip probe
├── RESULTS_FACTS_v6_AUTO_MODE_2026-05-16.md         # auto-mode launch + incident
├── RESULTS_FACTS_v7_TRAINED_ENCODER_2026-05-16.md   # SVD-aware probe of trained ckpts
├── RESULTS_FACTS_2026-05-16.md                      # superseded by v3-v7
├── RESULTS_FACTS_v2_2026-05-16.md                   # superseded by v3-v7
├── AGENT_PROPOSAL_2026-05-16.md                     # superseded
├── AGENT_PROPOSAL_v2_2026-05-16.md                  # superseded
├── AGENT_PROPOSAL_v4_2026-05-16.md                  # superseded by v5
├── AGENT_PROPOSAL_v5_FINAL_2026-05-16.md            # superseded by v7 then auto-mode
├── outputs/                                          # CSV data
├── scripts/                                          # diagnostic scripts (band, decomp, sensitivity, encoder probe)
├── cache/  / dev_cache/  / training_cache/  / etc.   # frame caches for analysis
└── ckpt_cache/                                       # trained T5C + Slot β backbones
```

Related upstream docs (RESCHAIN_GRL6 retro that opened this thread):

- `analysis/reschain_grl6_eval_2026-05-16/RESULTS_FACTS_2026-05-16.md`
- `docs/packet_retrospectives/packets/RESCHAIN_GRL6.md`
- `docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md`

Related downstream (post-scorecard):

- `analysis/auto_mode_2026-05-16_eval/PENDING_SCORECARD_PLAN.md`
- `docs/packet_retrospectives/packets/AUTO_MODE_ANCHOR_REBALANCE_PREVIEW.md`
