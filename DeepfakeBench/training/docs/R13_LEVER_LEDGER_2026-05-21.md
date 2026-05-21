# R13 Lever Ledger — FACTS only — frozen 2026-05-21

> Scope: decision-relevant R13 packets (from P8A forward). Pre-P8A packets
> (R13_A/B/C/FT*/RLP1-7) are closed and captured in
> `docs/packet_retrospectives/packets/RLP*.md` + auto-memory. Read those
> directly if you need to trace a specific pre-P8A lever.
>
> "Verdict" is literal — what did the contract scorecard / canary / Pareto
> say at the time the packet was retro'd. NOT a recommendation, NOT a
> ship/don't-ship. For deployment guidance see
> `analysis/iq_substrate_tau_2026-05-21/slot_a_v2_lockbox_pareto.md`.
>
> **Read alongside `MEMORY_CAVEATS_2026-05-21.md`** — some older claims have
> been re-examined and a few are wrong as originally framed.

## Glossary of terms used in the table

- **FT base** — the checkpoint used as initialization for fine-tuning. P8A is
  the most-used base; T5C step3500 (W&B `jrlldtem`) is the second base.
- **Single-lever delta** — the one thing changed from the FT base's
  configuration. R13 standard practice: one lever per packet.
- **Verdict at retro time** — the literal classification when the packet was
  reviewed: `ship-candidate` (matched or exceeded current deploy on its target
  metric), `partial` (showed mechanism activity but didn't ship), `refuted`
  (failed close criterion or regressed), `superseded` (later packet replaced
  it), `infra-only` (smoke test or framework piece).
- **W&B run ID** — when known; check `wandb.ai/dtect-vision/phase2-round13`
  or `phase2r13-experiments`.

## Table

| Date | Packet | FT base | Single-lever delta | Step | W&B | Verdict | FACTS doc |
|------|--------|---------|-------------------|------|-----|---------|-----------|
| 2026-04-24 | **P8A** (R13_RLP8_01_unfreeze_clip_codec) | RLP6_04 step23500 | unfreeze visual.proj + ln_post + apply_svd_to_mlp; teams_codec_sim_p=0.40 | 5000 | `9lmvb5b4` | **deploy-anchor** (rank-1 on contract through entire program) | `packets/P8A.md` |
| 2026-04-27 | RLP8_02 (fresh_head_plain_clip) | scratch | fresh head on plain CLIP backbone | — | — | refuted (no upstream-shortcut break) | `packets/RLP*.md` series |
| 2026-04-30 | P14_DATA_FIX | P8A step5000 | family_weight viso +8.0 + bundle | — | `xan4dfto` | refuted (value_composite 0.126) | `packets/P14.md` |
| 2026-04-30 | P14_FACE_SCALE_JITTER (mclioexb run) | P8A step5000 | face_scale_jitter@0.50 isolated | — | `mclioexb` | partial (5.7× value_composite over bundle but no τ promotes on scorecard) | `packets/P14.md` |
| 2026-04-30 | P16_DATA_AXIS | P8A step5000 | family weighting viso fw=2.0, no bundle | — | `rmic6wrc` | refuted (viso recall stays 1.1%) | `packets/P14.md` (combined retro) |
| 2026-05-01 | P17_LAYER3_HEAD / L4_HEAD (ArcFace + LINEAR) | scratch | layer-3 / layer-4 head readout | — | — | refuted (head learns AWAY from invariant signal by step 1000) | `packets/P15.md` |
| 2026-05-01 | P18_METHOD_DOMAIN_GRL | P8A step5000 | 12-class method-conditional GRL | — | — | refuted (P8A holds rank-1, P18T preserves 0% dor FPR, P18C regresses to 52%) | `packets/P15.md` |
| 2026-05-02 | **P22_AUG_CURRICULUM** | P8A step5000 | pipeline_randomization REPLACES teams_codec_sim | 8000 | `b5pmgrxn` | partial (3× dev fake_macro_recall; transport-shortcut cost invisible at retro time — see 2026-05-19 natural-experiment finding) | `packets/PC.md` |
| 2026-05-03 | S1/S2/S3 (P22 variants) | P22 step8000 / P22 step1000 | training-cap / earlier-base / viso fw=8.0 | varies | — | partial (S2 step600 = chain champion 91.5% teams_fake_lockbox; viso ceiling unbroken at 27%) | `packets/PC.md` |
| 2026-05-04 | E2B (R13_E2b_SCRATCH_B16_NO_ARCFACE) | scratch | B16 scratch + CE (no arcface) | 3200 | — | partial (deeplive 87.5% recall at FPR=10% but viso REGRESSES to 7%) | memory `project_e2b_breaks_deeplive_ceiling` |
| 2026-05-04 | E3 (R13_E3_L14_SCRATCH_EVAL_AUG) | scratch | L14 scratch + CE | 6600 | — | refuted (viso 11.6% at FPR=10%) | memory `project_l14_does_not_break_viso_ceiling` |
| 2026-05-09 | **T3_SLOT1** (DROP_HIGH_IQ_TEAMS_REALS) | P22 step8000 | drop top-25% high-IQ teams reals from training pool | 1500/2500 | — | partial (step1500: 77.5% teams_fake_lockbox recall, 2× P8A; doesn't formally promote due to stress FPR 9.92) | `packets/T3.md` |
| 2026-05-10 | T4_MULTI_AXIS_GRL | T3_SLOT1 step1500 | multi-axis L11 GRL on 4 axes (chronic_flag / is_dor / sharpness / color_a) | 10500 | — | partial (P8A wins contract; T4 dev AUCs +0.037-0.082 BUT lockbox AUC −0.174) | `packets/RESCHAIN_GRL6.md` |
| 2026-05-11 | T5A_CYCLIC_LAMBDA | T4 step10500 | cyclic λ schedule on GRL | — | — | superseded (T5C with bigger classifier wins T5 chain) | `packets/T6_T7_T5C.md` |
| 2026-05-11 | **T5C** (R13_T5C_T4_BIG_CLASSIFIER) | T4 step10500 | classifier hidden_dim 256→1024 | 3500 | `jrlldtem` | partial (rank-3, catches 1.7× more lockbox fakes than P8A at +0.0095 absolute FPR — used as FT-base going forward) | `packets/T6_T7_T5C.md` |
| 2026-05-11 | T6_T3_PLUS_JITTER / T7_T4_PLUS_JITTER | T3/T4 | face_scale_jitter stacked | — | — | refuted (6/6 ckpts fail dev_macro 0.30 floor — face_scale_jitter does NOT compose on T3/T4 base) | `packets/T6_T7_T5C.md`, memory `project_face_scale_jitter_load_bearing` |
| 2026-05-13 | LORA_L10_L11 (infra landed) | — | LoRA adapter on resblocks 10-11 | — | — | infra-only (10 unit tests + smoke wiring; D1-D5 + user auth pending) | memory `project_lora_l10_l11_infra_landed_2026-05-13` |
| 2026-05-13 | SLOT1_HEAD_RETRAIN / SLOT3_T5C_PLUS_JITTER030 / SLOT4_B16_SCRATCH_FOURIER | varies | head retrain / jitter@0.30 / fourier on B16 scratch | — | — | refuted: head retrain over-fires lockbox 80-92% on real frames | memory `project_job7_head_retrain_REFUTED_2026-05-04` |
| 2026-05-14 | LORA_T5C_L10_L11_R8 + LORA_T5C_L10_L11_PROPER_DATA | T5C step3500 | LoRA r=8 / r=16 on L10-L11 with various data | — | — | refuted (all 6 LoRA-T5C ckpts hard-couple to 24% lockbox_real_fpr vs P8A 1.84%) | memory `project_lora_l10_l11_chronic_fp_hard_coupled_2026-05-15` |
| 2026-05-14 | T5D_T5C_PLUS_FOURIER | T5C step3500 | fourier band aug on T5C base | — | — | refuted | memory cross-reference |
| 2026-05-15 | **T5C_RESCHAIN** (Slot α) | T5C step3500 | resolution_chain_aug | 3500 | `lsx4n0t7` | partial (cuts real score_range 25%, flip rate τ=0.9 from 57.7%→12.4%; needs scorecard for contract metrics) | memory `project_overnight_resolution_chain_2026-05-16` |
| 2026-05-15 | **T5C_6AXIS_GRL** (Slot β) | T5C step3500 | 6-axis multi_axis_grl (4 base + luma_high + ?) | 3500 | `gwntcld0` | refuted (real_rebal arm at 0.090 lockbox_real_fpr = 4.8× P8A; per encoder prediction) | memory `project_overnight_resolution_chain_2026-05-16` |
| 2026-05-16 | **T5C_ANCHOR_AWARE (Slot A v2)** | T5C step3500 | anchor_aware penalty (weight=5.0, target=0.10, pool=dor-real-webcam-false-flag-no-virtual-bg) | 3500 | `hp35c51p` | **ship-candidate** (29-suite scorecard: lockbox_fake_recall 0.688 vs P8A 0.387; lockbox_real_fpr 0.0191 vs P8A 0.0184 — rank-2 only by 0.07pp tiebreak. Dominates P8A at every lockbox-calibrated Pareto operating point per 2026-05-21 partial Pareto) | `packets/AUTO_MODE_ANCHOR_REBALANCE_PREVIEW.md` + `slot_a_v2_lockbox_pareto.md` |
| 2026-05-16 | T5C_REAL_REBALANCE (Slot B) | T5C step3500 | real-side rebalance | 3500 | — | refuted (lockbox FPR 0.090, 4.8× P8A) | memory `project_band_shortcut_ood_hypothesis_2026-05-16` |
| 2026-05-19 | T5C_TRIPLE Slot 1 (6AXIS_PLUS_ANCHOR) | T5C step3500 | 6-axis multi_axis_grl + anchor_aware stacked | 3500 | `afydiase` | refuted (lockbox_R FPR=5% trajectory monotonically DOWN: 0.53→0.39→0.30 over steps 1500/2500/3500; 3.5× worse than Slot A v2 step3500) | `packets/T5C_TRIPLE_2026-05-19.md` |
| 2026-05-19 | T5C_TRIPLE Slot 2 (LORA_L8_L9_R8) | T5C step3500 | LoRA r=8 on resblocks 8-9 | 2500 | `3yo9d1f3` | refuted (W&B canary at probe_step 3000: lockbox_R FPR=5% = 0.25 — middling, doesn't exceed Slot A v2's 0.59) | `packets/T5C_TRIPLE_2026-05-19.md` |
| 2026-05-19 | T5C_TRIPLE Slot 3 (5AXIS_NOLUMA) | T5C step3500 | 5-axis GRL (drop luma_mean_high) | 3500 | `9p1zo42l` | refuted (≈ Slot 1 on lockbox metrics) | `packets/T5C_TRIPLE_2026-05-19.md` |
| 2026-05-21 | T5C_ANCHOR_AWARE_PLUS_CODEC (Slot 1) | Slot A v2 step3500 | teams_codec_sim_p=0.40 ON TOP OF pipeline_randomization | 3500 | `ohcx4w0x` | refuted (natural-exp Δ closes 51% but absolute Δ=-0.083 > 0.05 criterion; may6 frac>0.5 regresses 0.043→0.065) | `analysis/codec_restoration_gates_2026-05-21/VERDICT_FACTS_2026-05-21.md` |
| 2026-05-21 | T5C_ANCHOR_AWARE_PLUS_CODEC_LIGHT (Slot 2) | Slot A v2 step3500 | teams_codec_sim_p=0.20 | 3500 | `xg35bae4` | refuted (Δ=-0.067, best closure 60%, but may6 0.087 vs Slot A v2 0.043; non-monotone w.r.t. Slot 1) | same VERDICT_FACTS |
| 2026-05-21 | T5C_CODEC_ONLY_NO_ANCHOR (Slot 3) | T5C step3500 | codec p=0.40, anchor=OFF | 3500 | `21alo5iu` | refuted (Δ=-0.106 worst of 3; may6 0.120 FAILS 0.10 criterion — anchor mechanism shown binding for production-drift) | same VERDICT_FACTS |
| 2026-05-21 | IQ-substrate τ feasibility (CPU only) | — | per-frame IQ k-means cluster as substrate proxy | — | — | refuted (at apples-to-apples lockbox FPR: per-mode oracle on IQ-substrate is −13 to −48pp vs naive global τ; ALSO surfaced that the May 5 "24.5pp lift" claim was at different operating points and the lift evaporates at matched FPR) | `analysis/iq_substrate_tau_2026-05-21/FACTS_2026-05-21.md` |

## Verdict counts

- **ship-candidate**: 1 (P8A; deploy-anchor)
- **ship-candidate / unresolved**: 1 (Slot A v2 — dominates P8A on partial Pareto but ranks rank-2 on contract by 0.07pp tiebreak that is inside sampling noise per Job B 2026-05-20)
- **partial / mechanism activity, didn't ship**: 8
- **refuted / failed close criterion**: 16
- **superseded**: 1
- **infra-only**: 1

## What this ledger does NOT capture

- **Pre-T3 in detail**: the ~6 weeks of pre-T3 work (P9-P21, RLP1-7, FT1-17) are visible in the packet retros and memory but not in this table. They collectively established that:
  - P8A's unfreeze-late-encoder-blocks recipe is the deploy anchor (no later packet has dominated it on the 29-suite contract on its terms);
  - the camera/transport-signature shortcut is upstream of any FT lever tested;
  - data-axis levers (P14/P16/PA) and head-only-retrain levers (P17, SLOT1_HEAD_RETRAIN) are dead.
- **The contract policy itself**: the 29-suite manifest + lex tiebreak rule is in `arena/score_teams_promotion_contract.py`; not relitigated here.
- **CPU diagnostic battery results** (lockbox tagging, IQ-shortcut audit, may6 production-drift): captured in `analysis/cpu_diagnostics_*` directories and per-packet FACTS docs.
- **Recommendations**: see `READ_FIRST.md` for the program's current state.
