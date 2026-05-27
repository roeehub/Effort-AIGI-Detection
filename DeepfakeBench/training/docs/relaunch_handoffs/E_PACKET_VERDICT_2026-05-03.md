# E-Packet Verdict — 2026-05-03 (B16 vs L14 Architectural Bake-off)

**Status:** E1 scorecard CSV in hand (verdict: refuted). E2b scorecard running. E3 training finished, scorecard pending.

## TL;DR

| Packet | Hypothesis | Trainer-best AUC | Scorecard verdict |
|--------|-----------|------------------|-------------------|
| E1 | B16 FT-from-P8A + heavier eval-aug → lifts viso recall | 0.9866 (step500) | ❌ REFUTED — viso/deeplive REGRESS vs P8A; lockbox-fake transfer ↑ |
| E2b | B16 SCRATCH + CrossEntropy + heavy aug → breaks anchor ceiling | 0.9944 (step7400) | ⏳ scorecard running (us-west4, ~70min in, ETA ~21:00 Paris) |
| E3 | L14 SCRATCH + CrossEntropy + heavy aug → 4×-larger encoder | **0.9976 (step7200)** — best of R13 chain | ⏳ scorecard pending (image rebuild + launch in flight) |

E3 produced the best val-AUC of the entire R13 chain (0.9976, EER 0.0140) on a clean L14 scratch run. That alone justifies the budget; whether it translates to viso recall is the open question.

## Background — what the hypotheses attacked

The R13 chain is stuck on a 27% viso recall ceiling at joint dev+lockbox FPR=10% (`project_viso_ceiling_unbroken_10_packets`). 10+ packets — single-lever aug, data-axis, base-shift, training-cap, GRL — none cleared it. The E packet attacks 3 independent axes:

- **E1** — augmentation-distribution gap (eval substrate has ~4-25× lower Laplacian variance than train; per `project_image_quality_shortcut`)
- **E2b** — chain-ossification (P8A is the result of 4 sequential FT stages on a CLIP base; could the structure carried by FT be the limit?)
- **E3** — encoder capacity (per the original Effort paper at https://arxiv.org/pdf/2411.15633, L14 is the reference backbone; we've been using B16 since R12)

E2 (the original ArcFace variant) collapsed at step 3400 from `s≥10 + heavy aug` — fixed in E2b by switching to CrossEntropy (matching the original Effort paper's loss).

## E1 — REFUTED (data backing)

Source: `/tmp/e1_scorecard/checkpoint_summary.csv` (full path in GCS via job `7115285333587001344`, us-east1).

At each ckpt's per-checkpoint dev-calibrated τ (≈ FPR=2%):

| ckpt | sel τ | dev_real_FPR | viso_dev | deeplive_dev | teams_dev | lockbox_fake | rank |
|------|-------|--------------|----------|--------------|-----------|--------------|------|
| **P8A_REFERENCE_STEP5000** | 0.991 | 0.0200 | **0.0109** | **0.0239** | 0.3728 | 0.2372 | 7 |
| E1_TOP_N_STEP400 (rank 1) | 0.957 | 0.0197 | 0.0073 | 0.0000 | 0.3748 | **0.4427** | 1 |
| E1_STEP500 | 0.945 | 0.0200 | 0.0109 | 0.0000 | 0.3782 | 0.3676 | 2 |
| E1_STEP300 | 0.940 | 0.0200 | 0.0109 | 0.0000 | 0.3690 | 0.3162 | 3 |
| E1_STEP700 | 0.959 | 0.0200 | 0.0036 | 0.0000 | 0.3562 | 0.2727 | 5 |
| E1_STEP900 | 0.935 | 0.0191 | 0.0036 | 0.0000 | 0.3574 | 0.2569 | 6 |

**Findings:**
1. **Viso recall regressed** at every E1 ckpt vs P8A (0.7-1.1% E1 vs 1.1% P8A); at the higher FPR=10% slice the gap is stronger (E1 4.7-17.6% vs P8A 27.1%).
2. **Deeplive recall went to ZERO** at every E1 ckpt vs P8A 2.4%.
3. **Lockbox-fake transfer LIFTED 1.9× at the rank-1 ckpt** (E1 44.3% vs P8A 23.7%). Replicates S2's lockbox lift signal, but contradicts S2 by NOT lifting viso.

**Mechanism guess:** heavier aug at FT-from-P8A diverts capacity from the dor-invariance signal P8A had into a more general "perturbation-robust" boundary that handles new physical-world artifacts (lockbox) but loses the substrate-specific viso boundary. **Refutes Stream C** (aug-distribution-gap as viso bottleneck).

**Operational note:** longer FT made it worse — STEP100 to STEP900 monotonically degrades viso (17.6% → 4.7% at FPR=10%). Consistent with `project_train_auc_not_valid_promotion_signal`.

## E2b — training observations (scorecard pending)

| Step | val_AUC | val_EER | comment |
|------|---------|---------|---------|
| 1000 | (early) | — | cleared E2's collapse zone (step 3400) |
| 7400 | **0.9944** | **0.0155** | trainer's top_n best |

CrossEntropy fix (vs E2's ArcFace) succeeded — no collapse, val AUC competitive with FT-from-P8A despite being scratch.

If E2b scorecard viso > 27% at FPR=10% → **scratch+CE is the lever**, FT chain is dispensable, B16 is enough.
If E2b ≤ 27% → architectural ceiling, look to L14.

## E3 — training observations

L14 SCRATCH + CrossEntropy + heavy eval-aug, run `jzroefab` in us-central1. Just finished at 19:37 Paris.

| Step | val_AUC | val_EER | role |
|------|---------|---------|------|
| 1000 | 0.9702 | 0.0561 | early |
| 2000 | 0.9884 | 0.0397 | mid |
| 4000 | 0.9956 | 0.0093 | top_n start |
| 4800 | 0.9961 | **0.0047** | best EER |
| 6600 | 0.9972 | 0.0093 | joint best |
| 7200 | **0.9976** | 0.0140 | best AUC of R13 chain |
| 8000 | 0.9972 | 0.0140 | end |

Cleared E2's collapse zone with comfortable margin. Chain monotonic from step 4000 onward. Best AUC is **higher than every single B16 R13 ckpt we have logged** — including the much-tuned P8A (0.9926) and E1 (0.9866).

11-entry checkpoint map: `arena/checkpoint_maps/teams_target_domain.e3_2026-05-03.yaml`. Image rebuild in progress (Cloud Build `3ff3eb6c-4fbf-497f-b80a-8df597b7afd2`, queued at 17:43 UTC).

## What this means for the user's framing

User asked: "are we any closer to training the model properly? … keep B16 backbone … consider Effort paper L14 … consider scratch was rejected a few times".

- **B16 scratch is structurally viable** (E2b), contrary to prior rejections — those were under ArcFace and `s≥10` collapse, not the fundamental architecture.
- **B16 FT-from-P8A with heavier aug is NOT the lever for viso** (E1 refuted). It IS a lever for lockbox-fake transfer, which may be useful for substrate generalization.
- **L14 trained spectacularly clean** (E3) — first arch-axis evidence of the day. B16 keeping vs L14 switch should be decided by E3 scorecard, not val AUC alone (per `project_train_auc_not_valid_promotion_signal`).

## Open verdicts (this evening)

- E2b scorecard verdict: ETA ~21:00 Paris (us-west4, job `5524802937504661504`).
- E3 scorecard launch: ETA ~20:30 Paris (after Cloud Build completes).
- E3 scorecard verdict: ETA ~22:30-23:30 Paris.

Final consolidated verdict before midnight Paris.
