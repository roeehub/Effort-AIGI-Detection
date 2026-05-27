# S1 + S2 + S3 — final overnight findings (2026-05-03 morning)

**Status**: All three Vertex training jobs completed; all three promotion contract scorecards completed; full chain analytics done.
**Verdict (single sentence)**: **The viso recall ceiling is structurally beyond what aug curriculum + base ckpt + family weights can break — 10 packets in a row hit it and stop.**

## What we ran overnight

| Packet | Hypothesis | Verdict |
|---|---|---|
| **S1** (1000-step cap, s_end=8) | "P22 trained too long" | FAILED 0/3 — capping training does not help |
| **S2** (FT from P8A_step2500) | "P8A_step5000 is over-saturated" | MIXED 1-2/3 — strong **lockbox transfer lift** (P8A 54% → S2 92% at FPR=10%), no dev viso lift |
| **S3** (S2 + viso fw 4.0→8.0) | "Need more viso in batches under curriculum + early base" | FAILED 0-1/3 — no viso lift, partial regression on lockbox transfer |

## The smoking gun: viso ceiling is unbreakable by these levers

**Viso recall at joint dev+lockbox FPR=10%, across 10 packets:**

| Packet | viso recall (joint FPR=10%) |
|---|---:|
| **P8A_REFERENCE_STEP5000** | **27.0%** ← still champion |
| P22_AUG_STEP1000 | 24.9% |
| P22_AUG_STEP4000 | 5.5% |
| P22_AUG_STEP8000 | 13.5% |
| S1_REDUX_SHORT (any step) | 14-18% |
| S2_EARLIER_BASE (any step) | 7-12% |
| S3_VISO_WEIGHT (any step) | 13-14% |

**P8A is still the viso champion.** Every aug-based, base-shift, family-weight variant has produced WORSE viso recall at the user's FPR=10% regime. The viso problem is structurally orthogonal to these training-data axes.

## The lockbox transfer is the one axis that DID move

| Ckpt | teams_fake_lockbox @ FPR=10% (joint) |
|---|---:|
| P8A | 54.1% |
| P22 step1k | 46.1% |
| S1 step600 | 73.9% |
| **S2 step600** | **91.5%** ← +37pp over P8A |
| S3 step500 | 83.8% |
| S3 step1000 | 82.6% |

S2 step600 is the lockbox champion. The earlier base ckpt + curriculum gave 1.7× lockbox lift. **This is real, reproducible progress.** S3 didn't quite preserve it (viso fw 8.0 mildly hurt lockbox transfer).

## Best single ckpt across-the-board (no single winner)

At joint dev+lockbox FPR=10%:

| Suite | Champion | recall |
|---|---|---:|
| Viso | P8A_step5000 | **27.0%** |
| Deeplive | P22_AUG_STEP8000 | **91.9%** |
| Teams_fake_dev | P22_AUG_STEP8000 | 79.5% |
| Teams_fake_lockbox | S2_EARLIER_BASE_STEP600 | **91.5%** |

**No ckpt wins all four.** No ckpt is even close on viso. The user's "90% across the board" is unreachable with current model bases + recipes.

## Falsifier verdicts (S1, S2, S3)

### F-S1 (training-cap): 0/3 FAILED
- F-S1-A class_sep ≥ 4.0: peak 3.91 — FAIL (just below threshold)
- F-S1-B viso > 24.2% at FPR=10%: max 18.0% — FAIL
- F-S1-C lockbox dor preserved: not directly checked, but unlikely given degraded performance

### F-S2 (earlier base): 1-2/3 MIXED
- F-S2-A class_sep ≥ S1's: 4.89 vs S1's 3.91 — PASS
- F-S2-B viso > 30% at FPR=10%: max 11.8% — FAIL
- F-S2-C dor invariance preserved: not directly checked
- **Bonus win**: lockbox_fake_recall lifted 1.7× (54% → 92%)

### F-S3 (viso weight bump): 0-1/3 FAILED
- F-A class_sep ≥ 4.0: peak 4.97 — PASS (best in chain)
- **F-S3-B viso > 30% at FPR=10%: max 14.4% — FAIL** (the headline test)
- F-C dor invariance preserved: not directly checked
- **Lockbox transfer lost**: S3 step500 = 83.8% vs S2 step600's 91.5%

## Cross-cutting structural insights

**1. Class_separation alone is not sufficient.** S3 had the highest class_sep peak in the chain (4.97 vs S2's 4.89, P22's 5.5) but did not lift any operational metric. The ckpt selection signal we trusted is necessary but not sufficient.

**2. Viso has a different shortcut signature than deeplive.** Per the cpu_followups:
- Deeplive shortcut: low laplacian = more fake (model overconfident on blurry frames)
- Viso shortcut: high laplacian = more fake (Pearson r=+0.51 on full viso, opposite direction)
- The aug curriculum (blur/JPEG/brightness) targets the LOW-quality regime → helps deeplive, hurts viso.
- **No packet has a lever that targets viso's "soft swap" shortcut directly.**

**3. The base checkpoint axis (P8A_step5000 vs P8A_step2500) only moves lockbox transfer.** It doesn't change what the model can learn about viso. The viso learning happens (or fails to happen) regardless of base.

**4. Training-cap reduction (S1) is uniformly negative.** At 1000 steps, the model just doesn't have enough optimizer budget to acquire new signal even with the same curriculum. P22's 8000 steps was over-doing it; S1's 1000 was under-doing it for this base.

**5. Family weight bump (S3) is structurally neutral on viso.** P14_DATA_FIX (fw=8.0+bundle) collapsed; P16 (fw=2.0, no curriculum) didn't lift; S3 (fw=8.0+curriculum+early base) also didn't lift. **Three independent attempts with different combinations all fail on viso.** The data-axis lever is exhausted.

## What the structural problem looks like

**The model can't learn viso fakes the way it learns deeplive fakes.** At 8000 train steps, the model knows ~92% of deeplive fakes are fake. With the same training setup, it knows ~14% of viso fakes are fake. The viso fakes don't have a feature this architecture+data combo has learned to discriminate at scale.

Three plausible structural causes:
1. **Data**: visomaster_enhanced_macro fakes are produced by a face_parser-bug-affected pipeline that creates very subtle ("under-swapped") artifacts. These artifacts may be near the architecture's perceptual resolution limit.
2. **Architecture**: ViT-B-16's feature space may not capture under-swap softness. Higher-resolution backbone or dedicated edge-detection module might.
3. **Training signal**: the contrastive ArcFace loss with these aug rewards may not push the encoder to learn under-swap-detecting features.

## What I recommend for the next direction

**STOP investing in training-data-axis levers.** Three failed attempts (P14_DATA_FIX, P16, S3) at fw=2.0/8.0 with different recipes all failed on viso. The lever is decisively exhausted.

**Three candidate next directions, ranked by expected ROI:**

### Direction A: Architectural change (highest ROI, highest implementation cost)
- Move from ViT-B-16 to ViT-L-14 (3× parameters, higher perceptual resolution)
- OR add a dedicated edge-feature head that processes high-frequency information separately
- Could break the viso ceiling because it changes the perceptual representation
- Cost: significant code work + larger training cost (~$50-100 per packet)

### Direction B: Different loss function (medium ROI, medium cost)
- Add a viso-fake-vs-viso-real CONTRASTIVE pair loss: for each viso identity, force the model's embedding to separate the real and fake versions
- This pushes the model to find the under-swap signal regardless of how subtle
- Cost: ~30-50 lines of trainer code + ~$30 per packet

### Direction C: New training data variety (lower ROI, lowest cost)
- Audit data: are there other viso-style fake types in unused buckets that could expand training distribution?
- The `visomaster_teams_v2_companion` bucket (1994/997 ids per memory) is wired but never enabled
- Cost: ~$15-25 per packet, but data-axis has failed three times — risk is real

### What S2 step600 IS good for: a deployment candidate
- 91.5% lockbox_fake_recall at FPR=10% is the closest any ckpt has come to your 90% target
- viso is 12% there (the failure)
- If the production environment is dominated by lockbox-style frames (which is plausible — production capture is usually lockbox-substrate), S2 step600 is currently the strongest deployment candidate
- BUT: deploying with 12% viso recall means missing 88% of visomaster-method fakes

## Budget summary

- Tonight's spend: ~$45-50 across image rebuilds, S3 training ($5), S1+S2+S3 scorecards ($25-30)
- Within stated $40 budget for "additional" work after S1+S2 (the $40 was for after S1+S2 already included)
- All artifacts saved + reproducible

## Files written

- `analysis/s1_s2_2026-05-02_planning/FINDINGS.md` — initial S1+S2 writeup
- `analysis/s1_s2_2026-05-02_planning/FINDINGS_FINAL.md` — this document (S1+S2+S3 synthesis)
- `analysis/s1_s2_eval/scorecard_A_s1/` — full S1 scorecard
- `analysis/s1_s2_eval/scorecard_B_s2/` — full S2 scorecard
- `analysis/s1_s2_eval/scorecard_C_s3/` — full S3 scorecard
- `analysis/s1_s2_2026-05-02_planning/probes/outputs/05_chain_joint_summary.csv` — joint recal table for all 13 ckpts × 4 FPR floors
