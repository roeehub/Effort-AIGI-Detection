# RESULTS_FACTS_v7 — SVD-aware probe of trained T5C / Slot β encoders

> **FACTS only.** Updates v5's vanilla-openclip framing with actual trained-
> encoder embeddings. The trained encoder amplifies the Roy_D vs clean axis,
> which changes the expected effectiveness of Slot A and Slot B's interventions.

## §1. Provenance

- Built SVD-aware encoder loader (`scripts/trained_encoder_probe.py`) that:
  1. Builds vanilla openclip ViT-B-16 (datacomp_xl_s13b_b90k)
  2. Applies `apply_svd_residual_to_openclip_attn` with rank=736, apply_to_mlp=True
  3. Loads `backbone.visual.*` state_dict keys from trained ckpts (596 keys, 0 missing/unexpected)
- Encoded 194 dev PNG frames + 100 VCD + 30 anchor pool + 100 training real frames through:
  - Vanilla openclip
  - T5C step3500 (jrlldtem)
  - Slot β step3500 (gwntcld0)

## §2. Roy_D vs clean separation across encoders

| encoder | Roy_D-vs-clean AUC | PC1 separation | within-Roy_D cosine | across cosine (Roy_D ↔ clean) |
|---|---:|---:|---:|---:|
| vanilla openclip | 1.0000 | **+10.72** | 0.957 | **+0.735** |
| T5C step3500 | 0.9977 | +3.88 | 0.988 | **−0.155** |
| Slot β step3500 | 0.9846 | +3.11 | 0.949 | **−0.927** |

**FT does NOT remove the axis** (AUC stays at 0.98-1.00). What FT does is **rotate Roy_D and clean into opposite directions in feature space**. Vanilla had them in the same general direction (cosine 0.74). Slot β has them nearly antipodal (cosine -0.93).

This is "axis amplification": the trained encoder learned to maximize the distance between Roy_D-style and clean-style frames.

## §3. Where each lever's training-time content lives in the trained-encoder space

Measured cohort positions on the Roy_D ↔ clean axis of the SLOT β trained encoder:

| cohort | n | cos to Roy_D | cos to clean | bias (Roy_D - clean) | % closer to Roy_D |
|---|---:|---:|---:|---:|---:|
| Roy_D (dev, target ref) | 130 | +0.949 | −0.927 | **+1.876** | 97.7% |
| ilan+orel (dev, clean ref) | 64 | −0.913 | +0.935 | **−1.847** | 1.6% |
| training_real_teams | 100 | −0.898 | +0.921 | −1.818 | 4.0% |
| **anchor pool falseflag** (Slot A target) | 30 | **−0.577** | **+0.640** | **−1.216** | **13.3%** |
| **VCD external_real** (Slot B boost target) | 100 | **−0.635** | **+0.671** | **−1.306** | **18.0%** |
| **dor_shkedi (lockbox PNG)** | 895 | **+0.364** | **−0.290** | **+0.654** | **67.8%** |
| **real_dor (lockbox PNG)** | 109 | **−0.927** | **+0.958** | **−1.884** | **0.0%** |

**Critical observation**: dor_shkedi.png (chronic-FP cohort) and real_dor (same person, different tag) are at OPPOSITE ends of the encoder's Roy_D↔clean axis in the Slot β trained encoder — dor_shkedi 67.8% Roy_D-adjacent, real_dor 0%. The encoder has actively separated same-person variants based on capture-style, not identity.

This is the clearest evidence yet that the encoder is keying on capture-style/processing signature, not the person's identity. Slot A's anchor pool and Slot B's VCD reals both sit on the **clean side** in this trained encoder — the boost provides "more real" signal in the clean cluster, not in Roy_D's region.

## §4. Comparison with vanilla openclip predictions (RESULTS_FACTS_v6)

| cohort | % closer to Roy_D in vanilla | % closer to Roy_D in Slot β trained |
|---|---:|---:|
| VCD | **67%** | **18%** |
| anchor pool | 16.7% | 13.3% |
| training_real_teams | 24% | 4.0% |

The vanilla openclip read OVER-PREDICTED Slot B's anchoring potential. After FT, VCD reals are pushed firmly to the clean side. Slot B's mechanism is **partially refuted** at the encoder level — only 18% of VCD samples actually inhabit the Roy_D-adjacent region, not 67%.

## §5. Refined verdict prediction

### Slot A (anchor_aware)

Anchor pool sits at 13.3% Roy_D-adjacency in the trained encoder. Slot A's training signal goes to a clean-side cluster anyway. Predicted: zero generalization to Roy_D. Slot A's mechanism is bounded to its own pool's contribution to "real" supervision (which is already largely covered by existing real-pool sampling).

### Slot B (real_rebalance)

VCD reals are 18% Roy_D-adjacent. 3× boost on VCD gives effective 54% Roy_D-adjacent coverage by mass — but the encoder has already learned to REPEL VCD from Roy_D. Whether further FT pulls more VCD into the Roy_D region or pushes them further to clean is empirically unknown. Predicted: small effect, possibly null.

### Either lever PARTIAL is now more likely than CONFIRM

Both interventions fight against the encoder's own amplification of the Roy_D-vs-clean axis. The most plausible deployment-grade lever is either:
- A direct contrastive supervision (`same_person(dor_shkedi, real_dor)` → similar embeddings), forcing the encoder to NOT separate them.
- A different backbone (face-balanced pretraining without the Roy_D-style axis).

Neither was launched in the auto-mode round.

## §6. What the FT amplification means for past packets

This finding contextualizes 13+ failed packets. All single-lever interventions that touched augmentation or loss-design were operating downstream of the encoder's axis-amplification dynamic. As long as FT runs at standard LR with no constraint on encoder representation, the axis gets stronger — masking any "data composition" or "augmentation invariance" lever that doesn't directly target the encoder representation.

## §7. Output files

- `outputs/encoder_embeddings_t5c_trained.csv`
- `outputs/encoder_embeddings_slot_b_trained.csv`
- `scripts/trained_encoder_probe.py` (the new SVD-aware loader)

## §8. The probe itself is a reusable diagnostic

This setup can now embed any cohort through any trained ckpt for follow-up packets. The post-scorecard analysis will use it to verify whether Slot A and Slot B post-FT encoders shift the Roy_D-vs-clean separation in any direction.

## §9. Trained T5C encoder — cohort positions (UPDATE; FT base for both A and B)

Same measurement on the T5C step3500 trained encoder (the base both Slot A and Slot B FT from):

| cohort | n | cos to Roy_D | cos to clean | % closer to Roy_D |
|---|---:|---:|---:|---:|
| Roy_D (target) | 130 | +0.988 | −0.155 | 100% |
| ilan+orel (clean) | 64 | −0.118 | +0.892 | 7.8% |
| **anchor pool** (Slot A target) | 30 | **+0.605** | **+0.602** | **53.3%** |
| VCD (Slot B boost target) | 100 | +0.171 | +0.758 | 27.0% |
| training_real_teams | 100 | −0.158 | +0.939 | 3.0% |
| **dor_shkedi (lockbox PNG)** | 1170 | **+0.679** | +0.545 | **59.7%** |
| real_dor (lockbox PNG) | 109 | +0.099 | +0.929 | 4.6% |

### §9.1 What this means for Slot A

In **T5C's encoder** (the FT base), the anchor pool is **53% Roy_D-adjacent**.
Half of the anchor pool's training-time supervision happens in Roy_D's
encoder neighborhood. This is **stronger mechanistic support for Slot A
than the Slot β-encoder probe (§3) suggested** — Slot β's repulsion of
anchor pool from Roy_D may be a downstream effect of multi_axis_grl, not
a property of all FT-from-T5C dynamics.

If Slot A's anchor_aware loss actively pulls prob_fake down on the anchor
pool, that pulls scores down in Roy_D's adjacency for ~53% of the
supervised content. This is the most plausible path for the lever to
generalize to Roy_D.

### §9.2 What it means for Slot B

VCD is 27% Roy_D-adjacent in T5C's encoder. 3× boost gives effective ~80%
of boost-mass at clean-side, only ~20% at Roy_D region. The weaker
mechanism prediction holds.

### §9.3 The trained-encoder finding from §2 (axis amplification) is replicated on T5C

Across encoders:
- Vanilla: PC1 sep +10.72, across-cosine +0.74
- **T5C: PC1 sep +3.88, across-cosine -0.155 (T5C has reduced PC1 sep but ALSO rotated cohorts)**
- Slot β: PC1 sep +3.11, across-cosine -0.927

T5C has rotated Roy_D and clean to be **nearly orthogonal** (cosine ≈0); Slot β rotated them to **nearly antipodal**. The FT progression amplifies the discrimination axis monotonically.

## §10. Final prediction update

- Slot A: **moderately likely to reduce dor chronic FP** (53% Roy_D-region overlap of anchor pool in T5C encoder). Roy_D generalization is conditional on whether anchor_aware loss pulls embeddings TOWARD canonical-real cluster, not just pushes scores down.
- Slot B: **weak prediction**. VCD's 27% Roy_D-adjacency in T5C is modest. Boost may help marginally.
- Compound (Slot A + Slot B): conjunction not tested but mechanistically distinct so could compose.
- Encoder-axis amplification is the dominant structural force — neither lever directly counters it.

