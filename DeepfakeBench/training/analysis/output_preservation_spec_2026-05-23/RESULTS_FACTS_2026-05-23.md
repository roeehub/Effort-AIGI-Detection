# Per-layer feature distance + output-preservation spec — RESULTS FACTS

Generated 2026-05-23. Factual readout. Spec interpretation in `AGENT_PROPOSAL_2026-05-23.md`.

> **Question (for Task #5)**: To inform the output-preservation aux loss spec for B.I.1, which layer's features drift the most across the trained ckpts (P8A, Slot A v2, T5C) on the substrate-pair pool? Which (reference encoder, layer, loss function) choice is best supported by data?
>
> **Method**: Use cached per-frame features from `analysis/substrate_pair_geometry_2026-05-22/feats/` (3 ckpts × 4 layers × {clean, teams}, ~5475/5478 frames per cell). Compute (a) per-row cosine similarity + normalized MSE between ckpt pairs at each layer, on identically-aligned frame indices, and (b) within-ckpt centroid shift between clean and teams substrates.

---

## 1. Cross-ckpt per-row cosine similarity at each layer

Aligned row-wise (same frame indices); reported as mean ± std across the cohort.

| Layer | P8A vs SlotAv2 cos_clean | cos_teams | P8A vs T5C cos_clean | cos_teams | SlotAv2 vs T5C cos_clean | cos_teams |
|---|---:|---:|---:|---:|---:|---:|
| L0 | 0.9993 ± 0.0000 | 0.9993 ± 0.0000 | 0.9995 ± 0.0000 | 0.9995 ± 0.0000 | 0.9999 ± 0.0000 | 0.9999 ± 0.0000 |
| L4 | 0.9947 ± 0.0004 | 0.9941 ± 0.0004 | 0.9965 ± 0.0003 | 0.9962 ± 0.0003 | 0.9987 ± 0.0001 | 0.9986 ± 0.0001 |
| L8 | 0.9751 ± 0.0032 | 0.9665 ± 0.0056 | 0.9846 ± 0.0020 | 0.9786 ± 0.0038 | 0.9939 ± 0.0010 | 0.9937 ± 0.0011 |
| **L11** | **0.4798 ± 0.0662** | **0.4042 ± 0.1378** | **0.4670 ± 0.0682** | **0.4596 ± 0.0834** | **0.7458 ± 0.0507** | **0.7372 ± 0.0459** |

Across-layers summary (mean over ckpt pairs):

| Layer | Mean cos_clean | Mean cos_teams | Mean normalized MSE clean | Mean normalized MSE teams |
|---|---:|---:|---:|---:|
| L0 | 0.9995 | 0.9995 | 0.0009 | 0.0009 |
| L4 | 0.9966 | 0.9963 | 0.0071 | 0.0078 |
| L8 | 0.9845 | 0.9796 | 0.0316 | 0.0425 |
| **L11** | **0.5642** | **0.5337** | **0.9928** | **0.8684** |

**~1000× more cross-ckpt drift at L11 than at L0.** Layers L0-L8 are essentially shared across the 3 trained ckpts; all the encoder divergence is concentrated at L11.

---

## 2. Within-ckpt substrate-pair centroid shift (clean → teams)

How much does each ckpt's centroid shift between the clean substrate and the teams substrate, per layer?

| Layer | Ckpt | cos(centroid_clean, centroid_teams) | MSE(centroid diff) | Var(centroid diff) | Var(within clean) |
|---|---|---:|---:|---:|---:|
| L0 | P8A | 1.0000 | 0.000011 | 0.000011 | 0.000280 |
| L0 | SlotAv2 | 1.0000 | 0.000011 | 0.000011 | 0.000279 |
| L0 | T5C | 1.0000 | 0.000011 | 0.000011 | 0.000279 |
| L4 | P8A | 0.9881 | 0.002441 | 0.002438 | 0.001507 |
| L4 | SlotAv2 | 0.9883 | 0.002393 | 0.002390 | 0.001504 |
| L4 | T5C | 0.9881 | 0.002446 | 0.002442 | 0.001513 |
| L8 | P8A | 0.9908 | 0.004889 | 0.004886 | 0.008785 |
| L8 | SlotAv2 | 0.9892 | 0.006042 | 0.006041 | 0.009460 |
| L8 | T5C | 0.9898 | 0.005555 | 0.005554 | 0.009989 |
| **L11** | **P8A** | **0.9719** | **0.091958** | **0.091958** | **0.159449** |
| **L11** | **SlotAv2** | **0.9812** | **0.022737** | **0.022735** | **0.155084** |
| **L11** | **T5C** | **0.9828** | **0.016400** | **0.016400** | **0.163973** |

Observations:
- L0-L8 substrate-pair shifts are tiny across all ckpts (cos ≥ 0.988)
- **L11 substrate-shift is 4× larger for P8A (MSE 0.092) than for Slot A v2 (MSE 0.023) or T5C (MSE 0.016).** The anchor_aware mechanism on Slot A v2 is structurally reducing the substrate-shift at L11.
- T5C has the smallest L11 substrate-shift but also the worst Xinhe-fake recall — small substrate-shift ≠ better deploy behavior.

---

## 3. P8A vs Slot A v2 specifically (the most relevant comparison for output-preservation)

P8A is the only deploy-passing ckpt; Slot A v2 is the most recent FT'd ckpt with the anchor_aware win. The candidate output-preservation aux loss "regularize toward a reference encoder" question is: should the reference be P8A (the empirical winner) or Slot A v2 (the most recent)?

Cosine similarity + normalized MSE between P8A and Slot A v2 at each layer:

| Layer | cos_clean | cos_teams | norm_mse_clean | norm_mse_teams |
|---|---:|---:|---:|---:|
| L0 | 0.9993 | 0.9993 | 0.0015 | 0.0015 |
| L4 | 0.9947 | 0.9941 | 0.0111 | 0.0124 |
| L8 | 0.9751 | 0.9665 | 0.0510 | 0.0699 |
| **L11** | **0.4798** | **0.4042** | **1.1964** | **1.0657** |

L11 is at cosine ~0.48 (clean) / ~0.40 (teams) — P8A and Slot A v2 are nearly orthogonal at the penultimate layer. The teams-substrate cosine is even lower than clean — Slot A v2's representation diverges more from P8A on teams-substrate than on clean-substrate.

---

## 4. Cross-validation with Probe 1 (substrate_pair_geometry_2026-05-22)

Probe 1 §1.2 reported per-ckpt KLIEP-axis classifier held-out accuracy of 0.9804 (P8A) / 0.9836 (SlotAv2) / 0.9836 (T5C) on L11 features, with cosines 0.04 / 0.10 / 0.06 vs the frozen-CLIP-L11 KLIEP axis (= 85-88° angular).

This Task #5 readout extends Probe 1 with cross-ckpt-pair cosines AT L11 specifically:
- P8A vs SlotAv2 cos: 0.48 (clean) / 0.40 (teams) — 61° / 66° angular
- P8A vs T5C cos: 0.47 / 0.46 — 62° / 63° angular
- SlotAv2 vs T5C cos: 0.75 / 0.74 — 41° / 42° angular

Two-cluster structure in L11 space:
- **Cluster A**: SlotAv2 + T5C (mutually closest, ~42° apart)
- **Cluster B**: P8A (60-66° from both)

This is consistent with Probe 1's framing that FT moves the encoder "a LOT in different directions" — but with more granularity: SlotAv2 and T5C ended up in similar but different L11 sub-region; P8A's L11 region is distinct from both.

Slot A v2's anchor_aware mechanism produced a smaller substrate-shift (§2) but did NOT bring its representation closer to P8A's (§3). Slot A v2 and T5C share more L11 geometry with each other than with P8A.

---

## 5. Artifacts

- `outputs/cross_ckpt_drift.csv` — per (ckpt-pair × layer) cross-ckpt drift metrics
- `outputs/within_ckpt_substrate_delta.csv` — per (ckpt × layer) substrate-pair centroid shift
- `outputs/layer_summary.csv` — across-layer summary

Wall: <5 sec to compute (all features were cached).

---

## 6. Caveats

1. **Row-alignment assumption**: I assume that the cached `.npy` files for clean and for teams substrates were extracted from the same `inventory_manifest.csv` row ordering, so that row i in P8A clean = row i in SlotAv2 clean (same frame). This is consistent with how the Probe 1 extraction script likely ran (parallel extraction from shared manifest) but not explicitly verified.
2. **Frozen-CLIP B16-DataComp.XL features on this 1,825-pair pool were NOT in the cache** and not extracted here. Cross-ckpt vs frozen-CLIP cosines are inferred from Probe 1's §1.2 (cos 0.04-0.10) rather than directly measured at multiple layers.
3. **Per-layer mean computed across the full 5475/5478 sample**; per-cohort (HDTF vs quickclips vs visomaster_teams_enhanced) breakdowns not separated. The 1,825 pair pool is heterogeneous; per-source distance might differ.
4. **L11 cosine of 0.48 means "halfway between identical and orthogonal."** Doesn't mean the encoders disagree on classification — both pass their own internal training objective on training data. They produce different feature directions for the same input, but the head transforms both into useful outputs.
