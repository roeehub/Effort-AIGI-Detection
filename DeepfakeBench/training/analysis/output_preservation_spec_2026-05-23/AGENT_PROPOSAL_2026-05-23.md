# AGENT_PROPOSAL — Output-preservation aux loss spec

Date: 2026-05-23. Interpretive doc — this IS the spec document that Task #5 was meant to produce. Accompanies `RESULTS_FACTS_2026-05-23.md` (feature-distance measurements).

Task #5 in the CPU-first sequence from `TRAINING_DIRECTIONS_REVIEW_2026-05-23.md`. The plan's B.I.1 lever was "output-preservation aux loss vs frozen anchor" with cost/EV estimates but no specification of layers, loss function, reference encoder, or weight. This doc names those choices with data-backed justification.

---

## 1. Spec: Output-preservation aux loss

### 1.1 Reference encoder: **P8A_REFERENCE_STEP5000**

Justification (in order of importance):
1. **P8A passes the team-identity bar** at per-ckpt-calibrated τ=0.59 (Tasks #1, #2). It's the only deploy-grade target we know works.
2. **P8A is the geometric outlier at L11** vs SlotAv2/T5C (RESULTS §3-4). Pulling new FT toward P8A is structurally different from pulling toward the SlotAv2/T5C cluster, which the plan's existing recipes already produce.
3. **Probe 1 finding** (FALLBACK1_PROBE1_FACTS_2026-05-22 §1.2): all FT'd encoders rotate 85-88° from the frozen-CLIP prior at L11. Using frozen-CLIP as reference would over-constrain — we WANT some FT drift, just not unbounded. P8A is FT'd to a known-good operating point; using it as reference defines "preserve known-good behavior."

Frozen-CLIP B16-DataComp.XL as an *alternative* reference is worth evaluating in Week 2 if P8A-reference fails; rejected as primary because we'd be preserving the wrong thing (the pre-FT prior, which doesn't pass the deploy bar).

### 1.2 Layer: **L11 (penultimate) + L8 with weighted contribution**

Justification:
- **L0-L4 are nearly identical across all FT'd ckpts** (cos ≥ 0.994; norm-MSE ≤ 0.011). Regularizing here is wasted compute — FT doesn't move these layers.
- **L8 has moderate cross-ckpt drift** (cos 0.97-0.99). Worth a small contribution (1/4 weight).
- **L11 has dramatic cross-ckpt drift** (cos 0.40-0.75). This is where the encoder divergence happens and where output-preservation needs to bite.

**Loss term**: `L_op = β_11 · L_cos(z_L11, z_L11_P8A) + β_8 · L_cos(z_L8, z_L8_P8A)` with `β_8 = β_11 / 4`.

### 1.3 Loss function: **Cosine-similarity-based** (negative cosine), not raw MSE

Justification:
- L11 features have meaningful magnitude variation (MSE differs 1000× between L0 and L11). Raw MSE would dominate the gradient unless heavily normalized.
- Probe 1 framed the problem as angular rotation (85-88° from frozen prior). Cosine is the natural metric for the angular structure.
- Cosine-loss has known stable gradient properties for representation preservation (CLIP training itself uses cosine-similarity objectives).

Specifically: `L_cos(z_a, z_b) = 1 - (z_a · z_b) / (||z_a|| · ||z_b||)`, per-frame, mean over batch.

Optionally add a small MSE regularizer to prevent magnitude collapse: `L_mse_norm = 0.1 · MSE(z_a / ||z_a||, z_b / ||z_b||)`. Probably not needed; CE loss + cosine already pins the geometry.

### 1.4 Training images for the aux loss

**The substrate-pair pool**: 1,825 matched (clean_i, teams_i) captures from `analysis/substrate_pair_geometry_2026-05-22/inventory_manifest.csv`. Sampler-fix from commit c2ed748 enables in-batch matched-pair co-occurrence.

For each substrate-pair sample (clean_i, teams_i):
1. Forward through the FT'd encoder → z_clean_i, z_teams_i
2. Forward through frozen P8A reference → z_clean_i_P8A, z_teams_i_P8A
3. Compute `L_op = L_cos(z_clean_i, z_clean_i_P8A) + L_cos(z_teams_i, z_teams_i_P8A)`

The substrate-pair pool is small relative to the training corpus (1,825 vs ~80K reals). Use a 50/50 batch composition: half from substrate-pair pool (for aux loss), half from regular training (for CE only).

For non-substrate-pair frames in the batch, aux loss is zero. This avoids the "more data" failure mode from the OPTB result (Task #4: 99.9% CV-separable from team-identity) — we're preserving on the substrate-relevant subset only.

### 1.5 Loss weight: **β_11 = 0.5, with warmup**

Justification:
- β=0 → no effect (Probe 1 says this matters; want effect)
- β=1.0 → equal to CE; risks dominating gradient on the small sub-batch where aux loss is computed
- β=10 → over-constrains; encoder can't depart from P8A
- **β=0.5 is the middle**: aux loss contributes meaningfully but doesn't dominate CE on the aux-active sub-batch

Warmup schedule:
- Steps 0-500: β_11 linearly ramps 0 → 0.5
- Steps 500-end: β_11 = 0.5 constant

Rationale for warmup: P8A reference features are not perfectly stable initially (the head learning curve isn't monotonic); ramping prevents an initial gradient explosion if the FT encoder starts far from P8A reference geometry. Inverse to typical IRM β-anneal-up; here anneal-up is correct because the loss is on representations, not classifier gradients.

### 1.6 Base ckpt: **P8A_REFERENCE_STEP5000** (same as reference) OR **P22 step1k**

Two variants worth running:

**Variant A**: Base P8A + preserve P8A (degenerate-by-construction)
- Aux loss is zero at step 0 (FT encoder = reference)
- Tests whether output-preservation can "hold the line" against further FT-drift while extending training
- Could just train more steps on P8A's data and see if Xinhe recall lifts without dor regression

**Variant B**: Base P22 step1k + preserve P8A
- Tests whether a different starting point on the gradient landscape + output-preservation regularization can find a better local optimum than P8A reached
- Per `project_p22_cpu_followups_reframe_2026-05-02`: P22 step1k has 140× wider score variance than step8k (the robust P22 ckpt)
- The combination is "explore from a different start, anchored toward P8A's known-good geometry"
- This is the morning reviewer's recommended combination

**Recommendation**: run Variant B as primary ($50, ~4h). If it bites, run Variant A as a $30 sanity check.

### 1.7 Anchor_aware compatibility

The Slot A v2 anchor_aware mechanism is the one packet-proven win to date. Should the aux loss compose with anchor_aware?

**No — at least not in the first run.** Per the morning reviewer's compound-experiment critique (and my TRAINING_DIRECTIONS_REVIEW §3.1), stacking known levers in one experiment obscures attribution. The first output-preservation run should be:

- Base: P22 step1k (NO anchor_aware)
- Aux loss: output-preservation vs P8A as specified above
- Standard CE on the binary task

If output-preservation alone shows a meaningful effect (anything > 5pp lift on Xinhe recall at per-ckpt-cal τ without dor regression), a Week 2 follow-up can stack anchor_aware.

### 1.8 Aux-loss ablation target

For attribution, include a $30 control run:
- Base: P22 step1k
- NO aux loss
- Otherwise identical training

This separates "P22 step1k as base" from "output-preservation aux loss" — the two ingredients my TRAINING_DIRECTIONS_REVIEW §3.1 said should be split.

---

## 2. Expected effect (calibration)

Per Task #1 (per-ckpt τ-recalibration), P8A's per-human metrics at τ=0.59:
- Xinhe fake recall: 0.603 (binding low cell)
- dor fake recall: 0.796
- max real-FPR: 0.037 (dor)

A "materially better" output-preservation result would:
- Lift Xinhe recall above 0.70 (closing the gap to face-pool's 0.849)
- Not regress dor recall below 0.75
- Not regress dor real-FPR above 0.05

A "marginal" result: any lift in Xinhe recall while not regressing others.

A "null result": Xinhe recall change ±5pp, all other cells unchanged.

A "negative result": Xinhe recall drop, OR any per-human FPR breach.

**Probability assessment**: 20-30% materially better, 40-50% informative (some directional movement), 20-30% null, 5-10% negative (would be surprising given P8A reference is a stable target).

These are slightly higher than the plan's 15-25% because the spec is more concrete and the reference choice (P8A) is empirically validated, not speculative.

---

## 3. Cost

- **GPU**: $50-60 single A100 4h run (Variant B, primary)
- Optional: $30 ablation (base P22 step1k, no aux loss)
- Optional Week 2: $50 stacking variant (output-preservation + anchor_aware)

If the spec doc is "1-page" justification of these choices and the ablation is included: total Week 1 commitment $80-110 for this lever class.

---

## 4. Risks

1. **Reference encoder drift over training**: P8A reference features are computed via forward pass through frozen P8A weights. If memory or wallclock requires fewer reference-forward passes (e.g., cached features only for the substrate-pair pool), we can precompute the 1,825 × 2 substrate-pair reference features once at the start (~5 min CPU/MPS) and reuse throughout training.
2. **Cosine loss can plateau at local optima**: if the FT encoder finds a representation that's cosine-close to P8A but still produces different head outputs (because the head transforms differently), the aux loss won't catch it. Mitigation: also report per-frame head-output L2 distance to P8A's head output during training as a diagnostic.
3. **The 1,825-pair pool is small**: aux loss may not bite if substrate-pair batches are rare. Mitigation: 50/50 batch composition ensures aux loss is active in every step.
4. **Anchor_aware combination might be necessary**: if output-preservation alone doesn't bite, the user might want to immediately compose with anchor_aware. Spec ensures clean attribution; if user accepts the attribution cost, the stacked run can come later.

---

## 5. What this spec does NOT cover

- **Variant for "preserve frozen-CLIP" (per Probe 1 angular framing)**: the spec uses P8A as reference. If the user wants the frozen-CLIP alternative explicitly, add a $50 Variant C run with frozen-CLIP as reference. My recommendation rejects this as primary but it's a legitimate alternative.
- **The face-pool / dual-pool head dependency**: this spec is encoder-output-preservation. The HEAD ALT lever (dual face+non-face) is a head-architecture change and orthogonal.
- **IRM combination**: IRM is a different mechanism (classifier-gradient invariance) and not stacked here per §1.7.

---

## 6. Self-correction log

- **Initial reference choice**: I initially considered frozen-CLIP as the reference (the morning reviewer's framing). Switched to P8A after reviewing Task #5's measurement: cross-ckpt drift is concentrated at L11, where the FT encoders ARE supposed to specialize. Pulling them toward frozen-CLIP would erase the specialization. P8A is "specialized in a useful direction"; pulling toward it preserves the useful direction.
- **Layer choice**: I initially considered "all layers, weighted by importance." Realized L0-L4 don't drift across FT'd ckpts anyway — no point regularizing where FT doesn't move. L11 alone with small L8 contribution is the right scope.
- **Loss function**: I initially leaned MSE for simplicity. Switched to cosine after seeing the 1000× MSE difference between L0 and L11 — raw MSE on L11 would dominate any L8 contribution and would over-emphasize magnitude over direction.
- **β weight**: I initially proposed β=0.1 (conservative). After seeing the magnitude of L11 cosine drift (~0.5), realized β=0.1 might be too weak to drive geometry change. β=0.5 with warmup is more aggressive but bounded.
- **Anchor_aware combination**: I initially considered stacking with anchor_aware (Slot A v2 mechanism + new aux loss). On reflection, this repeats the compound-experiment fault the review criticized. Spec is now "output-preservation alone first, stack in Week 2 if bites."

---

## 7. Followups (TODOs for user-decided application)

### Action items the user must commit to before launching the GPU run

1. **Confirm reference encoder is P8A** (or override with frozen-CLIP / Slot A v2).
2. **Confirm base ckpt is P22 step1k** (or override with P8A or Slot A v2).
3. **Pre-extract P8A reference features on the substrate-pair pool** — ~5 min CPU/MPS. Save to `analysis/output_preservation_spec_2026-05-23/outputs/p8a_substrate_pair_features.npz`. Trainer loads at startup.
4. **Implement the aux loss in the trainer** — ~50 LOC. Hook into the existing substrate-pair sampler from c2ed748.
5. **Run the $30 ablation** (P22 step1k base, no aux loss) in the same launch sequence for attribution.

### Memory updates

- **NEW**: `project_output_preservation_aux_loss_spec_2026-05-23.md` — "Spec for B.I.1 output-preservation aux loss landed 2026-05-23 PM. Reference: P8A_REFERENCE_STEP5000. Layers: L11 (β=0.5) + L8 (β=0.125). Loss: cosine-similarity per-frame. Pool: 1,825 substrate-pair captures (50/50 batch composition with regular training). Base: P22 step1k. Cost: $50-60 + $30 ablation. Justified by Task #5 finding that ~1000× more cross-ckpt drift is at L11 vs L0; P8A is geometric outlier vs SlotAv2/T5C cluster at L11. Spec doc: analysis/output_preservation_spec_2026-05-23/AGENT_PROPOSAL_2026-05-23.md."

### Threads to amend

- `docs/packet_retrospectives/threads/iq_shortcut_deconvolution_program_2026-05-08.md` — append section on output-preservation as a structurally-distinct lever now spec'd.

### TIMELINE

- Append: `2026-05-23 PM — output-preservation aux loss spec (analysis/output_preservation_spec_2026-05-23/) — per-layer feature-distance shows ~1000× cross-ckpt drift at L11 vs L0; P8A geometric outlier vs SlotAv2/T5C cluster (cos 0.48 vs 0.74). Spec: ref=P8A, layer=L11+L8, loss=cosine, pool=1825 substrate-pair, base=P22 step1k, β_11=0.5 with warmup. Cost $50-60 + $30 ablation.`

### OPEN_LOOPS

- Open: "Pre-extract P8A reference features on substrate-pair pool" — 5 min CPU/MPS, prerequisite to GPU launch.
- Open: "Implement output-preservation aux loss in trainer" — ~50 LOC.
- Open: "Launch B.I.1 with this spec + $30 ablation" — Week 1 GPU.

---

## 8. Gaps and blockers

- **Frozen-CLIP B16-DataComp.XL features on the 1,825-pair pool NOT cached** — would need ~6-10 min MPS extraction if the user wants to evaluate frozen-CLIP-reference as Variant C. Cheap to add if needed.
- **No head-output L2 distance diagnostic yet** — recommended as a training-time canary per Risk 2. Not blocking the run; just a nice-to-have.
- **Row-alignment in the cached features assumed (not verified)** — RESULTS §6.1 caveat. Verifying would take 1 min (load inventory_manifest.csv and check ordering). Worth doing before launch.
