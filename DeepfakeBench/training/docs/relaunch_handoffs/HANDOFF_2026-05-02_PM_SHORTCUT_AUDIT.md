# Handoff — image-quality shortcut audit + next-experiment planning (2026-05-02 PM)

**Date**: 2026-05-02 PM (after 11-diagnostic CPU-only forensic audit on the D contract scorecard)
**Branch**: `teams-relaunch-root-2026-04-17`
**Status**: No new GPU job. Comprehensive forensics complete. Image-quality shortcut characterised in detail. Three concrete next-experiment proposals below; user wants explicit hypotheses + expected outcomes + fail signals before any GPU spend.

---

## Canonical entry points (read in order)

> 1. **`docs/relaunch_handoffs/AGENT_GUIDE_2026-05-02.md`** — procedural rules, hard-rules checklist, validate-before-suggest. **Still authoritative.** The PM forensics did not change any of these rules.
> 2. **`docs/relaunch_handoffs/PSERIES_FACTS_2026-05-02.md`** — pure-data ledger of every P-series packet (P8A → P18). yaml paths, W&B runs, Vertex jobs, scorecard outcomes. **Zero interpretation.**
> 3. **THIS DOCUMENT** — what's new since the morning, the explicit shortcut framing, and the three candidate next experiments with hypotheses + expected outcomes.
> 4. **`analysis/score_distribution_2026-05-02/FINDINGS.md`** — 739 lines, pure-data ledger of the PM forensics audit. Cite this for any specific number you reproduce.
> 5. **`analysis/score_distribution_2026-05-02/outputs/REPORT.html`** — visual aggregator, 17 sections, 70 figures. Open directly or via the viewer at `/api/model-runs/p8a/static-report`.
> 6. **`docs/relaunch_handoffs/PSERIES_OPINIONS_2026-05-02.md`** — collected past framings with disclaimer. Read AFTER FACTS+FINDINGS.
> 7. **`CLAUDE.md`** + **`MEMORY.md`** — operational rules + memory index. New PM entries: `project_lockbox_identity_looseness`, `project_score_distribution_audit_2026_05_02`, `project_image_quality_shortcut`.

---

## TL;DR

**The model has an image-quality shortcut** — score correlates negatively with Laplacian variance, luminance, and skin-fraction across most suites. The eval data systematically violates the training distribution on these axes (eval is 4-25× LESS SHARP than training). This is the single most explanatory finding from the PM audit and it ties together:

- Why viso recall is ~1% at deployment τ but 35% at τ=0.5
- Why lockbox cam_test_s33 (Laplacian 9.5) is missed in 94% of frames by all 3 models
- Why training-side metrics improved across P14/P15/P16/P17/P18 but contract-grade τ-tail recall didn't move
- Why P18T's GRL "wins" are not sharper than missed (different mechanism — defensive against P18C regression, not additive over P8A)

**Lockbox identities are deliberately loose** (user-confirmed): the same person (PC_Generator, dor, bla_bla_chow, dor_shkedi) appears as both real and fake in both dev and lockbox. Identity-leakage across the dev↔lockbox boundary is intentional eval design, not an oversight. This affects how to interpret per-identity FPR.

**Operating-point arithmetic** (computed, not opined): if real-FPR floor relaxes from 2% (contract) to 10%, P18C unlocks 74% teams_fake + 77% deeplive; P8A unlocks 70% teams_fake + 54% lockbox + 42% deeplive. The model is much closer to "usable" than the contract scorecard implies.

---

## The shortcut problem — explicit framing

This is the section the user asked to highlight. **Read carefully** before proposing any packet.

### What the shortcut is

The model has learned that **darker + less sharp + more skin-dominated = fake**. This is a learned feature representation, not an architectural bug. It manifests as:

- Pearson r(score, luma_mean) negative on multiple suites — strongest -0.55 on teams_fake_all_dev for P8A
- Pearson r(score, laplacian_var) negative on real suites and on most fake suites — strongest -0.57 on dor reals for P18T
- Pearson r(score, skin_frac) negative on lockbox — -0.49 across all 3 models on teams_fake_all_lockbox
- Pearson r(score, h) negative on lockbox — -0.50 for P8A, -0.49 for P18C

**Source**: `analysis/score_distribution_2026-05-02/outputs/score_attribute_correlations.csv`, `figures/score_attribute_corr_heatmap_P8A.png`.

### Why the shortcut breaks at eval time

The training data has these distributions on Laplacian variance:
- visomaster_CSCS: 155, GhostFace: 162, SimSwap512: 186, InStyleSwapper: 190, Inswapper128: 140
- Enhanced (codeformer, gfpgan): 407
- Realpool (training reals): 132

The eval data has these distributions:
- visomaster_enhanced_macro_dev raw: 60
- visomaster_enhanced_macro_dev teams: 36
- teams_fake_all_lockbox (cam_test_s33 dominates): 17.5

**Eval is 4-25× LESS SHARP than training**. The shortcut the model learned ("fakes are sharp because that's how they are in training") fails systematically on eval data that isn't sharp.

Source: `analysis/score_distribution_2026-05-02/outputs/cross_suite_attribute_summary.csv`, `figures/cross_suite_attr_box.png`, `figures/train_vs_eval_attr_overlay.png`.

### The shortcut also explains the within-eval variation

Of the 275 paired viso sequences, 144 (52%) are missed by ALL three models in BOTH substrates at τ=0.5. Those 144 pairs are **systematically less sharp** than the 131 sometimes-caught pairs (Laplacian 43.6 vs 71.4, p<1e-4 Mann-Whitney). Pose / yaw / pitch / eye distance / mouth / bbox area are ALL non-significant — pose is RULED OUT.

Within cam_test_s33 (already at extreme low sharpness), sharpness no longer distinguishes caught vs missed (p=0.39). The model has effectively "given up" once it sees a low-sharpness regime.

### What's been tried against the shortcut

| packet | lever | outcome | shortcut targeting |
|---|---|---|---|
| P14_FACE_SCALE_JITTER (mclioexb) | jitter@0.50 only | no contract lift | indirect (jitter perturbs scale, not sharpness) |
| P14 bundle (anchor + pipeline_rand + jitter) | multi-lever | no lift, sometimes worse | indirect |
| P15_GRL_FROM_P8A (w5tky6ss) | GRL @ static λ=0.20 on capture-mode | no lift; wrong axis per Phase 1A | doesn't target shortcut at all |
| P16_DATA_AXIS (rmic6wrc) | visomaster_teams_enhanced fw=2.0 | no contract lift; viso 51% at τ=0.5, 1% at τ=0.99 | adds the same data the shortcut already learns |
| P14_DATA_FIX (xan4dfto) | same data + bundle fw=8.0 | trainer-side collapse | same |
| P17 layer-3/4 readout (ArcFace + LINEAR) | new head | failed, learned away from invariant signal | doesn't target shortcut |
| P18 12-bucket method-conditional GRL | architectural | defensive vs P18C regression, not additive over P8A | doesn't target shortcut |

**None of these directly targeted the image-quality shortcut.** They targeted: data presence (P14/P16), capture-mode axis (P15), method-cluster axis (P18), readout architecture (P17). The shortcut was implicit but unaddressed.

### How any next packet should reason about the shortcut

A packet proposal should answer **all** of these:

1. **What does this packet do to the image-quality shortcut?** Does it weaken it (e.g., training augmentation), bypass it (e.g., new feature path), or relax the operating point that exposes it (e.g., calibration loss / threshold relaxation)?
2. **What's the predicted Pearson r between score and laplacian_var on viso AFTER this packet?** If it doesn't reduce |r|, it doesn't fix the shortcut.
3. **What's the falsifier?** What outcome on a CPU diagnostic would update your belief away from this packet?

If a packet proposal can't answer these three, it isn't ready.

---

## What's new since the morning handoff

The morning agent (me, earlier in this session) wrote `HANDOFF.md` and the PM_AUDIT_RESTRUCTURE story. Since then:

### CPU diagnostics added (zero GPU spend)

11 diagnostics launched across two PM batches. Full list and outputs:

**Batch 1 (run_diagnostics, paired_substrate, pixel_diff, single_pair_diff)**:
- Per-suite histograms (9 suites × 3 models)
- Per-suite quantiles
- Viso winners identification (the 6 frames P8A scores correctly)
- Paired raw vs teams substrate analysis (n=275 pairs)
- Pixel-level diff between paired raw/teams images (mean L1=33, +34 luma shift, 2.5× HF reduction)
- Single-pair deep diff inspection (seq5402)

**Batch 2 (sharpness_validation, identity_fpr, lockbox_walkthrough, train_data_attrs, pose_audit, embedding, threshold_relaxation)**:
- Sharpness hypothesis validation per model (raw_only categories sharper vs both_missed across all 3)
- Identity-level FPR breakdown for teams_real_all_dev (13 unique identities, 96% of FPs in 3)
- Lockbox fake walkthrough (94% cam_test_s33 missed by ALL 3 at deployment τ)
- Train data attribute distribution (4-25× sharpness gap vs eval)
- Pose / face-detection audit (pose RULED OUT)
- ResNet-50 ImageNet feature embedding (raw_only pairs furthest apart in feature space)
- Threshold-relaxation grid + operating-point feasibility table

**Batch 3 (cross_suite_attribute_extraction, identity_overlap, identity_comparison_gallery)**:
- Cross-suite attribute landscape (lockbox = least sharp suite, real = sharp range)
- Identity overlap analysis (3 strings span dev↔lockbox; ZERO FPR-driver identities in training bucket)
- Identity comparison visual (dor / dor_shkedi / real_dor / deeplive_dor = same person)

### Viewer integration

- `viewer/model_dashboard_runs.yaml` updated with P18T and P18C entries; `per_frame_scores` artifact now wired for all 3 models
- `viewer/server.py` extended with `/api/model-runs/<run_id>/static-report` and `/api/static-report-asset/<path>` endpoints
- `viewer/model_dashboard.py` extended to fall back to GCS proxy for non-local frames in `_per_frame_scores`
- `analysis/score_distribution_2026-05-02/outputs/REPORT.html` accessible at `http://localhost:5000/api/model-runs/p8a/static-report` after `python -m viewer --no-auto-discover --port 5000`

### MEMORY index updated

Three new entries:
- `project_lockbox_identity_looseness.md`
- `project_score_distribution_audit_2026_05_02.md`
- `project_image_quality_shortcut.md`

These now load automatically in future sessions.

---

## Recommended next experiments (with hypotheses, expected outcomes, fail signals)

Three packets ranked by cost and likelihood of moving the needle. **Each is optional and requires user OK before launch.** Each must be CPU-validated first per AGENT_GUIDE Rule 3.

### Packet P21 — Operating-point recalibration (zero GPU spend, $0)

**Hypothesis**: the contract τ ≈ 0.99 is the bottleneck. At a relaxed operating point (e.g., 5% real_FPR floor), the model is much closer to deployable.

**What to do**:
- No training. Pure CPU + decision.
- Re-evaluate P8A and P18C on the existing per-frame predictions at fixed real_FPR floors of 5% and 10%.
- Decide whether the deployment specification can accept that operating point.

**Expected outcome (already computed in `outputs/operating_point_feasibility.csv`)**:
- At 10% real_FPR: P8A unlocks teams_fake 70% + lockbox 54% + deeplive 42%; P18C unlocks teams_fake 74% + deeplive 77% + viso 12%
- At 5% real_FPR: P8A unlocks teams_fake 53% + lockbox 30% + viso 6%; P18C unlocks teams_fake 60% + deeplive 50% + viso 7%
- dor real FPR at 5% floor: P8A 4% (was 0% at 2%), P18C 14%

**Fail signal**: deployment requires 2% FPR for compliance/UX reasons → P21 is moot, fall back to P22 or P23.

**Cost**: $0. Just policy.

**Recommended only if**: the user is willing to consider whether 2% FPR is the right operating spec.

---

### Packet P22 — Train-time blur/brightness/JPEG augmentation (estimated $30-45 GPU)

**Hypothesis**: the model relies on sharpness/brightness/skin_frac as fake predictors because training data has those distributions. If we explicitly augment training data to match the eval-side distribution (less sharp, brighter, more compressed), the model will learn detector features that generalise to the eval substrate.

**What to do**:
- FT from P8A_step5000 (proven baseline)
- Add training augmentations:
  - Gaussian blur with random sigma in [0, 4] applied with probability 0.5 to fake AND real
  - Random brightness shift in [-40, +40] with probability 0.5
  - JPEG re-encoding at random quality in [50, 95] with probability 0.5
- Otherwise: same hyperparameters as P8A. No GRL. No additional sources.
- Train for 8000 steps (matches P8A regime), checkpoint every 500.

**Expected outcomes (α/β/γ)**:
- α (success): viso recall at deployment τ moves from 1% → 5-10%; deeplive recall from 2% → 10-20%; lockbox cam_test_s33 from 5% → 15-25%. Pearson r(score, laplacian_var) on dev viso reduces in magnitude (e.g., from -0.4 → -0.2).
- β (defensive only): viso recall stays at ~1% but training-side composite holds steady. The augmentation didn't change the τ-tail. → P18T-style outcome.
- γ (collapse): training-side metrics regress (composite drops below 0.5). Augmentations were too aggressive and hurt the in-distribution learning. → like xan4dfto.

**CPU pre-validation (must run before GPU launch)**:
- Apply the same augmentations to a sample of 200 training frames; verify the post-augmentation Laplacian distribution overlaps the eval distribution. If the post-aug Laplacian still sits in [100, 400] range, the augmentation is too weak.

**Cost**: ~$30-45 Vertex (1 day us-east1 A100).

**Recommended if**: P21 isn't acceptable AND the user wants to try the most-direct shortcut intervention.

---

### Packet P23 — Score-margin / focal loss to push τ-tail (estimated $30 GPU)

**Hypothesis**: the model has the right ranking signal at τ=0.5 (P16 hit viso 51%) but loses it at τ=0.99 because the score distribution is too soft. A focal loss on the positive class (γ=2-3) or an explicit margin term will push true positives higher in the score distribution.

**What to do**:
- FT from P8A_step5000
- Replace BCE with focal loss (γ=2, α=0.5 to start) on the fake class
- Otherwise unchanged from P8A.
- Train for 6000 steps.

**Expected outcomes**:
- α: at fixed deployment τ, viso recall moves from 1% → 8-15%; deeplive from 2% → 10-20%. The shape of the histogram changes — fewer frames at p=0.5, more at p=0.99.
- β: τ-tail moves but real_FPR also increases. Need to adjust α to balance.
- γ: training instability or composite collapse if γ is too high.

**CPU pre-validation**:
- On the existing per-frame predictions, test whether re-weighting the predictions by a focal-style transform changes the recall-vs-FPR curve favourably. If it doesn't, focal won't help either.

**Cost**: ~$25-35 Vertex.

**Recommended if**: P21 + P22 don't appeal AND the user wants to try the τ-tail-direct intervention from the OPINIONS doc.

---

### Why we are NOT proposing more data-axis or GRL packets

Per memory `project_data_axis_lever_pulled_twice_no_lift.md`: data-axis lever has been tried twice (xan4dfto fw=8.0 collapse, rmic6wrc fw=2.0 no-lift). Per memory `project_p18_d_contract_p8a_wins_no_floor.md` and `project_phase1a_method_cluster_axis_2026-05-01.md`: GRL on multiple axes has been tried (P15 capture-mode, P18 12-bucket method-conditional). Neither lever is the right intervention for the image-quality shortcut. AGENT_GUIDE Rule 1 (validate-before-suggest) requires articulating what's structurally different from prior failures.

P21/P22/P23 each address an axis NOT previously tried (operating-point, train-time augmentation matching eval distribution, focal/margin loss for τ-tail).

---

## Pending decisions for the user

These are NOT to be made unilaterally:

1. **Strategic direction.** Pick from P21 (free, policy-only), P22 (augmentation, ~$30-45), P23 (focal loss, ~$25-35), or none-of-the-above + alternative direction.
2. **Whether to commit the score_distribution_2026-05-02 working tree.** All artifacts are uncommitted. Spans `analysis/score_distribution_2026-05-02/` (16 scripts, 33 CSVs, 70 figures), `viewer/server.py` (3 endpoint additions), `viewer/model_dashboard.py` (1 fallback change), `viewer/model_dashboard_runs.yaml` (P18T+P18C entries plus P8A score-distribution artifact). Would be a substantial commit.
3. **Whether to delete `outputs/viso_full_paired/` (~70 MB) and `outputs/cross_suite_samples/` (~50 MB) from local disk.** Re-downloadable from GCS at any time. The CSVs / figures / parquet are the persistent value.

---

## Files modified / created this session

```
A  analysis/score_distribution_2026-05-02/                                    (new directory)
A  analysis/score_distribution_2026-05-02/FINDINGS.md                          (739 lines)
A  analysis/score_distribution_2026-05-02/build_report.py
A  analysis/score_distribution_2026-05-02/run_diagnostics.py
A  analysis/score_distribution_2026-05-02/paired_substrate_analysis.py
A  analysis/score_distribution_2026-05-02/pixel_diff_analysis.py
A  analysis/score_distribution_2026-05-02/threshold_relaxation_grid.py
A  analysis/score_distribution_2026-05-02/crop_attribute_audit.py
A  analysis/score_distribution_2026-05-02/train_data_attribute_distribution.py
A  analysis/score_distribution_2026-05-02/sharpness_validation.py
A  analysis/score_distribution_2026-05-02/identity_fpr_breakdown.py
A  analysis/score_distribution_2026-05-02/lockbox_fake_walkthrough.py
A  analysis/score_distribution_2026-05-02/train_vs_eval_compare.py
A  analysis/score_distribution_2026-05-02/pose_audit.py
A  analysis/score_distribution_2026-05-02/feature_embedding.py
A  analysis/score_distribution_2026-05-02/cross_suite_attribute_extraction.py
A  analysis/score_distribution_2026-05-02/identity_overlap.py
A  analysis/score_distribution_2026-05-02/identity_comparison_gallery.py
A  analysis/score_distribution_2026-05-02/raw_reports/                          (27 CSVs, 8.3MB)
A  analysis/score_distribution_2026-05-02/outputs/                              (33 CSVs, 70 figs, 5 parquets, REPORT.html)
A  analysis/score_distribution_2026-05-02/outputs/viso_full_paired/             (550 frames, ~70 MB)
A  analysis/score_distribution_2026-05-02/outputs/lockbox_fake_walkthrough/     (425 frames)
A  analysis/score_distribution_2026-05-02/outputs/cross_suite_samples/          (450 frames)
A  analysis/score_distribution_2026-05-02/outputs/identity_gallery/             (48 frames)
A  analysis/score_distribution_2026-05-02/outputs/identity_comparison/          (64 frames)
A  analysis/score_distribution_2026-05-02/outputs/train_data_samples/           (359 frames)
A  analysis/score_distribution_2026-05-02/outputs/train_vs_eval/                (samples + manifest)
A  docs/relaunch_handoffs/HANDOFF_2026-05-02_PM_SHORTCUT_AUDIT.md               (THIS DOC)
M  viewer/server.py                                                              (+3 endpoints)
M  viewer/model_dashboard.py                                                     (1 GCS fallback)
M  viewer/model_dashboard_runs.yaml                                              (+2 runs, +1 P8A artifact)

Memory:
A  ~/.claude/.../memory/project_lockbox_identity_looseness.md
A  ~/.claude/.../memory/project_score_distribution_audit_2026_05_02.md
A  ~/.claude/.../memory/project_image_quality_shortcut.md
M  ~/.claude/.../memory/MEMORY.md                                                (+3 pointer lines)
```

Working tree is uncommitted per project pattern — do NOT commit without explicit user OK.

---

## Operational notes

- 0 active Vertex jobs.
- Image: 1.3.241 in GCR (current).
- Local /tmp/p18_ckpts/ ≈ 17 GB; /tmp/p17_ckpts/ ≈ 8 GB; downloadable from GCS.
- Spend this session: $0 GPU (entirely CPU-only forensics). Cumulative session: ~$10-15 (Cloud Build + earlier D scorecard).
- VERSION file: 1.3.241.

---

## Cron / wakeup

No active wakeups.

---

## What the next agent should NOT do

- Do NOT propose another data-axis lever (P14/P16-style) without articulating what's structurally different from the prior failures.
- Do NOT propose another GRL change (P15/P18-style) without justifying which axis it targets and why.
- Do NOT propose a new layer-X readout (P17-style) — structurally dead per the P17 verdict.
- Do NOT skip the AGENT_GUIDE validate-before-suggest checklist.
- Do NOT run the score-distribution audit again — it's done. Cite `FINDINGS.md` for any specific number.
- Do NOT re-download the 550 viso paired frames — they're cached at `analysis/score_distribution_2026-05-02/outputs/viso_full_paired/`.

## What the next agent SHOULD do

- Read AGENT_GUIDE → PSERIES_FACTS → THIS DOC → FINDINGS.md → REPORT.html in that order before doing anything.
- If the user wants to proceed, execute one of P21/P22/P23 OR justify a different proposal that addresses the image-quality shortcut.
- If a new diagnostic is needed, prefer CPU-only and reuse the cached frames.
- If proposing a GPU packet, follow the validate-before-suggest checklist explicitly and write each box's answer in the proposal.

---

*Authored 2026-05-02 PM by the agent that ran the 11-diagnostic forensics audit. The audit data is independent of these recommendations; the recommendations are opinion (per AGENT_GUIDE Rule 5).*
