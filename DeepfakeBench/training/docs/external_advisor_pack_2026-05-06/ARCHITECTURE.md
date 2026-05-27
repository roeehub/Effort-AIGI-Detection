# Architecture — Effort detector for face-swap manipulation detection

> Self-contained architecture description for an external advisor with no repo access. Implementation details cited from `detectors/effort_detector.py` (2,168 LOC) and `config/detector/effort.yaml`.

## Reference

Original method: Effort — "Towards Generalizable Deepfake Detection by Disentangled Representation Learning" (Li, Pinto, et al., NeurIPS 2024 / arXiv preprint).
- Paper: https://arxiv.org/pdf/2411.15633
- Core idea: Apply low-rank SVD-based residual modeling to a CLIP backbone's attention/MLP weights to disentangle forgery-relevant features from forgery-irrelevant features. The fixed (frozen) main weight is kept; a learnable low-rank residual `U @ diag(s) @ V` is added on top. Orthogonality regularization on `(U_r, U_residual)` and `(V_r, V_residual)` keeps the residual disjoint from preserved subspaces.

This codebase implements the Effort idea on OpenCLIP-B/16 + a 2-class classification head, with a substantial set of additions accumulated during the R13 iteration (April-May 2026).

## High-level model description

```
Input image (3 × 224 × 224, BGR loaded → INTER_LINEAR resize → BGR2RGB → CLIP normalize)
        │
        ▼
[Backbone: OpenCLIP ViT-B/16 visual encoder]
   • Patch embed (16×16 patches → 196 patch tokens + 1 CLS = 197 tokens × 768 dim)
   • 12 transformer resblocks
       Each resblock: LN → MHA(in_proj fused [Q;K;V]; out_proj) → LN → MLP(c_fc → GELU → c_proj)
       SVD residual is applied (see below) to selected linear layers in each resblock
   • ln_post (final LayerNorm on [CLS])
   • visual.proj: 768 → 512 (final linear projection)
        │ (output: [B, 512] = "feat")
        ▼
[Optional: feature normalize (L2) before head, when normalize_features_before_head=True]
        │
        ▼
[Head: nn.Linear(512, 2)  OR  ArcMarginProduct(512, 2, s, m)]
        │ (output: [B, 2] logits)
        ▼
[Softmax → prob_fake = output[:, 1]]
```

The detector class (`EffortDetector` at `detectors/effort_detector.py:329`) wires backbone + head + losses + augmentation hooks.

## SVD residual implementation

Implemented via `SVDResidualLinear(nn.Module)` (`detectors/effort_detector.py:1398`). Given an existing `nn.Linear(in, out)` with weight `W₀`:

1. **Fixed main weight** (`weight_main`): a frozen copy of `W₀`. `requires_grad=False`.
2. **Learnable residual**: three trainable parameters `U_residual`, `S_residual`, `V_residual` with shapes such that `U_residual @ diag(S_residual) @ V_residual` produces a low-rank matrix of the same shape as `W₀`. The rank parameter is `k = d - r` where `d` is the input/output dim and `r` is the configured rank.
3. **Effective weight at forward time** (`compute_current_weight()`):
 ```
 W_effective = W₀ + U_residual @ diag(S_residual) @ V_residual
 ```
4. **Frozen reserved subspace** (`U_r`, `V_r`): the top-`r` singular vectors of the original `W₀`; kept frozen and used by the orthogonality regularizer to enforce that the learned residual is orthogonal to the preserved subspace.

### Where SVD is applied (configurable)

- `apply_svd_to_mlp: true|false` — when true, replaces `c_fc` and `c_proj` (the two MLP linear layers in each resblock) with `SVDResidualLinear`. Default in `effort.yaml`: false. P8A enables this.
- `apply_svd_to_in_proj: true|false` — when true, applies SVD residual to the fused Q/K/V `in_proj_weight` of each resblock's `nn.MultiheadAttention`. Default true in newer configs.
- `out_proj` of each MHA is replaced by `SVDResidualLinear` (the output projection). This path was always autograd-correct.
- `block_indices` — optionally restrict SVD application to specific resblocks (by default all 12).

### The in_proj-SVD silent zero-gradient bug (fixed 2026-04-26, commit `2feea58`)

Pre-fix, the in_proj-SVD residual was installed via a `forward_pre_hook` that did:
```
module.in_proj_weight.data.copy_(module._svd_in_proj.weight)
```
where `_svd_in_proj.weight` is an autograd-tracked property. The `.data.copy_()` writes values **without participating in autograd**, so `F.multi_head_attention_forward` reads the fused weight tensor disconnected from the residual parameters. Result: `U_residual`, `S_residual`, `V_residual` for the q/k/v projections received gradient ONLY from the orthogonality regularizer — never from classification loss. They drifted toward orthogonality but learned nothing about the task.

**Affected runs**: every R12 / RLP / P-* run with `apply_svd_to_in_proj: true` from project start through 2026-04-25 (P8A). Implication: P8A's anchor-pool improvement came from the MLP-SVD path + unfrozen `visual.proj` + unfrozen `ln_post` ONLY — the in_proj-SVD capacity was a silent no-op.

Fix: override the `nn.MultiheadAttention.forward` to call `F.multi_head_attention_forward` with the autograd-tracked `_svd_in_proj.weight` property directly (`detectors/effort_detector.py:1876-1969`).

### SVD regularization losses

`SVDResidualLinear` exposes three optional regularizers:

- **`compute_orthogonal_loss`**: enforces `[U_r ; U_residual]` and `[V_r ; V_residual]` are orthogonal — `‖A^TA - I‖_F`. Discourages the learned residual from collapsing into the preserved subspace.
- **`compute_keepsv_loss`**: pulls the Frobenius norm of `W_effective` toward the original `‖W₀‖_F`. Limits how much the residual can grow the layer's overall capacity.
- **`compute_fn_loss`**: simple `‖W_effective‖_F^2` regularizer.

Total SVD regularizer weight is `lambda_reg` in config, optionally annealed from `lambda_reg_start` to `lambda_reg_end` over `lambda_reg_anneal_steps`. Default in `effort.yaml`: 0 (regularizer off unless explicitly enabled).

## Head and loss

### Standard linear head (`use_arcface_head: false`, default)
- `nn.Linear(512, 2)` directly on the pooled [CLS] feature.
- Loss: `CrossEntropyLossWithReduction` (with optional `label_smoothing`). May use Focal Loss in some configs.

### ArcFace head (`use_arcface_head: true`)
- `ArcMarginProduct(in_features=512, out_features=2, s, m)` with margin `m` (commonly 0.15) and scale `s` (commonly 30).
- Optional scale annealing: `s_start → s_end` over `anneal_steps`. P8A and successors used `m=0.15` per RLP3.5 finding.
- Loss: standard `CrossEntropyLoss` (the margin is in the head; not Focal Loss).

### Optional regularizers / auxiliary losses

Several have been added across R13 iterations and are individually configurable:

- **`mixup_alpha`** — embedding-space mixup (interpolate features before the head). Default 0 (off).
- **`feat_norm_reg_lambda`** — feature-norm regularization, pulls real- and fake-feature-norm distributions toward each other. Default 0.
- **`anchor_aware`** — penalty on `anchor_pool_mean_prob > 0.10` weight 5.0 with 16 samples per step. Used in P13 / P14 bundles.
- **`pipeline_random`** — symmetric chain of jpeg/downscale/chroma_blur/yuv_roundtrip/gamma stochastic perturbations applied at training time.
- **`face_scale_jitter`** — `scale_limit: 0.25` or `0.50` symmetric crop variation.
- **Quality-domain DANN head (GRL)** — gradient-reversed adversarial head over `K` quality domains, `λ=0.20` static (P15) or 12-class method-conditional (P18).
- **`correlation_penalty`** — train-time Pearson penalty `λ * Σ_axis |Pearson_batch(score, axis)|` over a configured set of nuisance axes (sharpness_laplacian, luma_mean, face_area_fraction). Used in PD (in flight at the time of this writeup). λ=1.0 default.

These can compose. `anti_shortcut_bundle_decomposition` discipline (post-P14) requires a single-lever ablation slot whenever ≥ 2 are stacked, because P14's bundle was net-negative against the strongest single component.

## Model variants in this audit

Three checkpoints are compared in the diagnostic probes that the facts pack references. All operate on the same backbone (OpenCLIP ViT-B/16, 768 → 512 final projection) and the same 224x224 INTER_LINEAR + CLIP-normalize preprocessing.

| ckpt name | base | recipe-distinguishing features | role |
|---|---|---|---|
| **P8A_REFERENCE_STEP5000** | FT chain through R12g → unfrozen visual.proj + ln_post + apply_svd_to_mlp | broke camera-signature anchor ceiling (Δ=−0.188 vs RLP6_04). NOTE: trained PRE-fix of the in_proj-SVD bug, so its in_proj-SVD residuals received zero classification gradient. | the production-anchor candidate; appears robust to may6 cross-day drift |
| **E2B_TOP_N_STEP3200** | scratch on OpenCLIP-B/16 + Cross-Entropy + heavy aug | broke deeplive ceiling (87.5% recall@10% FPR vs P8A 42.9%); regressed viso recall on v2 substrate. **Confirmed today as the deployed production model** (Pearson r = 1.000 with deployment scores). | the currently-deployed model |
| **PA_TOP_N_STEP3800** | FT-from-E2B + visomaster_enhanced + visomaster_teams_enhanced data sources at family_weight=4.0 | broke v2-substrate viso F4 ceiling (72% recall) but doesn't generalize to HDTF (7.87% vs P8A 93.57%) | substrate-bound winner; deployment-deprecated |
| **PD (in flight)** | FT-from-E2B + correlation_penalty(λ=1.0) on 3 axes (sharpness, luma, face_area), single-lever discipline (anchor_aware/face_scale_jitter/arcface_head all OFF) | first explicit-form anti-shortcut loss class on R13; GPU scorecard running at the time of this pack writeup | verdict pending |

The probes in the facts pack measure pairwise differences between these models on common substrates; they are NOT evaluations of new architectures.

## Training data sources (loader = `combined_paired`)

Frame batching is paired-style (a `(real, fake)` pair sampled together for arcface stability). Sources are configured via `combined_paired:` in each yaml:

- `deeplive` family (clean + enhanced + teams variants)
- `visomaster` family (clean + enhanced + teams variants; subset of these is "v2 substrate" — Dor-dominant, used in PA)
- `df40` (DeepFake40 dataset)
- `proper_data` lanes — HDTF / quickclips schema with `include_lanes` control: `proper_real_{clean,teams}`, `proper_visomaster_{clean,enhanced_clean,teams,enhanced_teams}`. Added at RLP6+.
- Family weights via `family_weights.<family>_<label> = float` control sampling probability per source.

## Augmentation pipeline

Layered config under `combined_paired.augmentation:`. Available primitives:

- Standard: flip, rotate, blur, brightness, contrast, JPEG compression (quality_lower/upper).
- ShiftScaleRotate (spatial backbone): `context_variation_shift: 0.08`, `context_variation_individual_p: 0.40` is the operating default (RLP3+).
- `face_scale_jitter` — face-crop scale variation.
- `pipeline_random` — symmetric chain of jpeg/downscale/chroma_blur/yuv_roundtrip/gamma applied to BOTH real and fake at training time.
- `TeamsCodecSimulation` (`data/augmentations/teams_simulation.py`) — calibrated against measured Teams transport deltas (sharpness −50.9%, brightness +19.0%, HF energy −77.4%, etc.) on an 18-video × 132-frame sample. Independently re-verified 2026-05-04 on different 30 paired (raw, teams) viso frames: cosine 0.87-0.99 across 8 IQ axes, magnitude ratio 97-106%.
- `webcam_harden` — drafted but caused NaN/Inf loss in P11 launch; dropped.

## Promotion contract / deployment scoring infrastructure

`arena/run_target_domain_validation_sequential.py` is the standard scorer. Inputs: a checkpoint map (yaml mapping logical name → GCS path) + a suite manifest (yaml with per-suite frame paths and labels). Output: per-frame CSV (`frame_path, label, prob, status, ...`) per (ckpt × suite), plus aggregated scorecard CSVs.

The promotion contract defines a τ-selection policy that lexicographically optimizes a tuple of metrics under FPR constraints. The v3 fix (recall-floor) is in working tree but uncommitted (open loop `contract-policy-bug-fix-not-committed`).

## Inference-side preprocessing parity

A 2026-04-24 audit found `cv2.INTER_AREA` (deployment) vs `cv2.INTER_LINEAR` (training) caused silent kernel drift. Fixed via commit `855871e` (`batch_inference_gcs.py:407`, `arena/model_arena.py:472`). Deployment server (`http://34.16.217.28:8999`) was untouched at audit time — separate workstream open loop `deploy-server-preprocessing-drift`.

Local inference path used in today's probes:
1. `cv2.imread(path, cv2.IMREAD_COLOR)` (BGR)
2. `cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)`
3. `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`
4. `T.ToTensor()` → `T.Normalize(mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276])`
5. `model({"image": batch}, inference=True)` → returns dict with `prob` (= `softmax(logits)[:, 1]`) and `feat` (the 512-dim [CLS] pooled feature).

## Known architectural caveats relevant to interpretation

1. The 2-class softmax means `prob_fake = sigmoid-like` over the [0, 1] range; common operating thresholds are τ=0.5 (default), τ=0.92-0.99 (lockbox-calibrated), or per-substrate τ.
2. The arcface scale `s` is annealed during training; this means the same checkpoint's logit scale at inference depends on which step it was saved at (memory `project_promotion_contract.md` notes this as a deployment caveat).
3. `combined_paired` batches pairs (real, fake) for arcface margin stability; `family_weights` controls per-source sampling. Memory `project_visomaster_hints_lanes_bad_data.md` records that some lanes (visomaster_hints, visomaster_hints_teams) are bad data and should never be enabled — this is a known training pitfall.
4. The `quality_enhancement` family-routing bug (memory `project_quality_enhancement_routing_2026-05-05.md`): `DEFAULT_ENHANCED_STRATEGIES` wrongly included `quality_enhancement` (no GFPGAN, base strategy). Result: ~5,120 / ~18,880 deeplive_enhanced_fake training frames (~27%) mislabeled as enhanced when they were base. Fix landed 2026-05-05; PD is the first post-fix R13 packet.

## Items that are NOT in this architecture (worth flagging for an external advisor)

- No SBI (Self-Blended Images) pseudo-fake lane. Not currently part of the loader; would be a structural addition.
- No Fourier-domain augmentation (amplitude perturbation / phase preservation). Not currently implemented.
- No AugMix-style consistency loss with JSD between augmented views. Not currently implemented.
- No HSIC-based independence penalty. Only Pearson correlation in PD.
- No GroupDRO / V-REx (worst-group / cross-group risk minimization). Only adversarial GRL/DANN, which is a different geometry.
- No test-time augmentation (TTA) at inference. Single forward per frame.
