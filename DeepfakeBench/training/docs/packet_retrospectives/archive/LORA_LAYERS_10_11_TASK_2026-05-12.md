# LoRA-on-layers-10-11 task spec — 2026-05-12

> **Status**: self-contained operational task spec for a fresh agent. The agent picking up this file should be able to deliver the code, smoke-tests, and a launch-ready yaml without re-reading the full session context.
>
> **Authoring**: drafted 2026-05-12 by the planning agent after Stage A + 4 CPU FACTS docs landed. The user has authorized building this lever as a parallel GPU slot ("Slot 4") in the next R13 packet (target name: `R13_LORA_L10_L11_2026-05-13`).
>
> **The task is to write CODE + YAML + SMOKE TESTS — not to launch the Vertex job.** Launch is gated on D1-D5 CPU diagnostics + user authorization.

---

## 0. Read first (15 min)

1. [`docs/packet_retrospectives/AGENT_GUIDE.md`](AGENT_GUIDE.md) — the 6-rule contract (especially Rule 1 validate-before-suggest, Rule 3 CPU-first-then-GPU, Rule 4 viewer integration).
2. [`docs/packet_retrospectives/MODEL_GOALS.md`](MODEL_GOALS.md) — three pillars + **NO ENSEMBLE** hard rule + B16-only + the promotion bar. **Critical**: this task is for a single-model deployment candidate, not a research probe.
3. [`docs/packet_retrospectives/STATE.md`](STATE.md) §"Where we stand right now" — current best ckpts (P8A rank-1, T5C step3500 rank-3) and the binding constraint (dev→lockbox transfer gap).
4. Memory `project_stage2_all_levers_regress_p8a_2026-05-09` — names "LoRA / parameter-efficient adapter FT" as an explicitly untested lever class. This task closes that loop.
5. Memory `project_per_layer_divergence_2026-05-06` — P8A↔E2B encoder cosine collapses 0.998→0.32 across 12 resblocks; near-orthogonal at layers 10-11. Layers 0-9 are largely shared between the two best ckpts.

---

## 1. Theory of why this lever matters

The Stage 2 packet (2026-05-09) found that **every FT-from-P8A regime tested drifts the dor cluster within 500-2500 steps** (memory `project_stage2_all_levers_regress_p8a_2026-05-09`). The convergent reading was "FT itself is the binding constraint, not any particular single lever."

LoRA is the structurally distinct intervention against that reading:

- **Why LoRA**: standard FT updates the full encoder + head. Any FT regime is free to drift the encoder. LoRA freezes the base weights and trains a low-rank delta — by construction it CANNOT drift the bulk of the encoder.
- **Why layers 10-11**: per `project_per_layer_divergence_2026-05-06`, the behavioral difference between P8A and E2B (and by extension, between P8A and any FT'd descendant) is concentrated at layers 10-11. The shared structure is at layers 0-9. So LoRA at layers 10-11 trains the specific region where ckpt-behavior differentiates.
- **Falsifier interpretation**: if LoRA at layers 10-11 reproduces the same Stage-2 dor-cluster regression that full FT does, the binding constraint is the data / objective, NOT the FT regime. That's a load-bearing result either way.

The hypothesis to test, in one sentence:

> Training a low-rank delta on resblocks[10] and resblocks[11] of P8A (frozen elsewhere) + a fresh trainable head produces a ckpt with T5C-step3500-class catch AND P8A-class chronic-cohort invariance.

---

## 2. Existing infrastructure to compose with (do not reinvent)

| Element | Path | Use |
|---|---|---|
| EffortDetector | `detectors/effort_detector.py` | host model; B16 backbone via `openclip_b16` builder |
| Resblocks | `detectors/effort_detector.py` (search for `resblocks`) | the LoRA targets are the attention/MLP layers within `visual.transformer.resblocks[10]` and `[11]` |
| `SVDResidualLinear` / `SVDInProjLinear` | `detectors/effort_detector.py:1850, 2207` | precedent for parameter-efficient linear-layer wrapping. **DO NOT reuse SVD**; LoRA is structurally different. Use these as architectural reference for how to wrap a linear layer in a residual-trainable adapter. |
| Multi-axis GRL | `detectors/effort_detector.py:370` (`MultiAxisGRLBlock`) | OPTIONAL — can compose LoRA with GRL; default for this packet is **WITHOUT GRL** to isolate the LoRA mechanism. |
| `AnchorAwarePenalty` | `loss/anchor_aware_penalty.py:37` | available but **NOT used by default** — see §6 constraint. |
| Trainer | `trainer/trainer.py` | reads yaml config, runs train loop; LoRA needs to plug into the optimizer's param group list. |
| Wandb config allowlist | `train_sweep.py:176-282` | **CRITICAL** — any new nested yaml block (`lora:`) MUST be added to the allowlist per memory `project_wandb_flattens_nested_dicts.md`. Trainer reads None for un-allowlisted blocks. |
| Image rebuild | `./dev.sh build-prod -y` | auto-bumps VERSION patch per memory `reference_image_rebuild.md`. |
| Vertex launch | `arena/launch_*.sh` patterns | reference existing R13 launches; US regions only per `CLAUDE.md` (us-east1 / us-west4 / us-central1). |

---

## 3. Concrete deliverables

Deliver these in one commit (or a sequence of commits, no PR yet):

### 3.1 New module: `detectors/lora_adapter.py` (~250-350 LOC)

A self-contained module with:

- **`LoRALinear`** — wraps a `nn.Linear` (or a frozen reference to one). Forward: `frozen_weight @ x + (B @ A @ x) * (alpha / rank)`. `A` is `nn.Linear(in_features, rank, bias=False)` initialized N(0, 1/rank); `B` is `nn.Linear(rank, out_features, bias=False)` initialized to zero (so the delta starts at zero and the model behaves identically to the frozen base at step 0).
- **`apply_lora_to_resblock(resblock, rank, alpha, target_modules)`** — utility that wraps the named linear layers inside one CLIP `ResidualAttentionBlock`. Default `target_modules = ("attn.in_proj", "attn.out_proj", "mlp.c_fc", "mlp.c_proj")` — i.e., the 4 linear layers per resblock that carry weight.
- **`freeze_base_clip_encoder(model)`** — sets `requires_grad=False` on every encoder parameter EXCEPT the LoRA `A` / `B` matrices and the head. Should be called after LoRA injection.
- **Unit tests** in `tests/test_lora_adapter.py`:
  1. **Zero-init parity**: after `apply_lora_to_resblock`, forward output matches the frozen-base output to within 1e-5 for a random input. (LoRA delta = 0 at init.)
  2. **Gradient targeting**: after one backward pass, only `A.weight`, `B.weight`, and head parameters have non-zero gradients; every frozen-base parameter has `grad == None` or zero.
  3. **Rank scaling sanity**: `LoRALinear(rank=4)` has fewer trainable params than `LoRALinear(rank=32)`.

### 3.2 Trainer wiring

Edit `trainer/trainer.py` (or wherever `EffortDetector` is built) to:

1. Read a new `lora:` block from yaml: `{enabled: bool, target_layers: [10, 11], rank: 16, alpha: 32, target_modules: [...], freeze_base: true}`.
2. If `enabled: true`, call `apply_lora_to_resblock` on the listed layers and `freeze_base_clip_encoder` after model build.
3. Log at init: `"[LoRA] enabled at layers {target_layers} | rank={rank} alpha={alpha} | trainable params: X (of Y total = Z%)"`. **Critical** — this log line MUST fire, per the open loop `silent-feature-failures-pattern` (memory `project_wandb_flattens_nested_dicts.md`).
4. Optimizer: ensure only trainable params end up in the optimizer's param groups (the optimizer-build step should query `model.parameters()` filtered on `requires_grad=True`).

### 3.3 Wandb allowlist

Add `lora` block to `train_sweep.py` `_REAPPLY_KEYS` list (or whatever the nested-block-allowlist mechanism is called as of 2026-05-12). Failure to do this = silent fallthrough = the trainer reads `None` and LoRA never activates. Verify by inspecting the W&B config after a smoke run.

### 3.4 Yaml: `experiments/phase2_round13/R13_LORA_L10_L11_2026-05-13.yaml`

- Base config: copy `R13_T5C_T4_BIG_CLASSIFIER_2026-05-11.yaml` as template.
- `gcs_base_checkpoint`: P8A_step5000 (same as T5C).
- **Remove**: `multi_axis_grl` block (set `enabled: false`) — isolate the LoRA mechanism.
- **Remove**: `anchor_aware:` block (or set `enabled: false`).
- **Keep**: T3 keep-list data filter (it's axis-level and orthogonal).
- **Keep**: `apply_svd_to_in_proj`/`apply_svd_to_mlp` — DECISION: **set both to `false`**. LoRA is replacing the SVD-residual recipe at layers 10-11; SVD on other layers stays frozen via the base-freeze.
- **Add**:
  ```yaml
  lora:
    enabled: true
    target_layers: [10, 11]
    rank: 16          # tune later; 16 is a safe starting point per LoRA literature
    alpha: 32
    target_modules: ["attn.in_proj", "attn.out_proj", "mlp.c_fc", "mlp.c_proj"]
    freeze_base: true
  ```
- LR: lower than T5C's (LoRA usually trains 2-5× higher LR than full FT — but base-LR comparison depends on how the existing recipe schedules it). Pick `1e-4` for the LoRA params, keep `1e-5` for the head. If the trainer only supports a single LR, use `5e-5`.
- Total training steps: 5000 (match T5C).
- Seed: 9801 (NOT 9501/9601/9701 — those are taken by T5C/T6/T7).
- Periodic save block: same as T5C.
- Canary probe: enable per `R13_P2_SCRATCH_BUNDLE.yaml` template (uses `arena/canaries/teams_chronic_diverse_800_2026-05-07.parquet`).

### 3.5 Smoke tests

Two-tier smoke before any GPU launch:

**Tier 1 — Unit tests pass** (laptop CPU, ~1 min):
```bash
cd /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training
python -m pytest tests/test_lora_adapter.py -v
```
All 3 unit tests must pass.

**Tier 2 — End-to-end 10-step smoke** (laptop MPS, ~5 min):
- Construct a minimal `R13_LORA_L10_L11_SMOKE.yaml` with the same blocks but `total_training_steps: 10` + `nEpochs: 1` + tiny dataset slice.
- Run the trainer for 10 steps.
- Verify in stdout/W&B:
  1. `[LoRA] enabled at layers [10, 11]` log line FIRES.
  2. Trainable param count is `<5%` of total (LoRA at 2 layers × 4 modules × rank=16 should be a small fraction).
  3. Training loss decreases over 10 steps (basic sanity).
  4. NO NaN in losses.

### 3.6 Memory + thread updates

After landing the code, before launching:

1. New memory entry: `project_lora_l10_l11_packet_pending_2026-05-13.md` — status of this packet (pending launch authorization).
2. Update `docs/packet_retrospectives/STATE.md` "In flight / pending authorization" section to include this packet.
3. Add to `docs/packet_retrospectives/TIMELINE.md` when launched (deferred to launch-time).
4. After scorecard verdict: author `docs/packet_retrospectives/packets/LORA_L10_L11.md` per `packet_template.md`.

---

## 4. Falsifier / close criterion

After GPU launch + promotion-contract scorecard:

**This lever's hypothesis is REFUTED if**:
- lockbox AUC stays at the dev-trained-head β-outcome level (~0.76 per memory `project_t4_substrate_overfit_inv_mean_misleading_2026-05-11`), AND
- Stage-2-style dor-cluster regression reproduces (Roy_D real-FPR at FPR-cal τ > 70%).

Interpretation: parameter-efficient FT did NOT escape the substrate-distribution-mismatch. The binding constraint is the data + objective, not the FT regime. Next move pivots to data ingestion.

**This lever's hypothesis is SUPPORTED if**:
- `lockbox_real_fpr` at FPR-cal τ ≤ 0.025 (≤ T5C step3500's 0.0240 + sampling noise), AND
- `lockbox_fake_recall` at FPR-cal τ ≥ 0.50 (≥ ~T5C step3500 level), AND
- `dev_fake_macro_recall` ≥ 0.40, AND
- No per-identity FPR > 30% on the chronic-6 set at FPR-cal τ.

Interpretation: parameter-efficient FT preserves P8A's chronic-cohort behavior AND adds T5C-style catch. The full-FT-from-P8A regression is structurally avoidable.

**Partial outcome** (some but not all of the above): record the per-axis pass/fail and decide whether a hidden_dim / rank / alpha sweep is justified.

---

## 5. Constraints (per `MODEL_GOALS.md`)

- **Single model only — no ensemble.** LoRA produces a single forward-pass model; this constraint is satisfied by construction.
- **B16 backbone only**. Do NOT use L14.
- **NO P8A-output-distillation, NO P8A-L11-anchor in this packet.** Per parent-session discussion (2026-05-12), P8A-anchor levers risk propagating P8A's IQ-shortcut alignment. This packet specifically isolates the LoRA-vs-FT mechanism question without confounding it with anchor losses.
- **Forbidden-word discipline applies to FACTS docs only**, not to this task spec. The eventual `packets/LORA_L10_L11.md` retro must follow the factual-only contract (see `eval_folder_template.md`).

---

## 6. Open questions for the next agent (do not block on these)

1. **Rank choice**: 16 is the suggested starting point. If smoke shows trainable param count is way under 1%, consider rank=32 or rank=64.
2. **Alpha schedule**: standard practice is `alpha = 2 × rank`. Keep alpha=32 for rank=16.
3. **Whether to include resblock[9] in LoRA targets**: per `project_per_layer_divergence_2026-05-06`, layer 9 is borderline (cos ≈ 0.6 between P8A and E2B). DEFAULT IS NO — keeping it tight to 10-11. Adding 9 is a follow-up sweep, not first attempt.
4. **Whether to compose with T3 keep-list data filter**: DEFAULT YES (axis-level, orthogonal to LoRA mechanism). If smoke or early training shows the data filter interferes with LoRA convergence, drop it.

---

## 7. Risk register

- **Silent failure mode**: yaml block not in W&B allowlist → trainer reads `None` → LoRA never activates → model behaves like frozen-base + head-only (effectively the head-retrain β-outcome we already know). Mitigation: Tier 2 smoke MUST verify the `[LoRA] enabled` log line.
- **Optimizer captures all params**: if `optimizer.param_groups` is built before `freeze_base_clip_encoder` is called (or built from `model.parameters()` without filtering on `requires_grad`), the optimizer will update frozen params silently. Mitigation: unit test #2 catches this.
- **LoRA on attention `in_proj` vs `q/k/v` separately**: CLIP B16 uses a fused `in_proj_weight` in `nn.MultiheadAttention`. The `SVDInProjLinear` precedent (`detectors/effort_detector.py:2207`) decomposes q/k/v separately. LoRA can be applied to the fused weight (simpler) or to q/k/v separately (more granular). DEFAULT: apply to the fused `in_proj_weight` for simplicity; revisit if results are weak.
- **Don't break SVD recipe for other layers**: the SVD-residual decomposition is still active on layers 0-9 (frozen). Verify by checking `model.named_parameters()` after freeze — SVD residual parameters should be in the frozen set if `apply_svd_to_in_proj: false` in the LoRA packet.

---

## 8. Time/cost estimate

- Code + unit tests: ~6-10h dev (one agent-session, no GPU needed).
- Tier 1 + Tier 2 smoke: ~10 min on laptop after code lands.
- Image rebuild: ~17 min (`./dev.sh build-prod -y`).
- Vertex training: ~5-8h on A100 (LoRA is fast because few trainable params).
- Promotion-contract scorecard: ~9-10h on Vertex (use the existing combined-scorecard batch with 3 other slots; see parent session).

**Net new GPU spend for this slot**: ~$30-40 (training) + share of combined scorecard.

---

## 9. Handback to the planning agent

When the code + smoke tests land:

1. Send a one-line summary back: "LoRA module ready, smoke passes, yaml at `<path>`, awaiting D1-D5 outcomes and launch authorization."
2. The planning agent will then either authorize the launch (if D1-D5 don't refute the framing) or pivot the design (if they do).
3. **Do not launch** the Vertex job from this task. Launch is gated on user authorization.

If you encounter blockers (test failures, smoke-test detects silent-fallthrough, etc.), DO NOT mark complete — report the blocker so the planning agent can adjust.

---

## 9b. Resolved decisions (2026-05-12, implementation agent + user)

The following two architectural choices were underspecified in §3.4 / §7 and were resolved before code was written. Both were approved by the user.

### D1. LoRA stacks on top of frozen SVD (NOT a replacement)

§3.4's instruction "set `apply_svd_to_in_proj` / `apply_svd_to_mlp` to `false`" is self-contradicting given `gcs_base_checkpoint: P8A_step5000`: P8A's weights are stored as `weight_main` + SVD residuals per layer, and MHA forward is monkey-patched by `_install_svd_in_proj_routing` (`detectors/effort_detector.py:2328`). With SVD off, the checkpoint won't load and the routing patch isn't installed.

**Resolution**: KEEP `apply_svd_to_in_proj: true` and `apply_svd_to_mlp: true` (matching P8A_step5000's training config). After SVD wrapping, the new `freeze_base_clip_encoder` helper sets `requires_grad=False` on every SVD residual parameter (`U_residual`, `S_residual`, `V_residual`) at all 12 layers, freezing them as part of the "base". LoRA A/B matrices at layers 10-11 are the only trainable encoder parameters.

Trainable set after build:
- `lora_A` / `lora_B` weight matrices at resblocks 10 and 11 (in_proj fused + out_proj + mlp.c_fc + mlp.c_proj)
- Head parameters (whatever the existing config trains: ArcFace / Linear)

Frozen set: everything else, including all SVD residuals at layers 0-11 and `visual.proj` / `visual.ln_post` (those unfreeze flags are turned OFF in this packet).

### D2. New `lora` param group in `choose_optimizer`

Per standard LoRA practice, A/B matrices should not receive weight decay. Add a 4th param group to `choose_optimizer` (`utils/setup.py:45`) alongside `svd_residual`, `backbone_native`, `other`:

```python
lora_param_names = ('lora_A', 'lora_B')
# ...
{'params': lora_params, 'weight_decay': 0.0, 'lr': base_lr * lora_lr_mult, 'name': 'lora'}
```

New yaml knob: `optimizer.adam.lora_lr_mult` (default `1.0` for bit-identical fallthrough when LoRA is off).

The R13_LORA yaml will set `lora_lr_mult: 10.0` so LoRA trains at `1e-4` while the head stays at `1e-5` (i.e., base `learning_rate: 1.0e-5` × `lora_lr_mult: 10.0` = `1e-4` for LoRA params). Numbers per §3.4.

### D3. LoRA-on-fused-in_proj routing pattern

Mirror `_install_svd_in_proj_routing` exactly. New function `_install_lora_in_proj_routing(mha_module)` overrides `mha.forward` to compute `in_proj_weight = base + (B @ A) * scaling`, where `base` is `mha._svd_in_proj.weight` if SVD is active on that layer, else `mha.in_proj_weight` (the raw frozen leaf). The LoRA A/B parameters live on the MHA module under `mha._lora_in_proj.lora_A` / `mha._lora_in_proj.lora_B`.

This composes cleanly: the SVD routing already replaced `forward`; the LoRA installer further wraps the SVD-routed weight by adding the LoRA delta. Both routings remain inspectable via marker attributes (`_svd_in_proj_routing_active`, `_lora_in_proj_routing_active`).

### D4. LoRA-on-non-MHA-linear pattern

For `attn.out_proj` and `mlp.c_fc` / `mlp.c_proj` (each currently an `SVDResidualLinear` when SVD is enabled), wrap with a `LoRALinear` adapter:

```python
class LoRALinear(nn.Module):
    """Holds reference to a frozen base layer; adds LoRA delta on top."""
    def forward(self, x):
        return self.base_layer(x) + self.lora_B(self.lora_A(x)) * self.scaling

    @property
    def weight(self):  # for F.multi_head_attention_forward compatibility (out_proj path)
        return self.base_layer.weight + (self.lora_B.weight @ self.lora_A.weight) * self.scaling

    @property
    def bias(self):
        return self.base_layer.bias
```

After wrapping, replace `module.out_proj = lora_wrapped` etc. The `.weight` and `.bias` properties keep `F.multi_head_attention_forward` happy.

---

## 10. Self-correction protocol

Per `AGENT_GUIDE.md` Rule 5: if your investigation surfaces evidence that contradicts §1 (Theory) or §4 (Falsifier), STOP and report. Do not silently re-scope the task. Examples of contradiction:

- LoRA at layers 10-11 fundamentally can't represent the changes T5C made (e.g., T5C's hidden_dim 1024 lever isn't expressible as a layer-10/11 delta) — report so the planning agent revises the scope to include the GRL/classifier block too.
- The existing `apply_svd_to_in_proj` infra makes LoRA conflict architecturally (e.g., SVDInProjLinear can't be combined with LoRALinear on the same fused weight) — report so the design is revised to either drop SVD residual at the LoRA layers OR stack the two.

These are the kinds of "the premise is wrong" findings that the parent agent NEEDS to know before authorizing GPU spend.
