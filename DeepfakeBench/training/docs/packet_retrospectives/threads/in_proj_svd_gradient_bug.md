# Thread: in_proj-SVD silent zero-gradient bug

## The question

The Effort detector's `apply_svd_to_in_proj` flag was meant to install learnable SVD residuals onto the fused `q/k/v` input projection of each transformer block's `nn.MultiheadAttention`. From the day this lever landed up to 2026-04-26, the residuals received zero classification gradient — every R12g / RLP1..7 / P8A / P9 / P10 run that flipped `apply_svd_to_in_proj: true` trained those parameters on regularizer gradient only (when `lambda_reg > 0`) and on nothing at all from the task loss. The question this thread answers: how many prior conclusions depend on `in_proj-SVD` being a real lever, and what is the post-fix re-attribution sweep?

## Initial belief

Through [RLP3](../packets/RLP3.md) → [RLP6](../packets/RLP6.md) → [RLP7](../packets/RLP7.md) → P8A, the team treated `apply_svd_to_in_proj` as a working lever: it appeared in every R12g/RLP/P-* yaml as `true`, contributed to the "trainable parameter count" in the README and handoffs, and was implicitly part of the "FT-from-RLP6_04 ceiling" narrative. The P7 anchor ceiling at ~−0.10 (`project_shortcut_is_upstream.md`) was attributed to `head + attention-only SVD reach + frozen visual.proj/ln_post + frozen MLP` — i.e., reading the in_proj-SVD residual capacity as load-bearing. P8A's headline single-variable claim ("unfreeze visual.proj + ln_post + apply_svd_to_mlp") was reported as adding *three* new levers on top of an already-trained in_proj-SVD baseline.

## What changed our mind

- **2026-04-26 static review of `_install_svd_in_proj_routing`** (`detectors/effort_detector.py:1710` per memory `project_in_proj_svd_gradient_bug.md`). The original pattern installed a `forward_pre_hook` that did `module.in_proj_weight.data.copy_(module._svd_in_proj.weight)` before each forward. `in_proj_weight` is a leaf `nn.Parameter` with `requires_grad=False` (intentionally frozen for the patched MHA). `.data.copy_` writes values without participating in autograd, so `F.multi_head_attention_forward` read a frozen leaf and gradients never propagated back to the SVD residual parameters.
- **2026-04-26 one-batch grad audit** (`analysis/probe_battery_2026-04-26/grad_audit.py`, committed in `7af72b1`). On the OLD pre-hook approach: 9/9 `svd_{q,k,v}.{U,V,S}_residual` parameters had `grad=None` or `grad.abs().sum() == 0` after a single classification-loss backward. On the fix: 9/9 in_proj residuals + 3/3 `SVDResidualLinear` residuals (out_proj path, never broken) receive non-zero gradient. PASS verdict locked into the audit script as a regression guard.
- **2026-04-26 fix lands as commit `2feea58`** ("Fix silent zero-gradient bug in apply_svd_to_in_proj path"). Routes MHA forward through `self._svd_in_proj.weight` (autograd-tracked property) instead of the frozen leaf `self.in_proj_weight`. The leaf is preserved on the module for checkpoint backward-compatibility but is no longer read at inference. `compute_orthogonal_loss` and `compute_keepsv_loss` continue to produce real gradient (they always did — when `lambda_reg > 0` the residuals drifted toward orthogonality, just learning nothing about the task).
- **2026-04-26 anchor-pool monitor + verification scaffolding** (commit `7af72b1`, `trainer/trainer.py` + `trainer/mixins/checkpointing.py`). Adds an in-trainer anchor-spread monitor that logs `anchor/composite` every step-validation cycle and a checkpoint-time gate. Designed to catch any future "broken lever" pattern at runtime by comparing anchor metrics against an expected curve; would have surfaced this bug if it had existed in 2026-04-22.
- **2026-04-26 LOG entry: phase-0-execution** (`april-26-training-master-plan-v2.LOG.md:88-123`). The bug fix and eval infrastructure had to be **split** at commit time because the working tree contained an unrelated older `feat_norm-reg` change (~38 lines, packet-6B 2026-04-23) interleaved with the in_proj fix (~130 lines). The split-via-patch-extract-and-apply pattern is documented at `:91`. Image rebuild produced `1.3.216` (`ac83ba1`); local probe-battery `grad_audit.py` confirmed 9/9 PASS pre-launch (`:136`), but in-image verification was deferred as paranoia-grade per user time pressure.
- **2026-04-26 retroactive correction of P8A attribution** (memory `project_p8a_breakthrough.md:24-31`, dated 2026-04-26). The "Why P8A worked" reasoning was rewritten with a CORRECTION block: "P8A's anchor breakthrough therefore came from MLP-SVD + unfrozen `visual.proj` + unfrozen `ln_post` ALONE; the in_proj-SVD piece was a no-op. The 'more reach' qualitative explanation still holds, but the lever set was smaller than reported here."
- **2026-04-26 retroactive correction of upstream-shortcut reasoning** (memory `project_shortcut_is_upstream.md:33-40`, dated 2026-04-26). The "FT-only is structurally bounded" claim was annotated with a CAVEAT: it was measured *with broken in_proj-SVD*. Every P7 run had `apply_svd_to_in_proj=true` but the q/k/v residuals received zero classification gradient. After the fix, the FT lever set is meaningfully larger (~3M trainable params per attention block × 12 blocks that weren't training). The shortcut-is-upstream evidence (RLP7_08 fork-point control) is unaffected; the "ceiling" claim itself needs re-validation post-fix.
- **2026-04-26 Phase C overnight slate launched to disentangle the ceiling re-attribution** (`april-26-training-master-plan-v2.LOG.md:206-273`). Five Vertex training jobs in us-west4 on image 1.3.218: C.1 canonical re-FT on P8A (image 1.3.216), C-ablation `apply_svd_to_in_proj: false` (commit `6665910`), C.3 codec hedge, C.4a one-step-back from RLP6_04 step 23500, C.4b two-effects-back from RLP6_04 step 8000. Designed so the morning readout produces a 5-corner read: C.1 vs C-ablation tells "is the in_proj fix the lever or co-passenger?"; C.3 vs C.1 tells "is the data axis the missing piece?"; C.4a vs C.1 tells "is P8A's specific unfreeze recipe necessary?"; C.4b vs C.4a tells "does FT-depth crystallization matter?"
- **2026-04-26 overnight readout: C.1 vs C-ablation tied** (`april-26-training-master-plan-v2.LOG.md:318-343`). Both runs completed; W&B summaries show no material lift from the in_proj fix in the canonical P10_SYM-on-P8A configuration. Surfaced verbatim: "in_proj-SVD bug fix produced no material lift in this configuration. Recommend de-prioritizing in_proj-SVD lever experiments until a recipe is built that actually exercises the unlocked residual capacity." The 5-corner read also concluded plain-CLIP-scratch (~48h, large $) is **not** warranted from this slate alone (C.4b vs C.4a tied within noise; Δ=0.0007 on best_anchor/composite).

## Current stance (2026-04-26)

The bug is fixed in source (`2feea58`) and verified in image `1.3.216` via local `grad_audit.py` PASS. The fix changes the parameter set that an `apply_svd_to_in_proj: true` run can train, but the post-fix Phase C overnight slate revealed that **the unlocked residual capacity is not being exercised by the canonical P10_SYM-on-P8A recipe** — C.1 (with fix) and C-ablation (`apply_svd_to_in_proj: false`) tied within noise. Two interpretations remain compatible with that evidence:

1. The in_proj-SVD lever is real but small relative to the MLP-SVD + visual.proj + ln_post unfreeze package P8A already exercises; further engagement requires a recipe that targets it (higher SVD rank, regularizer that uses the unlocked capacity, or a different fork point).
2. The lever is real but its useful direction is misaligned with the current data mix; "in_proj-SVD wants to learn" but "the supervision signal does not push it there in 10k steps from a checkpoint that was already pulled toward an in_proj-frozen optimum."

Either way, the retroactive correction is durable: every pre-2026-04-26 attribution that **counted** in_proj-SVD trainability as part of the lever set is now imprecise. The headline conclusions of P8A, P7, and the upstream-shortcut reasoning still hold qualitatively (P8A still cuts anchor FPR ~half vs RLP6_04; P7 still hits a ceiling at ~−0.10; the shortcut is still upstream of RLP6_04). What shifts is the *parameter accounting* and the question of "what is the maximum FT can reach with all levers actually on."

## Packet timeline

- *(pre-Slice 5)* — every R12g / RLP1..7 / P8A run with `apply_svd_to_in_proj: true` silently trained the q/k/v residuals on zero classification gradient. No packet-level evidence ever surfaced this; the silent-fallback class is invisible without a grad audit.
- [P8A retro](../packets/P8A.md) — first packet whose anchor headline is retroactively re-attributed: the breakthrough is from MLP-SVD + visual.proj + ln_post unfreeze, not from "more reach via in_proj residuals."
- [P9 retro](../packets/P9.md) — every P9 single-variable variant inherits the in_proj-SVD silent-bug confound. The "softened P8A" framing is fine, but the per-variant attribution of which lever moved which metric pre-dates the fix.
- [P10 retro](../packets/P10.md) — P10 yamls (commit `2c9778b`) were authored before the fix; the C-ablation slot in the Phase C overnight slate is the first packet to deliberately ablate `apply_svd_to_in_proj` against a fixed-code baseline. C-ablation tied with C.1 in the overnight readout — the fix is now operative but the lever is not being exercised by the canonical recipe.

## Evidence locations

- `detectors/effort_detector.py::_install_svd_in_proj_routing` (around line 1710 per memory) — the routing fix.
- `analysis/probe_battery_2026-04-26/grad_audit.py` — the regression-guard audit; PASS = 9/9 in_proj residuals + 3/3 `SVDResidualLinear` residuals receive non-zero gradient.
- `trainer/trainer.py` (anchor-pool monitor) + `trainer/mixins/checkpointing.py` (state init) — runtime guard against silent-bug recurrences (commit `7af72b1`).
- Commits: `2feea58` (the fix, in_proj-only via patch-extract-and-apply); `7af72b1` (eval infra: anchor monitor + grad_audit + manifest_overlap); `ac83ba1` (VERSION bump 1.3.216 = first image with the fix).
- `april-26-training-master-plan-v2.LOG.md:44-122` — phase-0-execution: discovery + split + commit + image build.
- `april-26-training-master-plan-v2.LOG.md:206-273` — phase-c-overnight-slate: C.1 vs C-ablation comparison plan.
- `april-26-training-master-plan-v2.LOG.md:318-343` — overnight-monitor-and-slate-readout: C.1 vs C-ablation tied within noise.
- Memory: `project_in_proj_svd_gradient_bug.md` (the bug doc with audit verification + how-to-apply); `project_p8a_breakthrough.md:24-31` (CORRECTION block); `project_shortcut_is_upstream.md:33-40` (CAVEAT block).
- Handoffs: `docs/relaunch_handoffs/PACKET_9_MID_FLIGHT_HANDOFF_2026-04-26.md` — context document; the bug fix was already in flight when this was authored, but the handoff focuses on P9 scoring rather than re-attribution.

## Open loops

### Open loop: apply-svd-in-proj-attribution-revision-needed
status: open
severity: high
first_seen: 2026-04-26
last_verified: 2026-04-29
close_criterion: every prior R12g / RLP / P-* run with `apply_svd_to_in_proj: true` is either (a) re-trained from a matched fork point with the fixed routing and the deltas vs broken-bug counterpart documented in a per-packet table, or (b) explicitly annotated in its packet retro with "in_proj-SVD trained on zero classification gradient — q/k/v residuals contributed only via regularizer drift." Until one of these is done the published attribution claims (P8A breakthrough = "more reach", P7 ceiling = "FT-only structurally bounded with all levers on") are imprecise.

The Phase C overnight slate (`april-26-training-master-plan-v2.LOG.md:206-273`) was the first attempt at the (a) branch — C.1 (with fix) vs C-ablation (`apply_svd_to_in_proj: false`) on the P10_SYM-on-P8A recipe. The result: tied within noise. That answers one specific configuration but does not close the loop for the broader retroactive sweep — the in_proj lever might be larger from a different fork point (R12g earliest, RLP6_04 step 8000) or under a different recipe (higher SVD rank, regularizer that exercises the residual capacity). The recommendation in the overnight readout was to **de-prioritize** in_proj-SVD lever experiments until a recipe is built that actually exercises the unlocked capacity, but the published P8A / P7 / upstream-shortcut attributions remain imprecise in their original form. Memory `project_p8a_breakthrough.md` and `project_shortcut_is_upstream.md` carry CORRECTION/CAVEAT blocks dated 2026-04-26 — that's the (b) branch for those two specific narratives. The full per-packet sweep (every R12/RLP/P-* run cited in any retro) has not been done.

### Open loop: in-proj-svd-residual-capacity-not-exercised
status: open
severity: medium
first_seen: 2026-04-26
last_verified: 2026-04-29
close_criterion: a recipe (yaml or code change) is identified that produces a measurable lift on a deployment-relevant metric (anchor pool composite, lockbox real-FPR, or per-method recall) when `apply_svd_to_in_proj: true` is enabled vs disabled at matched fork point + LR + schedule + data — i.e., the lever is shown to be live in some configuration, not just live in principle.

The Phase C overnight slate's C.1 vs C-ablation comparison showed the in_proj fix produced no material lift in the canonical P10_SYM-on-P8A configuration. Two hypotheses remain compatible with that evidence: (1) the lever is small relative to the MLP-SVD + visual.proj + ln_post unfreeze package P8A already exercises, so it would need a higher SVD rank or a regularizer to surface; (2) the supervision signal does not push the in_proj residuals in a useful direction within 10k steps from an in_proj-frozen-pretrained checkpoint. Either resolution closes the loop. A proactive next step would be a small ablation pack: e.g., (i) `apply_svd_to_in_proj: true` with `svd_rank_in_proj: 1024` (vs the default 736), (ii) `apply_svd_to_in_proj: true` with a small `in_proj_orthogonal_lambda` bump to exercise the unlocked capacity, (iii) `apply_svd_to_in_proj: true` from RLP6_04 step 8000 (less crystallized fork point, more room for in_proj to learn). None of these have been launched.

### Cross-thread refs

- [`processing_signature_shortcut`](processing_signature_shortcut.md) — the upstream-shortcut reasoning relied partly on in_proj-SVD being a real lever; the CAVEAT block in `project_shortcut_is_upstream.md` is dated 2026-04-26.
- [`promotion_contract_evolution`](promotion_contract_evolution.md) — every prior `value_composite` claim that depended on "all SVD residuals are training" inherits the in_proj-SVD imprecision; relevant only for parameter-accounting claims (composite numbers themselves are still measured against actual checkpoints).
- [`wandb_yaml_propagation_bugs`](wandb_yaml_propagation_bugs.md) — different bug class (yaml flattening vs autograd graph break) but same failure mode ("the lever was silently a no-op; only an explicit audit caught it"). Both belong to the broader "silent fallback patterns" hygiene story.
