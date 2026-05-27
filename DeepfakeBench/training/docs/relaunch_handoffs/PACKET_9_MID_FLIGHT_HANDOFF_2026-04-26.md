# Packet-9 Mid-Flight Handoff — 2026-04-26

## ⚠️ READ THIS FIRST — Why this handoff is unusual

**The user (Roee) has new findings that may invalidate the entire direction of Packet-9.**

He hasn't shown them to you yet. He's going to. Your job during the early part of the conversation is to **listen, integrate, and be willing to stop or pivot** — not to push forward with the P9 plan you'll read about below.

The P9 plan in the rest of this document is the *current working hypothesis*. Treat it as load-bearing context for understanding what you're looking at, not as a directive to execute. If Roee's new evidence contradicts the P8A premise, the right move may be to halt the running US relaunches, not to wait for the scorecard.

**The agent before you (me) was confident the P9 slate was the right move based on P8A's lockbox-FPR breakthrough. Roee is less confident. He may be right.**

So: read everything below for context. Then **wait for Roee to share what he's found before recommending any next action.**

---

## Where the experiments are right now (08:50 UTC, 2026-04-26)

**RUNNING (do not touch unless Roee redirects):**
- 4 US training relaunches in us-east1 (P9_01, P9_03, P9_AB_stack) and us-west4 (P9_FORK_AB_stack on r12g) — started 07:34 UTC, ~3.4 it/s, ETA ~5-6h to completion
- 1 US training PENDING in us-west4 (P9_FORK_p8a_on_r12g) — quota wait
- 1 europe-west4 training (P9_LONG_extended) — still RUNNING from yesterday
- **P9 promotion-contract scorecard** in us-east1 (job `2877145196556976128`) — started 07:46 UTC, ETA ~10:45 UTC. **This is the most important running job — it tells us whether any of the 4 finished P9 candidates beats P8A and RLP6_04 on lockbox.**

**FINISHED P9 ckpts awaiting scorecard verdict:**
- P9_05 (run `5oc7zyzw`, step 9000) — real-side codec uplift
- P9_FREEZE (run `ku6fdljz`, step 9000) — backbone_lr_mult 0.0
- P9_R (run `a5w629y1`, step 3500) — P8A reseed
- P9_FORK_DATA (run `6pd7o8iz`, step 5500) — fork experiment

**CANCELLED (don't worry about these):**
- All 5 asia-southeast1 P9 originals (cancelled 08:45 UTC after US replicas overtook them within 1h despite starting 10h later — GCS region locality issue, US-multi-region buckets are 10x slower from asia compute)

**Pending wakeup:** ScheduleWakeup at 09:48 UTC for scorecard verdict.

---

## The detection problem (don't skip this)

**Product**: Effort deepfake detector for Microsoft Teams deployment. Architecture is OpenCLIP ViT-B/16-DataComp-XL backbone + ArcFace head + SVD-rank-736 on backbone.

**Three success pillars** — all load-bearing, not just the first:
1. Fake recall on target methods (deeplive_enhanced, visomaster_enhanced_macro, etc.)
2. Real-pool FPR < 5% on Teams-camera-style data (5% target, 7% hard cap)
3. Robustness across lighting / camera / codec / color

**The crowning metric** is the **Teams Promotion Contract** scorecard, which:
- Calibrates τ on dev set lexicographically (lockbox_real_fpr ascending, then lockbox_fake_recall descending)
- Then reads out lockbox metrics at that τ
- Trainer's `value_composite` is **NOT** deployment-grade — multiple times we've crowned the wrong checkpoint by trusting it

**Known contract-policy bug**: the dev-calibration sometimes drives τ to ~0.99, crushing fake recall to ~23%. Always check `selected_threshold` and the τ value before trusting a scorecard. (memory: `project_contract_policy_bug.md`)

---

## The narrative arc you need to understand

### Discovery: the camera-signature shortcut (pre-Packet-7)

Detector achieves strong AUC on academic test sets but flips its prediction based on **camera/ISP signature** rather than face content. Same person (Dor or Roee), same lighting, same face, different camera → different output. Cross-subject confirmed. This is a **shortcut**, not a real signal.

The shortcut lives **upstream of RLP6_04** — i.e., it was inherited from the pretraining and isn't something later finetuning can remove with FT-only interventions. RLP7_08 ruled out the "late-RLP6_04 consolidation" hypothesis: forking from any RLP6_04 step with FT-only hits a ~0.84-0.89 anchor-pool ceiling. (memory: `project_shortcut_is_upstream.md`)

**Implication**: to actually fix the shortcut, we need to perturb the backbone, not just the head.

### Packet 7 (RLP7_*): augmentation-based attempts

Tried spatial-only and codec-only augmentation interventions to make the model camera-signature invariant:
- RLP7_02: heavy codec augmentation
- RLP7_04: spatial-only
- RLP7_05: spatial + moderate codec (best of P7)

**P7 outcome**: Partial gains on anchor pool. RLP7_05 at lockbox_real_fpr 0.514% / fake_recall 23.7% — slightly worse than RLP6_04 (0.441% / 23.3%) on FPR, similar on recall. Augmentation alone wasn't moving the needle enough. The shortcut survived because **the backbone itself encodes camera signatures** and we weren't training the backbone.

### Packet 8: unfreezing the backbone (the breakthrough)

Hypothesis: if the shortcut is upstream and the backbone is frozen, FT can't remove it. Unfreeze the backbone surgically.

**P8A** (`R13_RLP8_01_unfreeze_clip_codec.yaml`) introduced three flags simultaneously:
- `unfreeze_final_proj: true` — visual.proj layer
- `unfreeze_final_ln: true` — ln_post
- `apply_svd_to_mlp: true` — MLP layers in last 2 transformer blocks (c_fc, c_proj)

**P8A result** (run `9lmvb5b4`, step 5000):
- **lockbox_real_fpr 0.147%** — **half of RLP6_04's 0.441%**. This was the breakthrough — the camera-signature ceiling cracked.
- lockbox_fake_recall 23.7% — same as RLP6_04 (23.3%) at the macro level.

**BUT**: per-method scorecard analysis showed **329 fake-method regressions with 0 compensating wins**. Hits concentrated on codec-processed fake families:
- `visomaster_enhanced_macro`: 1.1% (vs RLP6_04 3.3%) — 2.2pt loss
- `deeplive_enhanced`: 2.4% (vs RLP6_04 6.4%) — 4pt loss

Only ~19% of those regressions were τ-recoverable (separability loss, not threshold drift).

So P8A traded **deployment FPR (good)** for **per-method fake recall (bad)** at exactly the methods we care about most.

### Packet 9: soften P8A and fix the model-selection failure mode

P9 is built on the assumption that **P8A is the right base, but trained too aggressively**. Three hypotheses for why P8A regressed on fake recall:
1. **Magnitude** — backbone trained too hot
2. **Topology** — `apply_svd_to_mlp` reaches too deep
3. **Data** — real/fake codec exposure asymmetry

P9 is single-variable experiments off the P8A baseline to disentangle these.

**This is where Roee's new findings might come in.** If the P8A premise is wrong — e.g., the FPR win is a measurement artifact, or the per-method regression is fundamental and not softening-recoverable, or there's a different shortcut hiding in P8A — then P9 is solving the wrong problem.

---

## The P9 experiment slate (10 experiments total)

### Original 5 (single-variable from P8A, 2026-04-25)

| Run | Delta from P8A | What it tests |
|-----|----------------|---------------|
| **P9_01** | `optimizer.adam.backbone_lr_mult: 0.3` | Magnitude softening — is regression from training too hot? |
| **P9_03** | `apply_svd_to_mlp: false` (keep proj+ln) | Topology — is regression from MLP-SVD reaching too deep? |
| **P9_05** | `augmentation.real_codec_uplift: true` (real-side codec aug) | Data — does real/fake codec symmetry recover separability? |
| **P9_R** | identical to P8A but seed 737 → 938 | Replication — is P8A's headline reproducible at all? (N=1 currently) |
| **P9_freeze** | `backbone_lr_mult: 0.0` | Disentangle magnitude vs topology — what does MLP-SVD buy with native unfreeze frozen? |

### Longshot pack (5 more, added 2026-04-25 evening)

These were launched after Roee asked for additional candidates exploring different fork points:
- **P9_FORK_p8a_on_r12g** — P8A recipe forked from RLP6_12g (different base ckpt)
- **P9_FORK_AB_stack_on_r12g** — A+B stack from RLP6_12g
- **P9_AB_stack** — combined A+B intervention from P8A base
- **P9_FORK_DATA** — fork with data axis variant
- **P9_LONG_extended** — extended-schedule variant

Each P9 candidate gets the same lockbox-anchored crowning protocol. Only the original 5 had clean single-variable framing; the longshot pack was deliberately exploratory.

---

## Crowning protocol (pre-committed before launch)

A P9 candidate is **promotable** only if all three hold on lockbox:

1. `lockbox_real_fpr ≤ RLP6_04.lockbox_real_fpr` (i.e., ≤ 0.441%)
2. `lockbox_fake_recall ≥ max(0.80, RLP6_04 - 0.02)` — i.e., ≥21.3% macro
3. Per-method recall on `deeplive_enhanced` ≥ 3.4% (RLP6_04 6.4% - 3pt slack), `visomaster_enhanced_macro` ≥ 0.3% (RLP6_04 3.3% - 3pt slack)

**Hard cap**: 7% lockbox FPR. Above that, never crown regardless of recall.

**Crown via** `arena/score_teams_promotion_contract.py:430-479` (lexicographic sort), **NOT** by trainer's `value_composite`. Trainer's value_composite has crowned wrong winners in past packets. (memory: `project_promotion_contract.md`)

---

## When the scorecard lands (~10:45 UTC)

1. Read `gs://training-job-outputs/test_results/teams_promotion_contract/p9-review-scorecard-20260426/promotion_contract/checkpoint_summary.csv`
2. Compare each P9 candidate against:
   - **P8A baseline** (lockbox_real_fpr 0.147%, fake_recall 23.7%, viso_macro 1.1%, deeplive_enh 2.4%, τ=0.991)
   - **RLP6_04 baseline** (lockbox_real_fpr 0.441%, fake_recall 23.3%, viso_macro 3.3%, deeplive_enh 6.4%, τ=0.992)
3. Apply crowning protocol — does any P9 candidate satisfy all three conditions?
4. **Critical: check `selected_threshold` for each row.** If τ is ≥0.99, the contract-policy bug may be inflating real_fpr metrics. (memory: `project_contract_policy_bug.md`)

**If a P9 candidate looks like a clean winner**: do not promote yet. Roee's pending findings may change the analysis.

**If no P9 candidate beats RLP6_04**: the P8A "soften" thesis is questionable. The P9 slate may have been the wrong investigation.

---

## Files to know

**P9 yamls** (in `experiments/phase2_round13/`):
- `R13_P9_01_softened_p8a.yaml`
- `R13_P9_03_no_mlp_svd.yaml`
- `R13_P9_05_real_codec_uplift.yaml`
- `R13_P9_R_p8a_reseed.yaml`
- `R13_P9_freeze_native.yaml`
- (+5 longshot yamls)

**P9 scorecard map**: `arena/checkpoint_maps/teams_target_domain.p9_review_2026-04-26.yaml`

**Key code** (don't modify without thinking):
- `data/augmentations/pipelines.py` — has the `real_codec_uplift` flag wired in (P9_05)
- `detectors/effort_detector.py:736,744` — backbone unfreeze flags
- `utils/setup.py` — `backbone_lr_mult` param-group wiring
- `arena/score_teams_promotion_contract.py:430-479` — crowning logic
- `arena/launch_teams_promotion_contract.sh` — scorecard launcher (must export WANDB_API_KEY/ENTITY/PROJECT, must override `REGION=us-east1` env var or it defaults to asia)

**Prior handoffs / context docs:**
- `docs/relaunch_handoffs/NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF_2026-04-19.md`
- `docs/relaunch_handoffs/RELAUNCH_UPGRADE_REVIEW_PACKET_2026-04-19.md`
- `docs/relaunch_handoffs/WT_B_AND_NEW_DATA_READINESS_2026-04-19.md`

**Plan file** (the current Packet-9 plan, mid-execution):
- `~/.claude/plans/warm-baking-cat.md`

**Memory** (read these before recommending anything):
- `project_promotion_contract.md` — contract is authoritative, value_composite is not
- `project_contract_policy_bug.md` — τ inflation issue, check before trusting scorecard
- `project_p8a_breakthrough.md` — what P8A did and why
- `project_shortcut_is_upstream.md` — why we unfroze the backbone
- `project_signature_shortcut_finding.md` — the camera-signature shortcut details
- `project_success_criteria.md` — three pillars, not one
- `project_gcs_region_locality.md` — US-only training, asia is 10x slower (just filed)
- `feedback_no_cancelling_vertex_jobs.md` — destructive class, ask first
- `feedback_decision_points.md` — present recommendation + tradeoffs, wait for explicit pick

---

## Resume instructions

1. **Greet Roee and ask what he's found.** He has potential evidence that could flip the direction. Don't preempt with the P9 plan — listen first.
2. Once you understand the new evidence, integrate it against the narrative above:
   - Does it invalidate the camera-signature-shortcut hypothesis?
   - Does it invalidate P8A's headline (the lockbox_real_fpr 0.147% number)?
   - Does it invalidate the "soften P8A" framing of P9?
   - Does it suggest a different intervention axis we haven't tried?
3. Decide jointly whether to:
   - Continue waiting for the scorecard verdict (~10:45 UTC) and act on P9 results
   - **Halt the running US relaunches** and pivot to Packet-10 with a different premise
   - Something else entirely
4. If continuing: when the scorecard lands, apply the crowning protocol with eyes wide open for τ inflation and per-method regressions, and report findings in a side-by-side table including τ.

---

## Failed approaches / warnings

- **Don't crown by `value_composite`.** Multiple past packets crowned wrong checkpoints this way. Always use the lockbox-anchored contract scorecard.
- **Don't trust scorecard rows without checking τ.** The contract-policy bug drives τ to ~0.99 sometimes, which makes everything look low-FPR but kills fake recall.
- **Don't run training in asia-southeast1 or europe-west4** against the US-multi-region buckets. 10x slower than US compute. Cancelled 5 jobs this morning over this.
- **Don't cancel Vertex training jobs without explicit Roee authorization.** Even hung jobs. Even obviously broken jobs. Ask first. (memory enforces this)
- **Don't add features, refactor, or introduce abstractions beyond the task.** Roee's repo discipline is strict. P9_05 was the only P9 experiment that needed code changes; everything else is yaml-only.
- **Don't bog down in per-pool details with small samples (<200 frames).** Aggregate, summarize, move to planning. (memory enforces this)
- **Don't push or amend commits without explicit OK.** Treat all git operations beyond local edits as confirmation-required.

---

## TL;DR for the next agent

Roee is going to show you something that may overturn the P8A premise. Until he does, sit with the context above and don't run anything. The scorecard is the only background process worth waiting on, and it doesn't need your help.
