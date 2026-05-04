# R13 Forward Plan — Master Plan 2026-04-29

> **What this document is.** A bird's-eye write-up of where the R13 Teams deepfake-detector training program stands as of 2026-04-29 afternoon, and a long-form forward-planning document covering the situation, strengths, weaknesses, what's been fixed and what hasn't, the paths still to investigate, and a decision tree for the next user-decision points. Length: long; verbosity intentional in this pass — the next pass compresses it.
>
> **Source of truth.** `docs/packet_retrospectives/` (the wiki built across 8 slice agents on 2026-04-29 plus the post-Slice-7 maintenance pass that landed the eval-substrate caveats). The wiki is treated as authoritative; source files (training code, yamls, scripts) are referenced only via wiki citations, never independently re-surveyed. Memory entries under `~/.claude/projects/.../memory/` are referenced where the wiki cites them, but the wiki is the canonical surface.
>
> **Citation rule (load-bearing).** Every load-bearing claim cites a wiki surface as `file:section` or `file:line`. Claims without a citation are tagged `[uncited]` inline and should be treated as inferential. The next pass (Codex via `/claudex:plan`) will verify citations and call out unsupported claims. **No claim in this document is invented from training data; everything traces back to a wiki surface.**
>
> **What this becomes.** This document is the long-form draft. The user will compress it to ≤ 400 lines as `PLAN.md` in the project root, then run `/claudex:plan --from-draft --rounds 2 "R13 forward plan"` to grill it adversarially. The refined `PLAN.md` will then update this master.

---

## 0. Read this first (orientation + scope)

This document has thirteen required sections plus appendices. A fresh agent or a tired user reading it cold should be able to:

1. State where the program stands (Section 1).
2. Identify the next user-decision point (Sections 11 and 12).
3. Name the top three to five risks (Section 10).
4. Follow the decision tree to a recommended action (Section 12).

**Confidence calibration**, applied throughout:

- **HIGH confidence** — measured directly on cached predictions or codified in repo state that is verified in the wiki: frame-level AUCs, the +207-line working-tree diff, the contract-policy bug structure, the bucket-gap distribution mismatch (with explicit GCS bucket citations), the wandb-flattening fix structure.
- **MEDIUM confidence** — derived from one or two wiki-cited measurements: shortcut Δ as residual deployment block (P13 Day-4 verdict measurement), expected lift from Layer-2 to ~24% viso recall (frame-level AUC ceiling argument), the in_proj-SVD lever's true magnitude post-fix.
- **LOW confidence** — items the wiki itself flags as open or substrate-conditional: production translation of eval-substrate FPR (open loop `eval-production-crop-tightness-mismatch`), whether the sharpness-metric bug materially shifts any headline, whether modern_v2's dor_shkedi-skew distorts the deployment-relevant FPR floor.

**Three-layer ship-readiness model** (referenced throughout):

- **Layer 1 — Contract scorecard.** The contract-policy v3 fix must commit + image-rebuild + scorecard-verify before any P-series scorecard read is trustworthy. Mechanical; awaits user authorization. (Loop `contract-policy-bug-fix-not-committed`, high.)
- **Layer 2 — Headline-metric magnitude (the bucket gap).** `R13_P14_DATA_FIX.yaml` wires the missing eval-distribution into training. Yaml-only intervention; ~$70 / 1-2 days; preconditioned by Move 1 (frozen-feature linear probe). (Loop `p14-data-fix-not-launched`, medium.)
- **Layer 3 — Residual shortcut at Axis 3.** P14_FT (in flight) tests whether P13's anti-shortcut interventions land additively on a working cross-domain base; P15 GRL is the structurally-different next lever, contingent on P14 verdict. (Loop `shortcut-deployment-block`, critical, in-progress.)

These three layers must close together for ship-readiness. Closing one without the others still leaves the candidate non-deployable. The decision tree in Section 12 sequences them.

---

## 1. Situation summary

### Where we stand at end-of-day 2026-04-29

The Effort detector for Teams deepfake detection is **frame-level strong on cross-domain fakes but contract-recall weak under the legacy policy, with a separately-fixable bucket gap on the headline metric and a residual camera-signature shortcut as the deployment block** (`STATE_2026-04-29.md:9-11`). This sentence is load-bearing — every other claim in this section unpacks one of its three clauses.

#### Frame-level signal is real

Frame-level AUCs computed directly from cached `P8A_step5000` predictions on the dev eval suites, no τ involved (`STATE_2026-04-29.md:74-77`, memory `project_p8a_frame_level_auc_2026-04-29.md`):

| Suite | Frame-level AUC |
|---|---:|
| `visomaster_enhanced_macro_dev` | **0.7527** |
| `deeplive_enhanced_dev` | **0.8614** |
| `teams_fake_all_dev` | **0.9106** |

These numbers are the load-bearing pivot of the Slice-7 narrative shift (`processing_signature_shortcut.md:51-58`). They reframe "low recall" headline numbers as **scorer artifact** rather than model failure. Through Slices 4-6 the dominant framing was "the model has learned a shortcut and consequently underperforms on cross-domain fakes." The frame-level AUC evidence reframes this as: **the model has substantial cross-domain signal (especially on deeplive: AUC 0.86; and teams_fake: AUC 0.91), but the contract scorecard was reading near-zero recall because the τ-policy was operating in a regime where the model's confident-fake region sat below the selected threshold** (`processing_signature_shortcut.md:60-61`).

The shortcut is real (Axis 3 Δ ~0.5 on best P13), but it was **not** the proximate cause of the headline-metric failure that motivated three packets of redesign (`processing_signature_shortcut.md:61`). The model is meaningfully less broken than headline metrics suggested — though still not deployment-ready, because:

- Corrected viso 13.6% is well below the 90% gate (`STATE_2026-04-29.md:84`).
- Shortcut residual at Axis 3 Δ = 0.52 still triggers the deployment block (`P13.md:14`).
- Modern_v2 FPR explodes on P13 from-scratch to 30.2% (`P13.md:73`).

#### Contract recall is weak under the legacy policy — but mostly artifact

Under the buggy contract policy that minimizes FPR with no recall budget, P8A reports near-zero recall on cross-domain fake suites (`contract_policy_bug.md:7-15`). Under the corrected policy with `target_fake_recall_min=0.30` (the v3 fix in working tree), P8A reports the consistent-with-AUC numbers (`STATE_2026-04-29.md:80-86`):

| Suite | Buggy policy (τ ≈ 0.99) | Corrected policy (τ ≈ 0.92, recall floor 0.30) |
|---|---:|---:|
| `visomaster_enhanced_macro_dev` | 1.1% | **13.6%** |
| `deeplive_enhanced_dev` | 1.6% | **23.9%** |
| `teams_fake_all_dev` | ~5% | **52.6%** |
| `lockbox_real_fpr` | 0.4% | 1.8% (still well within the 5% budget) |

**Reconciliation** (the load-bearing claim from Slice 7, `STATE_2026-04-29.md:87`): the buggy contract policy was reading near-zero recall as a **τ-tail-collapse artifact**, not a model failure. The corrected-policy recall numbers are consistent with the underlying frame-level AUCs (the (1 − AUC) margin inversion roughly maps to the operating-point recall under a sane FPR budget). **Every P-series scorecard read pre-fix is a lower bound on actual deployment performance.**

#### Bucket gap as separate-but-related layer on the headline metric

The eval suite `visomaster_enhanced_macro_dev` reads from `gs://teams-faces-data-test-2914-fake-4420-real-feb-28/fake/` — GAN-enhanced viso fakes recaptured through Teams (550 videos in this slice; `viso_bucket_gap.md:15`). Training viso reads `gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_*` with `tiers: null`. **Zero enhancers, no Teams transport** in training (`viso_bucket_gap.md:15`).

The deeplive control case explains the asymmetry. `deeplive_enhanced_dev` reads from the same Teams-recapture eval bucket. Deeplive *training* includes both `edge_cases_enhanced` / `minimal_processing_enhanced` strategies AND the `deeplive_teams_*` family from the separate `live-...-teams` bucket. So deeplive's training distribution approximately matches its eval distribution; viso's does not (`viso_bucket_gap.md:16`). This perfectly predicts the persistent ~3-5× viso-vs-deeplive recall gap on every candidate at every τ since RLP5 — which had stayed unexplained through six packets of recipe-tuning.

The asymmetry magnitude is **~2× in delta-from-1**, not 3-5× (`viso_bucket_gap.md:19`). Frame-level AUC: viso eval 0.7527 / deeplive eval 0.8614. So `(1 − 0.75) ≈ 2 × (1 − 0.86)` — the gap is real but quantitatively smaller than the recipe-tuning history implied. This sets a structural ceiling on what Layer-2 closure can deliver: viso plausibly lifts to ≈ deeplive's substrate-matched ~24% under the corrected contract policy, **not** to the 90% headline target.

The viso recall failure is approximately **two-thirds bucket-gap, one-third shortcut** (`viso_bucket_gap.md:29`) — a heuristic estimate consistent with the AUC ceiling argument. Layer-2 closure is yaml-only; `R13_P14_DATA_FIX.yaml` is drafted but not launched.

#### Camera-signature shortcut as residual deployment block

The 2026-04-24 controlled 2-camera test (`processing_signature_shortcut.md:14`, `R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md:5-19` cited in the wiki) promoted the camera/ISP-signature shortcut from "lockbox curiosity" to a hard deployment blocker: same person, same lighting, swap the camera and the score flips 0.02 ↔ 0.94 on Dor (laptop vs webcam) and 0.01 ↔ 0.90 on Roee (Windows vs Mac). The `dor_shkedi` vs `real_dor` flip on the Dor-annotated sanity manifest (`processing_signature_shortcut.md:13`) provides the same finding from the lockbox side: same person, different `identity_key`, model gives mean prob_fake = 0.457 on `dor_shkedi` vs 0.038 on `real_dor`.

P8A's CLIP backbone unfreeze (proj + ln_post + MLP-SVD) breaks the P7 anchor ceiling at single-variable cost (Δ=−0.188 vs RLP6_04 on `dor-real-webcam-no-VBG`; `processing_signature_shortcut.md:20-21`), but at the cost of −13.6 pp aggregate fake recall on `teams_fake_all_dev` (`processing_signature_shortcut.md:23`). The trade is structural: ~80% of misses sit below τ-recoverable band, separability collapse not threshold drift.

P13 anti-shortcut interventions (anchor-aware loss + pipeline-random aug + face-scale jitter, `P13.md:25-42`) moved Axis 3 directionally on best ckpt (P13_step18000 = 0.520 vs P8A 0.659, ~14pp improvement) but did not break the gate (still 3.5× the 0.15 gate; `processing_signature_shortcut.md:67-69`). The from-scratch trade also collapsed cross-domain capability (viso 5.5%, deeplive 48.8% at τ=0.5; `P13.md:73`).

#### What's running, what's drafted, what's uncommitted

In flight (`STATE_2026-04-29.md:18-19`):

- **`R13_P14_FT_FROM_P8A.yaml`** — Day-5 fallback per Plan v6 (FT-from-P8A_REFERENCE_step5000 + all three P13 anti-shortcut interventions). Currently launched on Vertex (us-east1 → us-west4 fallback). Cost: ~$60. Result expected within ~24h.

Designed-not-launched (`STATE_2026-04-29.md:22-25`):

- **`R13_P14_DATA_FIX.yaml`** — bucket-gap closure variant. Wires `VisoMasterEnhancedSample` loader into `combined_paired.py` as `visomaster_enhanced_fake` family. Drafted 2026-04-29 (untracked). Cost: ~$70 / 1-2 days. **Layer-2 closure is yaml-only — wiring is fully built** (`viso_bucket_gap.md:17`).
- **`R13_P15_GRL_FROM_P8A.yaml`** — gradient reversal on quality-domain head (DANN). Drafted 2026-04-29; smoke-test required before paying for a Vertex run. The cleanest structurally-different anti-shortcut lever. Five-line yaml change from P14. Cost: ~$60. Launch decision contingent on P14 verdict (α: don't / β: do / γ: do; per `P15.md:60-66`).
- **Move 1 — frozen-feature linear probe** of P8A on eval bucket vs training bucket viso. ~$0 / ~1-2 hours. The pre-spend gate that confirms or downgrades the bucket-gap finding before committing to P14_DATA_FIX. **Not yet executed as of 2026-04-29.**

Fixes in working tree, uncommitted (`STATE_2026-04-29.md:28-41`):

- **Contract policy v3 fix.** +207 net lines across `arena/score_teams_promotion_contract.py` (98 lines), `arena/run_target_domain_validation_sequential.py` (19 lines), `tests/test_score_teams_promotion_contract.py` (125 lines, 2 new tests). Adds `target_fake_recall_min=0.30` recall floor; bumps FPR budgets 0.02/0.05 → 0.07/0.10; tier-based sort at within-ckpt AND cross-ckpt summary levels. Six contract tests pass locally. Re-scoring P13 reports under the new policy lifts P8A from rank 4 to rank 1 (τ=0.916, viso recall 1.1% → 13.6%). **Third "fix" attempt in 6 days, uncommitted.** See `contract_policy_bug.md:25-66` for the canonical Slice 7 status block.
- **VisoMaster enhanced wiring tests** — `tests/test_visomaster_enhanced_wiring.py` (3 tests: domain map, sample tagging, bucket routing). Untracked.
- **`R13_P14_DATA_FIX.yaml`** — the bucket-fix retrain yaml. Untracked.
- **`R13_P14_FT_FROM_P8A.yaml`, `R13_P15_GRL_FROM_P8A.yaml`** — drafted, untracked.
- Several smaller patches (`feat_norm_reg_lambda` knob, `path_exclude_contains` filter, `entrypoint.sh` subcommands, `launch_batch_inference.sh` cd-vs-SCRIPT_DIR fix, the `CLAUDE.md` US-region preference paragraph).

#### Eval-substrate caveats added 2026-04-29 afternoon

A post-Slice-7 visual audit of the very-sharp-FP and is_no_face slices discovered three issues with the eval substrate that bear on how every FPR / recall number above should be read (`STATE_2026-04-29.md:54-64`):

1. **Eval-vs-production crop-tightness mismatch** — eval frames carry more background context around the face than production crops do (`eval_production_crop_tightness_gap.md:14-22`). Structurally upstream of the camera-signature shortcut and the FPR investigations the wiki tracks. Translation from eval FPR to production FPR is suspect for background-driven failure modes.
2. **`sharpness_laplacian` is computed on the full image, not the face** (`sharpness_metric_bug.md:14-26`, `analysis/lockbox_tagging/layers/quality.py:80-82`). The metric conflates face sharpness with non-face content sharpness. Per-quartile FPR analyses indexed on this column are partially confounded.
3. **Eval substrate has known data-quality issues** (`eval_substrate_data_hygiene.md`):
   - The `is_no_face` slice (n=219) is data degeneracy where MediaPipe's "0 faces" verdict is correct, not a model-weakness signal.
   - The eval substrate contains 99×110-pixel source crops (e.g., `PC_Generator__s15_1754.4_frame_058496_crop_002__d83b853d.jpg`) that pass the existing `face_pixel_area > 1k` floor but fail any reasonable source-resolution bound like `min(width, height) >= 200`.

**All FPR numbers in this document should be read with these caveats.** Modern_v2 already excludes `is_no_face` (so the v2 0.71% FPR is unaffected by the is_no_face finding), but v2 does **not** apply a source-image-resolution floor and v2's filter set does not address the crop-tightness gap. The `shortcut-deployment-block` close criterion is measured on the eval substrate and its production translation is now an open variable. **None of this changes the contract-recall reframing** (`STATE_2026-04-29.md:88-89`) — frame-level AUCs are computed directly on cached predictions, with all of the same eval-substrate caveats applied uniformly across baseline-vs-corrected-policy comparisons. What it does change is: a future agent who reads "the model has eval-FPR X" should not treat X as a production-FPR floor without auditing the eval-vs-production crop-tightness delta.

#### Confidence levels at this snapshot

- **HIGH confidence:**
  - Frame-level AUCs as measured (`STATE_2026-04-29.md:74-77`).
  - Bucket gap as the dominant headline-metric driver (citations explicit at `compute_verdict.py:51,335`, manifest pointers, training-side viso source — `viso_bucket_gap.md:15-17`).
  - Contract-policy bug structure (the +207-net-line working-tree diff is concrete; the 6 contract tests pass locally; `contract_policy_bug.md:31-37`).
  - Identity diversity is adequate (memory `project_data_inventory_identity_diversity.md`; `identity_audit.md:18-26` table).
  - Camera-signature shortcut is real on production captures (the 2-camera controlled test was production-side; `processing_signature_shortcut.md:91-97`).
- **MEDIUM confidence:**
  - The "two-thirds bucket-gap, one-third shortcut" decomposition (`viso_bucket_gap.md:29` is heuristic; consistent with AUCs but not strict).
  - Expected Move 2 lift to ~24% viso recall (frame-level AUC ceiling argument; `viso_bucket_gap.md:19`).
  - In_proj-SVD lever is small/inert in canonical recipe (Phase C overnight tied within noise; `in_proj_svd_gradient_bug.md:21-22`); other recipes might exercise it.
- **LOW confidence:**
  - Production translation of eval-substrate FPR (open loop `eval-production-crop-tightness-mismatch`, `eval_production_crop_tightness_gap.md:62-69`).
  - Whether sharpness-metric bug materially shifts any headline (the very-sharp-FP slice may simply be small enough that re-tagging doesn't move headline FPR; `sharpness_metric_bug.md:78-85`).
  - Whether modern_v2's dor_shkedi-skew (211/281 = 75%; `webcam_fpr_dominance.md:50`) makes v2 FPR identity-specific rather than deployment-distribution.

---

## 2. Strengths

What the model genuinely does well, and what we have solid evidence of. Cited by what's load-bearing about each.

### 2.1. Frame-level cross-domain signal is real and substantial

The single most consequential discovery of Slice 7. Frame-level AUCs computed directly from cached predictions, no τ involved:

| Suite | AUC | Implication |
|---|---:|---|
| `visomaster_enhanced_macro_dev` | 0.7527 | Cross-domain signal is real but bounded; substrate-matched ceiling for Layer-2 lift is ~24% recall (`viso_bucket_gap.md:19`). |
| `deeplive_enhanced_dev` | 0.8614 | Strong cross-domain signal where training distribution approximates eval. |
| `teams_fake_all_dev` | 0.9106 | Near-deployment-grade signal on the most deployment-relevant suite. |

Source: `STATE_2026-04-29.md:74-77`, memory `project_p8a_frame_level_auc_2026-04-29.md`.

These are **dispositive evidence** that the contract scorer's "viso recall = 1.1%" pre-fix reports were a measurement artifact. Memory codifies an operational rule: *"Always sanity-check operating-point recall against frame-level AUC before concluding a checkpoint is bad. If contract recall ≪ AUC-implied recall, suspect the τ-policy first"* (`processing_signature_shortcut.md:63`).

### 2.2. P8A breakthrough on the camera-signature anchor

P8A produced the first run that materially weakens the camera-signature shortcut (`processing_signature_shortcut.md:18-25`):

- Anchor pool `dor-real-webcam-false-flag-no-virtual-bg` Δ = **−0.188** vs RLP6_04 (best P7 was −0.097 ≈ 2× improvement).
- Roee-mac Δ = **−0.399** (best P7 was −0.228 ≈ 1.75×).
- All three real-correct pools also moved closer to zero (no FPR regression).
- Lockbox-layer: `lockbox_real_fpr` halved from 0.441% → 0.147% (`processing_signature_shortcut.md:167`).

Single-variable delta from RLP7_02: `unfreeze_final_proj=true`, `unfreeze_final_ln=true`, `apply_svd_to_mlp=true`. Memory `project_p8a_breakthrough.md` is the canonical anchor.

The breakthrough was not free: −13.6 pp aggregate fake recall on `teams_fake_all_dev` at default τ. But the breakthrough is real on the shortcut axis.

**Important caveat (2026-04-26 retroactive correction):** P8A's headline single-variable claim was originally reported as adding three new levers on top of an already-trained in_proj-SVD baseline. The in_proj-SVD silent zero-gradient bug (`in_proj_svd_gradient_bug.md`) means `apply_svd_to_in_proj=true` was a no-op pre-2026-04-26. The "more reach" qualitative explanation still holds, but the lever set was smaller than originally reported (memory `project_p8a_breakthrough.md:24-31` CORRECTION block).

### 2.3. Modern_lockbox_v2 0.71% FPR — data hygiene at P8A baseline

The modern_lockbox_v2 filter sweep on P8A step 5000 (`webcam_fpr_dominance.md:31-46`):

| Filter | n_real | n_fake | FPR @ τ=0.5 | FPR @ τ=0.9741 |
|---|---:|---:|---:|---:|
| baseline (all) | 414 | 425 | 21.98% | 4.59% |
| v2_recommended | 281 | 367 | 6.76% | **0.71%** |

v2_recommended definition:
```
clip_capture_mode not in {webcam, screen}
AND face_area_ratio >= 0.10
AND not is_pose_extreme
AND not is_no_face
```

At calibrated 5%-production τ, modern_v2 lockbox FPR is **0.71%** — a 6.5× reduction from baseline 4.59% with no retraining. **The FPR side of the 90/5 target is solved by data hygiene at P8A baseline** (`webcam_fpr_dominance.md:46`). The recall side (≥ 90% on viso/deeplive/teams_fake) remains the gating concern.

Caveat: v2_recommended is `dor_shkedi`-skewed (211/281 = 75% of v2 reals are dor_shkedi; `webcam_fpr_dominance.md:50`). Per-identity FPR breakdown is essential before treating v2 numbers as load-bearing. The two highest-FPR identities (Chikara, PC_Generator__s15) are exactly the ones almost-fully filtered by the webcam-drop rule; the v2 FPR drop is partly a survivor-bias effect.

Post-audit caveat (2026-04-29 afternoon): under the crop-tightness gap, even modern_v2 may not be a strict production-FPR floor (`webcam_fpr_dominance.md:80-88`). The two caveats compound. The webcam-mode tail this thread documents drives FPR via camera/ISP-pipeline cues; if the eval substrate gives those cues a larger surface than production does, dropping the webcam tail removes one axis of bias but leaves the residual eval-vs-production crop-tightness mismatch on every remaining slice.

### 2.4. Operational discipline accumulated through six packets

The wiki documents three operational guardrails that landed during Slices 5-6 in response to recurrent failure modes (cited from `TIMELINE.md` and the threads):

- **Pre-launch image-currency guard** (commit `ad76cd8`, `scripts/launch/check_image_currency.sh` ~95 lines + integration). Prevents "yaml/checkpoint-map newer than image" mistakes by failing-fast at submission time. Smoke-tested both pass and fail cases. `TIMELINE.md:88` — Slice 5.
- **`tests/test_train_sweep_reapply_allowlist.py`** (commit `c366026`). The first generalized regression-test guard for the wandb-flattening bug class — fails CI if any `TRAINER_NESTED_KEYS` entry is not re-applied in `train_sweep.py`. Memory `project_wandb_flattens_nested_dicts.md` enumerates the 14-key allowlist post-fix. `wandb_flattening.md:29-33`.
- **Six contract tests passing locally** for the v3 fix, including `test_promotion_summary_sort_key_demotes_low_recall_with_floor` and `test_recall_floor_changes_winner_in_score_promotion_contract` (`contract_policy_bug.md:54-57`).

These three artifacts are evidence that the team learns from incidents and lands structural fixes (not just point patches). The wiki's anti-whack-a-mole protocol (`AGENTS.md`) is the latest such artifact.

### 2.5. Identity diversity is adequate — refutes "more identity diversity" recipe

Hard counts from local discovery caches (`identity_audit.md:18-26`):

| Bucket | Source | Samples | Unique identities | In P13 training? |
|---|---|---:|---:|:---:|
| `live-deepfake-methods-real-and-fake-frames-cropped` | viso (live, no enhancers) | 960 | **429** | ✅ |
| same | deeplive | ~5,800 / 2,040 dirs | **1,916** | ✅ |
| `live-...-teams` | teams (deeplive_teams + viso_hints_teams) | 2,626 | 1,306 | ✅ partial |
| `live-...-teams-v2` | visomaster_teams_v2_companion | 1,994 | **997** | ❌ |
| `visomaster-enhanced-face-cropped-v2` | viso enhanced (direct) | 2,073 | many | ❌ |
| `hdtf_visomaster_cropped_frames` + `_teams` | proper-data wave 2026-04-19 | 1,322 captures | **705** | ❌ |

Active training has **~430 viso + ~1,900 deeplive identities** — not the "N=8 per method" mental model that had been on the table. **+2,300 additional identities sit in unused buckets** including the `visomaster_teams_v2_companion` bucket (the closest match to the eval distribution by transport pipeline) and the 2026-04-19 proper-data wave (purpose-built same-identity-clean-and-teams pairs).

**Implication for forward planning:** the recipe-tuning effort to lift cross-domain recall by adding identity diversity (or by aggressive sampling weights, which P11_HEAVY_DEEPLIVE tried at 6× and saw a regression) is **mis-aimed** (`identity_audit.md:32`). The right axis is *pipeline transport* — clean vs Teams-recapture, no-enhancer vs enhanced — not raw identity count.

### 2.6. Layer-2 wiring is fully built

The bucket-fix infrastructure exists in code; only YAML flags + family weights + smoke test + ~$70 spend are missing (`viso_bucket_gap.md:17`):

- `data/sources/visomaster.py:605-890` — `VisoMasterEnhancedSample` (the unused loader).
- `arena/build_visomaster_enhanced_v2_manifest.py` and `arena/manifests/visomaster_enhanced_v2_manifest_2026-04-13.json` (2,073 frames, 9 swap × 8 enhancers).
- `data/sources/combined_paired.py:646` — `create_unified_samples_from_visomaster_enhanced` with quality-domain mapping (`QUALITY_DOMAIN_MAP["visomaster_enhanced"] = 2`).
- `train_sweep.py` re-applies the whole `combined_paired` block, so the wandb-flattening trap does NOT bite for new nested children.
- `tests/test_visomaster_enhanced_wiring.py` (3 tests: domain map, sample tagging, bucket routing) — uncommitted, untracked file as of 2026-04-29.

The structural blocker is one yaml stanza: `visomaster_enhanced.enabled: true` + `visomaster_hints_teams.enabled: true` + family weights in `combined_paired.sampling.family_weights`.

### 2.7. Distinguishing real signal from scorer-favorable artifact

Two patterns the wiki makes explicit:

- **High lockbox-fake-recall on RLP6_04 (84.2% at τ=0.5) was partially shortcut-driven** (`processing_signature_shortcut.md:13-14`). The `dor_shkedi` flip case (mean prob_fake = 0.457 on `dor_shkedi` vs 0.038 on `real_dor`) shows the same person, different processing pipeline, flipped output. The lockbox recall is real but the signal it relies on is partly the camera/ISP fingerprint, not manipulation content.
- **The contract reading near-zero recall pre-fix was scorer artifact, not model failure** (frame-level AUCs prove this; `processing_signature_shortcut.md:51-58`). This is the Slice-7 reframing.

Both patterns are honest readouts: the wiki does not pretend the model is better than it is, but it also does not let the contract scorer's degenerate τ-policy obscure the real cross-domain signal.

---

## 3. Weaknesses and confounds

What we cannot yet trust, with the reason why. Cited per failure mode.

### 3.1. Contract scorer τ-tail collapse

The lexicographic τ-selection in `arena/score_teams_promotion_contract.py:_threshold_sort_key` minimizes `dev_primary_real_fpr` first **with no recall budget** (`contract_policy_bug.md:7`). On a sharp-prediction model (Effort with confident-fake p90 ≈ [0.94, 0.99]), τ snaps to ~0.995 and reported recall craters by ~10–30× vs the trainer's W&B numbers. The same scorecard "passes" because no recall floor blocks promotion.

The bug is **structural in the deployment artifact** (lines 456-479 of `arena/score_teams_promotion_contract.py`) and has been recurring since at least 2026-04-23. Three "fix" attempts in 6 days have not produced a committed change. See Section 4 for the whack-a-mole pattern.

The pattern is **"design lands but doesn't ship to deployment artifact"** (`contract_policy_bug.md:74`). Every readout post-2026-04-23 inspects `selected_threshold` first and falls back to τ=0.5 if it's in the τ-tail-collapse band — operator discipline absorbing what a code commit would resolve.

### 3.2. Sharpness-metric full-image bug

`sharpness_laplacian` at `analysis/lockbox_tagging/layers/quality.py:82` runs on the full grayscale frame, not a face crop (`sharpness_metric_bug.md:14-19`). The metric conflates face sharpness with non-face content sharpness.

Reproduction case (`sharpness_metric_bug.md:21-26`):

- `dor_shkedi__frame_000059_seq906__fec59bc1.jpg` (494×393 source, face 148×166): full-image laplacian **760.7**, face-crop laplacian **484.1**, **bg-only laplacian 844.9**, face-area-fraction 12.67%.
- `PC_Generator__s15_1754.4_frame_058496_crop_002__d83b853d.jpg` (99×110 source, face 55×65): full-image laplacian **555.1**, face-crop laplacian **443.6**, **bg-only laplacian 1018.1**, face-area-fraction 33.50%.

Re-derived face-crop laplacians are 484 vs 444 (essentially identical face sharpness). What drives the high laplacian on the small frame: virtual-background mountains, body silhouette, JPEG ringing.

**Downstream impact:** the 2026-04-27 investigation's "very sharp (>402): 45% FPR" finding now mixes two populations (genuinely sharp faces in sharp environments vs soft faces in tiny / busy / compressed crops where non-face content drives the laplacian). Any per-quartile FPR analysis indexed on `sharpness_laplacian` is partially confounded.

### 3.3. Eval-vs-production crop-tightness gap

The structurally most consequential audit finding from 2026-04-29 afternoon (`eval_production_crop_tightness_gap.md:14-22`). **Eval frames carry more background context around the face than production crops do.**

Implication compounds existing findings:

- The camera/ISP-signature shortcut lives in pipeline metadata that surrounds the face. **If the eval substrate's background-to-face ratio is higher than production's**, then any pipeline-signal cue the model uses gets a larger surface to read in eval than in production. (`eval_production_crop_tightness_gap.md:21-22`)
- Eval FPR overstates production FPR for shortcut-driven false flags.
- Eval recall might overstate or understate production recall depending on whether the manipulation method's pipeline signal is read mostly from the face region or from the surrounding context.
- The close criterion of `shortcut-deployment-block` is measured on the eval substrate, whose mismatch with production is now an open variable.

The cleanest closure path: **quantify the crop-tightness delta between eval and production, decide whether to retag eval at production tightness or apply a correction at scoring time**. The cheapest closure path: re-crop a sample of the lockbox at production tightness and re-score; if the score distribution shifts materially, the gap is real and load-bearing. **Until that is done, every existing eval readout should carry the caveat** (`eval_production_crop_tightness_gap.md:42`).

The 2-camera controlled test (2026-04-24) was production-side captures, so the shortcut is real on production. The eval-substrate amplification (if real) compounds with the production-side mechanism rather than replacing it. **Do not unilaterally revise the close criterion.**

### 3.4. In_proj-SVD retroactive attribution problem

Every R12g/RLP/P-* run with `apply_svd_to_in_proj=true` trained the q/k/v residuals on **zero classification gradient** until commit `2feea58` (2026-04-26; `in_proj_svd_gradient_bug.md:5, 13-14`). The original pattern installed a `forward_pre_hook` that did `module.in_proj_weight.data.copy_(module._svd_in_proj.weight)` before each forward. `.data.copy_` writes values without participating in autograd, so `F.multi_head_attention_forward` read a frozen leaf and gradients never propagated back to the SVD residual parameters.

One-batch grad audit (`analysis/probe_battery_2026-04-26/grad_audit.py`): on the OLD pre-hook approach: 9/9 `svd_{q,k,v}.{U,V,S}_residual` parameters had `grad=None` or `grad.abs().sum() == 0` after a single classification-loss backward. On the fix: 9/9 in_proj residuals + 3/3 `SVDResidualLinear` residuals (out_proj path, never broken) receive non-zero gradient.

Implication for the wiki's published attributions:

- P8A's headline single-variable claim ("unfreeze visual.proj + ln_post + apply_svd_to_mlp") was reported as adding *three* new levers on top of an already-trained in_proj-SVD baseline. **Memory `project_p8a_breakthrough.md:24-31` carries a CORRECTION block dated 2026-04-26**: "P8A's anchor breakthrough therefore came from MLP-SVD + unfrozen `visual.proj` + unfrozen `ln_post` ALONE; the in_proj-SVD piece was a no-op."
- P7 "FT-only ceiling" reading was measured **with broken in_proj-SVD**. After the fix, the FT lever set is meaningfully larger (~3M trainable params per attention block × 12 blocks that weren't training). Memory `project_shortcut_is_upstream.md:33-40` carries a CAVEAT block. The shortcut-is-upstream evidence (RLP7_08 fork-point control) is unaffected; the "ceiling" claim itself needs re-validation post-fix.

**Open loop `apply-svd-in-proj-attribution-revision-needed`** (open, high; `OPEN_LOOPS.md`) tracks the unfinished part of the retroactive sweep — every R12g/RLP/P-* run cited in any retro that has not been annotated with the bug-class CAVEAT.

### 3.5. Modern_v2 `dor_shkedi`-skew

Of 281 v2 reals, **211 (75%) are `dor_shkedi`** (`webcam_fpr_dominance.md:50`). The two highest-FPR identities (Chikara_Takahashi__s22 with 83.3% identity FPR; PC_Generator__s15 with 89.7% identity FPR) are almost-fully filtered by the webcam-drop rule. The v2 FPR drop is partly a survivor-bias effect.

Per-identity FPR breakdown (`webcam_fpr_dominance.md:53-57`):

- bla_bla_chow__s1 (n=68, 45k px²): FPR=1.47%
- dor_shkedi (n=275, 22k px²): FPR=10.55%
- Chikara_Takahashi__s22 (n=42, 17k px²): FPR=83.3%
- PC_Generator__s15 (n=29, 2.3k px²): FPR=89.7% — tiny artifact crops, the catastrophic outlier

Per-identity breakdown is **mandatory** before treating v2 numbers as load-bearing.

### 3.6. Source-bucket linear probe says P8A is NOT shortcut-clean

10-class source-bucket probe (chance=0.10, plan §6 PASS gate ≤ 0.25; `processing_signature_shortcut.md:31-35`):

- RLP6_04_BASELINE_STEP23500 (control): 0.97 test_acc — known-shortcut model saturates the probe.
- P8A_REFERENCE_STEP5000: 0.461 test_acc — 4.6× chance, FAIL.
- C3_CODEC_HEDGE_VC_STEP2000: 0.416 test_acc — 4.2× chance, FAIL.

Implication: P8A's 65% τ=0.5 lockbox fake recall is **partially shortcut-attributable**. Codec aug pulled some source-pipeline signal out (~9% relative reduction P8A → C3) but didn't clear the ≤0.25 PASS gate. The probe gradient (P8A 0.46 vs C3 0.42) shows codec aug as a **partial** lever.

### 3.7. Crop-tightness sweep flips 53% of frame predictions

47-frame × 5-tightness sweep: 25 of 47 frames (53%) flipped the predicted label across the tightness range t∈[0.7, 1.5] (`face_size_label_leak.md:15-18`). FAIL regime sits in a "FAKE valley" at native crop; any perturbation drops FPR ~30-40pp. Per-FAIL-tag breakdown is uneven: 2 of 3 FAIL tags are crop-shortcut artifacts; the third uses a non-crop signal (likely a webcam-specific signature).

The face-size leak is the **same shortcut from a different axis** as the camera-signature shortcut (memory `project_face_size_label_leak.md`). Confirmed empirically by the crop-sweep findings: same person flips real ↔ fake under tightness perturbation alone.

### 3.8. AUC ceiling caps Layer-2 expected lift

Layer-2 (bucket-gap closure) plausibly lifts viso recall to ≈ deeplive's substrate-matched ~24% under the corrected contract policy, **not** to the 90% headline target (`viso_bucket_gap.md:19`). The gap from 24% to 90% is the residual that the shortcut + architectural-class interventions (P15 GRL, future representation-loss work) need to close.

This is a **quantitative bound** on what a successful P14_DATA_FIX run can deliver; not a claim that the bucket fix is unimportant. A 10-15pp lift over P8A's current 13.6% is meaningful, but it's not deployment-sufficient on its own. Layer 3 still needs to close.

### 3.9. 70% of cross-pool FPR gap remains post per-camera calibration

WS-P1 calibration probe verdict: 30.1% gap closure on average across {5,10,15,25}% target FPRs (`OPEN_LOOPS.md` `residual-70-pct-fpr-gap-no-lever-past-rlp7`, `TIMELINE.md` 2026-04-24 entry). Per-target {5%: 0.000, 10%: 0.333, 15%: 0.286, 25%: 0.583}.

Verdict: **per-camera calibration is a complement, not a cure**. The remaining 70% is not addressed by any current lever past RLP7. The shortcut needs a training-side fix.

### 3.10. Frame-level AUCs themselves carry the eval-substrate caveats

Honestly stated: the frame-level AUCs above are computed on the same eval substrate the eval-substrate-caveats section flags as having (a) a crop-tightness mismatch with production, (b) a confounded `sharpness_laplacian` metric, and (c) data-quality issues (is_no_face degeneracy + 99×110 source crops) (`STATE_2026-04-29.md:88-89`). Treating "frame-level AUC 0.91 on teams_fake" as a strict production-deployment floor inherits all those caveats.

The narrative shift Slice 7 captured (low recall is scorer artifact, not model failure) was correct directionally but the substrate itself has additional load-bearing caveats that Slice 7 did not surface. The corrected-policy numbers above are still load-bearing for relative comparisons (P8A vs P13, baseline vs corrected policy) since the eval-substrate caveats apply uniformly across both sides of the comparison; what shifts is the absolute interpretation of any single AUC or recall number as a production floor.

---

## 4. Things that were fixed multiple times in strange ways

The wiki's flagship value proposition. Each entry: what kept breaking, why the fixes didn't stick, current canonical state.

### 4.1. Contract-policy bug — three fixes in 6 days, none committed

**The pattern this whole knowledge base was built to prevent surfaces here in concentrated form** (`contract_policy_bug.md:3`). Memory `project_contract_policy_bug.md` is the auto-memory anchor; the thread is the deliberated synthesis.

#### Attempt #1 — 2026-04-23 (slot-07 sanity)

First live evidence (`contract_policy_bug.md:15`). The slot-07 sanity check on the calibrated promotion contract (carrier `arena/checkpoint_maps/teams_target_domain.r13_rlp5_07_sanity_2026-04-23.yaml`) is the first time the τ-tail-collapse fires in the wild on a real artifact. Selected `τ = 0.9953`, dev real FPR = 0.0 across all 3 real suites, but recall craters: `teams_fake_all_dev = 27.6%`, `visomaster_enhanced_macro_dev = 1.6%`, `deeplive_enhanced_dev = 0.0%`, `lockbox_fake_recall = 15.8%`, `dev_fake_macro_recall = 9.7%`. Trainer-side `value_composite=0.7736` vs contract-side macro recall 9.7% on the same checkpoint, same data, different τ.

**The bug is not yet diagnosed in-session** — the readout was filed as "expected sanity, slot-07 passes the contract on FPR axis" and the diagnostic energy moves to RLP6 gate-alignment. **No fix lands.** The first live evidence existed for 4+ days before anyone framed it as a bug.

#### Attempt #2 — 2026-04-27 (codec-hedge readout; lost to /tmp wipe)

The codec_hedge A.2-style validation ran 4 checkpoints and **all 4 hit the contract policy bug regime (τ ≈ 0.97–0.99)** (`contract_policy_bug.md:17`). The agent surfaces a PLAN CHANGE PROPOSAL with three forks: (i) raise the FPR gate, (ii) **fix the contract policy bug first**, or (iii) commit to the §9 "no candidate" branch. User authorizes fork (ii) end-to-end revalidation.

The user-authorized fork-(ii) work re-scores the codec_hedge reports under an uncommitted policy fix in `arena/score_teams_promotion_contract.py` (+61/-16) — the patch added `target_real_fpr` / `target_stress_fpr` budget infrastructure with a tier-based sort. Two variants written to `analysis/policy_reruns_2026-04-27/{default,recall_floor_30}/`. Under `recall_floor=0.30`, P8A's macro_recall (0.300) sits exactly at the floor → byte-identical to the no-floor "default" variant, so the floor isn't binding for this slate.

**The fix is not committed in this session** (`contract_policy_bug.md:19`). The agent's own next-step note explicitly leaves staging for the user. **The agent then changes machines / sessions; the smoke-tested fix is lost to /tmp wipe** — confirmed by the 2026-04-29 session header (memory `project_contract_policy_bug.md` "Status 2026-04-29: FIXED IN WORKING TREE, uncommitted").

#### Attempt #3 — 2026-04-29 (the v3 fix; the design that becomes uncommitted)

`target_fake_recall_min=0.30` + `target_real_fpr=0.07` + `target_stress_fpr=0.10` is the corrected policy (`contract_policy_bug.md:23`). Test coverage added in `tests/test_score_teams_promotion_contract.py` (2 new tests, 6 contract tests pass total).

**Re-scoring P13 reports under the new policy** moves headline numbers dramatically (`contract_policy_bug.md:23`):

- P8A_REFERENCE_STEP5000 becomes rank 1 (was rank 4 under old policy).
- Selected τ 0.916 (was ~0.992).
- Viso_enhanced_macro_dev recall 1.1% → **13.6%**.
- Deeplive_enhanced_dev recall 1.6% → **23.9%**.
- Teams_fake_all_dev recall ~5% → **52.6%**.
- Lockbox real FPR 0.4% → 1.8% (still well within the 5% budget).

**The fix is in the working tree, uncommitted as of today.** `git diff --stat`:

```
arena/run_target_domain_validation_sequential.py                |  19 +
arena/score_teams_promotion_contract.py                          |  98 ++++++++++++++---
tests/test_score_teams_promotion_contract.py                     | 125 +++++++++++++++++++++
```

The patch consists of three logically-distinct changes that ship together (`contract_policy_bug.md:39-50`):

1. **`arena/score_teams_promotion_contract.py:_threshold_sort_key` (lines 456-479).** The within-checkpoint τ selector. Adds `budget_active`, `recall_floor_active`, three-tier sort (tier 0 satisfies both budgets, tier 1 satisfies budgets but fails recall floor, tier 2 violates budgets). Default `target_fake_recall_min = 0.0` preserves backward compatibility.
2. **`arena/score_teams_promotion_contract.py:_promotion_summary_sort_key` (lines 510-531).** Cross-checkpoint ranker. Without tiering at the cross-ckpt summary level, the cross-ckpt ranker picks the lowest-`lockbox_real_fpr` ckpt regardless of how degenerate its recall is — exactly how `P13_step2000` (3.6% lockbox fake recall, 2.9% dev fake macro recall) was crowned rank-1 on 2026-04-29. The patch adds tier 0 / tier 1 by recall-floor satisfaction at the summary layer.
3. **`arena/run_target_domain_validation_sequential.py` (+19 lines).** Adds three new CLI args: `--promotion_target_real_fpr`, `--promotion_target_stress_fpr`, `--promotion_target_fake_recall_min`. Plus `--promotion_readout_only_suites`.

#### Why the fixes didn't stick

- **Attempt #1 was misclassified.** Filed as "expected sanity, FPR axis passes." The first live evidence existed for 4+ days before anyone framed it as a bug.
- **Attempt #2 was lost to /tmp wipe** — and the agent's own next-step note explicitly left staging for the user without explicit "commit this now" guidance.
- **Attempt #3 lives in working tree as of today** — third design + test pass; no commit yet.

The structural failure mode: **design lands as analysis output, doesn't propagate to deployment artifact** (`contract_policy_bug.md:120-122`).

#### Current canonical state (end of Slice 7)

The bug is fixed in source (`arena/score_teams_promotion_contract.py:456-479` + `:510-531`), tested (6 contract tests passing locally), and verified by re-scoring P13 reports (P8A becomes rank 1 with consistent-with-AUC numbers). **Commit + image rebuild + scorecard run are mechanical and pending user authorization.**

The wiki's loop `contract-policy-bug-fix-not-committed` (open, high) is the surface that should now prevent attempt #4.

### 4.2. Wandb-side hygiene pattern recurrence

Three documented instances of the bug class across slices.

#### Slice 3 instance — `872502c` (2026-04-22)

`wandb.init` flattens nested dicts so `wandb.config.get('value_composite')` returns None despite yamls declaring the new gate. Trainer init logs of RLP35 slots 01/02 print legacy `(0.02 / 0.04 / max)` despite NEW gate yamls (`TIMELINE.md` 2026-04-22 entry). 8-line per-block re-apply added to existing allowlist (~lines 170–290). 6 RLP35 slots cancelled and relaunched.

Memory `project_wandb_flattens_nested_dicts.md` codifies the rule: "every new nested dict needs a re-apply line." **No test guard added at this point.**

#### Slice 5 instance — `f366368` (2026-04-26)

wandb artifact name >128 chars bug fix (image 1.3.214 via `d609e39`). Long `log_prefix` scorecards triggered W&B artifact-name length validation — second wandb-side hygiene fix in two slices. Pattern recurs across distinct wandb surfaces. `TIMELINE.md` 2026-04-26 entry.

#### Slice 6 instance — `c366026` (2026-04-28)

`anchor_aware`/`face_scale_jitter`/`periodic_saves` keys silently DISABLED on P13 first launch (`wandb_flattening.md:17-22`). ~$3.50 burned across two cancelled jobs; the first wrong-layered fix attempt (trainer-side `_to_plain_dict()` helper) lost ~$1.40.

Trainer init logs at job `8345130870696312832` showed:
```
PipelineRandomization ENABLED p_real=0.55 p_fake=0.45 …
AnchorAwarePenalty DISABLED (enabled=False weight=5.0)
FaceScaleJitter DISABLED
```

PipelineRandomization is correctly **ENABLED** because it lives nested *under* `augmentation:` (which IS in the allowlist at `train_sweep.py:207-214`); AnchorAwarePenalty + FaceScaleJitter are **DISABLED** despite the yaml asserting `enabled: true`. The pattern: **the wandb-flattening mechanism is sensitive to the depth of the new key relative to existing allowlist coverage** (`wandb_flattening.md:35`). Any nested feature added under an existing top-level allowlisted block silently works; any new top-level block silently fails.

#### Why the fixes didn't stick

- **Slice 3 fix had no test coverage.** It was a mechanical patch: add 8 lines to `train_sweep.py`. The rule was named in memory but didn't get pre-applied because the new yaml blocks landed in commits `cab2909` and `c7dc828` without a paired allowlist update (`wandb_flattening.md:37`).
- **Slice 5 fix addressed a different wandb-side surface** (artifact-name length, not flattening). The pattern was opened as the loop `wandb-side-surface-hygiene-not-systematic` (medium) but no systematic fix landed.

#### Slice 6 fix is structurally complete

`c366026` adds three new keys to `train_sweep.py:283-313` allowlist mirroring the Slice-3 `value_composite` pattern. **First generalized regression-test guard for the bug class:** `tests/test_train_sweep_reapply_allowlist.py` fails CI if any key in `TRAINER_NESTED_KEYS` is not re-applied in `train_sweep.py` (`wandb_flattening.md:29-33`).

The Slice-3 loop `train-sweep-allowlist-not-closed-under-schema-changes` was marked `resolved` at end of Slice 3, but Slice 6's recurrence demonstrates that "marking resolved because mechanical fix exists" was premature (`wandb_flattening.md:43-45`). The Slice-3 loop **stays resolved** post-Slice-6 because the closure mechanism (the test guard) now exists; pre-Slice-6 the loop was prematurely marked resolved without the closure mechanism.

#### Current canonical state

Memory `project_wandb_flattens_nested_dicts.md` enumerates the 14-key allowlist post-fix. The post-`c366026` P13_FROM_SCRATCH relaunch fired ALL three new yaml blocks correctly: trainer init logged `AnchorAwarePenalty ENABLED weight=5.0 target_mean_prob=0.10`, `FaceScaleJitter ENABLED scale_limit=0.25`, `PeriodicSaves ENABLED step_list=[2000,4000,...]` (`wandb_flattening.md:86`).

The broader **`wandb-side-surface-hygiene-not-systematic`** open loop (medium) tracks the broader pattern; it stays `open` because `c366026` closes the train_sweep allowlist instance but does not address the broader category (artifact-name validation, future flattening surfaces).

### 4.3. In_proj-SVD silent zero-gradient bug

Different bug class (autograd graph break vs yaml flattening) but **same failure mode**: "the lever was silently a no-op; only an explicit audit caught it" (`in_proj_svd_gradient_bug.md:74`). Three instances together belong to a broader "silent fallback patterns" hygiene story.

The bug existed from whenever `apply_svd_to_in_proj=true` first landed up to 2026-04-26 — every R12g/RLP/P-* run that flipped `apply_svd_to_in_proj: true` trained those parameters on regularizer gradient only (when `lambda_reg > 0`) and on nothing at all from the task loss.

Discovery + fix:

- **2026-04-26 static review** caught the `forward_pre_hook` + `.data.copy_` pattern (`in_proj_svd_gradient_bug.md:13`).
- **One-batch grad audit** (`analysis/probe_battery_2026-04-26/grad_audit.py`) verified: 9/9 in_proj residuals + 3/3 `SVDResidualLinear` residuals receive non-zero gradient post-fix.
- **Commit `2feea58`** lands the fix.
- **Commit `7af72b1`** adds anchor-pool monitor + grad_audit as a regression guard.

Why this bug went undetected for so long: the silent-fallback class is invisible without a grad audit. P8A's headline "more reach via in_proj residuals" qualitative reading was consistent with the bug being absent; only the static review caught the issue.

**Phase C overnight slate (2026-04-26)** was the first attempt at re-validation: C.1 (with fix) vs C-ablation (`apply_svd_to_in_proj: false`) **tied within noise** on the canonical P10_SYM-on-P8A recipe (`in_proj_svd_gradient_bug.md:21-22`). The lever exists but is not exercised by this recipe. Loop `in-proj-svd-residual-capacity-not-exercised` (medium) tracks the unfinished question of whether a different recipe surfaces a measurable lever-on lift.

### 4.4. Pattern summary — "design lands, ship discipline fails"

All three patterns above share a meta-observation: **the design work is correct; what fails is the path from design to deployed artifact.** The wiki names this in the `silent-feature-failures-pattern` open loop (medium, `wandb_flattening.md:69-83`), consolidating three instances:

1. `apply_svd_to_in_proj` zero-gradient (added 2026-03-ish; broken since first landing).
2. `anchor_aware` / `face_scale_jitter` / `periodic_saves` silent DISABLED.
3. `periodic_saves` silent no-checkpoint (P12_HEAVY_LONG dud).

Closure criterion: **a single pre-launch CI / smoke / lint step that verifies every yaml-declared trainer feature actually emits an "ENABLED" log line at trainer init.** This is the canonical answer; smoke + a YAML-declared-feature-vs-runtime-log audit at `Trainer.__init__` is a cheaper variant that catches the same class.

The wiki's existence is itself part of this pattern's resolution: by tracking open issues in `OPEN_LOOPS.md` and reaffirming them every session via the `AGENTS.md` read protocol, the structural failure mode (design lands but doesn't ship) is harder to reproduce.

---

## 5. Things that were marked fixed but probably aren't

Items with status `resolved` (or implied-resolved by virtue of having a "fix") whose closure criterion was retroactively earned rather than directly met. The risk is recurrence.

### 5.1. Periodic_saves silent failure

The canonical example. **Patch at commit `c7dc828`** added periodic-step-based saves to trainer; defensive `or {}` guard; **no `isinstance(dict)` defense** (`TIMELINE.md` 2026-04-28 entry). Silently failed on P12_HEAVY_LONG (2026-04-28 12:25): no `periodic_*.pth` files saved despite yaml + trainer code being in place.

Closed via the Slice 6 wandb-flattening fix (`c366026`) that re-applies `periodic_saves` block in `train_sweep.py:283-313`. **Loop status: closed for periodic_saves specifically; OPEN as `silent-feature-failures-pattern` meta-loop** because no pre-launch CI / smoke / lint step verifies that every yaml-declared trainer feature actually emits an "ENABLED" log line at trainer init (`wandb_flattening.md:69-82`).

**Recurrence risk:** any new nested-dict block in a different code path could repeat the pattern. Current closure is "memory rule + test guard for the train_sweep allowlist case specifically." A future code path that bypasses `train_sweep.py` (for example, a direct trainer invocation without the launcher) would not benefit from the test guard.

### 5.2. Slice-3 `train-sweep-allowlist-not-closed-under-schema-changes`

Marked `resolved` at end of Slice 3 with mechanical fix `872502c` (8-line allowlist patch). **Recurred 6 days later in Slice 6** as `c366026`-precondition (3 missing keys: `anchor_aware`, `face_scale_jitter`, `periodic_saves`).

**Stays resolved post-Slice-6 because the Slice 6 fix added the test guard that prevents future allowlist drift** (`wandb_flattening.md:43-45`). But the recurrence itself is evidence: "marked resolved because a mechanical fix exists" was premature in Slice 3. The closure mechanism (the test guard) was added in Slice 6, retroactively earning the Slice-3 close.

This is a **case study in why the wiki's `resolved` status carries weight** only when accompanied by a closure mechanism (test, CI step, structural artifact), not just a mechanical fix.

### 5.3. In_proj-SVD lever post-fix

Bug fixed 2026-04-26 (commit `2feea58`); 9/9 in_proj residuals receive non-zero gradient post-fix (verified via grad_audit). **But the lever is not exercised by the canonical recipe** — Phase C overnight: C.1 (with fix) vs C-ablation (`apply_svd_to_in_proj: false`) tied within noise (`in_proj_svd_gradient_bug.md:62-69`).

Loop stays open as `in-proj-svd-residual-capacity-not-exercised` (medium). "Marked fixed" but the FT-only ceiling claim from P7 / RLP7_08 remains imprecise — measured with broken in_proj-SVD; the ceiling has not been re-measured from a clean fork point.

The retroactive correction to memory (`project_p8a_breakthrough.md:24-31` CORRECTION block, `project_shortcut_is_upstream.md:33-40` CAVEAT block) covers the two specific narratives most cited. The full per-packet sweep (every R12/RLP/P-* run cited in any retro) has not been done — open loop `apply-svd-in-proj-attribution-revision-needed` (high) tracks this.

### 5.4. Recurrent meta-pattern: closure-mechanism-vs-mechanical-patch

The wiki's authority order rule is: "in the rare case of drift, threads win, because they are deliberated synthesis and memory is a fast index that lags" (`AGENTS.md:164-166`). The pattern in this section is the closure-mechanism analogue: **a status of `resolved` means little without a closure mechanism**. The Slice-3 → Slice-6 trajectory is the canonical example.

Implication for the master plan: any item below that's nominally "fixed" but lives in working tree without a corresponding test guard or CI step should be treated as a candidate for the same pattern. Specifically:

- **The contract-policy v3 fix** has 2 new tests + 6 total contract tests passing locally, but the broader "FPR-minimization-no-budget" loop (`fpr-minimization-no-budget-tau-collapse`, in-progress) requires the **runner** to enforce the recall floor or τ ceiling. The runner gets new CLI args in the v3 fix, but the default value in the runner is `target_fake_recall_min = 0.0` (`contract_policy_bug.md:50`). If a future operator runs the scorecard without explicitly passing `--promotion_target_fake_recall_min 0.30`, the bug regime can re-fire. The closure mechanism (the in-runner default + a CI test that asserts the runner refuses τ ≥ 0.99 unless the floor is explicit) is not yet in place. This is a future-Slice work item, not a current blocker, but it's a candidate for the recurrent meta-pattern.

---

## 6. Wrong paths and dead-end conclusions

What we thought, what was wrong, when we noticed.

### 6.1. "N too small" identity-diversity hypothesis (refuted)

**What we thought:** "We have only N=8 identities per fake method" — therefore the cross-domain failure is data-narrowness at the identity axis, addressable by adding more identities or by aggressive sampling weights (which P11_HEAVY_DEEPLIVE tried at 6× and saw a regression).

**What was wrong:** The mental model was off by 2 orders of magnitude. Hard counts: 429 viso identities + 1,916 deeplive identities in **active training** (`identity_audit.md:18-26`); +2,300 additional identities sit in **unused buckets** including the bucket whose distribution matches the eval suite. The right axis was *pipeline transport* (clean vs Teams-recapture, no-enhancer vs enhanced), not raw identity count.

**When we noticed:** 2026-04-29 ~10:00 CEST after the P13 γ verdict, when the user pushed back on the agent's "training data may be too narrow at the identity axis" line of reasoning and asked the agent to count the buckets (`viso_bucket_gap.md:15`). The audit ran hard counts from `.viewer_cache/discovery/*.json` plus `gsutil` ground-truth where available.

**Why it's a wrong path with utility preserved:** the identity hypothesis was not entirely wrong — it was detecting the right surface (the wrong distribution) but pinning it on the wrong axis (`identity_audit.md:27`). Memory `project_clean_teams_same_identity.md` codifies the corollary: the unused +2,300 identities are largely the *same people* as in existing buckets, just recaptured through Teams. So the gain from wiring them in is not "more faces" but "more transports of the same faces" — which is the exact ground truth signal for "what should look the same vs different to a deepfake detector."

### 6.2. "FT-only ceiling" reading from before in_proj-SVD fix

**What we thought:** P7 ceiling at ~−0.10 was attributed to "head + attention-only SVD reach + frozen visual.proj/ln_post + frozen MLP" — i.e., reading the in_proj-SVD residual capacity as **load-bearing**. P8A's headline single-variable claim ("unfreeze visual.proj + ln_post + apply_svd_to_mlp") was reported as adding *three* new levers on top of an already-trained in_proj-SVD baseline (`in_proj_svd_gradient_bug.md:9`).

**What was wrong:** Every R12g/RLP/P-* run with `apply_svd_to_in_proj=true` trained those residuals on **zero classification gradient** (`in_proj_svd_gradient_bug.md:13-14`). The in_proj-SVD piece was a no-op. P8A's anchor breakthrough therefore came from MLP-SVD + unfrozen `visual.proj` + unfrozen `ln_post` ALONE; the lever set was smaller than reported.

**When we noticed:** 2026-04-26 static review of `_install_svd_in_proj_routing` (`in_proj_svd_gradient_bug.md:13`). Commit `2feea58` lands the fix. Memory `project_p8a_breakthrough.md:24-31` adds a CORRECTION block dated 2026-04-26; memory `project_shortcut_is_upstream.md:33-40` adds a CAVEAT block.

**Why the qualitative narrative survives:** P8A still cuts anchor FPR ~half vs RLP6_04; P7 still hits a ceiling at ~−0.10; the shortcut is still upstream of RLP6_04 (`in_proj_svd_gradient_bug.md:30`). What shifts is the *parameter accounting* and the question of "what is the maximum FT can reach with all levers actually on." The Phase C overnight slate (C.1 vs C-ablation tied within noise) suggests the in_proj-SVD lever is small relative to MLP-SVD + visual.proj + ln_post in the canonical recipe; whether a different recipe would surface a bigger lever is open (`in_proj-svd-residual-capacity-not-exercised` loop).

### 6.3. `dor_shkedi` / `real_dor` flip misread initially

**What we thought:** The slot-07 sanity readout at τ=0.9953 with viso recall = 1.6% / lockbox fake recall 15.8% / dev fake macro recall 9.7% (`processing_signature_shortcut.md:13`) was filed in-session as "expected sanity, slot-07 passes the contract on FPR axis."

**What was wrong:** This was the first live evidence of the contract-policy τ-tail-collapse bug AND it interlocked with the camera-signature shortcut. Same person (`dor_shkedi` vs `real_dor`), different `identity_key`, model gives mean prob_fake = 0.457 on `dor_shkedi` vs 0.038 on `real_dor`. The flip itself was identity vs pipeline; it was read as a model failure when it was a scorer + shortcut entanglement.

**When we noticed:** 2026-04-23 was the first live evidence; 2026-04-27 was the first time the agent framed it as a bug (PCP fork (ii) "fix the contract policy bug first"; `contract_policy_bug.md:17`). The diagnosis was 4+ days late. This is the proximate cause of the wiki's anti-whack-a-mole protocol.

### 6.4. Hints-as-supervision hypothesis (closed; negative result)

**What we thought:** WT-B (Slice 1, 2026-04-17) preserved 480+202 hints rows as "weak training signal." If hints rows added training signal, the cross-domain capability lift would be measurable on a hints-ladder ablation.

**What was wrong:** RLP1 (2026-04-19) tested hints ladder (`01→02→03`) as training signal. Result: hints fail at 8/8 matched holdouts (Δ −0.0077..−0.0115; `TIMELINE.md` 2026-04-20). **Closed Slice-1 loop as negative result.**

**When we noticed:** RLP1 readout, 2026-04-20. Loop `hints-as-supervision-hypothesis` is in `OPEN_LOOPS.md` Resolved section (`OPEN_LOOPS.md:236-241`).

### 6.5. Codec_hedge as a Phase D promotion candidate (downgraded)

**What we thought:** 2026-04-26 Phase C overnight: codec_hedge gives best_anchor/composite **+30%** vs canonical C.1. Trainer-side anchor metric was meaningfully better. Surfaced as the highest-priority A.2-style validation target (`processing_signature_shortcut.md:26`).

**What was wrong:** 2026-04-27 codec_hedge A.2-style validation does NOT survive lockbox-anchored evaluation. P8A reference dominates BOTH lockbox axes vs all 3 C-slate checkpoints. All 4 hit τ ≈ 0.97-0.99 contract-policy bug regime. Source-bucket linear probe on C3_codec_hedge: 0.416 (FAIL the ≤0.25 gate; `processing_signature_shortcut.md:30, 34`).

**When we noticed:** 2026-04-27 codec_hedge readout vs Phase C anchor-composite. The "+30% trainer-side" was real; the "Phase D promotion candidate" reading was wrong.

**Why this matters for forward planning:** trainer-side metrics are not deployment-grade. The wiki codifies this as the `value_composite_semantics` thread's resolved loop (`OPEN_LOOPS.md` `value-composite-semantics-undefined`). The operational rule "always sanity-check operating-point recall against frame-level AUC before concluding a checkpoint is bad" (memory `project_p8a_frame_level_auc_2026-04-29.md`) is the Slice-7 generalization.

### 6.6. Inference-time stack as a recipe lever (exhausted)

**What we thought:** Calibration + ensembling on top of P8A + P11 predictions could break the 90/5 gate without retraining. The hypothesis was that the substrate's signal was there; we just needed the right operating point or the right combination.

**What was wrong:** 2026-04-28 13:30 Option A subagent ran 85 configurations (30 single-candidate × 3 calibrations + 45 noisy-OR + 10 weighted-noisy-OR) on existing P8A + 4 P11 predictions. **All 85 fail the 90/5 gate** (`processing_signature_shortcut.md:42`). Best ensemble: `p8a_reference + p11_heavy_step1000` isotonic-noisy-OR, τ=0.846 → viso 31.3%, deeplive 58.2%, teams_fake_all 75.0%, modern_v2 FPR 4.98%, teams_real_all_dev FPR 10.9% (fails 7% cap).

**AUC(visomaster fake vs modern_v2 real) caps at ~0.65 across all candidates and ensembles** (best ensemble AUC = 0.647). Calibration moves probabilities, not AUC; ensembles of same-deficit candidates can't break the substrate ceiling.

**When we noticed:** 2026-04-28 Option A verdict γ; motivated Plan v4 (G — substrate redesign / H — data-axis / I — honest report). P13_FROM_SCRATCH is the substrate-redesign track of Plan v4.

### 6.7. Pattern across wrong paths

Common features:

- Each wrong path **detected a real signal** but pinned it on the wrong cause.
- Each took at least one full packet's worth of work to resolve (the identity hypothesis required RLP1 + a year of recipe-tuning before the bucket-gap audit; the codec_hedge path took Phase C + the source-bucket probe; the inference-time stack took Option A's 85 configurations).
- **The wiki's job is to prevent re-derivation** of these wrong paths — once `identity_audit.md` exists, an agent looking at viso recall ~1.6% will not re-propose "add more viso fake identities."

---

## 7. Paths to investigate further

For each: what it tests, expected cost, current state, what it informs.

### 7.1. Frame-level vs clip-level scorer reconciliation

**What it tests:** does the contract scorer's clip-level recall reconcile with the frame-level AUCs from cached predictions?

- Either (a) the corrected contract policy with `target_fake_recall_min=0.30` produces clip-level recalls within ~5pp of `(1 − threshold-implied-FNR)` from the frame-level distribution, OR
- (b) the contract scorer is documented as measuring something genuinely different from the frame-level signal (e.g. clip-aggregation thresholds, video-level voting policy) and the headline reporting is normalized so future agents do not compare clip-level recall to AUC-implied recall.

Loop: `frame-level-vs-clip-level-scorer-mismatch` (open, medium; `viso_bucket_gap.md:73-79`).

**Cost:** documentation + analysis; no Vertex spend.

**Current state:** untouched. The reconciliation has not been written down anywhere as a normative rule.

**What it informs:** every future scorecard read. The bucket-gap finding makes this loop materially more important: a future P14_DATA_FIX scorecard read with the unfixed contract scorer would under-report the bucket-fix lift by 10-30×, exactly as the prior whack-a-mole pattern (`viso_bucket_gap.md:79`).

### 7.2. Eval-vs-production crop-tightness quantification (Move 1.5)

**What it tests:** does the eval substrate's crop-tightness mismatch with production materially shift the score distribution on a representative checkpoint?

Cheapest first move: **re-crop a sample of the lockbox at production tightness** and re-score `P8A_step5000`. If the score distribution shifts materially, the gap is load-bearing (`eval_production_crop_tightness_gap.md:42`).

The user-facing version (`STATE_2026-04-29.md:166`): "re-crop a sample of the lockbox at production tightness and re-score P8A_step5000 to test whether the eval-vs-production crop-tightness gap moves the score distribution materially."

Loop: `eval-production-crop-tightness-mismatch` (open, high; `eval_production_crop_tightness_gap.md:62-69`).

**Cost:** hours of analysis; no Vertex spend.

**Current state:** drafted as a "Move 1.5" companion to the Move 1 linear probe. Independent question; both probes give a cleaner read on what the bucket-gap retrain is actually trying to fix.

**What it informs:** how every existing eval FPR / recall number translates to production. This is the structurally most consequential audit finding because it potentially reframes how every existing eval result should be read — not just one slice or one method.

### 7.3. P14_DATA_FIX (drafted, not launched)

**What it tests:** does bucket-gap closure lift `visomaster_enhanced_macro_dev` recall above the P8A baseline?

Wires `VisoMasterEnhancedSample` loader (`data/sources/visomaster.py:605-890`, default bucket `visomaster-enhanced-face-cropped`) into `combined_paired.py` as a `visomaster_enhanced_fake` family. ~4-8h plumbing + ~$70 / 1-2 days for an FT-from-P8A retrain. Far cheaper than another recipe-tuning variant and far more likely to move the headline metric (`P14.md:38-51`).

Loop: `p14-data-fix-not-launched` (open, medium; `viso_bucket_gap.md:61-70`).

**Cost:** ~$70 / 1-2 days (FT-from-P8A retrain) + ~$10 / ~3h scorecard.

**Current state:** drafted (`R13_P14_DATA_FIX.yaml`, untracked); test coverage drafted (`tests/test_visomaster_enhanced_wiring.py`, untracked). Move 1 is the proposed pre-spend gate.

**What it informs:** Layer-2 closure as primary lift on headline metric. Expected lift: viso recall 13.6% → ~24% (substrate-matched ceiling per `viso_bucket_gap.md:19`). NOT to the 90% headline target — the gap from 24% to 90% is the residual that the shortcut + architectural-class interventions need to close.

### 7.4. P15 GRL (drafted, smoke test required)

**What it tests:** does direct anti-shortcut supervision (DANN) break the Axis 3 ceiling that input-space perturbations couldn't?

Five-line yaml change from P14: `use_quality_domain_head: true` + `quality_domain_loss_weight: 0.20` + `quality_domain_require_labels: true` + `quality_domain_count: 4` + `quality_head_hidden_dim: 128`. Everything else identical to P14_FT_FROM_P8A. Seed 1501 to avoid wandb collision (`P15.md:38-46`).

Quality-domain mapping (`P15.md:24-32`):

| Domain | Sources |
|---|---|
| 0 clean_academic | `df40` |
| 1 webcam_codec | `external` (VCD), **`deeplive_teams` (Teams traffic)** |
| 2 studio_capture | `deeplive`, `visomaster` |
| 3 social_media | (none in training; `youtube` is OOD-monitor only) |

Teams traffic — the same distribution that drives modern_v2 FPR — lands in domain 1. The backbone is penalized whenever its features can identify "this is webcam_codec." That is exactly the shortcut breakdown P13_step18000 demonstrated at τ=0.5 (modern_v2 FPR exploded to 30%+ because the backbone tied "webcam look" to "fake").

**Cost:** ~$60 / ~8h on A100.

**Current state:** drafted (`R13_P15_GRL_FROM_P8A.yaml`, untracked); pre-launch checklist explicit (smoke test, image rebuild not required, contract scorer floor flag, WANDB env vars, user authorization; `P15.md:71-79`). Decision logic: contingent on P14 verdict (α: don't launch / β: launch / γ: launch; `P15.md:60-66`).

**What it informs:** the only structurally-different anti-shortcut lever the codebase already supports — every prior shortcut intervention has been input-space (augmentation, FT-from-base, face-scale jitter). GRL on a quality-domain classifier is the cleanest available next class because the data labels exist, the detector hooks exist, and the unit tests pass.

**Slice-7 reframing of P15's expected gain** (`P15.md:55-66`): GRL on training-only pipelines does not teach the model the eval pipeline; expected gain is much higher *after* the bucket gap closes. So P15 is **still useful but is no longer the cheapest next move** — it's contingent on P14 verdict + Moves 1-3.

### 7.5. Sharpness-metric re-derivation on face crops

**What it tests:** what changes if `sharpness_laplacian` is computed on the face-crop region instead of the full grayscale frame?

Update `analysis/lockbox_tagging/layers/quality.py:82` to use the existing `face_pixel_area` and bbox tags from the face-geometry layer. Re-tag parquet `analysis/lockbox_tagging/full_tags_2026-04-27.parquet`. Audit downstream consumers — notably the 2026-04-27 investigation's "very sharp (>402): 45% FPR" table.

Loop: `sharpness-metric-computed-on-full-image-not-face` (open, high; `sharpness_metric_bug.md:69-74`).

**Cost:** ~15 lines in `quality.py` + one re-tag pass + auditing downstream consumers.

**Current state:** mechanical fix; no work done yet.

**What it informs:** every per-quartile FPR analysis indexed on `sharpness_laplacian`. The very-sharp-FP slice can be re-partitioned by face-crop laplacian, enabling per-population analysis (loop `very-sharp-fp-slice-mixes-two-populations`, medium).

### 7.6. Source-image-resolution floor on eval substrate

**What it tests:** what happens to headline FPR / recall on a canonical checkpoint (P8A_step5000) when a `min(width, height) >= 200` floor is applied as an eval-scope bound?

Compose with modern_lockbox_v2 filter set or report side-by-side. The 2026-04-27 investigation's `face_pixel_area > 57k` cut catches the small-face problem indexed by face area but **misses the small-source-image problem** (99×110 frames pass `face_pixel_area > 1k` but fail any reasonable source-resolution bound; `eval_substrate_data_hygiene.md:18-19`).

Loop: `source-image-resolution-floor-not-applied-to-eval` (open, medium; `eval_substrate_data_hygiene.md:65-70`).

**Cost:** filter + rerun on canonical checkpoint.

**Current state:** drafted as a parallel cut to the existing face-area filter; no work done yet.

**What it informs:** deployment-relevance of every existing eval FPR number. The eval substrate is genuinely scoring on thumbnail-resolution inputs that no production deployment would feed the model.

### 7.7. Path priority sequencing

The wiki recommends a **canonical priority sequence** (`viso_bucket_gap.md` § 6 "Recommended action sequence", referenced in `P15.md:60-66`):

1. **Move 1** — empirical confirmation (linear probe, ~$0).
2. **Move 2** — wire enhanced viso into training (P14_DATA_FIX, ~$70).
3. **Move 3** — superpose anti-shortcut interventions on top of P14_DATA_FIX.
4. **Move 4** — same-identity pair training using proper-data wave.
5. **Move 5** — only if Moves 2-4 still leave Axis 3 Δ > 0.30: revive P15 GRL with the broader pipeline label set the new data enables.

So the canonical Slice-7 framing: **P15 is still useful but is no longer the cheapest next move; it's contingent on P14 verdict + Moves 1-3**. The Section 9 experiment slate sequences these explicitly.

---

## 8. Open-loop close-path map

For each of the top ten open loops by severity, propose a concrete close path: prerequisite, what to do, expected cost (compute $ + wall-clock), expected signal. Cite loop ID and current status.

### 8.1. `shortcut-deployment-block` (in-progress, critical)

**Prerequisite:** contract-policy v3 commit (loop 8.2 below) so the close criterion's `lockbox_fake_recall ≥ 0.60` half is readable.

**What to do:** chain the canonical priority sequence — P14_FT verdict (in flight) → Move 1 → Move 2 (P14_DATA_FIX) → Move 3 (anti-shortcut interventions on top of P14_DATA_FIX) → if Δ still > 0.30: Move 5 (P15 GRL).

**Expected cost:** ~$60 P14_FT (in flight) + ~$70 P14_DATA_FIX + ~$60 P15 = ~$190 across the sequence.

**Expected signal:** `dor-real-webcam-false-flag-no-virtual-bg` mean prob_fake ≤ 0.30 AND `lockbox_fake_recall ≥ 0.60` at a τ that holds `teams_ood_real` FPR ≤ 5% (`OPEN_LOOPS.md` `shortcut-deployment-block`).

**Sub-loop disposition (`shortcut-block-criterion-may-be-scorer-artifact-bound`, medium):** when the close criterion is hit (or a packet gets close), record whether the criterion's `lockbox_fake_recall ≥ 0.60` half is reaffirmed under the corrected contract policy or revised to a frame-level-AUC-based version. **Do not change the criterion silently** (`processing_signature_shortcut.md:147-156`). The disposition needs to be written down so a future agent does not re-derive the question.

### 8.2. `contract-policy-bug-fix-not-committed` (open, high)

**Prerequisite:** none — the working tree diff is ready.

**What to do:** commit +207 net line diff + image rebuild (`./dev.sh build-prod -y` auto-bumps VERSION) + scorecard run with `--promotion_target_fake_recall_min 0.30` on a representative recent checkpoint (e.g. P8A reference step 5000). All three commit + image + verified-readout components are required (`OPEN_LOOPS.md`).

**Expected cost:** mechanical commit + ~3h Vertex run.

**Expected signal:** scorecard documents τ via the recall-floor path (not via the legacy no-budget minimize-FPR-only path).

**Decision-owner:** user. **The wiki was built specifically to prevent attempt #4** (`contract_policy_bug.md:115-122`).

**Standing pattern:** every future P-series scorecard read (P14_FT verdict, P14_DATA_FIX scorecard, P15 scorecard) requires this fix to be in image; otherwise τ-tail-collapse fires and the readouts are unreadable. Operator discipline ("always check `selected_threshold`; if τ ≈ 0.99, defer to τ=0.5") works as a fallback but is not durable across operators.

### 8.3. `fpr-minimization-no-budget-tau-collapse` (in-progress, high)

**Paired with 8.2; closes together when v3 fix lands on `main`.** This loop is the bug-itself surface; loop 8.2 is the commit-discipline failure mode.

**Close criterion** (`OPEN_LOOPS.md` `fpr-minimization-no-budget-tau-collapse`): the `score_teams_promotion_contract.py` runner enforces a recall floor or τ ceiling that prevents `selected_threshold ≈ 0.995` configurations passing silently.

**Note on durability:** the v3 fix's runner default is `target_fake_recall_min = 0.0` (`contract_policy_bug.md:50`). If a future operator runs the scorecard without explicitly passing `--promotion_target_fake_recall_min 0.30`, the bug regime can re-fire. A future-Slice work item (not blocking the current commit) is to add a CI test that asserts the runner refuses τ ≥ 0.99 unless the floor is explicit.

### 8.4. `apply-svd-in-proj-attribution-revision-needed` (open, high)

**Prerequisite:** none.

**What to do:** either (a) re-train every prior R12g/RLP/P-* run with `apply_svd_to_in_proj=true` from a matched fork point with the fixed routing and document the deltas vs broken-bug counterpart in a per-packet table, OR (b) explicitly annotate every packet retro with "in_proj-SVD trained on zero classification gradient — q/k/v residuals contributed only via regularizer drift" (`OPEN_LOOPS.md` `apply-svd-in-proj-attribution-revision-needed`).

**Expected cost:** documentation pass for (b) is the cheap option (~hours per packet retro; ~6-10 retros). (a) is full retrain sweep across many packets ($X00s + days).

**Recommendation:** **(b) for all packets EXCEPT P8A and the upstream-shortcut reasoning, which already have CORRECTION/CAVEAT blocks in memory.** The Phase C overnight slate's C.1 vs C-ablation tied within noise suggests the lever is small relative to other levers in the canonical recipe; absolute attribution corrections likely don't change the qualitative narratives.

**Expected signal:** every published retro carries the bug-class CAVEAT.

### 8.5. `face-size-label-leak` (open, high)

**Prerequisite:** P14 verdict (FT or DATA_FIX) so the post-`c366026` candidate (with `face_scale_jitter` actually live) can be re-tagged.

**What to do:** re-measure per-method face-pixel-area distributions on the post-P14 candidate; require Cohen's d on dev_fake vs dev_real ≤ 0.10 AND tightness-sweep at inference flips ≤ 10% of frame predictions in the FAIL regime (`OPEN_LOOPS.md` `face-size-label-leak`).

**Expected cost:** post-P14 re-tag pass + a small inference run.

**Note:** parallel cut `source-image-resolution-floor-not-applied-to-eval` (loop 8.10) is a different axis; the face-size leak is per-method face-pixel-area, the resolution floor is source-image dimensions. Both bear on the eval substrate but address different mechanisms.

### 8.6. `eval-production-crop-tightness-mismatch` (open, high)

**Prerequisite:** none.

**What to do:** re-crop lockbox sample at production tightness and re-score `P8A_step5000` (Move 1.5; loop 7.2). Either (a) quantify delta and document translation rule, OR (b) replace canonical reporting surface with re-cropped substrate (`OPEN_LOOPS.md` `eval-production-crop-tightness-mismatch`).

**Expected cost:** hours of analysis; no Vertex spend.

**Expected signal:** if score distribution shifts materially, gap is load-bearing; revise close criteria for `shortcut-deployment-block` per the disposition decision tracked by `shortcut-block-criterion-may-be-scorer-artifact-bound`.

### 8.7. `sharpness-metric-computed-on-full-image-not-face` (open, high)

**Prerequisite:** none.

**What to do:** ~15 lines in `analysis/lockbox_tagging/layers/quality.py:82` + re-tag parquet + audit downstream consumers (notably the 2026-04-27 investigation's per-quartile FPR table; `OPEN_LOOPS.md` `sharpness-metric-computed-on-full-image-not-face`).

**Expected cost:** short working session.

**Expected signal:** every per-quartile FPR analysis indexed on `sharpness_laplacian` either rerun under corrected metric or annotated as confounded.

**Sibling loop `very-sharp-fp-slice-mixes-two-populations`** (medium) closes naturally once this closes; the per-population FPR split is the actual analysis output downstream readers care about.

### 8.8. `p14-data-fix-not-launched` (open, medium)

**Prerequisite:** Move 1 (linear probe, ~$0) + user authorization.

**What to do:** commit `R13_P14_DATA_FIX.yaml` + the `tests/test_visomaster_enhanced_wiring.py` tests, smoke load enhanced viso samples cleanly, retrain, contract scorecard run with corrected policy (`OPEN_LOOPS.md` `p14-data-fix-not-launched`).

**Expected cost:** ~$70 / 1-2 days (retrain) + ~$10 / ~3h (scorecard). ~$0 / ~1-2h for Move 1 pre-gate.

**Expected signal:** `visomaster_enhanced_macro_dev` recall lifted materially above the P8A baseline (≥ 24% target under the corrected contract policy with `target_fake_recall_min=0.30`).

**Decision-owner:** user.

### 8.9. `webcam-mode-fpr-dominance-headline-misleading` (open, medium)

**Prerequisite:** modern_v2 filter audit (referenced in Plan v4 Track H per `webcam_fpr_dominance.md:114`).

**What to do:** contract scorecard reports v2-filtered lockbox FPR as a first-class metric alongside baseline lockbox FPR; per-identity breakdown as hard sub-gate (e.g. "no single v2 identity has FPR > 30%" — the catastrophic outlier filter; `OPEN_LOOPS.md` `webcam-mode-fpr-dominance-headline-misleading`).

**Expected cost:** contract-scorer integration pass.

**Expected signal:** a candidate that has 5% baseline lockbox FPR but 0.7% v2 FPR is read as "deployment-FPR ≤ 5%" by the contract.

### 8.10. `silent-feature-failures-pattern` (open, medium)

**Prerequisite:** none.

**What to do:** a single pre-launch CI / smoke / lint step that verifies every yaml-declared trainer feature actually emits an "ENABLED" log line at trainer init — i.e., for every nested-dict block in the launched yaml that the trainer reads via `self.config.get('<block>')`, a runtime check fails fast (and a CI test fails offline) if the block is silently fall-through-disabled (`OPEN_LOOPS.md` `silent-feature-failures-pattern`).

**Expected cost:** small CI script + tests.

**Expected signal:** any future "added-but-not-firing" instance fails fast at submission time, not after $X spent on cancelled jobs.

### 8.11. Medium-tier follow-ups

- **`shortcut-block-criterion-may-be-scorer-artifact-bound`**: written disposition — either reaffirm or revise close criterion under corrected policy. Closes in the same packet that closes its parent; disposition needs to be written down so a future agent does not re-derive the question.
- **`source-image-resolution-floor-not-applied-to-eval`**: small filter + rerun on canonical checkpoint; either fold into modern_lockbox_v2 or report side-by-side.
- **`frame-level-vs-clip-level-scorer-mismatch`**: documentation + analysis pass; either reconcile the two scorers or document the contract scorer as measuring something genuinely different from frame-level signal.
- **`split-mode-delta-unquantified`** (low; `OPEN_LOOPS.md`): re-run RLP1_01 recipe with `hash_stable` on the same data snapshot; numeric bound on AUC delta attributable to `shuffle → hash_stable` alone.
- **`deploy-server-preprocessing-drift`** (medium): the deployment server at `http://34.16.217.28:8999` is verified through a `tests/test_inference_train_preprocessing_parity`-style guard; smoke probe through production endpoint reproduces post-fix anchor-pool number within ±0.02.
- **`residual-70-pct-fpr-gap-no-lever-past-rlp7`** (medium): a Packet-7+ readout reports per-pool FPR spread on the 6 Dor/Roee pools — closes ≥70% of the cross-pool FPR gap; OR a documented next-lever proposal (representation loss, data-side intervention, architectural change) is filed.
- **`per-identity-reducer-not-a-contract-gate`** (low): contract hard-gates on `max(per-identity real_fpr) ≤ <budget>` rather than admitting it as a sort key only.
- **`prior-leader-rescore-sweep`** (low): every pre-fix leader (RLP1_01, RLP2_02, RLP3_05, RLP3.5_02, RLP5_07, RLP6_04 — RLP6_04 is the only one already done) is re-scored through the `INTER_LINEAR` path on a matched eval suite.
- **`is-no-face-slice-is-data-degeneracy`** (low): 219 frames removed or relocated; 2026-04-27 investigation's "small but high-lift, FN lift 2.09×" reading documented as data-degeneracy.

---

## 9. Recommended experiment slate

Priority-ordered. For each: what it tests, expected outcomes (success / failure), cost, decision branch (success → next; failure → next).

### Priority 1 — Move 1: Frozen-feature linear probe

**Source**: `viso_bucket_gap.md:25`, `STATE_2026-04-29.md:166`.

**What it tests:** is P8A's eval-bucket viso AUC materially lower than train-bucket viso AUC?

A frozen-feature linear probe of P8A on the eval bucket (`gs://teams-faces-data-test-2914-fake-4420-real-feb-28/fake/`) vs the training bucket (`live-...-cropped` viso). If train-bucket viso AUC is materially higher (>0.80) than eval-bucket viso AUC (~0.65), the bucket gap is dispositive.

**Cost:** ~$0 / ~1-2 hours.

**Expected outcomes:**

- **Success:** train-bucket > 0.80 vs eval-bucket ~0.65 → bucket gap dispositive → commit to P14_DATA_FIX (Priority 5).
- **Failure:** bucket-bucket AUC similar → bucket gap real but not dominant → P15 GRL keeps higher priority; consider scope-extension to Move 1.5 (Priority 2).

**Decision branch:**

- Success → Priority 3 (contract v3 commit) → Priority 5 (P14_DATA_FIX commit + launch).
- Failure → Priority 2 (Move 1.5) for cross-check; consider elevating P15 GRL.

**User decision required:** authorize Move 1. ~$0; informational; gates the P14_DATA_FIX investment (`STATE_2026-04-29.md:131`).

### Priority 2 — Move 1.5: Re-crop at production tightness

**Source**: `STATE_2026-04-29.md:166`, `eval_production_crop_tightness_gap.md:69`.

**What it tests:** does re-cropping a sample of lockbox at production tightness shift the score distribution materially?

The eval-substrate-vs-production crop-tightness mismatch is structurally upstream of multiple shortcut/FPR investigations. This probe tests whether the gap is load-bearing (production translation is suspect) or marginal (eval substrate is honest deployment proxy on this axis).

**Cost:** hours of analysis; no Vertex spend.

**Expected outcomes:**

- **Success (gap is load-bearing):** distribution shifts materially → revise close criteria for `shortcut-deployment-block`; eval FPR readouts gain a translation-to-production caveat.
- **Failure (gap is marginal):** distribution stable → eval substrate is honest deployment proxy on this axis; current FPR numbers stand at face value.

**Decision branch:** this is informational; runs in parallel to the Move 1 → P14_DATA_FIX chain. The two probes are independent; running both gives a cleaner read.

**User decision required:** authorize Move 1.5. ~$0; informational on a meta-risk (`STATE_2026-04-29.md:166`).

### Priority 3 — Commit + ship contract-policy v3 fix

**Source**: `contract_policy_bug.md:108-122`.

**What it tests:** does the v3 fix produce a representative scorecard with non-degenerate τ on a representative checkpoint?

**Cost:** mechanical commit + image rebuild (auto-bumps VERSION) + ~3h Vertex run.

**Expected outcomes:**

- **Success:** scorecard documents τ via the recall-floor path; P8A_REFERENCE_step5000 becomes rank 1 with τ=0.916, viso 13.6%, deeplive 23.9%, teams_fake 52.6% (matches the local re-scored numbers).
- **Failure:** re-scored numbers diverge from local re-score; investigate before launching any subsequent P-series scorecard.

**Decision branch:** closes loops `contract-policy-bug-fix-not-committed` (high) and `fpr-minimization-no-budget-tau-collapse` (in-progress, high) by construction. **Standing precondition for every P-series scorecard read after this point.**

**User decision required:** authorize commit + image rebuild + scorecard run. Mechanical (`STATE_2026-04-29.md:130`).

### Priority 4 — P14_FT_FROM_P8A (in flight)

**Source**: `P14.md:22-36`, `STATE_2026-04-29.md:19-21`.

**What it tests:** do P13's anti-shortcut interventions land additively on a working cross-domain base (FT-from-P8A_REFERENCE_step5000)?

Spec: init from `P8A_REFERENCE_STEP5000` checkpoint; apply ALL three P13 anti-shortcut interventions (anchor-aware loss, pipeline-random aug, face-scale jitter); light fine-tune (small LR, ~2K-4K steps to avoid catastrophic forgetting); validate against the same Day-4 contract on Day-6.

**Cost:** ~$60 / ~8h on A100 us-east1 → us-west4 fallback (already running).

**Expected outcomes** (per `P14.md:62-67`):

- **α (passes triple-axis):** ship P14, save GRL for next round.
- **β (Axis 3 Δ in [0.15, 0.45], cross-domain ≥ P8A):** launch P15. Goal: drive Δ further down without giving back cross-domain.
- **γ (Axis 3 Δ ≥ 0.45 OR cross-domain < P8A by > 8pp):** launch P15. Substrate needs an additional structural lever.

**Decision branch:** the verdict drives P15 launch decision (Priority 6). Verdict will likely look like P13's even if the bundle works (because the bucket gap is unchanged) — explicit cautionary note in `viso_bucket_gap.md` § 3.

### Priority 5 — P14_DATA_FIX (drafted, contingent on Move 1 success)

**Source**: `P14.md:38-51`.

**What it tests:** does bucket-gap closure lift `visomaster_enhanced_macro_dev` recall above ≥24% target under corrected policy?

Spec: add `visomaster_enhanced_fake` family wiring in `combined_paired.py` (uses existing `VisoMasterEnhancedSample` class); update yaml to enable `visomaster.enhanced_enabled: true` + add `family_weights.visomaster_enhanced_fake`; image rebuild; smoke test (load 16 enhanced viso samples in CPU dataloader, confirm shapes/labels); training run (~$60 / ~8h FT-from-P8A); scorecard (~$10 / ~3h with `--promotion_target_fake_recall_min 0.30`).

**Cost:** ~$70 / 1-2 days + ~$10 / 3h scorecard.

**Expected outcomes:**

- **Success:** viso recall ≥ 24% under corrected policy → Layer-2 closed; pivot focus to Layer 3 (residual shortcut at Axis 3).
- **Failure:** viso recall < 18% → bucket gap was not dominant; reconsider the heuristic estimate; downgrade Layer-2 priority.

**Decision branch:**

- Success + P14_FT verdict α → ship P14, save GRL.
- Success + P14_FT verdict β → launch P15 with goal of further Axis 3 movement.
- Success + P14_FT verdict γ → launch P15 as substrate's last cheap structural lever.
- Failure → reconsider; elevate P15 priority OR investigate Move 4 (paired-transport contrastive).

**User decision required:** authorize P14_DATA_FIX commit + launch (contingent on Move 1 result). ~$70 / 1-2 days (`STATE_2026-04-29.md:132`).

**Eval-split policy decision (load-bearing):** if `R13_P14_DATA_FIX.yaml` wires the eval bucket directly into training, `visomaster_enhanced_macro_dev` is 550 videos all from one Teams-recapture session; if it lands in training as-is, it stops being a deployment proxy. Identity-split out a held-out portion before wiring (`STATE_2026-04-29.md:50, P14.md:76, viso_bucket_gap.md:5.1`). **User judgment, not delegable.**

### Priority 6 — P15 GRL (drafted, contingent on P14 verdict)

**Source**: `P15.md:9-16`, `docs/relaunch_handoffs/P15_GRL_READINESS_NOTE_2026-04-29.md` (cited in `P15.md:113`).

**What it tests:** does direct anti-shortcut supervision (DANN) break the Axis 3 ceiling that input-space perturbations couldn't?

Five-line yaml change from P14_FT (Priority 4): `use_quality_domain_head: true` + `quality_domain_loss_weight: 0.20` + `quality_domain_require_labels: true` + `quality_domain_count: 4` + `quality_head_hidden_dim: 128`. Seed 1501.

**Cost:** ~$60 / ~8h on A100.

**Expected outcomes** (per `P15.md:85-89`):

- **Best-case:** Axis 3 Δ moves below P14 AND viso/deeplive recall stays at or above P8A's level. First candidate where direct anti-shortcut supervision is acting on a working cross-domain base.
- **Modal expected (β):** GRL @ static λ=0.20 will likely move Δ another 0.05-0.15, still above the 0.15 gate, but with cross-domain recall preserved by the FT init. Licenses P16 (GRL with ramped λ, larger weight) or escalation to spectral/dual-stream features.
- **Worst-case:** GRL at λ=0.20 catastrophically degrades cross-domain recall (similar to P13_FROM_SCRATCH, but for a different reason: backbone forced to throw away too much information). **Mitigation: kill switch at step 1500 if viso recall drops > 8pp below P8A** (`P15.md:15`).

**Decision branch:**

- α (P14_FT passes): do not launch P15; ship P14.
- β / γ (P14_FT fails or partial): launch P15.

**User decision required:** authorize P15 launch per α/β/γ matrix on P14 verdict. ~$60 / 8h (`STATE_2026-04-29.md:134`).

### Priority 7 — Substrate hygiene cleanup

**Source**: `sharpness_metric_bug.md:69-74`, `eval_substrate_data_hygiene.md:65-70`, plus the is_no_face dispositioning.

**What it tests:** do these substrate hygiene fixes change any headline FPR/recall conclusion?

Three fixes:

- Sharpness-metric re-derivation on face crops (~15 lines + re-tag).
- Source-image-resolution floor (`min(width, height) >= 200`) on eval substrate.
- `is_no_face` slice removal/relocation from eval substrates that inform FPR/recall/FN-lift conclusions.

**Cost:** short working session.

**Expected outcomes:**

- Most likely: headline FPR numbers don't move materially (slices are small relative to full substrate), but downstream interpretations change (the very-sharp-FP slice splits into two populations; the source-resolution floor catches degenerate inputs; the is_no_face slice is documented as data degeneracy).
- Less likely: a slice splits revealing a previously-hidden failure mode worth a packet-level intervention.

**Decision branch:** runs in parallel to the main Move 1 → P14_DATA_FIX → P15 chain. Closes three medium-severity loops by construction.

**User decision required:** prioritize relative to next packet, or accept partial confound on slice analyses (`STATE_2026-04-29.md` § "Decisions deferred").

### Priority 8 — Move 4 (same-identity pair training using proper-data wave)

**Source**: `STATE_2026-04-29.md:170`, `clean_teams_identity_pairing.md` (referenced in `viso_bucket_gap.md:84-85`).

**What it tests:** same-identity contrastive learning on `(clean[id_X], teams[id_X])` pairs — does explicit pairing help the model distinguish "what should look the same vs different to a deepfake detector"?

The 2026-04-19 proper-data wave has 705 paired clean+teams identities; companion_bucket plumbing exists in code (`data/sources/visomaster.py:748,1085,1105,1176,1523`); `_load_merged_teams_enhanced_paired` function exists.

**Cost:** TBD (yaml + smoke test + ~$70 retrain).

**Position in queue:** post-P14_DATA_FIX, post-P15. Move 4 in the priority sequence (`viso_bucket_gap.md` § 6).

**Decision branch:** revival contingent on Layer-2 closure + Layer-3 status after P15 verdict.

### Priority 9 — Substrate-redesign last-resort options

If P14_DATA_FIX + P15 + Move 4 still leave Axis 3 Δ > 0.30:

- **Move 5: P15 GRL with broader pipeline label set the new data enables** (`P15.md:64`).
- **Representation loss work** (`OPEN_LOOPS.md` `residual-70-pct-fpr-gap-no-lever-past-rlp7`).
- **Architectural change** (spectral/dual-stream features per `P15.md:88`).

These are deferred. The wiki does not name an experiment slate beyond P15 + Move 4; further levers are speculative until the bucket-gap closure status is known.

---

## 10. Risk register

What could go wrong with each top recommendation. Include the meta-risk that the entire reframing (low-recall = scorer artifact) is wrong directionally.

### 10.1. Move 1 (linear probe) risk

**Risk:** the probe is on eval-substrate features. If the eval substrate's crop-tightness doesn't match production (loop `eval-production-crop-tightness-mismatch`), the probe signal might also not translate to production. The probe answers "are P8A's features bucket-discriminative on the eval substrate?" — if eval substrate's crop-tightness is materially different, that answer might not generalize (`STATE_2026-04-29.md:166`).

**Mitigation:** pair with Move 1.5 (re-crop at production tightness; `STATE_2026-04-29.md:166` recommendation). The two probes are independent; running both gives a cleaner read on what the bucket-gap retrain is actually trying to fix.

**Severity:** medium. The probe is cheap; the worst-case is that we spend ~$0 to learn the probe's signal is conditional on the eval-substrate caveats, which we already knew.

### 10.2. Move 1.5 (re-crop) risk

**Risk:** small sample size for re-cropping; might not be representative of the full eval substrate. Modern_v2 already excludes some slices (webcam, screen, pose_extreme, no_face); a sample drawn from v2 vs from baseline gives different reads.

**Mitigation:** stratify by `clip_capture_mode`. Document the sampling strategy.

**Severity:** low. Worst-case is that the probe's signal is noisy; cost is hours of analysis, not money or wall-clock.

### 10.3. Contract v3 commit risk

**Risk:** image rebuild + scorecard run takes ~3h Vertex; selected τ might surprise (e.g., the recall floor not binding for the chosen checkpoint).

**Mitigation:** re-scored numbers locally already match expected pattern (P8A becomes rank-1 with τ=0.916, viso 13.6% etc.; `contract_policy_bug.md:23`). Local re-score at `/tmp/p13_repolicy/recall_floor_30/` is the reference; the Vertex run should reproduce within ε.

**Severity:** low-medium. The fix is well-tested locally (6 contract tests passing); the surprise risk is small.

**Operational note:** memory `feedback_promotion_contract_launch.md` requires `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT` exported before invocation (unlike `launch_batch_inference.sh`). Failure to export will fail the run early, not produce wrong numbers.

### 10.4. P14_FT risk

**Risk:** the from-scratch trade-off persists in light fine-tune; cross-domain capability might not be preserved at the claimed level. Alternatively, the anti-shortcut interventions might land but not move Axis 3 enough to license P15 (modal β verdict; `P15.md:88`).

**Mitigation:** kill switch on viso recall regression; per readiness pattern of P15. The verdict's α/β/γ structure is explicitly permissive of partial wins; β verdict is the modal expected outcome and is sufficient to license P15.

**Severity:** medium. P14_FT is in flight; ~$60 is committed. The verdict drives the next decision regardless; even γ is informative.

### 10.5. P14_DATA_FIX risk — eval split policy

**Risk:** if `R13_P14_DATA_FIX.yaml` wires `visomaster_enhanced_macro_dev` directly into training, the suite stops being a deployment proxy. The 550 videos are all from one Teams-recapture session; identity overlap with training would inflate validation metrics.

**Mitigation:** identity-split out a held-out portion before wiring (`STATE_2026-04-29.md:50, P14.md:76, viso_bucket_gap.md` § 5.1). **User judgment, not delegable.** A naive frame-split (vs identity-split) preserves identities in both sides; would not address the leakage.

**Severity:** high if not addressed; the eval suite is the primary headline metric. If wired naively, every subsequent scorecard reads inflated numbers.

### 10.6. P14_DATA_FIX risk — expected lift might not realize

**Risk:** the heuristic "two-thirds bucket-gap, one-third shortcut" decomposition is just that — a heuristic. The substrate-matched ceiling at ~24% recall is bounded above by the deeplive-substrate AUC, but the actual lift might be less (if the model's bucket-discriminative features don't generalize from training distribution to enhanced-viso distribution at the implied rate).

**Mitigation:** Move 1 (linear probe) is the pre-spend gate that confirms or downgrades the bucket-gap finding. If Move 1 succeeds, the lift expectation is validated; if Move 1 fails, P14_DATA_FIX's expected outcome distribution shifts.

**Severity:** medium. ~$70 is the spend; the worst case is that P14_DATA_FIX delivers <18% recall (still above P8A's 13.6% but below the 24% target).

### 10.7. P15 GRL risk

**Risk:** first-ever production exercise of GRL infrastructure (`P15.md:50`). The detector code (`detectors/effort_detector.py:236-274`), the data labels (`combined_paired.py:65-82`), and the trainer loss aggregation (`detectors/effort_detector.py:906-958`) all exist and are unit-tested (`tests/test_unpaired_reals_and_grl.py`), but never run end-to-end on Vertex.

**Mitigation:** smoke test with explicit log-line greps (`P15.md:71-76`):
- `Quality domain head ENABLED with 4 domains, hidden_dim=128, loss_weight=0.2, require_labels=True`
- `Applied anchor_aware: enabled=True weight=5.0 target_mean_prob=0.1`
- `Applied face_scale_jitter: enabled=True scale_limit=0.25`
- `Applied periodic_saves: enabled=True step_list=[500, 1000, ..., 4000]`

Plus kill switch at step 1500 if viso recall drops > 8pp below P8A.

**Severity:** medium. The mitigation surface (smoke + kill switch) is well-specified; the risk is mostly "infrastructure not exercised before."

### 10.8. P15 GRL risk — λ static at 0.20

**Risk:** `GradientReversalLayer.set_lambda(val)` exists but no caller in `trainer/trainer.py` invokes it; ramping λ from 0 → target over warmup steps would require a small trainer patch (~10 lines; `P15.md:51-52`). Skipping ramp keeps this draft no-code-change. If P15 either crashes from too-strong-too-fast adversarial pressure OR is too weak to bite, the static-λ design is the proximate cause.

**Mitigation:** P16 should add the ramp if P15 catastrophically degrades or has zero effect. The static-λ design is a deliberate first-cut choice; the ramp is the planned-for refinement.

**Severity:** low. The static-λ design is simpler; the ramp is a planned follow-up.

### 10.9. Substrate-redesign overshoot risk

**Risk:** P14_DATA_FIX + P15 + corrections might still leave Axis 3 Δ > 0.30. The wiki's experiment slate beyond P15 (Move 4 paired-transport, Move 5 GRL with broader label set) is speculative until measured.

**Mitigation:**

- Branch: revive Move 4 (paired-transport contrastive) as the next data-axis lever.
- Fall back to Move 5 (P15 GRL with broader pipeline label set the new data enables; `P15.md:64`).
- Last resort: representation loss work, architectural change (spectral/dual-stream features).

**Severity:** medium. The substrate-redesign track is bounded by what's already in the wiki; new ideas might be required.

### 10.10. Meta-risk — the entire reframing is wrong directionally

**Risk:** what if "low recall = scorer artifact" is false, and the model is genuinely weak on cross-domain fakes?

Counter-evidence load-bearing for the reframing:

- Frame-level AUCs computed directly on cached predictions (`STATE_2026-04-29.md:74-77`).
- Numbers consistent across (1 − AUC) margin inversion and corrected-policy operating-point recall.
- Three independent τ-policies (legacy contract τ ≈ 0.99, calibrated 5%-FPR τ, τ=0.5 diagnostic) tell consistent stories on P13 verdict (`P13.md:64`).

**But:** AUCs are computed on the same eval substrate the eval-substrate-caveats section flags. Treating "frame-level AUC 0.91 on teams_fake" as a strict production-deployment floor inherits all those caveats (`STATE_2026-04-29.md:88-89`). If the eval-substrate-vs-production crop-tightness delta is large enough to matter for the AUC, the reframing partially survives but its absolute magnitude shifts.

**Mitigation:** Move 1.5 (re-crop at production tightness and re-score) is the cleanest test of this meta-risk. If the score distribution shifts materially under re-crop, the reframing is partially overstated; if not, the reframing stands.

**Severity:** if the meta-risk fires, the entire forward plan is mis-priced (Layer-2 bucket-gap closure expected lift is overstated; Layer-3 shortcut is more dispositive than the wiki implies).

The wiki's stance: the narrative shift Slice 7 captured (low recall is scorer artifact, not model failure) was correct directionally but the substrate itself has additional load-bearing caveats that Slice 7 did not surface. The corrected-policy numbers are still load-bearing for relative comparisons (P8A vs P13, baseline vs corrected policy); what shifts is the absolute interpretation of any single AUC or recall number as a production floor.

---

## 11. Open questions for the user

The per-loop "what would close this" mapped to "is the user the one who needs to decide". Per memory `feedback_decision_points.md`: present recommendation + tradeoffs, wait for explicit pick.

### 11.1. User-decision (judgment calls)

These cannot be delegated to the agent.

1. **Authorize Move 1?** ($0, ~1-2h) — gates the P14_DATA_FIX investment. Recommendation per `STATE_2026-04-29.md:172`: yes; the cheapest decisive step on the largest pending uncertainty in the program.

2. **Authorize contract v3 commit + image rebuild + representative scorecard?** Mechanical; closes two high-severity loops (`contract-policy-bug-fix-not-committed` + `fpr-minimization-no-budget-tau-collapse`) by construction. The wiki was built specifically to prevent attempt #4 of this fix. Recommendation: yes.

3. **Authorize P14_DATA_FIX commit + launch?** Contingent on Move 1 result. ~$70 / 1-2 days. Recommendation per `STATE_2026-04-29.md:169`: if Move 1 confirms the bucket gap as dispositive, yes.

4. **Authorize P15 launch?** Per α/β/γ matrix on P14 verdict. ~$60 / 8h. Recommendation per `P15.md:60-66`: β/γ → launch; α → don't.

5. **Eval-split policy for P14_DATA_FIX**: identity-split out held-out portion before training? **User judgment, not delegable.** Without an identity-split, `visomaster_enhanced_macro_dev` stops being a deployment proxy after training.

6. **Cancel in-flight P14_FT once Move 1 lands?** Per memory `feedback_no_cancelling_vertex_jobs.md`, the default is no — cancellation is destructive class even for hung jobs; needs user OK first. P14_FT tests a separate hypothesis ("can the bundle of three anti-shortcut interventions land additively on a working cross-domain base?"); defensible $60 to let it finish even if Move 1 redirects priorities.

7. **Authorize Move 1.5 (re-crop sample at production tightness)?** ~$0, hours; informational on a meta-risk. Recommendation per `STATE_2026-04-29.md:166`: yes; runs in parallel to Move 1.

8. **Sharpness-metric fix priority** — block on it before next packet, or accept partial confound on slice analyses? Recommendation: do not block — fold into a routine analytics maintenance pass alongside the source-resolution floor and is_no_face cleanup (Priority 7 in Section 9).

9. **Source-image-resolution floor disposition** — fold into modern_lockbox_v2 or report side-by-side? Recommendation: side-by-side first to surface the delta, fold in once the delta is documented.

10. **Disposition on `shortcut-block-criterion-may-be-scorer-artifact-bound`** — reaffirm or revise close criterion under corrected policy? Recommendation: defer until P14 verdict + Move 1.5 lands; the disposition needs both inputs to be informed.

### 11.2. Agent-actionable (no user decision needed beyond standing approval)

These can run with standing approval:

- Run regenerator script after thread updates (`tools/regenerate_open_loops.py` per `AGENTS.md:101-107`).
- Re-tag parquet after sharpness-metric fix.
- Document cross-thread caveats in threads as new findings land.
- Update memory frontmatter `last_verified` after each session (`AGENTS.md:89-99`).
- Verify working-tree state matches threads' "fix in progress" annotations before any code change (`AGENTS.md:50-58`).

### 11.3. Pending decision points map (from `STATE_2026-04-29.md:128-138`)

| Decision | Recommendation | User action |
|---|---|---|
| Authorize Move 1 (linear probe) | Run before P14_DATA_FIX commit; ~$0 | OK / not OK |
| Authorize contract v3 commit + image rebuild | Commit per close criterion | OK / push back |
| Authorize P14_DATA_FIX (assuming Move 1 confirms bucket gap) | Highest-priority remaining unbuilt experiment | OK / not OK / change scope |
| Authorize P15 launch (post P14 verdict) | β/γ → launch per matrix; α → don't | OK / not OK |
| Eval-split policy for P14_DATA_FIX | Identity-split out a held-out portion | User judgment (not delegable) |
| Cancel in-flight P14_FT once Move 1 lands? | P14_FT tests separate hypothesis; defensible $60 | OK / cancel |

---

## 12. Decision tree

The forward plan as a branching decision tree. Each branch named with its trigger event, terminal nodes with the implied next action.

### 12.1. Top-level branching

The decision tree branches off two independent input signals:

- **Move 1 verdict** (success / failure) — gates the P14_DATA_FIX commit.
- **P14_FT verdict** (α / β / γ) — gates P15 launch.

Plus one parallel precondition that gates **everything** that uses contract scorecards:

- **Contract v3 commit + image rebuild + scorecard verify** — must happen before any P-series scorecard read is trustworthy.

### 12.2. The tree

```
Contract v3 commit (Priority 3)
├── DEFERRED: every P-series scorecard inherits τ-tail-collapse regime; manual τ=0.5 fallback per session
└── DONE: scorecards produce non-degenerate τ; P-series readouts trustworthy
    │
    └── (parallel) Move 1 (Priority 1) + Move 1.5 (Priority 2)
        │
        ├── Move 1 SUCCESS (train-bucket AUC >> eval-bucket AUC)
        │   │
        │   └── commit + launch P14_DATA_FIX (Priority 5)
        │       │
        │       ├── P14_DATA_FIX SUCCESS (viso recall ≥ 24% under corrected policy)
        │       │   │
        │       │   ├── P14_FT verdict α: ship P14_FT, save GRL for next round
        │       │   ├── P14_FT verdict β: launch P15 GRL with goal of further Axis 3 movement
        │       │   └── P14_FT verdict γ: launch P15 GRL as substrate's last cheap structural lever
        │       │
        │       └── P14_DATA_FIX FAILURE (viso recall < 18%)
        │           │
        │           ├── reconsider: bucket gap was not dominant; the heuristic 2/3 share was overstated
        │           ├── elevate P15 priority (the GRL bet now has higher expected gain)
        │           └── consider Move 4 (paired-transport contrastive) as alternative data lever
        │
        └── Move 1 FAILURE (bucket-bucket AUC similar)
            │
            ├── downgrade bucket-gap finding to "real but not dominant"
            ├── elevate P15 GRL to top priority (post-P14_FT verdict)
            └── (orthogonal) consider Move 1.5 scope-extension
                │
                ├── Move 1.5 SUCCESS (score distribution shifts under re-crop)
                │   └── crop-tightness gap is load-bearing
                │       └── revise `shortcut-deployment-block` close criterion to production-translation-aware version
                │
                └── Move 1.5 FAILURE (distribution stable under re-crop)
                    └── eval substrate is honest deployment proxy
                        └── focus moves to Layer 3 (residual shortcut) entirely
```

### 12.3. Side-channels and gates

- **P14_FT verdict drives P15 launch independent of Move 1.** P14_FT is in flight at end of Slice 7; its verdict will land within ~24h regardless of Move 1 status. The α/β/γ matrix at `P15.md:60-66` is the canonical decision rule.
- **Move 4 (paired-transport contrastive) sits below P14_DATA_FIX and P15 in the canonical priority sequence** (`viso_bucket_gap.md` § 6). It is the data-axis lever that becomes the next obvious move if P14_DATA_FIX delivers the substrate-matched ceiling but the residual shortcut at Axis 3 still fires modern_v2 FPR > 5%.
- **Substrate hygiene (Priority 7) runs in parallel** to the main decision tree. It does not gate any other priority; it does provide cleaner readouts for downstream substrate-level decisions.

### 12.4. Standing operational discipline

Every node in the tree assumes:

- Working-tree-diff check before any commit (`AGENTS.md:50-58`).
- Image-currency check before any Vertex launch (commit `ad76cd8`, `scripts/launch/check_image_currency.sh`).
- US-region preference: `us-west4` ↔ `us-east1` ↔ `us-central1` only; non-US requires explicit user authorization (`CLAUDE.md`).
- 30-minute pending threshold for region switch; never cancel original until replacement is RUNNING.
- WANDB env vars exported before any contract-scorecard launch (memory `feedback_promotion_contract_launch.md`).
- No cancellation of in-flight Vertex jobs without explicit user authorization (memory `feedback_no_cancelling_vertex_jobs.md`).

These are not nodes; they are the conditions under which any node executes.

### 12.5. Walk-through of the modal expected path

Here is what likely happens, end to end, if the modal expected outcomes hold:

1. **User approves Move 1 + Move 1.5.** Both probes run in parallel; ~$0; ~hours.
2. **Move 1 succeeds** (train-bucket viso AUC > 0.80, eval-bucket ~0.65). Bucket gap confirmed dispositive.
3. **Move 1.5 result is informational.** Either confirms eval substrate is honest proxy (status quo for `shortcut-deployment-block` close criterion) or flags eval-vs-production translation gap (caveat added to FPR readouts).
4. **User approves contract v3 commit.** Mechanical; ~3h Vertex run validates.
5. **User approves P14_DATA_FIX commit + launch.** ~$70 / 1-2 days. Identity-split done; eval suite preserved as deployment proxy.
6. **P14_FT verdict lands** (in parallel; ~24h from launch). Modal: β. Anti-shortcut interventions land additively but Axis 3 Δ stays in [0.15, 0.45]; cross-domain ≥ P8A.
7. **User approves P15 launch** (per β branch). ~$60 / 8h.
8. **P15 GRL verdict lands.** Modal: β again. Δ moves another 0.05-0.15, still above 0.15 gate, cross-domain preserved. Licenses P16 (GRL with ramped λ) or escalation to spectral/dual-stream features.
9. **P14_DATA_FIX scorecard lands.** Viso recall ≈ 24% under corrected policy. Layer-2 closed.
10. **Substrate hygiene cleanup runs in parallel.** Sharpness fix, source-resolution floor, is_no_face dispositioning. Headline numbers do not move materially; downstream interpretations clarified.

End state at the modal expected outcome: contract v3 in image; P14_DATA_FIX bucket-gap closed (viso ~24%); P14_FT verdict β; P15 verdict β; P16 drafted; Move 4 in queue. Axis 3 Δ residual at ~0.30-0.40; modern_v2 FPR still elevated relative to P8A baseline. Deployment block not yet closed; structural anti-shortcut work continues.

### 12.6. Unhappy paths

- **Move 1 fails:** bucket gap was not dominant. Plan reverts to "P15 GRL is top priority, possibly with broader pipeline labels." The 6 weeks of recipe-tuning history is partially redeemed (the recipes were under-powered against a non-bucket-gap-dominant cause; P11/P13's anti-shortcut interventions are vindicated as directionally correct).
- **P14_FT verdict α:** ship P14, no need for P15. Plan compresses.
- **P14_DATA_FIX fails:** the heuristic 2/3 share was overstated. Either the bucket gap is real-but-marginal or the substrate-matched ceiling argument is wrong. Plan moves to Move 4 (paired-transport contrastive) and parallel substrate-redesign work.
- **P15 catastrophically degrades cross-domain (worst-case):** kill switch at step 1500 fires; mitigation by lowering loss weight or removing `quality_domain_require_labels` safety belt. P16 with ramped λ is the next bet.
- **Meta-risk fires (frame-level AUC reframing partially wrong):** Move 1.5 surfaces this if the score distribution shifts materially under production-tightness re-crop. Plan re-prices: Layer-2 expected lift downgraded; Layer-3 (shortcut) elevated; modern_v2 FPR readouts gain a translation-to-production caveat.

---

## 13. Contradictions found in the wiki

Empty list is valid; don't manufacture contradictions.

After a careful read of the source-of-truth surfaces (AGENTS.md, STATE_2026-04-29.md, OPEN_LOOPS.md, TIMELINE.md, all 11 specified threads, P13/P14/P15 packet retros), **no load-bearing contradictions were found**.

Five candidate items were audited and resolved as not-contradictions:

1. **Open-loop count discrepancy.** STATE_2026-04-29.md says "19 open + 3 in-progress + 7 resolved at end of Slice 7" (`STATE_2026-04-29.md:122`); OPEN_LOOPS.md (regenerated 2026-04-29) shows "Open: 24, In progress: 3, Resolved: 7." Reconciliation: STATE notes "audit adds 5 new open loops"; 19+5=24. Not a contradiction.

2. **`viso_bucket_gap.md` 2/3-bucket-gap heuristic vs frame-level AUC ceiling argument.** The heuristic decomposition ("approximately two-thirds bucket-gap, one-third shortcut"; `viso_bucket_gap.md:29`) coexists with the AUC ceiling argument that bounds Layer-2 lift at ~24% (`viso_bucket_gap.md:19`). These are consistent — the heuristic is a decomposition of the residual headline-metric gap; the AUC is an upper bound on the absolute viso recall achievable under bucket-fix. Not a contradiction.

3. **`shortcut-deployment-block` close criterion measured on eval substrate vs `eval-production-crop-tightness-mismatch` loop.** The close criterion of the in-progress critical loop is measured on the eval substrate, whose mismatch with production is now an open variable. This is acknowledged via paired sub-loop `shortcut-block-criterion-may-be-scorer-artifact-bound` (`processing_signature_shortcut.md:147-156`). Not a contradiction; explicit cross-thread caveat.

4. **P13 retro mentions "post-fix relaunch" timing.** `P13.md:9` references "2026-04-28 EOD (post-`c366026` relaunch) → 2026-04-29 02:30 UTC (Day-4 verdict γ)." `TIMELINE.md` 2026-04-28 EOD entry confirms the post-`c366026` relaunch fired all three new yaml blocks live; verdict 02:30 UTC 2026-04-29. Consistent.

5. **`silent-feature-failures-pattern` first_seen 2026-04-28** but the in_proj-SVD bug (named as instance #1 of the meta-pattern) was fixed 2026-04-26. Resolution: the meta-loop was authored 2026-04-28 retroactively grouping prior instances; the loop's `first_seen` is the date of its consolidation, not the date of any particular instance. Not a contradiction.

The wiki appears coherent. Future passes (Codex via `/claudex:plan`, the user's compression to PLAN.md) may surface load-bearing contradictions that this read missed; if so, list them here in a numbered update with citations to the disagreeing surfaces.

---

## Appendix A: Glossary of metrics, suites, and packet names

For a fresh agent reading cold. Most of this is encoded in the wiki's threads; consolidated here for reference.

### A.1. Metrics

- **Frame-level AUC**: ROC-AUC computed directly on cached per-frame predictions, no τ involved. Slice-7 load-bearing pivot.
- **Operating-point recall (under contract policy)**: TPR at the τ selected by `arena/score_teams_promotion_contract.py:_threshold_sort_key`. Pre-fix: τ-tail-collapse regime ~0.99. Post-fix (recall floor 0.30): τ ~0.92.
- **Axis 1**: cross-domain fake recall (viso/deeplive/teams_fake_dev). P13 verdict gate: ≥ 90% on each, fail if ≥15pp short.
- **Axis 2**: lockbox real-FPR / lockbox-fake-recall composite. Gate: 80%+.
- **Axis 3**: `dor-real-webcam-false-flag-no-virtual-bg` mean prob_fake. Gate: ≤ 0.15 (P13 best is 0.520 = 3.5× the gate).
- **`value_composite`**: trainer-side directional metric; explicitly **not deployment-grade** (`value_composite_semantics` thread, resolved loop).
- **`worst_pool_fpr`**: real-pool-only FPR; documented invariant of the contract since RLP6.
- **`anchor/composite`**: in-trainer anchor-spread monitor (commit `7af72b1`); compares anchor metrics against an expected curve.

### A.2. Suites

- **`visomaster_enhanced_macro_dev`**: 550 GAN-enhanced viso videos recaptured through Teams. Reads from `gs://teams-faces-data-test-2914-fake-4420-real-feb-28/fake/`. Headline cross-domain fake suite for viso.
- **`deeplive_enhanced_dev`**: same eval bucket, deeplive method. Training distribution matches eval (deeplive_teams + edge_cases_enhanced + minimal_processing_enhanced). The control case for the bucket-gap finding.
- **`teams_fake_all_dev`**: aggregate Teams-traffic fake suite. Most deployment-relevant.
- **`teams_real_all_dev`**: aggregate Teams-traffic real suite. The primary FPR gate.
- **`modern_lockbox_v2`**: filtered subset of lockbox real (281 reals, 367 fakes) at `clip_capture_mode not in {webcam, screen} AND face_area_ratio >= 0.10 AND not is_pose_extreme AND not is_no_face`. Deployment-relevant FPR proxy.
- **`lockbox_real_fpr` / `lockbox_fake_recall`**: baseline lockbox readouts (without the v2 filter).
- **`dor-real-webcam-false-flag-no-virtual-bg`**: 30-frame anchor pool from the 2-camera controlled test. Axis 3 substrate.
- **`teams_ood_real`**: stress-OOD real pool; gates `≤ 5% FPR` in the close criterion.

### A.3. Packets

- **WT-A through WT-F**: Slice 1 (2026-04-17) parallel worktree tracks. WT-A = data-policy truth freeze; WT-B = weak-signal hints; WT-C = augmentation truth; WT-D = decision-system tooling; WT-E = promotion-contract scaffold; WT-F = proper-data schema.
- **RLP1 through RLP7**: relaunch packets 1-7. RLP6_04 is the gate-alignment leader (`value_composite=0.9006`; checkpoint at step 23500). RLP7 was launched as the structural response to the 2-camera shortcut finding.
- **WS-P0 through WS-P4**: weakspot probes. WS-P0 = preprocessing parity fix (commit `855871e`). WS-P1 = calibration probe (30.1% gap closure). WS-P2.b = per-identity reducer.
- **P8A**: first run to break the P7 anchor ceiling. Single-variable delta from RLP7_02. The deployment-relevant baseline as of 2026-04-29.
- **P8B**: scratch on plain CLIP-DataComp-XL. Refuted "the shortcut is in the weight chain" (scratch was worse than RLP6_04).
- **P9**: soften P8A along three single-variable axes (magnitude / topology / data) + replication + freeze-native control.
- **P10**: anti-shortcut packet (symmetric router + GRL slate, commit `2c9778b`). Phase C overnight slate ran the symmetric-router half.
- **P11**: 4-run portfolio (MILD / HEAVY / HEAVY_DEEPLIVE / WEBCAM_HARDEN); first packet to incorporate face-size axis disruption + codec axis at the same time.
- **P12_HEAVY_LONG**: long-schedule HEAVY recipe; periodic_saves silent failure means no usable mid-step checkpoints. P12 dud.
- **P13_FROM_SCRATCH**: substrate-redesign track of Plan v4. Anchor-aware loss + pipeline-random aug + face-scale jitter, from-scratch on P8A topology. Day-4 verdict γ 2026-04-29 02:30 UTC.
- **P14_FT_FROM_P8A**: in flight as of 2026-04-29 EOD. FT-from-P8A_REFERENCE_step5000 + same three P13 anti-shortcut interventions.
- **P14_DATA_FIX**: drafted, not launched. Bucket-gap closure (yaml-only wiring of `VisoMasterEnhancedSample`).
- **P15_GRL_FROM_P8A**: drafted, not launched. Five-line yaml change from P14: `use_quality_domain_head: true` + DANN config.

### A.4. Vertex job IDs

(For agents who need to look up provenance; not load-bearing for forward planning.)

- `978674200371789824` (us-west4): P13_FROM_SCRATCH scorecard, 2026-04-29 ~02:14 UTC `JOB_STATE_SUCCEEDED`.
- `8345130870696312832` (us-east1): P13 first launch (silent-DISABLED interventions, cancelled).
- `99039952980934656` (us-east1): P13 first-launch sibling (also cancelled).
- P14_FT_FROM_P8A: in flight at end of Slice 7; checkpoint map `arena/checkpoint_maps/teams_target_domain.p14_ft_from_p8a_2026-04-29.yaml` (untracked).

---

## Appendix B: Citation index

For citation discipline. Every claim in this document with a `file:line` or `file:section` cite traces back to one of these wiki surfaces. Memory entries and external handoffs are referenced only when explicitly cited in the wiki.

### B.1. Wiki orienting surfaces

- `docs/packet_retrospectives/AGENTS.md` (read/update protocol; lines 42-110 are the working-agent read protocol)
- `docs/packet_retrospectives/STATE_2026-04-29.md` (orienting surface; key sections at :74-77 (frame-level AUC), :80-86 (corrected-policy recall), :91-100 (P13 verdict), :104-122 (top-priority loops), :128-138 (decisions deferred), :161-170 (recommended next actions))
- `docs/packet_retrospectives/OPEN_LOOPS.md` (regenerated 2026-04-29; mechanical inventory)
- `docs/packet_retrospectives/TIMELINE.md` (chronological master index, append-only)

### B.2. Threads cited

- `threads/contract_policy_bug.md` (flagship thread; lines 7-23 attempts #1-3 chronology, :29-66 Slice 7 v3 fix subsection, :108-122 close criterion + whack-a-mole pattern)
- `threads/processing_signature_shortcut.md` (lines 5 question, :13-16 origin evidence, :18-26 Slice 5 P8A breakthrough, :28-43 Slice 6 codec_hedge + P11 + face-size, :47-76 Slice 7 frame-level AUC reframing + P13 γ verdict, :91-97 post-Slice-7 audit caveat, :147-169 close criterion sub-loops)
- `threads/eval_production_crop_tightness_gap.md` (the structurally most consequential audit finding; lines 14-22 mechanism + reproduction case, :42 close path, :62-69 open loop)
- `threads/sharpness_metric_bug.md` (lines 14-26 mechanism + reproduction, :30-44 current stance, :69-85 open loops)
- `threads/eval_substrate_data_hygiene.md` (companion thread; lines 17-21 audit findings, :28-35 current stance, :56-72 open loops)
- `threads/viso_bucket_gap.md` (lines 15-19 dispositive finding, :29-35 current stance, :61-79 open loops + frame-level AUC reconciliation, :81-87 cross-thread refs)
- `threads/identity_audit.md` (lines 18-26 dispositive identity-count table, :30-34 current stance, :47 evidence locations)
- `threads/face_size_label_leak.md` (lines 15-37 Slice 6 evidence, :43-58 current stance + Slice 7 update, :87-94 open loop)
- `threads/webcam_fpr_dominance.md` (lines 17-50 modern_v2 sweep, :64-72 current stance, :76-88 Slice 7 + audit caveats, :107-117 open loop)
- `threads/in_proj_svd_gradient_bug.md` (lines 13-22 discovery + Phase C readout, :23-30 current stance, :53-69 open loops)
- `threads/wandb_flattening.md` (lines 17-37 Slice 6 incident, :39-46 current stance, :69-94 open loops + cross-thread refs)

### B.3. Packet retros cited

- `packets/P13.md` (status card :7-16, results :54-93, conclusions :95-114)
- `packets/P14.md` (lines 6-16 status card, :22-56 two variants spec + Move 1 pre-spend gate)
- `packets/P15.md` (lines 7-16 status card, :20-66 readiness note + decision logic)

### B.4. Memory entries referenced (cited only via wiki)

- `project_p8a_frame_level_auc_2026-04-29.md` (Slice-7 reframing anchor)
- `project_contract_policy_bug.md` (auto-memory anchor for the whack-a-mole pattern)
- `project_signature_shortcut_finding.md` (Slice 4 origin)
- `project_p8a_breakthrough.md` (Slice 5 anchor; CORRECTION block dated 2026-04-26)
- `project_shortcut_is_upstream.md` (CAVEAT block dated 2026-04-26)
- `project_face_size_label_leak.md` (Slice 6 origin)
- `project_lockbox_fpr_dominated_by_webcam_mode.md` (Slice 6 webcam-mode finding)
- `project_wandb_flattens_nested_dicts.md` (the rule + 14-key allowlist)
- `project_in_proj_svd_gradient_bug.md` (the bug doc)
- `project_data_inventory_identity_diversity.md` (Slice 7 identity audit)
- `project_clean_teams_same_identity.md` (Slice 7 paired-transport)
- `project_viso_train_eval_bucket_gap.md` (Slice 7 dispositive finding)
- `project_promotion_contract.md` (the deployment readout the contract is supposed to gate)
- `feedback_decision_points.md` (user reserves judgment calls)
- `feedback_no_cancelling_vertex_jobs.md` (cancellation needs explicit user OK)
- `feedback_promotion_contract_launch.md` (env vars)
- `reference_image_rebuild.md` (`./dev.sh build-prod -y` auto-bumps VERSION)

### B.5. External handoffs referenced (cited only via wiki)

- `docs/relaunch_handoffs/R13_RELAUNCH_PACKET7_CAMERA_SIGNATURE_HANDOFF_2026-04-24.md` (2-camera controlled test, 2026-04-24)
- `docs/relaunch_handoffs/P13_DAY4_VERDICT_2026-04-29.md` (P13 γ verdict, 2026-04-29 02:30 UTC)
- `docs/relaunch_handoffs/TRAINING_EVAL_VISO_BUCKET_GAP_2026-04-29.md` (bucket-gap diagnosis, 2026-04-29 ~10:00 CEST)
- `docs/relaunch_handoffs/P15_GRL_READINESS_NOTE_2026-04-29.md` (P15 readiness + decision matrix)
- `docs/relaunch_handoffs/WANDB_FLATTENING_BUG_HANDOFF_2026-04-28.md` (the wandb-flattening incident writeup)

### B.6. Source files referenced (cited only via wiki)

- `arena/score_teams_promotion_contract.py:456-479` (`_threshold_sort_key`; the buggy + working-tree-fixed function)
- `arena/score_teams_promotion_contract.py:510-531` (`_promotion_summary_sort_key`; cross-checkpoint ranker added in v3)
- `arena/run_target_domain_validation_sequential.py` (runner; +19 lines new CLI args)
- `arena/checkpoint_maps/teams_target_domain.r13_rlp5_07_sanity_2026-04-23.yaml` (slot-07 sanity carrier)
- `data/sources/visomaster.py:605-890` (`VisoMasterEnhancedSample` loader)
- `data/sources/combined_paired.py:646` (`create_unified_samples_from_visomaster_enhanced`)
- `data/sources/combined_paired.py:65-82` (`QUALITY_DOMAIN_MAP`)
- `train_sweep.py:172-282` (pre-`c366026` allowlist; 11 nested-dict blocks)
- `train_sweep.py:283-313` (post-`c366026` allowlist; 14 keys total)
- `train_sweep.py:322-349` (P15 quality-domain fail-fast guards)
- `tests/test_train_sweep_reapply_allowlist.py` (Slice 6 regression test guard)
- `tests/test_score_teams_promotion_contract.py` (working-tree v3 fix tests; +125 lines)
- `tests/test_visomaster_enhanced_wiring.py` (untracked; 3 wiring tests)
- `tests/test_unpaired_reals_and_grl.py` (P15 unit tests)
- `analysis/lockbox_tagging/layers/quality.py:80-82` (sharpness-metric full-image bug)
- `analysis/lockbox_tagging/full_tags_2026-04-27.parquet` (the substrate parquet; n=7,334 dev+lockbox tagged frames)
- `analysis/modern_lockbox_v2_2026-04-27/build_modern_subset.py` (v2 filter sweep)
- `analysis/probe_battery_2026-04-26/grad_audit.py` (in_proj-SVD regression guard)
- `detectors/effort_detector.py:236-274` (quality-domain head + GRL hook)
- `detectors/effort_detector.py:906-958` (trainer loss aggregation for GRL)
- `scripts/launch/check_image_currency.sh` (pre-launch image-currency guard, commit `ad76cd8`)
- `experiments/phase2_round13/R13_P13_FROM_SCRATCH.yaml` (P13 yaml, commit `cab2909`)
- `experiments/phase2_round13/R13_P14_FT_FROM_P8A.yaml` (P14_FT yaml, in flight)
- `experiments/phase2_round13/R13_P14_DATA_FIX.yaml` (P14_DATA_FIX yaml, untracked)
- `experiments/phase2_round13/R13_P15_GRL_FROM_P8A.yaml` (P15 yaml, untracked)

### B.7. Commits cited

- `c7dc828` (periodic_saves trainer code + yaml schema, no allowlist update)
- `cab2909` (P13 anti-shortcut interventions including face_scale_jitter)
- `c366026` (the wandb-flattening fix; 3 new keys + test guard)
- `6c9a320` (image 1.3.226 bump after `c366026`)
- `2feea58` (in_proj-SVD silent zero-gradient fix)
- `7af72b1` (anchor-pool monitor + grad_audit + manifest_overlap)
- `ac83ba1` (VERSION bump 1.3.216 = first image with in_proj-SVD fix)
- `872502c` (Slice 3 train_sweep value_composite allowlist patch)
- `f366368` (Slice 5 wandb artifact name >128 chars fix)
- `855871e` (WS-P0 preprocessing parity fix)
- `deac44e` (WS-P1 calibration probe + WS-P2.b per-identity reducer)
- `ad76cd8` (pre-launch image-currency guard)
- `2c9778b` (P10 anti-shortcut packet authored; six GRL yamls drafted, never launched)

### B.8. Citation discipline note

This document was written from the wiki only. No source files were re-surveyed independently of wiki citations. Where a claim required a file:line check (e.g., the contract-policy v3 fix's exact line numbers), the wiki's cited surface was used directly without independent re-derivation. If a claim in this document does not trace back to a wiki cite, it should be considered inferential and flagged on the next pass.

---

## Appendix C: Read order for a fresh agent

Per `AGENTS.md:42-58`, the working-agent read protocol:

1. Read `AGENTS.md` (this protocol may have evolved).
2. Read `OPEN_LOOPS.md` (current open-issue inventory).
3. Read the last ~10 entries of `TIMELINE.md` (recent session arc).
4. Identify the relevant thread(s) for the user's task. Read at minimum the `## Current stance` and `## Open loops` sections of each.
5. Run a working-tree-diff check (`git status` + `git diff --stat`) before proposing or applying any fix.
6. Then read the user's specific task.

For an agent picking up the R13 forward plan specifically, recommended additional read after the protocol:

- `STATE_2026-04-29.md` (the orienting surface).
- This document (`MASTER_PLAN_2026-04-29.md`) — the bird's-eye write-up.
- The compressed `PLAN.md` once it lands (≤ 400 lines per user spec; the adversarially-refined target).
- Threads `contract_policy_bug.md`, `processing_signature_shortcut.md`, `viso_bucket_gap.md` (the three load-bearing threads for the current decision tree).
- Packet retros `P13.md`, `P14.md`, `P15.md` (the three most recent).

---

## Appendix D: What this document does not cover

For honest scope-of-applicability:

- **Pre-Slice-1 history.** The wiki's TIMELINE starts 2026-04-17; this master plan does not re-derive Phase 2's pre-relaunch trajectory.
- **Source-code review.** The master plan cites file:line via the wiki; it does not independently audit source-code correctness. Codex's `/claudex:plan` adversarial pass is the next surface that does.
- **Deployment-server status.** The deploy server at `http://34.16.217.28:8999` (loop `deploy-server-preprocessing-drift`, medium) is referenced via the wiki but not addressed in any forward priority. It is a separate workstream.
- **Production capture pipeline specifics.** The eval-vs-production crop-tightness gap is acknowledged but not quantified. Move 1.5 is the proposed path; this master plan does not predict the result.
- **Identity-split methodology for P14_DATA_FIX.** Named as a user judgment call; the master plan does not propose a specific split policy.
- **Cross-method generalization (FaceDancer blind spot, etc.).** `processing_signature_shortcut.md:131` flags this as an open question. No packet has attacked FaceDancer pipelines directly.

These are scope omissions, not contradictions. A future working-agent session may need to address one or more before the next packet decision lands.

---

## Appendix E: Closing notes

This document is the long-form draft. The user will compress it to ≤ 400 lines as `PLAN.md` in the project root, then run `/claudex:plan --from-draft --rounds 2 "R13 forward plan"` to grill it adversarially. The refined `PLAN.md` will then update this master.

**Compression discipline (for the user's next pass):**

- Preserve all decisions (every entry in Sections 11 and 12).
- Preserve all citations (Appendix B is the audit trail).
- Preserve all numbered open-loop close paths (Section 8).
- Preserve the decision tree intact (Section 12).
- Drop exposition. Drop redundant restatements. Drop the appendices except the citation index.

**Adversarial-refinement expectations:**

The Codex pass should call out:

- Citation gaps (claims tagged `[uncited]` or claims without explicit cites).
- Internal inconsistencies (e.g., if Section 8's close path for a loop disagrees with the loop's close criterion in `OPEN_LOOPS.md`).
- Risk-register items the master plan named but the decision tree didn't address.
- Open questions for the user that the master plan named but didn't sequence.

**Folding back into the master:** the refined `PLAN.md` becomes the canonical short form; this master plan becomes the audit-trail companion. Both live under `docs/packet_retrospectives/plans/`. A future agent reading the canonical short form should be able to drop into the master plan for any expanded justification.

---

*End of document.*

*Wiki source: `docs/packet_retrospectives/` as of 2026-04-29 afternoon. Date-stamped: 2026-04-29. Author session: Claude Opus 4.7 via plan mode + ExitPlanMode. Length: ~2000 lines (target 1800-2200).*
