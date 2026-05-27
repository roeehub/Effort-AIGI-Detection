# Training/Eval Distribution Gap on Viso — Dispositive Finding

**Generated**: 2026-04-29 ~10:00 CEST (Roee + Claude Opus 4.7)
**Branch**: `teams-relaunch-root-2026-04-17`
**Status**: Diagnosis complete; no code changes yet. Awaits user authorization for next steps (data plumbing or empirical confirmation probe).

---

## TL;DR

The headline `visomaster_enhanced_macro_dev` recall failure is largely a **train/eval distribution mismatch**, not a model-capacity or shortcut-only failure.

- **Eval `visomaster_enhanced_macro_dev` fake frames** come from `gs://teams-faces-data-test-2914-fake-4420-real-feb-28/fake/` — viso fakes that have been **GAN-enhanced** (codeformer / GFPGAN / GPEN-* / restoreformer / vqfr) **and recaptured through Teams**.
- **Training viso fakes** come from `gs://live-deepfake-methods-real-and-fake-frames-cropped/samples/visomaster_*` — **no enhancers** (the cache shows `enhancers: {'none': 960}`) and **no Teams transport** (the live bucket is direct-recorded).

The deeplive symmetric metric (`deeplive_enhanced_dev`) *also* uses the same Teams-recaptured eval bucket, but training **does** include both enhanced deeplive (`edge_cases_enhanced`, `minimal_processing_enhanced`) and Teams-transport deeplive (`deeplive_teams_*` strategies). This perfectly predicts the ~3–5× asymmetry between viso recall and deeplive recall on every candidate to date.

This finding does **not** invalidate the camera-signature shortcut memory or the inverse-correlation read. It does reframe the recipe-tuning effort: most of the past three weeks has been trying to make the model *generalize* from one viso pipeline to a strictly disjoint viso pipeline through augmentation. There is a dramatically cheaper move: include the eval pipeline in training.

---

## 1. Identity diversity — refutes the "N is too small" worry

A prior diagnosis path proposed that the viso/deeplive substrate might be too narrow at the identity axis (e.g., N≈8 per method) to expect cross-domain generalization. Hard counts from local discovery caches (`.viewer_cache/discovery/*.json`, last refreshed 2026-04-17) show otherwise:

| Bucket | Source | Samples | Unique identities | In P13_FROM_SCRATCH training? |
|---|---|---:|---:|:---:|
| `live-deepfake-methods-real-and-fake-frames-cropped` | viso (live, no enhancers) | 960 | **429** | ✅ |
| `live-deepfake-methods-real-and-fake-frames-cropped` | deeplive (3 strategies + 2 enhanced) | 5,800+ | **1,916** | ✅ |
| `live-...-teams` (1) | teams (deeplive_teams_* + viso_hints_teams) | 2,626 | 1,306 | ✅ partial (deeplive_teams yes; viso_hints_teams no) |
| `live-...-teams-v2` | visomaster_teams_v2_companion (9 swap × 8 enhancers) | 1,994 | **997** | ❌ |
| `visomaster-enhanced-face-cropped-v2` | viso enhanced (direct, no teams) | 2,073 | many (manifest-hashed) | ❌ |
| `hdtf_visomaster_cropped_frames` + `_teams` | proper-data wave 2026-04-19 (clean+teams pairs) | 1,322 captures | **705** | ❌ |

**Active training has ~430 viso identities and ~1,900 deeplive identities. ~2,300 additional identities sit in buckets that training never touches** — including the bucket whose distribution the eval suite measures recall on.

(1) — The teams source's swap-model field is empty for the deeplive_teams family in the cache; teams-transport visomaster (the `visomaster_hints_teams` 404-sample slice) is in the cache but is not the same as the `teams_v2_companion` bucket described above.

---

## 2. The smoking gun — eval bucket vs training bucket

### 2.1 Where the verdict's `viso_dev_recall` reads

`analysis/p13_day4_verdict_2026-04-29/compute_verdict.py:51,335`:

```python
SUITE_NAMES = ["visomaster_enhanced_macro_dev", ...]
viso = float(srow["visomaster_enhanced_macro_dev__fake_recall"])
```

`arena/target_domain_suites.r13_best_megaval_2026-04-13.yaml:44–49`:

```yaml
- name: visomaster_enhanced_macro_dev
  external_fake_manifest: "arena/manifests/teams_target_domain_manifest_2026-04-06_frozen.json"
  external_fake_manifest_split: "dev"
  external_fake_manifest_slices: "visomaster_enhanced_macro"
```

First `visomaster_enhanced_macro` slice item from that manifest (550 videos in this slice):

```
video_id:    visomaster_enhanced_raw__seq12349__fake
method:      visomaster_enhanced_macro
frame_path:  gs://teams-faces-data-test-2914-fake-4420-real-feb-28/fake/visomaster_enhanced_raw__frame_001595_seq12349.png
```

These are GAN-enhanced viso fakes that have been **transmitted through Teams** (the bucket name encodes "feb-28" recapture context). The model's job is to detect them as fake.

### 2.2 What training viso actually contains

`experiments/phase2_round13/R13_P13_FROM_SCRATCH.yaml:163–177`:

```yaml
visomaster:
  enabled: true
  gcs_bucket: "live-deepfake-methods-real-and-fake-frames-cropped"   # NOT the enhanced bucket
  swap_models: [CSCS, GhostFace-v1/v2/v3, InStyleSwapper256-A/B/C, Inswapper128, SimSwap512]
  tiers: null                                                          # NO enhancer dimension
```

Local cache `.viewer_cache/discovery/visomaster.json` (the 960-sample live snapshot):

```python
Counter(s.get('enhancer','') or 'none' for s in viso) == {'none': 960}
```

There is no Teams-transport viso source enabled in this yaml at all — the `teams:` block exists but feeds `deeplive_teams_*` (no `visomaster_*_teams` family is defined as a training source).

### 2.3 The deeplive control case

The same eval suite has `deeplive_enhanced_dev`, also reading from `gs://teams-faces-data-test-2914-fake-4420-real-feb-28/fake/`. Training **does** include matching distributions:

- `combined_paired.deeplive.include_strategies` includes `edge_cases_enhanced` and `minimal_processing_enhanced` (post-enhancer fake variants)
- `combined_paired.teams` is enabled with bucket `live-deepfake-methods-real-and-fake-frames-cropped-teams`, which is the Teams-transport companion to deeplive

So **deeplive's training distribution is approximately matched to its eval distribution**, and **viso's is not**. This is precisely consistent with every candidate's measured asymmetry between viso and deeplive recall (a 3–5× gap that survives every recipe tuning, every ensemble, and every τ-policy).

---

## 3. What this implies for the running and queued experiments

| Experiment | Hypothesis being tested | Status under this finding |
|---|---|---|
| P14 (FT-from-P8A + interventions) | "Anti-shortcut bundle becomes additive on a working base" | Tests a real but secondary effect. The bucket gap is unchanged, so cross-domain viso recall is bounded above by the same data mismatch. Most likely outcome: viso recall ≈ P8A's, slight Axis 3 movement. Verdict will look like P13's even if the bundle works. **Do not over-interpret.** |
| P15 GRL (drafted) | "Adversarial pipeline classifier removes the camera-signature shortcut" | Still useful, but expected gain is much higher *after* the eval distribution is in the training set. GRL on training-only pipelines does not teach the model the eval pipeline. |
| ArcFace identity audit | "Training-set labels may be identity-corrupt" | Independent and still worth completing. The 2026-04-27 partial audit (`analysis/identity_corruption_audit_2026-04-27/`) found only 1 of 5 audited labels multi-identity (PC_Generator__s15) and ~3.5% frame-level corruption. Low likelihood of being load-bearing. |
| Contract policy bug fix | "Contract picks τ-degenerate winners (project_contract_policy_bug.md)" | Orthogonal but still required for any future scorecard read to be trustworthy. |

---

## 4. The concrete unused data, and the cost to use it

### 4.1 Already-built loaders we are not using

`data/sources/visomaster.py:605–890` defines `VisoMasterEnhancedSample` with a dual-bucket structure (`original_bucket` for real frames + `enhanced_bucket` for enhanced fake frames; default `enhanced_bucket = "visomaster-enhanced-face-cropped"`). The arena builder `arena/build_visomaster_enhanced_v2_manifest.py` and the manifest `arena/manifests/visomaster_enhanced_v2_manifest_2026-04-13.json` (2,073 frames across 9 swap models × 8 enhancers) are already produced. **These are not wired into `combined_paired.py` for current training.**

### 4.2 Already-built data not in any training run

| Asset | Identities | Methods | Pipeline match to eval? |
|---|---:|---|---|
| `live-...-teams-v2` (visomaster_teams_v2_companion source) | 997 | 9 swap × 8 enhancers each | **Direct match** — enhanced + Teams transport |
| `arena/manifests/visomaster_enhanced_v2_manifest_2026-04-13.json` | many | 9 swap × 8 enhancers | Partial — enhanced but not Teams-transported |
| `hdtf_visomaster_cropped_frames` + `_teams` (proper-data wave) | 705 | 9 swap × 8 enhancers | **Direct match** — same-identity clean + Teams pairs |

### 4.3 Plumbing scope (estimate)

- Add a new family `visomaster_enhanced_fake` in `data/sources/combined_paired.py` (mirrors the existing `visomaster_fake` family wiring). Reuses the existing `VisoMasterEnhancedSample` class — no new loader class required. ~4–8 hours.
- Update `R13_*.yaml` to enable `visomaster.enhanced_enabled: true` and add a `family_weights.visomaster_enhanced_fake` entry. ~30 minutes.
- Image rebuild (auto-bumps VERSION). ~5 minutes build + 3 minutes push.
- Smoke test: load 16 enhanced viso samples in a CPU dataloader, confirm shapes/labels. ~30 minutes.
- Training run: ~$60 / ~8h on A100 us-east1.
- Scorecard: ~$10 / ~3h.

**Total: ~1–2 days, ~$70.** Far cheaper than another P11/P12-recipe variant, and far more likely to move the eval-headline metric.

---

## 5. Open questions before acting

1. **Is the eval substrate a fair production proxy?** `visomaster_enhanced_macro_dev` is 550 videos all from one Teams-recapture session. We are about to put it in training. That is fine for closing the train/eval gap on the headline metric, but it means we should split it into a train slice and a held-out eval slice carefully (split by identity, not by frame), or it stops being a deployment proxy.
2. **Does the proper-data wave (`hdtf_visomaster_*`) supersede the `teams_v2` companion bucket as the training source?** Both contain enhanced+teams viso. Proper-data is purpose-built and identity-paired (clean+teams variants of the same identity); `teams_v2_companion` has more swap models and more enhancers but lacks the explicit clean-pair link. Probably the right move is to use both, with proper-data driving same-identity-pair losses if any.
3. **Do we cancel P14 once Move 1 (the linear-probe confirmation) confirms the gap?** P14 is testing a separate hypothesis that remains valid; letting it finish is a defensible $60. But once we know the bucket gap is dominant, P14's outcome is largely predicted ahead of time.
4. **Does the enhanced-viso wiring need any new augmentation, or does the existing pipeline cover it?** Enhanced fakes have different chroma statistics than direct-swap fakes. The pipeline-randomization aug (Plan v5) was designed for general pipeline jitter; it should work, but a quick visual-smoke pass is warranted.

---

## 6. Recommended action sequence (in priority order)

1. **Move 1 — empirical confirmation (today, ~$0, no GPU).** Run a frozen-feature linear probe of P8A on the eval bucket (`gs://teams-faces-data-test-2914-fake-4420-real-feb-28/fake/`) vs the training bucket (`live-...-cropped` viso). If train-bucket viso AUC is materially higher (>0.80) than eval-bucket viso AUC (~0.65), the bucket gap is confirmed dispositive. If not, the gap is real but not the dominant cause and we keep prioritizing P15 GRL.
2. **Move 2 — wire enhanced viso into training (1–2 days, ~$70).** Add `visomaster_enhanced_fake` family using existing loader, retrain from P8A (light FT, ~3K steps). Call it `P14_DATA_FIX`. Score on the same triple-axis verdict.
3. **Move 3 — superpose anti-shortcut interventions on top of `P14_DATA_FIX`.** Once the data gap is closed, the anchor-aware loss + pipeline-rand + face scale-jitter trio should regain its expected effect.
4. **Move 4 — same-identity pair training using proper-data wave.** This becomes a real lever once the basic bucket gap is closed, because the model will then have multi-pipeline samples for the same identity (the actual ground truth for "same person, different rendering").
5. **Move 5 — only if Moves 2–4 still leave Axis 3 Δ > 0.30 on a working candidate**: revive P15 GRL with the broader pipeline label set the new data enables.

---

## 7. Pending commits (require explicit OK)

- `docs/relaunch_handoffs/TRAINING_EVAL_VISO_BUCKET_GAP_2026-04-29.md` (this file)
- `april-26-training-master-plan-v2.LOG.md` (session entry — being added in this batch)
- Memory files in `~/.claude/projects/.../memory/` (auto-memory; not in repo)
- All prior pending items from `HANDOFF.md` §"Pending User Decisions"

No code touched. No data wired. No experiments launched. This document is the basis for the next user decision; nothing escalates without authorization.
