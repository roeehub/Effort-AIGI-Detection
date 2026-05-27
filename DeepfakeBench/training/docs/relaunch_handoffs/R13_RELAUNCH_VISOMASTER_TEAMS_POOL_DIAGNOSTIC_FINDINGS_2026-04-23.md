# R13 Relaunch — VisoMaster Teams-Pool Diagnostic Findings

**Authored**: 2026-04-23 ~01:00 UTC
**Branch**: `teams-relaunch-root-2026-04-17` (HEAD `c7ac78c`; all diagnostic artifacts uncommitted)
**Audience**: The agent writing the next experiment-planning packet (Packet 5)
**Source checkpoint under test**: `gs://training-job-outputs/phase2r13_experiments/w92amaaa/value_composite_effort_20260422_step2500_auc0.9895_eer0.0116.pth` (slot 05 of Round 13 Packet 3 — FT_proper_low_arcface_spatial)

---

## 1. TL;DR

Prior chat built a local diagnostic (`analysis/compare_teams_pools.py`) to compare the `teams_ood` pool against the `proper_visomaster_teams` pool for the P3 slot-05 checkpoint. Its first run (30 videos × 4 frames per group) produced an alarming headline: the pool summary suggested a ~70% per-frame misclass rate on `proper_visomaster_teams_fake`, even though the slot-05 training yaml lists `proper_visomaster_teams_fake` with `family_weight = 1.0`.

Today's investigation shows the headline was inflated by a **sampling bug**. The GCS bucket `gs://hdtf_visomaster_cropped_frames_teams/samples/` stores frames for **two different manifest lanes** — the `proper_visomaster_teams` lane (included in slot-05 training) and the `proper_visomaster_enhanced_teams` lane (explicitly excluded by the slot-05 yaml's `include_lanes`). The diagnostic samples folders uniformly from the bucket, so on the 30-vid run 63% of the "fakes" came from the lane the model was never trained to detect.

After fixing the lane accounting post-hoc and running a full-scale 150 × 8 diagnostic, the true picture is:

| Pool / lane | Per-frame accuracy (p ≥ 0.5) | Per-video accuracy |
|---|---|---|
| `teams_ood_fake` | **91.4 %** | — |
| `teams_ood_real` | 90.4 % (← acc = 1 − FPR) | — |
| `proper_visomaster_teams_real` | **99.1 %** | — |
| `proper_visomaster_teams_fake` — **`proper_visomaster_teams` lane (IN training)** | **67.2 %** | 69.0 % (20/29) |
| `proper_visomaster_teams_fake` — `proper_visomaster_enhanced_teams` lane (NOT in training) | 18.6 % | 17.4 % (19/109) |
| `proper_visomaster_teams_fake` — not in manifest (bucket-only) | 35.5 % | 25.0 % (3/12) |

The residual finding, after lane contamination is removed, is a real but sub-catastrophic training weakness: **~67 % per-frame accuracy on the in-training `proper_visomaster_teams` fake lane vs. ~91 % on `teams_ood_fake`** — a ~22 pp gap. The identity-hold-out and preprocessing confounds are ruled out by the data.

Suggested experiments follow in § 6.

---

## 2. What the diagnostic is, in one paragraph

`analysis/compare_teams_pools.py` samples GCS frame folders, resizes and CLIP-normalizes each frame, runs the checkpoint under test for embedding + fake-prob, and renders a single HTML report with pool-level image/geometry stats, embedding centroid distances, and a failures gallery. CLI: `python analysis/compare_teams_pools.py --output-dir <dir>` with `--videos-per-group` and `--frames-per-video` (defaults 150 × 8). 17 unit tests in `tests/test_compare_teams_pools.py` all pass. Full design spec at `docs/superpowers/specs/2026-04-22-compare-teams-pools-design.md`.

Current `GROUP_DEFINITIONS` (`analysis/compare_teams_pools.py`) uses four pools keyed by bucket URI:

```
teams_ood_{real,fake}          -> gs://live-deepfake-methods-real-and-fake-frames-cropped-teams-v2
proper_visomaster_teams_{real,fake} -> gs://hdtf_visomaster_cropped_frames_teams,
                                       gs://quickclips_visomaster_cropped_frames_teams
```

Each sampled "video" is a folder in the bucket (`samples/<base_capture_id>/frames/fake/*.png`).

This bucket-keyed grouping is the root cause of the headline inflation. See § 4.2.

---

## 3. Initial headline (30 × 4 run, `scratch/teams_pool_diff/2026-04-22T23-51-00/`)

Raw per-group fake-prob stats:

| Group | N frames | mean | median | confidently-wrong (p<0.1 for fakes, p>0.9 for reals) |
|---|---|---|---|---|
| `teams_ood_fake` | 84 | 0.928 | 0.998 | 1/84 |
| `teams_ood_real` | 120 | 0.136 | 0.025 | 7/120 |
| `proper_visomaster_teams_real` | 120 | 0.004 | 0.001 | 0/120 |
| `proper_visomaster_teams_fake` | 120 | **0.301** | **0.036** | **72/120 (60 %)** |

Per-video picture (each video = 4 frames):

- **23 of 30 videos were "fully wrong"** (mean fake_prob < 0.5 across all 4 frames — often < 0.05).
- **7 of 30 videos were confidently correct** (mean fake_prob > 0.8).
- Bimodal split — not gradient noise.

The handoff doc summarizing this run quoted "30/120 = 25 % confidently wrong." That 30 is actually the gallery display cap (`per_bucket = 30`); the actual confidently-wrong rate is 72/120 (60 %).

---

## 4. Investigation log — what we ruled in and out

Working through the Resume-Instructions order from the handoff.

### 4.1 Ruled out: identity-split confound (manifest cross-reference)

The training yaml uses `identity_split_mode: hash_stable` with `train_split: 0.85, val_split: 0.10, test_split: 0.05`, seed 737. The manifest at `arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json` also carries a `split` field per video (dev / lockbox, 80/20). Cross-referencing each of the 30 sampled `video_id` values against the manifest's split:

| Manifest `split` | N sampled | predicted as FAKE | accuracy |
|---|---|---|---|
| `dev` (training-eligible) | 24 | 4 | **17 %** |
| `lockbox` (held out, never trained on) | 4 | 2 | **50 %** |
| not in manifest | 2 | 1 | 50 % |

If the held-out-identity hypothesis were true, we would expect the opposite direction: ≥ 85 % acc on dev (since hash_stable makes ~85 % of dev identities train-set) and random on lockbox. The observed direction is the inverse. **Confound rejected.**

### 4.2 Found the real confound: lane contamination (the headline bug)

The manifest records, per `base_capture_id`, which **lane** each fake variant belongs to. I checked each of the 30 sampled `base_capture_id` values. Result:

| `proper_visomaster_teams` | `proper_visomaster_enhanced_teams` | N of 30 |
|---|---|---|
| has variant | — | 9 |
| — | has variant | 19 |
| — | — (not in manifest) | 2 |

Nineteen of thirty sampled folders carry only `proper_visomaster_enhanced_teams` variants. The slot-05 training yaml is explicit:

```yaml
proper_data:
  enabled: true
  inventory_path: "arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml"
  manifest_path: "arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json"
  include_lanes:
    - "proper_visomaster_clean"
    - "proper_visomaster_teams"
```

`proper_visomaster_enhanced_teams` is **not** in `include_lanes`, and the manifest summary shows the bucket composition (by capture count):

```
proper_visomaster_teams:           342 captures  (in training)
proper_visomaster_enhanced_teams: 1484 captures  (NOT in training)
```

So a uniform sample of bucket folders is ~83 % enhanced / ~17 % unenhanced. The diagnostic's 30-vid draw hitting 19 enhanced / 9 unenhanced / 2 unknown is exactly the expected ratio.

**Recomputing per-lane accuracy on the 30-vid run:**

| Lane | N | correct | acc |
|---|---|---|---|
| `proper_visomaster_teams` (in training) | 9 | 4 | **44 %** |
| `proper_visomaster_enhanced_teams` (not in training) | 19 | 2 | 10.5 % |
| not in manifest | 2 | 1 | 50 % |

The 10.5 % on `enhanced_teams` is not a failure — the model was never asked to detect that lane. That term is responsible for most of the "70 % misclass" headline.

### 4.3 Corroborating asymmetry: why real is 100 % but fake is broken

`proper_visomaster_teams_real` is at **100 % accuracy** (mean p = 0.004) in the 30-vid run. If there were a preprocessing drift or identity-level distribution shift hitting this pool, it would affect real and fake symmetrically. It doesn't. That fact independently rules out preprocessing and distributional shift as primary explanations.

Mechanistically, the bucket contains real frames only in the unenhanced `proper_visomaster_teams` layout — there's no `real` label under the enhanced-teams lane (enhancement is applied to generated fakes, not to reals). So random bucket sampling for reals draws only from training-seen data, while random bucket sampling for fakes draws 83 % from excluded-lane data. Exactly the asymmetry we observe.

### 4.4 Full-scale run (`scratch/teams_pool_diff/2026-04-23T00-25-36/`)

Kicked off a `--videos-per-group 150 --frames-per-video 8` run to kill small-sample noise. Completed in ~27 minutes, 4,457 frames scored in total.

**Overall per-group metrics:**

| Group | N | mean p | median p | per-frame acc @ 0.5 |
|---|---|---|---|---|
| `teams_ood_fake` | 864 | 0.898 | 0.997 | **91.4 %** |
| `teams_ood_real` | 1,197 | 0.123 | 0.015 | 90.4 % (tnr) |
| `proper_visomaster_teams_real` | 1,199 | 0.016 | 0.002 | **99.1 %** (tnr) |
| `proper_visomaster_teams_fake` | 1,197 | 0.306 | 0.050 | 38.6 % (overall, **pre-lane-split**) |

**Per-lane accuracy on `proper_visomaster_teams_fake` (the key table):**

| Lane | N videos | N frames | Per-frame acc | Per-video acc | Mean p |
|---|---|---|---|---|---|
| **`proper_visomaster_teams`** (IN training) | 29 | 232 | **67.2 %** | **69.0 %** (20/29) | 0.669 |
| `proper_visomaster_enhanced_teams` (not in training) | 109 | 872 | 18.6 % | 17.4 % (19/109) | 0.201 |
| not in manifest (bucket-only) | 12 | 93 | 35.5 % | 25.0 % (3/12) | 0.376 |

**Centroid distances (cosine):**

| Pair | d |
|---|---|
| `proper_visomaster_teams_fake` ↔ `teams_ood_real` | **0.062** (closest pair of all 6) |
| `proper_visomaster_teams_fake` ↔ `proper_visomaster_teams_real` | 0.102 |
| `proper_visomaster_teams_fake` ↔ `teams_ood_fake` | 0.376 |
| `proper_visomaster_teams_real` ↔ `teams_ood_real` | 0.063 |
| `proper_visomaster_teams_real` ↔ `teams_ood_fake` | 0.716 (clean separation) |
| `teams_ood_fake` ↔ `teams_ood_real` | 0.477 |

### 4.5 Small-sample noise was partially at play too

- 30 × 4 run:  44 % acc on in-training lane (9 videos)
- 150 × 8 run: 67 % acc on in-training lane (29 videos)

~23 pp shift between sample sizes. The 150-vid number (29 videos, ~8 % of the 342-capture training lane) is far more trustworthy but still not tight. A stratified re-run pulling all 342 in-training captures would be the tightest possible estimate; ~3× more compute than the current default.

---

## 5. The real findings

1. **Training-eligible `proper_visomaster_teams_fake` accuracy ≈ 67 % per-frame / 69 % per-video.** Teams_ood_fake is 91.4 %. That ~22 pp gap is the actual signal the team should act on — it is meaningful but not emergency-level.
2. **`proper_visomaster_enhanced_teams` accuracy ≈ 18 % is expected.** Training excluded that lane. Do not treat this as a regression.
3. **`proper_visomaster_teams_real` at 99 % accuracy** demonstrates the model *can* discriminate the visomaster distribution when there's a clear real-signal; the weakness is specifically the *fake* side of this generator family.
4. **Embedding geometry: visomaster fakes cluster near real-space.** d(visomaster_fake, teams_ood_real) = 0.06 — closer than any other pair. The visomaster generators (cscs / ghostface_v{1,2,3} / inswapper128 / instyleswapper256_{a,b,c} / simswap512) produce frames the current model embeds "real-adjacent." This is structural — it tells you these generators are hard, not that the diagnostic is broken.
5. **Family-weight share for `proper_visomaster_teams_fake`**: with the slot-05 weights `{deeplive_teams_fake: 5.0, deeplive_enhanced_fake: 3.0, deeplive_non_enhanced_fake: 2.5, proper_visomaster_clean_fake: 1.0, proper_visomaster_teams_fake: 1.0, df40_fake: 0.15}`, proper_visomaster_teams_fake carries 1.0 / 12.65 ≈ 7.9 % of the fake gradient. Training saw roughly ~6,300 visomaster-teams fake frames over 2,500 steps (back-of-envelope). This is consistent with under-learning, not non-learning, which matches the 67 % empirical result.

---

## 6. Suggested experiments for Packet 5

Framed as concrete single-variable changes to the `R13_RLP3_05_FT_proper_low_arcface_spatial.yaml` baseline. None of these require data rebuilding. All can be launched from existing Vertex tooling using the same pipeline as Packet 4.

### E1 — Bump `proper_visomaster_teams_fake` family weight: 1.0 → 2.5

**Rationale**: Bring the gradient share from 7.9 % to ~17 %, roughly doubling exposure. Keeps the change minimal and falsifiable — a single-number edit.

**Change**:
```yaml
sampling:
  family_weights:
    proper_visomaster_teams_fake: 2.5   # was 1.0
```

**Hypothesis to test**: per-frame accuracy on the in-training lane moves from ~67 % → ~80 %. If it moves to ≥ 85 %, likely good enough. If it barely moves, the generator is hitting an embedding limit the model can't resolve under current architecture.

### E2 — Add the enhanced-teams lane to training

**Rationale**: 1,484 captures × (~1 method + ~8 enhancers) = a large pool currently excluded. Per the summary, `proper_visomaster_enhanced_teams` has 1,484 captures vs. 342 for unenhanced. Adding it ~5× the visomaster-teams fake data budget. Today the model fails on it (18 % acc) because it was never asked; that's also a lurking production weakness if any production call path routes through an enhanced-teams-like input.

**Change**:
```yaml
proper_data:
  include_lanes:
    - "proper_visomaster_clean"
    - "proper_visomaster_teams"
    - "proper_visomaster_enhanced_teams"   # NEW
sampling:
  family_weights:
    proper_visomaster_enhanced_teams_fake: 1.0   # NEW (tune)
```

**Caveat**: doubling the visomaster-family data may shift the deeplive-vs-visomaster balance; consider combining with a small down-weight of `deeplive_teams_fake` (e.g., 5.0 → 4.0) if compute-neutrality matters.

**Hypothesis to test**: enhanced-teams accuracy moves from 18 % → ≥ 80 %; unenhanced teams accuracy holds or improves (the two are structurally related).

### E3 — Combined E1 + E2

**Rationale**: If (E1) by itself is insufficient and (E2) by itself might cannibalize deeplive acc, the union may be the pragmatic best — wider coverage plus heavier weighting on the harder lane. Only worth running if E1 and E2 results show the two changes don't fully cover each other.

### D1 — Diagnostic patch (enabling better eval for Packet 5)

Independent of any training experiment, the diagnostic in `analysis/compare_teams_pools.py` should accept a `--manifest-path` flag and intersect sampled bucket folders with manifest-listed captures filtered by lane. Without this, the diagnostic will continue to blur in/out-of-training lanes for anyone running it against the proper_visomaster_teams pool. Estimated patch: ~50 lines in `analysis/compare_teams_pools.py` + 3 new tests. Not a blocker for launching E1/E2, but a blocker for trusting their post-run evaluation.

---

## 7. Artifacts (paths are relative to `DeepfakeBench/training/`)

| Path | Role |
|---|---|
| `analysis/compare_teams_pools.py` | Diagnostic code (~1100 lines, one file) |
| `tests/test_compare_teams_pools.py` | 17 passing tests |
| `docs/superpowers/specs/2026-04-22-compare-teams-pools-design.md` | Full design spec |
| `docs/superpowers/plans/2026-04-22-compare-teams-pools.md` | Implementation plan |
| `scratch/teams_pool_diff/2026-04-22T23-51-00/report.html` | 30 × 4 run (the one that produced the inflated headline) |
| `scratch/teams_pool_diff/2026-04-23T00-25-36/report.html` | **150 × 8 run — the authoritative numbers in § 4.4** |
| `scratch/teams_pool_diff/2026-04-23T00-25-36/raw_scores.parquet` | Per-frame fake_prob + embeddings |
| `scratch/teams_pool_diff/2026-04-23T00-25-36/stats_pass_model.json` | Per-group model metrics |
| `arena/manifests/proper_visomaster_target_domain_manifest_2026-04-19_provisional.json` | 7,304 video entries with per-video `lane` + `split` — the source of truth for lane assignment |
| `arena/inventories/proper_visomaster_wave_2026_04_19_provisional.yaml` | Wave inventory referenced by training yaml |
| `experiments/phase2_round13/R13_RLP3_05_FT_proper_low_arcface_spatial.yaml` | Slot-05 training config (see lines 150–212 for `proper_data.include_lanes` and `sampling.family_weights`) |
| `data/sources/combined_paired.py:1186-1338` | `split_samples_by_identity()` — hash-stable identity split |
| `data/sources/combined_paired.py:1274-1282` | Shuffle-mode fallback; confirms 85/10/5 split logic |

All diagnostic artifacts are **uncommitted** (`git status` shows them as untracked). Prior-session HANDOFF.md has a proposed commit bundle.

---

## 8. Known gaps and caveats

1. **Sample size on the in-training lane is modest** — 29 unique videos / 232 frames in the 150 × 8 run. A stratified re-run drawing all 342 in-training captures (8 frames each = 2,736 frames) would shrink the confidence interval on the 67 % figure meaningfully. Cheap (one-time ~15 min CPU).
2. **The 29 videos straddle the 85/10/5 identity sub-split.** Within `dev`, ~15 % of identities by hash land in val/test and are not gradient-exposed. Approximately 4 of the 29 videos are val/test. The 67 % figure is therefore a mix of "true training" and "in-dev but held out." Not corrected here.
3. **Diagnostic uses CPU single-frame inference, no augmentation.** Training uses GPU, augmentations, and multi-frame batches. Not a confound for accuracy comparisons between pools (same preprocessing across pools) but means the absolute numbers are not identical to the training-time loss surface.
4. **Checkpoint is slot-05 only** (`w92amaaa/value_composite_effort_20260422_step2500_auc0.9895_eer0.0116.pth`). Other P3 slots (01–08) may behave differently on this pool. The diagnostic accepts `--checkpoint <gs://...pth>` to re-run against another checkpoint; iterating is cheap.
5. **Diagnostic inlines `_load_state_dict_into_model`** from `retro_score_value_composite.py:94-145`. If retro_score's helper changes, the inlined copy silently drifts — worth a comment in both files or an extraction to a shared helper without the `data.sources` import chain.
6. **Identity-level training/holdout assignment within dev is not verified in this report.** I confirmed the 80/20 dev/lockbox split from the manifest; I did not confirm which of the 29 in-training-lane videos fell into train vs val vs test per the hash_stable seed 737 split. Doing so would tighten the 67 % to a "true training-set accuracy" vs "in-dev held-out accuracy" comparison and would be worth a few hundred lines of code.

---

## 9. Recommended decision flow for Packet 5

1. If (E1) is launch-ready from existing infra, launch it as the minimum-risk single-variable experiment. Expected cost: one slot of compute, ~2,500 steps.
2. In parallel, patch the diagnostic (D1). Once the E1 checkpoint completes, run the patched diagnostic against it, with lane filtering enabled, to produce a clean before/after on the same 150-vid evaluation grid.
3. If E1 moves the in-training-lane accuracy from 67 % to ≥ 85 %, Packet 5 can stop there and move on to the next axis.
4. If E1 moves it to ~75 % (partial), queue E3 (combined E1 + E2).
5. If E1 moves it barely (< 70 %), the bottleneck is likely architectural / data-quality, not weight — consider investigating specific generator methods (cscs vs ghostface vs inswapper vs simswap) separately; the 150-vid run shows cscs / ghostface_v1 / inswapper128 / instyleswapper256_a are already mostly detected at step 2500, while simswap512, instyleswapper256_c, and ghostface_v3 are the harder subset.

---

**End of report.**
