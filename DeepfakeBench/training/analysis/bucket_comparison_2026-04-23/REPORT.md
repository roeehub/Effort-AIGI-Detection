# Bucket comparison — why the model isn't generalizing (2026-04-23)

**Question**: A model trained on DeepLive + proper_visomaster doesn't generalize to
the `teams_ood` / `wma_failure` OOD pools that the `value_composite` gate grades
against. Is that a data-distribution problem, a weight-imbalance problem, or
something else?

**Method**: sampled ~15–60 frames each from every source referenced by the
packet-5 yaml + the candidate `visomaster-enhanced-face-cropped-v2` bucket the
user asked me to include. Computed per-image stats: resolution, file size,
luminance, luminance std, Laplacian-variance sharpness, Canny edge density,
high-pass-residual noise std, and an 8-boundary blockiness proxy. 443 images
analyzed in total. Full per-image data in `per_image_stats.csv`; summary in
`per_source_summary.csv`; thumbnail grid in `thumbnails_grid.png`.

**One-sentence conclusion**: the packet-5 training set is drawn from a narrow
224²-resolution / moderate-sharpness distribution, while the gate grades the
model against pools whose medians differ in resolution (up to 342×436), sharpness
(5–10× lower for `wma_failure`), and brightness (15–40 luma points higher). The
model is being asked to generalize across a structural distribution gap that no
amount of margin/weight tuning is going to close by itself.

---

## Critical corrections to prior assumptions

1. **`quality_enhancement` is NON-enhanced** despite the name (user correction
   2026-04-23). It goes under the `deeplive_non_enhanced_fake` family weight,
   not `deeplive_enhanced_fake`. Any previous reading that assumed the label
   matched the family was wrong.
2. **The DeepLive bucket contains 9 `visomaster_*` method-family sessions that
   are currently excluded from training** (`visomaster.enabled: false` in the
   yaml). By the sharpness metric, the real frames from these sessions are
   **the second-sharpest user-created source** (Laplacian 196, only beaten by
   external_vcd_real at 207). Unused high-quality training data.
3. **`proper_visomaster_enhanced_clean` is not in training AND not in eval** —
   1484 user-created captures sitting idle.
4. **The Teams-v2 bucket also has a visomaster_* method family** that gets
   bundled under `deeplive_teams_*` family weights — it is the smallest-
   resolution user-created source (177×177 median).

---

## Per-source stats

| Source | n | res (median) | file kB | luma | sharp (Laplacian) | edges | noise |
|---|---:|---|---:|---:|---:|---:|---:|
| **DeepLive bucket (user-created, in training)** | | | | | | | |
| `deeplive_non_enh_fake` | 16 | 224² | 77 | 98 | **55** ← low | 0.040 | 4.3 |
| `deeplive_enh_fake` | 16 | 224² | 76 | 109 | 112 | 0.039 | 5.1 |
| `deeplive_non_enh_real` | 16 | 224² | 70 | 94 | 83 | 0.047 | 5.3 |
| `deeplive_enh_real` | 15 | 224² | 58 | **157** ← bright | 116 | 0.042 | 6.1 |
| **DeepLive bucket (user-created, UNUSED)** | | | | | | | |
| `dl_bucket_visomaster_fake` | 16 | 224² | 73 | 111 | 103 | 0.039 | 5.6 |
| `dl_bucket_visomaster_real` | 16 | 224² | 68 | 95 | **196** ← very sharp | 0.051 | 6.7 |
| **Teams-v2 bucket (user-created, in training + OOD — bundled `deeplive_teams_*`)** | | | | | | | |
| `tv2_deeplive_fake` | 21 | 214² | 60 | 125 | 86 | 0.041 | 5.6 |
| `tv2_visomaster_fake` | 15 | **177²** ← smallest | 46 | 151 | 161 | 0.054 | 7.3 |
| `tv2_deeplive_real` | 20 | 213² | 67 | 109 | 71 | 0.041 | 4.8 |
| `tv2_visomaster_real` | 31 | 210² | 59 | 144 | 160 | 0.047 | 6.8 |
| **proper_visomaster manifest lanes** | | | | | | | |
| `proper_visomaster_clean_fake` | 16 | 224² | 69 | 103 | **56** ← very smooth | 0.028 | 4.2 |
| `proper_visomaster_teams_fake` | 13 | 261² | 108 | 108 | **51** ← very smooth | 0.025 | 4.0 |
| `proper_visomaster_enhanced_clean_fake` **UNUSED** | 16 | 224² | 79 | 106 | **215** ← sharpest fake | 0.056 | 6.6 |
| `proper_visomaster_enhanced_teams_fake` | 16 | 255² | 109 | 116 | 148 | 0.062 | 6.4 |
| `proper_real_clean__paired` | 16 | 224² | 80 | 111 | 193 | 0.063 | 7.4 |
| `proper_real_teams__paired` | 15 | 252² | 93 | 114 | 113 | 0.052 | 6.0 |
| **Candidate new bucket** | | | | | | | |
| `visomaster_enhanced_v2_fake` | 60 | 195² | 63 | 137 | 100 | 0.050 | 5.3 |
| **External (OOD gate targets)** | | | | | | | |
| `external_vcd_real` | 28 | 224² | 73 | 122 | 207 | 0.039 | 6.2 |
| `external_youtube_avspeech_real` | 27 | 175² | 46 | 122 | 241 | 0.063 | 7.1 |
| `wma_failure_fake` | 54 | **342×436** | **190** | 153 | **18** ← 5–10× less sharp | **0.013** | **2.5** ← much less noise |

See `thumbnails_grid.png` for a visual row-per-source grid (6 exemplars each).

---

## What the numbers say

### 1. Resolution clusters into three regimes

| Regime | Median res | Sources |
|---|---|---|
| **Large** (~250²) | 252–261² | proper_visomaster_teams, enhanced_teams, proper_real_teams |
| **Standard** (224²) | 224² | deeplive_*, dl_bucket_visomaster_*, proper_visomaster_clean, enhanced_clean, proper_real_clean, external_vcd |
| **Teams-small** (~190–215²) | 177–214² | tv2_deeplive_*, tv2_visomaster_*, visomaster_enhanced_v2, external_avspeech |
| **Outlier** | 342×436 non-square | wma_failure_fake |

The **Teams-v2 bucket** data (177–214²) and the **candidate visomaster_enhanced_v2**
bucket (195²) sit in the same small-resolution cluster. That's what the user
meant by "the Teams captures are smaller / from an old pipeline." It isn't a
subset of teams-v2 — the whole bucket is in this cluster. Training mixes
"standard 224²" and "teams-small 190–214²" images; the model implicitly learns
resolution cues, which leak into `worst_pool_fpr` when it hits an OOD pool at a
third resolution regime.

### 2. `wma_failure_fake` is structurally broken for this gate

- 342×436 median (not square, not a face crop in the same sense as everything
  else — probably full frames or different crop logic)
- Laplacian variance **18** — everything else is 51–241. 5–10× less sharp.
- Noise std **2.5** — everything else is 4–7.4. Half the natural-image noise floor.
- Edge density **0.013** — everything else is 0.025–0.063.

Every metric points to the same thing: this data has been heavily smoothed /
upscaled / denoised, and it isn't a face crop in the same framing as the
training data. **The `value_composite` gate uses this as the fake-side OOD
driver.** So the metric that blocks 6 of 8 packet-5 runs is being computed
against a pool the model has never seen anything resembling.

### 3. proper_visomaster non-enhanced fake lanes are oddly smooth

- `proper_visomaster_clean_fake`: Laplacian 56, edges 0.028, noise 4.2
- `proper_visomaster_teams_fake`: Laplacian 51, edges 0.025, noise 4.0

These are noticeably smoother than their paired real sources (193 and 113
Laplacian) and dramatically smoother than the enhanced variants (215, 148).
The non-enhanced fake lanes look like the generators' raw output was lightly
blurred by some downstream processing. That's a clue about what
`proper_visomaster_teams_fake` is actually teaching the model — not
"deepfake artifacts" but "generator-output-smoothing."

### 4. `proper_visomaster_enhanced_clean` (UNUSED) is the sharpest fake source

Laplacian **215**, higher than every other fake source and higher than most
reals. This is the user's best-quality fake training data and it isn't in
training. Confirms the user's instinct to add it.

### 5. `dl_bucket_visomaster_*` (UNUSED) is similarly high-quality

Nine method families sitting in the DeepLive bucket with `visomaster.enabled:
false`. The real frames show Laplacian 196 (close to the top) and the fake
frames 103 (similar to deeplive_enh). User-created, comparable quality to the
training data — currently ignored by the yaml.

### 6. Brightness is also bucket-systematic

Teams-bucket sources skew brighter (luma 108–161) than proper_visomaster
sources (103–116) and the DeepLive bucket (83–157). Teams call lighting / gain
control; this is a real domain artifact that shows up in the pixel statistics.

---

## Why the model isn't generalizing (what the data says)

1. **The model learns a bimodal resolution distribution (224² + 190–215²)** and
   no exemplars in between or outside. Hand it a 342×436 face region and the
   feature extractor encounters an image region it has never statistically
   seen — scale-space, aliasing, texture frequencies are all off.
2. **Training-fake sharpness spans 51 to 215** (4× ratio). OOD-fake target
   sharpness is 18 (another 3× gap below training). The model has no basis to
   calibrate "fakeness" at that blur level.
3. **The OOD-real targets are noisier and sharper than training reals.**
   `external_vcd_real` has Laplacian 207 + noise 6.2. `realpool_real` draws
   from sources with Laplacian 71–196 + noise 4.8–6.7. The overlap isn't bad,
   but it's far from identical.
4. **The non-enhanced fake lanes in proper_visomaster look post-smoothed.** If
   the model learns to key off "this level of residual noise = fake," it
   generalizes poorly to `wma_failure_fake` (even smoother) and OOD captures
   whose noise floor differs from training.

The `value_composite` gate reveals this in the cleanest way possible: it grades
against distributions the model has essentially zero training support for,
then rejects all checkpoints.

---

## Recommendations (for packet 6 design)

### A. Include the user-created data that's currently unused

**A1.** Add `proper_visomaster_enhanced_clean` to `include_lanes` across all E3
slots. Add `proper_visomaster_enhanced_clean_fake` family weight (match
`proper_visomaster_enhanced_teams_fake`, probably 2.0–2.5). **1484 high-quality
captures, sharpest fake source we have.**

**A2.** Flip `visomaster.enabled: true` in the yaml, include the 9
`visomaster_*` method families from the DeepLive bucket. Set a family weight
similar to `deeplive_non_enhanced_fake`. **Adds a comparable-quality source
that's currently free.**

**A3.** Register the `visomaster-enhanced-face-cropped-v2` candidate bucket
(2073 fakes across 16 enhancer families). It sits in the Teams-small
resolution cluster, so it extends training coverage of the 190–215² regime.
Add under a new family weight, e.g., `visomaster_enhanced_v2_fake: 1.5`.

### B. Fix the OOD gate before any further training

**B1.** `wma_failure_fake` should not be a driver of `worst_pool_fpr`. Either
re-crop it to match the training face-crop framing (so Laplacian rises to the
training range), or drop it from the hard gate and keep it as a
diagnostic-only monitor. Current state: the gate's primary blocker is a pool
the model has no training basis to classify.

**B2.** Replace / supplement the OOD fake pool with identity-held-out slices
of the four `proper_visomaster_*` lanes. Each lane has an identity split key
in the manifest; a 20–30% identity-held-out subset gives ~70–450 captures per
lane — enough to drive a gate signal grounded in training-like distributions.

### C. Quality-normalize the non-enhanced proper lanes

The `proper_visomaster_clean_fake` and `proper_visomaster_teams_fake` lanes
look like they've been post-smoothed (Laplacian 51–56 vs everything else ≥
70). If that's an artifact of the generation / crop pipeline, a re-crop from
the source videos with matched processing to the real lane should restore the
sharpness. If it's intentional (anti-aliasing in the generator), fine — but
the family weight (currently 1.0–2.5) should probably drop, since this data
is teaching the model "smoothing = fake."

### D. Address the resolution gap

Train with an augmentation that randomly downscales+upscales by a factor
uniform in [0.6, 1.0] BEFORE the 224² resize. Every source today is fed
through a resize-to-224 step; without this variation, the model doesn't see
the resolution-invariance it needs to generalize to the OOD 190–214² pools.
Cheap to add, likely to help all three distribution gaps simultaneously.

### E. Identity-held-out test splits for every user-created lane

Per the user's "test properly" ask: carve 20–30% of identities out of each
of the four `proper_visomaster_*` lanes (and the new `dl_bucket_visomaster_*`
lanes if included) and add them as OOD pools. This gives the gate a fake-side
signal tied to distributions the model is trained on.

---

## Files in this analysis

- `sample_and_analyze.py` — sampler + stats script (reproducible with `SEED=737`)
- `make_thumbnail_grid.py` — renders a per-source thumbnail grid
- `per_image_stats.csv` — per-image raw stats (443 rows)
- `per_source_summary.csv` — per-source aggregated stats
- `thumbnails_grid.png` — visual grid for eyeballing differences
- `cache/<source>/*.png` — the sampled frames themselves (kept for inspection)

## Caveats / what this analysis does NOT do

- **No embedding-space comparison.** Image-level stats catch resolution / noise /
  sharpness gaps but not semantic/style gaps. A CLIP or trained-detector
  embedding + FID/MMD would strengthen the case. Worth doing before committing
  to packet 6.
- **Small samples per source (15–60).** Medians are stable but tail behavior
  isn't measured. Good enough to surface structural gaps; not a substitute for
  full-distribution analysis.
- **Doesn't directly probe "old vs new capture system" within teams-v2.** The
  whole bucket is in the small-resolution cluster; whether there's further
  sub-clustering by session number / timestamp wasn't tested.
- **JPEG quantization proxy is 0 everywhere** because every source is PNG.
  File-size-per-pixel is the closer-to-useful compression signal but wasn't
  tabulated here.
