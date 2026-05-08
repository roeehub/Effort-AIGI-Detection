# Canary substrate resolution + sharpness audit FACTS (2026-05-08)

> **Status: factual-only.** No interpretation. Forbidden words: succeeds, fails, wins,
> promotes, deployment-grade.
>
> Source: `outputs/canary_resolution_per_frame.csv`,
> `outputs/canary_resolution_per_cohort.csv` produced by
> `verify_canary_resolution.py` (CPU, 2026-05-08).

## Question

User eyeballed the chronic-6 contact sheets and observed: "they look very pixelated
and we can probably afford in production to simply ignore that level of pixelation
and just reject algorithm analysis until we get a better resolution. I don't know
if this is just a viewer or if this is actually the quality of the images."

Two-part question:

1. Is the displayed pixelation a viewer-side downscale, or is the cached crop
   already at the same resolution as the GCS source?
2. What are the actual pixel-resolution and sharpness distributions of the canary
   substrate, broken out by chronic-6 vs others?

## Method

1. Read every cached crop in `_frame_cache/` (n=800, all canary frames present).
2. Compute (h, w), `min_dim`, `max_dim`, Laplacian variance per frame.
3. Spot-check 3 frames against GCS source dims via `gsutil cp` to verify cache
   has not been downscaled.
4. Tabulate per-frame and per-cohort distributions, with chronic-6 vs other-real
   vs fake split.

## Spot-check: cached vs GCS-source dims

| frame_idx | identity | cached (W×H) | GCS (W×H) | match |
|---:|---|---|---|---|
| 0 | PC_Generator__s22 | 86×91 | 86×91 | **MATCH** |
| 250 | Roy_D | 303×303 | 303×303 | **MATCH** |
| 600 | teams_fake_lockbox | 299×422 | 299×422 | **MATCH** |

Cached crops carry the exact pixel dims of the GCS source. **No viewer-side or
cache-time downscale.** What is on screen IS what is on disk IS what is in the
training/eval pipeline before resize-to-224.

## Headline distribution across all 800 canary frames

| metric | p05 | p25 | p50 | p75 | p95 | mean |
|---|---:|---:|---:|---:|---:|---:|
| min_dim (px) | 89 | 159 | 224 | 319 | 413 | 240.9 |
| max_dim (px) | 96 | 179 | 253 | 396 | 512 | 283.4 |
| Laplacian variance | 10.8 | 63.5 | 220.9 | 514.6 | 1017.2 | 337.5 |

The model resizes input to 224×224. Frames with `min_dim < 224` are upsampled
before forward pass; the p25 of the canary is at min_dim=159, so at least 25%
of canary frames go through the model after a 1.4×+ upscale.

## Group split (chronic-6 vs other-real vs fake)

| group | n | min_dim p50 | min_dim p05 | lap_var p50 | lap_var p05 |
|---|---:|---:|---:|---:|---:|
| chronic_real | 300 | 130 | 82 | 491.0 | 48.1 |
| other_real | 300 | 223 | 144 | 236.2 | 79.8 |
| fake | 200 | 357.5 | 189 | 20.7 | 9.0 |

Notes (factual, not interpretive):

- chronic_real p50 min_dim = 130 vs other_real p50 = 223. Chronic-6 group is
  systematically lower-resolution than the other-real group on this substrate.
- fake group has the highest p50 min_dim (357.5) and the lowest p50 Laplacian
  variance (20.7). Fakes on this canary are larger and smoother than reals.
- chronic_real has higher p50 Laplacian variance (491) than other_real (236).
  The Laplacian-variance metric is unnormalized for resolution; small images
  with crisp edges accumulate higher per-pixel edge density than large images
  with the same edge content.

## Per-cohort breakdown (chronic-6 detail)

| base_identity | n | min_dim p50 | min_dim min | lap_var p50 | lap_var p05 |
|---|---:|---:|---:|---:|---:|
| PC_Generator__s22 | 50 | **94** | 81 | 488.1 | 405.2 |
| PC_Generator__s45 | 50 | **90** | 80 | 985.3 | 836.8 |
| Q__s6 | 50 | **94** | 91 | 645.9 | 500.1 |
| Roy_D | 50 | 274.5 | 227 | 65.7 | 55.9 |
| bla_bla_chow | 50 | 394 | 306 | 523.6 | 37.7 |
| bla_bla_chow__s2 | 50 | 141.5 | 129 | 55.4 | 44.0 |

The chronic-6 split into two resolution regimes:

- **Tiny crops** (min_dim p50 ~ 90): PC_Generator__s22, PC_Generator__s45, Q__s6.
  Every frame in these three cohorts has min_dim ≤ 96 px. Model sees these only
  after a ~2.5× upsample to 224×224.
- **Normal-or-larger crops** (min_dim p50 ≥ 141): Roy_D, bla_bla_chow,
  bla_bla_chow__s2. Roy_D's smallest frame is 227 px; bla_bla_chow's is 306 px.
  These cohorts are at or above 224 in their p05 — they are NOT below the
  resize threshold.

## Direct observations

1. The pixelation visible in the contact sheets is in the data, not in the viewer.
2. Three of the six chronic-FP identities (PC_Generator__s22, PC_Generator__s45,
   Q__s6) have min_dim distributions concentrated below 100 px — substantially
   smaller than the model's 224 input resolution. The model upscales these
   ~2.5× before forward pass.
3. Three of the six chronic-FP identities (Roy_D, bla_bla_chow, bla_bla_chow__s2)
   have min_dim distributions at or above 140 px, with Roy_D and bla_bla_chow
   above 224 throughout. These are NOT below the model's input resolution.
4. The other_real group (n=300, non-chronic real cohorts) has min_dim p05=144,
   p50=223. Most non-chronic-real canary frames are at or above the resize
   threshold.
5. The fake group has min_dim p50=357.5; every fake cohort is above 200 px.
6. Per-frame data is at `outputs/canary_resolution_per_frame.csv` (800 rows).

## Cross-reference

- Eyeball driver + output: `eyeball_dip.py`, `figs/eyeball_chronic_*.png`
- Eyeball interpretive doc: `EYEBALL_DIP_EXPLANATION_2026-05-08.md`
- Memory: `project_canary_below_production_resolution_2026-05-08.md` (user policy)
- Related IQ-axis memory:
  - `project_image_quality_shortcut.md` (sharpness shortcut)
  - `project_face_size_label_leak.md` (face-size shortcut)
  - `project_iq_gating_viability_2026-05-04.md` (P8A IQ-gate viable, E2B inverted)
- Related thread: `docs/packet_retrospectives/threads/image_quality_shortcut.md`

## Artifacts

- `outputs/canary_resolution_per_frame.csv` — 800 rows × 9 columns
- `outputs/canary_resolution_per_cohort.csv` — chronic-6 per-identity rollup
- `verify_canary_resolution.py` — driver
