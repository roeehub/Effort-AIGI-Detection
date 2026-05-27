# Identity Frame Browser

Self-contained HTML browser of the lockbox eval substrate + internal-test
identity (`PC_Generator`-style) frames, grouped by base identity then by
real/fake label and suite.

Created 2026-05-05 from `scope_manifest.csv` (7,808 unique frames, 34 base
identities).  This is **iteration 1** of an iterative workflow: start with raw
frame inspection; later add more buckets / per-checkpoint scores; finally
overlay scores so you can visually pick the best checkpoint per identity.

## Quick view

Open `index.html` in any browser:

```bash
open analysis/identity_browser_2026-05-05/index.html
```

Or browse to the `file://` URL.  All thumbnails embed via relative paths from
`thumbs/`; clicking a thumbnail opens the original full-resolution image from
`frames/`.

## Layout

```
analysis/identity_browser_2026-05-05/
  scope_manifest.csv                      # input: 7,808 rows
  scripts/
    build_browser.py                      # reusable end-to-end builder
    add_bucket.py                         # stub for adding more buckets
  data/
    grouped_manifest.csv                  # input + base_identity column (legacy schema)
    grouped_manifest_v2.csv               # + face_size, quality, score_<ckpt> columns
    summary.json                          # per-identity counts + suite breakdown
  frames/<base_identity>/<suite>__<src>.{jpg,png}   # full-size originals
  thumbs/<base_identity>/<suite>__<src>.jpg         # 256-px thumbnails (q=80)
  index.html                              # the browser
  run.log                                 # phase timings + per-frame skips
```

## Re-running

The build is **idempotent**.  Frames already on disk are skipped, thumbnails
already on disk are skipped.  Re-run after edits or after dropping new frames
in:

```bash
python scripts/build_browser.py
```

Pass `--skip-download` if you only changed something HTML-side.

## Adding a new bucket / suite

Two paths.

**Path A — you already have a fully-formed scope-manifest CSV** with the
columns `suite, video_id, frame_path, label, is_lockbox, bucket`:

```bash
python scripts/build_browser.py \
    --manifest scope_manifest.csv \
    --manifest /path/to/new_bucket_manifest.csv
```

`build_browser.py` accepts `--manifest` multiple times and merges them.

**Path B — you have a raw scorecard CSV** (e.g. produced by the promotion
contract launcher) with `video_id, frame_path, label`:

```bash
python scripts/add_bucket.py \
    --from-scorecard /path/to/scorecard.csv \
    --suite-name my_new_suite \
    --is-lockbox \
    --then-rebuild
```

This converts the scorecard to a manifest under `data/manifest_my_new_suite.csv`
and re-runs the build.  Edit `add_bucket.py::build_manifest_from_scorecard_csv`
if your scorecard column names differ.

## Adding per-checkpoint scores (iter 2 — wired)

The build script now accepts repeatable `--score-csv NAME=PATH_OR_GLOB`.
Each ckpt becomes one row in the `<div class="scores">` block under every
thumbnail.  Color logic: green text = model agrees with label (real<0.5
or fake>=0.5); red text = model disagrees (eyeball-debug target).

Default — wired to P8A reference @ step 5000:

```bash
python scripts/build_browser.py --skip-download --skip-thumb
# == python scripts/build_browser.py \
#       --score-csv 'P8A=/Users/roeedar/.../analysis/cpu_followups_2026-05-04/raw_reports/*p8a_reference_step5000_frames_report.csv'
```

Add another checkpoint later (e.g., E2B) — copy/paste form:

```bash
python scripts/build_browser.py \
    --skip-download --skip-thumb \
    --score-csv 'P8A=/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/analysis/cpu_followups_2026-05-04/raw_reports/*p8a_reference_step5000_frames_report.csv' \
    --score-csv 'E2B=/path/to/your/e2b_top_n_step3200/*frames_report.csv' \
    --score-csv 'PA_5600=/path/to/pa_top_n_step5600/*frames_report.csv'
```

Each CSV needs columns `frame_path, frame_prob`.  PATH may be a glob
matching multiple suite reports for the same ckpt; they're concatenated.
Order of `--score-csv` flags matters: the **first** ckpt is the primary
sort key (descending for real frames, ascending for fakes — surfaces
worst FPs / FNs at the top of each sub-bucket).

### Binary property tags

Iter 2 also derives two binary property axes from the lockbox tag parquet
(`analysis/lockbox_tagging/full_tags_2026-04-27.parquet`):

- **face_size**: `close` / `far` / `unknown`
  - per-suite median split on `face_area_ratio`
  - 0% coverage on `teams_real_dor_dev` → all marked `unknown`
  - ~29% coverage on `teams_real_all_lockbox` → ~1004 marked `unknown`
- **quality**: `hi-q` / `lo-q` / `unknown` (from `is_low_quality`)

Frames within each `(identity, label)` section are sorted by `face_size`
→ `quality` → score, with sub-bucket headers like
`▸ close + hi-q (37 frames)` between groups.

Two new toggle filter rows in the toolbar (face / quality) let you
narrow the view to a specific sub-bucket without losing the search box.
Override the parquet via `--tags-parquet PATH` (or omit and accept the
default).

## Constraints honored

- CPU only (Mac, 10 cores, 32 GB RAM).
- No `sklearn n_jobs=-1` — uses `multiprocessing.Pool(8)` for thumbs and
  `ThreadPoolExecutor(16)` for `gsutil cp`.
- All paths quoted in shell calls; identities with spaces / special chars
  remain valid path segments (special chars only present in `/` are
  remapped, otherwise preserved).
- No base64 image embedding — output HTML is small (text-only) and points at
  on-disk thumbnails.

## Highlight panel (iter 3 — 2026-05-05 evening)

A sticky right-side panel (`#highlight-panel`, `position: fixed; right: 16px; top: 80px;`)
gives you live model-correctness feedback while you scroll thumbnails:

- **Model selector** — radio buttons: `Off` (default) plus one per loaded ckpt
  (`P8A`, `E2B`, `PA_3800`).
- **Threshold slider** — `[0.00, 1.00]` step `0.01`, default `0.50`. The
  numeric value shows next to the slider in real time.
- **Live counters** — `correct / wrong / total` for the **visible** frames
  (i.e. respecting the toolbar's filters), counted only over frames that have
  a score for the highlighted ckpt.

Border-color logic per thumbnail (3 px solid, same width as the default
transparent border so there's no layout shift):

| frame label | model score vs τ | border |
|---|---|---|
| real (label=0) | score < τ | green (correct) |
| real (label=0) | score ≥ τ | red (wrong) |
| fake (label=1) | score ≥ τ | green (correct) |
| fake (label=1) | score < τ | red (wrong) |
| any            | no score for ckpt | none (defensive) |
| any            | model = `Off` | none |

When a ckpt is highlighted, the other ckpt rows in each thumbnail's score
table are dimmed (`.score-row.muted { opacity: 0.45; }`) so your eye locks
on the active row's number.

Implementation: each thumbnail carries `data-score-<ckpt-lowercased>="<float>"`
plus the existing `data-label`. JS reads via `getAttribute()` (deterministic
under HTML's lower-casing rule) and toggles `highlight-correct` /
`highlight-wrong` classes on the `.frame.thumb` anchor.

The panel and existing filter UI are independent — flipping suite checkboxes
or typing in the search box just re-runs the highlight pass on the new
visible set, so counters always reflect what's actually on screen.

### Errors section (iter 4 — 2026-05-05)

Once a model is highlighted, a new `#errors-section` block renders at the
top of `<main>` with two sub-groups, each a `<details>`:

- **Wrong: REAL flagged as fake — N frames** (the FP cohort)
- **Wrong: FAKE missed — N frames** (the FN cohort)

Each grid is populated by **cloning** the matching `.frame.thumb` originals
from the identity sections (`cloneNode(true)` carries score row, pills, image,
caption, and link). Clones are tagged `data-error-clone="true"` and
`data-error-class="fp"|"fn"` so the highlight-repaint loop and the panel
counter strip skip them.

Behavior:
- Section hidden when model selector is `Off`.
- Hidden by `applyFilter()` indirectly: clones are sourced only from
  visible (non-`.hidden`) originals, so toggling a suite checkbox or
  typing in search filters the errors section too.
- FP grid sorted by score **descending** (worst FP first).
- FN grid sorted by score **ascending** (worst FN first).
- Empty cohort renders an italic *"(no errors at this threshold + filter)"*
  placeholder instead of an empty grid.
- Performance: cloning is on the slider's `change` event (release), not
  `input` (drag), so dragging stays smooth even with ~1k wrong frames.
  The originals' borders still update live on `input`; clones' borders go
  stale during drag and refresh on release.

Default counts at τ=0.50 with all filters checked:

| ckpt | FP (real→fake) | FN (fake→real) | total wrong |
|---|---|---|---|
| P8A | 668 | 242 | 910 |
| E2B | 560 | 155 | 715 |
| PA_3800 | 874 | 146 | 1020 |

To wire all three ckpts as in the iter-3 rebuild:

```bash
python analysis/identity_browser_2026-05-05/scripts/build_browser.py \
    --skip-download --skip-thumb \
    --score-csv 'P8A=analysis/cpu_followups_2026-05-04/raw_reports/*_p8a_reference_step5000_frames_report.csv' \
    --score-csv 'E2B=analysis/cpu_followups_2026-05-04/raw_reports/*_e2b_top_n_step3200_frames_report.csv' \
    --score-csv 'PA_3800=analysis/pa_pc_eval_2026-05-05/raw_reports/*_pa_top_n_step3800_frames_report.csv'
```

## Identity extraction (the bug-fix)

The `identity` column in `scope_manifest.csv` was wrong (it included
`__seg_X.Y` markers, which over-grouped identities).  Re-extracted via:

```python
def base_identity(video_id: str) -> str:
    s = re.sub(r"__seg_[\d.]+", "", video_id)
    s = re.sub(r"__seq_?\d+", "", s)
    s = re.sub(r"__(?:real|fake)$", "", s)
    s = re.sub(r"__frame_\d+_crop_\d+__[a-f0-9]+", "", s)  # ilan/orel-style ids
    s = re.sub(r"_+$", "", s)
    return s
```

Result: **34 base identities** (down from the 5,758 unique `video_id`s and
from the over-segmented 96 produced by the original column).
