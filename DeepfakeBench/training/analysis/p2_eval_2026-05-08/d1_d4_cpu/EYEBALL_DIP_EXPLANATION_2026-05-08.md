# Eyeball — Slot D step3000 vs step19000 chronic-real disagreement (2026-05-08)

> **Status: explanation + interpretive guidance.** This is *not* a FACTS-only doc; it
> tells you what to look for in the contact sheets and why. The numbers are FACTS;
> the framings are working hypotheses, not verdicts.

## What I'm showing you

For each of the 6 chronic-real identities (PC_Generator__s22, PC_Generator__s45,
Q__s6, Roy_D, bla_bla_chow, bla_bla_chow__s2), one contact sheet at
[`figs/eyeball_chronic_<id>.png`](figs/) showing the **6 frames where Slot D
`top_n_step19000` and `periodic_step3000` disagreed the most** in absolute
prob_fake distance. Each tile is annotated with:

- `frame N` — the canary frame_idx (joinable to the parquet)
- `Δ = +X.YYY` — `score(step19000) − score(step3000)`. Positive = the late-trained
  model is MORE confident this is fake than the early-trained model.
- `step3k=X.YYY` — Slot D `periodic_step3000` score (low-saturation state; matches
  canary `_step=3000` fire exactly per D5)
- `step19k=X.YYY` — Slot D `top_n_step19000` score (late-saturated state; matches
  canary `_step=24000` fire)
- `P8A=X.YYY` — P8A reference (production anchor)

All these frames are **labeled real** (`label=0`) — they come from the chronic-6 set
which is the canary's hardest-real cohort. A score near 0.0 = "model thinks real" =
correct on these frames. A score near 1.0 = "model thinks fake" = false positive.

## Why this comparison

The canary fire at `_step=18000` showed a striking transient pattern: Roy_D's
mean prob_fake dropped from 0.94 → **0.68** between `_step=15000` and `_step=18000`,
then recovered to 0.89 at `_step=21000`. bla_bla_chow showed a parallel dip 0.91 →
**0.41** → 0.81. `lockbox_recall@FPR_10pct` simultaneously PEAKED at 0.41 (highest
across all 8 fires of Slot D).

The implication, if real: for ~1000 optimizer steps mid-training, Slot D's model
state was BETTER on chronic reals AND BETTER on lockbox fakes — the kind of
joint-improvement pattern that's rare in this project's history. Memory
`project_train_auc_not_valid_promotion_signal.md` has been warning that train AUC
keeps rising while operationally relevant metrics oscillate; the canary's `_step=18000`
captured one such oscillation in a useful direction.

That dip-state was NOT checkpointed. Per D5 (FACTS doc
`D5_CKPT_MAPPING_FACTS_2026-05-08.md`):
- The closest savable approximation is `slotD_periodic_step3000` (L2=0.437 over
  17 metrics — 4× the exact-match distance of 0.10).
- step3000 has the highest `lockbox_recall_at_FPR_10pct` (0.37) of any saved D ckpt.
- step3000 is the only D ckpt with `score_p95_on_reals` below 0.95 (it's 0.88).

So we use **step3000 as the operational stand-in for the dip-state**, and **step19000
as the late-saturated peer**. Comparing the two on chronic-real frames tells us:
"between low-saturation and saturated states, which faces did the model decide are
fake?"

## What to look for in each contact sheet

For each identity, ask the questions in this order:

### Question 1 — Are these frames CORRECTLY labeled as real?

These come from `teams_real_*` GCS sources, but data hygiene memory
`project_eval_substrate_data_hygiene` flags this canary substrate as having known
issues. If you see a face that looks GFPGAN-enhanced or otherwise manipulated:
that's a candidate label-leak. Note the frame_idx and we can audit.

### Question 2 — Do the high-Δ frames cluster on a visual axis?

Specifically watch for:
- **Lighting** — flat indoor lighting, harsh shadows, low-key lighting. Memory
  `project_image_quality_shortcut` shows the model uses Laplacian variance (sharpness)
  as a fake predictor; if the high-Δ frames are uniformly soft-focused, that's the
  IQ-shortcut speaking through.
- **Camera distance / framing** — close-up vs medium-shot. Memory
  `project_face_size_label_leak` shows fake methods cluster at tight face-size
  bands; if high-Δ frames have unusual framing, that's the size-shortcut signal.
- **Skin / color cast** — yellow tint, color-graded skin, unusual saturation. Memory
  `project_dor_drift_named_axes_2026-05-06` shows ~30% of P8A's drift on Dor frames
  comes from `color_b_dev` — color is a known shortcut axis.
- **Specific person angle / expression** — if the high-Δ frames are all the same
  person at the same angle (e.g., front-facing webcam mode), the model may be
  keying on capture-mode rather than face content.
- **Background / context** — eval frames carry more background context than
  production crops (memory `project_eval_production_crop_tightness_gap`). High-Δ
  frames clustered on background features suggest the crop-tightness gap is
  load-bearing.

### Question 3 — Where does P8A sit relative to step3k / step19k?

Three patterns are diagnostic:

- **P8A LOW + step3k LOW + step19k HIGH** → step3k preserves P8A's correct
  classification; step19k REGRESSED. This is the "what did late training break"
  pattern. **Roy_D shows this most clearly** (P8A 0.07, step3k 0.37, step19k 0.93 in
  the surfaced top-Δ frames).
- **P8A HIGH + step3k LOW + step19k HIGH** → step3k is the OUTLIER; both P8A and
  step19k call these fake while step3k briefly went the other way. **PC_Generator__s22
  shows this pattern** (P8A 0.81, step3k 0.17, step19k 0.96). The "dip" here is step3k
  being wrong-direction-confident vs P8A — possibly worth-it on average, but means
  the dip mechanism is "early training hadn't yet learned PC_Gen's signature."
- **P8A LOW + step3k LOW + step19k LOW** → no disagreement to look at; these frames
  wouldn't appear in the top-6 by definition.

### Question 4 — Are PC_Gen-s22 and Roy_D the same kind of failure?

The headline numbers (delta_mean):
- PC_Gen__s22: +0.782 (step3k 0.17 → step19k 0.96), but P8A is also high (0.81)
- Roy_D: +0.554 (step3k 0.37 → step19k 0.93), but P8A is LOW (0.07) — P8A handles Roy_D!
- bla_bla_chow: +0.792 (step3k 0.14 → step19k 0.93), P8A LOW (0.14) — P8A handles!

Roy_D and bla_bla_chow regress vs P8A. PC_Generator does not (P8A had the same
problem). If the visual content of Roy_D / bla_bla_chow high-Δ frames looks
substantively different from the PC_Generator high-Δ frames — different lighting,
different framing, different anything — that hints at TWO distinct failure modes
in step19000. If they look the same, then it's a single shortcut affecting all
chronic identities.

## Per-identity headline (from this analysis, n=6 surfaced frames per identity)

| identity | step3k mean | step19k mean | Δ mean | P8A mean | reading |
|---|---:|---:|---:|---:|---|
| PC_Generator__s22 | 0.173 | 0.955 | **+0.782** | 0.810 | step3k disagrees with P8A; both step19k and P8A high |
| PC_Generator__s45 | 0.473 | 0.883 | +0.410 | 0.802 | similar to s22 but smaller magnitude |
| Q__s6 | 0.489 | 0.671 | +0.182 | 0.945 | smallest Δ; P8A is highest fake-confidence here |
| Roy_D | 0.374 | 0.927 | +0.554 | **0.066** | step19k REGRESSED vs P8A; step3k closer to P8A direction |
| bla_bla_chow | 0.140 | 0.932 | **+0.792** | 0.144 | step19k REGRESSED vs P8A; step3k matches P8A direction |
| bla_bla_chow__s2 | 0.252 | 0.707 | +0.455 | 0.596 | P8A also moderately confident-fake |

## Why this matters for the next packet

If the eyeball reveals a CLEAR visual axis on which the high-Δ frames cluster
(e.g., low Laplacian variance, specific framing, color cast), the next anti-shortcut
intervention has a concrete target. The Fourier-aug packet (Slot D) was an attempt
at one such intervention — it dampened the high-frequency capture-pipeline signature.
If the eyeball shows step19k still regresses on a non-frequency axis (e.g., color),
that's a candidate axis for a future packet.

If the eyeball shows NO clear visual cluster — the high-Δ frames look randomly
selected — that suggests the model's decision is on a non-pixel feature
(face-embedding-level identity signature, learned through training data
co-occurrence). That's harder to attack with an aug-side intervention.

## Scope caveats

- These 6 frames per identity are the EXTREME-disagreement frames. They are not
  representative of the full 50-frame cohort. The full 50 frames probably include
  many low-Δ frames where step3k and step19k agree.
- step3000 is an early-training ckpt (epoch 1 of 9, training-AUC 0.97). Its
  "low-saturation" property reflects "hasn't learned the full signal yet" as much
  as "hasn't learned the shortcut yet." Distinguishing these two from a single
  ckpt is fundamentally limited.
- The canary `_step=18000` dip itself cannot be directly inspected — the model
  state at that point was not saved. step3000 is a structural proxy, not a
  one-to-one match.

## Artifacts

- 6 contact sheets at `figs/eyeball_chronic_<cohort>.png` (1 per chronic identity)
- Summary CSV at `outputs/eyeball_dip_summary.csv` (36 rows = 6 identities × 6 frames)
- Driver: `eyeball_dip.py`
- Source per-frame scores: `scores/slotD_periodic_step3000.csv`,
  `scores/slotD_top_n_step19000.csv`, `scores/_canary_meta.csv` (P8A reference)
