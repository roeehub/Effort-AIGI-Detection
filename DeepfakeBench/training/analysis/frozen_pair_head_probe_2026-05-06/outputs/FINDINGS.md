# Frozen Pair-Head Probe — FINDINGS

**Job:** `FROZEN_PAIR_HEAD_PROBE_2026-05-06`
**Date:** 2026-05-06
**Verdict:** **INSUFFICIENT_COVERAGE** — no on-disk feature cache contains same-source opposite-label pairs. Probe cannot run as specified; produces a precise blueprint instead.

> Note: this file mirrors the sub-agent's report. The sub-agent wrote `coverage_report.md`, `summary.json`, `verdict.json`, and `run_probe.py` (re-runnable; activates the heads-training path as soon as a paired feature cache exists).

## What the probe was supposed to do

Train tiny linear heads {CE-only / CE+pair-rank / CE+pair-rank+group-weighting} on cached frozen P8A/E2B features at paired indices. If pair-rank-trained head improves pair metrics over CE-only, pair-rank has **head-side signal** (frozen-encoder + new head suffices). If not, pair-rank is **encoder-side** (full FT justified). Job 7 already refuted CE-only head-only retrain (`project_job7_head_retrain_REFUTED_2026-05-04.md`); this probe asked the different pair-rank question.

## Why coverage is insufficient

Exhaustive audit in `coverage_report.md`:

- **`analysis/clip_vs_p8a_viso_2026-05-03/outputs/p8a__features.npz`** — 1100-frame viso eval substrate. `frame_path` keys present. But of the 1,812 paired reals from `pair_gaps.csv`, only 131 overlap; of the 3,499 paired fakes, **0 overlap**. The cache's 550 fakes are `visomaster_enhanced` with no co-bucketed real partner; its 550 reals are broad `teams_real_dev`. Asymmetric, no real-fake pairs.
- **`analysis/_features_cache_2026-04-30/*.npz`** (31 files) — store positional `valid_idx` only, no `frame_path`, no co-located `sampling_manifest.json` locally. Unmappable.
- **No other cache** found containing P8A or E2B features at paired indices.

**Net coverage of same-source paired features: 0 pairs across all caches.**

## Implication for P1 (`PE_PAIR_RANK_DRO`)

**P1 should be DEFERRED, not just AMBER-conditional, until this probe returns.** The reason is GPU-cost asymmetry:

- Encoder-FT for P1 = GPU-weeks of training compute (~10-30× the cost of feature extraction).
- Frozen-feature extraction at paired indices = **~$5-10, 1-2 A100-hours** over ~5,311 unique paired frames.

If frozen-feature pair-rank shows head-side signal, P1's encoder-FT is overkill (a head-only retrain might suffice — though Job 7 refutes the CE version, the pair-rank version is open). If frozen-feature pair-rank shows no signal, encoder-FT is justified or P1 is dead.

Either outcome strictly improves the prior on P1 vs launching encoder-FT blind.

## Extraction blueprint

**Step 1 — extract features at paired indices** (~$5-10, 1-2 A100-hours):
1. Reuse `analysis/feature_space_2026-04-23/extract_features.py` as the template.
2. Replace its bucket-scan with a CSV path-list discovery from `analysis/pair_gap_audit_2026-05-06/outputs/pair_gaps.csv` (5,311 unique frame paths covering both real and fake pair members).
3. Output to `analysis/frozen_pair_features_2026-05-07/{p8a,e2b,clip_b16_raw}_paired_features.npz` (~32 MB total per ckpt). Include CLIP-B16-raw as the un-fine-tuned baseline (matches Probe 4's reference).
4. Each `.npz` carries `frame_paths` (array of strings) and `features` (N × 768 array).

**Step 2 — run `run_probe.py`** with the new caches:
```
python analysis/frozen_pair_head_probe_2026-05-06/run_probe.py \
  --features analysis/frozen_pair_features_2026-05-07/p8a_paired_features.npz \
  --pair_gaps_csv analysis/pair_gap_audit_2026-05-06/outputs/pair_gaps.csv \
  --output_dir analysis/frozen_pair_head_probe_2026-05-06/outputs/
```
The script's coverage-check path is active today; the heads-training path activates the moment a paired feature cache exists.

## Promotion gate for P1

After the extraction + probe run:

- **Head B (CE + pair-rank) ≥ 3pp lift on `P(fake_score > real_score)` over Head A (CE-only)** → pair-rank has head-side signal → **defer P1 in favour of head-only retrain** (faster, cheaper). Job 7's CE-only head-only refutation does not apply here.
- **Head B no lift over Head A** → pair-rank is encoder-bound or absent → P1 justified IF Phase 0g is also GREEN; otherwise P1 dead.
- **Head C (CE + pair-rank + group-weighting) > Head B** → multi-axis GroupDRO is additive over pair-rank.

## Parallel deployable lever (no extraction needed)

`coverage_report.md` notes an adjacent finding: **per-substrate τ-calibration gives 21pp lockbox recall lift on P8A today** (Job 7 result, memory `project_job7_head_retrain_REFUTED_2026-05-04`). This is a different lever (operating-point, not weights) and is deployable today without any training. Worth capturing in parallel — does not interact with P1/P2/P3 packets and provides immediate production lift.

## Outputs

- `coverage_report.md` — exhaustive cache audit, per-cache overlap counts, per-subject distribution.
- `summary.json` — coverage counts; null head metrics; INSUFFICIENT_COVERAGE verdict.
- `verdict.json` — head-side/encoder-side localisation = UNDETERMINED; recommended followup name; cost; P1-blocking flag.
- `run_probe.py` — re-runnable; coverage-check path active today, heads-training path activates as soon as paired feature cache exists.
- (`head_metrics.csv`, `pair_gap_distributions.csv`, `head_models.npz` not produced — gated on extraction step.)
- `FINDINGS.md` — this file.

## Cross-references

- Plan: `docs/relaunch_handoffs/NEXT_STEPS_PLAN_2026-05-06.md` §8.2 P1 (now blocked on extraction or 0g+encoder-FT acceptance), §8.1 0b (this probe).
- Pair source data: `analysis/pair_gap_audit_2026-05-06/outputs/pair_gaps.csv`.
- Feature extraction template: `analysis/feature_space_2026-04-23/extract_features.py`.
- Job 7 framing: `memory/project_job7_head_retrain_REFUTED_2026-05-04.md`.
