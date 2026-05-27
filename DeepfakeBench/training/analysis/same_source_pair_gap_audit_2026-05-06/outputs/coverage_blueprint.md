# Forward-pass blueprint — `SAME_SOURCE_PAIR_GAP_AUDIT_2026-05-06`

This blueprint specifies **exactly** what would be needed to answer the load-bearing
go/no-go for `PE_PAIR_RANK_DRO` (Phase 0g, P1) at the trainer's tight
`(sample_id, frame_idx)` opposite-label pairing key.

**Triggered by:** Step-1 cache-first audit found **0 tight (sample_id, frame_idx)**
opposite-label pairs in any local score cache for the six paired training lanes.
See `summary.json::step_1_cache_first_audit` and `coverage_summary.json` for the
full coverage table.

**Do NOT execute this blueprint without explicit user authorization.** The user reserves
judgment calls at decision points (memory `feedback_decision_points.md`).

---

## 1. Scope

Run forward-pass inference with 3 checkpoints (P8A, E2B, PA_3800) on a stratified
sample of frame indices from the 6 paired training lanes. Compute pair_gap at each
trainer-emitted `(sample_id, frame_idx)`.

| Lane | n_paired_base_samples | Pair key | Bucket(s) | Suggested sample size |
|---|---:|---|---|---:|
| df40 | 4,698 | `pair_id`, `frame_idx` | `gs://df40-frames-recropped-rfa85/{real,fake}/...` | 500 base × 4 frames = 2,000 pairs |
| deeplive | 4,067 | `sample_id`, `frame_idx` | `gs://deeplive-frames-*` (training bucket) | 500 base × 4 frames = 2,000 pairs |
| visomaster_v1_base | 342 | `sample_id`, `frame_idx` | `gs://visomaster-face-cropped-v1/...` | all 342 base × 4 frames = 1,368 pairs |
| visomaster_enhanced | 1,484 | `sample_id`, `frame_idx` | `gs://visomaster-enhanced-face-cropped-v2/...` | 500 base × 4 frames = 2,000 pairs |
| visomaster_teams_enhanced | 997 | `sample_id`, `frame_idx` | mixed (54 teams_v2 + 943 clean_fallback) | all 997 × 4 frames; partition into transport sub-buckets |
| deeplive_teams | ~1,300 | `sample_id`, `frame_idx` | `gs://deeplive-teams-*` (count not verified) | 500 base × 4 frames = 2,000 pairs |

**Total estimated forward passes:** ~9,400 base samples × 2 sides × 4 frames × 3 ckpts
= **~225K image inferences.** At ~50 fps on an A100 with batch=64, that is **~75 minutes
of GPU time per ckpt**, ~**3.75 GPU-hours total**. Fits comfortably in a single
1×A100 node, ~$5–8 of Vertex AI compute (us-east1).

**Scope sanity check vs `pair_coverage_audit_2026-05-06.coverage_summary.json`:**
matches the 12,888 paired base samples (~91.5% paired fraction) reported. We propose
sub-sampling to 9,400 to keep cost down while preserving per-lane statistical power
(min n_missed_fakes ≥ 100 per lane is achievable assuming the catch-rate priors
from the cross-product audit hold).

## 2. Frames-to-score CSV schema

```
pair_id,sample_id,frame_idx,lane,base_identity,method,enhancer,transport,
gcs_real,gcs_fake,
real_score_P8A,fake_score_P8A,pair_gap_P8A,
real_score_E2B,fake_score_E2B,pair_gap_E2B,
real_score_PA_3800,fake_score_PA_3800,pair_gap_PA_3800
```

Final aggregate output is the same `summary.json` / `verdict.json` structure as this
audit, but with **measured tight-pair P(pair_gap ≤ 0 | missed_fake)** per lane per
ckpt, replacing the cross-product proxy.

## 3. Implementation

Re-use `scripts/launch/launch_batch_inference.sh` with three modifications:

1. **Input manifest:** build a stratified pair manifest (`paired_inference_manifest.csv`)
   from the 6 paired training-lane sources by walking `combined_paired.py`'s discovery
   functions in **inference mode** (no augmentation):

   ```python
   from data.sources.combined_paired import (
       create_combined_paired_pipeline,
       UnifiedPairedSample,
   )
   # iterate over the unified samples; for each sample emit (real, fake) at
   # frame_idx ∈ {0, 8, 16, 24} (matches the trainer's typical 32-frame stride)
   ```

   Write that to a CSV the existing inference launcher can ingest.

2. **Sample 4 frame indices per base sample.** Trainer emits 32 frames per pair on
   DF40, fewer on others; uniform stride at indices {0, 8, 16, 24} keeps coverage
   comparable across lanes.

3. **Run inference 3× (P8A, E2B, PA_3800)** with the existing `prob_fake` head; join
   on `(pair_id, frame_idx)` post-hoc.

**Where to run:** us-east1 or us-west4 (per `feedback` policy; data lives in US
multi-region). PA_3800 ckpt path is in `experiments/<latest>/checkpoint_map.yaml`;
P8A and E2B paths are well-established (`gs://...workspace.../P8A_step5000.pth`,
`gs://...E2B_top_n_step3200.pth`).

## 4. Decision rule (unchanged from §8.2 of NEXT_STEPS_PLAN)

For each lane × ckpt:

- `P(pair_gap ≤ 0 | missed_fake) > 25%` → **GREEN** (pair-rank loss has training signal).
- `10–25%` → **AMBER** (marginal — depends on margin and DRO weighting).
- `<10%` → **RED** (lever is dead — the encoder already orders fake ≥ real on missed fakes).

**Aggregate verdict:** P1 launches if **GREEN in ≥2 paired lanes** on the chosen
FT-base (P8A per Agent 3 cohort diagnosis). RED across all 6 → P1 demoted, P2
elevated.

## 5. Cost vs information ledger

| Option | Cost | Information | Trade-off |
|---|---|---|---|
| Skip and use cross-product proxy (this audit) | $0 | 2 of 6 lanes covered | Verdict is partial — RED on E2B, mixed on P8A/PA |
| Run full forward-pass blueprint | ~$5–8, ~75 min | 6 of 6 lanes measured directly | Definitive verdict |
| Run forward pass on viso_enhanced + viso_teams_enhanced ONLY | ~$2, ~25 min | Confirms/refutes the GREEN signal in cross-product cell `dor_local | visomaster_v2_dor` (28.1%) | Cheaper but doesn't cover df40 / deeplive lanes (which have ~1/3 of paired base samples each — significant skip) |
| Run forward pass on df40 + deeplive ONLY | ~$3, ~40 min | Resolves the largest unmeasured lanes; the two viso lanes have proxy coverage already | Best $-per-bit |

**Recommendation if user authorizes:** run **df40 + deeplive only** as a $3 first
pass; that's the largest unmeasured cohort and where the cross-product audit had no
visibility. If that returns GREEN on either, P1 launches lane-restricted; if both
RED, the visomaster GREEN signal alone (one lane) is below the 2-lane minimum and
P1 demotes to P2.

## 6. Open question for user

> Authorize the forward-pass audit (df40 + deeplive lanes only, ~$3, ~40 min)?
> Or proceed with lane-restricted P1 launch on the cross-product-proxy GREEN
> visomaster_enhanced lane while running P2 (SBI) in parallel?

(The user-flagged decision-point pattern from `feedback_decision_points.md`
applies — present the recommendation + tradeoffs, wait for explicit pick.)
