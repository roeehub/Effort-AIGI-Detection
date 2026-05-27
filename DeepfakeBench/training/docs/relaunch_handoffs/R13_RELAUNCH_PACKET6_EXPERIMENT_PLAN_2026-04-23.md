# R13 Packet-6 Experiment Plan — align gate + training to the actual deployment target

**Generated**: 2026-04-23 ~18:00 UTC
**Revised**: 2026-04-23 ~22:00 UTC — corrected gate-block root cause + added gate-hygiene lever
**Branch**: `teams-relaunch-root-2026-04-17`
**Status**: **Draft — pending packet-5 final state + user approval.** No yamls generated yet.

## TL;DR

Packet 5 E3 successfully closed the headline VisoMaster-Teams training gap
(proper lanes from 67%/18% → 94%/96% per feature-space on slot-07's
`value_composite` checkpoint at step 20500, composite = 0.7736).

Packet 6 pivots to the **remaining reason composite is ~0.77 instead of
~0.90**: the value-composite gate forces τ high to keep
`external_youtube_avspeech_real` FPR ≤ 5%, which depresses
`teams_fakes_tpr` / `other_fakes_tpr`. avspeech is at **54% accuracy**
(46% FPR) at natural thresholds and is the dominant real-pool gate
driver. Packet 6 targets that explicitly, plus two cleanup levers.

**Three levers in packet 6** (all yaml-only, one-line code change already
landed):

1. **Gate real-pool alignment** — remove `external_youtube_avspeech_real`
   from `ood_monitoring.external_real_sources`; move it to a readout-only
   block (or drop entirely). It is **not deployment-distribution** — the
   deployment target is Teams-passed reals, covered by `teams_ood_real`.
2. **Gate fake-pool hygiene** (code landed 2026-04-23) — add
   `path_exclude_contains: ["/visomaster_"]` to the `teams_ood_fake`
   external_fake_sources entry so the gate no longer grades the detector
   against ~33% failed-deepfake hint folders (these look near-real by
   design — "hints" are VisoMaster runs where the face swap failed and
   the frame is mostly the original real with subtle processing
   artifacts).
3. **Drop `wma_failure_fake` from the gate** — it sits 14.55 Fréchet from
   `deeplive_enh_fake` per feature-space (effectively in-distribution,
   99.4% detected), contributes nothing actionable to `other_fakes_tpr`,
   and the earlier bucket-comparison hypothesis (that it drives
   `worst_pool_fpr`) was wrong — `worst_pool_fpr` is computed over REAL
   pools only (trainer.py:182–216, `_find_threshold_for_mean_fpr`).

## Corrected gate-block analysis

Earlier packet-6 framing said `wma_failure_fake` drives
`worst_pool_fpr` and removing it unblocks the gate. That was wrong. The
composite code at `trainer/trainer.py:161–216` computes
`worst_pool_fpr` over `real_pools` only. `wma_failure_fake` is a fake
pool and has no effect on gate blocking.

**The actual gate blockers are the three real pools in
`external_real_sources`:**

| Real pool | Acc at τ≈0.44 | FPR | In deployment distribution? |
|---|---:|---:|---|
| `external_youtube_avspeech_real` | **54.2%** | **45.8%** | No (not Teams-passed) |
| `zoom_vcd_real` | 92.0% | 8.0% | No (not Teams-passed) |
| `teams_ood_real` | 97.8% | 2.2% | **Yes** |

Because the gate requires every real pool's FPR ≤ 5%, the bisection
has to push τ extremely high to tame avspeech, which crushes
`teams_fakes_tpr` and `other_fakes_tpr`. The best recoverable composite
is ~0.77. Without avspeech in the gate (or with it gated via a much
lower cap), τ can be chosen to yield ≥0.90.

## Why composite = 0.7736, not 0.90 (the math)

From `_compute_value_composite` (trainer.py:219–294):

```
composite = (0.6·teams_fakes_tpr + 0.3·other_fakes_tpr + 0.1·stability) / active_weight
```

At step 20500 with a viable τ, stability ≈ 0.81 (p95 score_jitter ≈
0.19; dominated by `ood_lighting_stress_general` 0.20 and
`ood_spatial_stress_scale` 0.19). That leaves ~0.69 for the two TPR
terms: 0.6·T + 0.3·O ≈ 0.69 → T, O roughly in the 0.70–0.85 band.
Those are the pool TPRs *at a τ high enough to hold avspeech ≤ 5%*.
At natural τ, teams_ood_fake is 98% and wma_failure_fake is 100%.

## The 8 experiments

All slots inherit from `R13_RLP5_07_E3_seedB` (packet-5 leader at
composite 0.7736) unless otherwise noted.

| # | Name | Lever (delta from RLP5_07) | Hypothesis |
|---|------|------------------------------|------------|
| 1 | `R13_RLP6_01_gate_align_canary` | Drop `external_youtube_avspeech_real` and `external_vcd_real` from `external_real_sources`; keep `teams_ood_real`. Drop `wma_failure_fake` from `external_fake_sources`. Add `path_exclude_contains: ["/visomaster_"]` to `teams_ood_fake`. | The deployment-aligned gate (`teams_ood_*` only) unblocks the composite. Expect metric ≥ 0.88. Canary — launch alone first. |
| 2 | `R13_RLP6_02_hint_clean_only` | Only change: add `path_exclude_contains` to `teams_ood_fake`. Keep avspeech+VCD in the real gate. | Isolates whether hint contamination in the fake-side pool was materially depressing composite (expect ≤ 0.01 swing; confirms lever 2 is hygiene, not a driver). |
| 3 | `R13_RLP6_03_drop_wma_only` | Only change: remove `wma_failure_fake`. Keep everything else. | Confirms wma is not a gate blocker (expect composite unchanged). If this slot also stays around 0.77, corroborates the corrected gate-block model. |
| 4 | `R13_RLP6_04_add_enh_clean` | Slot 1 + include `proper_visomaster_enhanced_clean` in `proper_data.include_lanes` + `family_weights.proper_visomaster_enhanced_clean_fake: 2.0` | Does the sharpest unused user-created fake source improve composite beyond the gate-alignment baseline? |
| 5 | `R13_RLP6_05_seed_variance` | Slot 1 + `seed: 742 → 745` (keep `split_seed`/`identity_split_seed` at 737) | Seed-variance measurement on the deployment-aligned gate. Needed because the packet-5 leader was a seed-variance result. |
| 6 | `R13_RLP6_06_avspeech_readout_only` | Slot 1 + move avspeech/VCD into a readout-only OOD block (so we still see their FPR but they do not gate). Requires a tiny yaml convention — keeps monitoring visibility. | Same intent as slot 1, but preserves external_real_sources observability. Checks that readout-only logging survives the yaml change. |
| 7 | `R13_RLP6_07_heavy_enh` | Slot 4 + `proper_visomaster_enhanced_clean_fake: 2.0 → 4.0` | Saturation probe — does loading more of the sharpest user-created data help further, or saturate at 2.0? |
| 8 | `R13_RLP6_08_E1_gate_align` | Based on `R13_RLP5_08_E1_teams2_5` (E1 base, packet-5 robust 0.7691) + slot-1 gate changes | Confirms the E1 backbone also benefits from gate alignment and serves as a second anchor. |

### Coverage matrix

| Lever | Slots covering it |
|---|---|
| Drop avspeech + VCD from real gate | 1, 4, 5, 6, 7, 8 |
| Hint-exclude `teams_ood_fake` | 1, 2, 4, 5, 6, 7, 8 |
| Drop `wma_failure_fake` | 1, 3, 4, 5, 6, 7, 8 |
| A1 `proper_visomaster_enhanced_clean` in training | 4, 7 |
| Saturation probe | 7 |
| Seed-variance measurement | 5 |
| E1 backbone (vs E3) | 8 |

### Region allocation

```
us-west4         slot 1  gate_align_canary        ★ launch alone first
us-west4         slot 2  hint_clean_only
us-east1         slot 3  drop_wma_only
us-east1         slot 4  add_enh_clean
asia-southeast1  slot 5  seed_variance
asia-southeast1  slot 6  avspeech_readout_only
asia-southeast1  slot 7  heavy_enh
us-central1      slot 8  E1_gate_align
```

- `us-west4` (2 jobs): canary region.
- `us-east1` (2 jobs).
- `asia-southeast1` (3 jobs): launch after packet-5 slots 02/05/06 finish.
- `us-central1` (1 job): slot 8.
- `europe-west4`: not used.

## Code change (landed this session)

`data/validation_sources.py` and `data/sources/combined_paired.py` now
accept `path_exclude_contains: Optional[Iterable[str]]` on the external
real / fake loaders + stress-OOD loaders. No yaml rewrites required for
legacy yamls — the flag is opt-in. Packet-6 yamls set it explicitly on
`teams_ood_fake`:

```yaml
- bucket: "live-deepfake-methods-real-and-fake-frames-cropped-teams-v2"
  prefix: "samples"
  method: "teams_ood_fake"
  grouping: "by_folder"
  path_contains: "/frames/fake/"
  path_exclude_contains: ["/visomaster_"]   # NEW — drops 535 failed-hint folders
  video_id_depth: -4
  max_videos: 300
  deterministic: true
```

Rationale: every `visomaster_*` folder in the teams-v2 bucket is a
failed VisoMaster run (the `VISOMASTER_BAD_DATA_POLICY_MANIFEST` marks
all 4,904 as `ignore` with reason
`teams_pair_complete_<method>_excluded`). Training's `teams.enabled`
loader already applies that policy via `_select_clean_teams_samples`.
The `external_fake_sources` loader did **not**, so the gate was
grading the detector against near-real content labeled fake.

## Deferred (unchanged from prior draft)

- **A3** (`visomaster-enhanced-face-cropped-v2` loader): per feature-space
  this bucket sits in-distribution already (84% recall); low urgency
  given the deployment-relevant pools are now well-covered by packet 5.
- **B2** (identity-held-out proper slices as OOD fake driver): nicer
  fake-side gate diversity once avspeech is out; not a blocker.
- **D** (random-rescale augmentation): still cheap; useful for further
  robustness if avspeech-like reals re-enter the gate in a later packet.
- **E** (per-lane identity-held-out test splits): still valuable.

Recommended order post-packet-6: **B2 → D → A3 → E**.

## Launch plan (pending approval)

```bash
# VERSION is at 1.3.198 (uncommitted vs HEAD's 1.3.193). The path_exclude_contains
# code change landed this session — bump VERSION to 1.3.199 before the image rebuild.
./dev.sh build-prod -y

# Launch slot 1 alone first — canary. Confirm RUNNING + first-eval composite
# jumps to ≥0.85 (gate unblocked). Then launch the rest.
./scripts/launch/launch_experiment.sh -y enhanced-aug-test us-west4 \
    experiments/phase2_round13/R13_RLP6_01_gate_align_canary.yaml

# After canary looks healthy:
./scripts/launch/launch_experiment.sh -y enhanced-aug-test us-west4 \
    experiments/phase2_round13/R13_RLP6_02_hint_clean_only.yaml

./scripts/launch/launch_experiment.sh -y enhanced-aug-test us-east1 \
    experiments/phase2_round13/R13_RLP6_03_drop_wma_only.yaml
./scripts/launch/launch_experiment.sh -y enhanced-aug-test us-east1 \
    experiments/phase2_round13/R13_RLP6_04_add_enh_clean.yaml

./scripts/launch/launch_experiment.sh -y enhanced-aug-test us-central1 \
    experiments/phase2_round13/R13_RLP6_08_E1_gate_align.yaml

# asia-southeast1: launch after RLP5 02/05/06 SUCCEEDED.
./scripts/launch/launch_experiment.sh -y enhanced-aug-test asia-southeast1 \
    experiments/phase2_round13/R13_RLP6_05_seed_variance.yaml
./scripts/launch/launch_experiment.sh -y enhanced-aug-test asia-southeast1 \
    experiments/phase2_round13/R13_RLP6_06_avspeech_readout_only.yaml
./scripts/launch/launch_experiment.sh -y enhanced-aug-test asia-southeast1 \
    experiments/phase2_round13/R13_RLP6_07_heavy_enh.yaml
```

## Expected decision points post-packet-6

1. **Does slot 1 reach composite ≥ 0.88?** Confirms the
   gate-alignment hypothesis. If yes, gate pools going forward should
   be teams_ood_* only.
2. **Does slot 2 stay near 0.77?** Confirms the hint-clean fix is
   hygiene, not a materially-different signal.
3. **Does slot 3 stay near 0.77?** Confirms wma is not a gate driver.
4. **Does slot 4 beat slot 1?** Does loading
   `proper_visomaster_enhanced_clean` add signal beyond the gate fix?
5. **Does slot 5 agree with slot 1 ± 0.01?** Seed-stability check.
6. **Does slot 7 beat slot 4?** Saturation probe on the enh weight.

Primary result to track: `best_value_composite/metric` vs the packet-5
leader `0.7736`. Expected jump from gate alignment alone is ≥ +0.10.

## Dropped from the prior draft

- **"Drop `wma_failure_fake` unblocks the gate"** — wrong premise (see
  § Corrected gate-block analysis). Still dropping `wma` is fine as
  cleanup; slot 3 isolates this.
- **"`visomaster.enabled: true` in the DeepLive bucket"** — not in
  packet 6. The 9 method families there are not deployment-distribution
  (they are the non-Teams proper VisoMaster variants, already covered
  by `proper_visomaster_clean`). We can revisit after gate alignment.
- **"Deweight smooth proper lanes"** — the packet-5 results already
  show proper_visomaster_teams at 100% and proper_visomaster_enhanced_teams
  at 98.6% per-method accuracy; no evidence those weights hurt.

## References

- `analysis/feature_space_2026-04-23/REPORT.md` — per-source prob_fake
  + distance matrices (slot-07 checkpoint).
- `analysis/bucket_comparison_2026-04-23/REPORT.md` — bucket-level
  distribution analysis (still useful but its gate-block causation
  claim is superseded — see above).
- `docs/relaunch_handoffs/R13_RELAUNCH_VISOMASTER_TEAMS_POOL_DIAGNOSTIC_FINDINGS_2026-04-23.md` —
  what motivated packet 5.
- `trainer/trainer.py:161–294` — `_find_threshold_for_mean_fpr` +
  `_compute_value_composite`. Authoritative on gate semantics.
- `experiments/phase2_round13/R13_RLP5_07_E3_seedB.yaml` — base config.
- `policy/visomaster_bad_data/VISOMASTER_BAD_DATA_POLICY_MANIFEST_2026-04-17.csv` —
  why `visomaster_*` folders in teams-v2 are hints.
