# Autonomous overnight plan — 2026-05-02 → 03

**User context**: leaving for ~7h overnight. Budget: ~$40 for additional work
after S1/S2 scorecards. Goal: keep iterating toward "single model with strong
across-the-board recall." Be thorough, don't waste budget.

**Sequence**:

## Phase 0 — current state (23:00 UTC)
- S1 (`padjfsoq`, us-west4): training complete at step 1000 @ 22:50, post-eval
- S2 (`evm4y66r`, us-east1): SUCCEEDED 22:32, Probe 1 done.
  Class_sep peak 4.89 at step 200 (PASSES F-S2-A threshold of 4.0).

## Phase 1 — when S1 SUCCEEDS (~23:15-23:30 UTC)
1. Run Probe 1 on S1 (`01_select_best_ckpt_by_class_sep.py --run-id padjfsoq`)
2. Update `arena/checkpoint_maps/teams_target_domain.s1_s2_2026-05-02.yaml`
   replacing the S1 placeholders with the top-3 ckpts by class_separation
3. Run pytest tests (must pass)
4. Rebuild image (`./dev.sh build-prod -y`) → 1.3.246 (~10 min)
5. Launch 2 parallel scorecards:
   - **Scorecard A** to us-west4: P8A + P22 step1k + S1 top-3 (5 ckpts × 9 suites)
   - **Scorecard B** to us-east1: P8A + P22 step1k + S2 top-3 (5 ckpts × 9 suites)
   Use `--checkpoints` filter on the same map yaml.

## Phase 2 — when both scorecards SUCCEED (~02:30-03:00 UTC)
1. Pull threshold_grids + reports for both
2. Run Probe 2 (compute_falsifiers_s1_s2.py) for S1's best AND S2's best
3. Run Probe 3 (p22_redux_diff.py) for both
4. Run unified chain analytics (04_unified_chain_analytics.py) — combines
   P22 + S1 + S2 grids into one operating-point comparison
5. Write `analysis/s1_s2_2026-05-02_planning/FINDINGS.md`

## Phase 3 — autonomous decision tree (~03:00-04:00 UTC)

Based on the unified chain analytics, **branch on viso recall at FPR=10% (joint)**:

### Branch A — neither S1 nor S2 hits viso > 30% at FPR=10%
**Most likely outcome based on chain history.**
The "training cap + base ckpt" axis is exhausted. Viso needs structural intervention.

**Next packet (S3-VISO-WEIGHT, ~$8-12):**
- FT from BEST ckpt of S1 vs S2 (whichever has best viso recall at FPR=10%)
- Single lever: increase `combined_paired.sampling.family_weights.visomaster_fake`
  from 4.0 → 8.0 (push more viso fakes into batches)
- Same training cap (1000 steps), same s_end=8
- **Hypothesis**: viso under-representation in training batches caps the viso recall
  ceiling. More viso exposure during the short FT window helps viso specifically.
- Falsifier: viso recall at FPR=10% (joint) > 35% (must beat best S1/S2 result)
- Run scorecard, do CPU follow-ups, decide if budget allows another packet

### Branch B — S1 OR S2 hits 3/3 falsifiers
**Less likely but possible.** The training-cap + base axis worked.

**Next packet (S3-CONSOLIDATE, ~$8-12):**
- Stack one more lever on the winner — luma jitter (the F2 shifted-shortcut hypothesis)
- FT from S1 or S2 winner, same 1000-step cap
- New lever: pipeline_randomization with luma_jitter [0.7, 1.3] symmetric
- Falsifier: viso recall ≥ winner's level (don't regress) AND F2 R² drops by 0.05

### Branch C — One of S1/S2 surprisingly succeeds on viso
Investigate before launching anything new. Use saved CPU budget.

## Budget guardrails
- **Hard stop**: do NOT spend more than $30 in additional GPU after Phase 2
- **Per-packet cap**: $15 max
- **If viso doesn't move with S3**: stop launching packets. Document the structural
  problem clearly; user will pick the next direction in the morning.
- **If a scorecard fails**: investigate cause, ONE retry max, then move on.

## What to write up regardless of branch
- `analysis/s1_s2_2026-05-02_planning/FINDINGS.md` — full S1/S2 verdict with all probes
- `docs/relaunch_handoffs/PSERIES_FACTS_2026-05-02.md` — append S1/S2 sections (pure data)
- `HANDOFF.md` — top-of-tree update with current best ckpt + next-step recommendation
- Memory entries for any structural insights

## Things NOT to do autonomously
- Cancel any already-running training/scorecard jobs (per memory)
- Run more than 2 additional training packets (budget)
- Propose stacked-bundle packets (single-lever discipline)
- Change to non-US regions (CLAUDE.md)
- FT from a degraded ckpt (P22 step8k or any post-collapse ckpt)
