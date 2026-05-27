# R13 Relaunch Packet 1 Evaluator Handoff - 2026-04-19

Use the prompt below for an independent evaluator review of the current
relaunch packet.

The goal of this handoff is to get a fresh read, not to pass along conclusions.
The evaluator should use the docs below as context, then inspect the code and
configs directly and challenge anything that does not hold up.

## Prompt To Evaluator

```text
Review-only task.

Repo:
`/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision`

Goal:
Do an independent evaluator review of the current R13 relaunch packet-1
experiment plan and configs.

Important:
- Please do not assume the packet plan is correct just because it exists.
- Please do not anchor on prior summaries or conclusions from other agents.
- Use the markdown docs below as context, but treat them as claims to verify.
- If docs and code disagree, trust the code/config/runtime surfaces.
- If any counts or assumptions in the docs look stale or wrong, say so plainly
  and replace them with your own independently derived understanding.
- This is a review-only task. Do not make code changes.

Start by reading these markdown files for context:

1. `DeepfakeBench/training/docs/relaunch_handoffs/R13_RELAUNCH_PACKET1_EXPERIMENT_PLAN_2026-04-19.md`
2. `DeepfakeBench/training/docs/relaunch_handoffs/RELAUNCH_UPGRADE_REVIEW_PACKET_2026-04-19.md`
3. `DeepfakeBench/training/docs/relaunch_handoffs/NEW_DATA_LOADER_AND_EXPERIMENT_HANDOFF_2026-04-19.md`
4. `DeepfakeBench/training/docs/relaunch_handoffs/WT_B_AND_NEW_DATA_READINESS_2026-04-19.md`
5. `DeepfakeBench/training/docs/relaunch_handoffs/WT-C_2026-04-17.md`
6. `DeepfakeBench/training/docs/relaunch_handoffs/WT-F_2026-04-17.md`
7. `DeepfakeBench/training/docs/relaunch_handoffs/TASK_BOARD_2026-04-17.md`
8. `DeepfakeBench/training/docs/TEAMS_TARGET_DOMAIN_UPGRADE_PLAN_2026-04-06.md`

Then review these implementation / runtime surfaces:

- `DeepfakeBench/training/data/sources/combined_paired.py`
- `DeepfakeBench/training/data/sources/proper_data.py`
- `DeepfakeBench/training/data/augmentations/pipelines.py`
- `DeepfakeBench/training/utils/grouping.py`
- `DeepfakeBench/training/arena/build_visomaster_proper_data_artifacts.py`

Then review these packet configs:

- `DeepfakeBench/training/experiments/phase2_round13/R13_RLP1_01_FT_WTB1_no_hints_live.yaml`
- `DeepfakeBench/training/experiments/phase2_round13/R13_RLP1_02_FT_WTB2_hints_only_live.yaml`
- `DeepfakeBench/training/experiments/phase2_round13/R13_RLP1_03_FT_WTB3_hints_plus_teams_hints_live.yaml`
- `DeepfakeBench/training/experiments/phase2_round13/R13_RLP1_04_FT_WTB3_plus_proper_unenhanced_live.yaml`
- `DeepfakeBench/training/experiments/phase2_round13/R13_RLP1_05_FT_WTB3_plus_proper_full_live.yaml`
- `DeepfakeBench/training/experiments/phase2_round13/R13_RLP1_06_FT_WTB3_plus_proper_full_truthful_gammaup.yaml`
- `DeepfakeBench/training/experiments/phase2_round13/R13_RLP1_07_FT_WTB3_plus_proper_full_teams_shadow.yaml`
- `DeepfakeBench/training/experiments/phase2_round13/R13_RLP1_08_SCRATCH_WTB3_plus_proper_full_live.yaml`

What I want from you:

- Review whether the 8-run packet tells a coherent story across WT-B, WT-C,
  WT-F, and the original April 6 target-domain upgrade plan.
- Assess correctness, regressions, config/runtime consistency, and launch
  readiness.
- Specifically assess whether the packet mix makes sense at high level:
  - the data-composition ladder
  - the use of hints
  - the use of proper-data
  - the choice to include 2 WT-C sidecars
  - the choice to include 1 scratch hedge
- Check whether the unchanged scaffold is being held fixed where intended:
  backbone, rank, sampling regime, split policy, eval cadence, etc.
- Check whether the cache / freshness story is actually safe:
  - mutable discovery sources should not be stuck on stale cache manifests
  - proper-data should not silently miss fresh uploads if the builder refresh is
    required
- Check whether any config names, comments, tags, or documented rationale are
  misleading.
- Check whether the packet is overfitting to a narrative rather than asking the
  most informative first-round questions.

Specific questions to answer:

1. Is this a coherent first 8-slot packet for tonight?
2. What is strong about it?
3. What is risky or weak about it?
4. Is anything still missing or mis-specified before launch?
5. Does the fine-tune vs scratch split make sense?
6. Are the WT-C sidecars attached in the right way, or would you change that?
7. Is the proper-data freshness story operationally safe, or is there still a
   hidden trap?

Output format:

1. Findings first, ordered by severity, with file/line references.
2. Then open questions / assumptions.
3. Then a short verdict:
   - what is solid
   - what is still risky
   - whether you would launch this packet as-is, with minor fixes, or not yet

Please keep the review independent. If you agree with the packet, say why. If
you disagree with the packet, say so plainly and explain what you would change.
```
