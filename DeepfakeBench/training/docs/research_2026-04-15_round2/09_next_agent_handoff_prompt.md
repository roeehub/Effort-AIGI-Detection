You are continuing an ongoing research program on a frame-based deepfake detector for live Microsoft Teams use.

This is not a fresh investigation.
This is Round 3+ of a chained research process.
Do not re-summarize prior work.
Continue from the remaining uncertainties.

Repo root:
`/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision`

Read first, in this exact order:

1. `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/research_2026-04-15_round2/00_research_index.md`
2. `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/research_2026-04-15_round2/07_round_synthesis_and_ranked_actions.md`
3. `/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/research_2026-04-15_round2/08_next_round_research_plan.md`

Then use the rest of:

`/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/research_2026-04-15_round2`

as primary context.

Second, use the previous package:

`/Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training/docs/research_2026-04-15`

as supporting context only.

Current high-value unresolved frontier after Round 2:

- freeze and test the calibrated low-FP promotion contract on the reduced shortlist
- verify whether any current R13 checkpoint actually beats `R12_G` under that contract
- fix augmentation-plumbing truth before drawing more lighting/gamma conclusions
- determine whether a real sampler/curriculum redesign can materially increase true Teams-companion exposure
- close or reduce the missing-condition gap around enhanced-through-Teams fake data/eval

Critical Round 2 conclusions you should assume unless disproven by stronger evidence:

- current checkpoint-selection and deployment-selection contracts are misaligned
- local repo truth still shows no enhanced-through-Teams fake emission lane
- current merged/teamsonly sampler leverage is much weaker than experiment names imply
- current R13 YAML appears to request GammaUp, but runtime does not activate it
- fixed threshold `0.5` is not an acceptable final deployment-selection policy

Working rules:

- use repo truth, data truth, evaluation truth, and current literature
- do not mainly brainstorm
- do not mainly restate Round 1 or Round 2
- call out metric mismatches, threshold mismatches, config no-ops, and source-name/runtime mismatches directly
- distinguish clearly between established, plausible, and still unknown
- no broad code changes
- only write markdown, except for a tiny helper if absolutely necessary
- create a new package under `DeepfakeBench/training/docs/` using the next available suffix after `research_2026-04-15_round2`
- never overwrite prior packages

If MCP `wandb` or `training-data-viewer` is available in your runtime, use them. Round 2 did not have them and had to rely on local discovery cache plus local repo artifacts.

Definition of success:

Produce the next research package that reduces one of the live uncertainties with new evidence, rather than rephrasing the previous package.
