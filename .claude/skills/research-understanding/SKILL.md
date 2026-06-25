---
name: research-understanding
description: Convert ML papers and technical reports into implementation and experiment plans for Effort/AIGI. Use when reviewing model architectures, losses, datasets, training strategies, ablations, assumptions, limitations, or reproducibility gaps.
---

# Research understanding

1. Extract the architecture, loss functions, training strategy, datasets,
   evaluation protocol, and reported ablations.
2. Separate stated facts from inferred or missing implementation details.
3. Translate the method into repository-relevant implementation blocks.
4. Propose falsifiable training hypotheses and the minimum experiments needed
   to reproduce or reject them.

Return a structured summary, assumptions and limitations, missing details, and
a prioritized experiment list.
