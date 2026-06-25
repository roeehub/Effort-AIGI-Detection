---
name: experiment-memory
description: Analyze historical Effort/AIGI experiments across configs, local reports, and selectively queried Weights & Biases metrics. Use when comparing runs, recovering prior conclusions, identifying best or failed patterns, diagnosing overfitting, or finding gaps in the experiment matrix.
---

# Experiment memory

Prefer local configs, logs, Markdown reports, and the `ml-training` memory domain
before querying Weights & Biases. Use W&B only for missing metrics and keep
queries narrow.

1. Group runs by architecture, dataset, objective, and material
   hyperparameters.
2. Normalize metric names and distinguish validation, test, and production
   measurements.
3. Identify best runs, repeated failures, overfitting, and diminishing returns.
4. Preserve provenance for every conclusion and mark uncertain comparisons.
5. Record durable new conclusions as append-only `ml-training` inbox notes.

Return a compact comparison table, key learnings, failed patterns, and
high-value gaps in exploration.
