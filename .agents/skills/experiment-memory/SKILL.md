# Skill: Experiment Memory

Goal:
Understand historical experiments.

Inputs:
- experiment configs
- logs / metrics on wandb (access is heavy, so we want to minimize it, we use wandb MCP)
- .md files summarizing experiments and other relevant information (e.g. DeepfakeBench/training/docs/JAN_13_L14_vs_B16_LAION_Investigation.md)

Steps:
1. Cluster experiments by:
   - architecture
   - hyperparameters
2. Identify:
   - best runs
   - failed patterns
3. Detect:
   - overfitting trends
   - diminishing returns

Output:
- experiment summary table
- key learnings
- gaps in exploration