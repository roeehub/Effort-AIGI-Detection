# Skill: Repo Understanding

Goal:
Build a mental model of the ML system.

Steps:
1. Identify:
   - entrypoints (train.py, main.py)
   - config system
   - data pipeline
   - model definition
2. Trace:
   data → preprocessing → model → loss → metrics
3. Detect:
   - dead code
   - duplicated logic
   - hidden dependencies

Output:
- architecture map
- key files list
- risks / unknowns