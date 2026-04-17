# Round 2 Research Index

This round was not a recap round. It did six concrete things:

1. Froze a deployment-facing evaluation and checkpoint-selection contract.
2. Rebuilt target-domain data truth from the frozen Teams manifest, loader code, and local discovery cache.
3. Recomputed sampler leverage from actual runtime behavior, including the effect of filtering before the identity split.
4. Audited augmentation propagation down to the config-key level.
5. Separated decision-layer weakness from model weakness for low-FP operation.
6. Deepened the literature only where the repo still has live uncertainty.

## Read this round in this order

1. `07_round_synthesis_and_ranked_actions.md`
2. `01_evaluation_contract_and_shortlist.md`
3. `02_target_domain_data_truth.md`
4. `03_sampler_curriculum_leverage.md`
5. `04_nuisance_invariance_and_augmentation_truth.md`
6. `05_decision_system_low_fp_analysis.md`
7. `06_literature_deepening.md`
8. `08_next_round_research_plan.md`
9. `09_next_agent_handoff_prompt.md`

## What changed versus the previous package

- The evaluation story is now stricter. Training checkpoint selection, frozen Teams promotion, and mixed-source regression are separated instead of being informally blended.
- The target-domain data story is tighter. Local repo truth supports this stronger claim: there is still no local train/eval lane that emits an actually enhanced-through-Teams fake.
- The sampler story changed materially. Under the current identity-balanced sampler, most weight tweaks are cosmetic; the meaningful competition is concentrated in a small overlap set, and the teamsonly lane is weaker than its YAML name suggests.
- The teamsonly analysis from Round 1 needed correction. The `companion_domains` filter is applied before the identity split, so teamsonly slightly reshuffles the whole train pool; even after correcting for that, true Teams-companion merged exposure remains small.
- The augmentation story tightened. `gamma_up_*` in current R13 YAML is not active in runtime. The current direct Teams branch also still gets only a minimal passthrough pipeline.
- The low-FP story is now explicitly a system story, not just a backbone story. The repo already contains calibration, temporal voting, and gating machinery that is cheaper to test than more retraining.

## Evidence base used this round

- Repo code and config truth:
  - `data/sources/combined_paired.py`
  - `data/augmentations/pipelines.py`
  - `trainer/trainer.py`
  - `arena/run_target_domain_validation_sequential.py`
- Frozen target-domain artifacts:
  - `arena/manifests/teams_target_domain_manifest_2026-04-06_frozen.json`
  - `arena/target_domain_suites.teams_manifest.frozen_2026-04-06.yaml`
  - `arena/target_domain_suites.r13_best_megaval_2026-04-13.yaml`
- Local discovery cache truth:
  - `.viewer_cache/discovery/df40.json`
  - `.viewer_cache/discovery/deeplive.json`
  - `.viewer_cache/discovery/visomaster.json`
  - `.viewer_cache/discovery/visomaster_teams_enhanced.json`
  - `.viewer_cache/discovery/teams.json`
  - `.viewer_cache/discovery/external_reals.json`
- Selective arena strategy artifacts:
  - `arena/strategy_results/best_strategies_summary.csv`
  - `arena/strategy_results/extremity_gate_full.csv`

## Tool limitation that matters

Round 2 did not have live MCP access to `wandb` or `training-data-viewer`. Their intended role was replaced with local repo artifacts and local viewer discovery cache files. That means:

- Established claims in this package are grounded in repo/runtime/data artifacts that are present locally.
- Claims that would require fresh remote run inspection remain explicitly labeled as plausible or still unknown.
