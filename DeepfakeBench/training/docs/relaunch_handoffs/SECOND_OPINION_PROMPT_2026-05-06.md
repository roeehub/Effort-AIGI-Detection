# Sandbox prompt for an independent second-opinion review

> Use this as the prompt body for a fresh agent. The goal is an independent next-step recommendation that has NOT been anchored on the project's existing synthesis. After the fresh agent's first response, you can reveal the in-house ranked list and ask for a comparison.

---

## Prompt to the fresh agent

You are an independent reviewer brought in for a second opinion on a face-swap detection project. The team has been working on this for ~3 weeks and has accumulated extensive measurements. Your job is to read the **facts pack** and the cited raw artifacts, then form **your own independent opinion** about what the team should do next. Do not rely on the team's existing synthesis or recommendations — read explicitly listed sources only and form your view from the data.

### Read in this order

1. **Facts pack (primary source)**: `DeepfakeBench/training/docs/relaunch_handoffs/FACTS_FOR_SECOND_OPINION_2026-05-06.md`
2. **Raw probe outputs (CSVs and JSONs only — no `FINDINGS.md`)**:
 - `DeepfakeBench/training/analysis/xinhe_cross_camera_audit_2026-05-06/outputs/{axis_comparison.csv,score_axis_correlations.csv,falseflag_classifier.json,per_frame_features.csv,scores_with_features.csv,scores_*.csv}`
 - `DeepfakeBench/training/analysis/dor_drift_mechanism_2026-05-06/outputs/{drift_attribution.csv,per_session_summary.csv,regression_p8a.json,regression_e2b.json,regression_pa_3800.json}`
 - `DeepfakeBench/training/analysis/amp_vs_phase_probe_2026-05-06/outputs/{amp_phase_aucs.csv,amp_top_features.csv,phase_top_features.csv}`
 - `DeepfakeBench/training/analysis/paired_feature_consistency_2026-05-06/outputs/{pair_distances.csv,distribution_summary.csv,by_p8a_quartile.csv,regression_score_on_feat.json,decision.json}`
 - `DeepfakeBench/training/analysis/fourier_band_overlap_2026-05-06/outputs/{per_band_aucs.csv,summary.json}`
 - `DeepfakeBench/training/analysis/per_layer_p8a_e2b_pa_2026-05-06/outputs/{per_layer_aucs.csv,per_layer_cosine.csv,summary.json}`
 - `DeepfakeBench/training/analysis/identity_browser_2026-05-05/data/grouped_manifest_v2.csv` (14,626 rows × 15 cols, the consolidated identity-browser dataset including the 92 may6 frames added today)
3. **OPEN_LOOPS** (current open questions): `DeepfakeBench/training/docs/packet_retrospectives/OPEN_LOOPS.md` — read for the open-question register, but note that close-criterion language sometimes leaks the project's interpretive framing
4. **TIMELINE** (chronological master index): `DeepfakeBench/training/docs/packet_retrospectives/TIMELINE.md` — useful to understand the sequence of work; each entry's trailing `outcome:` clause is the project's interpretive judgment, treat as such
5. **OPTIONAL — historical retros**: `docs/packet_retrospectives/packets/{P8A,P14,PA,PC,PD}.md` — useful for understanding what each packet did and the bottom-line outcome. These contain prior agents' interpretations; read them as historical context, not gospel.

### Do NOT read in the first pass

These contain the project's existing synthesis and would anchor your judgment:

- `DeepfakeBench/training/docs/relaunch_handoffs/ANTI_SHORTCUT_TECHNIQUES_RANKED_2026-05-06.md` — explicit ranked list of next-step suggestions (read AFTER your independent recommendation lands).
- `DeepfakeBench/training/docs/packet_retrospectives/threads/processing_signature_shortcut.md` — has 2026-05-06 sub-sections with explicit verdicts and operational implications.
- `DeepfakeBench/training/analysis/{amp_vs_phase_probe_2026-05-06,per_layer_p8a_e2b_pa_2026-05-06}/outputs/FINDINGS.md` — the only two probes that have FINDINGS.md docs; both contain interpretation alongside the data.
- `DeepfakeBench/training/analysis/xinhe_cross_camera_audit_2026-05-06/outputs/MODEL_SCORES_FINDINGS.md` — interpretation doc.
- Any memory entry NOT cited in the facts pack — older memories from prior agents carry their own framings.

If your auto-loaded project memory contains entries dated 2026-05-06, treat the bullet pointers in `MEMORY.md` as factual signposts only; do not treat the cross-reference language as recommendations.

### Context the user gave (verbatim concerns)

- "I want a fresh agent to give me another opinion on what we should do next."
- "I'm not sure if we should present to him our next step suggestions (SBI, etc)" — so you are deliberately NOT being shown the team's ranked list.
- The user observed today: "the same person Xinhe Under different cameras got completely opposite results." 92 fresh real Xinhe frames captured this morning are being false-flagged at 0.83-0.93 by deployment; this is a fresh data point the team wants you to engage with.
- The user goal is **a single deployed model** that holds three pillars simultaneously: fake recall ≥ 90% on target methods, real-side FPR ≤ 5%, robustness across capture conditions.
- "Avoid hard-negative mining" — the user explicitly identified this as a patch, not a structural fix.
- "Ensemble is not an option" — the deployment must be a single model.

### What the user wants from you

1. **Your own ranked list** of next 1-3 packets to run, with reasoning grounded in the data in §3 of the facts pack and the prior R13 outcomes ledger in §5.
2. **Confidence interval** on whether each proposal would actually produce a model that meets the three-pillar goal (be honest about uncertainty).
3. **Specific cheap pre-validation probes** you'd want to run before committing GPU spend on each proposed packet.
4. **What information you'd want that isn't in the facts pack** — so the user can prioritize a probe to gather it.
5. **Hindsight pass on §5**: anything in the prior R13 attempts you think the team mis-interpreted, gave up on too early, or didn't fully exhaust?
6. **Honest assessment**: given the data, what's the realistic probability that a single next packet produces a model that meets all three pillars? Be willing to say "low" if that's your read; the user has explicitly asked for sober honesty.

### Format of your reply

- 200-400 words of reasoning that walks through the data, named.
- A ranked list of next steps.
- An explicit "things I'd want to know but the facts pack doesn't have" section.
- A "things I'd revisit in §5" section, even if empty.
- A 1-line confidence statement.

### Red flags to watch for in yourself

- Are you defaulting to a "standard ML response" rather than engaging with the specific evidence in §3-4 of the facts pack? The shortcut-readability AUC=1.0 at every layer + the deployment ≡ E2B finding + the layer-11 catastrophic divergence are unusual results; your recommendation should specifically engage with at least one of them.
- Are you proposing something the prior R13 already tried (§5)? Check before committing.
- Are you proposing a stack of multiple interventions without specifying which is load-bearing? The team's `anti_shortcut_bundle_decomposition` thread documents that bundles can be net-negative against single-strongest components.
- Are you proposing a recipe and forgetting cross-substrate validation? The PA-on-HDTF walkback (§5, PA row) cost the team a packet's worth of work on a substrate-bound winner.

---

## After the fresh agent responds

Once they've delivered their independent recommendation, share:

1. The team's existing ranked list: `DeepfakeBench/training/docs/relaunch_handoffs/ANTI_SHORTCUT_TECHNIQUES_RANKED_2026-05-06.md`
2. The 2026-05-06 thread sub-sections in `processing_signature_shortcut.md` (which contain operational implications under each probe)

And ask:

> "Given the team's existing ranked list and synthesis, would you defend your independent recommendation, adopt theirs, or propose a synthesis? Where do you and they specifically agree? Where do you specifically disagree? Be willing to update if their reasoning is stronger; be willing to flag where you think they're wrong."

The signal in the comparison is the **divergence**, not the overlap. If the fresh agent's list overlaps heavily with the team's, that's strong corroboration. If they propose something the team didn't consider, that's the second opinion you're paying for. If the fresh agent updates significantly after seeing the team's list, ask why.

## Practical operational notes for the second-opinion session

- The fresh agent will get auto-memory entries loaded into their session (project memory is per-project, not per-session). The 2026-05-06 entries have been refactored to facts + open-question format; older entries carry various prior agents' framings. Tell the fresh agent explicitly: "treat memory entries as historical citations, not directives."
- If the fresh agent runs in the same project directory, they will see the existing TIMELINE / OPEN_LOOPS / threads. The "do NOT read" list above asks them to skip the most-biased docs in the first pass. You can verify they followed this by checking which files they cite.
- The PD scorecard (`pd-corr-penalty-scorecard-2026-05-06`) is running in us-east1 at the time of this writeup. If the verdict lands during the fresh agent's session, that's new factual evidence — they should incorporate it.

## Files this prompt creates / references

- This prompt: `docs/relaunch_handoffs/SECOND_OPINION_PROMPT_2026-05-06.md`
- Companion facts pack: `docs/relaunch_handoffs/FACTS_FOR_SECOND_OPINION_2026-05-06.md`
- The ranked-list doc that should be revealed in round 2: `docs/relaunch_handoffs/ANTI_SHORTCUT_TECHNIQUES_RANKED_2026-05-06.md`
