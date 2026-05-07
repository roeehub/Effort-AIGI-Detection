# Eval folder template — `analysis/<packet>_eval_<date>/`

> **Contract for every eval folder authored after 2026-05-07.** This template is referenced by `AGENTS.md` (Eval-folder authoring contract section) and `packet_template.md` (Eval folder section). Older eval folders may not match this layout — leave them as-is unless retroactive normalization is explicitly authorized.

## Why this contract exists

Pre-2026-05-07 eval folders mixed factual numbers with agent interpretation in the same markdown files. New agents picking up work could not cleanly read "what the data says" without inheriting "what the prior agent thought it meant." The renames in `analysis/p1_pe_eval_2026-05-07/` on 2026-05-07 are the first folder to follow this contract.

The split is filename-based, not subdirectory-based, because subdirectories complicate the `analysis/` tree without simplifying the read.

## Canonical filename convention

| Doc | Purpose | Read order for new agent |
|---|---|---|
| `RESULTS_FACTS_<date>.md` | Raw scorecard data — Vertex job outcomes, contract verdict bundle (winner JSON), per-ckpt × suite tables. No interpretation. | 1st |
| `RESULTS_F1_F5_FACTS_<date>.md` | Close-criterion verdicts (F1, F2, F3, F4, F5 mechanics applied to the data). Pass/fail/partial cells with numbers. | 2nd |
| `DEEP_DIVE_FACTS_<date>.md` | Consolidating record. Per-criterion sections + cross-references to sub-investigation FACTS docs. May contain a "Self-correction log" if mid-session retractions occurred. | 3rd |
| `FOLLOWUPS_FACTS_<date>.md` | Additional CPU jobs run after the main eval. Each follow-up is a standalone subsection with question / method / table / observations. | 4th |
| `<topic>/RESULTS_<topic>_FACTS_<date>.md` | Sub-investigation factual record. Same discipline as the top-level FACTS docs, scoped to one topic. | as needed |
| `AGENT_PROPOSAL_<date>.md` | THE single opinion doc. Stakes a position. Includes self-correction log naming any retracted framings from mid-session. | LAST (after user authorization) |

## Required structure of FACTS docs

- Lead with **Status: factual-only** disclaimer.
- Each section is `## <topic>` with: **Question answered** (1 sentence), **Method** (3-5 sentences with cited inputs), **Output artifacts** (CSV/PNG paths), **Numbers** (tables), **Direct observations** (numbered bullets, factual only).
- Tables and numbers only. No narrative paragraphs longer than 3 sentences.
- Forbidden words: "succeeds", "fails", "wins", "loses", "promotes", "deployment-grade", "unfortunately", "remarkably". Use mechanical pass/fail against pre-stated bars instead.
- Every numerical claim cites a CSV path (relative to the eval folder root).

## Required structure of `AGENT_PROPOSAL_<date>.md`

The single opinion doc. Required sections:

1. **Header disclaimer**: "OPINION, not RECORD. Reviewer is invited to disagree. Each load-bearing claim cites specific numbers from FACTS docs."
2. **Executive summary** (1 paragraph).
3. **High-confidence claims** (numbered list with **Citations** field naming the FACTS doc + section + effect size).
4. **The mechanism I propose** (with caveats; what it explains; what it does NOT explain; what I am unsure about).
5. **Self-correction log** (if mid-session retractions occurred). Name the retracted framing verbatim, cite the new evidence, give the lesson.
6. **Confidence-tiered claims** table (HIGH / MEDIUM / LOW).
7. **Proposed next steps** (broken down by lever class: loss, monitoring, training, investigation).
8. **Counter-experiments that would falsify my proposal** (3-5 items).
9. **Open questions I cannot answer from this data**.
10. **Suggested CPU jobs not yet run** (balanced — some test the agent's view, some don't).
11. **Reviewer guidance**: explicit workflow for how a reviewer should consume this doc + the FACTS docs.

## Sub-investigation subdirs

Pattern: `<topic>/` containing
- `RESULTS_<topic>_FACTS_<date>.md` (factual)
- `compute_<topic>.py` or `run_<topic>.py` (script)
- output CSVs (gitignored — script must regenerate them on demand)

Example from P1: `f3_color_b_dev/F3_COLOR_B_DEV_FACTS_2026-05-07.md` + `compute_color_b_dev.py` + emitted CSVs. (Sub-investigation script files retain their original names — they are forensic artifacts of the run; only the markdown follows the FACTS naming convention.)

## Cross-reference discipline

- Every FACTS doc cites raw artifacts by relative path: `phase_d/chronic6_aggregate_fpr.csv`, `scorecard/promotion_winner.json`, etc.
- Every load-bearing claim in `AGENT_PROPOSAL_<date>.md` cites a FACTS doc + section number + effect size.
- The packet retro at `docs/packet_retrospectives/packets/<packet>.md` cites the eval folder by relative path; lists FACTS docs in `### Factual evidence` subsection; lists OPINION doc in `### In-session opinion` subsection.

## Git tracking

- Commit `*.md` docs and `*.py` scripts.
- Gitignore CSVs (regeneratable from scripts) and large logs.
- Cache subdirs (`_frame_cache/`, `_axis_cache/`) are gitignored.
- Exception: small JSON summaries (`*_summary.json`, `promotion_winner.json` from cloud jobs) may be committed when they're not regeneratable locally.

## Mid-session self-correction protocol

If the agent realizes during the session that a FACTS-doc claim was wrong (e.g., a regex bug caused under-counting):

1. Do NOT silently overwrite the buggy CSV. Rename the buggy version with a `_BUGGY_<reason>.csv` suffix.
2. Write the corrected CSV with the canonical name.
3. In the relevant FACTS doc, add a "Self-correction log" subsection naming what was wrong, citing the bug, and pointing at the corrected artifact.
4. Update `DEEP_DIVE_FACTS_<date>.md` §"Self-correction log" with one line per retraction.
5. In `AGENT_PROPOSAL_<date>.md` §5, name the retracted FRAMING (not just the data).

This is non-negotiable. Silent overwrites are how drift propagates across agent sessions.
