# Packet <ID>  ·  <one-line identity>

> **Template contract**: fill every section. If a section truly has no content, write `*(none — reason)*` rather than deleting the heading. Keep the status-card table intact.

## Status card

| Field | Value |
|---|---|
| Dates | YYYY-MM-DD → YYYY-MM-DD |
| Slots | N (e.g. RLPX_01, RLPX_02, …) |
| Headline lever | what this packet was actually testing |
| Leader slot | e.g. RLPX_0Y |
| Leader metric | e.g. value_composite=0.7442 |
| Verdict | ✅ confirmed / ⚠️ muddled / ❌ failed / 🔬 superseded / 🟡 in-flight |
| Next-packet decision | one line — what this packet chose for the next |
| Themes touched | links to `../threads/*.md` |

## Configuration

- What changed vs the prior packet (one bullet per lever)
- Control slot definition
- Variants tested (bullet list — slot → what it probed)
- Links to yamls: `experiments/phase2_round13/R13_<ID>_*.yaml`

## Results at the time

- Leading metric readouts (value_composite / AUC / scorecard verdict)
- Leader vs control delta
- Suite-level numbers that mattered (real-pool FPR, fake recall, pool-specific gaps)
- Links to scorecards / analysis artifacts / checkpoint maps

## Conclusions drawn in-session

> **Required structure (added 2026-05-07).** New agents read the FACTS subsection first to form an independent view. Read the OPINION subsection only after the user authorizes (or after independent view is committed). This separation is what makes the retro replayable across agent sessions.

### Factual evidence (read first)

- Pointers to `*_FACTS_<date>.md` docs in the eval folder. List by canonical name (RESULTS_FACTS, RESULTS_F1_F5_FACTS, DEEP_DIVE_FACTS, FOLLOWUPS_FACTS).
- Headline numerical findings — close-criterion verdicts, scorecard outcomes, surprising deltas.
- Each line cites the specific FACTS doc + section that backs it.
- No interpretation, no proposal language ("we should", "this means").

### In-session opinion (read second, with skepticism)

- Pointer to `AGENT_PROPOSAL_<date>.md` in the eval folder.
- One paragraph summary of the agent's mechanism claim (NOT verdict).
- **Required**: name any prior framings retracted mid-session and where the retraction lives. If the agent self-corrected, link the §4-style self-correction log.
- Caveat that the next agent should form their own view from the FACTS subsection BEFORE engaging with this section.

## Eval folder

The packet's eval folder lives at `analysis/<packet>_eval_<date>/` and follows the canonical contract in `../eval_folder_template.md`. Required entries:

- **FACTS docs** (factual; safe-to-read for forming independent view): `RESULTS_FACTS_<date>.md`, `RESULTS_F1_F5_FACTS_<date>.md`, `DEEP_DIVE_FACTS_<date>.md`, `FOLLOWUPS_FACTS_<date>.md`.
- **OPINION doc** (single, clearly-marked): `AGENT_PROPOSAL_<date>.md`.
- **Sub-investigations**: per-topic subdir (`<topic>/`) with its own `RESULTS_<topic>_FACTS_<date>.md` + scripts + CSVs.

Cross-reference rule: every claim in a FACTS doc cites a raw artifact (CSV, script, GCS object) by relative path; every load-bearing claim in the OPINION doc cites a FACTS doc.

## Retrospective

- Which later packet challenged or confirmed these conclusions (date each entry)
- If reinterpreted: what we now believe + why, with file citation
- Preprocessing-parity note: are this packet's numbers pre- or post- commit `855871e` (INTER_LINEAR fix)?
- Cross-reference to `../threads/*.md` where the story continues
- Use `*(YYYY-MM-DD — placeholder)*` to mark a slot a future agent should fill in when downstream evidence lands

## Source files

- **Handoffs**: `docs/relaunch_handoffs/...` (with `:line_number` where it anchors a conclusion)
- **Yamls**: `experiments/phase2_round13/R13_<ID>_*.yaml`
- **Scorecards / analysis**: `arena/...`, `analysis/...`
- **Memory pointers**: entries from `~/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/` if load-bearing
