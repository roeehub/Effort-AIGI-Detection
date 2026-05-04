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

- Bullet conclusions, each traceable to a handoff doc line or a convmem session quote
- Short verbatim quotes (≤25 words) when they carry non-obvious reasoning
- **Session IDs**: `<ID1>`, `<ID2>` (convmem-resumable)

## Retrospective (as of 2026-04-24)

- Which later packet challenged or confirmed these conclusions
- If reinterpreted: what we now believe + why, with file citation
- Preprocessing-parity note: are this packet's numbers pre- or post- commit `855871e` (INTER_LINEAR fix)?
- Cross-reference to `../threads/*.md` where the story continues

## Source files

- **Handoffs**: `docs/relaunch_handoffs/...` (with `:line_number` where it anchors a conclusion)
- **Yamls**: `experiments/phase2_round13/R13_<ID>_*.yaml`
- **Scorecards / analysis**: `arena/...`, `analysis/...`
- **Memory pointers**: entries from `~/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/` if load-bearing
