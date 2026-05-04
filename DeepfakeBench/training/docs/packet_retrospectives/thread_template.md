# Thread: <one-line topic>

> **Template contract**: fill every section. If a section truly has no content for this topic, write `*(none — reason)*` rather than deleting the heading. Threads are deliberated synthesis: every claim must cite a source (file:line, memory entry, handoff doc, or commit hash).
>
> **Canonical example**: [`threads/promotion_contract_evolution.md`](./threads/promotion_contract_evolution.md). Read it before authoring a new thread — it is the cleanest existing instance of this format.
>
> **Read protocol**: see `AGENTS.md`. **Update protocol**: see `AGENTS.md`. Run `tools/regenerate_open_loops.py` after editing any `### Open loop:` block.

## The question

One paragraph stating the question this thread is answering. The question is more durable than the answer; framing it well lets the thread accumulate updates without becoming a chronological mess.

## Initial belief

What the team believed when this question first surfaced. Cite the packet/handoff/commit where the belief was first recorded. Do not edit this section once written — it is a historical anchor.

## What changed our mind

Bullet list of evidence that moved the team's understanding, in chronological order. Each bullet must cite a source:

- **<Date or packet ID>**: <one-sentence finding> (`<file:line>` or memory entry name or commit hash).
- **<Date or packet ID>**: <one-sentence finding> (`<source>`).

This section grows over time. New bullets append to the end. Do **not** rewrite prior bullets — if a later finding contradicts an earlier one, add a new bullet documenting the contradiction; the prior bullet stays for historical fidelity.

## Current stance (YYYY-MM-DD)

The current synthesized answer to the thread's question, dated. One paragraph or a short numbered list. When the stance changes materially, replace this section's date and content (do not preserve the old stance verbatim — that history lives in `## What changed our mind`).

If a slice agent disagrees with the prior agent's stance, **do not overwrite**. Add a `### DEBATE — YYYY-MM-DD` subsection below the stance with the disagreement, citing evidence. The user resolves debates; agents do not unilaterally rewrite the stance.

## Packet timeline

Compact map of which packets touched this thread and what they contributed. One line per packet:

- [PacketID](../packets/PacketID.md) — one-sentence contribution to this thread.

The timeline lets a fresh reader follow the chronological development without reading all the packet retros.

## Evidence locations

Concrete files, scorecards, scripts, memory entries, and handoff docs that anchor the claims in this thread. Group by source type:

- `<file path>:<line range>` — what it shows.
- Memory: `<memory entry name>` — what it captures.
- Handoffs: `docs/relaunch_handoffs/<file>.md:<line>` — what it documents.
- Commits: `<short hash>` — what it changes.

This section is the citation pool — every claim above should reference something here (or cite directly inline).

## Open loops

One or more **structured open-loop blocks**. These are parsed by `tools/regenerate_open_loops.py`; the format must be exact.

```
### Open loop: short-id-kebab-case
status: open
severity: high
first_seen: 2026-04-29
last_verified: 2026-04-29
close_criterion: what observable would resolve this loop
```

Format rules:

- The heading must start with `### Open loop: ` (three hashes, the literal word "Open loop", colon, space). The id is kebab-case.
- The five fields below the heading appear on consecutive lines, each as `<field>: <value>`. Order is required: `status`, `severity`, `first_seen`, `last_verified`, `close_criterion`.
- `status` ∈ `{open, in-progress, resolved, superseded}`.
- `severity` ∈ `{critical, high, medium, low}`.
- Dates are `YYYY-MM-DD`.
- `close_criterion` is one line — the observable that would let a future agent mark the loop `resolved`.
- After the structured block, you may add free-form prose about the loop (context, what's been tried, leads). The regenerator only reads the structured fields.

When a loop's status changes, edit the block in place and bump `last_verified:`. **Do not delete resolved loops** — they stay in the thread as evidence the issue was tracked. The regenerator surfaces them in a separate section of `OPEN_LOOPS.md`.

If `last_verified:` is older than 60 days, the regenerator flags the entry as **stale** in its output (warning only — the agent re-verifies and bumps the date, or escalates if the loop is unresolved and forgotten).
