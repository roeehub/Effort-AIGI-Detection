# AGENTS.md — Knowledge-Base Read & Update Protocol

> **Mandatory entry point** for any agent (build agent or working agent) that touches `docs/packet_retrospectives/` after the Slice-0 build of 2026-04-29.
>
> **Authoritative rule:** when memory and threads disagree, **threads win**. Memory is a fast index that lags; threads are deliberated synthesis. The agent that observes drift updates the lagging surface in the same session.

---

## Why this protocol exists

The R13 program accumulated ~6 weeks of context across handoffs, auto-memory, code comments, the master plan log, and `analysis/` dirs. Knowledge was well-captured but poorly indexed. The whack-a-mole pattern that triggered this build: the same contract-scorer policy bug was "fixed" three times in 6 days (2026-04-23, 2026-04-27, 2026-04-29) without ever being committed, because no canonical surface tracked the fix as an open issue.

This protocol structurally prevents rediscovery: every open issue lives in a `### Open loop` block in a `threads/*.md` file, surfaces in `OPEN_LOOPS.md`, and is reaffirmed every session via the read protocol below.

---

## What lives here

```
docs/packet_retrospectives/
├── AGENTS.md                  ← this file (entry point)
├── README.md                  ← executive index of the wiki
├── TIMELINE.md                ← chronological master index (one line per session/packet)
├── OPEN_LOOPS.md              ← ⚙ MECHANICALLY GENERATED — do not hand-edit
├── BUILD_SCAFFOLD.md          ← chronological map for slice agents (build-time)
├── thread_template.md         ← canonical structure for a thread doc
├── packet_template.md         ← canonical structure for a per-packet retro
├── packets/                   ← per-packet retros (RLP1.md, RLP2.md, …)
├── threads/                   ← cross-cutting topic docs (the cleanest pattern)
└── tools/
    └── regenerate_open_loops.py   ← reads structured open-loop blocks from threads/, writes OPEN_LOOPS.md
```

Adjacent surfaces this protocol references but does not own:

- `docs/relaunch_handoffs/` — 35 dated session handoffs. **Read-only** for build/working agents. Treat as primary source material; do not reorganize.
- `april-26-training-master-plan-v2.LOG.md` — chronological session log; reference, not a wiki surface.
- `~/.claude/projects/-Users-roeedar-Documents-repos-Effort-AIGI-Detection-DtectVision/memory/` — auto-memory, loads every session. Has frontmatter (`status`, `wiki_ref`, `last_verified`) that links to threads here.

---

## Working agent read protocol

Run these steps **before** reading the user's task description. They are ordered so that the most-likely-relevant context is loaded first.

1. **Read `AGENTS.md`** (this file). The protocol may have evolved since your last session.
2. **Read `OPEN_LOOPS.md`.** This is the current open-issue inventory. If the user's task touches an open loop's domain, do not redo a fix that is already in progress; consult the owning thread first.
3. **Read the last ~10 entries of `TIMELINE.md`.** Gives you the recent session arc — what was decided, what's pending, what just landed.
4. **Identify the relevant thread(s)** for the user's task. Read at minimum the `## Current stance` and `## Open loops` sections of each. If the task spans multiple threads, read all of them.
5. **Run a working-tree-diff check** before proposing or applying any fix:

   ```bash
   git status
   git diff --stat
   ```

   If a file mentioned in the relevant thread or in `OPEN_LOOPS.md` has uncommitted hunks, treat it as **"fix in progress, do not redo without context"**. Read the diff. Read the thread's open-loop entry. The fix may already exist — your job is to commit/test it, not to re-derive it.

   This step exists because the contract-policy bug was rediscovered and "re-fixed" three times. The first fix was sitting uncommitted in the working tree the whole time.

6. **Then read the user's specific task.** Frame your response in terms of the open loops you just loaded.

---

## Working agent update protocol

Run these steps **before** declaring a session complete (i.e., before handing back to the user). The order matters — `regenerate_open_loops.py` is last because it consumes the structured blocks you may have just edited.

1. **Promote new findings to threads.** If you discovered something the existing threads don't capture:
   - If a thread doc exists for the topic → extend it (add a new dated section). Cite the source (file:line, memory entry, handoff doc, or commit hash).
   - If no thread exists → create one using `thread_template.md`. The first sentence of any new section must be a citation.
   - **Do not** leave findings only in handoffs. Handoffs are baton-passing artifacts; threads are the canonical record.

2. **Update structured open-loop blocks** for any status change. Block format (parsed by `regenerate_open_loops.py`):

   ```
   ### Open loop: short-id-here
   status: open | in-progress | resolved | superseded
   severity: critical | high | medium | low
   first_seen: YYYY-MM-DD
   last_verified: YYYY-MM-DD
   close_criterion: what would resolve this
   ```

   When a loop changes state (open → in-progress → resolved), update `status:` and bump `last_verified:`. Do **not** delete resolved loops — leave them in the thread; the regenerator groups them into a "resolved" section. Resolved loops are evidence that the issue was tracked, not noise.

3. **Append a `TIMELINE.md` entry** for the session. Format: one line per session — `YYYY-MM-DD — what happened — owning thread(s) — key outcome`. Append-only; do not edit prior entries.

4. **Update memory frontmatter** for any memory entry your session touched:

   ```yaml
   ---
   status: open | in-progress | resolved
   wiki_ref: docs/packet_retrospectives/threads/<thread_name>.md
   last_verified: YYYY-MM-DD
   ---
   ```

   Bump `last_verified:` to today's date. Update `status:` if it changed. Do not edit memory entries unrelated to your session.

5. **Run the regenerator** from `docs/packet_retrospectives/`:

   ```bash
   python tools/regenerate_open_loops.py
   ```

   This rewrites `OPEN_LOOPS.md` from the structured blocks in `threads/*.md`. The script will warn (but not fail) if any open-loop block has `last_verified:` older than 60 days — re-verify those entries when you encounter them.

6. **The handoff doc is for cross-agent baton-passing only.** Findings must also be promoted to threads, not left in handoffs alone. A handoff dies when the next agent reads it; a thread persists.

---

## Build-agent special rules (Slice 1..N)

This protocol applies recursively. A build agent on its second session reads `AGENTS.md` first, just like a working agent.

In addition, build agents must:

1. **Validate the prior slice before processing their own.** Re-read the prior slice's threads' "Open loops" sections + "Current stance" paragraphs. If agent N–1 made a claim that later evidence contradicts, add a dated `### DEBATE — YYYY-MM-DD` block to the thread. **Do not delete or rewrite the prior agent's text.** If agent N–1 marked an open loop "resolved" but the evidence doesn't support it, reopen it (set status back to `open`, bump `last_verified:`).

2. **Process the slice per `BUILD_SCAFFOLD.md` entry.** Read all listed source material; identify topics; author or extend thread docs; cite file:line for every claim.

3. **Mark slice complete in `BUILD_SCAFFOLD.md`** with date + agent identifier; mark next slice ready. Then **stop**. Do not start the next slice. Hand back to user.

---

## Thread consolidation pattern

Topics merge over time. If a Slice-N agent finds two threads covering substantially the same question, **consolidate**:

1. Pick the canonical thread (typically the larger/older one).
2. Move the content from the deprecated thread into the canonical thread, preserving dated section headers so the timeline is intact.
3. Replace the deprecated thread's contents with a stub:

   ```markdown
   # Thread: <old name> (consolidated)

   This thread was consolidated into [<canonical thread>](./<canonical_thread>.md) on YYYY-MM-DD.
   Reason: <one line — why these were really the same question>.

   Search this name in the canonical thread for the migrated content.
   ```

4. Update any cross-links in `README.md`, `OPEN_LOOPS.md` (will regenerate from blocks — make sure the blocks now live in the canonical thread), `TIMELINE.md`, and the `wiki_ref:` frontmatter of memory entries that pointed at the deprecated path.

5. Run `tools/regenerate_open_loops.py` after consolidation. The new `source_thread:line` references should point at the canonical file.

The stub stays — never delete a thread file. Old `wiki_ref:` references in memory or in commit messages will still resolve, just to a redirect.

---

## Anti-whack-a-mole mechanisms (why this works)

1. **Working-tree diff check** is step 5 of the read protocol — would have caught the contract-bug whack-a-mole at the first session that triggered it.
2. **Memory `status: in-progress`** flags an unfinished fix; the agent must read the entry's `wiki_ref` thread before touching the same area.
3. **Open-loop blocks** in threads accumulate across slices — three "contract policy fix attempted" entries in the same thread is visually obvious.
4. **Overlap validation** during build: each slice agent re-reads the prior slice's open loops, so unresolved issues are reaffirmed rather than fading.
5. **`last_verified:` decay**: claims older than 60 days are flagged stale by `regenerate_open_loops.py` (warns but does not auto-mark resolved).

---

## Authority order (one rule)

Memory and threads must agree. The build process makes them agree. The update protocol keeps them agreeing. **In the rare case of drift, threads win**, because they are deliberated synthesis and memory is a fast index that lags. The agent that observes drift updates the lagging surface in the same session.

If a memory entry says X and the thread says ¬X with citation: trust the thread. Update the memory entry's `status:` and `last_verified:`, and append a dated note in the thread acknowledging the drift was caught.
