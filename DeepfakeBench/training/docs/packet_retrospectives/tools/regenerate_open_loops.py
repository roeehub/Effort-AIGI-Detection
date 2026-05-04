#!/usr/bin/env python3
"""regenerate_open_loops.py

Walks `docs/packet_retrospectives/threads/*.md`, parses every structured
`### Open loop:` block, and writes a sorted, grouped, grep-friendly inventory
to `docs/packet_retrospectives/OPEN_LOOPS.md`.

Run from the `docs/packet_retrospectives/` directory:

    python tools/regenerate_open_loops.py

Stdlib only — no external dependencies.

The structured block format (per `thread_template.md`) is:

    ### Open loop: short-id-kebab-case
    status: open | in-progress | resolved | superseded
    severity: critical | high | medium | low
    first_seen: YYYY-MM-DD
    last_verified: YYYY-MM-DD
    close_criterion: <one-line description>

The fields appear on consecutive lines immediately after the heading.
Free-form prose may follow the block; this script ignores it.

Behavior:
  - Open + in-progress entries are surfaced first, then resolved, then
    superseded. Within each group, sorted by severity (critical first)
    then by first_seen (oldest first).
  - Entries with `last_verified` older than 60 days are flagged STALE
    (warning only — never auto-resolved).
  - Each entry shows id, status, severity, first_seen, last_verified,
    close_criterion, and the source thread:line where the block lives.
  - Empty thread set or threads without any blocks: produces a valid
    OPEN_LOOPS.md with an empty body section. Does not crash.
"""

from __future__ import annotations

import datetime as _dt
import re
import sys
from pathlib import Path

VALID_STATUS = {"open", "in-progress", "resolved", "superseded"}
VALID_SEVERITY = {"critical", "high", "medium", "low"}

# Severity sort order (lower = more urgent).
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

# Group display order: open and in-progress at top, then resolved, then superseded.
GROUP_ORDER = [
    ("open", "Open"),
    ("in-progress", "In progress"),
    ("resolved", "Resolved"),
    ("superseded", "Superseded"),
]

STALE_DAYS = 60

OPEN_LOOP_HEADING_RE = re.compile(r"^###\s+Open loop:\s*(.+?)\s*$")
FIELD_RE = re.compile(r"^([a-z_]+):\s*(.*?)\s*$")


def _project_root() -> Path:
    """Locate the packet_retrospectives root regardless of where the script is invoked from.

    Resolves to the directory containing `threads/` and `tools/`.
    """
    script_path = Path(__file__).resolve()
    # script lives at <root>/tools/regenerate_open_loops.py
    return script_path.parent.parent


def _today() -> _dt.date:
    return _dt.date.today()


def _parse_date(raw: str) -> _dt.date | None:
    """Parse YYYY-MM-DD; return None on malformed input."""
    raw = raw.strip()
    try:
        return _dt.date.fromisoformat(raw)
    except ValueError:
        return None


def _parse_thread_file(path: Path) -> tuple[list[dict], list[str]]:
    """Extract structured open-loop blocks from a thread file.

    Returns (blocks, warnings). A block is a dict with keys: id, status,
    severity, first_seen, last_verified, close_criterion, source_path,
    source_line. Warnings are human-readable strings about malformed
    blocks (printed to stderr; never raised).
    """
    blocks: list[dict] = []
    warnings: list[str] = []

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        warnings.append(f"{path}: could not read ({exc!r}); skipping")
        return blocks, warnings

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        match = OPEN_LOOP_HEADING_RE.match(line)
        if not match:
            i += 1
            continue

        heading_lineno = i + 1  # 1-indexed
        loop_id = match.group(1).strip()
        if not loop_id:
            warnings.append(f"{path}:{heading_lineno}: open-loop heading missing id; skipping")
            i += 1
            continue

        # Read the next 5 lines as field lines, allowing blank lines between
        # heading and fields. Be lenient: scan up to 12 lines past the
        # heading looking for the 5 fields, in order.
        fields_required = ["status", "severity", "first_seen", "last_verified", "close_criterion"]
        collected: dict[str, str] = {}
        scan_idx = i + 1
        scan_limit = min(len(lines), i + 14)
        while scan_idx < scan_limit and len(collected) < len(fields_required):
            field_line = lines[scan_idx]
            if field_line.strip() == "":
                scan_idx += 1
                continue
            field_match = FIELD_RE.match(field_line)
            if not field_match:
                # Stop scanning — we've left the structured block.
                break
            field_name, field_value = field_match.group(1), field_match.group(2)
            if field_name not in fields_required:
                # Unknown field — stop scanning, treat as end of block.
                break
            if field_name in collected:
                # Duplicate field — first wins; warn.
                warnings.append(
                    f"{path}:{scan_idx + 1}: duplicate field '{field_name}' for loop "
                    f"'{loop_id}'; ignoring duplicate"
                )
                scan_idx += 1
                continue
            collected[field_name] = field_value
            scan_idx += 1

        missing = [f for f in fields_required if f not in collected]
        if missing:
            warnings.append(
                f"{path}:{heading_lineno}: open-loop '{loop_id}' missing field(s) "
                f"{', '.join(missing)}; skipping"
            )
            i = scan_idx
            continue

        status = collected["status"].strip().lower()
        severity = collected["severity"].strip().lower()
        first_seen_raw = collected["first_seen"].strip()
        last_verified_raw = collected["last_verified"].strip()
        close_criterion = collected["close_criterion"].strip()

        if status not in VALID_STATUS:
            warnings.append(
                f"{path}:{heading_lineno}: open-loop '{loop_id}' has invalid status "
                f"'{status}'; valid: {sorted(VALID_STATUS)}"
            )
        if severity not in VALID_SEVERITY:
            warnings.append(
                f"{path}:{heading_lineno}: open-loop '{loop_id}' has invalid severity "
                f"'{severity}'; valid: {sorted(VALID_SEVERITY)}"
            )

        first_seen = _parse_date(first_seen_raw)
        last_verified = _parse_date(last_verified_raw)
        if first_seen is None:
            warnings.append(
                f"{path}:{heading_lineno}: open-loop '{loop_id}' has malformed "
                f"first_seen '{first_seen_raw}' (expected YYYY-MM-DD)"
            )
        if last_verified is None:
            warnings.append(
                f"{path}:{heading_lineno}: open-loop '{loop_id}' has malformed "
                f"last_verified '{last_verified_raw}' (expected YYYY-MM-DD)"
            )

        blocks.append({
            "id": loop_id,
            "status": status,
            "severity": severity,
            "first_seen": first_seen_raw,
            "first_seen_date": first_seen,
            "last_verified": last_verified_raw,
            "last_verified_date": last_verified,
            "close_criterion": close_criterion,
            "source_path": path,
            "source_line": heading_lineno,
        })

        i = scan_idx

    return blocks, warnings


def _is_stale(block: dict, today: _dt.date) -> bool:
    lvd = block.get("last_verified_date")
    if lvd is None:
        return False
    if block["status"] in {"resolved", "superseded"}:
        return False  # don't flag closed loops as stale
    return (today - lvd).days > STALE_DAYS


def _sort_key(block: dict) -> tuple:
    sev = SEVERITY_ORDER.get(block["severity"], 99)
    fsd = block.get("first_seen_date") or _dt.date.max
    return (sev, fsd, block["id"])


def _render_entry(block: dict, root: Path, *, stale: bool) -> str:
    rel_path = block["source_path"].relative_to(root).as_posix()
    src = f"{rel_path}:{block['source_line']}"
    stale_marker = " ⚠ STALE" if stale else ""
    return (
        f"### `{block['id']}`{stale_marker}\n"
        f"- **status**: {block['status']}\n"
        f"- **severity**: {block['severity']}\n"
        f"- **first_seen**: {block['first_seen']}\n"
        f"- **last_verified**: {block['last_verified']}\n"
        f"- **close_criterion**: {block['close_criterion']}\n"
        f"- **source**: `{src}`\n"
    )


def _render_output(blocks: list[dict], root: Path, *, today: _dt.date) -> str:
    timestamp = today.isoformat()
    header = (
        "# OPEN_LOOPS — Mechanically Generated Issue Inventory\n\n"
        f"> ⚙ **GENERATED on {timestamp}** by `tools/regenerate_open_loops.py`. "
        "Do not hand-edit. To change an entry, edit the corresponding `### Open loop:` "
        "block in the **owning thread file** under `threads/` and re-run the script.\n\n"
    )

    if not blocks:
        return header + (
            "## Entries\n\n"
            "*(No structured open-loop blocks found in `threads/*.md`. Either the wiki "
            "build is incomplete or every loop has been closed and removed.)*\n"
        )

    # Group by status using GROUP_ORDER. Entries with unknown status sink to the bottom.
    groups: dict[str, list[dict]] = {key: [] for key, _ in GROUP_ORDER}
    unknown: list[dict] = []
    for block in blocks:
        if block["status"] in groups:
            groups[block["status"]].append(block)
        else:
            unknown.append(block)

    stale_count = 0
    for group_blocks in groups.values():
        for b in group_blocks:
            if _is_stale(b, today):
                stale_count += 1

    summary_lines = []
    for key, label in GROUP_ORDER:
        n = len(groups[key])
        if n:
            summary_lines.append(f"- **{label}**: {n}")
    if unknown:
        summary_lines.append(f"- **Unknown status**: {len(unknown)}")
    if stale_count:
        summary_lines.append(f"- ⚠ **Stale (last_verified > {STALE_DAYS} days)**: {stale_count}")

    body_parts = [header]
    if summary_lines:
        body_parts.append("## Summary\n\n" + "\n".join(summary_lines) + "\n\n")

    body_parts.append("## Entries\n\n")

    for key, label in GROUP_ORDER:
        group_blocks = sorted(groups[key], key=_sort_key)
        if not group_blocks:
            continue
        body_parts.append(f"### {label} ({len(group_blocks)})\n\n")
        for block in group_blocks:
            stale = _is_stale(block, today)
            body_parts.append(_render_entry(block, root, stale=stale))
            body_parts.append("\n")

    if unknown:
        body_parts.append(f"### Unknown status ({len(unknown)})\n\n")
        for block in sorted(unknown, key=_sort_key):
            stale = _is_stale(block, today)
            body_parts.append(_render_entry(block, root, stale=stale))
            body_parts.append("\n")

    return "".join(body_parts)


def main(argv: list[str] | None = None) -> int:
    root = _project_root()
    threads_dir = root / "threads"
    output_path = root / "OPEN_LOOPS.md"

    if not threads_dir.exists() or not threads_dir.is_dir():
        sys.stderr.write(
            f"regenerate_open_loops.py: threads dir not found at {threads_dir}; "
            "writing empty inventory\n"
        )
        all_blocks: list[dict] = []
    else:
        all_blocks = []
        all_warnings: list[str] = []
        # Sorted for deterministic output across filesystems.
        thread_files = sorted(threads_dir.glob("*.md"))
        for thread_file in thread_files:
            blocks, warnings = _parse_thread_file(thread_file)
            all_blocks.extend(blocks)
            all_warnings.extend(warnings)

        for warning in all_warnings:
            sys.stderr.write(f"warn: {warning}\n")

    today = _today()
    output_text = _render_output(all_blocks, root, today=today)

    output_path.write_text(output_text, encoding="utf-8")

    open_count = sum(1 for b in all_blocks if b["status"] == "open")
    inprog_count = sum(1 for b in all_blocks if b["status"] == "in-progress")
    resolved_count = sum(1 for b in all_blocks if b["status"] == "resolved")
    superseded_count = sum(1 for b in all_blocks if b["status"] == "superseded")
    stale_count = sum(1 for b in all_blocks if _is_stale(b, today))

    sys.stdout.write(
        f"regenerate_open_loops.py: wrote {output_path.relative_to(root)} "
        f"(open={open_count}, in-progress={inprog_count}, resolved={resolved_count}, "
        f"superseded={superseded_count}, stale={stale_count})\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
