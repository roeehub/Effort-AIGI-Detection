"""One-off: remove `quality_enhancement` from R13 yaml `enhanced_strategy_names`
ONLY (NOT from `include_strategies`).

Companion to the 2026-05-05 fix landing the misrouting bug. Removes the line
`      - "quality_enhancement"` (6-space indent) ONLY when it appears inside
an `enhanced_strategy_names:` block. The same line in an `include_strategies:`
block is preserved — we still want to train on quality_enhancement frames,
just routed to the correct family.

Approach: walk each yaml line by line, track whether we're inside an
`enhanced_strategy_names:` list (entered at that key, exited when indent drops
below the list-item indent or a non-list-item key appears). Drop only the
quality_enhancement line within that scope.

Sanity checks:
- For each affected file: report exactly 1 line removed (expected).
- Files with 0 or >1 removals are flagged and left untouched.

See: docs/packet_retrospectives/threads/quality_enhancement_strategy_misrouting.md
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path('experiments/phase2_round13')

ENHANCED_KEY_RE = re.compile(r'^(\s*)enhanced_strategy_names:\s*$')
QE_LINE_RE = re.compile(r'^      - "quality_enhancement"\s*$')


def process_lines(lines: list[str]) -> tuple[list[str], int]:
    """Return (new_lines, n_removed)."""
    out: list[str] = []
    in_enhanced_block = False
    enhanced_key_indent = -1
    n_removed = 0

    for line in lines:
        # Detect entering enhanced_strategy_names block
        m = ENHANCED_KEY_RE.match(line)
        if m:
            in_enhanced_block = True
            enhanced_key_indent = len(m.group(1))
            out.append(line)
            continue

        # Inside the block, list items are at indent = key_indent + 2 (or 4 for "  - ").
        # Exit when we hit a line whose indent is <= key_indent and it's not a list item.
        if in_enhanced_block:
            stripped = line.lstrip()
            line_indent = len(line) - len(stripped)
            is_list_item = stripped.startswith('- ')
            if line_indent <= enhanced_key_indent and stripped and not is_list_item:
                # Exited the block via a sibling/parent key
                in_enhanced_block = False
            elif not stripped:
                # Empty line — yaml may exit block; safer: stay in block
                # (subsequent list items still part of it). But if multiple
                # blank lines, also exit. Conservative: stay in.
                pass
            elif is_list_item and line_indent <= enhanced_key_indent:
                # List item at same indent as key — malformed; assume exit
                in_enhanced_block = False

            if in_enhanced_block and QE_LINE_RE.match(line):
                n_removed += 1
                continue  # drop this line

        out.append(line)

    return out, n_removed


def main() -> int:
    if not ROOT.exists():
        print(f"ERROR: {ROOT} not found; run from training/ root", file=sys.stderr)
        return 1

    affected = 0
    skipped_no_match = 0
    anomalies = []

    for yaml_path in sorted(ROOT.glob('*.yaml')):
        with open(yaml_path, 'r') as f:
            lines = f.readlines()

        new_lines, n_removed = process_lines(lines)

        if n_removed == 0:
            skipped_no_match += 1
            continue
        if n_removed != 1:
            anomalies.append((yaml_path, n_removed))
            continue

        with open(yaml_path, 'w') as f:
            f.writelines(new_lines)
        affected += 1

    print(f"Affected files: {affected}")
    print(f"Files without target line in enhanced_strategy_names block: {skipped_no_match}")
    print(f"Anomalies (removed != 1): {len(anomalies)}")
    for p, d in anomalies:
        print(f"  WARN {p}: removed {d} lines (expected 1) — left untouched")
    return 0 if not anomalies else 2


if __name__ == '__main__':
    sys.exit(main())
