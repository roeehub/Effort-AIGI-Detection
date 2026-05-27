"""Offline sessionization for per-request observability logs.

The live server (app3.py + observability.py) writes, for every inference
request, a compact JSON line to:

    gs://<bucket>/<prefix>/date=YYYY-MM-DD/_index/part-<pid>-<uuid>.jsonl

plus the original frames + a meta.json under
`date=.../ip=.../req=<utc>_<reqid>/`. This script reads the `_index` lines for
one or more days, groups them into **sessions** (consecutive requests from the
same client IP, split on an inactivity gap), and prints a summary per session.

It is read-only and runs locally/offline — it is NOT imported by the server.

Examples
--------
    # Sessions for one day from the live bucket (uses ADC):
    python -m analysis.observability_sessions.sessionize --date 2026-05-25

    # A date range, custom gap, JSON output:
    python -m analysis.observability_sessions.sessionize \
        --bucket remote-live-data --start 2026-05-20 --end 2026-05-25 \
        --gap-minutes 20 --json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from typing import Any, Dict, Iterable, List


# ──────────────────────────────────────────
# Pure logic (unit-tested)
# ──────────────────────────────────────────
def _as_int(value: Any, default: int = 0) -> int:
    """Coerce a possibly-missing/None/non-numeric field to int (fail-soft)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def sessionize(records: Iterable[Dict[str, Any]], gap_minutes: int = 30) -> List[List[Dict[str, Any]]]:
    """Group index lines into sessions by (client_ip, inactivity gap).

    * De-duplicates on `request_id` (uploads are at-least-once; an index part
      file may be re-written on retry).
    * Sorts by (client_ip, ts_epoch_ms) so each IP's requests are contiguous and
      chronological.
    * Starts a new session when the same IP is idle for STRICTLY MORE than
      `gap_minutes` (a gap exactly equal to the threshold stays in-session), or
      when the IP changes.
    """
    seen: Dict[Any, Dict[str, Any]] = {}
    for r in records:
        rid = r.get("request_id")
        if rid is None:
            continue  # malformed/foreign line: can't identify or dedup it — skip
        if rid not in seen:
            seen[rid] = r
    uniq = sorted(seen.values(), key=lambda r: (r.get("client_ip") or "", _as_int(r.get("ts_epoch_ms"))))

    gap_ms = gap_minutes * 60 * 1000
    sessions: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for r in uniq:
        if not current:
            current = [r]
            continue
        prev = current[-1]
        same_ip = r.get("client_ip") == prev.get("client_ip")
        within_gap = (_as_int(r.get("ts_epoch_ms")) - _as_int(prev.get("ts_epoch_ms"))) <= gap_ms
        if same_ip and within_gap:
            current.append(r)
        else:
            sessions.append(current)
            current = [r]
    if current:
        sessions.append(current)
    return sessions


def summarize_session(session: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One-line summary of a session (sorted ascending by ts)."""
    start = session[0]
    end = session[-1]
    return {
        "client_ip": start.get("client_ip"),
        "n_requests": len(session),
        "n_frames": sum(_as_int(r.get("n_frames")) for r in session),
        "n_scored": sum(_as_int(r.get("n_scored")) for r in session),
        "n_gated": sum(_as_int(r.get("n_gated")) for r in session),
        "fake_requests": sum(1 for r in session if r.get("pred_label") == "FAKE"),
        "start_ts_epoch_ms": _as_int(start.get("ts_epoch_ms")),
        "end_ts_epoch_ms": _as_int(end.get("ts_epoch_ms")),
        "start_utc": start.get("utc"),
        "end_utc": end.get("utc"),
        "duration_ms": _as_int(end.get("ts_epoch_ms")) - _as_int(start.get("ts_epoch_ms")),
        "first_request_id": start.get("request_id"),
        "gcs_prefixes": [r.get("gcs_prefix") for r in session],
    }


# ──────────────────────────────────────────
# GCS read side (CLI; not unit-tested — needs a real/emulated bucket)
# ──────────────────────────────────────────
def _daterange(start: str, end: str) -> List[str]:
    d0 = _dt.date.fromisoformat(start)
    d1 = _dt.date.fromisoformat(end)
    if d0 > d1:
        raise ValueError(f"--start ({start}) is after --end ({end})")
    out, d = [], d0
    while d <= d1:
        out.append(d.isoformat())
        d += _dt.timedelta(days=1)
    return out


def read_index_lines(bucket_name: str, prefix: str, dates: List[str]) -> List[Dict[str, Any]]:
    """Download and parse every _index/*.jsonl line for the given dates."""
    from google.cloud import storage  # lazy import; only needed for the CLI
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    rows: List[Dict[str, Any]] = []
    for date in dates:
        idx_prefix = f"{prefix}/date={date}/_index/"
        for blob in client.list_blobs(bucket, prefix=idx_prefix):
            if not blob.name.endswith(".jsonl"):
                continue
            for line in blob.download_as_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return rows


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Group observability index lines into per-IP sessions.")
    p.add_argument("--bucket", default="remote-live-data")
    p.add_argument("--prefix", default="v1")
    p.add_argument("--date", help="single day YYYY-MM-DD")
    p.add_argument("--start", help="range start YYYY-MM-DD")
    p.add_argument("--end", help="range end YYYY-MM-DD (inclusive)")
    p.add_argument("--gap-minutes", type=int, default=30)
    p.add_argument("--json", action="store_true", help="emit session summaries as JSON")
    args = p.parse_args(argv)

    if args.date:
        dates = [args.date]
    elif args.start and args.end:
        try:
            dates = _daterange(args.start, args.end)
        except ValueError as e:
            p.error(str(e))
    else:
        p.error("provide --date OR (--start and --end)")

    rows = read_index_lines(args.bucket, args.prefix, dates)
    sessions = sessionize(rows, gap_minutes=args.gap_minutes)
    summaries = [summarize_session(s) for s in sessions]

    if args.json:
        print(json.dumps(summaries, indent=2))
    else:
        print(f"{len(rows)} requests across {len(dates)} day(s) → {len(sessions)} sessions "
              f"(gap={args.gap_minutes}m)\n")
        for s in summaries:
            dur_min = s["duration_ms"] / 60000.0
            print(f"  ip={s['client_ip']:<20} reqs={s['n_requests']:<4} frames={s['n_frames']:<5} "
                  f"fake_reqs={s['fake_requests']:<3} dur={dur_min:6.1f}m  {s['start_utc']}→{s['end_utc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
