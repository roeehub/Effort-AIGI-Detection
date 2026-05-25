"""Tests for the offline sessionization helper (analysis/observability_sessions).

Groups per-request observability index lines into sessions by client IP +
inactivity gap. This is the read-side counterpart to the GCS layout written by
observability.ObservabilityUploader.
"""
from __future__ import annotations

import pytest

from analysis.observability_sessions.sessionize import (
    _daterange,
    sessionize,
    summarize_session,
)

MIN = 60 * 1000  # one minute in ms


def _line(ip, ts_min, rid, n_frames=1, pred="REAL"):
    return {
        "request_id": rid,
        "client_ip": ip,
        "ts_epoch_ms": ts_min * MIN,
        "utc": f"t+{ts_min}m",
        "endpoint": "/check_frame_batch",
        "pred_label": pred,
        "n_frames": n_frames,
        "n_scored": n_frames,
        "n_gated": 0,
        "status": "ok",
    }


def test_empty_input_yields_no_sessions():
    assert sessionize([]) == []


def test_requests_close_in_time_same_ip_are_one_session():
    rows = [_line("1.1.1.1", 0, "a"), _line("1.1.1.1", 5, "b"), _line("1.1.1.1", 9, "c")]
    sessions = sessionize(rows, gap_minutes=30)
    assert len(sessions) == 1
    assert [r["request_id"] for r in sessions[0]] == ["a", "b", "c"]


def test_gap_larger_than_threshold_splits_sessions():
    rows = [_line("1.1.1.1", 0, "a"), _line("1.1.1.1", 45, "b")]  # 45 min > 30
    sessions = sessionize(rows, gap_minutes=30)
    assert len(sessions) == 2
    assert [r["request_id"] for r in sessions[0]] == ["a"]
    assert [r["request_id"] for r in sessions[1]] == ["b"]


def test_different_ips_are_separate_sessions():
    rows = [_line("1.1.1.1", 0, "a"), _line("2.2.2.2", 1, "b"), _line("1.1.1.1", 2, "c")]
    sessions = sessionize(rows, gap_minutes=30)
    # one session per IP (each IP's requests are within the gap)
    ips = sorted({s[0]["client_ip"] for s in sessions})
    assert ips == ["1.1.1.1", "2.2.2.2"]
    by_ip = {s[0]["client_ip"]: [r["request_id"] for r in s] for s in sessions}
    assert by_ip["1.1.1.1"] == ["a", "c"]
    assert by_ip["2.2.2.2"] == ["b"]


def test_duplicate_request_ids_are_deduped():
    rows = [_line("1.1.1.1", 0, "a"), _line("1.1.1.1", 0, "a"), _line("1.1.1.1", 1, "b")]
    sessions = sessionize(rows, gap_minutes=30)
    assert sum(len(s) for s in sessions) == 2  # "a" counted once


def test_boundary_exactly_at_gap_stays_in_same_session():
    # gap == threshold is inclusive (still same session); strictly greater splits.
    rows = [_line("1.1.1.1", 0, "a"), _line("1.1.1.1", 30, "b")]
    assert len(sessionize(rows, gap_minutes=30)) == 1


def test_sessionize_skips_lines_without_request_id():
    # #11: a line lacking request_id must NOT collapse other distinct requests
    # into a single None-keyed entry; such malformed lines are skipped.
    rows = [_line("1.1.1.1", 0, "a"),
            {"client_ip": "1.1.1.1", "ts_epoch_ms": 60000},  # no request_id
            _line("1.1.1.1", 2, "b")]
    sessions = sessionize(rows, gap_minutes=30)
    ids = [r["request_id"] for s in sessions for r in s]
    assert "a" in ids and "b" in ids
    assert sum(len(s) for s in sessions) == 2  # the id-less line dropped, a/b intact


def test_summarize_session_tolerates_null_numeric_fields():
    # #10: a present-but-null count/timestamp must not crash the whole report.
    session = [
        {"request_id": "a", "client_ip": "9.9.9.9", "ts_epoch_ms": 0, "n_frames": None, "pred_label": "REAL"},
        {"request_id": "b", "client_ip": "9.9.9.9", "ts_epoch_ms": 1000, "n_frames": 2, "pred_label": "FAKE"},
    ]
    s = summarize_session(session)  # must not raise
    assert s["n_frames"] == 2      # None coerced to 0
    assert s["duration_ms"] == 1000


def test_daterange_rejects_start_after_end():
    # #14: a transposed range must error loudly, not silently return no days.
    with pytest.raises(ValueError):
        _daterange("2026-05-25", "2026-05-20")


def test_summarize_session_reports_span_and_totals():
    session = [_line("9.9.9.9", 0, "a", n_frames=3, pred="FAKE"),
               _line("9.9.9.9", 10, "b", n_frames=2, pred="REAL")]
    s = summarize_session(session)
    assert s["client_ip"] == "9.9.9.9"
    assert s["n_requests"] == 2
    assert s["n_frames"] == 5
    assert s["duration_ms"] == 10 * MIN
    assert s["start_ts_epoch_ms"] == 0
    assert s["end_ts_epoch_ms"] == 10 * MIN
    assert s["fake_requests"] == 1
