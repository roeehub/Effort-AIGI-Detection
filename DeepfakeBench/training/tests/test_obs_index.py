"""Tests for the viewer's parallel index part-file reader.

A busy day's `_index/` holds hundreds of small `part-*.jsonl` files (one per
flush). Reading them one-by-one stalls on a slow/high-latency link, so
`_download_index_rows` fans the downloads out across threads. It must still parse
correctly, skip non-jsonl objects + malformed lines, and tolerate a failed
download without dropping the whole day.
"""
from __future__ import annotations

import json

import viewer.obs_server as obs


class _FakeBlob:
    def __init__(self, name, text, boom=False):
        self.name = name
        self._text = text
        self._boom = boom

    def download_as_text(self):
        if self._boom:
            raise RuntimeError("simulated network error")
        return self._text


class _FakeClient:
    def __init__(self, blobs):
        self._blobs = blobs

    def bucket(self, name):
        return ("bucket", name)

    def list_blobs(self, bucket, prefix=None):
        return list(self._blobs)


def _line(rid, ip="1.1.1.1"):
    return json.dumps({"request_id": rid, "client_ip": ip, "ts_epoch_ms": 1, "n_frames": 1})


def test_parses_all_jsonl_parts(monkeypatch):
    blobs = [
        _FakeBlob("v1/date=D/_index/part-1.jsonl", _line("a") + "\n" + _line("b")),
        _FakeBlob("v1/date=D/_index/part-2.jsonl", _line("c")),
    ]
    monkeypatch.setattr(obs, "_get_gcs_client", lambda: _FakeClient(blobs))
    rows = obs._download_index_rows("D")
    assert sorted(r["request_id"] for r in rows) == ["a", "b", "c"]


def test_ignores_non_jsonl_and_malformed_lines(monkeypatch):
    blobs = [
        _FakeBlob("v1/date=D/_index/part-1.jsonl", _line("a") + "\nNOT JSON\n"),
        _FakeBlob("v1/date=D/_index/manifest.txt", _line("ignored")),   # not .jsonl
    ]
    monkeypatch.setattr(obs, "_get_gcs_client", lambda: _FakeClient(blobs))
    rows = obs._download_index_rows("D")
    assert [r["request_id"] for r in rows] == ["a"]


def test_tolerates_a_failed_download(monkeypatch):
    blobs = [
        _FakeBlob("v1/date=D/_index/ok.jsonl", _line("a")),
        _FakeBlob("v1/date=D/_index/boom.jsonl", "", boom=True),        # one download fails
    ]
    monkeypatch.setattr(obs, "_get_gcs_client", lambda: _FakeClient(blobs))
    rows = obs._download_index_rows("D")
    assert [r["request_id"] for r in rows] == ["a"]                     # skipped, not fatal


def test_empty_index_returns_empty(monkeypatch):
    monkeypatch.setattr(obs, "_get_gcs_client", lambda: _FakeClient([]))
    assert obs._download_index_rows("D") == []
