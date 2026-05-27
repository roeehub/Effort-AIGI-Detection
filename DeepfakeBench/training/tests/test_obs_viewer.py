"""Tests for the local observability viewer (viewer/obs_server.py).

Covers the pure transforms (date parsing, session payload shaping, IP directory
aggregation, frame-URL attachment, config normalize/label) and the bucket
config read-modify-write (generation-match + conflict retry) against an
in-memory fake, plus a few Flask route smokes with data-access monkeypatched.

No network: GCS is faked / monkeypatched throughout.
"""
from __future__ import annotations

import json

import pytest
from google.api_core.exceptions import NotFound, PreconditionFailed

import viewer.obs_server as obs


# ──────────────────────────────────────────
# helpers
# ──────────────────────────────────────────
def row(rid, ip, ts, *, label="REAL", status="ok", n_frames=2, n_gated=0, prefix=None):
    ip_dash = ip.replace(".", "-")
    return {
        "schema_version": "obs-1", "request_id": rid, "ts_epoch_ms": ts,
        "utc": f"utc-{ts}", "client_ip": ip, "endpoint": "/check_frame_batch",
        "model_type": "custom", "pred_label": label, "confidence": 0.5, "status": status,
        "n_frames": n_frames, "n_scored": n_frames - n_gated, "n_gated": n_gated, "n_failed": 0,
        "gcs_prefix": prefix or f"v1/date=2026-05-25/ip={ip_dash}/req={ts}_{rid}",
    }


# ──────────────────────────────────────────
# parse_dates
# ──────────────────────────────────────────
def test_parse_dates_extracts_dedups_and_sorts_desc():
    prefixes = ["v1/date=2026-05-24/", "v1/date=2026-05-25/", "v1/date=2026-05-25/", "v1/_viewer/"]
    assert obs.parse_dates(prefixes) == ["2026-05-25", "2026-05-24"]


def test_parse_dates_ignores_non_date_prefixes():
    assert obs.parse_dates(["v1/_index/", "garbage", ""]) == []


# ──────────────────────────────────────────
# sessions_payload
# ──────────────────────────────────────────
def test_sessions_payload_newest_first_with_requests_attached():
    rows = [
        row("a1", "1.1.1.1", 1000),
        row("a2", "1.1.1.1", 2000, label="FAKE"),
        row("b1", "2.2.2.2", 5000),
    ]
    out = obs.sessions_payload(rows, gap_minutes=30)
    assert out["n_requests"] == 3
    assert out["n_sessions"] == 2
    # sessions newest-first by start ts: B (start 5000) before A (start 1000)
    assert out["sessions"][0]["client_ip"] == "2.2.2.2"
    assert out["sessions"][1]["client_ip"] == "1.1.1.1"
    a = out["sessions"][1]
    assert a["n_requests"] == 2
    assert a["fake_requests"] == 1
    # requests within a session are newest-first
    assert [r["request_id"] for r in a["requests"]] == ["a2", "a1"]


def test_sessions_payload_gap_splits_same_ip():
    big_gap_ms = 40 * 60 * 1000
    rows = [row("a1", "1.1.1.1", 0), row("a2", "1.1.1.1", big_gap_ms)]
    out = obs.sessions_payload(rows, gap_minutes=30)
    assert out["n_sessions"] == 2


# ──────────────────────────────────────────
# ip_directory
# ──────────────────────────────────────────
def test_ip_directory_aggregates_and_applies_names():
    rows = [
        row("a1", "1.1.1.1", 1000, label="FAKE"),
        row("a2", "1.1.1.1", 9000),
        row("b1", "2.2.2.2", 2000),
    ]
    labels = {"1.1.1.1": {"name": "Alice"}}
    out = obs.ip_directory(rows, labels, gap_minutes=30)
    by_ip = {e["ip"]: e for e in out}
    assert by_ip["1.1.1.1"]["name"] == "Alice"
    assert by_ip["1.1.1.1"]["n_requests"] == 2
    assert by_ip["1.1.1.1"]["fake_requests"] == 1
    assert by_ip["2.2.2.2"]["name"] is None
    # sorted by last activity desc → 1.1.1.1 (ts 9000) first
    assert out[0]["ip"] == "1.1.1.1"


def test_ip_directory_counts_sessions_per_ip():
    big_gap_ms = 40 * 60 * 1000
    rows = [row("a1", "1.1.1.1", 0), row("a2", "1.1.1.1", big_gap_ms)]
    out = obs.ip_directory(rows, {}, gap_minutes=30)
    assert out[0]["n_sessions"] == 2


# ──────────────────────────────────────────
# attach_frame_urls
# ──────────────────────────────────────────
def test_attach_frame_urls_only_for_stored_frames():
    meta = {
        "frames": [
            {"seq": 0, "bytes_size": 1024, "gcs_object_path": "v1/d/ip/req/frame_000.png", "capture_skipped": False},
            {"seq": 1, "bytes_size": 2048, "gcs_object_path": None, "capture_skipped": False},
            {"seq": 2, "bytes_size": 4096, "gcs_object_path": "v1/d/ip/req/frame_002.png", "capture_skipped": True},
        ]
    }
    out = obs.attach_frame_urls(meta, "remote-live-data")
    f = out["frames"]
    assert f[0]["frame_url"] == "/api/frame/remote-live-data/v1/d/ip/req/frame_000.png"
    assert f[1]["frame_url"] is None          # gcs_object_path None
    assert f[2]["frame_url"] is None          # capture_skipped
    assert f[0]["bytes_human"] == "1.0 KB"


# ──────────────────────────────────────────
# config normalize / apply_label
# ──────────────────────────────────────────
def test_normalize_config_fills_shape():
    cfg = obs.normalize_config(None)
    assert cfg["version"] == 1
    assert cfg["ip_labels"] == {}
    assert "gap_minutes" in cfg["settings"]


def test_apply_label_sets_entry():
    cfg = obs.apply_label(obs.normalize_config(None), "1.2.3.4", "Bob", notes="vip")
    assert cfg["ip_labels"]["1.2.3.4"] == {"name": "Bob", "notes": "vip"}


# ──────────────────────────────────────────
# config read-modify-write against a fake bucket
# ──────────────────────────────────────────
class _Store:
    def __init__(self, text=None, gen=0, fail_next_upload=False):
        self.text = text
        self.gen = gen
        self.uploads = []
        self.fail_next_upload = fail_next_upload


class _FakeBlob:
    def __init__(self, store, path):
        self.store = store
        self.path = path
        self.generation = store.gen if store.text is not None else None

    def download_as_text(self):
        if self.store.text is None:
            raise NotFound(self.path)
        return self.store.text

    def upload_from_string(self, data, content_type=None, if_generation_match=None):
        s = self.store
        s.uploads.append((data, if_generation_match))
        if s.fail_next_upload:
            s.fail_next_upload = False
            if s.text is not None:      # simulate a concurrent writer bumping the generation
                s.gen += 1
            raise PreconditionFailed("simulated concurrent write")
        cur = s.gen if s.text is not None else 0
        if (if_generation_match or 0) != cur:
            raise PreconditionFailed("generation mismatch")
        s.text = data
        s.gen = cur + 1


class _FakeBucket:
    def __init__(self, store):
        self.store = store

    def get_blob(self, path):
        if self.store.text is None:
            return None
        return _FakeBlob(self.store, path)

    def blob(self, path):
        return _FakeBlob(self.store, path)


class _FakeClient:
    def __init__(self, store):
        self.store = store

    def bucket(self, name):
        return _FakeBucket(self.store)


def test_read_config_absent_returns_empty_gen0():
    client = _FakeClient(_Store(text=None))
    cfg, gen = obs.read_config(client)
    assert gen == 0 and cfg["ip_labels"] == {}


def test_set_label_creates_config_when_absent():
    store = _Store(text=None)
    cfg = obs.set_label(_FakeClient(store), "9.9.9.9", "Zed")
    assert cfg["ip_labels"]["9.9.9.9"]["name"] == "Zed"
    assert store.text is not None and store.gen == 1
    assert json.loads(store.text)["ip_labels"]["9.9.9.9"]["name"] == "Zed"


def test_set_label_merges_into_existing():
    existing = json.dumps({"version": 1, "ip_labels": {"1.1.1.1": {"name": "A"}}, "settings": {}})
    store = _Store(text=existing, gen=5)
    cfg = obs.set_label(_FakeClient(store), "2.2.2.2", "B")
    assert cfg["ip_labels"]["1.1.1.1"]["name"] == "A"   # preserved
    assert cfg["ip_labels"]["2.2.2.2"]["name"] == "B"   # added
    assert store.gen == 6


def test_set_label_retries_on_generation_conflict():
    existing = json.dumps({"version": 1, "ip_labels": {}, "settings": {}})
    store = _Store(text=existing, gen=5, fail_next_upload=True)
    cfg = obs.set_label(_FakeClient(store), "3.3.3.3", "C")
    assert cfg["ip_labels"]["3.3.3.3"]["name"] == "C"   # succeeded after retry
    assert len(store.uploads) == 2                       # one failed, one succeeded


# ──────────────────────────────────────────
# Flask route smokes (data access monkeypatched)
# ──────────────────────────────────────────
@pytest.fixture
def client():
    obs.app.config["TESTING"] = True
    return obs.app.test_client()


def test_route_dates(client, monkeypatch):
    monkeypatch.setattr(obs, "list_dates", lambda: ["2026-05-25", "2026-05-24"])
    r = client.get("/api/dates")
    assert r.status_code == 200
    assert r.get_json()["dates"] == ["2026-05-25", "2026-05-24"]


def test_route_sessions(client, monkeypatch):
    rows = [row("a1", "1.1.1.1", 1000), row("a2", "1.1.1.1", 2000)]
    monkeypatch.setattr(obs, "read_index_rows", lambda date: rows)
    r = client.get("/api/sessions?date=2026-05-25&gap_minutes=30")
    assert r.status_code == 200
    body = r.get_json()
    assert body["n_sessions"] == 1 and body["sessions"][0]["n_requests"] == 2


def test_route_request_attaches_frame_url(client, monkeypatch):
    meta = {"request_id": "x", "frames": [
        {"seq": 0, "bytes_size": 10, "gcs_object_path": "v1/d/ip/req/frame_000.jpg", "capture_skipped": False}]}
    monkeypatch.setattr(obs, "download_meta", lambda prefix: dict(meta))
    r = client.get("/api/request?gcs_prefix=v1/d/ip/req")
    assert r.status_code == 200
    assert r.get_json()["frames"][0]["frame_url"].endswith("/frame_000.jpg")


def test_route_request_requires_prefix(client):
    assert client.get("/api/request").status_code == 400


def test_route_label_post(client, monkeypatch):
    captured = {}

    def fake_set_label(c, ip, name, notes=None):
        captured.update(ip=ip, name=name, notes=notes)
        return {"version": 1, "ip_labels": {ip: {"name": name}}, "settings": {}}

    monkeypatch.setattr(obs, "set_label", fake_set_label)
    monkeypatch.setattr(obs, "_get_gcs_client", lambda: object())
    r = client.post("/api/label", json={"ip": "5.5.5.5", "name": "Eve"})
    assert r.status_code == 200
    assert captured["ip"] == "5.5.5.5" and captured["name"] == "Eve"
    assert r.get_json()["ip_labels"]["5.5.5.5"]["name"] == "Eve"


def test_route_label_requires_ip_and_name(client):
    assert client.post("/api/label", json={"ip": "5.5.5.5"}).status_code == 400
    assert client.post("/api/label", json={"name": "Eve"}).status_code == 400
