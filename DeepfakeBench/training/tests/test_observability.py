"""Tests for the per-request observability capture/upload pipeline (observability.py).

The feature captures, for every /check_frame and /check_frame_batch request, the
client IP + original image bytes + params + per-frame results, and uploads them
asynchronously to GCS via background worker threads — without ever blocking,
slowing, or crashing the inference request path.

These tests use an injected in-memory fake GCS bucket (no network, no Docker) so
the upload logic, byte-budgeting, retry, index flushing, graceful drain and
fail-open behaviour are all exercised deterministically.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading

import numpy as np
import pytest

try:
    import cv2  # noqa: F401
    _HAVE_CV2 = True
except Exception:  # pragma: no cover
    _HAVE_CV2 = False

from starlette.requests import Request

import observability as obs


# ──────────────────────────────────────────
# In-memory fake GCS bucket (no network, no Docker)
# ──────────────────────────────────────────
class FakeBlob:
    def __init__(self, bucket, name):
        self._bucket = bucket
        self.name = name

    def upload_from_string(self, data, content_type=None):
        self._bucket._record_upload(self.name, data, content_type)


class FakeBucket:
    def __init__(self):
        self.uploads = []                # ordered [(name, data, content_type)]
        self.fail_counts = {}            # name -> remaining transient failures
        self.fail_substr = None          # if set, any blob name containing it always fails
        self._lock = threading.Lock()

    def blob(self, name):
        return FakeBlob(self, name)

    def _record_upload(self, name, data, content_type):
        with self._lock:
            if self.fail_substr is not None and self.fail_substr in name:
                raise RuntimeError("simulated persistent GCS error")
            if self.fail_counts.get(name, 0) > 0:
                self.fail_counts[name] -= 1
                raise RuntimeError("simulated transient GCS error")
            self.uploads.append((name, data, content_type))

    # convenience views
    def names(self):
        with self._lock:
            return [n for (n, _d, _c) in self.uploads]

    def get(self, name):
        with self._lock:
            for (n, d, c) in self.uploads:
                if n == name:
                    return d, c
        return None, None


# ──────────────────────────────────────────
# Test helpers
# ──────────────────────────────────────────
def make_request(client=("203.0.113.7", 5555), headers=None):
    """Build a real Starlette Request with the given peer + headers."""
    raw_headers = [
        (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
    ]
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/check_frame",
        "headers": raw_headers,
        "client": client,  # (host, port) or None
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)


# ──────────────────────────────────────────
# get_client_ip
# ──────────────────────────────────────────
def test_get_client_ip_uses_peer_when_no_forwarded_header():
    ip, remote_addr, chain = obs.get_client_ip(make_request(client=("203.0.113.7", 1)))
    assert ip == "203.0.113.7"
    assert remote_addr == "203.0.113.7"
    assert chain == []


def test_get_client_ip_prefers_first_forwarded_hop():
    req = make_request(
        client=("10.0.0.1", 1),
        headers={"X-Forwarded-For": "198.51.100.23, 10.0.0.1"},
    )
    ip, remote_addr, chain = obs.get_client_ip(req)
    assert ip == "198.51.100.23"           # real client = first hop
    assert remote_addr == "10.0.0.1"        # direct peer kept separately
    assert chain == ["198.51.100.23", "10.0.0.1"]


def test_get_client_ip_handles_missing_client_without_raising():
    ip, remote_addr, chain = obs.get_client_ip(make_request(client=None))
    assert ip == "unknown"
    assert remote_addr == "unknown"
    assert chain == []


def test_get_client_ip_trims_and_drops_empty_forwarded_entries():
    req = make_request(headers={"X-Forwarded-For": " 198.51.100.23 ,, "})
    ip, _remote, chain = obs.get_client_ip(req)
    assert ip == "198.51.100.23"
    assert chain == ["198.51.100.23"]


# ──────────────────────────────────────────
# new_record (request envelope)
# ──────────────────────────────────────────
def test_new_record_builds_envelope_from_request():
    req = make_request(
        client=("10.0.0.1", 1),
        headers={"X-Forwarded-For": "198.51.100.23", "User-Agent": "UA/1", "X-Client-Id": "cid-9"},
    )
    rec = obs.new_record(req, "/check_frame", model_type="custom", threshold=0.5,
                         gate_profile="legacy", gate_spec={"min_dim": 80},
                         yolo_conf_threshold=0.2, recrop=False, debug=False)
    assert rec.endpoint == "/check_frame"
    assert rec.client_ip == "198.51.100.23"
    assert rec.remote_addr == "10.0.0.1"
    assert rec.user_agent == "UA/1"
    assert rec.x_client_id == "cid-9"
    assert rec.model_type == "custom" and rec.threshold == 0.5
    assert len(rec.request_id) >= 8
    assert rec.status == "handler_exited_without_status"
    # received_utc is a compact, lexicographically-sortable UTC stamp
    # (YYYYMMDDThhmmssfffZ) so listing req= prefixes is chronological.
    assert re.match(r"^\d{8}T\d{9}Z$", rec.received_utc)


# ──────────────────────────────────────────
# sanitize_ip
# ──────────────────────────────────────────
def test_sanitize_ipv4_replaces_dots():
    assert obs.sanitize_ip("198.51.100.23") == "198-51-100-23"


def test_sanitize_ipv6_replaces_colons_and_lowercases():
    assert obs.sanitize_ip("2001:DB8::1") == "2001-db8--1"


def test_sanitize_ip_empty_becomes_unknown():
    assert obs.sanitize_ip("") == "unknown"


def test_sanitize_ip_output_is_gcs_path_safe():
    # No characters outside [a-z0-9-] survive (slashes etc. would create
    # spurious "directories" in the object key).
    out = obs.sanitize_ip("1.2.3.4/../evil")
    assert all(c.isalnum() or c == "-" for c in out)
    assert "/" not in out


# ──────────────────────────────────────────
# sanitize_filename
# ──────────────────────────────────────────
def test_sanitize_filename_none_defaults_to_seq():
    assert obs.sanitize_filename(None, 3) == "frame_3"


def test_sanitize_filename_empty_defaults_to_seq():
    assert obs.sanitize_filename("   ", 7) == "frame_7"


def test_sanitize_filename_strips_path_separators():
    # Path-traversal-style names must not create nested keys.
    out = obs.sanitize_filename("../../etc/passwd", 0)
    assert "/" not in out
    assert ".." not in out
    assert out == "passwd"


def test_sanitize_filename_caps_length():
    out = obs.sanitize_filename("a" * 500 + ".jpg", 0)
    assert len(out) <= 128


# ──────────────────────────────────────────
# FrameCapture
# ──────────────────────────────────────────
def test_frame_capture_computes_size_and_sha256():
    raw = b"\xff\xd8\xff some jpeg bytes"
    fc = obs.FrameCapture(seq=0, raw_bytes=raw)
    assert fc.bytes_size == len(raw)
    assert fc.sha256 == hashlib.sha256(raw).hexdigest()


def test_frame_capture_to_meta_omits_raw_bytes():
    fc = obs.FrameCapture(seq=2, raw_bytes=b"abc", filename="f.jpg",
                          content_type="image/jpeg", dims_hw=(224, 256),
                          gate_pass=True, prob=0.81, scored=True, verdict="FAKE")
    meta = fc.to_meta()
    assert "raw_bytes" not in meta
    assert meta["seq"] == 2
    assert meta["filename"] == "f.jpg"
    assert meta["content_type"] == "image/jpeg"
    assert meta["dims"] == {"h": 224, "w": 256}
    assert meta["bytes_size"] == 3
    assert meta["sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert meta["gate_pass"] is True
    assert meta["prob"] == 0.81
    assert meta["scored"] is True
    assert meta["verdict"] == "FAKE"


def test_frame_capture_dims_none_when_decode_failed():
    fc = obs.FrameCapture(seq=0, raw_bytes=b"garbage", dims_hw=None)
    assert fc.to_meta()["dims"] is None


def test_frame_capture_face_found_is_tristate():
    # None = recrop not requested (N/A); False = recrop on but no face.
    assert obs.FrameCapture(seq=0, raw_bytes=b"x").to_meta()["face_found"] is None
    assert obs.FrameCapture(seq=0, raw_bytes=b"x", face_found=False).to_meta()["face_found"] is False


# ──────────────────────────────────────────
# CaptureRecord
# ──────────────────────────────────────────
def _record(frames, **over):
    base = dict(
        request_id="abcd1234", received_utc="20260525T141233456Z",
        received_epoch_ms=1779000000000, endpoint="/check_frame_batch",
        client_ip="198.51.100.23", remote_addr="10.0.0.1", xff_chain=["198.51.100.23", "10.0.0.1"],
        user_agent="pytest", x_client_id=None, model_type="custom",
        threshold=0.5, gate_profile="t5c", gate_spec={"min_dim": 120},
        yolo_conf_threshold=0.2, recrop=False, debug=False, frames=frames,
    )
    base.update(over)
    return obs.CaptureRecord(**base)


def test_record_total_bytes_sums_frames():
    rec = _record([obs.FrameCapture(0, b"aaaa"), obs.FrameCapture(1, b"bb")])
    assert rec.total_bytes == 6


def test_record_counts_scored_gated_failed():
    frames = [
        obs.FrameCapture(0, b"x", gate_pass=True, scored=True, prob=0.7),   # scored
        obs.FrameCapture(1, b"x", gate_pass=False, scored=False),            # gated (min_dim)
        obs.FrameCapture(2, b"x", face_found=False, scored=False),           # gated (no face)
        obs.FrameCapture(3, b"x", gate_pass=None, scored=False),             # decode/processing failed
    ]
    counts = _record(frames).counts()
    assert counts == {"total": 4, "scored": 1, "gated": 2, "failed": 1}


def test_record_to_meta_dict_structure_and_static_merge():
    rec = _record([obs.FrameCapture(0, b"x", scored=True, prob=0.9, verdict="FAKE")],
                  status="ok", pred_label="FAKE", confidence=0.9, latency_ms=12.3)
    static = {
        "checkpoint_paths": {"custom": "gs://b/ck.pth", "base": "weights/base.pth"},
        "use_arcface": False, "device": "cuda", "app_version": "0.7.0",
        "git_sha": "deadbeef", "hostname": "deepfakebench2", "pid": 4242,
    }
    rec.gcs_prefix = "v1/date=2026-05-25/ip=198-51-100-23/req=20260525T141233456Z_abcd1234"
    meta = rec.to_meta_dict(static)

    assert meta["schema_version"] == obs.SCHEMA_VERSION
    assert meta["request_id"] == "abcd1234"
    assert meta["status"] == "ok"
    assert meta["latency_ms"] == 12.3
    assert meta["gcs_prefix"].endswith("_abcd1234")
    # client block
    assert meta["client"]["client_ip"] == "198.51.100.23"
    assert meta["client"]["remote_addr"] == "10.0.0.1"
    assert meta["client"]["xff_chain"] == ["198.51.100.23", "10.0.0.1"]
    # model identity resolved from static via model_type
    assert meta["model"]["checkpoint_path"] == "gs://b/ck.pth"
    assert meta["model"]["device"] == "cuda"
    assert meta["model"]["hostname"] == "deepfakebench2"
    # params + aggregate + frames
    assert meta["params"]["threshold"] == 0.5
    assert meta["params"]["gate_profile"] == "t5c"
    assert meta["aggregate"]["pred_label"] == "FAKE"
    assert meta["counts"] == {"total": 1, "scored": 1, "gated": 0, "failed": 0}
    assert len(meta["frames"]) == 1 and meta["frames"][0]["prob"] == 0.9


def test_record_build_index_line_is_compact_and_self_sufficient():
    rec = _record([obs.FrameCapture(0, b"x", scored=True, prob=0.9)],
                  status="ok", pred_label="FAKE", confidence=0.9)
    rec.gcs_prefix = "v1/date=2026-05-25/ip=198-51-100-23/req=20260525T141233456Z_abcd1234"
    line = rec.build_index_line()
    assert line["request_id"] == "abcd1234"
    assert line["ts_epoch_ms"] == 1779000000000
    assert line["client_ip"] == "198.51.100.23"
    assert line["endpoint"] == "/check_frame_batch"
    assert line["pred_label"] == "FAKE"
    assert line["status"] == "ok"
    assert line["n_frames"] == 1 and line["n_scored"] == 1
    assert line["gcs_prefix"] == rec.gcs_prefix
    # Index lines must NOT carry image bytes or the full per-frame array.
    assert "frames" not in line and "raw_bytes" not in line


# ──────────────────────────────────────────
# ObservabilityUploader
# ──────────────────────────────────────────
_STATIC = {
    "checkpoint_paths": {"custom": "gs://b/ck.pth"}, "use_arcface": False,
    "device": "cuda", "app_version": "0.7.0", "git_sha": "deadbeef",
    "hostname": "deepfakebench2", "pid": 4242,
}


def make_uploader(bucket, **over):
    kw = dict(
        bucket=bucket, num_workers=2, max_queue_bytes=10_000, max_record_bytes=5_000,
        max_retries=3, index_flush_every=1, retry_sleep=lambda _s: None,
        static=_STATIC, prefix="v1",
    )
    kw.update(over)
    return obs.ObservabilityUploader("remote-live-data", **kw)


def frame(seq, nbytes=10, content_type="image/jpeg", **over):
    return obs.FrameCapture(seq=seq, raw_bytes=b"x" * nbytes, content_type=content_type, **over)


def test_enqueue_within_budget_counts_and_queues():
    up = make_uploader(FakeBucket())  # not started: nothing drains the queue
    up.enqueue(_record([frame(0, 100)]))
    s = up.stats()
    assert s["enqueued"] == 1
    assert s["queue_depth"] == 1
    assert s["bytes_in_flight"] == 100
    assert s["dropped_records"] == 0


def test_enqueue_drops_record_exceeding_per_record_cap():
    up = make_uploader(FakeBucket(), max_record_bytes=500)
    up.enqueue(_record([frame(0, 600)]))
    s = up.stats()
    assert s["enqueued"] == 0
    assert s["dropped_records"] == 1
    assert s["dropped_bytes"] == 600
    assert s["queue_depth"] == 0


def test_enqueue_drops_when_total_byte_budget_would_be_exceeded():
    up = make_uploader(FakeBucket(), max_queue_bytes=1000)
    up.enqueue(_record([frame(0, 600)]))   # ok -> 600 in flight
    up.enqueue(_record([frame(0, 600)]))   # 1200 > 1000 -> dropped
    s = up.stats()
    assert s["enqueued"] == 1
    assert s["dropped_records"] == 1
    assert s["queue_depth"] == 1


def test_enqueue_is_noop_and_never_raises_when_disabled():
    up = obs.ObservabilityUploader("remote-live-data",
                                   bucket_factory=lambda: (_ for _ in ()).throw(RuntimeError("no creds")))
    assert up.enabled is False
    up.enqueue(_record([frame(0, 100)]))   # must not raise
    assert up.stats()["enabled"] is False


def test_enqueue_never_raises_even_on_internal_error():
    up = make_uploader(FakeBucket())
    up.enqueue("not a record")   # bad input must be swallowed, not raised
    assert up.stats()["dropped_records"] >= 1


def test_upload_record_uploads_frames_then_meta_last():
    bucket = FakeBucket()
    up = make_uploader(bucket)
    rec = _record([frame(0, content_type="image/jpeg"), frame(1, content_type="image/png")],
                  status="ok", pred_label="REAL", confidence=0.2)
    up._upload_record(rec)

    names = bucket.names()
    # meta.json is written AFTER all of the record's frame objects (commit marker).
    # (An _index part file may legitimately be flushed afterwards.)
    frame_idxs = [i for i, n in enumerate(names) if "/frame_" in n]
    meta_idx = next(i for i, n in enumerate(names) if n.endswith("/meta.json"))
    assert frame_idxs and meta_idx > max(frame_idxs)
    # frame object keys carry seq + correct extension under the computed prefix
    assert any(n.endswith("/frame_000.jpg") for n in names)
    assert any(n.endswith("/frame_001.png") for n in names)
    # prefix layout: v1/date=YYYY-MM-DD/ip=.../req=<utc>_<reqid8>
    assert re.match(r"^v1/date=\d{4}-\d{2}-\d{2}/ip=198-51-100-23/req=.+_abcd1234$",
                    rec.gcs_prefix)
    # frame.gcs_object_path was recorded for the meta
    assert rec.frames[0].gcs_object_path.endswith("/frame_000.jpg")


def test_uploaded_meta_json_roundtrips_with_schema_and_model_identity():
    bucket = FakeBucket()
    up = make_uploader(bucket)
    rec = _record([frame(0, scored=True, prob=0.9, verdict="FAKE")],
                  status="ok", pred_label="FAKE", confidence=0.9)
    up._upload_record(rec)

    data, ctype = bucket.get(rec.frames[0].gcs_object_path.rsplit("/", 1)[0] + "/meta.json")
    assert ctype == "application/json"
    meta = json.loads(data)
    assert meta["schema_version"] == obs.SCHEMA_VERSION
    assert meta["model"]["checkpoint_path"] == "gs://b/ck.pth"
    assert meta["frames"][0]["prob"] == 0.9


def test_upload_blob_retries_transient_failures_then_succeeds():
    bucket = FakeBucket()
    up = make_uploader(bucket, max_retries=3)
    name = "v1/probe/x.jpg"
    bucket.fail_counts[name] = 2          # fail twice, succeed on the 3rd
    assert up._upload_blob(name, b"data", "image/jpeg") is True
    assert name in bucket.names()


def test_upload_blob_gives_up_after_max_retries():
    bucket = FakeBucket()
    up = make_uploader(bucket, max_retries=2)
    name = "v1/probe/y.jpg"
    bucket.fail_counts[name] = 99          # always fails
    assert up._upload_blob(name, b"data", "image/jpeg") is False
    assert name not in bucket.names()


def test_upload_record_writes_index_part_file():
    bucket = FakeBucket()
    up = make_uploader(bucket, index_flush_every=1)
    rec = _record([frame(0, scored=True, prob=0.9)], status="ok", pred_label="FAKE")
    up._upload_record(rec)
    up.flush_index(force=True)

    index_objs = [n for n in bucket.names() if "/_index/part-" in n and n.endswith(".jsonl")]
    assert index_objs, "expected an _index/part-*.jsonl object"
    data, _c = bucket.get(index_objs[0])
    lines = [json.loads(l) for l in data.decode().splitlines() if l.strip()]
    assert any(l["request_id"] == "abcd1234" for l in lines)


def test_threaded_drain_uploads_everything_and_releases_bytes():
    bucket = FakeBucket()
    up = make_uploader(bucket, num_workers=3)
    up.start()
    for i in range(7):
        r = _record([frame(0, 50, scored=True, prob=0.3)])
        r.request_id = f"req{i:04d}"
        up.enqueue(r)
    up.stop(timeout=10)

    meta_objs = [n for n in bucket.names() if n.endswith("/meta.json")]
    assert len(meta_objs) == 7
    s = up.stats()
    assert s["uploaded_records"] == 7
    assert s["bytes_in_flight"] == 0


# ──────────────────────────────────────────
# Regression tests for code-review fixes
# ──────────────────────────────────────────
def test_failed_frame_upload_clears_gcs_object_path():
    # #3: a frame whose blob upload permanently fails must NOT advertise a
    # gcs_object_path in meta (it would 404 for the reader).
    bucket = FakeBucket()
    bucket.fail_substr = "frame_000"
    up = make_uploader(bucket, max_retries=0)
    rec = _record([frame(0, content_type="image/jpeg")], status="ok")
    up._upload_record(rec)
    assert rec.frames[0].gcs_object_path is None
    # meta.json still uploaded (commit marker for the rest of the record)
    assert any(n.endswith("/meta.json") for n in bucket.names())


def test_no_index_line_when_meta_upload_fails():
    # #4: if meta.json (the commit marker) fails, do NOT emit an index line —
    # otherwise sessionize would surface a request with no meta at its prefix.
    bucket = FakeBucket()
    bucket.fail_substr = "meta.json"
    up = make_uploader(bucket, max_retries=0, index_flush_every=1)
    rec = _record([frame(0)], status="ok")
    up._upload_record(rec)
    up.flush_index(force=True)
    assert not [n for n in bucket.names() if "/_index/" in n]
    assert up.stats()["failed"] >= 1


def test_failed_index_flush_rebuffers_and_retries():
    # #8: a failed index flush must keep the lines buffered (not advance past
    # them) so a later flush re-uploads them.
    bucket = FakeBucket()
    bucket.fail_substr = "_index"
    up = make_uploader(bucket, max_retries=0, index_flush_every=1000)
    up._buffer_index("2026-05-25", {"request_id": "x", "ts_epoch_ms": 1})
    up.flush_index(force=True)            # fails → re-buffered
    assert not [n for n in bucket.names() if "/_index/" in n]
    bucket.fail_substr = None
    up.flush_index(force=True)            # now succeeds
    idx = [n for n in bucket.names() if "/_index/" in n]
    assert idx
    assert b'"request_id":"x"' in bucket.get(idx[0])[0]


def test_req_prefix_uses_full_request_id():
    # #15: the req= segment must use the full request_id (not a 32-bit prefix)
    # so two same-IP same-ms requests cannot collide and overwrite each other.
    bucket = FakeBucket()
    up = make_uploader(bucket)
    rec = _record([frame(0)])
    rec.request_id = "abcd1234ef567890deadbeefcafef00d"
    up._upload_record(rec)
    assert rec.gcs_prefix.endswith("_abcd1234ef567890deadbeefcafef00d")


def test_background_flusher_emits_index_without_a_second_request():
    # #7: a single request followed by idle must still get its index line
    # flushed (a background flusher), not wait for the next request.
    import time as _t
    bucket = FakeBucket()
    up = make_uploader(bucket, num_workers=1, index_flush_every=1000, index_flush_seconds=0.1)
    up.start()
    try:
        up.enqueue(_record([frame(0, 50, scored=True, prob=0.3)]))
        deadline = _t.time() + 5
        while _t.time() < deadline and not [n for n in bucket.names() if "/_index/" in n]:
            _t.sleep(0.05)
        assert [n for n in bucket.names() if "/_index/" in n], "background flusher never wrote the index"
    finally:
        up.stop(timeout=5)


# ──────────────────────────────────────────
# app3.py integration (real handler code paths via TestClient)
#
# The heavy startup (GCS weights + YOLO + CUDA) is skipped; a fake model and a
# fake-bucket uploader are injected into app.state so the capture hooks in
# /check_frame and /check_frame_batch run end-to-end on CPU with no network.
# ──────────────────────────────────────────
pytestmark_cv2 = pytest.mark.skipif(not _HAVE_CV2, reason="cv2 required for image encode")


def _png_bytes(h=100, w=100, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)  # noise -> passes lap_var gate
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture
def app_client():
    import torch
    from fastapi.testclient import TestClient
    import app3

    # Snapshot the shared (import-once) app singleton so we can restore it and
    # not leak fake state into any other test that imports app3.
    saved_startup = list(app3.app.router.on_startup)
    saved_shutdown = list(app3.app.router.on_shutdown)
    saved_state = dict(app3.app.state._state)

    # Skip the heavy startup/shutdown; inject state directly.
    app3.app.router.on_startup = []
    app3.app.router.on_shutdown = []

    class FakeModel:
        def __init__(self, prob):
            self.prob = prob

        def __call__(self, batch, inference=False):
            n = batch["image"].shape[0]
            return {"prob": torch.full((n,), float(self.prob))}

    app3.app.state.models = {"custom": FakeModel(0.9)}
    app3.app.state.loaded_weights_paths = {"custom": "gs://b/ck.pth"}
    app3.app.state.yolo_available = False

    bucket = FakeBucket()
    up = obs.ObservabilityUploader(
        "remote-live-data", bucket=bucket, num_workers=1, index_flush_every=1,
        retry_sleep=lambda _s: None,
        static={"checkpoint_paths": {"custom": "gs://b/ck.pth"}, "device": "cpu",
                "app_version": "0.7.0", "hostname": "testhost", "pid": 1},
    )
    up.start()
    app3.app.state.obs = up

    try:
        with TestClient(app3.app) as client:
            yield client, bucket, up
    finally:
        up.stop(timeout=5)
        # Restore the app singleton so other tests see the real handlers/state.
        app3.app.router.on_startup = saved_startup
        app3.app.router.on_shutdown = saved_shutdown
        app3.app.state._state.clear()
        app3.app.state._state.update(saved_state)


@pytestmark_cv2
def test_check_frame_happy_path_captures_record(app_client):
    client, bucket, up = app_client
    png = _png_bytes()
    resp = client.post(
        "/check_frame?model_type=custom&threshold=0.5&gate_profile=legacy",
        files={"file": ("myframe.png", png, "image/png")},
        headers={"X-Forwarded-For": "198.51.100.23, 10.0.0.1", "X-Client-Id": "user-7"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["pred_label"] == "FAKE" and abs(body["fake_prob"] - 0.9) < 1e-6

    up.stop(timeout=5)  # drain (idempotent; fixture teardown re-calls safely)
    metas = [n for n in bucket.names() if n.endswith("/meta.json")]
    assert len(metas) == 1
    meta = json.loads(bucket.get(metas[0])[0])
    assert meta["status"] == "ok"
    assert meta["endpoint"] == "/check_frame"
    assert meta["client"]["client_ip"] == "198.51.100.23"   # first XFF hop
    assert meta["client"]["x_client_id"] == "user-7"
    assert meta["params"]["threshold"] == 0.5
    assert meta["model"]["checkpoint_path"] == "gs://b/ck.pth"
    assert len(meta["frames"]) == 1
    f0 = meta["frames"][0]
    assert f0["scored"] is True and abs(f0["prob"] - 0.9) < 1e-6 and f0["verdict"] == "FAKE"
    # original bytes were uploaded verbatim
    raw, ctype = bucket.get(f0["gcs_object_path"])
    assert raw == png and ctype == "image/png"


@pytestmark_cv2
def test_check_frame_gated_frame_is_still_captured(app_client):
    client, bucket, up = app_client
    tiny = _png_bytes(h=50, w=50)  # min_dim 50 < legacy 80 -> gated
    resp = client.post(
        "/check_frame?model_type=custom&gate_profile=legacy",
        files={"file": ("tiny.png", tiny, "image/png")},
    )
    assert resp.status_code == 200
    up.stop(timeout=5)
    meta = json.loads(bucket.get([n for n in bucket.names() if n.endswith("/meta.json")][0])[0])
    assert meta["status"] == "gate_failed"
    assert meta["frames"][0]["gate_pass"] is False
    assert meta["frames"][0]["scored"] is False


@pytestmark_cv2
def test_check_frame_batch_mixed_captures_per_frame_truth(app_client):
    client, bucket, up = app_client
    good = _png_bytes(h=150, w=150, seed=1)  # >= t5c min_dim (120) -> scored
    tiny = _png_bytes(h=40, w=40, seed=2)    # < 120 -> gated
    resp = client.post(
        "/check_frame_batch?model_type=custom&threshold=0.5&gate_profile=t5c",
        files=[
            ("files", ("a.png", good, "image/png")),
            ("files", ("b.png", tiny, "image/png")),
        ],
    )
    assert resp.status_code == 200
    up.stop(timeout=5)
    meta = json.loads(bucket.get([n for n in bucket.names() if n.endswith("/meta.json")][0])[0])
    assert meta["endpoint"] == "/check_frame_batch"
    assert len(meta["frames"]) == 2
    counts = meta["counts"]
    assert counts["total"] == 2 and counts["scored"] == 1 and counts["gated"] == 1
    # both original frames uploaded
    assert len([n for n in bucket.names() if "/frame_" in n]) == 2


@pytestmark_cv2
def test_check_frame_batch_caps_captured_bytes(app_client):
    # #1: the batch handler must bound the raw bytes it retains by the uploader's
    # max_record_bytes, so a large batch can't pin unbounded memory (OOM the box).
    client, bucket, up = app_client
    up.max_record_bytes = 100_000  # ~100KB cap
    f1 = _png_bytes(h=150, w=150, seed=1)  # ~60-70KB
    f2 = _png_bytes(h=150, w=150, seed=2)  # cumulative > cap -> not retained
    resp = client.post(
        "/check_frame_batch?model_type=custom&gate_profile=legacy",
        files=[("files", ("a.png", f1, "image/png")), ("files", ("b.png", f2, "image/png"))],
    )
    assert resp.status_code == 200
    assert len(resp.json()["probs"]) == 2  # both frames still SCORED
    up.stop(timeout=5)
    meta = json.loads(bucket.get([n for n in bucket.names() if n.endswith("/meta.json")][0])[0])
    assert len(meta["frames"]) == 2
    # only the first frame (within budget) had its image uploaded
    assert len([n for n in bucket.names() if "/frame_" in n]) == 1
    assert meta["frames"][0]["capture_skipped"] is False
    assert meta["frames"][1]["capture_skipped"] is True
    assert meta["frames"][1]["gcs_object_path"] is None
    assert sum(fr["bytes_size"] for fr in meta["frames"]) <= 100_000


@pytestmark_cv2
def test_inference_still_succeeds_when_obs_is_none(app_client):
    client, bucket, up = app_client
    import app3
    app3.app.state.obs = None  # observability disabled mid-flight
    resp = client.post(
        "/check_frame?model_type=custom",
        files={"file": ("f.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["pred_label"] == "FAKE"


@pytestmark_cv2
def test_inference_not_blocked_by_failing_bucket(app_client):
    client, bucket, up = app_client
    # Make every upload fail: response must still return promptly (uploads are
    # off the request path). This is the "never slow/break inference" guarantee.
    import observability as _obs
    orig = up._upload_blob
    up._upload_blob = lambda *a, **k: False
    try:
        resp = client.post(
            "/check_frame?model_type=custom",
            files={"file": ("f.png", _png_bytes(), "image/png")},
        )
        assert resp.status_code == 200
        assert abs(resp.json()["fake_prob"] - 0.9) < 1e-6
    finally:
        up._upload_blob = orig


def test_obs_stats_endpoint(app_client):
    client, bucket, up = app_client
    resp = client.get("/obs_stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert "queue_depth" in body and "dropped_records" in body


def test_ping_unchanged(app_client):
    client, bucket, up = app_client
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert resp.json() == {"message": "pong"}
