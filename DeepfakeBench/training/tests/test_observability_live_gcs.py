"""Real-GCS round-trip test for the observability pipeline.

This test talks to the ACTUAL `remote-live-data` bucket (via ADC), so it is
gated behind OBS_LIVE_BUCKET_TEST=1 and skipped otherwise. It writes under a
unique, clearly-marked self-test prefix (`__obs_selftest__/<runid>/`) — never
the production `v1/` namespace — and DELETES everything it wrote in a finally,
so it leaves no residue in the bucket.

Run with:
    OBS_LIVE_BUCKET_TEST=1 python -m pytest tests/test_observability_live_gcs.py -q -s
"""
from __future__ import annotations

import json
import os
import time
import uuid

import pytest

import observability as obs
from analysis.observability_sessions.sessionize import sessionize, summarize_session

pytestmark = pytest.mark.skipif(
    os.getenv("OBS_LIVE_BUCKET_TEST") != "1",
    reason="real-GCS test; set OBS_LIVE_BUCKET_TEST=1 to run",
)

BUCKET = os.getenv("OBS_LIVE_BUCKET", "remote-live-data")


def _record(request_id, client_ip, epoch_ms, raw, ct="image/jpeg"):
    fc = obs.FrameCapture(seq=0, raw_bytes=raw, filename="f.jpg", content_type=ct,
                          dims_hw=(120, 120), gate_pass=True, prob=0.83, scored=True, verdict="FAKE")
    return obs.CaptureRecord(
        request_id=request_id, received_utc=time.strftime("%Y%m%dT%H%M%S000Z", time.gmtime(epoch_ms / 1000)),
        received_epoch_ms=epoch_ms, endpoint="/check_frame_batch",
        client_ip=client_ip, remote_addr=client_ip, xff_chain=[client_ip],
        user_agent="live-selftest", x_client_id=None, model_type="custom",
        threshold=0.5, gate_profile="t5c", gate_spec={"min_dim": 120},
        yolo_conf_threshold=0.2, recrop=False, debug=False, frames=[fc],
        status="ok", pred_label="FAKE", confidence=0.83,
    )


def test_live_gcs_round_trip_write_then_read_then_sessionize():
    from google.cloud import storage

    run_prefix = f"__obs_selftest__/{uuid.uuid4().hex[:12]}"
    client = storage.Client()
    bucket = client.bucket(BUCKET)

    up = obs.ObservabilityUploader(
        BUCKET, num_workers=2, index_flush_every=1, prefix=run_prefix,
        retry_sleep=lambda _s: None,
        static={"checkpoint_paths": {"custom": "gs://b/ck.pth"}, "device": "cpu",
                "app_version": "0.7.0", "hostname": "live-selftest", "pid": os.getpid()},
    )
    assert up.enabled, "uploader failed to bind the real bucket (check ADC)"

    now_ms = int(time.time() * 1000)
    ip = "203.0.113.200"
    records = [
        _record("live-req-aaaa1111", ip, now_ms, b"\xff\xd8\xff selftest-bytes-A" * 8),
        _record("live-req-bbbb2222", ip, now_ms + 2000, b"\xff\xd8\xff selftest-bytes-B" * 8),
        _record("live-req-cccc3333", "203.0.113.201", now_ms + 3000, b"\xff\xd8\xff selftest-bytes-C" * 8),
    ]
    date = obs._date_from_epoch_ms(now_ms)

    try:
        up.start()
        for r in records:
            up.enqueue(r)
        up.stop(timeout=60)

        # ---- assert the objects really landed in GCS ----
        names = [b.name for b in client.list_blobs(bucket, prefix=run_prefix + "/")]
        metas = [n for n in names if n.endswith("/meta.json")]
        frames = [n for n in names if "/frame_000.jpg" in n]
        idx = [n for n in names if "/_index/part-" in n and n.endswith(".jsonl")]
        assert len(metas) == 3, f"expected 3 meta.json, got {len(metas)}: {names}"
        assert len(frames) == 3, f"expected 3 frame objects, got {len(frames)}"
        assert idx, "expected at least one _index part file"

        # ---- meta.json round-trips with the bytes we sent ----
        meta = json.loads(bucket.blob(metas[0]).download_as_bytes())
        assert meta["schema_version"] == obs.SCHEMA_VERSION
        assert meta["client"]["client_ip"] in (ip, "203.0.113.201")
        f0 = meta["frames"][0]
        assert f0["gcs_object_path"] and f0["sha256"]
        stored = bucket.blob(f0["gcs_object_path"]).download_as_bytes()
        import hashlib
        assert hashlib.sha256(stored).hexdigest() == f0["sha256"], "uploaded image bytes != meta sha256"

        # ---- read the index back through the real read path + sessionize ----
        rows = []
        for n in idx:
            for line in bucket.blob(n).download_as_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        got_ids = {r["request_id"] for r in rows}
        assert {"live-req-aaaa1111", "live-req-bbbb2222", "live-req-cccc3333"} <= got_ids

        sessions = sessionize(rows, gap_minutes=30)
        by_ip = {s[0]["client_ip"]: s for s in sessions}
        assert by_ip[ip] and len(by_ip[ip]) == 2          # two requests from the same IP → one session
        assert len(by_ip["203.0.113.201"]) == 1
        summary = summarize_session(by_ip[ip])
        assert summary["n_requests"] == 2 and summary["n_frames"] == 2
        print(f"\n[live-gcs] round-trip OK under gs://{BUCKET}/{run_prefix}/  "
              f"({len(names)} objects, {len(sessions)} sessions)")
    finally:
        # ---- clean up: delete everything this test wrote ----
        to_delete = list(client.list_blobs(bucket, prefix=run_prefix + "/"))
        for b in to_delete:
            try:
                b.delete()
            except Exception:
                pass
        leftover = list(client.list_blobs(bucket, prefix=run_prefix + "/"))
        assert not leftover, f"cleanup failed, {len(leftover)} objects remain under {run_prefix}"
