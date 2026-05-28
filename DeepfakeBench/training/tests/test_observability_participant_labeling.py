"""Failing tests (TDD RED) for the participant-labeling enrichment of
observability.py.

What this adds to the per-request capture → GCS pipeline:

  1. A `parse_pid_from_filename` helper that extracts the participant id
     (and the per-participant seq) from the encoded multipart filename the
     WMA client writes (`pid=<pid>__seq=<n>__frame_<i>.<ext>`).
  2. A `participant_id` field on `FrameCapture` (+ optional `participant_seq`)
     that flows through `to_meta()` and into the per-request meta.json so
     offline analysis can group frames by participant identity.
  3. A per-participant breakdown in `CaptureRecord.build_index_line()` and
     in `to_meta_dict()`'s `counts` block.
  4. A pid-partitioned GCS object key — frames carrying a participant id
     land under `…/req=…/pid=<pid>/frame_NNN.<ext>` so any reader can list
     a participant's frames with a single prefix scan. Frames WITHOUT a pid
     keep the old flat layout (`…/req=…/frame_NNN.<ext>`) so legacy
     callers stay byte-identical in the bucket.

The tests reuse the existing FakeBucket / `_record` / `make_uploader` /
`frame()` helpers from `tests.test_observability`. They MUST fail on the
current code; B6 makes them pass.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

# Reuse the in-memory FakeBucket + builders from the main observability test
# module so we don't reimplement them and stay aligned with their conventions.
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from test_observability import (  # noqa: E402
    FakeBucket,
    _STATIC,
    _record,
    frame,
    make_uploader,
)

import observability as obs  # noqa: E402


# ──────────────────────────────────────────
# B3 — parse_pid_from_filename
# ──────────────────────────────────────────
class TestParsePidFromFilename:
    def test_parses_simple_encoded_filename(self):
        pid, seq = obs.parse_pid_from_filename("pid=Alice__seq=42__frame_0.png")
        assert pid == "Alice"
        assert seq == 42

    def test_parses_pid_with_spaces_and_dots(self):
        # WMA-side `_encode_pid_for_filename` keeps `[A-Za-z0-9._ -]` as-is,
        # so real-name pids like "John Q. Public" survive intact.
        pid, seq = obs.parse_pid_from_filename(
            "pid=John Q. Public__seq=7__frame_3.jpg"
        )
        assert pid == "John Q. Public"
        assert seq == 7

    def test_parses_pid_with_single_underscore(self):
        # Single-underscore inside the pid must NOT be confused with the `__`
        # separator. The regex is non-greedy on the pid, so it stops at the
        # FIRST `__seq=` boundary, leaving any single `_` inside the pid.
        pid, _seq = obs.parse_pid_from_filename(
            "pid=Bob_Smith__seq=1__frame_0.png"
        )
        assert pid == "Bob_Smith"

    def test_returns_none_for_legacy_generic_filename(self):
        # Backward compat: any caller that hasn't adopted the encoding yet
        # uploads `frame_<i>.png`. The parser must return (None, None), and
        # downstream the FrameCapture.participant_id stays None.
        assert obs.parse_pid_from_filename("frame_0.png") == (None, None)
        assert obs.parse_pid_from_filename("anything.png") == (None, None)

    def test_returns_none_for_blank_or_none_filename(self):
        assert obs.parse_pid_from_filename(None) == (None, None)
        assert obs.parse_pid_from_filename("") == (None, None)

    def test_returns_none_for_malformed_seq(self):
        # Anything that fails the integer parse safely degrades to (None, None).
        assert obs.parse_pid_from_filename("pid=A__seq=NOTANUMBER__frame_0.png") == (None, None)


# ──────────────────────────────────────────
# B4 — FrameCapture.participant_id + per-pid GCS key
# ──────────────────────────────────────────
class TestFrameCaptureParticipantId:
    def test_participant_id_defaults_to_none(self):
        fc = obs.FrameCapture(seq=0, raw_bytes=b"x")
        assert fc.participant_id is None
        assert fc.to_meta()["participant_id"] is None

    def test_participant_id_round_trips_through_to_meta(self):
        fc = obs.FrameCapture(
            seq=0, raw_bytes=b"x",
            participant_id="Alice", participant_seq=42,
        )
        meta = fc.to_meta()
        assert meta["participant_id"] == "Alice"
        assert meta["participant_seq"] == 42


class TestPerPidGCSKey:
    def test_pid_partitions_blob_key(self):
        bucket = FakeBucket()
        up = make_uploader(bucket)
        rec = _record([
            obs.FrameCapture(seq=0, raw_bytes=b"a" * 10, content_type="image/png",
                             participant_id="Alice", participant_seq=1),
            obs.FrameCapture(seq=1, raw_bytes=b"b" * 10, content_type="image/png",
                             participant_id="Bob_Smith", participant_seq=2),
        ])
        up._upload_record(rec)
        names = bucket.names()
        # Each pid lives under its own `pid=<pid>/` segment inside the
        # request prefix. seq stays unique per-request so frames across
        # pids never collide.
        assert any(re.search(r"/pid=Alice/frame_000\.png$", n) for n in names), names
        assert any(re.search(r"/pid=Bob_Smith/frame_001\.png$", n) for n in names), names
        # And the FrameCapture records the chosen pid-partitioned path.
        assert rec.frames[0].gcs_object_path is not None
        assert "/pid=Alice/" in rec.frames[0].gcs_object_path

    def test_no_pid_keeps_legacy_flat_layout(self):
        # Backward compat: a frame with NO participant_id (legacy caller,
        # parse miss) must land at `req=…/frame_NNN.ext` exactly like
        # before — readers of historical data don't get broken.
        bucket = FakeBucket()
        up = make_uploader(bucket)
        rec = _record([obs.FrameCapture(seq=0, raw_bytes=b"x" * 10,
                                        content_type="image/png")])
        up._upload_record(rec)
        names = bucket.names()
        assert any(n.endswith("/frame_000.png") for n in names)
        # No spurious pid= segment got injected.
        assert not any("/pid=" in n for n in names)

    def test_pid_with_unsafe_chars_is_sanitized_in_key(self):
        # Defense in depth: WMA already sanitizes the pid before encoding it,
        # but the key MUST stay GCS-path-safe regardless of what arrives.
        bucket = FakeBucket()
        up = make_uploader(bucket)
        rec = _record([obs.FrameCapture(seq=0, raw_bytes=b"x" * 10,
                                        content_type="image/png",
                                        participant_id="../bad/name")])
        up._upload_record(rec)
        names = bucket.names()
        for n in names:
            assert "/.." not in n, n
            # The pid= segment must not introduce a literal `/` either.
            head = n.split("/pid=", 1)[1] if "/pid=" in n else ""
            assert "/" not in head.split("/", 1)[0], n


# ──────────────────────────────────────────
# B5 — participants in build_index_line + to_meta_dict
# ──────────────────────────────────────────
class TestParticipantsIndexAndMeta:
    def test_build_index_line_contains_per_pid_counts(self):
        # The index line is the offline-sessionization read path. Carrying
        # per-pid counts here means readers can filter by participant
        # without dereferencing each meta.json.
        rec = _record([
            obs.FrameCapture(0, b"x", participant_id="Alice", gate_pass=True, scored=True, prob=0.7),
            obs.FrameCapture(1, b"x", participant_id="Alice", gate_pass=True, scored=True, prob=0.6),
            obs.FrameCapture(2, b"x", participant_id="Bob",   gate_pass=False, scored=False),
            obs.FrameCapture(3, b"x"),  # no pid -> unknown bucket
        ], status="ok", pred_label="REAL", confidence=0.4)
        line = rec.build_index_line()
        assert "participants" in line
        # {pid: {total, scored, gated, failed}} (failed = decode/processing)
        assert line["participants"]["Alice"] == {"total": 2, "scored": 2, "gated": 0, "failed": 0}
        assert line["participants"]["Bob"]["gated"] == 1
        # Frames with no pid roll up under the literal key 'unknown' so
        # the breakdown remains exhaustive — total over all entries equals
        # the request's frame count.
        assert "unknown" in line["participants"]
        total = sum(p["total"] for p in line["participants"].values())
        assert total == 4

    def test_to_meta_dict_carries_participants_block(self):
        rec = _record([
            obs.FrameCapture(0, b"x", participant_id="Alice", scored=True, prob=0.9, verdict="FAKE"),
            obs.FrameCapture(1, b"x", participant_id="Bob",   scored=True, prob=0.1, verdict="REAL"),
        ], status="ok", pred_label="REAL", confidence=0.5)
        meta = rec.to_meta_dict(_STATIC)
        # `participants` is a top-level block (not nested under `counts`) so
        # the legacy `counts` shape stays byte-identical for any downstream
        # that strict-compares it.
        assert "participants" in meta
        assert set(meta["participants"].keys()) >= {"Alice", "Bob"}
        # `counts` keeps its original 4-key shape.
        assert set(meta["counts"].keys()) == {"total", "scored", "gated", "failed"}


# ──────────────────────────────────────────
# B6 (precondition) — uploaded meta.json carries pid info
# ──────────────────────────────────────────
class TestUploadedMetaRoundTrip:
    def test_uploaded_meta_json_carries_per_frame_participant_ids(self):
        # The captured meta.json must let offline tooling map prob → pid
        # WITHOUT reaching for the blob layout.
        bucket = FakeBucket()
        up = make_uploader(bucket)
        rec = _record([
            obs.FrameCapture(0, b"a" * 10, content_type="image/png",
                             participant_id="Alice", participant_seq=5,
                             scored=True, prob=0.7, verdict="FAKE"),
        ], status="ok", pred_label="FAKE", confidence=0.7)
        up._upload_record(rec)
        meta_blob = [n for n in bucket.names() if n.endswith("/meta.json")][0]
        data, ctype = bucket.get(meta_blob)
        assert ctype == "application/json"
        meta = json.loads(data)
        assert meta["frames"][0]["participant_id"] == "Alice"
        assert meta["frames"][0]["participant_seq"] == 5
