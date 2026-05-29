"""Tests for per-participant grouping in the observability viewer.

`group_frames_by_participant` takes the flat `frames` list from one or more
request meta.json records (the frames carry the WMA-encoded filename) and groups
them by participant identity — the missing axis for the session→participant
board. Identity is parsed from the filename via the SAME helper the server uses
(`observability.parse_pid_from_filename`), so the percent-encoded wire form
(pid%3D…) groups correctly on today's already-captured data.

Pure transform — no GCS, no Flask. Mirrors the style of test_obs_viewer.py.
"""
from __future__ import annotations

import pytest

import viewer.obs_server as obs


def fr(filename, *, prob=None, scored=False, verdict=None, gate_pass=None,
       face_found=None, seq=0):
    """Build a frame dict shaped like a meta.json `frames[]` entry."""
    return {
        "filename": filename, "prob": prob, "scored": scored, "verdict": verdict,
        "gate_pass": gate_pass, "face_found": face_found, "seq": seq,
    }


def test_groups_frames_by_participant():
    frames = [
        fr("pid=Alice__seq=1__frame_0.png", prob=0.9, scored=True, verdict="FAKE", seq=0),
        fr("pid=Bob__seq=1__frame_1.png", prob=0.1, scored=True, verdict="REAL", seq=1),
        fr("pid=Alice__seq=2__frame_2.png", prob=0.8, scored=True, verdict="FAKE", seq=2),
    ]
    by = {p["participant_id"]: p for p in obs.group_frames_by_participant(frames, threshold=0.5)}
    assert set(by) == {"Alice", "Bob"}
    assert by["Alice"]["n_frames"] == 2
    assert by["Bob"]["n_frames"] == 1


def test_mean_score_and_verdict_over_scored_frames_only():
    frames = [
        fr("pid=A__seq=1__frame_0.png", prob=0.9, scored=True, verdict="FAKE"),
        fr("pid=A__seq=2__frame_1.png", prob=0.7, scored=True, verdict="FAKE"),
        fr("pid=A__seq=3__frame_2.png", prob=-1.0, scored=False, gate_pass=False),  # gated sentinel
    ]
    p = obs.group_frames_by_participant(frames, threshold=0.5)[0]
    assert p["n_frames"] == 3
    assert p["n_scored"] == 2
    assert p["n_gated"] == 1
    assert p["mean_score"] == pytest.approx(0.8)   # (0.9+0.7)/2, sentinel excluded
    assert p["verdict"] == "FAKE"


def test_verdict_real_when_mean_below_threshold():
    frames = [fr("pid=A__seq=1__frame_0.png", prob=0.2, scored=True, verdict="REAL")]
    p = obs.group_frames_by_participant(frames, threshold=0.5)[0]
    assert p["verdict"] == "REAL"


def test_verdict_none_when_nothing_scored():
    frames = [fr("pid=A__seq=1__frame_0.png", prob=-1.0, scored=False, gate_pass=False)]
    p = obs.group_frames_by_participant(frames, threshold=0.5)[0]
    assert p["n_scored"] == 0
    assert p["mean_score"] is None
    assert p["verdict"] is None


def test_unlabeled_frames_group_as_unknown():
    p = obs.group_frames_by_participant([fr("frame_0.png", prob=0.2, scored=True)])[0]
    assert p["participant_id"] == "unknown"


def test_percent_encoded_filenames_group_by_decoded_pid():
    # The real wire form (aiohttp quote_fields). Ties to the parser URL-decode
    # fix — this is what today's bucket actually contains.
    frames = [
        fr("pid%3DAlice%20Smith__seq%3D1__frame_0.png", prob=0.9, scored=True, verdict="FAKE"),
        fr("pid%3DAlice%20Smith__seq%3D2__frame_1.png", prob=0.8, scored=True, verdict="FAKE"),
    ]
    parts = obs.group_frames_by_participant(frames)
    assert len(parts) == 1
    assert parts[0]["participant_id"] == "Alice Smith"
    assert parts[0]["n_frames"] == 2


def test_participants_sorted_flagged_first():
    frames = [
        fr("pid=Low__seq=1__frame_0.png", prob=0.10, scored=True, verdict="REAL"),
        fr("pid=High__seq=1__frame_1.png", prob=0.95, scored=True, verdict="FAKE"),
    ]
    parts = obs.group_frames_by_participant(frames, threshold=0.5)
    assert [p["participant_id"] for p in parts] == ["High", "Low"]


def test_frames_within_participant_sorted_by_participant_seq():
    frames = [
        fr("pid=A__seq=3__frame_0.png", prob=0.5, scored=True),
        fr("pid=A__seq=1__frame_1.png", prob=0.5, scored=True),
        fr("pid=A__seq=2__frame_2.png", prob=0.5, scored=True),
    ]
    p = obs.group_frames_by_participant(frames)[0]
    assert [f["participant_seq"] for f in p["frames"]] == [1, 2, 3]


# ──────────────────────────────────────────
# /api/participants route (data access monkeypatched; no network)
# ──────────────────────────────────────────
@pytest.fixture
def client():
    obs.app.config["TESTING"] = True
    return obs.app.test_client()


def _meta(frames, threshold=0.8):
    return {"frames": frames, "params": {"threshold": threshold}}


def _mframe(seq, filename, prob, prefix, scored=True, verdict="REAL"):
    return {
        "seq": seq, "filename": filename, "prob": prob, "scored": scored,
        "verdict": verdict, "gate_pass": True,
        "gcs_object_path": f"{prefix}frame_{seq:03d}.png",
    }


def test_route_participants_groups_across_requests(client, monkeypatch):
    pa, pb = "v1/d/ip/req=A/", "v1/d/ip/req=B/"
    metas = {
        pa: _meta([
            _mframe(0, "pid=Alice__seq=1__frame_0.png", 0.9, pa, verdict="FAKE"),
            _mframe(1, "pid=Bob__seq=1__frame_1.png", 0.1, pb, verdict="REAL"),
        ]),
        pb: _meta([
            _mframe(0, "pid=Alice__seq=2__frame_0.png", 0.8, pb, verdict="FAKE"),
        ]),
    }
    monkeypatch.setattr(obs, "download_meta", lambda pfx: dict(metas[pfx]))
    r = client.post("/api/participants", json={"prefixes": list(metas), "threshold": 0.5})
    assert r.status_code == 200
    body = r.get_json()
    by = {p["participant_id"]: p for p in body["participants"]}
    assert set(by) == {"Alice", "Bob"}
    assert by["Alice"]["n_frames"] == 2          # Alice's frames grouped across req A + B
    assert body["n_frames"] == 3
    # every frame carries a frame_url so the browser can render the image
    assert all(f.get("frame_url") for f in by["Alice"]["frames"])


def test_route_participants_requires_prefixes(client):
    assert client.post("/api/participants", json={}).status_code == 400


def test_route_participants_skips_unreadable_meta(client, monkeypatch):
    good = "v1/d/ip/req=good/"

    def fake_dl(pfx):
        if pfx == "bad":
            raise Exception("boom")
        return _meta([_mframe(0, "pid=A__seq=1__frame_0.png", 0.5, good)])

    monkeypatch.setattr(obs, "download_meta", fake_dl)
    r = client.post("/api/participants", json={"prefixes": ["bad", good]})
    assert r.status_code == 200
    body = r.get_json()
    # the good request still yields a participant; the bad one is skipped, not fatal
    assert body["n_frames"] == 1
    assert len(body["participants"]) == 1
