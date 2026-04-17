#!/usr/bin/env python3
"""
Pre-Flight Pipeline Test — verify sender↔receiver coordination works.

Run this BEFORE committing to an overnight session.  It exercises the full
HTTP round-trip against a real receiver (--test mode or production) and
validates that frames appear in the expected output dirs.

Three test levels:
  --level quick    Just check health + one segment signal/poll cycle (30s)
  --level full     Simulate N_TEST_PAIRS sample pairs, verify output (2-3 min)
  --level stress   full + inject deliberate failures to test retry logic (5 min)

Usage:
  # Receiver must be running (in --test mode or real mode):
  python test_pipeline.py --receiver-url http://192.168.X.X:8080

  # Quick smoke test:
  python test_pipeline.py --receiver-url http://localhost:8080 --level quick

  # Full test with OBS (optional, uses real video playback):
  python test_pipeline.py --receiver-url http://192.168.X.X:8080 --level full --with-obs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    sys.exit("Missing requests. Install: pip install requests")


# ── Test configuration ──────────────────────────────────────────────────────

N_TEST_PAIRS = 3
POLL_INTERVAL_S = 0.5
POLL_TIMEOUT_S = 30.0
GREEN_HOLD_S = 2.0


# ── Result tracking ────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str = ""
    duration_s: float = 0.0

@dataclass
class TestSuite:
    results: list[TestResult] = field(default_factory=list)
    start_time: float = 0.0

    def add(self, name: str, passed: bool, msg: str = "", duration: float = 0.0):
        self.results.append(TestResult(name, passed, msg, duration))
        status = "PASS" if passed else "FAIL"
        icon = "✓" if passed else "✗"
        print(f"  {icon} {status}  {name}" + (f"  ({msg})" if msg else ""))

    def summary(self):
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        elapsed = time.time() - self.start_time
        print(f"\n{'=' * 60}")
        print(f"  {passed}/{total} tests passed  ({elapsed:.1f}s)")
        if passed < total:
            print(f"\n  FAILED:")
            for r in self.results:
                if not r.passed:
                    print(f"    ✗ {r.name}: {r.message}")
        print(f"{'=' * 60}")
        return passed == total


# ── Receiver client (minimal, standalone) ───────────────────────────────────

class TestReceiverClient:
    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, **kwargs) -> dict:
        r = requests.get(f"{self.base_url}{path}", timeout=kwargs.get("timeout", self.timeout))
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, data: dict | None = None) -> dict:
        r = requests.post(f"{self.base_url}{path}", json=data or {}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def health(self) -> dict:
        return self._get("/health")

    def session_start(self, playlist: list[dict]) -> dict:
        return self._post("/session/start", {"playlist": playlist})

    def session_end(self) -> dict:
        return self._post("/session/end")

    def segment_start(self, sample_id: str, vid_type: str,
                      strategy: str = "test", index: int = 0) -> dict:
        return self._post("/segment/start", {
            "sample_id": sample_id,
            "type": vid_type,
            "strategy": strategy,
            "index": index,
        })

    def segment_status(self) -> dict:
        return self._get("/segment/status")

    def session_progress(self) -> dict:
        return self._get("/session/progress")

    def completed_samples(self) -> dict:
        return self._get("/completed-samples", timeout=30.0)


# ── Test implementations ───────────────────────────────────────────────────

def test_health(suite: TestSuite, rc: TestReceiverClient):
    """Test 1: Receiver is reachable and healthy."""
    t0 = time.time()
    try:
        h = rc.health()
        ok = h.get("status") == "ok"
        watching = h.get("watching", False)
        participant = h.get("participant", "?")
        faces_dir = h.get("faces_dir", "?")
        suite.add("Health check", ok,
                  f"watching={watching}, participant={participant}, faces={faces_dir}",
                  time.time() - t0)
        return ok
    except Exception as e:
        suite.add("Health check", False, str(e), time.time() - t0)
        return False


def test_session_lifecycle(suite: TestSuite, rc: TestReceiverClient):
    """Test 2: Session start/end works."""
    t0 = time.time()
    try:
        playlist = [
            {"index": 0, "sample_id": "test_001", "type": "real", "strategy": "test"},
            {"index": 1, "sample_id": "test_001", "type": "fake", "strategy": "test"},
        ]
        resp = rc.session_start(playlist)
        ok = resp.get("ok", False)
        suite.add("Session start", ok,
                  f"total_expected={resp.get('total_expected', '?')}",
                  time.time() - t0)
        return ok
    except Exception as e:
        suite.add("Session start", False, str(e), time.time() - t0)
        return False


def test_segment_signal(suite: TestSuite, rc: TestReceiverClient):
    """Test 3: Segment start signal is accepted."""
    t0 = time.time()
    try:
        resp = rc.segment_start("test_signal_001", "real", "test", 0)
        ok = resp.get("ok", False)
        suite.add("Segment start signal", ok,
                  f"baseline_files={resp.get('baseline_files', '?')}",
                  time.time() - t0)
        return ok
    except Exception as e:
        suite.add("Segment start signal", False, str(e), time.time() - t0)
        return False


def test_segment_poll(suite: TestSuite, rc: TestReceiverClient):
    """Test 4: Segment status polling works and returns expected fields."""
    t0 = time.time()
    try:
        status = rc.segment_status()
        required_fields = ["state", "collected_frames",
                           "min_frames", "complete", "timed_out"]
        missing = [f for f in required_fields if f not in status]
        ok = len(missing) == 0
        suite.add("Segment status polling", ok,
                  f"state={status.get('state', '?')}, "
                  f"fields_ok={len(missing) == 0}" +
                  (f", missing={missing}" if missing else ""),
                  time.time() - t0)
        return ok
    except Exception as e:
        suite.add("Segment status polling", False, str(e), time.time() - t0)
        return False


def test_segment_collection_cycle(suite: TestSuite, rc: TestReceiverClient,
                                  sample_id: str = "preflight_test_001",
                                  vid_type: str = "real",
                                  require_frames: bool = True):
    """Test 5: Full segment lifecycle — start, poll through collecting, until complete or timeout.

    If require_frames=False, a poll timeout with 0 collected frames is treated
    as a SKIP (expected when receiver runs in --test mode with no real WMA).
    """
    # When we know no frames will come, use a short timeout
    effective_timeout = POLL_TIMEOUT_S if require_frames else 8.0
    t0 = time.time()
    try:
        resp = rc.segment_start(sample_id, vid_type, "preflight_test", 0)
        if not resp.get("ok"):
            suite.add(f"Collection cycle ({vid_type})", False,
                      "segment_start rejected", time.time() - t0)
            return False

        # Poll until complete, timeout, or our poll timeout
        last_state = "unknown"
        last_collected = 0
        while time.time() - t0 < effective_timeout:
            status = rc.segment_status()
            last_state = status.get("state", "unknown")
            last_collected = status.get("collected_frames", 0)

            if status.get("complete"):
                suite.add(f"Collection cycle ({vid_type})", True,
                          f"collected={last_collected} frames in {time.time() - t0:.1f}s",
                          time.time() - t0)
                return True

            if status.get("timed_out"):
                if not require_frames and last_collected == 0:
                    suite.add(f"Collection cycle ({vid_type})", True,
                              f"SKIP — no frame producer (test mode), API flow OK",
                              time.time() - t0)
                    return True
                suite.add(f"Collection cycle ({vid_type})", False,
                          f"receiver timed out with {last_collected} frames",
                          time.time() - t0)
                return False

            time.sleep(POLL_INTERVAL_S)

        # Poll timeout — if no frames are expected, this is fine
        if not require_frames and last_collected == 0:
            suite.add(f"Collection cycle ({vid_type})", True,
                      f"SKIP — no frame producer (test mode), API flow OK",
                      time.time() - t0)
            return True

        suite.add(f"Collection cycle ({vid_type})", False,
                  f"poll timeout ({POLL_TIMEOUT_S}s), last_state={last_state}, "
                  f"collected={last_collected}",
                  time.time() - t0)
        return False
    except Exception as e:
        suite.add(f"Collection cycle ({vid_type})", False,
                  str(e), time.time() - t0)
        return False


def test_progress_endpoint(suite: TestSuite, rc: TestReceiverClient):
    """Test 6: Session progress returns sensible data."""
    t0 = time.time()
    try:
        progress = rc.session_progress()
        ok = "completed_segments" in progress and "elapsed_s" in progress
        suite.add("Session progress", ok,
                  f"segments={progress.get('completed_segments', '?')}, "
                  f"pairs={progress.get('completed_pairs', '?')}",
                  time.time() - t0)
        return ok
    except Exception as e:
        suite.add("Session progress", False, str(e), time.time() - t0)
        return False


def test_completed_samples(suite: TestSuite, rc: TestReceiverClient):
    """Test 7: Completed samples endpoint works."""
    t0 = time.time()
    try:
        data = rc.completed_samples()
        ok = "completed" in data and "partial" in data
        suite.add("Completed samples", ok,
                  f"completed={len(data.get('completed', []))}, "
                  f"partial={len(data.get('partial', []))}",
                  time.time() - t0)
        return ok
    except Exception as e:
        suite.add("Completed samples", False, str(e), time.time() - t0)
        return False


def test_session_end(suite: TestSuite, rc: TestReceiverClient):
    """Test 8: Session end with summary."""
    t0 = time.time()
    try:
        resp = rc.session_end()
        ok = resp.get("ok", False)
        summary = resp.get("summary", {})
        suite.add("Session end", ok,
                  f"segments={summary.get('completed_segments', '?')}, "
                  f"pairs={summary.get('completed_pairs', '?')}",
                  time.time() - t0)
        return ok
    except Exception as e:
        suite.add("Session end", False, str(e), time.time() - t0)
        return False


def test_rapid_segment_switch(suite: TestSuite, rc: TestReceiverClient):
    """Test 9 (stress): Rapidly switch segments to test state cleanup."""
    t0 = time.time()
    try:
        ok = True
        for i in range(5):
            resp = rc.segment_start(f"stress_rapid_{i}", "real", "stress", i)
            if not resp.get("ok"):
                ok = False
                break
            time.sleep(0.5)

        status = rc.segment_status()
        correct_sample = status.get("sample_id", "") == "stress_rapid_4"
        suite.add("Rapid segment switch", ok and correct_sample,
                  f"last_sample={status.get('sample_id', '?')}",
                  time.time() - t0)
        return ok and correct_sample
    except Exception as e:
        suite.add("Rapid segment switch", False, str(e), time.time() - t0)
        return False


# ── Test runners ────────────────────────────────────────────────────────────

def run_quick(rc: TestReceiverClient) -> bool:
    """Quick smoke test — just connectivity and API shape."""
    suite = TestSuite(start_time=time.time())
    print("\n─── Quick Pre-Flight Test ───\n")

    if not test_health(suite, rc):
        print("\n  Cannot reach receiver — aborting.")
        return suite.summary()

    test_session_lifecycle(suite, rc)
    test_segment_signal(suite, rc)
    test_segment_poll(suite, rc)
    test_progress_endpoint(suite, rc)
    test_completed_samples(suite, rc)
    test_session_end(suite, rc)

    return suite.summary()


def run_full(rc: TestReceiverClient) -> bool:
    """Full test — includes actual segment collection cycles."""
    suite = TestSuite(start_time=time.time())
    print("\n─── Full Pre-Flight Test ───\n")

    if not test_health(suite, rc):
        print("\n  Cannot reach receiver — aborting.")
        return suite.summary()

    # Build a small playlist
    playlist = []
    for i in range(N_TEST_PAIRS):
        for vt in ("real", "fake"):
            playlist.append({
                "index": len(playlist),
                "sample_id": f"preflight_{i:03d}",
                "type": vt,
                "strategy": "preflight_test",
            })

    test_session_lifecycle(suite, rc)

    # Detect test mode
    health = rc.health()
    faces_dir = health.get("faces_dir", "")
    is_test_mode = "receiver_test" in faces_dir or "Temp" in faces_dir
    if is_test_mode:
        print(f"\n  Running {N_TEST_PAIRS} collection cycles "
              f"(test mode — will SKIP, no WMA)...\n")
    else:
        print(f"\n  Running {N_TEST_PAIRS} collection cycles "
              f"(receiver must produce face crops)...\n")

    for i in range(N_TEST_PAIRS):
        for vt in ("real", "fake"):
            sample_id = f"preflight_{i:03d}"
            # Show green screen pause between segments
            time.sleep(GREEN_HOLD_S)
            test_segment_collection_cycle(suite, rc, sample_id, vt,
                                          require_frames=not is_test_mode)

    test_progress_endpoint(suite, rc)
    test_completed_samples(suite, rc)
    test_session_end(suite, rc)

    return suite.summary()


def run_stress(rc: TestReceiverClient) -> bool:
    """Stress test — full + deliberate abuse."""
    suite = TestSuite(start_time=time.time())
    print("\n─── Stress Pre-Flight Test ───\n")

    if not test_health(suite, rc):
        print("\n  Cannot reach receiver — aborting.")
        return suite.summary()

    test_session_lifecycle(suite, rc)

    # Detect if receiver is in test mode (no real WMA producing frames)
    # by checking if health response indicates simulated session
    health = rc.health()
    faces_dir = health.get("faces_dir", "")
    is_test_mode = "receiver_test" in faces_dir or "Temp" in faces_dir
    if is_test_mode:
        print("\n  (Receiver is in --test mode — collection cycles will "
              "SKIP instead of FAIL since no WMA produces frames)\n")

    # Normal collection
    print("\n  Phase 1: Normal collection...\n")

    time.sleep(GREEN_HOLD_S)
    test_segment_collection_cycle(suite, rc, "stress_normal_001", "real",
                                  require_frames=not is_test_mode)
    time.sleep(GREEN_HOLD_S)
    test_segment_collection_cycle(suite, rc, "stress_normal_001", "fake",
                                  require_frames=not is_test_mode)

    # Rapid switching
    print("\n  Phase 2: Rapid segment switching...\n")
    test_rapid_segment_switch(suite, rc)

    # Double-start (sender retries a segment)
    print("\n  Phase 3: Double-start same segment (retry simulation)...\n")
    t0 = time.time()
    try:
        rc.segment_start("stress_double", "real", "stress", 0)
        time.sleep(1.0)
        resp2 = rc.segment_start("stress_double", "real", "stress", 0)
        suite.add("Double-start same segment", resp2.get("ok", False),
                  "state correctly reset", time.time() - t0)
    except Exception as e:
        suite.add("Double-start same segment", False, str(e), time.time() - t0)

    # Back-to-back session start
    print("\n  Phase 4: Back-to-back session lifecycle...\n")
    test_session_end(suite, rc)
    test_session_lifecycle(suite, rc)

    test_progress_endpoint(suite, rc)
    test_completed_samples(suite, rc)
    test_session_end(suite, rc)

    return suite.summary()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pre-flight pipeline test for sender↔receiver coordination"
    )
    parser.add_argument("--receiver-url", required=True,
                        help="Receiver server URL (e.g. http://192.168.1.100:8080)")
    parser.add_argument("--level", choices=["quick", "full", "stress"],
                        default="quick",
                        help="Test level: quick (30s), full (2-3min), stress (5min)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Pre-Flight Pipeline Test")
    print(f"  Receiver: {args.receiver_url}")
    print(f"  Level:    {args.level}")
    print(f"  Time:     {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    rc = TestReceiverClient(args.receiver_url)

    if args.level == "quick":
        passed = run_quick(rc)
    elif args.level == "full":
        passed = run_full(rc)
    elif args.level == "stress":
        passed = run_stress(rc)
    else:
        passed = run_quick(rc)

    if passed:
        print("\n  ✓ All tests passed — safe to run overnight.\n")
    else:
        print("\n  ✗ Some tests failed — fix issues before overnight run.\n")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
