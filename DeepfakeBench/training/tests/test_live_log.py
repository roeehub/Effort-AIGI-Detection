"""TDD tests for `live_log` — app3's live console logging.

`live_log.py` is import-light (stdlib only) so the formatting/aggregation logic
can be unit-tested WITHOUT importing the model / torch / GCS / observability
stack or spinning up the FastAPI server. See
`docs/superpowers/specs/2026-05-31-app3-live-logging-design.md`.

Visible-text assertions strip ANSI; color is asserted by SGR-code membership.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# live_log.py lives in the training/ dir (parent of tests/).
_TRAIN = Path(__file__).parent.parent
if str(_TRAIN) not in sys.path:
    sys.path.insert(0, str(_TRAIN))

import live_log  # noqa: E402


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def strip(s: str) -> str:
    return _ANSI_RE.sub("", s)


def fake_parser(filename):
    """Test pid parser: 'Name#seq' -> (Name, seq); anything else -> (None, None).

    Stands in for observability.parse_pid_from_filename so the tests don't pull
    in the observability module.
    """
    if not filename or "#" not in filename:
        return None, None
    name, _, seq = filename.partition("#")
    try:
        return name, int(seq)
    except ValueError:
        return None, None


# --------------------------------------------------------------------------- #
# Badges                                                                       #
# --------------------------------------------------------------------------- #
class TestBadgeRegistry:
    def test_letters_assigned_in_first_seen_order(self):
        reg = live_log.BadgeRegistry()
        assert reg.badge("1.1.1.1").letter == "A"
        assert reg.badge("2.2.2.2").letter == "B"
        # Re-querying the first IP keeps its original letter.
        assert reg.badge("1.1.1.1").letter == "A"
        assert reg.badge("3.3.3.3").letter == "C"

    def test_color_never_red_or_green(self):
        reg = live_log.BadgeRegistry()
        for i in range(40):
            color = reg.badge(f"10.0.0.{i}").color
            assert color in live_log.PALETTE
            assert color != live_log.RED
            assert color != live_log.GREEN

    def test_color_is_deterministic_across_instances(self):
        # md5-based, not salted hash() — so two processes agree on an IP's color.
        a = live_log.BadgeRegistry().badge("203.0.113.7").color
        b = live_log.BadgeRegistry().badge("203.0.113.7").color
        assert a == b

    def test_two_ips_get_distinct_letters_even_if_color_collides(self):
        reg = live_log.BadgeRegistry()
        b1 = reg.badge("9.9.9.9")
        b2 = reg.badge("8.8.8.8")
        assert b1.letter != b2.letter

    def test_tag_renders_dot_and_letter(self):
        reg = live_log.BadgeRegistry()
        assert strip(reg.badge("1.2.3.4").tag()) == "●A"


# --------------------------------------------------------------------------- #
# Grouping                                                                     #
# --------------------------------------------------------------------------- #
class TestGroupByParticipant:
    def _grouped(self):
        filenames = ["Guest#0", "Guest#1", "dor#0", "anon0"]
        status = [
            {"kind": "tensor", "prob": 0.9},
            {"kind": "gated", "prob": -1.0, "reason": "no_face_detected"},
            {"kind": "tensor", "prob": 0.1},
            {"kind": "failed"},
        ]
        return live_log.group_by_participant(filenames, status, fake_parser, threshold=0.5)

    def test_splits_by_participant_with_unknown_bucket(self):
        parts = self._grouped()
        assert [p.name for p in parts] == ["Guest", "dor", "unknown"]

    def test_most_suspicious_first(self):
        parts = self._grouped()
        # Guest mean 0.9 > dor mean 0.1 > unknown (no score -> last).
        assert parts[0].name == "Guest"
        assert parts[-1].name == "unknown"

    def test_mean_uses_voting_frames_only(self):
        guest = self._grouped()[0]
        # sentinel (-1.0) gated frame is excluded from the mean.
        assert abs(guest.mean - 0.9) < 1e-9
        assert guest.verdict == "FAKE"
        assert guest.n == 2
        assert guest.n_voting == 1
        assert guest.n_gated == 1
        assert guest.n_no_face == 1

    def test_unknown_participant_has_no_mean(self):
        unknown = self._grouped()[-1]
        assert unknown.mean is None
        assert unknown.verdict == "—"
        assert unknown.n_failed == 1

    def test_real_verdict_below_threshold(self):
        dor = self._grouped()[1]
        assert dor.verdict == "REAL"
        assert abs(dor.mean - 0.1) < 1e-9


# --------------------------------------------------------------------------- #
# Frame cells + block                                                          #
# --------------------------------------------------------------------------- #
class TestFrameCellAndBlock:
    def _one(self, status, filename="x#0", threshold=0.5):
        parts = live_log.group_by_participant([filename], [status], fake_parser, threshold)
        return parts[0].frames[0]

    def test_scored_fake_cell_is_red_and_shows_prob(self):
        cfg = live_log.LogConfig()
        cell = live_log.format_frame_cell(self._one({"kind": "tensor", "prob": 0.931}), 0.5, cfg)
        assert live_log.RED in cell
        assert "931" in strip(cell)
        assert "█" in strip(cell)

    def test_scored_real_cell_is_green(self):
        cfg = live_log.LogConfig()
        cell = live_log.format_frame_cell(self._one({"kind": "tensor", "prob": 0.12}), 0.5, cfg)
        assert live_log.GREEN in cell

    def test_gated_cell_is_dim_with_reason_tag(self):
        cfg = live_log.LogConfig()
        frame = self._one({"kind": "gated", "prob": -1.0, "reason": "no_face_detected"})
        cell = live_log.format_frame_cell(frame, 0.5, cfg)
        assert live_log.DIM in cell
        assert "no-face" in strip(cell)
        assert "·" in strip(cell)

    def test_block_header_carries_identity_and_meta(self):
        cfg = live_log.LogConfig()
        reg = live_log.BadgeRegistry()
        badge = reg.badge("93.19.226.214")
        parts = live_log.group_by_participant(
            ["Guest#0", "Guest#1"],
            [{"kind": "tensor", "prob": 0.9}, {"kind": "tensor", "prob": 0.8}],
            fake_parser, 0.5,
        )
        block = strip(live_log.format_batch_block(badge, "93.19.226.214", parts, 0.5, 18.0, cfg, clock="14:32:07"))
        assert "93.19.226.214" in block
        assert "●A" in block
        assert "BATCH" in block
        assert "18ms" in block
        assert "14:32:07" in block
        assert "Guest" in block
        assert "FAKE" in block

    def test_block_caps_frames_and_summarizes_overflow(self):
        cfg = live_log.LogConfig(frame_cap=2)
        reg = live_log.BadgeRegistry()
        badge = reg.badge("1.2.3.4")
        names = [f"dor#{i}" for i in range(5)]
        status = [{"kind": "tensor", "prob": 0.2} for _ in range(5)]
        parts = live_log.group_by_participant(names, status, fake_parser, 0.5)
        block = strip(live_log.format_batch_block(badge, "1.2.3.4", parts, 0.5, None, cfg, clock="00:00:00"))
        assert "+3 more" in block


# --------------------------------------------------------------------------- #
# Anomalies                                                                    #
# --------------------------------------------------------------------------- #
class TestDetectAnomalies:
    def _parts(self, filenames, status, threshold=0.5):
        return live_log.group_by_participant(filenames, status, fake_parser, threshold)

    def test_no_face_majority_flagged(self):
        cfg = live_log.LogConfig(no_face_frac=0.5)
        parts = self._parts(
            ["Guest#0", "Guest#1", "Guest#2", "Guest#3"],
            [
                {"kind": "tensor", "prob": 0.9},
                {"kind": "gated", "prob": -1.0, "reason": "no_face_detected"},
                {"kind": "gated", "prob": -1.0, "reason": "no_face_detected"},
                {"kind": "gated", "prob": -1.0, "reason": "no_face_detected"},
            ],
        )
        lines = [strip(x) for x in live_log.detect_anomalies(None, "1.2.3.4", parts, None, cfg)]
        assert any("NO-FACE" in x and "3/4" in x and "Guest" in x for x in lines)

    def test_no_face_below_threshold_not_flagged(self):
        cfg = live_log.LogConfig(no_face_frac=0.5)
        parts = self._parts(
            ["Guest#0", "Guest#1"],
            [
                {"kind": "tensor", "prob": 0.9},
                {"kind": "gated", "prob": -1.0, "reason": "no_face_detected"},
            ],
        )
        lines = [strip(x) for x in live_log.detect_anomalies(None, "1.2.3.4", parts, None, cfg)]
        assert not any("NO-FACE" in x for x in lines)

    def test_processing_failure_flagged(self):
        cfg = live_log.LogConfig()
        parts = self._parts(["dor#0", "dor#1"], [{"kind": "tensor", "prob": 0.1}, {"kind": "failed"}])
        lines = [strip(x) for x in live_log.detect_anomalies(None, "1.2.3.4", parts, None, cfg)]
        assert any("FAILED" in x and "dor" in x for x in lines)

    def test_nothing_scored_flagged(self):
        cfg = live_log.LogConfig()
        parts = self._parts(
            ["Guest#0", "Guest#1"],
            [{"kind": "gated", "prob": -1.0, "reason": "min_dim"}, {"kind": "failed"}],
        )
        lines = [strip(x) for x in live_log.detect_anomalies(None, "84.12.7.9", parts, None, cfg)]
        assert any("nothing scored" in x and "84.12.7.9" in x for x in lines)

    def test_slow_latency_flagged(self):
        cfg = live_log.LogConfig(slow_ms=1000.0)
        parts = self._parts(["dor#0"], [{"kind": "tensor", "prob": 0.1}])
        lines = [strip(x) for x in live_log.detect_anomalies(None, "1.2.3.4", parts, 1500.0, cfg)]
        assert any("slow" in x and "1500ms" in x for x in lines)

    def test_clean_batch_no_anomalies(self):
        cfg = live_log.LogConfig()
        parts = self._parts(["dor#0", "dor#1"], [{"kind": "tensor", "prob": 0.1}, {"kind": "tensor", "prob": 0.2}])
        assert live_log.detect_anomalies(None, "1.2.3.4", parts, 20.0, cfg) == []


# --------------------------------------------------------------------------- #
# Rolling dashboard                                                            #
# --------------------------------------------------------------------------- #
class TestActivityRegistry:
    def test_should_print_toggles_around_render(self):
        reg = live_log.BadgeRegistry()
        act = live_log.ActivityRegistry()
        assert act.should_print() is False
        act.record("1.2.3.4", ["Guest"], n_frames=6, n_fake=2, n_gated=1, now=1000.0)
        assert act.should_print() is True
        act.render(reg, now=1001.0, window_seconds=300.0)
        assert act.should_print() is False

    def test_render_single_user_counts_and_percentages(self):
        reg = live_log.BadgeRegistry()
        act = live_log.ActivityRegistry()
        act.record("1.2.3.4", ["Guest", "dor"], n_frames=6, n_fake=2, n_gated=1, now=1000.0)
        out = strip(act.render(reg, now=1001.0, window_seconds=300.0))
        assert "1 user ·" in out
        assert "1.2.3.4" in out
        assert "6f" in out
        assert "33%fake" in out
        assert "17%gated" in out
        assert "seen 1s" in out
        assert "Guest" in out and "dor" in out

    def test_render_counts_multiple_users(self):
        reg = live_log.BadgeRegistry()
        act = live_log.ActivityRegistry()
        act.record("1.1.1.1", ["a"], 3, 0, 0, now=1000.0)
        act.record("2.2.2.2", ["b"], 3, 3, 0, now=1000.5)
        out = strip(act.render(reg, now=1001.0, window_seconds=300.0))
        assert "2 users" in out

    def test_stale_ip_pruned_outside_window(self):
        reg = live_log.BadgeRegistry()
        act = live_log.ActivityRegistry()
        act.record("1.2.3.4", ["Guest"], 5, 0, 0, now=0.0)
        # 400s later, with a 300s window, nothing is active.
        assert act.render(reg, now=400.0, window_seconds=300.0) is None
        assert act.should_print() is False

    def test_window_aggregates_only_recent_events(self):
        reg = live_log.BadgeRegistry()
        act = live_log.ActivityRegistry()
        act.record("1.2.3.4", ["Guest"], 10, 10, 0, now=0.0)      # stale
        act.record("1.2.3.4", ["Guest"], 4, 0, 0, now=350.0)      # recent
        out = strip(act.render(reg, now=351.0, window_seconds=300.0))
        # Only the recent 4 frames (0 fake) should count -> 0%fake, 4f.
        assert "4f" in out
        assert "0%fake" in out


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
class TestLogConfig:
    def test_defaults(self, monkeypatch):
        for var in (
            "OBS_LOG_FRAME_CAP", "OBS_LOG_FRAMES_PER_LINE", "OBS_LOG_BAR_WIDTH",
            "OBS_LOG_NOFACE_FRAC", "OBS_LOG_SLOW_MS", "OBS_LOG_DASHBOARD_SECONDS",
            "OBS_LOG_WINDOW_SECONDS",
        ):
            monkeypatch.delenv(var, raising=False)
        cfg = live_log.LogConfig.from_env()
        assert cfg.frame_cap == 8
        assert cfg.frames_per_line == 3
        assert cfg.bar_width == 12
        assert cfg.no_face_frac == 0.5
        assert cfg.slow_ms == 1000.0
        assert cfg.dashboard_seconds == 15.0
        assert cfg.window_seconds == 300.0

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("OBS_LOG_FRAME_CAP", "4")
        monkeypatch.setenv("OBS_LOG_NOFACE_FRAC", "0.8")
        monkeypatch.setenv("OBS_LOG_SLOW_MS", "500")
        cfg = live_log.LogConfig.from_env()
        assert cfg.frame_cap == 4
        assert cfg.no_face_frac == 0.8
        assert cfg.slow_ms == 500.0


# --------------------------------------------------------------------------- #
# LiveLog orchestration                                                        #
# --------------------------------------------------------------------------- #
class TestLiveLog:
    def _live(self, captured):
        return live_log.LiveLog(
            log_fn=captured.append,
            cfg=live_log.LogConfig(),
            clock_fn=lambda: "00:00:00",
            pid_parser=fake_parser,
        )

    def test_log_batch_emits_block_and_records_activity(self):
        captured = []
        live = self._live(captured)
        live.log_batch(
            ["Guest#0", "dor#0"],
            [{"kind": "tensor", "prob": 0.9}, {"kind": "tensor", "prob": 0.1}],
            threshold=0.5, ip="1.2.3.4", latency_ms=20.0,
        )
        assert len(captured) == 1
        block = strip(captured[0])
        assert "Guest" in block and "dor" in block and "1.2.3.4" in block
        assert live.activity.should_print() is True

    def test_tick_prints_dashboard_when_active_then_silent(self):
        captured = []
        live = self._live(captured)
        live.log_batch(
            ["Guest#0"], [{"kind": "tensor", "prob": 0.9}],
            threshold=0.5, ip="1.2.3.4", latency_ms=20.0,
        )
        assert len(captured) == 1          # the batch block
        live._tick(now=1.0)
        assert len(captured) == 2          # + dashboard
        assert "live" in strip(captured[1])
        live._tick(now=2.0)
        assert len(captured) == 2          # nothing new since last render

    def test_tick_silent_when_no_traffic(self):
        captured = []
        live = self._live(captured)
        live._tick(now=1.0)
        assert captured == []

    def test_start_stop_does_not_raise(self):
        live = self._live([])
        live.start()
        live.stop()
