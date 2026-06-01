"""Live console logging for app3 — IP-badged, per-participant batch readouts
plus a rolling activity dashboard.

Design: docs/superpowers/specs/2026-05-31-app3-live-logging-design.md

This module is intentionally **import-light** (stdlib only). It must NOT import
torch / cv2 / google-cloud-storage / observability, so the formatting and
aggregation logic stays unit-testable in milliseconds and can never break model
serving by pulling a heavy/optional dependency.

Two axes of identity, kept separate on purpose:
  * IP        — which remote/connection. Colored ●badge on the block header and
                the dashboard rows. The same value the viewer sessionizes on.
  * participant — which person inside one call (pid parsed from the filename).
                Names group the per-frame lines; one IP/request can carry many.

Color policy: RED/GREEN are reserved for the verdict, so the IP badge palette
deliberately excludes them — an IP's color never competes with FAKE/REAL.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

# --- ANSI ------------------------------------------------------------------ #
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Badge palette — every entry must avoid RED (91) and GREEN (92) so the IP
# color can never be confused with a verdict.
PALETTE = [
    "\033[96m",  # bright cyan
    "\033[95m",  # bright magenta
    "\033[94m",  # bright blue
    "\033[93m",  # bright yellow
    "\033[36m",  # cyan
    "\033[35m",  # magenta
    "\033[34m",  # blue
    "\033[33m",  # yellow
]

def _stable_hash(s: str) -> int:
    """Deterministic across processes (unlike salted builtin hash())."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def _fmt_prob(p: float) -> str:
    """0.928 -> '.928', 1.000 -> '1.000' (drop the leading zero for density)."""
    return f"{p:.3f}".lstrip("0") or "0"


def _fmt_thr(t: float) -> str:
    return f"{t:.2f}".lstrip("0") or "0"


def _bar(prob: float, width: int) -> str:
    filled = max(0, min(width, int(round(prob * width))))
    return "█" * filled + "░" * (width - filled)


# --- Config ---------------------------------------------------------------- #
@dataclass
class LogConfig:
    frame_cap: int = 8
    bar_width: int = 12
    box_width: int = 72                # width of the ═ frame drawn around each batch
    no_face_frac: float = 0.5          # flag when MORE THAN this fraction is no-face
    slow_ms: float = 1000.0
    dashboard_seconds: float = 15.0
    window_seconds: float = 300.0

    @classmethod
    def from_env(cls) -> "LogConfig":
        def _i(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, "").strip())
            except (TypeError, ValueError):
                return default

        def _f(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, "").strip())
            except (TypeError, ValueError):
                return default

        return cls(
            frame_cap=_i("OBS_LOG_FRAME_CAP", 8),
            bar_width=_i("OBS_LOG_BAR_WIDTH", 12),
            box_width=_i("OBS_LOG_BOX_WIDTH", 72),
            no_face_frac=_f("OBS_LOG_NOFACE_FRAC", 0.5),
            slow_ms=_f("OBS_LOG_SLOW_MS", 1000.0),
            dashboard_seconds=_f("OBS_LOG_DASHBOARD_SECONDS", 15.0),
            window_seconds=_f("OBS_LOG_WINDOW_SECONDS", 300.0),
        )


# --- Badges ---------------------------------------------------------------- #
@dataclass(frozen=True)
class Badge:
    letter: str
    color: str

    def tag(self) -> str:
        return f"{self.color}●{self.letter}{RESET}"

    def plain(self) -> str:
        return f"●{self.letter}"


class BadgeRegistry:
    """Assigns each IP a first-seen letter (A, B, C…) and a stable hashed color."""

    def __init__(self, palette: Optional[List[str]] = None):
        self._palette = palette or PALETTE
        self._by_ip: dict = {}
        self._next = 0
        self._lock = threading.Lock()

    @staticmethod
    def _letter_for(idx: int) -> str:
        if idx < 26:
            return chr(ord("A") + idx)
        if idx < 52:
            return chr(ord("a") + idx - 26)
        return "*"

    def badge(self, ip: str) -> Badge:
        with self._lock:
            b = self._by_ip.get(ip)
            if b is None:
                letter = self._letter_for(self._next)
                self._next += 1
                color = self._palette[_stable_hash(ip) % len(self._palette)]
                b = Badge(letter, color)
                self._by_ip[ip] = b
            return b


# --- Grouping -------------------------------------------------------------- #
@dataclass
class FrameView:
    kind: Optional[str]
    prob: Optional[float]
    reason: Optional[str]
    is_voting: bool


@dataclass
class Participant:
    name: str
    frames: List[FrameView]
    mean: Optional[float]
    verdict: str
    n: int
    n_voting: int
    n_gated: int
    n_failed: int
    n_no_face: int


def group_by_participant(filenames, per_frame_status, pid_parser, threshold) -> List[Participant]:
    """Group a batch's frames by participant (pid), most-suspicious-first.

    `pid_parser(filename) -> (pid, seq)`; a None pid lands in the `unknown`
    bucket (legacy / unlabeled frames). "Voting" frames are the model-scored
    ones (kind == "tensor"); gated sentinels and failures are excluded from the
    participant mean — matching the server's confidence calc.
    """
    buckets: dict = {}
    order: List[str] = []
    for fn, st in zip(filenames, per_frame_status):
        pid = pid_parser(fn)[0] if pid_parser else None
        name = pid or "unknown"
        prob = st.get("prob")
        kind = st.get("kind")
        reason = st.get("reason")
        is_voting = kind == "tensor" and prob is not None and prob >= 0.0
        if name not in buckets:
            buckets[name] = []
            order.append(name)
        buckets[name].append(FrameView(kind=kind, prob=prob, reason=reason, is_voting=is_voting))

    parts: List[Participant] = []
    for name in order:
        frames = buckets[name]
        voting = [f.prob for f in frames if f.is_voting]
        mean = (sum(voting) / len(voting)) if voting else None
        verdict = "—" if mean is None else ("FAKE" if mean >= threshold else "REAL")
        parts.append(Participant(
            name=name,
            frames=frames,
            mean=mean,
            verdict=verdict,
            n=len(frames),
            n_voting=len(voting),
            n_gated=sum(1 for f in frames if f.kind == "gated"),
            n_failed=sum(1 for f in frames if f.kind == "failed"),
            n_no_face=sum(1 for f in frames if (f.reason or "") == "no_face_detected"),
        ))
    # most-suspicious first: highest mean, None last, then name.
    parts.sort(key=lambda p: (p.mean is None, -(p.mean or 0.0), p.name))
    return parts


# --- Formatting ------------------------------------------------------------ #
def format_frame_cell(frame: FrameView, threshold: float, cfg: LogConfig) -> str:
    """One frame as its own line: scored → coloured bar + prob + verdict word; a
    rejected frame → dim dots + the FULL reason (e.g. 'min_dim=89<120'), kept
    whole so you can see exactly why each frame was rejected."""
    w = cfg.bar_width
    if frame.is_voting and frame.prob is not None:
        color = RED if frame.prob >= threshold else GREEN
        verdict = "FAKE" if frame.prob >= threshold else "REAL"
        return f"{color}{_bar(frame.prob, w)} {_fmt_prob(frame.prob)} {verdict}{RESET}"
    reason = frame.reason or ("failed" if frame.kind == "failed" else "gated")
    return f"{DIM}{'·' * w} {reason}{RESET}"


def format_batch_block(badge: Badge, ip, participants, threshold, latency_ms, cfg, clock="") -> str:
    total = sum(p.n for p in participants)
    scored = sum(p.n_voting for p in participants)
    gated = sum(p.n_gated for p in participants)
    failed = sum(p.n_failed for p in participants)
    lat = f"  {int(round(latency_ms))}ms" if latency_ms is not None else ""
    counts = f"  ·  {scored} scored"
    if gated:
        counts += f" / {gated} gated"
    if failed:
        counts += f" / {failed} failed"
    head = f" {clock}  {badge.tag()}  {ip}  BATCH·{total}  thr={_fmt_thr(threshold)}{lat}{counts}"
    lines = [head, f"{DIM}{'─' * cfg.box_width}{RESET}"]
    for p in participants:
        if p.mean is None:
            lines.append(f" {p.name}  — none scored ({p.n}f)")
        else:
            vcolor = RED if p.verdict == "FAKE" else GREEN
            lines.append(f" {p.name}  mean {_fmt_prob(p.mean)} {vcolor}{p.verdict}{RESET} ({p.n}f)")
        shown = p.frames[: cfg.frame_cap]
        for fr in shown:                       # one line per frame (incl. rejects)
            lines.append("   " + format_frame_cell(fr, threshold, cfg))
        if p.n > len(shown):
            lines.append(f"   +{p.n - len(shown)} more")
    return "\n".join(lines)


def detect_anomalies(badge: Optional[Badge], ip, participants, latency_ms, cfg) -> List[str]:
    out: List[str] = []
    for p in participants:
        if p.n > 0 and (p.n_no_face / p.n) > cfg.no_face_frac:
            out.append(f"{BOLD}{YELLOW} ⚠ {p.name} {p.n_no_face}/{p.n} NO-FACE — check crop{RESET}")
        if p.n_failed > 0:
            out.append(f"{BOLD}{RED} ⚠ {p.name} {p.n_failed} frame(s) FAILED{RESET}")
    if participants and sum(p.n_voting for p in participants) == 0:
        btag = badge.tag() if badge is not None else "●?"
        out.append(f"{BOLD}{YELLOW} ⚠ {btag} {ip} BATCH all-gated/failed — nothing scored{RESET}")
    if latency_ms is not None and latency_ms > cfg.slow_ms:
        out.append(f"{BOLD}{YELLOW} ⚠ slow {int(round(latency_ms))}ms{RESET}")
    return out


def render_batch(badge: Badge, ip, participants, threshold, latency_ms, cfg, clock="") -> str:
    """The full framed batch: a ═ rule, the block (+ any anomaly lines), a ═ rule.
    The frame brackets each batch so consecutive ones don't blur together."""
    bar = f"{BOLD}{'═' * cfg.box_width}{RESET}"
    block = format_batch_block(badge, ip, participants, threshold, latency_ms, cfg, clock)
    anomalies = detect_anomalies(badge, ip, participants, latency_ms, cfg)
    lines = [bar, block]
    if anomalies:
        lines.append("\n".join(anomalies))
    lines.append(bar)
    return "\n".join(lines)


# --- Rolling dashboard ----------------------------------------------------- #
class ActivityRegistry:
    """Per-IP rolling counters for the periodic dashboard.

    Stores one event per recorded batch; `render` aggregates events inside the
    window, prunes anything older, and reports nothing when idle.
    """

    def __init__(self):
        self._events: dict = {}   # ip -> list[(t, n_frames, n_fake, n_gated, names)]
        self._dirty = False
        self._lock = threading.Lock()

    def record(self, ip, participant_names, n_frames, n_fake, n_gated, now=None):
        t = time.time() if now is None else now
        names = tuple(n for n in (participant_names or []) if n)
        with self._lock:
            self._events.setdefault(ip, []).append((t, n_frames, n_fake, n_gated, names))
            self._dirty = True

    def should_print(self) -> bool:
        return self._dirty

    def render(self, badges: BadgeRegistry, now=None, window_seconds=300.0) -> Optional[str]:
        t = time.time() if now is None else now
        with self._lock:
            self._dirty = False
            rows = []
            for ip in list(self._events.keys()):
                recent = [e for e in self._events[ip] if t - e[0] <= window_seconds]
                if not recent:
                    del self._events[ip]
                    continue
                self._events[ip] = recent
                frames = sum(e[1] for e in recent)
                fake = sum(e[2] for e in recent)
                gated = sum(e[3] for e in recent)
                last_seen = max(e[0] for e in recent)
                names: List[str] = []
                for e in recent:
                    for nm in e[4]:
                        if nm not in names:
                            names.append(nm)
                rows.append((last_seen, ip, frames, fake, gated, names))
            if not rows:
                return None
            rows.sort(key=lambda r: -r[0])  # most-recently-active first
            n = len(rows)
            clock = time.strftime("%H:%M:%S", time.localtime(t))
            out = [f"──── live · {n} user{'s' if n != 1 else ''} · {clock} ────"]
            for last_seen, ip, frames, fake, gated, names in rows:
                badge = badges.badge(ip)
                fpct = int(round(100 * fake / frames)) if frames else 0
                gpct = int(round(100 * gated / frames)) if frames else 0
                dt = int(round(t - last_seen))
                nm = ", ".join(names) if names else "—"
                out.append(
                    f" {badge.tag()} {ip}  {frames}f  {fpct}%fake  {gpct}%gated  seen {dt}s  {nm}"
                )
            return "\n".join(out)


# --- Orchestration --------------------------------------------------------- #
class LiveLog:
    """Glue: turns a request batch into a logged block + feeds the dashboard.

    `log_batch` is fail-open — a logging bug must never surface as a 500 on the
    inference path. The dashboard runs on a daemon thread (`start`/`stop`);
    `_tick` is the testable single step.
    """

    def __init__(self, log_fn: Callable[[str], None], cfg: Optional[LogConfig] = None,
                 badges: Optional[BadgeRegistry] = None, activity: Optional[ActivityRegistry] = None,
                 clock_fn: Optional[Callable[[], str]] = None, pid_parser=None):
        self.log_fn = log_fn
        self.cfg = cfg or LogConfig()
        self.badges = badges or BadgeRegistry()
        self.activity = activity or ActivityRegistry()
        self._clock_fn = clock_fn or (lambda: time.strftime("%H:%M:%S"))
        self._pid_parser = pid_parser
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def log_batch(self, filenames, per_frame_status, threshold, ip, latency_ms):
        try:
            badge = self.badges.badge(ip)
            parts = group_by_participant(filenames, per_frame_status, self._pid_parser, threshold)
            self.log_fn(render_batch(badge, ip, parts, threshold, latency_ms, self.cfg, clock=self._clock_fn()))
            n_frames = sum(p.n for p in parts)
            n_fake = sum(
                1 for p in parts for f in p.frames
                if f.is_voting and f.prob is not None and f.prob >= threshold
            )
            n_gated = sum(p.n_gated for p in parts)
            self.activity.record(ip, [p.name for p in parts], n_frames, n_fake, n_gated)
        except Exception:
            # Telemetry/console only — never break serving.
            pass

    def _tick(self, now=None):
        if self.activity.should_print():
            txt = self.activity.render(self.badges, now=now, window_seconds=self.cfg.window_seconds)
            if txt:
                self.log_fn(txt)

    def _loop(self):
        while not self._stop.wait(self.cfg.dashboard_seconds):
            try:
                self._tick()
            except Exception:
                pass

    def start(self):
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="live-log-dashboard", daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
