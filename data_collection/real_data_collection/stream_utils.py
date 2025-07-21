#!/usr/bin/env python3
"""
stream_utils.py – progressive-stream helpers + tiny HTTP-range fetch.
"""

from __future__ import annotations
import subprocess, tempfile, pathlib
from typing import Optional

from pytube import YouTube


def pick_stream_urls(video_id: str) -> tuple[str, str]:
    """
    Returns (low_res_url, high_res_url) – best progressive MP4 at ≥480p and highest overall.
    Raises ValueError if no progressive streams.
    """
    yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
    prog = yt.streams.filter(progressive=True, file_extension="mp4")
    if not prog:
        raise ValueError("No progressive MP4 streams found")
    low = max((s for s in prog if s.resolution and int(s.resolution.rstrip("p")) >= 480),
              key=lambda s: int(s.resolution.rstrip("p")))
    high = max(prog, key=lambda s: int(s.resolution.rstrip("p")))
    return low.url, high.url


def fetch_snippet(url: str, start: float, dur: float) -> pathlib.Path:
    """
    HTTP-range download using ffmpeg, returns a temp file path.
    """
    tmp = pathlib.Path(tempfile.mkstemp(suffix=".mp4")[1])
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-ss", str(start), "-t", str(dur),
        "-i", url, "-c", "copy", "-y", str(tmp)
    ]
    subprocess.run(cmd, check=True)
    return tmp
