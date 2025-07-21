#!/usr/bin/env python3
"""
probe.py – probe phase (spec §2.1).
"""

from __future__ import annotations
import numpy as np  # noqa
import cv2  # noqa
import os
import logging
from typing import Tuple
import pytube  # noqa

from face import detect_faces_bgr
from stream_utils import pick_stream_urls, fetch_snippet

log = logging.getLogger("probe")


def probe_video(video_id: str, cfg_probe: dict, cfg_face: dict) -> Tuple[bool, float, str, str, str]:
    """
    Returns (accepted, duration_sec, low_url, high_url).
    accepted = False if face threshold not met or any step fails.
    """
    try:
        low_url, high_url = pick_stream_urls(video_id)
    except Exception as e:
        log.debug(f"{video_id}: stream pick failed – {e}")
        # returns (..., reason)  reason = "" if accepted
        return False, 0.0, "", "", "stream"

    S = cfg_probe["samples"]
    snip_sec = cfg_probe["snippet_sec"]
    hits_needed = cfg_probe["hits_needed"]

    # pytube already gave us duration
    duration = pytube.YouTube(f"https://youtu.be/{video_id}").length or 0
    if duration < snip_sec:  # too short to probe
        return False, float(duration), low_url, high_url, "too-short"

    ts = np.linspace(0, max(duration - snip_sec, 1), S)
    hits = 0
    for t in ts:
        try:
            tmp = fetch_snippet(low_url, t, snip_sec)
            cap = cv2.VideoCapture(str(tmp))
            ret, frame = cap.read();
            cap.release()
            os.unlink(tmp)
            if not ret:
                continue
            if detect_faces_bgr(frame, det_conf=cfg_face["min_confidence"]):
                hits += 1
                if hits >= hits_needed:
                    return True, float(duration), low_url, high_url, ""
        except Exception as e:
            log.debug(f"{video_id}: snippet @ {t:.1f}s failed – {e}")

    return False, float(duration), low_url, high_url, "no-face"
