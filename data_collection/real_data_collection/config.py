#!/usr/bin/env python3
"""
config.py – Load scraper configuration from YAML or JSON into frozen dataclasses.
"""
from __future__ import annotations
import json, pathlib, types
from dataclasses import dataclass, field, asdict
from typing import Any, Mapping

import yaml  # pip install pyyaml

DEFAULTS: dict[str, Any] = {
    "youtube_api_key": "TO BE SET",
      "output_dir": "clips",
      "sources_csv": "sources.csv",
      "queries_txt": None,
    "probe": {
        "samples": 5,
        "snippet_sec": 10,
        "hits_needed": 2,
        "min_res_p": 480,
    },
    "mining": {
        "window_sec": 5,
        "sample_fps": 5,
        "stride_sec": 30,
        "max_dead": 10,
        "sparse_interval": 20,
        "oversample": 35,
        "keep_frames": 32,
        "yaw_threshold": 0.35,
    },
    "face_detector": {"backend": "mediapipe_gpu", "min_confidence": 0.5},
    "dedup": {"phash_bits": 64},
    "bandwidth": {"max_bytes_per_video": 150_000_000},
}


# ---------- dataclass tree ---------- #

@dataclass(frozen=True, slots=True)
class ProbeCfg:
    samples: int
    snippet_sec: int
    hits_needed: int
    min_res_p: int


@dataclass(frozen=True, slots=True)
class MiningCfg:
    window_sec: int
    sample_fps: int
    stride_sec: int
    max_dead: int
    sparse_interval: int
    oversample: int
    keep_frames: int
    yaw_threshold: float


@dataclass(frozen=True, slots=True)
class FaceCfg:
    backend: str
    min_confidence: float


@dataclass(frozen=True, slots=True)
class DedupCfg:
    phash_bits: int


@dataclass(frozen=True, slots=True)
class BandwidthCfg:
    max_bytes_per_video: int


@dataclass(frozen=True, slots=True)
class Config:
    youtube_api_key: str
    output_dir: str
    probe: ProbeCfg
    mining: MiningCfg
    face_detector: FaceCfg
    dedup: DedupCfg
    bandwidth: BandwidthCfg
    sources_csv: str | None
    queries_txt: str | None


# ---------- loader ---------- #

def _merge(d: Mapping[str, Any], default: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge user dict over default dict."""
    out = dict(default)
    for k, v in d.items():
        if isinstance(v, Mapping) and k in out and isinstance(out[k], Mapping):
            out[k] = _merge(v, out[k])
        else:
            out[k] = v
    return out


def load_config(path: str | pathlib.Path | None = None) -> Config:
    """
    Load YAML/JSON config, merge with defaults, return frozen Config dataclass.
    If *path* is None, looks for 'config.yaml' in CWD.
    """
    path = pathlib.Path(path or "config.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} not found")

    text = path.read_text()
    try:
        user_cfg = yaml.safe_load(text)
    except yaml.YAMLError:
        user_cfg = json.loads(text)

    merged = _merge(user_cfg or {}, DEFAULTS)

    # Build dataclasses
    return Config(
        youtube_api_key=merged["youtube_api_key"],
        output_dir=merged["output_dir"],
        probe=ProbeCfg(**merged["probe"]),
        mining=MiningCfg(**merged["mining"]),
        face_detector=FaceCfg(**merged["face_detector"]),
        dedup=DedupCfg(**merged["dedup"]),
        bandwidth=BandwidthCfg(**merged["bandwidth"]),
        sources_csv=merged.get("sources_csv"),
        queries_txt=merged.get("queries_txt", None),
    )


def asdict_recursive(cfg: Config) -> dict[str, Any]:
    """Nicely convert nested frozen dataclasses to dict (for pretty‐printing)."""
    return json.loads(json.dumps(asdict(cfg)))
