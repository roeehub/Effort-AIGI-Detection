"""Tests for the viewer's bounded thumbnail cache eviction.

`proxy_frame` caches downscaled thumbnails on disk so re-views are instant, but
nothing evicted them — a long QC session over a 67k-frame meeting could fill the
disk. `_evict_frame_cache` keeps the cache under a size cap by deleting the
oldest files first.
"""
from __future__ import annotations

import os
from pathlib import Path

import viewer.obs_server as obs


def _mkfile(d: Path, name: str, size: int, mtime: float) -> Path:
    p = Path(d) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)
    os.utime(p, (mtime, mtime))
    return p


def test_no_eviction_when_under_max(tmp_path):
    _mkfile(tmp_path, "a", 100, 1000)
    _mkfile(tmp_path, "b", 100, 2000)
    freed = obs._evict_frame_cache(max_bytes=1000, target_bytes=800, cache_dir=tmp_path)
    assert freed == 0
    assert (tmp_path / "a").exists() and (tmp_path / "b").exists()


def test_evicts_oldest_first_until_under_target(tmp_path):
    _mkfile(tmp_path, "old", 500, 1000)   # oldest
    _mkfile(tmp_path, "mid", 500, 2000)
    _mkfile(tmp_path, "new", 500, 3000)   # newest
    freed = obs._evict_frame_cache(max_bytes=1000, target_bytes=600, cache_dir=tmp_path)
    assert freed == 1000
    assert not (tmp_path / "old").exists()
    assert not (tmp_path / "mid").exists()
    assert (tmp_path / "new").exists()    # newest kept


def test_evicts_nested_files(tmp_path):
    sub = tmp_path / "remote-live-data" / "v1"
    _mkfile(sub, "x", 1200, 1000)
    freed = obs._evict_frame_cache(max_bytes=1000, target_bytes=500, cache_dir=tmp_path)
    assert freed == 1200
    assert not (sub / "x").exists()


def test_missing_dir_is_safe(tmp_path):
    assert obs._evict_frame_cache(1000, 800, cache_dir=tmp_path / "does-not-exist") == 0
