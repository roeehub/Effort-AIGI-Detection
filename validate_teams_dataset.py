#!/usr/bin/env python3
"""
Post-Run Validation — audit teams_dataset/ output and spot-check quality.

Can run locally on Machine B (--dataset-dir) OR remotely from Machine A
(--receiver-url) by querying the receiver's HTTP API.

Produces a coverage report + optional HTML viewer comparing Teams crops
against GCS ground-truth originals.

Usage:
  # REMOTE from Machine A (recommended — has GCS creds + receiver API):
  python validate_teams_dataset.py --receiver-url http://10.0.0.19:8080 \\
    --compare-gcs --n-compare 20

  # LOCAL on Machine B:
  python validate_teams_dataset.py --dataset-dir C:\\path\\to\\teams_dataset \\
    --session-log C:\\path\\to\\session_log.json

  # With GCS visual comparison (downloads originals for spot-check):
  python validate_teams_dataset.py --receiver-url http://10.0.0.19:8080 \\
    --compare-gcs --n-compare 20

  # Generate rerun manifest for missing samples:
  python validate_teams_dataset.py --receiver-url http://10.0.0.19:8080 \\
    --playlist playlist.json --gen-rerun
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

# ── Config ──────────────────────────────────────────────────────────────────

MIN_FRAMES_PER_SIDE = 3       # minimum frames to consider a side "done"
EXPECTED_FRAMES_PER_SIDE = 10  # ideal frame count per side
MIN_FILE_SIZE_BYTES = 5000     # frames below this are suspect
GCS_PROJECT = "train-cvit2"
GCS_FRAMES_BUCKET = "live-deepfake-methods-real-and-fake-frames-cropped"
N_COMPARE_DEFAULT = 0          # 0 = compare ALL complete pairs


# ── Remote receiver client ──────────────────────────────────────────────────

class ReceiverClient:
    """Thin HTTP client for the receiver's dataset inspection API."""

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def _get(self, path: str, **params) -> dict:
        r = requests.get(f"{self.base}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def health(self) -> dict:
        return self._get("/health")

    def dataset_samples(self) -> dict:
        return self._get("/dataset/samples")

    def dataset_frames(self, sample_id: str, side: str,
                       max_frames: int = 6) -> dict:
        return self._get(f"/dataset/frames/{sample_id}/{side}",
                         max_frames=max_frames)

    def session_log(self) -> dict:
        return self._get("/dataset/session-log")


def scan_dataset_remote(rc: ReceiverClient) -> dict:
    """Fetch sample inventory from receiver API and return in same format as scan_dataset."""
    resp = rc.dataset_samples()
    raw = resp.get("samples", {})

    samples: dict[str, dict] = {}
    for sid, info in raw.items():
        for side in ("real", "fake"):
            s = info.get(side, {"count": 0, "files": [], "sizes": []})
            sizes = s.get("sizes", [])
            s["min_size"] = min(sizes) if sizes else 0
            s["max_size"] = max(sizes) if sizes else 0
            s["avg_size"] = sum(sizes) / len(sizes) if sizes else 0
            s["suspect_count"] = sum(1 for sz in sizes if sz < MIN_FILE_SIZE_BYTES)
            info[side] = s
        info["sample_id"] = sid
        samples[sid] = info

    return samples


# ── Directory scanning (local) ──────────────────────────────────────────────

def scan_dataset(dataset_dir: Path) -> dict:
    """Scan teams_dataset/ and return per-sample stats."""
    samples: dict[str, dict] = {}

    if not dataset_dir.exists():
        print(f"  ERROR: Dataset directory does not exist: {dataset_dir}")
        return samples

    for sample_dir in sorted(dataset_dir.iterdir()):
        if not sample_dir.is_dir():
            continue

        sample_id = sample_dir.name
        info: dict = {"sample_id": sample_id, "real": {}, "fake": {}}

        for side in ("real", "fake"):
            side_dir = sample_dir / side
            if not side_dir.exists():
                info[side] = {"count": 0, "files": [], "sizes": []}
                continue

            files = sorted(side_dir.glob("*.jpg"))
            sizes = []
            for f in files:
                try:
                    sizes.append(f.stat().st_size)
                except OSError:
                    sizes.append(0)

            info[side] = {
                "count": len(files),
                "files": [f.name for f in files],
                "sizes": sizes,
                "min_size": min(sizes) if sizes else 0,
                "max_size": max(sizes) if sizes else 0,
                "avg_size": sum(sizes) / len(sizes) if sizes else 0,
                "suspect_count": sum(1 for s in sizes if s < MIN_FILE_SIZE_BYTES),
            }

        samples[sample_id] = info

    return samples


def classify_samples(samples: dict) -> dict:
    """Classify samples into complete, partial, empty."""
    complete = []
    partial_real_only = []
    partial_fake_only = []
    partial_both_low = []
    empty = []

    for sid, info in samples.items():
        rc = info["real"]["count"]
        fc = info["fake"]["count"]

        if rc >= MIN_FRAMES_PER_SIDE and fc >= MIN_FRAMES_PER_SIDE:
            complete.append(sid)
        elif rc == 0 and fc == 0:
            empty.append(sid)
        elif rc >= MIN_FRAMES_PER_SIDE and fc < MIN_FRAMES_PER_SIDE:
            partial_real_only.append(sid)
        elif fc >= MIN_FRAMES_PER_SIDE and rc < MIN_FRAMES_PER_SIDE:
            partial_fake_only.append(sid)
        else:
            partial_both_low.append(sid)

    return {
        "complete": complete,
        "partial_real_only": partial_real_only,
        "partial_fake_only": partial_fake_only,
        "partial_both_low": partial_both_low,
        "empty": empty,
    }


# ── Coverage report ─────────────────────────────────────────────────────────

def print_coverage_report(samples: dict, classification: dict,
                          session_log: dict | None = None,
                          playlist: list[dict] | None = None):
    """Print a comprehensive coverage report."""
    total = len(samples)
    n_complete = len(classification["complete"])
    n_partial = (len(classification["partial_real_only"]) +
                 len(classification["partial_fake_only"]) +
                 len(classification["partial_both_low"]))
    n_empty = len(classification["empty"])

    # Frame count stats
    all_real_counts = [s["real"]["count"] for s in samples.values()]
    all_fake_counts = [s["fake"]["count"] for s in samples.values()]
    all_real_nonzero = [c for c in all_real_counts if c > 0]
    all_fake_nonzero = [c for c in all_fake_counts if c > 0]

    print(f"\n{'=' * 60}")
    print(f"  TEAMS DATASET VALIDATION REPORT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")

    print(f"  Total sample directories:  {total}")
    print(f"  Complete (real+fake ≥{MIN_FRAMES_PER_SIDE}):  {n_complete}  "
          f"({n_complete / total * 100:.1f}%)" if total else "")
    print(f"  Partial:                   {n_partial}")
    if classification["partial_real_only"]:
        print(f"    - real only:             {len(classification['partial_real_only'])}")
    if classification["partial_fake_only"]:
        print(f"    - fake only:             {len(classification['partial_fake_only'])}")
    if classification["partial_both_low"]:
        print(f"    - both low:              {len(classification['partial_both_low'])}")
    print(f"  Empty:                     {n_empty}")

    print(f"\n  Frame count statistics:")
    if all_real_nonzero:
        print(f"    Real:  min={min(all_real_nonzero)}, "
              f"max={max(all_real_nonzero)}, "
              f"avg={sum(all_real_nonzero) / len(all_real_nonzero):.1f}, "
              f"zero={all_real_counts.count(0)}")
    if all_fake_nonzero:
        print(f"    Fake:  min={min(all_fake_nonzero)}, "
              f"max={max(all_fake_nonzero)}, "
              f"avg={sum(all_fake_nonzero) / len(all_fake_nonzero):.1f}, "
              f"zero={all_fake_counts.count(0)}")

    # File size check
    total_suspect = sum(
        s["real"].get("suspect_count", 0) + s["fake"].get("suspect_count", 0)
        for s in samples.values()
    )
    if total_suspect:
        print(f"\n  ⚠ Suspect frames (<{MIN_FILE_SIZE_BYTES}B): {total_suspect}")

    # Cross-reference with playlist if available
    if playlist:
        playlist_sample_ids = set()
        for entry in playlist:
            playlist_sample_ids.add(entry["sample_id"])
        n_playlist = len(playlist_sample_ids)
        captured_ids = set(samples.keys())
        missing_ids = playlist_sample_ids - captured_ids
        extra_ids = captured_ids - playlist_sample_ids

        print(f"\n  Playlist cross-reference:")
        print(f"    Playlist samples:    {n_playlist}")
        print(f"    Captured samples:    {len(captured_ids)}")
        print(f"    Missing from output: {len(missing_ids)}")
        if missing_ids and len(missing_ids) <= 20:
            for sid in sorted(missing_ids)[:20]:
                print(f"      - {sid}")
        elif missing_ids:
            print(f"      (showing first 20 of {len(missing_ids)})")
            for sid in sorted(missing_ids)[:20]:
                print(f"      - {sid}")
        if extra_ids:
            print(f"    Extra (not in playlist): {len(extra_ids)}")

    # Cross-reference with session log if available
    if session_log:
        log_segments = session_log.get("segments", [])
        n_timed_out = sum(1 for s in log_segments if s.get("timed_out"))
        n_zero_copied = sum(1 for s in log_segments if s.get("copied_frames", 0) == 0)
        print(f"\n  Session log cross-reference:")
        print(f"    Total segments logged: {len(log_segments)}")
        print(f"    Timed out:             {n_timed_out}")
        print(f"    Zero frames copied:    {n_zero_copied}")

    print()


# ── Strategy breakdown ──────────────────────────────────────────────────────

def print_strategy_breakdown(samples: dict, classification: dict):
    """Break down results by strategy (extracted from sample_id)."""
    import re

    strategy_stats: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "complete": 0, "partial": 0, "empty": 0}
    )

    complete_set = set(classification["complete"])
    empty_set = set(classification["empty"])

    for sid in samples:
        m = re.match(r"^(.+?)_(\d{4,})$", sid)
        strat = m.group(1) if m else "unknown"
        strategy_stats[strat]["total"] += 1
        if sid in complete_set:
            strategy_stats[strat]["complete"] += 1
        elif sid in empty_set:
            strategy_stats[strat]["empty"] += 1
        else:
            strategy_stats[strat]["partial"] += 1

    print(f"\n  Strategy breakdown:")
    print(f"  {'Strategy':<35s} {'Total':>6s} {'Done':>6s} {'Part':>6s} {'Empty':>6s} {'Rate':>7s}")
    print(f"  {'-' * 35} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 7}")
    for strat in sorted(strategy_stats):
        s = strategy_stats[strat]
        rate = s["complete"] / s["total"] * 100 if s["total"] else 0
        print(f"  {strat:<35s} {s['total']:>6d} {s['complete']:>6d} "
              f"{s['partial']:>6d} {s['empty']:>6d} {rate:>6.1f}%")
    print()


# ── GCS visual comparison ──────────────────────────────────────────────────

def build_comparison_html(dataset_dir: Path | None, samples: dict,
                          n_compare: int, output_path: Path,
                          rc: ReceiverClient | None = None):
    """Build an HTML viewer comparing Teams crops against GCS originals.

    Teams frames are loaded from local dataset_dir (if provided) or fetched
    from the receiver API (if rc is provided).
    """
    try:
        from google.cloud import storage as gcs
    except ImportError:
        print("  Cannot build GCS comparison: pip install google-cloud-storage")
        return

    print(f"\n  Building HTML comparison viewer ({n_compare} pairs)...")
    client = gcs.Client(project=GCS_PROJECT)
    bucket = client.bucket(GCS_FRAMES_BUCKET)

    # Select samples to compare — 0 means ALL complete samples
    complete_ids = [sid for sid, info in samples.items()
                    if info["real"]["count"] >= MIN_FRAMES_PER_SIDE
                    and info["fake"]["count"] >= MIN_FRAMES_PER_SIDE]
    if not complete_ids:
        print("  No complete samples to compare.")
        return

    if n_compare <= 0 or n_compare >= len(complete_ids):
        selected = complete_ids  # show ALL
    else:
        step = max(1, len(complete_ids) // n_compare)
        selected = complete_ids[::step][:n_compare]
    print(f"  Selected {len(selected)} samples for comparison "
          f"(from {len(complete_ids)} complete)")

    # Cache dir for GCS downloads
    cache_dir = output_path.parent / "gcs_comparison_cache"
    cache_dir.mkdir(exist_ok=True)

    def get_gcs_frames(sample_id: str, frame_type: str, max_frames: int = 0) -> list[Path]:
        """Download GCS original frames. max_frames=0 means all."""
        prefix = f"samples/{sample_id}/frames/{frame_type}/"
        local_dir = cache_dir / sample_id / frame_type
        if local_dir.exists() and list(local_dir.glob("*")):
            files = sorted(local_dir.glob("*"))
            return files if max_frames == 0 else files[:max_frames]
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            blobs = list(bucket.list_blobs(prefix=prefix))
            blobs = [b for b in blobs if b.name.endswith(('.png', '.jpg'))]
            blobs.sort(key=lambda b: b.name)
            if max_frames > 0 and len(blobs) > max_frames:
                step_b = max(1, len(blobs) // max_frames)
                sel = blobs[::step_b][:max_frames]
            else:
                sel = blobs
            paths = []
            for b in sel:
                fname = os.path.basename(b.name)
                local = local_dir / fname
                if not local.exists():
                    b.download_to_filename(str(local))
                paths.append(local)
            return paths
        except Exception as e:
            print(f"    GCS error for {sample_id}/{frame_type}: {e}")
            return []

    def get_teams_frames_b64(sample_id: str, side: str) -> list[dict]:
        """Get ALL Teams frames as [{filename, size, b64}, ...].

        Uses receiver API if rc is available, otherwise reads from local disk.
        Returns every single frame — no subsampling.
        """
        if rc is not None:
            try:
                resp = rc.dataset_frames(sample_id, side, max_frames=0)
                return resp.get("frames", [])
            except Exception as e:
                print(f"    Receiver API error for {sample_id}/{side}: {e}")
                return []
        elif dataset_dir is not None:
            side_dir = dataset_dir / sample_id / side
            if not side_dir.exists():
                return []
            files = sorted(side_dir.glob("*.jpg"))
            result = []
            for fr in files:
                with open(fr, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode()
                result.append({
                    "filename": fr.name,
                    "size": fr.stat().st_size,
                    "b64": b64,
                })
            return result
        return []

    def img_b64_from_path(path: Path) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def img_type(path: Path) -> str:
        return "png" if path.suffix == ".png" else "jpeg"

    html_parts = ["""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Teams Dataset Validation — Visual Comparison</title>
<style>
body { font-family: -apple-system, 'Segoe UI', monospace; background: #1a1a2e;
       color: #eee; padding: 20px; max-width: 1800px; margin: 0 auto; }
h1, h2, h3 { color: #fff; }
.summary { background: #16213e; padding: 15px; border-radius: 8px;
           margin-bottom: 20px; border-left: 4px solid #0f3460; }
.pair { margin: 15px 0; padding: 15px; border: 1px solid #333;
        border-radius: 8px; background: #16213e; }
.pair h3 { margin: 0 0 8px 0; font-size: 14px; }
.row { display: flex; gap: 15px; margin: 5px 0; }
.side { flex: 1; }
.label { font-size: 12px; font-weight: bold; margin-bottom: 4px; }
.teams-label { color: #66bb6a; }
.gcs-label { color: #4fc3f7; }
.faces { display: flex; gap: 3px; flex-wrap: wrap; align-items: flex-start; }
.face-img { border: 2px solid #444; border-radius: 3px; max-height: 80px; }
.divider { border-left: 2px solid #4fc3f7; padding-left: 10px; }
.good { border-color: #66bb6a; }
.warn { border-color: #ffab40; }
.fail { border-color: #ef5350; }
.meta { color: #777; font-size: 11px; }
.verdict { font-weight: bold; padding: 3px 8px; border-radius: 4px;
           display: inline-block; font-size: 11px; }
.verdict-ok { background: #1b5e20; color: #a5d6a7; }
.verdict-warn { background: #e65100; color: #ffcc80; }
.verdict-fail { background: #b71c1c; color: #ef9a9a; }
</style></head><body>
<h1>Teams Dataset — Visual Comparison</h1>
<div class="summary">
<p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
<p>Comparing <b>Teams face crops</b> (left, green) against <b>GCS originals</b> (right, blue).</p>
<p>Verify: do the faces in each pair match? Are there misaligned or corrupt frames?</p>
</div>
"""]

    for i, sid in enumerate(selected):
        info = samples[sid]
        html_parts.append(f'<div class="pair">')
        html_parts.append(f'<h3>#{i + 1}  {sid}  '
                          f'(real: {info["real"]["count"]}f, '
                          f'fake: {info["fake"]["count"]}f)</h3>')

        for side in ("real", "fake"):
            html_parts.append(f'<div class="row">')

            # Teams frames (from receiver API or local disk)
            html_parts.append(f'<div class="side">')
            html_parts.append(f'<div class="label teams-label">'
                              f'Teams {side} ({info[side]["count"]} frames)</div>')
            html_parts.append(f'<div class="faces">')

            teams_frames = get_teams_frames_b64(sid, side)
            for fr in teams_frames:
                sz = fr["size"]
                border = "good" if sz >= MIN_FILE_SIZE_BYTES else "fail"
                html_parts.append(
                    f'<img class="face-img {border}" '
                    f'src="data:image/jpeg;base64,{fr["b64"]}" '
                    f'title="{fr["filename"]} ({sz:,}B)">'
                )
            if not teams_frames:
                html_parts.append('<span class="meta">No frames available</span>')

            html_parts.append('</div></div>')

            # GCS originals
            html_parts.append(f'<div class="side divider">')
            html_parts.append(f'<div class="label gcs-label">GCS {side}</div>')
            html_parts.append(f'<div class="faces">')

            gcs_paths = get_gcs_frames(sid, side, 0)  # all GCS frames
            for gp in gcs_paths:
                html_parts.append(
                    f'<img class="face-img" '
                    f'src="data:image/{img_type(gp)};base64,{img_b64_from_path(gp)}" '
                    f'title="{gp.name}">'
                )
            if not gcs_paths:
                html_parts.append('<span class="meta">No GCS frames found</span>')

            html_parts.append('</div></div>')
            html_parts.append('</div>')  # row

        html_parts.append('</div>')  # pair

    html_parts.append("""
<div class="summary" style="margin-top: 20px; border-left-color: #ffab40;">
<b>Checklist:</b><br>
1. Do Teams faces match GCS originals in identity and orientation?<br>
2. Are there any black/corrupt frames (red border)?<br>
3. Are real/fake sides correctly labeled (real = natural, fake = deepfake)?<br>
4. Is frame quality sufficient for training?
</div>
</body></html>""")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    size_kb = output_path.stat().st_size / 1024
    print(f"  HTML viewer saved: {output_path} ({size_kb:.0f} KB)")
    print(f"  Open in browser to visually inspect.")


# ── Rerun manifest generation ──────────────────────────────────────────────

def generate_rerun_manifest(dataset_dir: Path, samples: dict,
                            classification: dict,
                            playlist: list[dict] | None,
                            output_path: Path):
    """Generate a rerun manifest for incomplete samples."""
    import re

    incomplete_ids = set()
    for cat in ("partial_real_only", "partial_fake_only", "partial_both_low", "empty"):
        incomplete_ids.update(classification[cat])

    # Also add playlist entries that have no output at all
    if playlist:
        captured_ids = set(samples.keys())
        playlist_ids = set(e["sample_id"] for e in playlist)
        missing_ids = playlist_ids - captured_ids
        incomplete_ids.update(missing_ids)

    if not incomplete_ids:
        print("  No incomplete samples — no rerun manifest needed.")
        return

    # Build manifest entries with strategy info
    rerun_samples = []
    for sid in sorted(incomplete_ids):
        m = re.match(r"^(.+?)_(\d{4,})$", sid)
        strat = m.group(1) if m else "unknown"
        info = samples.get(sid, {"real": {"count": 0}, "fake": {"count": 0}})
        rerun_samples.append({
            "sample_id": sid,
            "strategy": strat,
            "existing_real_frames": info["real"]["count"] if isinstance(info["real"], dict) else 0,
            "existing_fake_frames": info["fake"]["count"] if isinstance(info["fake"], dict) else 0,
        })

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_failed_samples": len(rerun_samples),
        "rerun_samples": rerun_samples,
    }

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Rerun manifest: {output_path} ({len(rerun_samples)} samples)")
    print(f"  Use with: python obs_coordinated_sender.py ... "
          f"--rerun-manifest {output_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate teams_dataset/ output and report coverage"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-dir", type=str,
                        help="Path to teams_dataset/ directory (local mode)")
    source.add_argument("--receiver-url", type=str,
                        help="Receiver server URL (remote mode, e.g. http://10.0.0.19:8080)")
    parser.add_argument("--session-log", type=str, default=None,
                        help="Path to session_log.json (local mode; auto-fetched in remote mode)")
    parser.add_argument("--playlist", type=str, default=None,
                        help="Path to playlist.json from sender (for missing-sample detection)")
    parser.add_argument("--compare-gcs", action="store_true",
                        help="Build HTML viewer comparing Teams crops vs GCS originals")
    parser.add_argument("--n-compare", type=int, default=N_COMPARE_DEFAULT,
                        help="Number of pairs to compare with GCS (0 = ALL, default: ALL)")
    parser.add_argument("--gen-rerun", action="store_true",
                        help="Generate rerun_manifest.json for incomplete samples")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to write HTML/manifest (default: current dir or next to dataset-dir)")
    args = parser.parse_args()

    # ── Determine mode ──
    rc: ReceiverClient | None = None
    dataset_dir: Path | None = None

    if args.receiver_url:
        # Remote mode: fetch everything from receiver API
        rc = ReceiverClient(args.receiver_url)
        print(f"\n  Remote mode: connecting to {args.receiver_url} ...")
        try:
            health = rc.health()
            print(f"  Receiver OK: watching={health.get('watching')}, "
                  f"participant={health.get('participant')}")
        except Exception as e:
            sys.exit(f"  Cannot reach receiver: {e}")

        output_dir = Path(args.output_dir) if args.output_dir else Path(".")

        # Scan via API
        print(f"  Fetching sample inventory from receiver ...")
        samples = scan_dataset_remote(rc)

        # Fetch session log from receiver
        session_log = None
        try:
            session_log = rc.session_log()
            print(f"  Session log fetched ({len(session_log.get('segments', []))} segments)")
        except Exception:
            print(f"  No session log available on receiver.")
    else:
        # Local mode: scan disk
        dataset_dir = Path(args.dataset_dir)
        output_dir = Path(args.output_dir) if args.output_dir else dataset_dir.parent

        print(f"\n  Scanning {dataset_dir} ...")
        samples = scan_dataset(dataset_dir)

        session_log = None
        if args.session_log:
            with open(args.session_log) as f:
                session_log = json.load(f)

    if not samples:
        print("  No samples found. Check the path / receiver.")
        sys.exit(1)

    playlist = None
    if args.playlist:
        with open(args.playlist) as f:
            playlist = json.load(f)

    classification = classify_samples(samples)

    # Reports
    print_coverage_report(samples, classification, session_log, playlist)
    print_strategy_breakdown(samples, classification)

    # GCS comparison
    if args.compare_gcs:
        html_path = output_dir / "validation_comparison.html"
        build_comparison_html(dataset_dir, samples, args.n_compare, html_path,
                              rc=rc)

    # Rerun manifest
    if args.gen_rerun:
        manifest_path = output_dir / "rerun_manifest.json"
        generate_rerun_manifest(dataset_dir, samples, classification,
                                playlist, manifest_path)

    # Final verdict
    n_complete = len(classification["complete"])
    total = len(samples)
    rate = n_complete / total * 100 if total else 0
    if rate >= 95:
        print(f"  ✓ Excellent: {rate:.1f}% complete ({n_complete}/{total})")
    elif rate >= 80:
        print(f"  ~ Good: {rate:.1f}% complete ({n_complete}/{total}) — "
              f"consider rerunning failures")
    else:
        print(f"  ✗ Low completion: {rate:.1f}% ({n_complete}/{total}) — "
              f"check logs and rerun")
    print()


if __name__ == "__main__":
    main()
