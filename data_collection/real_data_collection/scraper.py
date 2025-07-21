#!/usr/bin/env python3
"""
scraper.py – CLI front-end for the talking-head scraper.

*Only* handles arg-parsing and config loading for now.
"""

import argparse, pprint
from pathlib import Path
from config import load_config, asdict_recursive
from youtube_api import (
    yt_build, search_videos, video_details,
    iter_sources_csv,
)


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CUDA-enabled talking-head scraper")
    p.add_argument("--config", type=Path, default=Path("config.yaml"),
                   help="YAML/JSON config file (default: ./config.yaml)")

    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preview", help="List candidate videos without downloading")
    sub.add_parser("scrape", help="Run full scrape pipeline")

    return p.parse_args()


def run_preview(cfg):
    svc = yt_build(cfg.youtube_api_key)
    candidates = []

    # 1. collect IDs
    if cfg.sources_csv:
        candidates += list(iter_sources_csv(svc, cfg.sources_csv, max_per_playlist=200))

    if cfg.queries_txt:
        with open(cfg.queries_txt) as fh:
            for q in (l.strip() for l in fh):
                if q:
                    candidates += search_videos(svc, q, max_results=10)

    print(f"\n🔍 Collected {len(candidates)} raw IDs, showing first 50…\n")
    candidates = candidates[:50]
    info = video_details(svc, candidates)

    # 2. markdown table
    print("| # | Video ID | Title | Channel | Duration | Link |")
    print("|---|----------|-------|---------|----------|------|")
    for i, vid in enumerate(candidates, 1):
        det = info.get(vid)
        if not det:
            continue
        title = det["snippet"]["title"][:60].replace("|", "\\|")
        chan = det["snippet"]["channelTitle"][:30].replace("|", "\\|")
        dur = det["contentDetails"]["duration"]
        print(f"|{i}|{vid}|{title}|{chan}|{dur}|https://youtu.be/{vid}|")


def main() -> None:
    args = parse_cli()
    cfg = load_config(args.config)

    # For now, just echo what we loaded
    print("✅ Loaded configuration:")
    pprint.pp(asdict_recursive(cfg), width=100, compact=True)

    if args.cmd == "preview":
        run_preview(cfg)
    elif args.cmd == "scrape":
        print("\n👉 SCRAPE mode is not implemented yet.")


if __name__ == "__main__":
    main()
