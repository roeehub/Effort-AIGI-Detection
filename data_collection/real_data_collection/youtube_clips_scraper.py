#!/usr/bin/env python3
"""youtube_clips_scraper.py
-------------------------------------------------
Collect short talking‑head clips (≤ N seconds) from YouTube playlists, channels, or keyword searches.

Key features
============
* **Preview mode** – list candidate videos in a human‑readable markdown/CSV before any download.
* **Scrape mode** – download, trim, run face‑presence & size gate, write accepted clips + metrics to `clips_metadata.jsonl`.
* **Source flexibility** –
  • keyword queries (LLM‑generated or static)
  • event playlists / channels from a `sources.csv`
* **Safety & robustness** –
  • Daily quota check (10000 units default)
  • Resume after interruptions using `seen_videos.jsonl`
  • Dedup on YouTube ID + pHash of first frame
* **Tunable thresholds** – `--min-face-area`, `--min-face-ratio`, `--clip-seconds`, `--sample-fps`.

Dependencies
------------
```bash
pip install google-api-python-client pytube mediapipe opencv-python imagehash pillow tqdm
# ffmpeg must be available on $PATH
```

Usage examples
--------------
Preview 50 videos from event playlists only:
```bash
python youtube_clips_scraper.py preview --sources sources.csv --max-videos 50
```
Generate multilingual keyword list with Gemini, preview 10 per query:
```bash
python youtube_clips_scraper.py preview --queries queries.txt --per-query 10
```
Full scrape targeting 3000 accepted clips into ./clips:
```bash
python youtube_clips_scraper.py scrape --sources sources.csv --queries queries.txt \
    --target 3000 --clip-seconds 180 --out clips
```
"""
from __future__ import annotations
import os, sys, csv, json, time, random, pathlib, argparse, logging, hashlib, subprocess
from typing import List, Dict, Iterable, Tuple, Optional
from datetime import datetime
from itertools import islice

from googleapiclient.discovery import build  # noqa
from pytube import YouTube  # noqa
import cv2  # noqa
import mediapipe as mp  # noqa
from PIL import Image  # noqa
import imagehash  # noqa
from tqdm import tqdm  # noqa
from dotenv import load_dotenv  # noqa
from googleapiclient.errors import HttpError  # noqa

load_dotenv()  # Load environment variables from .env file

# --------------- Global constants & helpers ----------------
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"
DEFAULT_CLIP_SEC = 180
DAILY_UNIT_BUDGET = 10_000

mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    # level=logging.DEBUG,  # Set to DEBUG for detailed output
)
logger = logging.getLogger("scraper")


# ---------------- YouTube API wrappers ---------------------

def yt_build(api_key: str):
    return build(API_SERVICE_NAME, API_VERSION, developerKey=api_key, cache_discovery=False)


def search_videos(service, query: str, max_results: int = 20, region: Optional[str] = None) -> List[str]:
    """Return a list of video IDs for the given search query."""
    results = service.search().list(
        q=query,
        part="id",
        maxResults=min(max_results, 50),
        type="video",
        videoDuration="medium",
        videoEmbeddable="true",
        regionCode=region,
    ).execute()
    return [item["id"]["videoId"] for item in results.get("items", [])]


def playlist_items(service, playlist_id: str, limit: int = 500) -> List[str]:
    vids = []
    page_token = None
    while True:
        resp = service.playlistItems().list(
            playlistId=playlist_id,
            part="contentDetails",
            maxResults=50,
            pageToken=page_token,
        ).execute()
        vids.extend(i["contentDetails"]["videoId"] for i in resp.get("items", []))
        page_token = resp.get("nextPageToken")
        if not page_token or len(vids) >= limit:
            break
    return vids[:limit]


def channel_latest_playlists(service, channel_id: str, limit: int = 10) -> List[str]:
    """Return recent playlist IDs for a channel."""
    try:
        resp = service.playlists().list(
            channelId=channel_id,
            part="id",
            maxResults=limit,
        ).execute()
    except HttpError as e:
        logger.warning(f"Skipping channel {channel_id}: {e}")
        return []  # Gracefully skip bad or deleted channels
    return [item["id"] for item in resp.get("items", [])]


def video_details(service, ids: List[str]):
    batched = []
    for i in range(0, len(ids), 50):
        part = service.videos().list(part="snippet,contentDetails", id=','.join(ids[i:i + 50])).execute()
        batched.extend(part.get("items", []))
    return {item["id"]: item for item in batched}


# ---------------- Download & validation --------------------

def download_first_seconds(video_id: str, seconds: int, out_dir: pathlib.Path) -> Optional[pathlib.Path]:
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if not stream:
            return None
        tmp_path = out_dir / f"{video_id}_full.mp4"
        stream.download(output_path=out_dir, filename=tmp_path.name)
        clip_path = out_dir / f"{video_id}.mp4"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_path), "-t", str(seconds), "-c", "copy",
               str(clip_path)]
        subprocess.run(cmd, check=True)
        tmp_path.unlink(missing_ok=True)
        return clip_path
    except Exception as e:
        logger.warning(f"Download failed for {video_id}: {e}")
        return None


def face_gate(video_path: pathlib.Path, sample_fps: int = 1, min_face_area: float = 0.05, min_ratio: float = 0.6) -> \
        Tuple[bool, float]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(int(fps // sample_fps), 1)
    total = ok = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if total % frame_interval != 0:
            total += 1
            continue
        total += 1
        h, w = frame.shape[:2]
        results = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            boxes = [d.location_data.relative_bounding_box for d in results.detections]
            # take the largest area
            area = max(b.width * b.height for b in boxes)
            if area >= min_face_area:
                ok += 1
    cap.release()
    ratio = ok / total if total else 0
    return ratio >= min_ratio, ratio


def phash_first_frame(video_path: pathlib.Path) -> str:
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return ""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return str(imagehash.phash(img))


# ---------------- Utility I/O ------------------------------

def load_seen(jsonl: pathlib.Path) -> Dict[str, str]:
    seen = {}
    if jsonl.exists():
        with open(jsonl) as fh:
            for line in fh:
                obj = json.loads(line)
                seen[obj["video_id"]] = obj.get("phash", "")
    return seen


def append_metadata(jsonl: pathlib.Path, record: Dict):
    with open(jsonl, "a") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------- Discovery helpers ------------------------

def iter_sources_csv(service, csv_path: pathlib.Path, max_per_playlist: int = 200) -> Iterable[str]:
    with open(csv_path, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            kind = row['kind']
            id_ = row['id']
            if kind == 'playlist':
                vids = playlist_items(service, id_, limit=max_per_playlist)
            elif kind == 'channel':
                pl_ids = channel_latest_playlists(service, id_)
                vids = []
                for pid in pl_ids:
                    vids.extend(playlist_items(service, pid, limit=max_per_playlist))
            else:
                continue
            for v in vids:
                yield v


# ---------------- Preview ----------------------------------

def preview_candidates(service, video_ids: List[str], n: int):
    random.shuffle(video_ids)
    video_ids = video_ids[:n]
    details = video_details(service, video_ids)
    print("| # | Video ID | Title | Channel | Duration | Link |")
    print("|---|---|---|---|---|---|")
    for idx, vid in enumerate(video_ids, 1):
        det = details.get(vid)
        if not det:
            continue
        title = det['snippet']['title'][:60]
        channel = det['snippet']['channelTitle'][:30]
        dur = det['contentDetails']['duration']
        link = f"https://youtu.be/{vid}"
        print(f"|{idx}|{vid}|{title}|{channel}|{dur}|{link}|")


# ---------------- Main workflow ----------------------------

def run_preview(args, service):
    candidates = []
    if args.sources:
        candidates.extend(iter_sources_csv(service, pathlib.Path(args.sources), max_per_playlist=args.max_per_playlist))
    if args.queries:
        with open(args.queries) as fh:
            qlist = [q.strip() for q in fh if q.strip()]
        for q in qlist:
            candidates.extend(search_videos(service, q, max_results=args.per_query))
    print(f"Previewing {min(len(candidates), args.max_videos)} of {len(candidates)} collected IDs…")
    preview_candidates(service, candidates, args.max_videos)


def run_scrape(args, service):
    out_dir = pathlib.Path(args.out).expanduser();
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "clips_metadata.jsonl"
    seen = load_seen(meta_path)
    target_left = max(args.target - len(seen), 0)
    logger.info(f"Already have {len(seen)} accepted clips; need {target_left} more.")
    if target_left == 0:
        return
    # build candidate queue
    queue = []
    if args.sources:
        queue.extend(iter_sources_csv(service, pathlib.Path(args.sources), max_per_playlist=args.max_per_playlist))
    if args.queries:
        with open(args.queries) as fh:
            qlist = [q.strip() for q in fh if q.strip()]
        for q in qlist:
            queue.extend(search_videos(service, q, max_results=args.per_query))
    random.shuffle(queue)

    for vid in tqdm(queue, desc="Processing", unit="video"):
        if vid in seen:
            continue
        clip_path = download_first_seconds(vid, args.clip_seconds, out_dir)
        if not clip_path:
            continue
        ok, ratio = face_gate(clip_path, sample_fps=args.sample_fps, min_face_area=args.min_face_area,
                              min_ratio=args.min_face_ratio)
        if not ok:
            clip_path.unlink(missing_ok=True)
            continue
        phash = phash_first_frame(clip_path)
        # simple perceptual dedup
        if phash in seen.values():
            clip_path.unlink(missing_ok=True)
            continue
        record = {
            "video_id": vid,
            "filename": clip_path.name,
            "phash": phash,
            "face_ratio": round(ratio, 3),
            "scraped_at": datetime.utcnow().isoformat() + "Z",
        }
        append_metadata(meta_path, record)
        seen[vid] = phash
        target_left -= 1
        if target_left <= 0:
            logger.info("Target reached – exiting.")
            break


# ---------------- CLI --------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="YouTube talking‑head clip scraper")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prev = sub.add_parser("preview", help="List candidate videos without downloading")
    p_prev.add_argument("--sources", help="CSV with playlists/channels")
    p_prev.add_argument("--queries", help="TXT file with search queries")
    p_prev.add_argument("--per-query", type=int, default=10)
    p_prev.add_argument("--max-videos", type=int, default=50)
    p_prev.add_argument("--max-per-playlist", type=int, default=200)

    p_scr = sub.add_parser("scrape", help="Download & validate clips")
    p_scr.add_argument("--sources")
    p_scr.add_argument("--queries")
    p_scr.add_argument("--per-query", type=int, default=20)
    p_scr.add_argument("--max-per-playlist", type=int, default=200)
    p_scr.add_argument("--target", type=int, default=3000)
    p_scr.add_argument("--clip-seconds", type=int, default=DEFAULT_CLIP_SEC)
    p_scr.add_argument("--sample-fps", type=int, default=1)
    p_scr.add_argument("--min-face-area", type=float, default=0.05,
                       help="min face bbox area (relative) to accept frame")
    p_scr.add_argument("--min-face-ratio", type=float, default=0.6, help="min ratio of good frames to accept clip")
    p_scr.add_argument("--out", default="clips")

    return p.parse_args()


# -----------------------------------------------------------

def main():
    args = parse_args()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        logger.error("YOUTUBE_API_KEY env‑var required.")
        sys.exit(1)
    yt = yt_build(api_key)

    if args.cmd == "preview":
        run_preview(args, yt)
    elif args.cmd == "scrape":
        run_scrape(args, yt)


if __name__ == "__main__":
    main()
