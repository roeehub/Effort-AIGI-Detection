"""youtube_avspeech_scraper.py
================================================
Scrape, validate and upload clipped talking-head segments defined by an
AVSpeech-style CSV (video_id,start_sec,end_sec,x_norm,y_norm).

v2.8 (Optimized for Reprocessing) — Added multiprocessing and batched inference for significant speed-up.
v2.7  (2025-08-11) — Added --reprocess-frames-only mode to source videos from GCS.
v2.6  (2025-08-11) — Added PNG output and --frames-only mode.
v2.5  (2025-08-10) — Added human-like delay to avoid rate-limiting.
v2.4  (2025-08-10) — Added cookie support to bypass bot detection.
------------------------------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse, csv, json, logging, re, subprocess, tempfile, time
import collections
import shutil
import random
from pathlib import Path
from typing import Dict, Tuple, List, Optional
### SPEED-UP CHANGE ### - Import libraries for multiprocessing
import multiprocessing
from functools import partial
import numpy as np

import cv2  # type: ignore
import yt_dlp  # type: ignore
from google.cloud import storage  # type: ignore
from google.api_core import exceptions as gcs_exceptions  # type: ignore
from tqdm import tqdm  # type: ignore

# suppress logs from yt-dlp
logging.getLogger("yt_dlp").setLevel(logging.WARNING)

# ──────────────────────────
# 📍 Configuration
# ──────────────────────────
GCS_VIDEO_BUCKET = "gs://youtube-real-videos"
GCS_AUDIO_BUCKET = "gs://youtube-real-audio-clips"
GCS_FRAMES_BUCKET = "gs://effort-collected-data"

QA_DIR = Path("qa_frames")
QA_DIR.mkdir(exist_ok=True)

YOLO_VARIANT = "s"
DEFAULT_BBOX_MARGIN = 20
MIN_DURATION = 4.0  # s
SAMPLE_FRAMES_FOR_FACE = 5
QA_FRAMES_PER_CLIP = 32

FACE_CONF_NORMAL = 0.20
FACE_CONF_DEBUG = 0.15
MAX_ALLOWED_FACES = 2

# New constants for retry logic
MAX_RETRIES = 5
INITIAL_BACKOFF_S = 2.0

YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
DEBUG = False  # set in main()

# Constants for rate-limiting delay
MIN_REQUEST_DELAY_S = 0.8  # Minimum seconds to wait between requests
MAX_REQUEST_DELAY_S = 2.5  # Maximum seconds to wait between requests


# ──────────────────────────
# 🐛 Debugging Tool & 🧰 sub-process helpers
# ──────────────────────────
def _run(cmd: List[str], *, show_cmd: bool = False):
    if show_cmd or DEBUG: logging.debug("RUN: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, stdout=None if DEBUG else subprocess.PIPE,
                       stderr=None if DEBUG else subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logging.error("Command failed: %s", " ".join(cmd))
        if e.stdout: logging.error("stdout:\n%s", e.stdout.decode(errors="ignore"))
        if e.stderr: logging.error("stderr:\n%s", e.stderr.decode(errors="ignore"))
        raise


def download_clip_direct(url: str, start: float, dur: float, out: Path):
    _run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start}", "-i", url, "-t", f"{dur}", "-c", "copy", str(out)])


def extract_audio(mp4: Path, out: Path):
    _run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vn", "-acodec", "libmp3lame", "-ab", "192k", str(out)])


def extract_frames(mp4: Path, dir_: Path, n: int, duration: float):
    dir_.mkdir(parents=True, exist_ok=True)
    fps = max(n / duration, 0.001)
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf", f"fps={fps}", str(dir_ / "%04d.png")])


# ──────────────────────────
# 🖼️ YOLO face helper
# ──────────────────────────
from ultralytics import YOLO  # noqa

_model_cache = {}


def detect_faces(img, margin=DEFAULT_BBOX_MARGIN) -> List[Tuple[int, int, int, int]]:
    # This function is now mostly a wrapper; batch prediction is preferred.
    if YOLO_VARIANT not in _model_cache:
        from urllib.request import urlretrieve
        w = Path(f"yolov8{YOLO_VARIANT}-face.pt")
        if not w.exists():
            print("Downloading YOLOv8-face weights…")
            urlretrieve(
                f"https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8{YOLO_VARIANT}-face-lindevs.pt",
                w)
        _model_cache[YOLO_VARIANT] = YOLO(str(w))
    conf = FACE_CONF_DEBUG if DEBUG else FACE_CONF_NORMAL
    # The 'img' can be a single image or a list of images for batch prediction
    results = _model_cache[YOLO_VARIANT].predict(img, conf=conf, iou=0.4, verbose=False)

    # Handle both single and batch results
    if not isinstance(results, list):
        results = [results]

    batch_boxes = []
    for res in results:
        if res.boxes.shape[0] == 0:
            batch_boxes.append([])
            continue

        boxes = []
        h, w = res.orig_shape
        for box in res.boxes.xyxy.cpu().numpy():
            x0, y0, x1, y1 = box
            x0 -= margin;
            y0 -= margin
            x1 += margin;
            y1 += margin
            x0, y0 = int(max(0, x0)), int(max(0, y0))
            x1, y1 = int(min(w, x1)), int(min(h, y1))
            boxes.append((x0, y0, x1, y1))
        batch_boxes.append(boxes)

    # If input was a single image, return a single list of boxes
    return batch_boxes[0] if not isinstance(img, list) else batch_boxes


# ──────────────────────────
# ☁️ GCS helper
# ──────────────────────────
gcs_client = None


def initialize_gcs_client():
    """Initializes a GCS client for each worker process."""
    global gcs_client
    if gcs_client is None:
        gcs_client = storage.Client()


class GCSUploadError(Exception): pass


class GCSDownloadError(Exception): pass


def gcs_upload(local: Path, uri: str):
    if gcs_client is None: initialize_gcs_client()
    if not uri.startswith("gs://"): raise ValueError("GCS URI must start with gs://")
    bucket_name, *rest = uri[5:].split("/", 1)
    blob_name = rest[0] if rest else ""
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    try:
        blob.upload_from_filename(str(local))
    except gcs_exceptions.Forbidden as e:
        raise GCSUploadError(f"GCS Upload Forbidden (403) for {uri}.") from e
    except Exception as e:
        raise GCSUploadError(f"GCS_upload_failed: {e}") from e


def gcs_download(uri: str, local_path: Path):
    if gcs_client is None: initialize_gcs_client()
    if not uri.startswith("gs://"): raise ValueError("GCS URI must start with gs://")
    bucket_name, *rest = uri[5:].split("/", 1)
    blob_name = rest[0] if rest else ""
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    try:
        blob.download_to_filename(str(local_path))
    except gcs_exceptions.NotFound:
        raise GCSDownloadError(f"File not found in GCS: {uri}")
    except Exception as e:
        raise GCSDownloadError(f"GCS download failed for {uri}: {e}") from e


# ... (status helpers and yt-dlp functions are unchanged) ...
STATUS_FILE = Path("status.jsonl")


def load_status() -> Dict[str, dict]:
    if not STATUS_FILE.exists(): return {}
    with STATUS_FILE.open() as fh: return {j["video_id"]: j for j in (json.loads(l) for l in fh)}


def append_status(rec: dict):
    with STATUS_FILE.open("a") as fh: fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def report_lengths(rows: List[Tuple[str, float, float]]):
    val = [r for r in rows if (r[2] - r[1]) >= MIN_DURATION]
    n = len(val)
    est = int(n * 0.85)
    print(f"\n[Report] {n} clips have duration ≥ {MIN_DURATION}s.")
    print(f"[Report] Roughly {est} expected after face-gate.\n")


class RateLimitException(Exception): pass


def best_progressive_mp4(video_id: str, cookies_file: Optional[str] = None):
    opts = {"quiet": True, "skip_download": True, "forcejson": True, "no_warnings": True, "no_color": True,
            "noprogress": True, "logger": None}
    if cookies_file:
        if Path(cookies_file).exists():
            opts["cookiefile"] = cookies_file
        else:
            logging.warning(f"Cookie file not found at '{cookies_file}', proceeding without it.")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(video_id, download=False)
    prog = [f for f in info["formats"] if
            f.get("ext") == "mp4" and f.get("vcodec") != "none" and f.get("acodec") != "none"]
    prog.sort(key=lambda f: (f.get("height") or 0, f.get("tbr") or 0), reverse=True)
    return prog[0] if prog else None


def get_video_format_with_retry(video_id: str, cookies_file: Optional[str] = None):
    for attempt in range(MAX_RETRIES):
        try:
            return best_progressive_mp4(video_id, cookies_file)
        except yt_dlp.DownloadError as e:
            if "HTTP Error 429" in str(e):
                if attempt < MAX_RETRIES - 1:
                    backoff = INITIAL_BACKOFF_S * (2 ** attempt) + random.uniform(0, 1)
                    logging.warning(f"Rate limit hit for {video_id}. Retrying in {backoff:.2f}s...")
                    time.sleep(backoff)
                    continue
                else:
                    raise RateLimitException(f"Rate limit persisted for {video_id}") from e
            else:
                raise e
    return None


def find_alternative_intro_clip(vid: str, url: str, bbox: int, tmp_dir: Path) -> Tuple[bool, Path | None, str]:
    start_time, duration = 1.0, 10.0
    backup_mp4 = tmp_dir / f"{vid}_backup.mp4"
    try:
        download_clip_direct(url, start_time, duration, backup_mp4)
    except subprocess.CalledProcessError:
        return False, None, "backup_download_failed"
    cap = cv2.VideoCapture(str(backup_mp4))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not fps or total_frames < 1: cap.release(); return False, None, "backup_video_meta_failed"
    timestamps = [i * 0.5 for i in range(20)]
    frame_indices = [idx for idx in [int(t * fps) for t in timestamps] if idx < total_frames]
    random.shuffle(frame_indices)
    if len(frame_indices) < 20: cap.release(); return False, None, "backup_not_enough_samples"
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: cap.release(); return False, None, f"backup_read_frame_{frame_idx}_failed"
        bboxes = detect_faces(frame, margin=bbox)
        if len(bboxes) != 1: cap.release(); return False, None, f"backup_found_{len(bboxes)}_faces_at_frame_{i + 1}"
        x0, y0, x1, y1 = bboxes[0]
        width, height = x1 - x0, y1 - y0
        if not (max(width, height) >= 200 and min(width,
                                                  height) >= 150): cap.release(); return False, None, f"backup_face_too_small_{width}x{height}_at_frame_{i + 1}"
    cap.release()
    return True, backup_mp4, "backup_clip_valid"


# ─────────────────
# 🔁 row processor
# ──────────────────────────
### SPEED-UP CHANGE ### - Function signature changed to accept a single tuple for easier mapping
def process_row_worker(row_tuple: Tuple, *, bbox: int, buckets: Tuple[str, str, str],
                       cookies_file: Optional[str], reprocess_frames_only: bool):
    """Worker function designed to be used with multiprocessing.Pool"""
    if reprocess_frames_only:
        vid, = row_tuple
        start, end = 0, 0
    else:
        vid, start, end = row_tuple

    # Each process needs its own GCS client and YOLO model.
    initialize_gcs_client()
    if YOLO_VARIANT not in _model_cache:
        # FIX: Use a valid NumPy array as a dummy image for initialization
        dummy_image = np.zeros((8, 8, 3), dtype=np.uint8)
        detect_faces(dummy_image)  # Dummy call to initialize model

    return process_row(vid, start, end, bbox, buckets, cookies_file, reprocess_frames_only)


def process_row(vid: str, start: float, end: float, bbox: int, buckets: Tuple[str, str, str],
                cookies_file: Optional[str], reprocess_frames_only: bool):
    dur = end - start
    # Use a unique temp dir per process/video to avoid collisions
    tmp_prefix = f"ytclip_{vid}_"
    tmp = Path(tempfile.mkdtemp(prefix=tmp_prefix))
    st = {"video_id": vid, "status": "skipped", "reason": "unknown_error", "duration": round(dur, 2)}

    try:
        if reprocess_frames_only:
            # No logging here to prevent noisy output from parallel workers
            vb, ab, fb = buckets
            mp4_path = tmp / f"{vid}.mp4"
            gcs_video_uri = f"{vb.rstrip('/')}/{vid}.mp4"

            try:
                gcs_download(gcs_video_uri, mp4_path)
            except GCSDownloadError:
                st.update({"status": "skipped", "reason": "reprocess_video_not_found"})
                return st

            cap = cv2.VideoCapture(str(mp4_path))
            if not cap.isOpened():
                st.update({"status": "skipped", "reason": "reprocess_video_corrupt"})
                return st
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if not fps or fps <= 0 or frame_count <= 0:
                st.update({"status": "skipped", "reason": "reprocess_video_bad_meta"})
                return st

            clip_to_process = mp4_path
            clip_duration = frame_count / fps
            st["duration"] = round(clip_duration, 2)
            clip_person_prefix = "person"

        else:  # Original scraping logic... (unchanged)
            if DEBUG: logging.debug("Processing %s (%.1fs)", vid, dur)
            mp4 = tmp / f"{vid}.mp4"
            try:
                fmt = get_video_format_with_retry(vid, cookies_file=cookies_file)
            except RateLimitException:
                st["reason"] = "rate_limit_persistent";
                return st
            except yt_dlp.DownloadError as e:
                msg = str(e).lower()
                if "private" in msg:
                    st["reason"] = "private_video"
                elif "copyright" in msg:
                    st["reason"] = "copyright_blocked"
                elif "unavailable" in msg:
                    st["reason"] = "video_unavailable"
                elif "age restricted" in msg:
                    st["reason"] = "age_restricted"
                else:
                    st["reason"] = f"yt_dlp_error: {str(e).splitlines()[0].replace('ERROR: ', '')}"
                return st
            if not fmt: st["reason"] = "no_progressive_mp4"; return st
            url = fmt["url"]
            try:
                download_clip_direct(url, start, dur, mp4)
            except subprocess.CalledProcessError:
                st["reason"] = "ffmpeg_download_failed";
                return st
            cap = cv2.VideoCapture(str(mp4))
            if not cap.isOpened(): st["reason"] = "video_file_corrupt_or_unreadable"; return st
            tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(tot // SAMPLE_FRAMES_FOR_FACE, 1)
            idxs = [i * step for i in range(SAMPLE_FRAMES_FOR_FACE)]
            person_validity_tracker = collections.defaultdict(list)
            has_face_in_sample = []
            for ix in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, ix)
                r, frm = cap.read()
                if not r: has_face_in_sample.append(False); continue
                bboxes = detect_faces(frm, margin=bbox)
                has_face_in_sample.append(len(bboxes) > 0)
                bboxes.sort(key=lambda b: b[0])
                for i, (x0, y0, x1, y1) in enumerate(bboxes):
                    person_idx = i + 1
                    width, height = x1 - x0, y1 - y0
                    is_valid = (max(width, height) >= 150) and (min(width, height) >= 110)
                    person_validity_tracker[person_idx].append(is_valid)
            cap.release()
            clip_to_process = None
            if sum(has_face_in_sample) < 3:
                st["reason"] = f"face_gate_failed_hits_{sum(has_face_in_sample)}_of_5"
            else:
                valid_person_indices = [p_idx for p_idx, v_list in person_validity_tracker.items() if
                                        all(v_list) and len(v_list) >= 3]
                if valid_person_indices:
                    clip_to_process = mp4
                    clip_duration = dur
                    clip_person_indices = valid_person_indices
                    clip_person_prefix = "person"
                    st.update({"status": "parsed", "reason": None, "valid_persons_count": len(valid_person_indices)})
                else:
                    st["reason"] = "no_valid_persons_found"
                    if DEBUG: logging.info(f"[{vid}] Primary gate failed. Attempting backup intro-clip strategy...")
                    is_valid, backup_mp4_path, backup_reason = find_alternative_intro_clip(vid, url, bbox, tmp)
                    if is_valid:
                        logging.info(f"[{vid}] Backup strategy SUCCEEDED. Processing 10s intro clip.")
                        clip_to_process = backup_mp4_path
                        clip_duration = 10.0
                        clip_person_indices = [1]
                        clip_person_prefix = "backup_person"
                        st.update({"status": "parsed_backup", "reason": None, "duration": clip_duration,
                                   "valid_persons_count": 1})
                    else:
                        st["reason"] = f"no_valid_persons_found_and_{backup_reason}"
            if not clip_to_process: return st

        # 5. Process and Upload Frames (Common logic for both modes)
        try:
            fdir = tmp / "frames_full"
            extract_frames(clip_to_process, fdir, QA_FRAMES_PER_CLIP, clip_duration)
        except subprocess.CalledProcessError as e:
            st.update({"status": "skipped", "reason": "ffmpeg_processing_failed"})
            return st

        fdir_cropped = tmp / "frames_cropped"
        fdir_cropped.mkdir()

        ### SPEED-UP CHANGE ### - Batched face detection
        frame_paths = sorted(fdir.glob("*.png"))
        images = [cv2.imread(str(p)) for p in frame_paths]
        # Filter out any frames that failed to load
        valid_frames = [(p, img) for p, img in zip(frame_paths, images) if img is not None]
        if not valid_frames:
            st.update({"status": "skipped", "reason": "no_valid_frames_extracted"})
            return st

        valid_paths, valid_images = zip(*valid_frames)

        # Run YOLO prediction on all frames at once
        batch_bboxes = detect_faces(list(valid_images), margin=bbox)

        for i, frame_path in enumerate(valid_paths):
            img = valid_images[i]
            frame_h, frame_w = img.shape[:2]
            bboxes = batch_bboxes[i]  # Get pre-computed bboxes for this frame
            bboxes.sort(key=lambda b: b[0])

            for j, (x0, y0, x1, y1) in enumerate(bboxes):
                person_idx = j + 1
                width, height = x1 - x0, y1 - y0
                center_x, center_y = x0 + width / 2, y0 + height / 2
                side_length = max(width, height)
                sq_x0 = max(0, int(center_x - side_length / 2))
                sq_y0 = max(0, int(center_y - side_length / 2))
                sq_x1 = min(frame_w, int(center_x + side_length / 2))
                sq_y1 = min(frame_h, int(center_y + side_length / 2))
                cropped_img = img[sq_y0:sq_y1, sq_x0:sq_x1]
                if cropped_img.size == 0: continue
                new_filename = f"{clip_person_prefix}_{person_idx}_{frame_path.stem}.png"
                cv2.imwrite(str(fdir_cropped / new_filename), cropped_img)
        shutil.rmtree(fdir)

        vb, ab, fb = buckets
        if not reprocess_frames_only:
            mp3 = tmp / f"{vid}.mp3"
            extract_audio(clip_to_process, mp3)
            gcs_upload(clip_to_process, f"{vb.rstrip('/')}/{vid}.mp4")
            gcs_upload(mp3, f"{ab.rstrip('/')}/{vid}.mp3")

        for f in fdir_cropped.glob("*.png"):
            try:
                person_idx_str = f.name.split('_')[1]
            except IndexError:
                continue
            gcs_path = f"{fb.rstrip('/')}/real/external_youtube_avspeech/{vid}_{clip_person_prefix}_{person_idx_str}/{f.name}"
            gcs_upload(f, gcs_path)

        if reprocess_frames_only:
            st.update({"status": "reprocessed_frames", "reason": None})

        return st

    except (GCSUploadError, GCSDownloadError) as e:
        # Don't raise the exception here to allow other parallel jobs to continue
        st.update({"status": "skipped", "reason": f"gcs_error_{type(e).__name__}"})
        return st
    except Exception as e:
        st["reason"] = f"unexpected_error_{type(e).__name__}"
        return st
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ──────────────────────────
# 🏃 CLI entry-point
# ──────────────────────────
def main():
    ap = argparse.ArgumentParser(description="AVSpeech YouTube clip scraper · v2.8 (Optimized)")
    ap.add_argument("source",
                    help="Path to the input CSV file (for scraping) or GCS URI like gs://bucket-name (for reprocessing).")
    ap.add_argument("--bbox-margin", type=int, default=DEFAULT_BBOX_MARGIN)
    ap.add_argument("--dry-run", action="store_true",
                    help="Process only a few rows and send all artifacts to the VIDEO bucket.")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--cookies", type=str, default=None, help="Path to a netscape-style cookies.txt file for scraping.")
    ap.add_argument("--reprocess-frames-only", action="store_true",
                    help="Reprocess frames from existing videos. 'source' must be a GCS URI.")
    ### SPEED-UP CHANGE ### - Add argument for number of worker processes
    ap.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                    help="Number of parallel worker processes to use.")

    args = ap.parse_args()

    global DEBUG
    DEBUG = args.debug
    logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")

    print("\n" + "─" * 60)
    print("✨ AVSpeech Scraper v2.8 (Optimized) ✨")
    print(f"⚙️  Using up to {args.workers} parallel processes.")

    if args.reprocess_frames_only:
        print("🖼️  Running in REPROCESS FRAMES ONLY mode.")
        print(f"    Source videos will be read from: {args.source}")
    # ... (rest of the intro print statements are the same)
    else:
        if args.cookies:
            print(f"🍪 Using cookies from: {args.cookies}")
        else:
            print("⚠️ Running without cookies. May be blocked by YouTube for high-volume scraping.")
        print(f"🐌 Adding random delay between {MIN_REQUEST_DELAY_S}-{MAX_REQUEST_DELAY_S}s to avoid rate-limiting.")
    print("─" * 60 + "\n")

    buckets = (GCS_VIDEO_BUCKET,) * 3 if args.dry_run else (GCS_VIDEO_BUCKET, GCS_AUDIO_BUCKET, GCS_FRAMES_BUCKET)
    if args.dry_run: print("[Dry-run] Using VIDEO bucket for audio & frames too.")

    rows = []
    if args.reprocess_frames_only:
        if not args.source.startswith("gs://"):
            raise ValueError("In reprocess mode, 'source' must be a GCS URI (e.g., gs://my-bucket).")

        print(f"Listing videos from GCS source: {args.source} (this may take a moment)...")
        # Initialize the GCS client in the main process for listing blobs
        initialize_gcs_client()
        bucket_name = args.source[5:].split('/')[0]
        prefix_parts = args.source[5:].split('/', 1)
        prefix = prefix_parts[1] if len(prefix_parts) > 1 else ""

        # Check for existing statuses to avoid reprocessing already completed videos
        status = load_status()

        blobs = gcs_client.list_blobs(bucket_name, prefix=prefix)
        all_vids = [blob.name.split('/')[-1].replace('.mp4', '') for blob in blobs if blob.name.endswith(".mp4")]

        # Filter out vids that have already been successfully reprocessed
        rows_to_process = []
        for vid in all_vids:
            if status.get(vid, {}).get("status") != "reprocessed_frames":
                rows_to_process.append((vid,))  # Must be a tuple for mapping

        print(f"Found {len(all_vids)} total videos. {len(rows_to_process)} need reprocessing.")
        rows = rows_to_process
    else:  # Original scraping logic... (unchanged)
        status = load_status()
        with open(args.source) as fh:
            for i, (vid, s, e, *_) in enumerate(csv.reader(fh)):
                if not YT_ID_RE.match(vid):
                    if vid not in status: append_status(
                        {"video_id": vid, "status": "skipped", "reason": "invalid_id"}); continue
                if vid in status: continue
                s_f, e_f = float(s), float(e)
                if e_f - s_f < MIN_DURATION:
                    if vid not in status: append_status(
                        {"video_id": vid, "status": "skipped", "reason": "too_short"}); continue
                rows.append((vid, s_f, e_f))
        report_lengths(rows)
        print(f"Loaded {len(status)} previous statuses. Need to process {len(rows)} new clips …")

    if not rows:
        print("No new clips to process. Exiting.")
        return

    t0 = time.perf_counter()
    ok_primary = 0;
    ok_backup = 0;
    ok_reprocessed = 0
    skipped_counts = collections.Counter()

    ### SPEED-UP CHANGE ### - Use a multiprocessing Pool
    # Create a partial function with fixed arguments
    worker_func = partial(process_row_worker, bbox=args.bbox_margin, buckets=buckets,
                          cookies_file=args.cookies, reprocess_frames_only=args.reprocess_frames_only)

    try:
        # Use imap_unordered to get results as they complete, which is great for tqdm
        with multiprocessing.Pool(processes=args.workers) as pool:
            pbar = tqdm(total=len(rows), desc="Clips", unit="clip")
            for rec in pool.imap_unordered(worker_func, rows):
                append_status(rec)
                if rec["status"] == "parsed":
                    ok_primary += 1
                elif rec["status"] == "parsed_backup":
                    ok_backup += 1
                elif rec["status"] == "reprocessed_frames":
                    ok_reprocessed += 1
                else:
                    reason = rec.get("reason", "unknown")
                    skipped_counts[reason] += 1
                    if not DEBUG: tqdm.write(f"[Skipped] {rec['video_id']} – {reason}")
                pbar.update(1)
            pbar.close()

    except KeyboardInterrupt:
        print("\n🛑 User interrupted. Exiting.")
    finally:
        print("\n" + "=" * 20 + " Run Summary " + "=" * 20)
        total_attempted = ok_primary + ok_backup + ok_reprocessed + sum(skipped_counts.values())
        if total_attempted == 0:
            print("No new clips were processed.")
        else:
            print(f"Attempted to process {total_attempted} clips in {(time.perf_counter() - t0) / 60:.1f} min.")
            if ok_primary > 0: print(f"  ✅ Parsed (Primary): {ok_primary}")
            if ok_backup > 0: print(f"  ✨ Parsed (Backup):  {ok_backup}")
            if ok_reprocessed > 0: print(f"  🖼️  Reprocessed Frames: {ok_reprocessed}")
            print(f"  ➡️ Total Skipped:      {sum(skipped_counts.values())}")

        if skipped_counts:
            print("\n" + "-" * 18 + " Skip Reason Breakdown " + "-" * 18)
            for reason, count in skipped_counts.most_common():
                reason_short = (reason[:75] + '...') if len(reason) > 78 else reason
                print(f"  {count:>4} | {reason_short}")
        print("=" * 53)


if __name__ == "__main__":
    main()
