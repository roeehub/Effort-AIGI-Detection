"""youtube_avspeech_scraper.py
================================================
Scrape, validate and upload clipped talking-head segments defined by an
**AVSpeech-style CSV** (video_id,start_sec,end_sec,x_norm,y_norm).

v2.2  (2025-08-08) — yt-dlp backend · relaxed face-gate · graceful skips · backup intro-clip logic
------------------------------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse, csv, json, logging, re, subprocess, tempfile, time
import collections
import shutil
import random
from pathlib import Path
from typing import Dict, Tuple, List

import cv2  # type: ignore
import yt_dlp  # type: ignore
from google.cloud import storage  # type: ignore
from tqdm import tqdm  # type: ignore

# suppress logs from yt-dlp
logging.getLogger("yt_dlp").setLevel(logging.WARNING)

# ──────────────────────────
# 📍 Configuration
# ──────────────────────────
GCS_VIDEO_BUCKET = "gs://youtube-real-videos"
GCS_AUDIO_BUCKET = "gs://youtube-real-audio-clips"
GCS_FRAMES_BUCKET = "gs://df40-frames"

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


# ──────────────────────────
# 🐛 Debugging Tool
# ──────────────────────────
def debug_video(video_id: str, csv_path: str, bbox_margin: int = DEFAULT_BBOX_MARGIN):
    """
    Runs the full processing logic for a single video ID and generates a
    local debug report instead of uploading to GCS.
    """
    print(f"\n--- 🐛 Debugging Video: {video_id} 🐛 ---")
    debug_dir = Path("debug_output") / video_id
    debug_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Debug artifacts will be saved to: {debug_dir.resolve()}")

    # 1. Find video info in CSV
    start, end = None, None
    with open(csv_path) as fh:
        for row in csv.reader(fh):
            if row[0] == video_id:
                start, end = float(row[1]), float(row[2])
                break
    if start is None:
        print(f"[ERROR] Video ID '{video_id}' not found in '{csv_path}'.")
        return

    dur = end - start
    print(f"[*] Found clip info: Start={start:.2f}s, End={end:.2f}s, Duration={dur:.2f}s")

    # 2. Download clip to a temporary directory
    tmp = Path(tempfile.mkdtemp(prefix="ytclip_debug_"))
    mp4 = tmp / f"{video_id}.mp4"
    try:
        print("[*] Downloading video clip...")
        fmt = best_progressive_mp4(video_id)
        if not fmt:
            print("[VERDICT] SKIPPED: No progressive MP4 stream found.")
            return
        download_clip_direct(fmt["url"], start, dur, mp4)
        print("[*] Download complete.")

        # 3. Perform the 5-frame analysis
        print("\n--- Analyzing 5 Sample Frames ---")
        cap = cv2.VideoCapture(str(mp4))
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(tot // SAMPLE_FRAMES_FOR_FACE, 1)
        idxs = [i * step for i in range(SAMPLE_FRAMES_FOR_FACE)]

        person_validity_tracker = collections.defaultdict(list)
        has_face_in_sample = []

        for i, ix in enumerate(idxs):
            print(f"\n[Sample {i + 1}/5, Frame Index: {ix}]")
            cap.set(cv2.CAP_PROP_POS_FRAMES, ix)
            r, frm = cap.read()
            if not r:
                print("  - Failed to read frame.")
                has_face_in_sample.append(False)
                continue

            # Draw on a copy of the frame
            annotated_frm = frm.copy()
            frame_h, frame_w = annotated_frm.shape[:2]

            bboxes = detect_faces(frm, margin=bbox_margin)
            has_face_in_sample.append(len(bboxes) > 0)
            print(f"  - Detected {len(bboxes)} face(s).")

            bboxes.sort(key=lambda b: b[0])  # Sort by x-coordinate

            for p_idx, (x0, y0, x1, y1) in enumerate(bboxes):
                person_id = p_idx + 1
                width, height = x1 - x0, y1 - y0
                is_valid = (max(width, height) >= 170) and (min(width, height) >= 130)

                validity_str = "VALID" if is_valid else "INVALID - TOO SMALL"
                person_validity_tracker[person_id].append(is_valid)
                print(
                    f"    - Person {person_id}: bbox=({x0},{y0},{x1},{y1}), size=({width}x{height}) -> {validity_str}")
                color = (0, 255, 0) if is_valid else (0, 0, 255)  # Green for valid, Red for invalid
                cv2.rectangle(annotated_frm, (x0, y0), (x1, y1), color, 2)
                label = f"P{person_id}: {width}x{height}"
                cv2.putText(annotated_frm, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # 2. Calculate and draw the final square crop area
                center_x, center_y = x0 + width / 2, y0 + height / 2
                side_length = max(width, height)
                sq_x0 = max(0, int(center_x - side_length / 2))
                sq_y0 = max(0, int(center_y - side_length / 2))
                sq_x1 = min(frame_w, int(center_x + side_length / 2))
                sq_y1 = min(frame_h, int(center_y + side_length / 2))

                # Draw the square box in a different color (e.g., Cyan)
                cv2.rectangle(annotated_frm, (sq_x0, sq_y0), (sq_x1, sq_y1), (255, 255, 0), 2, cv2.LINE_AA)

            frame_filename = debug_dir / f"sample_{i + 1}_frame_{ix}.jpg"
            cv2.imwrite(str(frame_filename), annotated_frm)
            print(f"  - Saved annotated frame to: {frame_filename.name}")

        cap.release()

        # 4. Final Verdict Logic
        print("\n--- Final Gate Analysis ---")

        # Rule 1 Check
        hits = sum(has_face_in_sample)
        print(f"[*] Rule 1 (Face Hits): {hits} of 5 samples had at least one face.")
        if hits < 3:
            print(f"[VERDICT] SKIPPED: Fails Rule 1 (needs at least 3 hits).")
            return

        # Rule 2 Check
        print("[*] Rule 2 (Person Validity Tracker):")
        if not person_validity_tracker:
            print("  - No persons were ever detected.")
        for p_id, v_list in person_validity_tracker.items():
            print(f"  - Person {p_id}: Appeared {len(v_list)} times. Was always valid: {all(v_list)}.")

        valid_person_indices = []
        for person_idx, validity_list in person_validity_tracker.items():
            if all(validity_list) and len(validity_list) >= 3:
                valid_person_indices.append(person_idx)

        print(f"[*] Final list of valid persons (appeared >=3 times and were never too small): {valid_person_indices}")

        # Rule 3 Check
        if not valid_person_indices:
            print(f"[VERDICT] SKIPPED: Fails Rule 3 (no single person was consistently valid).")
            return

        print(f"\n[VERDICT] PASSED: Found {len(valid_person_indices)} valid person(s).")
        print(f"[*] This clip would be processed, and frames for Person(s) {valid_person_indices} would be saved.")

    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}")
        logging.error("Debug function failed", exc_info=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        print("\n--- Debug Session Complete ---")


# ──────────────────────────
# 🧰 sub-process helper
# ──────────────────────────
def _run(cmd: List[str], *, show_cmd: bool = False):
    if show_cmd or DEBUG:
        logging.debug("RUN: %s", " ".join(cmd))
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=None if DEBUG else subprocess.PIPE,
            stderr=None if DEBUG else subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        logging.error("Command failed: %s", " ".join(cmd))
        if e.stdout: logging.error("stdout:\n%s", e.stdout.decode(errors="ignore"))
        if e.stderr: logging.error("stderr:\n%s", e.stderr.decode(errors="ignore"))
        raise


def download_clip_direct(url: str, start: float, dur: float, out: Path):
    _run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start}", "-i", url,
          "-t", f"{dur}", "-c", "copy", str(out)])


def extract_audio(mp4: Path, out: Path):
    """Re-encodes the audio stream to MP3 format."""
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
          "-vn", "-acodec", "libmp3lame", "-ab", "192k", str(out)])


def extract_frames(mp4: Path, dir_: Path, n: int, duration: float):
    """
    Grab exactly *n* evenly-spaced frames by setting fps = n / duration.
    We compute the value in Python so the command needs no shell expansion.
    """
    dir_.mkdir(parents=True, exist_ok=True)
    fps = max(n / duration, 0.001)  # guard against div-by-zero
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
          "-vf", f"fps={fps}", str(dir_ / "%04d.jpg")])


# ──────────────────────────
# 🖼️ YOLO face helper
# ──────────────────────────
from ultralytics import YOLO  # noqa

_model_cache = {}


def detect_faces(img, margin=DEFAULT_BBOX_MARGIN) -> List[Tuple[int, int, int, int]]:
    """Detects all faces and returns a list of bounding boxes."""
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
    res = _model_cache[YOLO_VARIANT].predict(img, conf=conf, iou=0.4, verbose=False)[0]

    if res.boxes.shape[0] == 0:
        return []

    boxes = []
    h, w = img.shape[:2]
    for box in res.boxes.xyxy.cpu().numpy():
        x0, y0, x1, y1 = box
        x0 -= margin
        y0 -= margin
        x1 += margin
        y1 += margin
        x0, y0 = int(max(0, x0)), int(max(0, y0))
        x1, y1 = int(min(w, x1)), int(min(h, y1))
        boxes.append((x0, y0, x1, y1))

    return boxes


# ──────────────────────────
# ☁️ GCS helper
# ──────────────────────────
gcs_client = storage.Client()


def gcs_upload(local: Path, uri: str):
    if not uri.startswith("gs://"): raise ValueError("GCS URI must start with gs://")
    bucket, *rest = uri[5:].split("/", 1)
    blob = rest[0] if rest else ""
    gcs_client.bucket(bucket).blob(blob).upload_from_filename(str(local))


# ──────────────────────────
# 📄 status helpers
# ──────────────────────────
STATUS_FILE = Path("status.jsonl")


def load_status() -> Dict[str, dict]:
    if not STATUS_FILE.exists(): return {}
    with STATUS_FILE.open() as fh: return {j["video_id"]: j for j in (json.loads(l) for l in fh)}


def append_status(rec: dict):
    with STATUS_FILE.open("a") as fh: fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ──────────────────────────
# ℹ️ CSV report
# ──────────────────────────
def report_lengths(rows: List[Tuple[str, float, float]]):
    val = [r for r in rows if (r[2] - r[1]) >= MIN_DURATION]
    n = len(val)
    est = int(n * 0.85)
    print(f"\n[Report] {n} clips have duration ≥ {MIN_DURATION}s.")
    print(f"[Report] Roughly {est} expected after face-gate.\n")


# ──────────────────────────
# 🎥 yt-dlp stream picker & robust retrier
# ──────────────────────────
class RateLimitException(Exception):
    """Custom exception for when all retries fail due to rate limiting."""
    pass


def best_progressive_mp4(video_id: str):
    opts = {
        "quiet": True,
        "skip_download": True,
        "forcejson": True,
        "no_warnings": True,  # Suppress warnings
        "no_color": True,  # Remove color codes
        "noprogress": True,  # Ensure no progress bars
        "logger": None  # Disable logger
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(video_id, download=False)
    prog = [f for f in info["formats"]
            if f.get("ext") == "mp4" and f.get("vcodec") != "none" and f.get("acodec") != "none"]
    prog.sort(key=lambda f: (f.get("height") or 0, f.get("tbr") or 0), reverse=True)
    return prog[0] if prog else None


def get_video_format_with_retry(video_id: str):
    """
    Calls best_progressive_mp4 with a retry mechanism for rate limiting.
    Implements exponential backoff with jitter.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return best_progressive_mp4(video_id)
        except yt_dlp.DownloadError as e:
            if "HTTP Error 429" in str(e):
                if attempt < MAX_RETRIES - 1:
                    backoff = INITIAL_BACKOFF_S * (2 ** attempt)
                    jitter = random.uniform(0, 1)
                    sleep_time = backoff + jitter
                    logging.warning(
                        f"Rate limit hit for {video_id} on attempt {attempt + 1}. "
                        f"Retrying in {sleep_time:.2f} seconds..."
                    )
                    time.sleep(sleep_time)
                    continue
                else:
                    logging.error(f"All {MAX_RETRIES} retries failed for {video_id} due to rate limiting.")
                    raise RateLimitException(f"Rate limit persisted for {video_id}") from e
            else:
                raise e
    return None


# ──────────────────────────
# ⭐ NEW: Backup intro-clip finder
# ──────────────────────────
def find_alternative_intro_clip(
        vid: str, url: str, bbox: int, tmp_dir: Path
) -> Tuple[bool, Path | None, str]:
    """
    Backup strategy: Download the first 11s of a video and check for a
    consistent, single, large face. This is a "short-circuiting" check.

    Returns: (is_valid, path_to_mp4_if_valid, reason_string)
    """
    start_time, duration = 1.0, 10.0
    backup_mp4 = tmp_dir / f"{vid}_backup.mp4"
    try:
        download_clip_direct(url, start_time, duration, backup_mp4)
    except subprocess.CalledProcessError:
        return False, None, "backup_download_failed"

    cap = cv2.VideoCapture(str(backup_mp4))
    if not cap.isOpened():
        return False, None, "backup_video_read_failed"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or total_frames < 1:
        cap.release()
        return False, None, "backup_video_meta_failed"

    # Create a list of 20 timestamps to check (every 0.5s for 10s)
    timestamps = [i * 0.5 for i in range(20)]
    frame_indices = [int(t * fps) for t in timestamps]
    # Filter out any indices that might exceed the clip's actual frame count
    frame_indices = [idx for idx in frame_indices if idx < total_frames]

    # Critical: Randomize the order to enable fast short-circuiting
    random.shuffle(frame_indices)

    if len(frame_indices) < 20:  # Ensure we have enough frames to check
        logging.warning(f"[{vid}] Backup clip too short or low FPS, only found {len(frame_indices)} samples.")
        cap.release()
        return False, None, "backup_not_enough_samples"

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            # This frame failed, so the whole clip fails the check
            cap.release()
            return False, None, f"backup_read_frame_{frame_idx}_failed"

        bboxes = detect_faces(frame, margin=bbox)

        # Rule 1: Must be exactly ONE face
        if len(bboxes) != 1:
            cap.release()
            return False, None, f"backup_found_{len(bboxes)}_faces_at_frame_{i + 1}"

        # Rule 2: The single face must meet size requirements
        x0, y0, x1, y1 = bboxes[0]
        width, height = x1 - x0, y1 - y0
        if not (max(width, height) >= 200 and min(width, height) >= 150):
            cap.release()
            return False, None, f"backup_face_too_small_{width}x{height}_at_frame_{i + 1}"

    # If the loop completes, all 20 random checks passed.
    cap.release()
    return True, backup_mp4, "backup_clip_valid"


# ─────────────────
# 🔁 row processor
# ──────────────────────────
def process_row(vid: str, start: float, end: float, bbox: int, buckets: Tuple[str, str, str]):
    dur = end - start
    tmp = Path(tempfile.mkdtemp(prefix="ytclip_"))
    mp4 = tmp / f"{vid}.mp4"
    st = {"video_id": vid, "status": "skipped", "reason": "", "duration": round(dur, 2)}
    if DEBUG: logging.debug("Processing %s (%.1fs)", vid, dur)

    try:
        try:
            fmt = get_video_format_with_retry(vid)
        except RateLimitException:
            st["reason"] = "rate_limit_persistent"
            return st
        except yt_dlp.DownloadError as e:
            msg = str(e).lower()
            st["reason"] = "private_video" if "private" in msg else "video_unavailable"
            return st

        if not fmt:
            st["reason"] = "no_progressive_mp4"
            return st
        url = fmt["url"]

        # --- Primary Processing Logic (from AVSpeech timestamps) ---
        download_clip_direct(url, start, dur, mp4)

        cap = cv2.VideoCapture(str(mp4))
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(tot // SAMPLE_FRAMES_FOR_FACE, 1)
        idxs = [i * step for i in range(SAMPLE_FRAMES_FOR_FACE)]

        person_validity_tracker = collections.defaultdict(list)
        has_face_in_sample = []

        for ix in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, ix)
            r, frm = cap.read()
            if not r:
                has_face_in_sample.append(False)
                continue
            bboxes = detect_faces(frm, margin=bbox)
            has_face_in_sample.append(len(bboxes) > 0)
            bboxes.sort(key=lambda b: b[0])
            for i, (x0, y0, x1, y1) in enumerate(bboxes):
                person_idx = i + 1
                width, height = x1 - x0, y1 - y0
                is_valid = (max(width, height) >= 170) and (min(width, height) >= 130)
                person_validity_tracker[person_idx].append(is_valid)
        cap.release()

        if sum(has_face_in_sample) < 3:
            st["reason"] = f"face_gate_failed_hits_{sum(has_face_in_sample)}_of_5"
        else:
            valid_person_indices = []
            for person_idx, validity_list in person_validity_tracker.items():
                if all(validity_list) and len(validity_list) >= 3:
                    valid_person_indices.append(person_idx)

            if not valid_person_indices:
                st["reason"] = "no_valid_persons_found"
            else:
                st["valid_persons_count"] = len(valid_person_indices)
                # Primary logic passed, proceed to process THIS clip
                # This `else` block now contains the success path for the primary clip.
                # We rename `mp4` to `clip_to_process` for clarity.
                clip_to_process = mp4
                clip_duration = dur
                clip_person_indices = valid_person_indices
                clip_person_prefix = "person"  # for filename
                st["status"] = "parsed"
                st["reason"] = None

        # --- MODIFIED: Check if primary logic failed and if we should try the backup ---
        if st["status"] == "skipped" and st["reason"] == "no_valid_persons_found":
            if DEBUG: logging.info(f"[{vid}] Primary gate failed. Attempting backup intro-clip strategy...")

            is_valid, backup_mp4_path, backup_reason = find_alternative_intro_clip(vid, url, bbox, tmp)

            if is_valid:
                logging.info(f"[{vid}] Backup strategy SUCCEEDED. Processing 10s intro clip.")
                # Success! Overwrite variables to process the backup clip instead
                clip_to_process = backup_mp4_path
                clip_duration = 10.0
                clip_person_indices = [1]  # Backup logic guarantees one person, index 1
                clip_person_prefix = "backup_person"  # for filename
                st["status"] = "parsed_backup"
                st["reason"] = None
                st["duration"] = clip_duration
                st["valid_persons_count"] = 1
            else:
                # Backup also failed, update reason and let it be skipped.
                st["reason"] = f"no_valid_persons_found_and_{backup_reason}"
                return st  # Exit early

        # --- Final Processing Block (runs only if status is 'parsed' or 'parsed_backup') ---
        if st["status"].startswith("parsed"):
            # QA frame
            cap = cv2.VideoCapture(str(clip_to_process))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # First frame of the valid clip
            r, f0 = cap.read()
            cap.release()
            if r: cv2.imwrite(str(QA_DIR / f"{vid}.jpg"), f0)

            # Audio and Frame Extraction
            mp3 = tmp / f"{vid}.mp3"
            extract_audio(clip_to_process, mp3)

            fdir = tmp / "frames_full"
            extract_frames(clip_to_process, fdir, QA_FRAMES_PER_CLIP, clip_duration)

            fdir_cropped = tmp / "frames_cropped"
            fdir_cropped.mkdir()

            for frame_path in sorted(fdir.glob("*.jpg")):
                img = cv2.imread(str(frame_path))
                if img is None: continue
                frame_h, frame_w = img.shape[:2]

                bboxes = detect_faces(img, margin=bbox)
                bboxes.sort(key=lambda b: b[0])

                for i, (x0, y0, x1, y1) in enumerate(bboxes):
                    person_idx = i + 1
                    if person_idx in clip_person_indices:
                        width, height = x1 - x0, y1 - y0
                        center_x, center_y = x0 + width / 2, y0 + height / 2
                        side_length = max(width, height)
                        sq_x0 = max(0, int(center_x - side_length / 2))
                        sq_y0 = max(0, int(center_y - side_length / 2))
                        sq_x1 = min(frame_w, int(center_x + side_length / 2))
                        sq_y1 = min(frame_h, int(center_y + side_length / 2))
                        cropped_img = img[sq_y0:sq_y1, sq_x0:sq_x1]

                        if cropped_img.size == 0: continue
                        new_filename = f"{clip_person_prefix}_{person_idx}_{frame_path.name}"
                        cv2.imwrite(str(fdir_cropped / new_filename), cropped_img)

            shutil.rmtree(fdir)

            # Upload
            vb, ab, fb = buckets
            gcs_upload(clip_to_process, f"{vb.rstrip('/')}/{vid}.mp4")
            gcs_upload(mp3, f"{ab.rstrip('/')}/{vid}.mp3")

            for f in fdir_cropped.glob("*.jpg"):
                try:
                    person_idx_str = f.name.split('_')[1]
                except IndexError:
                    continue
                gcs_path = f"{fb.rstrip('/')}/real/external_youtube_avspeech/{vid}_{clip_person_prefix}_{person_idx_str}/{f.name}"
                gcs_upload(f, gcs_path)

            return st  # Return the successful status dict

        # If we reach here, it means the clip was skipped for a reason other than 'no_valid_persons_found'
        return st

    except Exception as e:
        logging.error(f"Error processing {vid}: {e}", exc_info=DEBUG)
        st["reason"] = str(e)
        return st
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ──────────────────────────
# 🏃 CLI entry-point
# ──────────────────────────
def main():
    ap = argparse.ArgumentParser(description="AVSpeech YouTube clip scraper · v2.2")
    ap.add_argument("csv")
    ap.add_argument("--bbox-margin", type=int, default=DEFAULT_BBOX_MARGIN)
    ap.add_argument("--dry-run", action="store_true",
                    help="Process only first eight rows send everything to VIDEO bucket")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    global DEBUG
    DEBUG = args.debug
    logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%H:%M:%S")

    buckets = (GCS_VIDEO_BUCKET,) * 3 if args.dry_run else \
        (GCS_VIDEO_BUCKET, GCS_AUDIO_BUCKET, GCS_FRAMES_BUCKET)
    if args.dry_run:
        print("[Dry-run] Using VIDEO bucket for audio & frames too.")

    status = load_status()
    rows = []
    with open(args.csv) as fh:
        for i, (vid, s, e, *_) in enumerate(csv.reader(fh)):
            if vid in status: continue
            # if i >= 50:
            #     break
            if not YT_ID_RE.match(vid):
                append_status({"video_id": vid, "status": "skipped", "reason": "invalid_id"})
                continue
            s_f, e_f = float(s), float(e)
            if e_f - s_f < MIN_DURATION:
                append_status({"video_id": vid, "status": "skipped", "reason": "too_short"})
                continue
            rows.append((vid, s_f, e_f))

    report_lengths(rows)
    print(f"Need to process {len(rows)} clips …")

    t0 = time.perf_counter()
    ok = 0
    parsed_backup = 0
    for vid, s, e in tqdm(rows, desc="Clips", unit="clip"):
        rec = process_row(vid, s, e, args.bbox_margin, buckets)
        append_status(rec)
        if rec["status"] == "parsed":
            ok += 1
        elif rec["status"] == "parsed_backup":
            parsed_backup += 1
        else:
            tqdm.write(f"[Skipped] {vid} – {rec['reason']}")
    print(f"\nDone. Parsed {ok}/{len(rows)} clips via primary method.")
    if parsed_backup > 0:
        print(f"        rescued {parsed_backup} clips via backup method.")
    print(f"Total processed: {ok + parsed_backup}/{len(rows)} in {(time.perf_counter() - t0) / 60:.1f} min.")
    if args.dry_run:
        print("[Dry-run] Artefacts uploaded to video bucket – remember to delete them if not needed.")


if __name__ == "__main__":
    main()
