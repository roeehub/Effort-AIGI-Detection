# face_curator.py
"""
A multi-stage, human-in-the-loop tool for curating a high-quality face dataset.

Workflow:
1. `process`: Download videos and extract all potential person chunks into a local
   directory. No cross-video consolidation is performed.
2. `review`: Launch an interactive GUI to approve or disapprove each person chunk.
   Disapproved chunks are logged to a file.
3. `prune`: Delete all the disapproved chunks from the local directory.
4. `upload`: Upload the final, curated set of person chunks to a GCP bucket.
"""
from __future__ import annotations
import sys
import logging
import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import face_recognition
from sklearn.cluster import DBSCAN
from google.cloud import storage
from google.api_core.exceptions import NotFound
from skimage.metrics import structural_similarity as ssim

# Platform-specific imports for reading single characters from the terminal
try:
    # Unix-like systems (macOS, Linux)
    import tty
    import termios
except ImportError:
    # Windows
    try:
        import msvcrt
    except ImportError:
        msvcrt = None

# ──────────────────────────
# 📍 Default Configuration
# ──────────────────────────
MIN_FACE_SIZE = (80, 80)
MIN_FRAMES_PER_CHUNK = 16
MAX_FRAMES_PER_CHUNK = 32
STATIC_FACE_THRESHOLD = 0.5
YOLO_MODEL_PATH = "/Users/roeedar/Library/Application Support/JetBrains/PyCharmCE2024.2/scratches/yolov8s-face.pt"
YOLO_CONF_THRESHOLD = 0.3
DBSCAN_EPS = 0.4
MODEL_IMG_SIZE = 224
YOLO_BBOX_MARGIN = 20
STATIC_FRAME_COUNT_THRESHOLD = 5
STATIC_BOX_DISTANCE_THRESHOLD = 10.0
BLUR_THRESHOLD = 100.0
STATIC_CONTENT_SSIM_THRESHOLD = 0.98
CLUSTER_COMPACTNESS_THRESHOLD = 0.35
DISAPPROVED_FILENAME = "disapproved.txt"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])


# ──────────────────────────
# 📦 Data Structures & Helpers
# ──────────────────────────
@dataclass
class Face:
    frame_image: np.ndarray
    encoding: np.ndarray
    frame_number: int
    box: tuple[int, int, int, int]


def _get_char():
    """Reads a single character from the user's terminal without waiting for Enter."""
    if 'termios' in sys.modules:
        # Unix-like system
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    elif msvcrt:
        # Windows
        return msvcrt.getch().decode('utf-8')
    else:
        # Fallback for unknown systems
        return sys.stdin.read(1)


def is_blurry(image: np.ndarray, threshold: float) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def sparsely_sample_frames(frames: list, num_samples: int) -> list:
    if len(frames) <= num_samples: return frames
    indices = np.linspace(0, len(frames) - 1, num=num_samples, dtype=int)
    return [frames[i] for i in indices]


def draw_box_on_image(image: np.ndarray, box: tuple, color=(0, 0, 255), thickness=2) -> np.ndarray:
    img_copy = image.copy()
    x0, y0, x1, y1 = map(int, box)
    return cv2.rectangle(img_copy, (x0, y0), (x1, y1), color, thickness)


# ──────────────────────────
# 🚀 STAGE 1: Process
# ──────────────────────────
class VideoProcessor:
    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        self.output_dir = Path(self.config.output_dir)

    def _filter_static_frames_from_chunk(self, faces: list[Face]) -> list[Face]:
        if len(faces) < self.config.static_frame_count: return faces
        box_clusters = defaultdict(list)
        for face in faces:
            x0, y0, x1, y1 = face.box
            center_x, center_y = int((x0 + x1) / 20), int((y0 + y1) / 20)
            box_clusters[(center_x, center_y)].append(face)

        potential_static = [c for c in box_clusters.values() if len(c) >= self.config.static_frame_count]
        if not potential_static: return faces

        true_static_centers = []
        for cluster in potential_static:
            img1_gray = cv2.cvtColor(cluster[0].frame_image, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(cluster[-1].frame_image, cv2.COLOR_BGR2GRAY)
            if ssim(img1_gray, img2_gray) > self.config.ssim_threshold:
                boxes = np.array([f.box for f in cluster])
                avg_center = np.mean([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
                true_static_centers.append(avg_center)

        if not true_static_centers: return faces

        dynamic_faces = []
        for face in faces:
            is_static = False
            face_center = np.array([(face.box[0] + face.box[2]) / 2, (face.box[1] + face.box[3]) / 2])
            for static_center in true_static_centers:
                if np.linalg.norm(face_center - static_center) < self.config.static_frame_dist:
                    is_static = True
                    break
            if not is_static:
                dynamic_faces.append(face)
        return dynamic_faces

    def _extract_faces_from_video(self, video_path: Path) -> list[Face]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.error(f"Could not open video file: {video_path}");
            return []

        extracted_faces, frame_num = [], 0
        frame_h, frame_w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            results = self.yolo_model.predict(frame, conf=self.config.yolo_conf, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for box in boxes:
                x0, y0, x1, y1 = box
                x0, y0 = max(0, x0 - self.config.margin), max(0, y0 - self.config.margin)
                x1, y1 = min(frame_w, x1 + self.config.margin), min(frame_h, y1 + self.config.margin)
                original_box = (int(x0), int(y0), int(x1), int(y1))
                w, h = x1 - x0, y1 - y0
                if w < self.config.min_face_size[0] or h < self.config.min_face_size[1]: continue

                center_x, center_y, side = x0 + w / 2, y0 + h / 2, max(w, h)
                sq_x0, sq_y0 = max(0, int(center_x - side / 2)), max(0, int(center_y - side / 2))
                sq_x1, sq_y1 = min(frame_w, int(center_x + side / 2)), min(frame_h, int(center_y + side / 2))
                face_crop = frame[sq_y0:sq_y1, sq_x0:sq_x1]
                if face_crop.size == 0: continue

                final_face = cv2.resize(face_crop, (self.config.img_size, self.config.img_size),
                                        interpolation=cv2.INTER_AREA)
                if is_blurry(final_face, self.config.blur_threshold): continue

                face_crop_rgb = cv2.cvtColor(final_face, cv2.COLOR_BGR2RGB)
                locs = [(0, self.config.img_size, self.config.img_size, 0)]
                encodings = face_recognition.face_encodings(face_crop_rgb, known_face_locations=locs)

                if encodings:
                    extracted_faces.append(Face(final_face, encodings[0], frame_num, original_box))
            frame_num += 1
        cap.release()
        return extracted_faces

    def process_and_save_video(self, video_path: Path):
        logging.info(f"Processing video: {video_path.name}")
        all_faces = self._extract_faces_from_video(video_path)
        if not all_faces:
            logging.warning(f"No valid faces found in {video_path.name}. Skipping.");
            return

        encodings = [face.encoding for face in all_faces]
        clt = DBSCAN(metric="euclidean", eps=self.config.dbscan_eps, min_samples=self.config.min_frames)
        clt.fit(encodings)
        persons_in_video = defaultdict(list)
        for face, label in zip(all_faces, clt.labels_):
            if label != -1: persons_in_video[label].append(face)

        logging.info(f"Identified {len(persons_in_video)} potential individuals in {video_path.name}.")
        for person_label, faces in persons_in_video.items():
            if len(faces) > 1:
                cluster_encodings = np.array([f.encoding for f in faces])
                centroid = np.mean(cluster_encodings, axis=0)
                avg_distance = np.mean(np.linalg.norm(cluster_encodings - centroid, axis=1))
                if avg_distance > self.config.compactness:
                    logging.warning(f"  - Discarding loose cluster {person_label} (Compactness: {avg_distance:.3f})")
                    continue

            cleaned_faces = self._filter_static_frames_from_chunk(faces)
            if len(cleaned_faces) < self.config.min_frames:
                logging.info(f"  - Discarding person {person_label} (only {len(cleaned_faces)} dynamic frames remain).")
                continue

            sorted_faces = sorted(cleaned_faces, key=lambda f: f.frame_number)
            sampled_faces = sparsely_sample_frames(sorted_faces, self.config.max_frames)

            video_stem = video_path.stem
            unique_id = random.randint(100000, 999999)
            chunk_name = f"{video_stem}_{unique_id}_person_{person_label}"
            chunk_dir = self.output_dir / chunk_name
            chunk_dir.mkdir(parents=True, exist_ok=True)

            logging.info(f"  - Saving person {person_label} to '{chunk_dir.name}' ({len(sampled_faces)} frames).")
            for i, face in enumerate(sampled_faces):
                cv2.imwrite(str(chunk_dir / f"{i:04d}.png"), face.frame_image)


def run_process(args):
    """Main function for the 'process' stage."""
    output_dir = Path(args.output_dir)
    temp_dir = output_dir / "temp_videos"
    output_dir.mkdir(exist_ok=True, parents=True)

    video_paths = []
    # --- START OF MODIFIED LOGIC ---
    if args.in_bucket:
        logging.info(f"Connecting to GCP bucket: '{args.in_bucket}'")
        temp_dir.mkdir(exist_ok=True, parents=True)
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(args.in_bucket)
            # Find all common video file types
            blobs = [b for b in bucket.list_blobs() if b.name.lower().endswith((".mp4", ".mov", ".avi"))]
            if args.sample:
                blobs = blobs[:args.sample]
            logging.info(f"Found {len(blobs)} videos to download...")
            for blob in blobs:
                dest = temp_dir / Path(blob.name).name
                logging.info(f"  -> Downloading {blob.name}...")
                blob.download_to_filename(dest)
                video_paths.append(dest)
        except Exception as e:
            logging.error(f"GCP download failed: {e}");
            sys.exit(1)

    elif args.in_folder:
        logging.info(f"Scanning local folder: '{args.in_folder}'")
        # Find all common video file types
        all_files = sorted([p for p in Path(args.in_folder).glob("*") if p.suffix.lower() in [".mp4", ".mov", ".avi"]])
        video_paths = all_files
        if args.sample:
            video_paths = all_files[:args.sample]
        logging.info(f"Found {len(video_paths)} videos to process.")
    # --- END OF MODIFIED LOGIC ---

    if not video_paths:
        logging.info("No videos to process. Exiting.")
        return

    processor = VideoProcessor(args)
    for path in video_paths:
        try:
            processor.process_and_save_video(path)
        except Exception as e:
            logging.error(f"FATAL ERROR processing {path.name}: {e}", exc_info=True)


# ──────────────────────────
# 🚀 STAGE 2: Review
# ──────────────────────────
def run_review(args):
    """Launch interactive review tool that uses the terminal for input."""
    root_dir = Path(args.dir)
    disapproved_path = root_dir / DISAPPROVED_FILENAME

    person_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    if not person_dirs:
        logging.info(f"No person directories found in '{root_dir}'.");
        return

    disapproved = set(line.strip() for line in open(disapproved_path)) if disapproved_path.exists() else set()

    total_dirs = len(person_dirs)
    for i, person_dir in enumerate(person_dirs):
        if person_dir.name in disapproved:
            logging.info(f"({i + 1}/{total_dirs}) Skipping already disapproved: {person_dir.name}")
            continue

        images = sorted(list(person_dir.glob("*.png")))
        if not images: continue

        sample_images = sparsely_sample_frames(images, 16)

        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        fig.suptitle(f"Reviewing ({i + 1}/{total_dirs}): {person_dir.name}", fontsize=12)
        axes = axes.flatten()

        for ax in axes: ax.axis('off')

        for ax, img_path in zip(axes, sample_images):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show(block=False)

        # --- FIX: Explicitly draw the canvas to prevent the first image from being blank ---
        # This forces matplotlib to render everything immediately, especially
        # important during the first loop's slow initialization.
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)  # A tiny pause is still good practice

        # Prompt in the terminal and wait for a single keypress
        print(f"Approve (y) / Disapprove (n) / Quit (q)? ", end="", flush=True)
        key_pressed = _get_char().lower()
        print(key_pressed)  # Echo the key back to the user

        plt.close(fig)  # Immediately close the plot window

        if key_pressed == 'n':
            disapproved.add(person_dir.name)
            with open(disapproved_path, "a") as f:
                f.write(f"{person_dir.name}\n")
            logging.info(f"-> DISAPPROVED: {person_dir.name}")
        elif key_pressed == 'y':
            logging.info(f"-> Approved: {person_dir.name}")
        elif key_pressed == 'q':
            logging.info("Quit command received. Exiting review.")
            break
        else:
            logging.warning(f"-> Unknown command '{key_pressed}'. Approving by default.")

    logging.info("Review session finished.")


# ──────────────────────────
# 🚀 STAGE 3: Prune
# ──────────────────────────
def run_prune(args):
    """Delete disapproved directories."""
    root_dir = Path(args.dir)
    disapproved_path = root_dir / DISAPPROVED_FILENAME
    if not disapproved_path.exists():
        logging.error(f"'{DISAPPROVED_FILENAME}' not found in '{root_dir}'. Run the review step first.");
        return

    with open(disapproved_path, "r") as f:
        to_delete = [line.strip() for line in f if line.strip()]

    if not to_delete:
        logging.info("No directories listed for deletion.");
        return

    logging.info(f"Found {len(to_delete)} directories to prune.")
    for dir_name in to_delete:
        dir_path = root_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            logging.info(f"Deleting '{dir_path}'...")
            shutil.rmtree(dir_path)
        else:
            logging.warning(f"Directory not found, skipping: '{dir_path}'")
    logging.info("Pruning complete.")


# ──────────────────────────
# 🚀 STAGE 4: Upload
# ──────────────────────────
def run_upload(args):
    """Upload curated directories to GCP under a specific label and optional method."""
    local_dir = Path(args.dir)
    bucket_name = args.bucket
    label = args.label
    method = args.method  # <-- New: Get the method from arguments

    # Build a more descriptive log message
    log_message = f"--- Uploading results from '{local_dir}' to GCP Bucket '{bucket_name}' under label '{label}'"
    if method:
        log_message += f" with method '{method}'"
    log_message += " ---"
    logging.info(log_message)

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        person_dirs = [d for d in local_dir.iterdir() if d.is_dir() and d.name != "temp_videos"]

        for person_dir in person_dirs:
            for local_file in person_dir.rglob('*'):
                if local_file.is_file():
                    # --- MODIFIED LOGIC: Conditionally add the method to the path ---
                    base_path_parts = [label]
                    if method:
                        base_path_parts.append(method)

                    base_path = "/".join(base_path_parts)
                    remote_path = f"{base_path}/{person_dir.name}/{local_file.name}"

                    logging.info(f"Uploading {local_file.name} to gs://{bucket_name}/{remote_path}")
                    blob = bucket.blob(remote_path)
                    blob.upload_from_filename(str(local_file))

    except Exception as e:
        logging.error(f"Failed to upload results to GCP: {e}");
        sys.exit(1)
    logging.info("Upload complete.")


# ──────────────────────────
# 🛠️ Main Arg Parser
# ──────────────────────────
def main():
    parser = argparse.ArgumentParser(description="A multi-stage tool for curating a face dataset.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Parser for "process" ---
    p_process = subparsers.add_parser("process", help="Extract face chunks from videos.")
    p_process.add_argument("output_dir", type=str, help="Local directory to save the extracted face chunks.")

    input_group = p_process.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--in_bucket", type=str, help="GCP bucket to read videos from.")
    input_group.add_argument("--in_folder", type=Path, help="Local folder with videos.")
    p_process.add_argument("--sample", type=int, help="Optional: Process only a small sample of N videos for testing.")
    # Add all tunable parameters to the process command
    p_process.set_defaults(min_face_size=MIN_FACE_SIZE, min_frames=MIN_FRAMES_PER_CHUNK,
                           max_frames=MAX_FRAMES_PER_CHUNK,
                           yolo_conf=YOLO_CONF_THRESHOLD, dbscan_eps=DBSCAN_EPS, margin=YOLO_BBOX_MARGIN,
                           img_size=MODEL_IMG_SIZE, static_frame_count=STATIC_FRAME_COUNT_THRESHOLD,
                           static_frame_dist=STATIC_BOX_DISTANCE_THRESHOLD, blur_threshold=BLUR_THRESHOLD,
                           ssim_threshold=STATIC_CONTENT_SSIM_THRESHOLD, compactness=CLUSTER_COMPACTNESS_THRESHOLD)

    # --- Parser for "review" ---
    p_review = subparsers.add_parser("review", help="Interactively review and approve/disapprove chunks.")
    p_review.add_argument("dir", type=str, help="The output directory from the 'process' stage.")

    # --- Parser for "prune" ---
    p_prune = subparsers.add_parser("prune", help="Delete all disapproved chunks.")
    p_prune.add_argument("dir", type=str, help="The output directory from the 'process' stage.")

    # --- Parser for "upload" ---
    p_upload = subparsers.add_parser("upload", help="Upload the curated chunks to a GCP bucket under a specific label.")
    p_upload.add_argument("dir", type=str, help="The local directory with the final, curated chunks.")
    p_upload.add_argument("bucket", type=str, help="The destination GCP bucket name.")
    p_upload.add_argument("--label", type=str, help="The label for this dataset (e.g., 'real' or 'fake').")
    p_upload.add_argument("--method", type=str, help="Optional sub-folder for the source or method (e.g., 'tiktok').")

    args = parser.parse_args()

    if args.command == "process":
        run_process(args)
    elif args.command == "review":
        run_review(args)
    elif args.command == "prune":
        run_prune(args)
    elif args.command == "upload":
        run_upload(args)


if __name__ == "__main__":
    # Ensure YOLO model exists
    if not Path(YOLO_MODEL_PATH).exists():
        logging.error(f"YOLO model not found at '{YOLO_MODEL_PATH}'.")
        logging.error("Download from: https://github.com/akanametov/yolo-face/releases/download/v8.0/yolov8s-face.pt")
        sys.exit(1)
    main()
