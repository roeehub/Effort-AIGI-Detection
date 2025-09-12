# gcp_face_processor_v2.py
"""
Processes videos from a local folder or GCP bucket to extract, identify,
and curate face chunks, with a comprehensive test mode for parameter tuning.
"""
from __future__ import annotations
import os
import sys
import logging
import argparse
import random
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import face_recognition
from sklearn.cluster import DBSCAN
from google.cloud import storage
from google.api_core.exceptions import NotFound

# ──────────────────────────
# 📍 Default Configuration (can be overridden by CLI args)
# ──────────────────────────
# --- Processing Parameters ---
MIN_FACE_SIZE = (64, 64)
MIN_FRAMES_PER_CHUNK = 16
MAX_FRAMES_PER_CHUNK = 32
MAX_CHUNKS_PER_PERSON = 3
# --- Model & Algorithm Parameters ---
YOLO_MODEL_PATH = "/Users/roeedar/Library/Application Support/JetBrains/PyCharmCE2024.2/scratches/yolov8s-face.pt"
YOLO_CONF_THRESHOLD = 0.3
DBSCAN_EPS = 0.4

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# ──────────────────────────
# 📦 Data Structures (Same as before)
# ──────────────────────────
@dataclass
class Face:
    frame_image: np.ndarray
    encoding: np.ndarray
    frame_number: int


@dataclass
class VideoChunk:
    video_name: str
    person_id_local: int
    frames: list[Face]
    representative_embedding: np.ndarray = field(init=False)

    def __post_init__(self):
        all_encodings = [face.encoding for face in self.frames]
        if all_encodings:
            self.representative_embedding = np.mean(all_encodings, axis=0)
        else:
            self.representative_embedding = None

    @property
    def chunk_name(self) -> str:
        return f"{Path(self.video_name).stem}_person_{self.person_id_local}"


# ──────────────────────────
# 🛠️ Helper Functions
# ──────────────────────────
def initialize_yolo_model() -> YOLO:
    if not Path(YOLO_MODEL_PATH).exists():
        logging.error(f"YOLO model not found at '{YOLO_MODEL_PATH}'.")
        logging.error("Download from https://github.com/akanametov/yolo-face")
        sys.exit(1)
    logging.info(f"Initializing YOLO model from {YOLO_MODEL_PATH}...")
    return YOLO(YOLO_MODEL_PATH)


def get_local_videos(path: Path) -> list[Path]:
    """Finds all .mp4 files in a local directory."""
    if not path.exists():
        logging.error(f"Local path not found: {path}")
        return []
    logging.info(f"Scanning for .mp4 files in '{path}'...")
    return sorted(list(path.glob("*.mp4")))


def download_videos_from_gcp(bucket_name: str, local_dir: Path) -> list[Path]:
    """Downloads all .mp4 files from a GCP bucket."""
    # (Implementation is the same as before, omitted for brevity but included in final script)
    logging.info(f"Connecting to GCP bucket: '{bucket_name}'")
    video_paths = []
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blobs = [b for b in bucket.list_blobs() if b.name.lower().endswith(".mp4")]
        if not blobs:
            logging.warning(f"No .mp4 files found in bucket '{bucket_name}'.")
            return []
        logging.info(f"Found {len(blobs)} mp4 files. Starting download...")
        for blob in blobs:
            destination_path = local_dir / Path(blob.name).name
            blob.download_to_filename(destination_path)
            logging.info(f" -> Downloaded '{blob.name}' to '{destination_path}'")
            video_paths.append(destination_path)
    except NotFound:
        logging.error(f"GCP bucket '{bucket_name}' not found or access denied.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during GCP download: {e}")
        sys.exit(1)
    return video_paths


def upload_results_to_gcp(local_dir: Path, bucket_name: str):
    """Recursively uploads the contents of a local directory to a GCP bucket."""
    logging.info(f"\n--- Uploading results to GCP Bucket '{bucket_name}' ---")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        for local_file in local_dir.rglob('*'):
            if local_file.is_file():
                remote_path = Path(local_dir.name) / local_file.relative_to(local_dir)
                logging.info(f"Uploading {local_file} to gs://{bucket_name}/{remote_path}")
                blob = bucket.blob(str(remote_path))
                blob.upload_from_filename(str(local_file))
    except Exception as e:
        logging.error(f"Failed to upload results to GCP: {e}")
        sys.exit(1)
    logging.info("Upload complete.")


def sparsely_sample_frames(frames: list, num_samples: int) -> list:
    if len(frames) <= num_samples: return frames
    indices = np.linspace(0, len(frames) - 1, num=num_samples, dtype=int)
    return [frames[i] for i in indices]


def draw_box_on_image(image: np.ndarray, box: tuple, color=(0, 0, 255), thickness=2) -> np.ndarray:
    """Draws a bounding box on an image."""
    img_copy = image.copy()
    x0, y0, x1, y1 = map(int, box)
    return cv2.rectangle(img_copy, (x0, y0), (x1, y1), color, thickness)


# ──────────────────────────
# 🚀 Core Processing Logic
# ──────────────────────────
class VideoProcessor:
    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.yolo_model = initialize_yolo_model()
        self.output_dir = Path(self.config.output_label)
        self.all_processed_chunks = []

        # State for test mode
        self.test_stats = defaultdict(int)
        self.test_examples = {}

    def process_video(self, video_path: Path):
        logging.info(f"Processing video: {video_path.name}")
        all_faces = self._extract_faces_from_video(video_path)
        if not all_faces:
            logging.warning(f"No valid faces found in {video_path.name}. Skipping.")
            return

        logging.info(f"Extracted {len(all_faces)} valid faces from {video_path.name}.")
        encodings = [face.encoding for face in all_faces]
        cluster_labels = self._cluster_embeddings(encodings)
        persons_in_video = defaultdict(list)
        for face, label in zip(all_faces, cluster_labels):
            if label != -1: persons_in_video[label].append(face)

        logging.info(f"Identified {len(persons_in_video)} potential individuals in {video_path.name}.")
        person_counter = 0
        successful_persons = []
        for person_label, faces in persons_in_video.items():
            if len(faces) < self.config.min_frames:
                logging.info(
                    f"  - Discarding person {person_label} (only {len(faces)} frames, needs {self.config.min_frames}).")
                if self.config.test_mode and f"participant_dropped_{person_label}" not in self.test_examples:
                    self.test_stats['participants_dropped_for_size'] += 1
                    self.test_examples[f"participant_dropped_{person_label}"] = faces[0].frame_image
                continue

            sorted_faces = sorted(faces, key=lambda f: f.frame_number)
            sampled_faces = sparsely_sample_frames(sorted_faces, self.config.max_frames)
            logging.info(f"  - Keeping person {person_label}. Sampled {len(sampled_faces)} frames.")

            successful_persons.append(sampled_faces[0].frame_image)
            chunk = VideoChunk(
                video_name=video_path.name,
                person_id_local=person_counter,
                frames=sampled_faces
            )
            self.all_processed_chunks.append(chunk)
            person_counter += 1

        if self.config.test_mode:
            for i, face_img in enumerate(successful_persons):
                self.test_examples[f"participant_success_{i}"] = face_img

    def _extract_faces_from_video(self, video_path: Path) -> list[Face]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.error(f"Could not open video file: {video_path}");
            return []

        if self.config.test_mode:
            self.test_stats['total_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        extracted_faces = []
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            results = self.yolo_model.predict(frame, conf=self.config.yolo_conf, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()

            if len(boxes) == 0 and self.config.test_mode:
                self.test_stats['frames_with_no_face'] += 1
                if 'no_face_detected' not in self.test_examples:
                    self.test_examples['no_face_detected'] = frame

            for box in boxes:
                x0, y0, x1, y1 = map(int, box)
                w, h = x1 - x0, y1 - y0

                if w < self.config.min_face_size[0] or h < self.config.min_face_size[1]:
                    if self.config.test_mode:
                        self.test_stats['small_faces_dropped'] += 1
                        if 'face_too_small' not in self.test_examples:
                            self.test_examples['face_too_small'] = draw_box_on_image(frame, (x0, y0, x1, y1))
                    continue

                face_crop_bgr = frame[y0:y1, x0:x1]
                if face_crop_bgr.size == 0: continue

                face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                face_locations = [(0, w, h, 0)]
                encodings = face_recognition.face_encodings(face_crop_rgb, known_face_locations=face_locations)

                if encodings:
                    extracted_faces.append(Face(face_crop_bgr, encodings[0], frame_num))
            frame_num += 1

        cap.release()
        return extracted_faces

    def _cluster_embeddings(self, encodings: list[np.ndarray]) -> np.ndarray:
        if not encodings: return np.array([])
        logging.info(f"Clustering {len(encodings)} embeddings with DBSCAN (eps={self.config.dbscan_eps})...")
        clt = DBSCAN(metric="euclidean", eps=self.config.dbscan_eps, min_samples=self.config.min_frames)
        clt.fit(encodings)
        return clt.labels_

    def consolidate_and_save(self):
        # (This method is the same as before, omitted for brevity but included in final script)
        if not self.all_processed_chunks:
            logging.warning("No valid chunks were processed. Nothing to save.")
            return
        logging.info(f"\n--- Starting Cross-Video Consolidation ({len(self.all_processed_chunks)} total chunks) ---")
        rep_embeddings = [chunk.representative_embedding for chunk in self.all_processed_chunks]
        logging.info("Clustering representative embeddings to find unique individuals...")
        clt = DBSCAN(metric="euclidean", eps=self.config.dbscan_eps, min_samples=1)
        clt.fit(rep_embeddings)
        global_labels = clt.labels_
        global_identities = defaultdict(list)
        for chunk, label in zip(self.all_processed_chunks, global_labels):
            global_identities[label].append(chunk)
        logging.info(f"Identified {len(global_identities)} unique individuals across the dataset.")
        final_chunks_to_save = []
        for global_id, chunks in global_identities.items():
            if len(chunks) > self.config.max_chunks:
                logging.info(
                    f"  - Global Person {global_id} has {len(chunks)} chunks. Randomly selecting {self.config.max_chunks}.")
                selected_chunks = random.sample(chunks, self.config.max_chunks)
                final_chunks_to_save.extend(selected_chunks)
            else:
                logging.info(f"  - Global Person {global_id} has {len(chunks)} chunks. Keeping all.")
                final_chunks_to_save.extend(chunks)
        logging.info(f"\n--- Saving Final {len(final_chunks_to_save)} Curated Chunks ---")
        for chunk in final_chunks_to_save:
            chunk_dir = self.output_dir / chunk.chunk_name
            chunk_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Saving chunk to '{chunk_dir}'...")
            for i, face in enumerate(chunk.frames):
                save_path = chunk_dir / f"{i:04d}.jpg"
                cv2.imwrite(str(save_path), face.frame_image)
        logging.info("Processing complete.")

    def generate_test_report(self):
        """Generates a markdown report and example images for test mode."""
        logging.info("\n--- Generating Test Mode Report ---")
        report_dir = self.output_dir / "test_report"
        examples_dir = report_dir / "examples"
        examples_dir.mkdir(parents=True, exist_ok=True)

        # Save example images
        for name, img in self.test_examples.items():
            path = examples_dir / f"{name}.jpg"
            cv2.imwrite(str(path), img)
            logging.info(f"Saved example image: {path}")

        # Create report content
        report = f"""
# Test Mode Report for `{self.config.in_folder.name}`

## Summary Statistics
- **Total Frames in Video:** {self.test_stats['total_frames']}
- **Frames with No Face Detected:** {self.test_stats['frames_with_no_face']}
- **Faces Dropped (Too Small):** {self.test_stats['small_faces_dropped']}
- **Participants Dropped (Too Few Frames):** {self.test_stats['participants_dropped_for_size']}
- **Participants Successfully Kept:** {len([k for k in self.test_examples if 'success' in k])}

## Visual Examples
Below are examples of key events during processing.

"""
        # Add images to markdown
        for name in sorted(self.test_examples.keys()):
            report += f"### {name.replace('_', ' ').title()}\n"
            report += f"![{name}](examples/{name}.jpg)\n\n"

        report_path = report_dir / "report.md"
        with open(report_path, "w") as f:
            f.write(report)
        logging.info(f"Report saved to: {report_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process videos for face analysis from local or GCP.")
    parser.add_argument("output_label", type=str, help="A label for the local output directory.")

    # --- Input Sources (mutually exclusive) ---
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--in_bucket", type=str, help="GCP bucket to read .mp4 files from.")
    input_group.add_argument("--in_folder", type=Path,
                             help="Local folder with .mp4 files or a single .mp4 for test mode.")

    # --- Output ---
    parser.add_argument("--out_bucket", type=str, help="Optional: GCP bucket to upload final results to.")

    # --- Test Mode ---
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode on a single video for diagnostics.")

    # --- Tunable Parameters ---
    parser.add_argument("--min_face_size", type=int, nargs=2, default=MIN_FACE_SIZE, help="Min face (width, height).")
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES_PER_CHUNK, help="Min frames per person chunk.")
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES_PER_CHUNK, help="Max frames per person chunk.")
    parser.add_argument("--max_chunks", type=int, default=MAX_CHUNKS_PER_PERSON,
                        help="Max video chunks per unique person.")
    parser.add_argument("--yolo_conf", type=float, default=YOLO_CONF_THRESHOLD,
                        help="YOLO detection confidence threshold.")
    parser.add_argument("--dbscan_eps", type=float, default=DBSCAN_EPS, help="DBSCAN epsilon for clustering.")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # --- Validation ---
    if args.test_mode and args.in_bucket:
        logging.error("Test mode can only be used with a local file via --in_folder.")
        sys.exit(1)
    if args.test_mode and args.in_folder and not args.in_folder.is_file():
        logging.error(f"In test mode, --in_folder must point to a single video file, not a directory.")
        sys.exit(1)

    # --- Setup Directories ---
    output_dir = Path(args.output_label)
    temp_dir = output_dir / "temp_videos"
    output_dir.mkdir(exist_ok=True, parents=True)

    # --- Get Video Paths ---
    video_paths = []
    if args.in_bucket:
        temp_dir.mkdir(exist_ok=True, parents=True)
        video_paths = download_videos_from_gcp(args.in_bucket, temp_dir)
    elif args.in_folder:
        if args.in_folder.is_dir():
            video_paths = get_local_videos(args.in_folder)
        else:  # is a file
            video_paths = [args.in_folder]

    if not video_paths:
        logging.info("No videos to process. Exiting.")
        return

    # --- Process Videos ---
    processor = VideoProcessor(args)

    if args.test_mode:
        processor.process_video(video_paths[0])
        processor.generate_test_report()
    else:
        for path in video_paths:
            try:
                processor.process_video(path)
            except Exception as e:
                logging.error(f"FATAL ERROR while processing {path.name}: {e}", exc_info=True)
        processor.consolidate_and_save()

    # --- Upload Results if Requested ---
    if args.out_bucket and not args.test_mode:
        upload_results_to_gcp(output_dir, args.out_bucket)
    elif args.out_bucket and args.test_mode:
        logging.info("In test mode, results are only saved locally. No upload to GCP was performed.")


if __name__ == "__main__":
    main()
