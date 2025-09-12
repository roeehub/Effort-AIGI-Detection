# gcp_face_processor_v3.py
"""
Processes videos from a local folder to extract, identify, and curate face chunks,
with robust data consistency checks and artifact removal.

V3 Changes:
- CRITICAL FIX: Face cropping now matches the 'video_preprocessor.py' logic:
  adds a margin, creates a square crop, and resizes to a fixed 224x224.
- FEATURE: Added a check to detect and discard "static" faces (e.g., from a
  picture in the video) by analyzing bounding box stability.
- ENHANCEMENT: Test mode now generates both the diagnostic report AND the
  final output folders for the single test video.
"""
from __future__ import annotations
import sys
import logging
import argparse
import random
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO
import face_recognition
from sklearn.cluster import DBSCAN
from google.cloud import storage
from google.api_core.exceptions import NotFound

# ──────────────────────────
# 📍 Default Configuration (can be overridden by CLI args)
# ──────────────────────────
# --- Processing Parameters ---
MIN_FACE_SIZE = (80, 80)
MIN_FRAMES_PER_CHUNK = 16
MAX_FRAMES_PER_CHUNK = 32
MAX_CHUNKS_PER_PERSON = 3
STATIC_FACE_THRESHOLD = 0.5  # Std. dev. of bbox coordinates. If lower, it's a static image.

# --- Model & Algorithm Parameters ---
YOLO_MODEL_PATH = "/Users/roeedar/Library/Application Support/JetBrains/PyCharmCE2024.2/scratches/yolov8s-face.pt"
YOLO_CONF_THRESHOLD = 0.3
DBSCAN_EPS = 0.4

# --- DATA CONSISTENCY PARAMETERS (from video_preprocessor.py) ---
MODEL_IMG_SIZE = 224  # The final output size of the face crop
YOLO_BBOX_MARGIN = 20  # Margin to add around the detected face
STATIC_FRAME_COUNT_THRESHOLD = 5
# How far a face's center can be from a static cluster's center to be removed (in pixels).
STATIC_BOX_DISTANCE_THRESHOLD = 10.0
BLUR_THRESHOLD = 100.0

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def is_blurry(image: np.ndarray, threshold: float) -> bool:
    """
    Determines if an image is blurry by calculating the variance of the Laplacian.
    """
    # The face crops are already BGR, so we convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.CV_64F is a 64-bit float to avoid overflow
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


# ──────────────────────────
# 📦 Data Structures
# ──────────────────────────
@dataclass
class Face:
    """Represents a single detected face instance, including its original box."""
    frame_image: np.ndarray  # The final processed (cropped, resized) image
    encoding: np.ndarray
    frame_number: int
    box: tuple[int, int, int, int]  # Original bbox (x0, y0, x1, y1) for static check


@dataclass
class VideoChunk:
    """Represents a sequence of frames for one person in one video."""
    video_name: str
    person_id_local: int
    frames: list[Face]
    unique_id: int = field(default_factory=lambda: random.randint(100000, 999999))
    representative_embedding: np.ndarray = field(init=False)

    def __post_init__(self):
        all_encodings = [face.encoding for face in self.frames]
        if all_encodings:
            self.representative_embedding = np.mean(all_encodings, axis=0)
        else:
            self.representative_embedding = None

    @property
    def chunk_name(self) -> str:
        video_stem = Path(self.video_name).stem
        return f"{video_stem}_{self.unique_id:06d}_person_{self.person_id_local}"


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


# (Other helpers like get_local_videos, download_videos_from_gcp, upload_results_to_gcp,
# sparsely_sample_frames, and draw_box_on_image are omitted for brevity but are unchanged
# from your provided script and included in the final code.)
def get_local_videos(path: Path) -> list[Path]:
    """Finds all .mp4 files in a local directory."""
    if not path.exists():
        logging.error(f"Local path not found: {path}")
        return []
    logging.info(f"Scanning for .mp4 files in '{path}'...")
    return sorted(list(path.glob("*.mp4")))


def download_videos_from_gcp(bucket_name: str, local_dir: Path) -> list[Path]:
    """Downloads all .mp4 files from a GCP bucket."""
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
        self.test_stats = defaultdict(int)
        self.test_examples = {}

    def _filter_static_frames_from_chunk(self, faces: list[Face]) -> list[Face]:
        """
        Scans a list of faces for a single person and removes any subset that
        originates from a static image in the background.
        """
        if len(faces) < self.config.static_frame_count:
            return faces  # Not enough frames to analyze

        # Group boxes by rounding their centers to handle minor detection jitter
        box_clusters = defaultdict(list)
        for face in faces:
            x0, y0, x1, y1 = face.box
            center_x = int((x0 + x1) / 20)  # Discretize by 20px grid
            center_y = int((y0 + y1) / 20)
            box_clusters[(center_x, center_y)].append(face)

        # Find potential static sources (clusters larger than threshold)
        static_sources = [cluster for cluster in box_clusters.values() if
                          len(cluster) >= self.config.static_frame_count]
        if not static_sources:
            return faces  # No dominant static source found

        # Calculate the average center of each static source
        static_centers = []
        for source in static_sources:
            boxes = np.array([f.box for f in source])
            avg_center = np.mean([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
            static_centers.append(avg_center)

        # Filter out any face that is too close to the center of a static source
        dynamic_faces = []
        for face in faces:
            is_static = False
            face_box = face.box
            face_center = np.array([(face_box[0] + face_box[2]) / 2, (face_box[1] + face_box[3]) / 2])
            for static_center in static_centers:
                distance = np.linalg.norm(face_center - static_center)
                if distance < self.config.static_frame_dist:
                    is_static = True
                    break
            if not is_static:
                dynamic_faces.append(face)

        num_removed = len(faces) - len(dynamic_faces)
        if num_removed > 0:
            logging.info(f"    -> Cleaned {num_removed} static frames from this person's chunk.")
            self.test_stats['static_frames_cleaned'] += num_removed

        return dynamic_faces

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
        successful_persons_examples = []
        for person_label, faces in persons_in_video.items():
            # --- START OF CHANGED LOGIC ---

            # Filter 1: Quick check for entirely static chunks (e.g., a photo filling the screen)
            boxes = np.array([face.box for face in faces])
            box_std_dev = np.mean(np.std(boxes, axis=0))
            if box_std_dev < self.config.static_threshold:
                logging.warning(
                    f"  - Discarding entirely static chunk for person {person_label} (Box std dev: {box_std_dev:.2f}).")
                if self.config.test_mode:
                    self.test_stats['participants_dropped_for_static_chunk'] += 1
                    self.test_examples[f"participant_dropped_static_chunk_{person_label}"] = faces[0].frame_image
                continue

            # Filter 2: NEW - Clean out static frames from within a potentially dynamic chunk
            cleaned_faces = self._filter_static_frames_from_chunk(faces)

            # Filter 3: Check if enough frames remain *after* cleaning
            if len(cleaned_faces) < self.config.min_frames:
                logging.info(
                    f"  - Discarding person {person_label} (only {len(cleaned_faces)} dynamic frames remain, needs {self.config.min_frames}).")
                if self.config.test_mode:
                    self.test_stats['participants_dropped_post_cleaning'] += 1
                    self.test_examples[f"participant_dropped_post_cleaning_{person_label}"] = faces[0].frame_image
                continue

            # --- END OF CHANGED LOGIC ---

            # Sort, sample, and save the valid, cleaned chunk
            sorted_faces = sorted(cleaned_faces, key=lambda f: f.frame_number)
            sampled_faces = sparsely_sample_frames(sorted_faces, self.config.max_frames)
            logging.info(f"  - Keeping person {person_label}. Sampled {len(sampled_faces)} frames.")

            if self.config.test_mode:
                successful_persons_examples.append(sampled_faces[0].frame_image)

            chunk = VideoChunk(video_name=video_path.name, person_id_local=person_counter, frames=sampled_faces)
            self.all_processed_chunks.append(chunk)
            person_counter += 1

        if self.config.test_mode:
            for i, face_img in enumerate(successful_persons_examples):
                self.test_examples[f"participant_success_{i}"] = face_img

    def _extract_faces_from_video(self, video_path: Path) -> list[Face]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.error(f"Could not open video file: {video_path}")
            return []

        if self.config.test_mode: self.test_stats['total_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        extracted_faces = []
        frame_num = 0
        frame_h, frame_w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            results = self.yolo_model.predict(frame, conf=self.config.yolo_conf, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()

            if len(boxes) == 0 and self.config.test_mode:
                self.test_stats['frames_with_no_face'] += 1
                if 'no_face_detected' not in self.test_examples: self.test_examples['no_face_detected'] = frame

            for box in boxes:
                # 1. Add margin to the initial bounding box
                x0, y0, x1, y1 = box
                x0 = max(0, x0 - self.config.margin)
                y0 = max(0, y0 - self.config.margin)
                x1 = min(frame_w, x1 + self.config.margin)
                y1 = min(frame_h, y1 + self.config.margin)
                original_box_tuple = (int(x0), int(y0), int(x1), int(y1))

                # Filter by size AFTER adding margin
                w, h = x1 - x0, y1 - y0
                if w < self.config.min_face_size[0] or h < self.config.min_face_size[1]:
                    if self.config.test_mode:
                        self.test_stats['small_faces_dropped'] += 1
                        if 'face_too_small' not in self.test_examples:
                            self.test_examples['face_too_small'] = draw_box_on_image(frame, original_box_tuple)
                    continue

                # 2. Create a SQUARE crop centered on the face
                center_x, center_y = x0 + w / 2, y0 + h / 2
                side_length = max(w, h)
                sq_x0 = max(0, int(center_x - side_length / 2))
                sq_y0 = max(0, int(center_y - side_length / 2))
                sq_x1 = min(frame_w, int(center_x + side_length / 2))
                sq_y1 = min(frame_h, int(center_y + side_length / 2))

                face_crop_bgr = frame[sq_y0:sq_y1, sq_x0:sq_x1]
                if face_crop_bgr.size == 0: continue

                # 3. Resize to the final fixed model size
                final_face = cv2.resize(face_crop_bgr, (self.config.img_size, self.config.img_size),
                                        interpolation=cv2.INTER_AREA)

                # Get embedding from the resized, square crop
                face_crop_rgb = cv2.cvtColor(final_face, cv2.COLOR_BGR2RGB)
                # Location is the whole image since it's already cropped
                face_locations = [(0, self.config.img_size, self.config.img_size, 0)]
                encodings = face_recognition.face_encodings(face_crop_rgb, known_face_locations=face_locations)

                if encodings:
                    extracted_faces.append(Face(final_face, encodings[0], frame_num, original_box_tuple))

            frame_num += 1

        cap.release()
        return extracted_faces

    def _cluster_embeddings(self, encodings: list[np.ndarray]) -> np.ndarray:
        # (This method is unchanged)
        if not encodings: return np.array([])
        logging.info(f"Clustering {len(encodings)} embeddings with DBSCAN (eps={self.config.dbscan_eps})...")
        clt = DBSCAN(metric="euclidean", eps=self.config.dbscan_eps, min_samples=self.config.min_frames)
        clt.fit(encodings)
        return clt.labels_

    def consolidate_and_save(self):
        # (This method is unchanged)
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
        # (This method is updated to include new stats)
        logging.info("\n--- Generating Test Mode Report ---")
        report_dir = self.output_dir / "test_report"
        examples_dir = report_dir / "examples"
        examples_dir.mkdir(parents=True, exist_ok=True)

        for name, img in self.test_examples.items():
            path = examples_dir / f"{name}.jpg"
            cv2.imwrite(str(path), img)
            logging.info(f"Saved example image: {path}")

        report = f"""
# Test Mode Report for `{self.config.in_folder.name}`

## Summary Statistics
- **Total Frames in Video:** {self.test_stats['total_frames']}
- **Frames with No Face Detected:** {self.test_stats['frames_with_no_face']}
- **Faces Dropped (Too Small):** {self.test_stats['small_faces_dropped']}
- **Participants Dropped (Too Few Frames):** {self.test_stats['participants_dropped_for_size']}
- **Participants Dropped (Static Face):** {self.test_stats['participants_dropped_for_static']}
- **Participants Successfully Kept:** {len([k for k in self.test_examples if 'success' in k])}

## Visual Examples
"""
        for name in sorted(self.test_examples.keys()):
            report += f"### {name.replace('_', ' ').title()}\n![{name}](examples/{name}.jpg)\n\n"
        report_path = report_dir / "report.md"
        with open(report_path, "w") as f:
            f.write(report)
        logging.info(f"Report saved to: {report_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process videos for face analysis from local or GCP.")
    parser.add_argument("output_label", type=str, help="A label for the local output directory.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--in_bucket", type=str, help="GCP bucket to read .mp4 files from.")
    input_group.add_argument("--in_folder", type=Path,
                             help="Local folder with .mp4 files or a single .mp4 for test mode.")
    parser.add_argument("--out_bucket", type=str, help="Optional: GCP bucket to upload final results to.")
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode on a single video for diagnostics.")
    # Tunable Parameters, including new ones
    parser.add_argument("--min_face_size", type=int, nargs=2, default=MIN_FACE_SIZE, help="Min face (width, height).")
    parser.add_argument("--min_frames", type=int, default=MIN_FRAMES_PER_CHUNK, help="Min frames per person chunk.")
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES_PER_CHUNK, help="Max frames per person chunk.")
    parser.add_argument("--max_chunks", type=int, default=MAX_CHUNKS_PER_PERSON,
                        help="Max video chunks per unique person.")
    parser.add_argument("--yolo_conf", type=float, default=YOLO_CONF_THRESHOLD,
                        help="YOLO detection confidence threshold.")
    parser.add_argument("--dbscan_eps", type=float, default=DBSCAN_EPS, help="DBSCAN epsilon for clustering.")
    parser.add_argument("--static_threshold", type=float, default=STATIC_FACE_THRESHOLD,
                        help="Std. dev. threshold for static face detection.")
    parser.add_argument("--margin", type=int, default=YOLO_BBOX_MARGIN, help="Margin to add to face bounding box.")
    parser.add_argument("--img_size", type=int, default=MODEL_IMG_SIZE, help="Final size of the square face crop.")
    parser.add_argument("--static_frame_count", type=int, default=STATIC_FRAME_COUNT_THRESHOLD,
                        help="Min frame count to identify a static image within a chunk.")
    parser.add_argument("--static_frame_dist", type=float, default=STATIC_BOX_DISTANCE_THRESHOLD,
                        help="Pixel distance threshold for cleaning static frames from a chunk.")
    parser.add_argument("--blur_threshold", type=float, default=BLUR_THRESHOLD,
                        help="Blurriness threshold (variance of Laplacian). Lower values are more blurry. Higher threshold is stricter.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.test_mode and args.in_bucket:
        logging.error("Test mode can only be used with a local file via --in_folder.")
        sys.exit(1)
    if args.test_mode and args.in_folder and not args.in_folder.is_file():
        logging.error(f"In test mode, --in_folder must point to a single video file.")
        sys.exit(1)

    output_dir = Path(args.output_label)
    temp_dir = output_dir / "temp_videos"
    output_dir.mkdir(exist_ok=True, parents=True)

    video_paths = []
    if args.in_bucket:
        temp_dir.mkdir(exist_ok=True, parents=True)
        video_paths = download_videos_from_gcp(args.in_bucket, temp_dir)
    elif args.in_folder:
        video_paths = [args.in_folder] if args.in_folder.is_file() else get_local_videos(args.in_folder)

    if not video_paths:
        logging.info("No videos to process. Exiting.")
        return

    processor = VideoProcessor(args)

    if args.test_mode:
        logging.info("--- RUNNING IN TEST MODE ---")
        processor.process_video(video_paths[0])
        processor.generate_test_report()
        logging.info("--- Test mode also generating final output for inspection ---")
        # In test mode, consolidate will only run on the chunks from the single video
        processor.consolidate_and_save()
    else:
        for path in video_paths:
            try:
                processor.process_video(path)
            except Exception as e:
                logging.error(f"FATAL ERROR while processing {path.name}: {e}", exc_info=True)
        processor.consolidate_and_save()

    if args.out_bucket and not args.test_mode:
        upload_results_to_gcp(output_dir, args.out_bucket)
    elif args.out_bucket and args.test_mode:
        logging.info("In test mode, results are saved locally. No upload to GCP was performed.")


if __name__ == "__main__":
    main()
