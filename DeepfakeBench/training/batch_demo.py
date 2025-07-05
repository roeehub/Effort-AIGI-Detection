#!/usr/bin/env python3
# training/batch_demo.py
import argparse
import sys
from pathlib import Path
import cv2
import dlib
import torch
from tqdm import tqdm
from infer import load_detector, infer_single_image, collect_image_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch inference with Effort-AIGI-Detection"
    )
    parser.add_argument(
        "--detector_config", required=True,
        help="Path to detector YAML config"
    )
    parser.add_argument(
        "--weights", required=True,
        help="Path to pretrained detector weights (.pth)"
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Folder of images or single image path to process"
    )
    parser.add_argument(
        "--landmark_model", default=None,
        help="Path to dlib 81-landmark .dat file, or omit for no face cropping"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_detector(args.detector_config, args.weights).to(device)
    model.eval()

    if args.landmark_model:
        face_det = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(args.landmark_model)
    else:
        face_det, predictor = None, None

    img_paths = collect_image_paths(args.input_dir)
    total = len(img_paths)
    print(f"Found {total} image(s) to process.")

    for idx, img_path in enumerate(img_paths, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[Warning] Failed to load, skip: {img_path}", file=sys.stderr)
            continue

        cls, prob = infer_single_image(img, face_det, predictor, model)
        print(
            f"[{idx}/{total}] {img_path.name:>30} | Pred Label: {cls} "
            f"(0=Real, 1=Fake) | Fake Prob: {prob:.4f}"
        )

if __name__ == "__main__":
    main()

