import os
import cv2
import argparse
import numpy as np
import pandas as pd
from time import perf_counter
from deepface import DeepFace


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate performance of Detector and Recognition model combinations"
    )
    parser.add_argument(
        "--db_path", required=True,
        help="Path to the face database directory (each subfolder per identity)"
    )
    parser.add_argument(
        "--test_dir", required=True,
        help="Path to test images directory"
    )
    parser.add_argument(
        "--detectors", nargs='+', default=["opencv", "ssd", "mtcnn", "dlib", "retinaface", "mediapipe", "yunet"],
        help="List of detector backends to test"
    )
    parser.add_argument(
        "--models", nargs='+', default=["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"],
        help="List of face recognition models to test"
    )
    parser.add_argument(
        "--metric", default="cosine",
        help="Distance metric to use for recognition"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Distance threshold for recognizing a face (if None, uses default from DeepFace)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of test images to process (random order). Default is all."
    )
    return parser.parse_args()


def load_db(db_path):
    db = {}
    for folder in os.listdir(db_path):
        full = os.path.join(db_path, folder)
        if os.path.isdir(full):
            img = os.path.join(full, f"{folder}.jpg")
            if os.path.exists(img):
                db[folder] = img
    return db


def evaluate(db_path, test_dir, detectors, models, metric, threshold, limit):
    img_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
                 if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if limit:
        img_files = img_files[:limit]

    # prepare results storage
    results = []

    # prebuild recognition models
    model_cache = {}
    for model_name in models:
        print(f"Loading model {model_name}...")
        model_cache[model_name] = DeepFace.build_model(model_name)

    for detector in detectors:
        for model_name, model in model_cache.items():
            print(f"Testing Detector={detector} Model={model_name}...")
            detected = 0
            recognized = 0
            total_time = 0.0

            # determine threshold
            thr = threshold if threshold is not None else DeepFace.verification.find_threshold(model_name, metric)

            for img_path in img_files:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                start = perf_counter()
                try:
                    df = DeepFace.find(
                        img_path=img,
                        db_path=db_path,
                        detector_backend=detector,
                        model_name=model_name,
                        distance_metric=metric,
                        enforce_detection=False,
                        silent=True
                    )
                except Exception:
                    dur = perf_counter() - start
                    total_time += dur
                    continue
                dur = perf_counter() - start
                total_time += dur

                # detection: count unique face entries
                count_faces = len(df)
                detected += count_faces

                # recognition: distances below threshold
                matches = df[df["distance"] <= thr]
                recognized += len(matches)

            # avg time per recognized face
            avg_time = (total_time / recognized) if recognized else 0
            results.append({
                "detector": detector,
                "model": model_name,
                "detected_faces": detected,
                "recognized_faces": recognized,
                "avg_time_per_recognition_s": round(avg_time, 4)
            })

    # print results as table
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    args = parse_args()
    db = load_db(args.db_path)
    evaluate(args.db_path, args.test_dir, args.detectors,
             args.models, args.metric, args.threshold, args.limit)
