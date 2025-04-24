import os
import cv2
import argparse
import pandas as pd
from time import perf_counter
from deepface import DeepFace


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate performance of detector and recognition model combinations"
    )
    parser.add_argument(
        "--db_path", required=True,
        help="Path to the face database directory (each subfolder per identity)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--test_dir",
        help="Path to test images directory"
    )
    group.add_argument(
        "--video",
        help="Path to input video file; all frames will be tested"
    )
    parser.add_argument(
        "--detectors", nargs='+',
        default=["opencv", "ssd", "mtcnn", "dlib", "retinaface", "mediapipe", "yunet"],
        help="List of detector backends to test"
    )
    parser.add_argument(
        "--models", nargs='+',
        default=["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepID", "ArcFace", "Dlib", "SFace"],
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
        help="Maximum number of test items to process (images or video frames). Default is all."
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


def evaluate(db_path, test_dir, video_path, detectors, models, metric, threshold, limit):
    use_video = video_path is not None
    img_files = []
    if not use_video:
        img_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
                     if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if limit:
            img_files = img_files[:limit]

    results = []

    # Pre-build recognition models
    model_cache = {}
    for model_name in models:
        print(f"Loading model {model_name}...")
        model_cache[model_name] = DeepFace.build_model(model_name)

    for detector in detectors:
        for model_name in model_cache:
            print(f"Testing Detector={detector} Model={model_name}...")
            detected = 0
            recognized = 0
            total_time = 0.0
            thr = threshold if threshold is not None else DeepFace.verification.find_threshold(model_name, metric)

            # Helper to process a frame or image
            def process_image(img):
                nonlocal detected, recognized, total_time
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
                    total_time += perf_counter() - start
                    return
                dur = perf_counter() - start
                total_time += dur

                # DeepFace.find kann DataFrame oder Liste von DataFrames zurückgeben
                if isinstance(df, list):
                    try:
                        df = pd.concat(df, ignore_index=True)
                    except Exception:
                        # Liste leer oder nicht konkateniert
                        return

                detected += len(df)
                recognized += len(df[df["distance"] <= thr])

            if use_video:
                cap = cv2.VideoCapture(video_path)
                frame_idx = 0
                while True:
                    ret, img = cap.read()
                    if not ret:
                        break
                    frame_idx += 1
                    if limit and frame_idx > limit:
                        break
                    process_image(img)
                cap.release()
            else:
                for img_path in img_files:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    process_image(img)

            avg_time = (total_time / recognized) if recognized else 0
            results.append({
                "detector": detector,
                "model": model_name,
                "detected_faces": detected,
                "recognized_faces": recognized,
                "avg_time_per_recognition_s": round(avg_time, 4)
            })

    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    args = parse_args()
    db = load_db(args.db_path)
    evaluate(
        db_path=args.db_path,
        test_dir=args.test_dir,
        video_path=args.video,
        detectors=args.detectors,
        models=args.models,
        metric=args.metric,
        threshold=args.threshold,
        limit=args.limit
    )