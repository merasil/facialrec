import os
import cv2
import argparse
import time
from deepface import DeepFace

def parse_args():
    parser = argparse.ArgumentParser(
        description="Einfacher Testlauf: DeepFace.find auf Bilder anwenden"
    )
    parser.add_argument(
        "--db_path", required=True,
        help="Pfad zur Face-DB"
    )
    parser.add_argument(
        "--test_dir", required=True,
        help="Pfad zu Testbildern"
    )
    parser.add_argument(
        "--detectors", nargs='+',
        default=["opencv", "mtcnn"],
        help="Welche Detektoren testen"
    )
    parser.add_argument(
        "--models", nargs='+',
        default=["Facenet", "ArcFace"],
        help="Welche Erkennungsmodelle testen"
    )
    parser.add_argument(
        "--metrics", nargs='+',
        default=["cosine"],
        help="Welche Distanzmetriken testen"
    )
    parser.add_argument(
        "--threshold", type=float,
        default=None,
        help="Schwellenwert (None = Default)"
    )
    parser.add_argument(
        "--enforce", type=lambda x: x.lower() in ("true","1","yes","y"),
        default=True,
        help="Enforce face detection (True/False, Default: True)"
    )
    return parser.parse_args()


def evaluate(db_path, test_dir, detectors, models, metrics, threshold, enforce):
    # Lade alle Bilder im Test-Verzeichnis
    img_files = [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    stats = {}

    for det in detectors:
        DeepFace.build_model(model_name=det, task="face_detector")
        for mod in models:
            # Model einmal pro Kombination laden
            DeepFace.build_model(model_name=mod, task="facial_recognition")
            for met in metrics:
                detected = 0
                recognized = 0
                calls = 0
                total_time = 0.0
                i = 0

                print(f"\n>> Detector={det} | Model={mod} | Metric={met}")
                for img_path in img_files:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"  Kann Bild nicht laden: {img_path}")
                        continue

                    start = time.perf_counter()
                    try:
                        result = DeepFace.find(
                            img_path=img,
                            db_path=db_path,
                            detector_backend=det,
                            model_name=mod,
                            distance_metric=met,
                            enforce_detection=enforce,
                            align=False,
                            silent=True
                        )
                        elapsed = time.perf_counter() - start
                        df = result[0] if isinstance(result, list) else result

                        calls += 1
                        if i != 0:
                            total_time += elapsed
                        print(f"Processing Image{i} took: {elapsed} ms")
                        i += 1
                        detected += 1
                    except ValueError:
                        continue

                    if not df.empty:
                        recognized += 1

                avg_time = (total_time / (calls-1)) if calls-1 else 0.0
                stats[(det, mod, met)] = {
                    "enforce": enforce,
                    "detected": detected,
                    "recognized": recognized,
                    "calls": calls,
                    "avg_time": avg_time
                }

    # ASCII-Tabelle erzeugen
    headers = ["Name", "Metric", "Enforce", "Avg Time (ms)", "Detected", "Recognized"]
    rows = []
    for (det, mod, met), data in stats.items():
        name = f"{mod}/{det}"
        rows.append([
            name,
            met,
            str(data["enforce"]),
            f"{data['avg_time']*1000:.1f}",
            str(data["detected"]),
            str(data["recognized"])
        ])

    # Spaltenbreiten berechnen
    col_widths = [max(len(cell) for cell in col) for col in zip(headers, *rows)]
    sep = "+" + "+".join("-"*(w+2) for w in col_widths) + "+"

    def print_row(cells):
        print(
            "| " + " | ".join(cell.ljust(w) for cell, w in zip(cells, col_widths)) + " |"
        )

    print("\n=== Zusammenfassung aller Tests ===")
    print(sep)
    print_row(headers)
    print(sep.replace("-", "="))
    for row in rows:
        print_row(row)
        print(sep)


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        db_path=args.db_path,
        test_dir=args.test_dir,
        detectors=args.detectors,
        models=args.models,
        metrics=args.metrics,
        threshold=args.threshold,
        enforce=args.enforce
    )
