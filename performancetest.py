#!/usr/bin/env python3
import argparse
import logging
import sys
from time import perf_counter, sleep
from datetime import datetime

import tensorflow as tf
from deepface import DeepFace
from lib.streamreader import StreamReader
from include.functions import resetDB

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark Face‑Recognition throughput on GPU"
    )
    p.add_argument("--rtsp-url", required=True, help="RTSP‑URL der Kamera")
    p.add_argument("--db-path", required=True, help="Pfad zur Face‑DB (Ordner mit Subverzeichnissen)")
    p.add_argument("-n", "--num-faces", type=int, default=50,
                   help="Anzahl gefundener Gesichter bis zum Stoppen")
    p.add_argument("--detector", default="retinaface",
                   help="DeepFace detector (yunet, ssd, retinaface, yolov8, …)")
    p.add_argument("--model", default="Facenet512",
                   help="DeepFace model (VGG-Face, Facenet, ArcFace, …)")
    p.add_argument("--metric", default="euclidean_l2",
                   help="Distance‑Metric (euclidean, cosine, …)")
    return p.parse_args()

def main():
    args = parse_args()

    # GPU-Memory Growth aktivieren
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except Exception:
            logging.warning("GPU memory growth konnte nicht gesetzt werden")

    # Model laden
    logging.info(f"Building model {args.model} …")
    DeepFace.build_model(args.model)
    logging.info("Model ready.")

    # StreamReader starten
    stream = StreamReader(rtsp_url=args.rtsp_url, reconnect_delay=5, queue_size=1)
    stream.start()
    sleep(2)  # kurz warten, bis erster Frame gepuffert

    # Variablen für Benchmark
    total_times = []
    found_faces = 0

    try:
        while found_faces < args.num_faces:
            frame = stream.read(timeout=5)
            if frame is None:
                logging.error("Kein Frame erhalten, versuche erneut …")
                continue

            # Messung starten
            t0 = perf_counter()
            faces = DeepFace.find(
                img_path=frame,
                detector_backend=args.detector,
                db_path=args.db_path,
                distance_metric=args.metric,
                model_name=args.model,
                enforce_detection=False,
                silent=True
            )
            dt = perf_counter() - t0

            # Anzahl der erkannten Gesichter in diesem Frame
            # faces ist eine Liste von DataFrames
            count_in_frame = sum(len(df) for df in faces if not df.empty)
            if count_in_frame > 0:
                found_faces += count_in_frame
                total_times.append(dt)
                logging.info(f"Frame {found_faces}/{args.num_faces} – "
                             f"{count_in_frame} face(s), find() in {dt:.3f}s")
            else:
                logging.debug(f"No faces in {dt:.3f}s")

        # Benchmark auswerten
        avg_time = sum(total_times) / len(total_times)
        logging.info("=== Benchmark Ergebnis ===")
        logging.info(f"Gesichter erkannt: {found_faces}")
        logging.info(f"Aufrufe mit ≥1 Face: {len(total_times)}")
        logging.info(f"Durchschnittliche DeepFace.find Dauer: {avg_time:.3f}s")

    except KeyboardInterrupt:
        logging.info("Benchmark per Strg+C abgebrochen")
    finally:
        stream.stop()
        logging.info("StreamReader gestoppt, Programm beendet")


if __name__ == "__main__":
    main()
