from deepface import DeepFace
import cv2 as cv
import numpy as np
from time import sleep, perf_counter
from datetime import datetime
import os
import signal
import sys
import logging
import configparser
import tensorflow as tf
from include.functions import *
from lib.streamreader import StreamReader
from lib.motionchecker import MotionChecker

# GPU memory growth
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except Exception:
    logging.info("Couldn't set Memory Growth for GPU or no GPU found. Continuing...")

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Read config file
config = configparser.ConfigParser()
config.read("config.ini")

stream_url = config["basic"]["stream_url"]
push_url = config["basic"]["push_url"]
motion_url = config["basic"]["motion_url"]
debug = str2bool(config["basic"]["debug"])

# Database setup
path_db = config["database"]["path"]
db = {}
for folder in os.scandir(path_db):
    if folder.is_dir():
        db[folder.name] = {
            "path": f"{path_db}/{folder.name}/{folder.name}.jpg",
            "last_seen": datetime.now(),
            "cnt": 0
        }

# Face recognition model setup
model = config["face_recognition"]["model"]
if model == "yunet":
    os.environ["yunet_score_threshold"] = "0.8"
detector = config["face_recognition"]["detector"]
metric = config["face_recognition"]["metric"]
alignment = str2bool(config["face_recognition"]["alignment"])
face_detect_enf = str2bool(config["face_recognition"]["face_detection_enforce"])

# Thresholds setup
threshold_model = DeepFace.verification.find_threshold(model, metric)
threshold_clearance = int(config["thresholds"]["clearance"])
threshold_last_seen = int(config["thresholds"]["last_seen"])
threshold_pretty_sure = threshold_model - (threshold_model * float(config["thresholds"]["pretty_sure"]))

# Initialize StreamReader and MotionChecker
stream = StreamReader(stream_url)
stream.start()

motion = MotionChecker(motion_url)
motion.start()

# Warm-up
sleep(5)
logging.info(f"Database: {db}")
logging.info("Loading Model...")
DeepFace.build_model(model)
logging.info("Finished loading Model...")

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("Killing Process...", file=sys.stderr)
    stream.stop()
    motion.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Main loop with timing measurements
try:
    while True:
        # Wait for motion
        start = perf_counter()
        motion.wait_for_motion()
        dur_wait = perf_counter() - start

        if debug:
            logging.info(f"wait_for_motion took {dur_wait:.4f}s")

        start = perf_counter()
        resetDB(db, threshold_last_seen)
        dur_reset = perf_counter() - start

        if debug:
            logging.info(f"resetDB took {dur_reset:.4f}s")

        # Read frame
        start = perf_counter()
        frame = stream.read()
        dur_read = perf_counter() - start
        if debug:
            logging.info(f"stream.read took {dur_read:.4f}s")

        if frame is None:
            if debug:
                logging.error("Couldn't receive Frame after motion. Continuing...")
            continue

        # Face find
        start = perf_counter()
        try:
            faces = DeepFace.find(
                img_path=frame,
                detector_backend=detector,
                align=alignment,
                enforce_detection=face_detect_enf,
                db_path=path_db,
                distance_metric=metric,
                model_name=model,
                silent=True
            )
        except ValueError as e:
            if debug:
                logging.error("No Face found! Continuing...")
                logging.error(e)
            continue
        except Exception as e:
            if debug:
                logging.error("Unknown Error! Exiting...")
                logging.error(e)
            signal_handler(None, None)
        dur_find = perf_counter() - start
        if debug:
            logging.info(f"DeepFace.find took {dur_find:.4f}s")

        # Process faces
        start = perf_counter()
        for face in faces:
            if face.empty:
                continue
            for identity in db:
                if identity in face.iloc[0]["identity"]:
                    db[identity]["cnt"] += 1
                    db[identity]["last_seen"] = datetime.now()
                    if face.iloc[0]["distance"] <= threshold_pretty_sure or db[identity]["cnt"] >= threshold_clearance:
                        openDoor(identity, push_url)
        dur_proc = perf_counter() - start
        if debug:
            logging.info(f"face processing took {dur_proc:.4f}s")

except KeyboardInterrupt:
    signal_handler(None, None)
except Exception as e:
    logging.error(f"Unhandled error in main loop: {e}", exc_info=True)
    signal_handler(None, None)
