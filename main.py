from deepface import DeepFace
import cv2 as cv
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
config.read("./config/config.ini")

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
recognition_model = config["face_recognition"]["recognition_model"]
detector_model = config["face_recognition"]["detector_model"]
metric = config["face_recognition"]["metric"]
alignment = str2bool(config["face_recognition"]["alignment"])
enforce = str2bool(config["face_recognition"]["enforce"])

# Thresholds setup
threshold_model = DeepFace.verification.find_threshold(recognition_model, metric)
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
detector_backend_model = DeepFace.build_model(model_name=detector_model, task="face_detector")
recognition_backend_model = DeepFace.build_model(model_name=recognition_model, task="facial_recognition")
logging.info("Finished loading Model...")

# Precompute database embeddings to speed up recognition
for identity, data in db.items():
    img = cv.imread(data["path"])
    try:
        reps = DeepFace.represent(
            img_path=img,
            model_name=recognition_model,
            model=recognition_backend_model,
            detector_backend=detector_model,
            enforce_detection=False,
            align=alignment,
        )
        if len(reps) > 0:
            data["embedding"] = reps[0]["embedding"]
        else:
            logging.warning(f"No face found for {identity} in database image.")
            data["embedding"] = None
    except Exception as e:
        logging.warning(f"Embedding creation failed for {identity}: {e}")
        data["embedding"] = None

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
        motion.wait_motion()
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

        # Compute embeddings for detected faces
        start = perf_counter()
        try:
            reps = DeepFace.represent(
                img_path=frame,
                model_name=recognition_model,
                model=recognition_backend_model,
                detector_backend=detector_model,
                align=alignment,
                enforce_detection=enforce,
            )
        except Exception as e:
            if debug:
                logging.error("No Face found! Continuing...")
                logging.error(e)
            continue
        dur_find = perf_counter() - start
        if debug:
            logging.info(f"DeepFace.represent took {dur_find:.4f}s")

        # Process embeddings
        start = perf_counter()
        for rep in reps:
            embedding = rep["embedding"]
            for identity, data in db.items():
                db_emb = data.get("embedding")
                if db_emb is None:
                    continue
                dist = calc_distance(db_emb, embedding, metric)
                if dist <= threshold_model:
                    data["cnt"] += 1
                    data["last_seen"] = datetime.now()
                    if dist <= threshold_pretty_sure or data["cnt"] >= threshold_clearance:
                        openDoor(identity, push_url)
        dur_proc = perf_counter() - start
        if debug:
            logging.info(f"embedding processing took {dur_proc:.4f}s")

except KeyboardInterrupt:
    signal_handler(None, None)
except Exception as e:
    logging.error(f"Unhandled error in main loop: {e}", exc_info=True)
    signal_handler(None, None)
