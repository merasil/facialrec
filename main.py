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

# Read config file first to get debug setting
config = configparser.ConfigParser()
config_path = "./config/config.ini"

if not os.path.exists(config_path):
    print(f"ERROR: Config file not found at {config_path}", file=sys.stderr)
    print("Please create config.ini based on config-example.ini", file=sys.stderr)
    sys.exit(1)

config.read(config_path)

try:
    stream_url = config["basic"]["stream_url"]
    stream_url_lowres = config.get("basic", "stream_url_lowres", fallback="").strip()
    push_url = config["basic"]["push_url"]
    motion_url = config["basic"]["motion_url"]
    debug = str2bool(config["basic"]["debug"])
except KeyError as e:
    print(f"ERROR: Missing required config key: {e}", file=sys.stderr)
    sys.exit(1)

# Logging setup based on debug setting
log_level = logging.DEBUG if debug else logging.INFO
logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Database setup
path_db = config["database"]["path"]

if not os.path.exists(path_db):
    logging.error(f"Database path does not exist: {path_db}")
    sys.exit(1)

db = {}
for folder in os.scandir(path_db):
    if folder.is_dir():
        img_path = f"{path_db}/{folder.name}/{folder.name}.jpg"
        if os.path.exists(img_path):
            db[folder.name] = {
                "path": img_path,
                "last_seen": datetime.now(),
                "cnt": 0
            }
            logging.debug(f"Loaded identity: {folder.name}")
        else:
            logging.warning(f"Image not found for {folder.name}: {img_path}")

if not db:
    logging.error("No valid identities found in database!")
    sys.exit(1)

# Face recognition model setup
try:
    recognition_model = config["face_recognition"]["recognition_model"]
    detector_model = config["face_recognition"]["detector_model"]
    metric = config["face_recognition"]["metric"]
    alignment = str2bool(config["face_recognition"]["alignment"])
    enforce = str2bool(config["face_recognition"]["enforce"])
except KeyError as e:
    logging.error(f"Missing required face_recognition config: {e}")
    sys.exit(1)

# Thresholds setup
try:
    threshold_clearance = int(config["thresholds"]["clearance"])
    threshold_last_seen = int(config["thresholds"]["last_seen"])
    pretty_sure_factor = float(config["thresholds"]["pretty_sure"])
except KeyError as e:
    logging.error(f"Missing required thresholds config: {e}")
    sys.exit(1)

try:
    threshold_model = DeepFace.verification.find_threshold(recognition_model, metric)
    threshold_pretty_sure = threshold_model - (threshold_model * pretty_sure_factor)
except Exception as e:
    logging.error(f"Error calculating threshold for model '{recognition_model}' with metric '{metric}': {e}")
    logging.error("Valid metrics are usually: cosine, euclidean, euclidean_l2")
    sys.exit(1)

# Motion detection setup
try:
    use_internal_motion = str2bool(config.get("motion", "use_internal", fallback="False"))
    motion_threshold = int(config.get("motion", "threshold", fallback="25"))
    motion_min_area = float(config.get("motion", "min_area_percent", fallback="0.2"))
    motion_cooldown = int(config.get("motion", "cooldown_seconds", fallback="5"))
except (KeyError, ValueError) as e:
    logging.warning(f"Motion config error, using defaults: {e}")
    use_internal_motion = False
    motion_threshold = 25
    motion_min_area = 0.2
    motion_cooldown = 5

# Initialize StreamReaders
# Main stream for face recognition
stream = StreamReader(stream_url)
stream.start()

# Low-res stream for motion detection (if configured and using internal motion)
stream_motion = None
if use_internal_motion and stream_url_lowres:
    logging.info(f"Using separate low-res stream for motion detection: {stream_url_lowres}")
    stream_motion = StreamReader(stream_url_lowres)
    stream_motion.start()
elif use_internal_motion:
    logging.info("Using main stream for motion detection (no low-res stream configured)")
    stream_motion = stream

# Initialize MotionChecker (internal or external)
if use_internal_motion:
    logging.info("Using internal motion detection")
    motion = MotionChecker(
        motion_url=None,
        stream_reader=stream_motion,
        use_internal=True,
        threshold=motion_threshold,
        min_area=motion_min_area,
        cooldown_seconds=motion_cooldown
    )
else:
    logging.info("Using external motion detection (Frigate)")
    motion = MotionChecker(motion_url, cooldown_seconds=motion_cooldown)
motion.start()

# Warm-up
sleep(5)
logging.info(f"Database: {db}")
logging.info("Loading Model...")
ddm = DeepFace.build_model(model_name=detector_model, task="face_detector")
drm = DeepFace.build_model(model_name=recognition_model, task="facial_recognition")
logging.info("Finished loading Model...")

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("Killing Process...", file=sys.stderr)
    stream.stop()
    if stream_motion and stream_motion != stream:
        stream_motion.stop()
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
            logging.debug(f"wait_for_motion took {dur_wait:.4f}s")

        start = perf_counter()
        resetDB(db, threshold_last_seen)
        dur_reset = perf_counter() - start

        if debug:
            logging.debug(f"resetDB took {dur_reset:.4f}s")

        # Read frame
        start = perf_counter()
        frame = stream.read()
        dur_read = perf_counter() - start
        if debug:
            logging.debug(f"stream.read took {dur_read:.4f}s")

        if frame is None:
            if debug:
                logging.debug("Couldn't receive Frame after motion. Continuing...")
            continue

        # Face find
        start = perf_counter()
        try:
            faces = DeepFace.find(
                img_path=frame,
                detector_backend=detector_model,
                align=alignment,
                enforce_detection=enforce,
                db_path=path_db,
                distance_metric=metric,
                model_name=recognition_model,
                silent=True
            )
        except ValueError as e:
            if debug:
                logging.debug("No Face found! Continuing...")
                logging.debug(e)
            continue
        except Exception as e:
            logging.error(f"Error during face recognition: {e}")
            continue
        dur_find = perf_counter() - start
        if debug:
            logging.debug(f"DeepFace.find took {dur_find:.4f}s")

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
            logging.debug(f"face processing took {dur_proc:.4f}s")

except KeyboardInterrupt:
    signal_handler(None, None)
except Exception as e:
    logging.error(f"Unhandled error in main loop: {e}", exc_info=True)
    signal_handler(None, None)
