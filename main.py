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
from concurrent.futures import ThreadPoolExecutor
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
ddm = DeepFace.build_model(model_name=detector_model, task="face_detector")
drm = DeepFace.build_model(model_name=recognition_model, task="facial_recognition")
logging.info("Finished loading Model...")

# Thread pool for parallel frame processing
max_workers = int(config.get("performance", "max_workers", fallback="2"))
executor = ThreadPoolExecutor(max_workers=max_workers)

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("Killing Process...", file=sys.stderr)
    executor.shutdown(wait=False)
    stream.stop()
    motion.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Face processing function for parallel execution
def process_frame(frame, frame_id):
    """Process a single frame and return results"""
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
        dur = perf_counter() - start
        if debug:
            logging.debug(f"Frame {frame_id}: DeepFace.find took {dur:.4f}s")
        return (True, faces, frame_id)
    except ValueError as e:
        if debug:
            logging.debug(f"Frame {frame_id}: No face found")
        return (False, None, frame_id)
    except Exception as e:
        logging.error(f"Frame {frame_id}: Error during face recognition: {e}")
        return (False, None, frame_id)

# Performance mode configuration
use_parallel = str2bool(config.get("performance", "parallel_processing", fallback="False"))
frames_per_motion = int(config.get("performance", "frames_per_motion", fallback="1"))

# Main loop with timing measurements
try:
    frame_counter = 0
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

        # Read multiple frames for better accuracy
        frames_to_process = []
        for i in range(frames_per_motion):
            frame = stream.read()
            if frame is not None:
                frames_to_process.append((frame, frame_counter))
                frame_counter += 1
            elif debug:
                logging.debug(f"Couldn't receive frame {i+1}/{frames_per_motion}")

        if not frames_to_process:
            if debug:
                logging.error("Couldn't receive any frames after motion. Continuing...")
            continue

        # Process frames (parallel or sequential)
        if use_parallel and len(frames_to_process) > 1:
            # Submit all frames for parallel processing
            futures = [executor.submit(process_frame, frame, fid) for frame, fid in frames_to_process]

            # Collect results
            for future in futures:
                success, faces, fid = future.result()
                if not success:
                    continue

                # Process recognized faces
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
                    logging.debug(f"Frame {fid}: face processing took {dur_proc:.4f}s")
        else:
            # Sequential processing (original behavior)
            for frame, fid in frames_to_process:
                success, faces, fid = process_frame(frame, fid)
                if not success:
                    continue

                # Process recognized faces
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
                    logging.debug(f"Frame {fid}: face processing took {dur_proc:.4f}s")

except KeyboardInterrupt:
    signal_handler(None, None)
except Exception as e:
    logging.error(f"Unhandled error in main loop: {e}", exc_info=True)
    signal_handler(None, None)
