import os
import signal
import sys
import logging
import configparser

from datetime import datetime
from time import sleep
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import tensorflow as tf
from deepface import DeepFace

from include.functions import *
from lib.streamreader import StreamReader
from lib.motionchecker import MotionChecker

gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    logging.info("Couldn't set Memory Growth for GPU or no GPU found. Continuing...")

############## Setting up Logging #############
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

############## Reading Config-File #############
config = configparser.ConfigParser()
config.read("config.ini")

stream_url = config["basic"]["stream_url"]
push_url = config["basic"]["push_url"]
motion_url = config["basic"]["motion_url"]
debug = str2bool(config["basic"]["debug"])

############## Setting up Database #############
db_path = config["database"]["path"]
db = {}
for folder in os.scandir(db_path):
        if folder.is_dir():
            db[folder.name] = {"path":"{}/{}/{}.jpg".format(db_path,folder.name,folder.name), "last_seen":datetime.now(), "cnt":0}

############## Setting up Face Recognition Model #############
model = config["face_recognition"]["model"]
if model == "yunet":
    os.environ["yunet_score_threshold"] = "0.8"
detector = config["face_recognition"]["detector"]
metric = config["face_recognition"]["metric"]
alignment = str2bool(config["face_recognition"]["alignment"])
face_detect_enf = str2bool(config["face_recognition"]["face_detection_enforce"])

############## Setting up Thresholds #############
threshold_model = DeepFace.verification.find_threshold(model, metric)
threshold_clearance = int(config["thresholds"]["clearance"])
threshold_last_seen = int(config["thresholds"]["last_seen"])
threshold_pretty_sure = threshold_model - (threshold_model * float(config["thresholds"]["pretty_sure"]))

############## Settings for Camera (URL, Thread, etc.) #############
stream = StreamReader(stream_url)
stream.start()

############## Settings for MotionChecker (URL, Thread, etc.) ##############
motion = MotionChecker(motion_url)
motion.start()

############## Settings for ThreadPoolExecuter ##############
MAX_WORKERS = os.getenv('MAX_WORKERS', 4)
executor = ThreadPoolExecutor(max_workers=int(MAX_WORKERS))
futures = set()

sleep(5)
logging.info("Database: {}".format(db))
logging.info("Loading Model...")
DeepFace.build_model(model)
logging.info("Finished loading Model...")

############## Setting up Signal Handler #############
def signal_handler(sig, frame):
    logging.info("Killing Process...")
    stream.stop()
    motion.stop()
    executor.shutdown(wait=False)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

############## Function for Processing Frames #############
def process_frame(frame, db_snapshot):
    import tensorflow as tf
    from deepface import DeepFace
    from include.functions import openDoor
    from datetime import datetime
    detector = db_snapshot['detector']
    model_name = db_snapshot['model_name']
    metric = db_snapshot['metric']
    alignment = db_snapshot['alignment']
    face_detect_enf = db_snapshot['face_detect_enf']
    threshold_pretty_sure = db_snapshot['threshold_pretty_sure']
    threshold_clearance = db_snapshot['threshold_clearance']
    push_url = db_snapshot['push_url']
    db_entries = db_snapshot['db_entries']
    try:
        faces = DeepFace.find(
            img_path=frame,
            detector_backend=detector,
            align=alignment,
            enforce_detection=face_detect_enf,
            db_path=db_snapshot['db_path'],
            distance_metric=metric,
            model_name=model_name,
            silent=True
        )
        for face in faces:
            if face.empty:
                continue
            identity = face.iloc[0]["identity"]
            dist = face.iloc[0]["distance"]
            if identity in db_entries:
                db_entries[identity]["cnt"] += 1
                db_entries[identity]["last_seen"] = datetime.now()
                if dist <= threshold_pretty_sure or db_entries[identity]["cnt"] >= threshold_clearance:
                    openDoor(identity, push_url)
    except Exception as e:
        logging.error(f"Error processing frame in worker: {e}")

# Package necessary context for worker processes
context = {
    'detector': detector,
    'model_name': model,
    'metric': metric,
    'alignment': alignment,
    'face_detect_enf': face_detect_enf,
    'threshold_pretty_sure': threshold_pretty_sure,
    'threshold_clearance': threshold_clearance,
    'push_url': push_url,
    'db_path': db_path,
    'db_entries': db
}

############## Starting the Application #############
try:
    while True:
        resetDB(db, threshold_last_seen)
        if not motion.result:
            if debug:
                logging.info("No Motion detected. Continuing with next...")
            sleep(1.0)
            continue

        frame = stream.read()
        if frame is None:
            if debug:
                logging.error("Couldn't receive Frame. Continuing with next...")
            continue

        done, futures = wait(futures, timeout=0, return_when=FIRST_COMPLETED)
        futures -= done

        if debug:
                logging.info("Starting Frame analyzing Worker...")
        future = executor.submit(process_frame, frame, context)
        futures.add(future)

        if len(futures) >= MAX_WORKERS:
            if debug:
                logging.info("No more Workers available! Waiting for one to finish...")
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            futures -= done

except KeyboardInterrupt:
    signal_handler(None, None)
except Exception as e:
    logging.error(f"Unhandled error in main loop: {e}", exc_info=True)
    signal_handler(None, None)
