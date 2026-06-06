from time import sleep, perf_counter
from datetime import datetime
import os
import signal
import sys
import logging
import threading
import configparser
from include.functions import *
from include.face_runtime import face_load
from lib.streamreader import StreamReader
from lib.motionchecker import MotionChecker

# Read config file first to get verbose setting
config = configparser.ConfigParser()
config_path = "./config/config.ini"

if not os.path.exists(config_path):
    print(f"ERROR: Config file not found at {config_path}", file=sys.stderr)
    print("Please create config.ini based on config-example.ini", file=sys.stderr)
    sys.exit(1)

config.read(config_path)

try:
    verbose = int(config.get("basic", "verbose", fallback="1"))
    if verbose < 0 or verbose > 4:
        print(f"ERROR: verbose level must be 0-4, got {verbose}", file=sys.stderr)
        sys.exit(1)
except KeyError as e:
    print(f"ERROR: Missing required config key: {e}", file=sys.stderr)
    sys.exit(1)
except ValueError as e:
    print(f"ERROR: verbose level must be an integer: {e}", file=sys.stderr)
    sys.exit(1)

# Logging setup based on verbose level (level 4 enables DEBUG logging)
log_level = logging.DEBUG if verbose >= 4 else logging.INFO
logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Database setup
path_db = config["database"]["path"]

if not os.path.exists(path_db):
    logging.error(f"Database path does not exist: {path_db}")
    sys.exit(1)

db = {}
supported_extensions = (".jpg", ".jpeg", ".png")
with os.scandir(path_db) as entries:
    for folder in entries:
        if not folder.is_dir():
            continue
        img_path = None
        for ext in supported_extensions:
            candidate = os.path.join(path_db, folder.name, folder.name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path:
            db[folder.name] = {
                "path": img_path,
                "last_seen": datetime.now(),
                "last_opened": None,
                "cnt": 0
            }
            logging.debug(f"Loaded identity: {folder.name}")
        else:
            logging.warning(f"No image found for {folder.name} in {path_db}/{folder.name}/ (supported: {', '.join(supported_extensions)})")

if not db:
    logging.error("No valid identities found in database!")
    sys.exit(1)

# Basic setup
try:
    stream_url = config["basic"]["stream_url"]
    stream_url_lowres = config.get("basic", "stream_url_lowres", fallback="").strip()
    push_url = config["basic"]["push_url"]
    motion_url = config["basic"]["motion_url"]
    stream_resize = str2bool(config.get("basic", "stream_resize", fallback="False"))
except KeyError as e:
    logging.error(f"Missing required basic config: {e}")
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
    open_door_cooldown = int(config.get("thresholds", "open_door_cooldown", fallback="30"))
except KeyError as e:
    logging.error(f"Missing required thresholds config: {e}")
    sys.exit(1)

# Motion detection setup
try:
    use_internal_motion = str2bool(config.get("motion", "use_internal", fallback="False"))
    motion_threshold = int(config.get("motion", "threshold", fallback="25"))
    motion_min_area = float(config.get("motion", "min_area_percent", fallback="0.2"))
    motion_cooldown = int(config.get("motion", "cooldown_seconds", fallback="5"))
    motion_bg_alpha = float(config.get("motion", "background_alpha", fallback="0.05"))
except (KeyError, ValueError) as e:
    logging.warning(f"Motion config error, using defaults: {e}")
    use_internal_motion = False
    motion_threshold = 25
    motion_min_area = 0.2
    motion_cooldown = 5
    motion_bg_alpha = 0.05

# Initialize StreamReaders
# Main stream for face recognition
stream = StreamReader(stream_url)
stream.start()

# Dedicated stream for motion detection (separate from main stream to avoid queue contention)
stream_motion = None
if use_internal_motion:
    motion_stream_url = stream_url_lowres if stream_url_lowres else stream_url
    if stream_url_lowres:
        logging.info(f"Using separate low-res stream for motion detection: {motion_stream_url}")
    else:
        logging.info("Using second connection to main stream for motion detection (no low-res stream configured)")
    stream_motion = StreamReader(motion_stream_url)
    stream_motion.start()

# Initialize MotionChecker (internal or external)
if use_internal_motion:
    logging.info("Using internal motion detection")
    motion = MotionChecker(
        motion_url=None,
        stream_reader=stream_motion,
        use_internal=True,
        threshold=motion_threshold,
        min_area=motion_min_area,
        cooldown_seconds=motion_cooldown,
        verbose=verbose,
        resize=stream_resize,
        background_alpha=motion_bg_alpha
    )
else:
    logging.info("Using external motion detection (Frigate)")
    motion = MotionChecker(motion_url, cooldown_seconds=motion_cooldown, verbose=verbose)
motion.start()

# Warm-up
sleep(5)
logging.info(f"Database: {db}")
logging.info("Loading Model...")
try:
    DeepFace = face_load(detector_model, recognition_model)
    threshold_model = DeepFace.verification.find_threshold(recognition_model, metric)
    threshold_pretty_sure = threshold_model - (
        threshold_model * pretty_sure_factor
    )
except Exception as e:
    logging.error(
        f"Error loading detector '{detector_model}' and recognition model "
        f"'{recognition_model}': {e}"
    )
    logging.error("Valid metrics are usually: cosine, euclidean, euclidean_l2")
    sys.exit(1)
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

        if verbose >= 4:
            logging.debug(f"wait_for_motion took {dur_wait:.4f}s")

        start = perf_counter()
        resetDB(db, threshold_last_seen)
        dur_reset = perf_counter() - start

        if verbose >= 4:
            logging.debug(f"resetDB took {dur_reset:.4f}s")

        # Read frame
        start = perf_counter()
        frame = stream.read(timeout=5)
        dur_read = perf_counter() - start
        if verbose >= 4:
            logging.debug(f"stream.read took {dur_read:.4f}s")

        if frame is None:
            if verbose >= 4:
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
            if verbose >= 4:
                logging.debug("No Face found! Continuing...")
                logging.debug(e)
            continue
        except Exception as e:
            logging.error(f"Error during face recognition: {e}")
            continue
        dur_find = perf_counter() - start
        if verbose >= 4:
            logging.debug(f"DeepFace.find took {dur_find:.4f}s")

        total_faces = len(faces)
        recognized_count = sum(1 for face in faces if not face.empty)
        unknown_count = total_faces - recognized_count
        if verbose >= 2:
            logging.info(f"Detected {total_faces} face(s)")

        # Process faces
        start = perf_counter()
        for face in faces:
            if face.empty:
                continue
            identity = face.iloc[0]["identity"].split('/')[-2]
            if identity in db:
                db[identity]["cnt"] += 1
                db[identity]["last_seen"] = datetime.now()
                if verbose >= 2:
                    distance = face.iloc[0]["distance"]
                    logging.info(f"Recognized: {identity} (distance: {distance:.4f}, count: {db[identity]['cnt']})")
                if face.iloc[0]["distance"] <= threshold_pretty_sure or db[identity]["cnt"] >= threshold_clearance:
                    last_opened = db[identity]["last_opened"]
                    if last_opened is None or (datetime.now() - last_opened).total_seconds() >= open_door_cooldown:
                        db[identity]["last_opened"] = datetime.now()
                        threading.Thread(target=openDoor, args=(identity, push_url, verbose), daemon=True).start()

        # Log unknown faces count if any
        if unknown_count > 0 and verbose >= 2:
            logging.info(f"Detected {unknown_count} unknown face{'s' if unknown_count != 1 else ''}")

        dur_proc = perf_counter() - start

        if verbose >= 4:
            logging.debug(f"Face processing took {dur_proc:.4f}s")

except KeyboardInterrupt:
    signal_handler(None, None)
except Exception as e:
    logging.error(f"Unhandled error in main loop: {e}", exc_info=True)
    signal_handler(None, None)
