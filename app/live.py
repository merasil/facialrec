import logging
import signal
import threading
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

from app.config import cfg_get, cfg_get_bool, cfg_get_float, cfg_get_int, cfg_pick
from app.face import face_find, face_load, face_missing, face_threshold
from include.functions import db_reset, door_open
from lib.motionchecker import MotChecker
from lib.streamreader import StrReader


LIVE_EXTS = (".jpg", ".jpeg", ".png")


def live_db(live_path: str) -> dict[str, dict[str, Any]]:
    live_root = Path(live_path)
    if not live_root.is_dir():
        raise ValueError(f"Database path does not exist: {live_path}")

    live_data = {}
    for live_folder in live_root.iterdir():
        if not live_folder.is_dir():
            continue
        live_image = next(
            (
                live_folder / f"{live_folder.name}{live_ext}"
                for live_ext in LIVE_EXTS
                if (live_folder / f"{live_folder.name}{live_ext}").is_file()
            ),
            None,
        )
        if live_image is None:
            logging.warning("No identity image found in %s", live_folder)
            continue
        live_data[live_folder.name] = {
            "path": str(live_image),
            "last_seen": datetime.now(),
            "last_opened": None,
            "cnt": 0,
        }
    if not live_data:
        raise ValueError(f"No valid identities found in database: {live_path}")
    return live_data


def live_run(live_args: Any, live_cfg: Any) -> int:
    live_verbose = cfg_pick(
        live_args.live_verbose,
        cfg_get_int(live_cfg, "basic", "verbose", 1),
    )
    if live_verbose < 0 or live_verbose > 4:
        raise ValueError("--verbose must be between 0 and 4")

    live_url = cfg_pick(
        live_args.live_url,
        cfg_get(live_cfg, "basic", "stream_url"),
    )
    live_low = cfg_pick(
        live_args.live_low,
        cfg_get(live_cfg, "basic", "stream_url_lowres", ""),
    )
    live_push = cfg_pick(
        live_args.live_push,
        cfg_get(live_cfg, "basic", "push_url"),
    )
    live_motion = cfg_pick(
        live_args.live_motion,
        cfg_get(live_cfg, "basic", "motion_url", "None"),
    )
    live_resize = cfg_pick(
        live_args.live_resize,
        cfg_get_bool(live_cfg, "basic", "stream_resize", False),
    )
    live_dbpath = cfg_pick(
        live_args.face_db,
        cfg_get(live_cfg, "database", "path"),
    )
    live_detector = cfg_pick(
        live_args.face_detector,
        cfg_get(live_cfg, "face_recognition", "detector_model"),
    )
    live_recognizer = cfg_pick(
        live_args.face_recognizer,
        cfg_get(live_cfg, "face_recognition", "recognition_model"),
    )
    live_metric = cfg_pick(
        live_args.face_metric,
        cfg_get(live_cfg, "face_recognition", "metric"),
    )
    live_align = cfg_pick(
        live_args.face_align,
        cfg_get_bool(live_cfg, "face_recognition", "alignment", False),
    )
    live_enforce = cfg_pick(
        live_args.face_enforce,
        cfg_get_bool(live_cfg, "face_recognition", "enforce", True),
    )
    live_clearance = cfg_pick(
        live_args.live_clearance,
        cfg_get_int(live_cfg, "thresholds", "clearance"),
    )
    live_last = cfg_pick(
        live_args.live_last,
        cfg_get_int(live_cfg, "thresholds", "last_seen"),
    )
    live_pretty = cfg_pick(
        live_args.live_pretty,
        cfg_get_float(live_cfg, "thresholds", "pretty_sure"),
    )
    live_door = cfg_pick(
        live_args.live_door,
        cfg_get_int(live_cfg, "thresholds", "open_door_cooldown", 30),
    )
    live_internal = cfg_pick(
        live_args.live_internal,
        cfg_get_bool(live_cfg, "motion", "use_internal", False),
    )
    live_mot_threshold = cfg_pick(
        live_args.live_mot_threshold,
        cfg_get_int(live_cfg, "motion", "threshold", 25),
    )
    live_mot_area = cfg_pick(
        live_args.live_mot_area,
        cfg_get_float(live_cfg, "motion", "min_area_percent", 0.2),
    )
    live_mot_cool = cfg_pick(
        live_args.live_mot_cool,
        cfg_get_int(live_cfg, "motion", "cooldown_seconds", 5),
    )
    live_mot_alpha = cfg_pick(
        live_args.live_mot_alpha,
        cfg_get_float(live_cfg, "motion", "background_alpha", 0.05),
    )

    logging.info("Loading detector and recognition models")
    face_load(live_detector, live_recognizer)

    live_data = live_db(live_dbpath)
    live_model_threshold = face_threshold(live_recognizer, live_metric)
    live_sure = live_model_threshold - (live_model_threshold * live_pretty)

    live_warm = next(iter(live_data.values()))["path"]
    try:
        face_find(
            live_warm,
            live_dbpath,
            live_detector,
            live_recognizer,
            live_metric,
            live_align,
            live_enforce,
            True,
        )
    except Exception as live_err:
        if not face_missing(live_err):
            raise
        logging.warning("Warm-up image contains no detectable face: %s", live_warm)
    logging.info("Models loaded")

    live_stream = StrReader(live_url)
    live_stream.str_start()
    live_mot_stream = None
    if live_internal:
        live_mot_stream = StrReader(live_low or live_url)
        live_mot_stream.str_start()
        live_checker = MotChecker(
            None,
            mot_stream=live_mot_stream,
            mot_internal=True,
            mot_threshold=live_mot_threshold,
            mot_area=live_mot_area,
            mot_cooldown=live_mot_cool,
            mot_verbose=live_verbose,
            mot_resize=live_resize,
            mot_alpha=live_mot_alpha,
        )
    else:
        live_checker = MotChecker(
            live_motion,
            mot_cooldown=live_mot_cool,
            mot_verbose=live_verbose,
        )
    live_checker.mot_start()

    live_stop = threading.Event()

    def live_shutdown(live_signum: Any, live_frame: Any) -> None:
        live_stop.set()

    signal.signal(signal.SIGINT, live_shutdown)
    signal.signal(signal.SIGTERM, live_shutdown)

    sleep(5)

    try:
        while not live_stop.is_set():
            live_wait = perf_counter()
            if not live_checker.mot_wait(mot_timeout=1):
                continue
            live_wait = perf_counter() - live_wait
            if live_verbose >= 4:
                logging.debug("Motion wait took %.4fs", live_wait)

            db_reset(live_data, live_last)
            live_frame = live_stream.str_read(str_timeout=5)
            if live_frame is None:
                logging.debug("No frame received after motion")
                continue

            live_start = perf_counter()
            try:
                live_faces = face_find(
                    live_frame,
                    live_dbpath,
                    live_detector,
                    live_recognizer,
                    live_metric,
                    live_align,
                    live_enforce,
                    False,
                )
            except Exception as live_err:
                if face_missing(live_err):
                    logging.debug("No face found")
                    continue
                logging.error("Face recognition failed: %s", live_err)
                continue

            live_total = len(live_faces)
            live_known = sum(1 for live_face in live_faces if not live_face.empty)
            live_unknown = live_total - live_known
            if live_verbose >= 2:
                logging.info("Detected %d face(s)", live_total)

            for live_face in live_faces:
                if live_face.empty:
                    continue
                live_path = str(live_face.iloc[0]["identity"])
                live_name = Path(live_path).parent.name
                if live_name not in live_data:
                    continue
                live_item = live_data[live_name]
                live_item["cnt"] += 1
                live_item["last_seen"] = datetime.now()
                live_distance = float(live_face.iloc[0]["distance"])
                if live_verbose >= 2:
                    logging.info(
                        "Recognized %s (distance %.4f, count %d)",
                        live_name,
                        live_distance,
                        live_item["cnt"],
                    )

                live_ready = (
                    live_distance <= live_sure
                    or live_item["cnt"] >= live_clearance
                )
                live_opened = live_item["last_opened"]
                live_cooled = (
                    live_opened is None
                    or (datetime.now() - live_opened).total_seconds() >= live_door
                )
                if live_ready and live_cooled:
                    live_item["last_opened"] = datetime.now()
                    threading.Thread(
                        target=door_open,
                        args=(live_name, live_push, live_verbose),
                        daemon=True,
                    ).start()

            if live_unknown > 0 and live_verbose >= 2:
                logging.info("Detected %d unknown face(s)", live_unknown)
            if live_verbose >= 4:
                logging.debug(
                    "Face recognition and processing took %.4fs",
                    perf_counter() - live_start,
                )
    finally:
        live_stream.str_stop()
        if live_mot_stream:
            live_mot_stream.str_stop()
        live_checker.mot_stop()
    return 0
