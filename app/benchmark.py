import itertools
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from app.config import cfg_get, cfg_get_bool, cfg_list, cfg_pick
from app.face import face_find, face_load, face_missing, face_result, face_tf
from app.media import med_first, med_iter
from app.table import tab_output, tab_render


BENCH_HEADS = [
    "Detector",
    "Recognizer",
    "Metric",
    "Inputs",
    "Detected",
    "Det %",
    "Correct",
    "Rec %",
    "Wrong",
    "Unknown",
    "Errors",
    "Total s",
    "ms/input",
    "Status",
]


def bench_rate(bench_count: int, bench_total: int) -> str:
    if bench_total == 0:
        return "0.00"
    return f"{(bench_count / bench_total) * 100:.2f}"


def bench_status(bench_err: Exception, bench_detector: str) -> str:
    bench_text = " ".join(str(bench_err).split())
    bench_lower = bench_text.lower()
    if "facenet_pytorch" in bench_lower or "facenet-pytorch" in bench_lower:
        return "missing facenet-pytorch"
    bench_missing = {
        "dlib": "missing dlib",
        "mediapipe": "missing mediapipe",
        "ultralytics": "missing ultralytics",
    }
    for bench_module, bench_hint in bench_missing.items():
        if bench_module in bench_lower:
            return bench_hint
    if bench_detector == "yolov8":
        return "invalid detector; use yolov8n, yolov8m or yolov8l"
    if "invalid model_name" in bench_lower or "unimplemented" in bench_lower:
        return f"invalid detector: {bench_detector}"
    if not bench_text:
        return type(bench_err).__name__
    return bench_text[:120]


def bench_warm(
    bench_image: Any,
    bench_db: str,
    bench_detector: str,
    bench_recognizer: str,
    bench_metric: str,
    bench_align: bool,
    bench_enforce: bool,
) -> None:
    face_load(bench_detector, bench_recognizer)
    try:
        face_find(
            bench_image,
            bench_db,
            bench_detector,
            bench_recognizer,
            bench_metric,
            bench_align,
            bench_enforce,
            True,
        )
    except Exception as bench_err:
        if not face_missing(bench_err):
            raise
        logging.warning(
            "Warm-up input had no detectable face for %s/%s/%s",
            bench_detector,
            bench_recognizer,
            bench_metric,
        )


def bench_one(
    bench_input: str,
    bench_name: str,
    bench_db: str,
    bench_detector: str,
    bench_recognizer: str,
    bench_metric: str,
    bench_align: bool,
    bench_enforce: bool,
    bench_first: Any,
) -> list[str]:
    logging.info(
        "Benchmarking detector=%s recognizer=%s metric=%s",
        bench_detector,
        bench_recognizer,
        bench_metric,
    )
    bench_warm(
        bench_first,
        bench_db,
        bench_detector,
        bench_recognizer,
        bench_metric,
        bench_align,
        bench_enforce,
    )

    bench_total = 0
    bench_detected = 0
    bench_correct = 0
    bench_wrong = 0
    bench_unknown = 0
    bench_errors = 0
    bench_time = 0.0

    for bench_pos, bench_image in med_iter(bench_input):
        bench_total += 1
        bench_start = perf_counter()
        try:
            bench_frames = face_find(
                bench_image,
                bench_db,
                bench_detector,
                bench_recognizer,
                bench_metric,
                bench_align,
                bench_enforce,
                False,
            )
        except Exception as bench_err:
            bench_time += perf_counter() - bench_start
            if face_missing(bench_err):
                continue
            bench_errors += 1
            logging.debug("Benchmark input %d failed: %s", bench_pos, bench_err)
            continue
        bench_time += perf_counter() - bench_start

        bench_found, bench_any, bench_match = face_result(bench_frames, bench_name)
        if not bench_found:
            continue
        bench_detected += 1
        if bench_match:
            bench_correct += 1
        elif bench_any:
            bench_wrong += 1
        else:
            bench_unknown += 1

    bench_avg = (bench_time / bench_total) * 1000 if bench_total else 0.0
    return [
        bench_detector,
        bench_recognizer,
        bench_metric,
        str(bench_total),
        str(bench_detected),
        bench_rate(bench_detected, bench_total),
        str(bench_correct),
        bench_rate(bench_correct, bench_detected),
        str(bench_wrong),
        str(bench_unknown),
        str(bench_errors),
        f"{bench_time:.3f}",
        f"{bench_avg:.2f}",
        "ok",
    ]


def bench_run(bench_args: Any, bench_cfg: Any) -> int:
    bench_db = cfg_pick(
        bench_args.face_db,
        cfg_get(bench_cfg, "database", "path"),
    )
    bench_detectors = cfg_list(
        bench_args.face_detectors,
        cfg_get(
            bench_cfg,
            "benchmark",
            "detectors",
            cfg_get(bench_cfg, "face_recognition", "detector_model"),
        ),
    )
    bench_recognizers = cfg_list(
        bench_args.face_recognizers,
        cfg_get(
            bench_cfg,
            "benchmark",
            "recognizers",
            cfg_get(bench_cfg, "face_recognition", "recognition_model"),
        ),
    )
    bench_metrics = cfg_list(
        bench_args.face_metrics,
        cfg_get(
            bench_cfg,
            "benchmark",
            "metrics",
            cfg_get(bench_cfg, "face_recognition", "metric"),
        ),
    )
    bench_align = cfg_pick(
        bench_args.face_align,
        cfg_get_bool(bench_cfg, "face_recognition", "alignment", False),
    )
    bench_enforce = cfg_pick(
        bench_args.face_enforce,
        cfg_get_bool(bench_cfg, "face_recognition", "enforce", True),
    )

    if not bench_detectors or not bench_recognizers or not bench_metrics:
        raise ValueError("Detector, recognizer and metric lists must not be empty")
    if not Path(bench_args.bench_input).exists():
        raise ValueError(f"Input does not exist: {bench_args.bench_input}")
    if not Path(bench_db).is_dir():
        raise ValueError(f"Database directory does not exist: {bench_db}")
    if not (Path(bench_db) / bench_args.bench_name).is_dir():
        raise ValueError(
            f"Expected identity is not an exact database folder: {bench_args.bench_name}"
        )

    face_tf()
    bench_first = med_first(bench_args.bench_input)
    bench_rows = []
    bench_combos = itertools.product(
        bench_detectors,
        bench_recognizers,
        bench_metrics,
    )
    for bench_detector, bench_recognizer, bench_metric in bench_combos:
        try:
            bench_row = bench_one(
                bench_args.bench_input,
                bench_args.bench_name,
                bench_db,
                bench_detector,
                bench_recognizer,
                bench_metric,
                bench_align,
                bench_enforce,
                bench_first,
            )
        except Exception as bench_err:
            logging.exception(
                "Model combination failed: %s/%s/%s",
                bench_detector,
                bench_recognizer,
                bench_metric,
            )
            bench_row = [
                bench_detector,
                bench_recognizer,
                bench_metric,
                "0",
                "0",
                "0.00",
                "0",
                "0.00",
                "0",
                "0",
                "1",
                "0.000",
                "0.00",
                bench_status(bench_err, bench_detector),
            ]
            logging.error("Combination error: %s", bench_err)
        bench_rows.append(bench_row)

    bench_table = tab_render(BENCH_HEADS, bench_rows)
    tab_output(bench_table, bench_args.tab_output)
    return 0
