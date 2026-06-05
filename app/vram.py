import itertools
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from app.config import cfg_get, cfg_get_bool, cfg_get_int, cfg_list, cfg_pick
from app.table import tab_output, tab_render


VRAM_HEADS = [
    "Detector",
    "Recognizer",
    "TF base",
    "TF loaded",
    "TF peak",
    "TF delta",
    "NVML base",
    "NVML loaded",
    "NVML peak",
    "NVML delta",
    "Status",
]


def vram_mb(vram_value: Optional[int]) -> str:
    if vram_value is None:
        return "N/A"
    return f"{vram_value / (1024 * 1024):.1f}"


def vram_delta(vram_peak: Optional[int], vram_base: Optional[int]) -> str:
    if vram_peak is None or vram_base is None:
        return "N/A"
    return vram_mb(max(0, vram_peak - vram_base))


def vram_worker(vram_data: dict[str, Any]) -> tuple[Optional[dict[str, Any]], str]:
    vram_root = Path(__file__).resolve().parent.parent
    vram_cmd = [
        sys.executable,
        "-m",
        "app.vram_worker",
        json.dumps(vram_data),
    ]
    vram_proc = subprocess.run(
        vram_cmd,
        check=False,
        capture_output=True,
        text=True,
        cwd=vram_root,
    )

    for vram_line in reversed(vram_proc.stdout.splitlines()):
        if vram_line.startswith("VRAM_JSON="):
            vram_payload = json.loads(vram_line.removeprefix("VRAM_JSON="))
            if "ok" not in vram_payload:
                return vram_payload, "ok"
            if vram_payload["ok"]:
                return vram_payload["result"], "ok"
            return None, str(vram_payload.get("error", "worker failed"))

    if vram_proc.returncode != 0:
        vram_stage = next(
            (
                vram_line.removeprefix("VRAM_STAGE=").strip()
                for vram_line in reversed(vram_proc.stdout.splitlines())
                if vram_line.startswith("VRAM_STAGE=")
            ),
            "",
        )
        vram_lines = [
            vram_line.strip()
            for vram_line in vram_proc.stderr.splitlines()
            if vram_line.strip()
        ]
        vram_detail = next(
            (
                vram_line.removeprefix("VRAM worker failed:").strip()
                for vram_line in reversed(vram_lines)
                if vram_line.startswith("VRAM worker failed:")
            ),
            "",
        )
        vram_error = f"worker exit {vram_proc.returncode}"
        if vram_stage:
            vram_error += f" during {vram_stage}"
        if vram_detail:
            vram_error += f": {vram_detail}"
        return None, vram_error
    return None, "worker returned no result"


def vram_run(vram_args: Any, vram_cfg: Any) -> int:
    vram_db = cfg_pick(
        vram_args.face_db,
        cfg_get(vram_cfg, "database", "path"),
    )
    vram_detectors = cfg_list(
        vram_args.face_detectors,
        cfg_get(
            vram_cfg,
            "vram",
            "detectors",
            cfg_get(vram_cfg, "face_recognition", "detector_model"),
        ),
    )
    vram_recognizers = cfg_list(
        vram_args.face_recognizers,
        cfg_get(
            vram_cfg,
            "vram",
            "recognizers",
            cfg_get(vram_cfg, "face_recognition", "recognition_model"),
        ),
    )
    vram_metric = cfg_pick(
        vram_args.face_metric,
        cfg_get(vram_cfg, "face_recognition", "metric"),
    )
    vram_align = cfg_pick(
        vram_args.face_align,
        cfg_get_bool(vram_cfg, "face_recognition", "alignment", False),
    )
    vram_enforce = cfg_pick(
        vram_args.face_enforce,
        cfg_get_bool(vram_cfg, "face_recognition", "enforce", True),
    )
    vram_runs = cfg_pick(
        vram_args.vram_runs,
        cfg_get_int(vram_cfg, "vram", "runs", 3),
    )
    vram_gpu = cfg_pick(
        vram_args.vram_gpu,
        cfg_get_int(vram_cfg, "vram", "gpu", 0),
    )

    if not vram_detectors or not vram_recognizers:
        raise ValueError("Detector and recognizer lists must not be empty")
    if not Path(vram_args.vram_input).exists():
        raise ValueError(f"Input does not exist: {vram_args.vram_input}")
    if not Path(vram_db).is_dir():
        raise ValueError(f"Database directory does not exist: {vram_db}")
    if vram_runs <= 0:
        raise ValueError("--runs must be greater than zero")
    if vram_gpu < 0:
        raise ValueError("--gpu must not be negative")

    vram_rows = []
    for vram_detector, vram_recognizer in itertools.product(
        vram_detectors,
        vram_recognizers,
    ):
        logging.info(
            "Measuring VRAM for detector=%s recognizer=%s",
            vram_detector,
            vram_recognizer,
        )
        vram_data = {
            "input": str(Path(vram_args.vram_input).resolve()),
            "db": str(Path(vram_db).resolve()),
            "detector": vram_detector,
            "recognizer": vram_recognizer,
            "metric": vram_metric,
            "align": vram_align,
            "enforce": vram_enforce,
            "runs": vram_runs,
            "gpu": vram_gpu,
        }
        vram_result, vram_status = vram_worker(vram_data)
        if vram_result is None:
            vram_rows.append(
                [
                    vram_detector,
                    vram_recognizer,
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    vram_status,
                ]
            )
            continue

        vram_rows.append(
            [
                vram_detector,
                vram_recognizer,
                vram_mb(vram_result["tf_base"]),
                vram_mb(vram_result["tf_loaded"]),
                vram_mb(vram_result["tf_peak"]),
                vram_delta(vram_result["tf_peak"], vram_result["tf_base"]),
                vram_mb(vram_result["nv_base"]),
                vram_mb(vram_result["nv_loaded"]),
                vram_mb(vram_result["nv_peak"]),
                vram_delta(vram_result["nv_peak"], vram_result["nv_base"]),
                vram_status,
            ]
        )

    vram_table = tab_render(VRAM_HEADS, vram_rows)
    tab_output(vram_table, vram_args.tab_output)
    return 0
