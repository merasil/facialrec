import json
import os
import sys
import threading
from time import sleep
from typing import Any, Optional

from app.face import (
    face_find,
    face_load_detector,
    face_load_recognizer,
    face_missing,
    face_prepare_detector,
    face_torch_detector,
    face_tf,
)
from app.media import med_first


def vram_stage(vram_name: str) -> None:
    print(f"VRAM_STAGE={vram_name}", flush=True)


def vram_nvml(vram_gpu: int) -> tuple[Any, Any]:
    try:
        import pynvml
        pynvml.nvmlInit()
        vram_handle = pynvml.nvmlDeviceGetHandleByIndex(vram_gpu)
    except Exception:
        return None, None
    return pynvml, vram_handle


def vram_used(vram_nv: Any, vram_handle: Any) -> Optional[int]:
    if vram_nv is None:
        return None
    try:
        vram_items = vram_nv.nvmlDeviceGetComputeRunningProcesses(vram_handle)
        for vram_item in vram_items:
            if int(vram_item.pid) == os.getpid():
                return int(vram_item.usedGpuMemory)
    except Exception:
        return None
    return 0


def vram_tfinfo(vram_tf: Any) -> dict[str, int]:
    try:
        vram_info = vram_tf.config.experimental.get_memory_info("GPU:0")
    except Exception as vram_err:
        raise RuntimeError(
            f"TensorFlow memory information is unavailable: {vram_err}"
        ) from vram_err
    return {
        "current": int(vram_info["current"]),
        "peak": int(vram_info["peak"]),
    }


def vram_run(vram_data: dict[str, Any]) -> dict[str, Any]:
    vram_gpu = int(vram_data["gpu"])
    vram_nv, vram_handle = vram_nvml(vram_gpu)
    vram_base = vram_used(vram_nv, vram_handle)
    vram_stop = threading.Event()
    vram_thread = None
    vram_peak = [0]
    try:
        vram_stage("configure")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(vram_gpu)

        if face_torch_detector(vram_data["detector"]):
            vram_stage("detector runtime import")
            face_prepare_detector(vram_data["detector"])
            vram_stage("detector loading")
            face_load_detector(vram_data["detector"])
            vram_stage("tensorflow configure")
            vram_tf = face_tf(0)
        else:
            vram_stage("tensorflow configure")
            vram_tf = face_tf(0)
            vram_stage("detector loading")
            face_load_detector(vram_data["detector"])
        vram_stage("recognizer loading")
        face_load_recognizer(vram_data["recognizer"])
        if not vram_tf.config.list_logical_devices("GPU"):
            raise RuntimeError("TensorFlow did not find a GPU")
        vram_tfbase = 0
        vram_loaded = vram_used(vram_nv, vram_handle)
        vram_tfload = vram_tfinfo(vram_tf)["current"]

        # Build the DeepFace datastore and initialize all runtime workspaces
        # before peak measurement starts.
        vram_stage("warm-up")
        vram_image = med_first(vram_data["input"])
        try:
            face_find(
                vram_image,
                vram_data["db"],
                vram_data["detector"],
                vram_data["recognizer"],
                vram_data["metric"],
                bool(vram_data["align"]),
                bool(vram_data["enforce"]),
                True,
            )
        except Exception as vram_err:
            if not face_missing(vram_err):
                raise

        try:
            vram_tf.config.experimental.reset_memory_stats("GPU:0")
        except Exception:
            pass

        vram_current = vram_used(vram_nv, vram_handle)
        vram_peak[0] = vram_current or 0

        def vram_poll() -> None:
            while not vram_stop.is_set():
                vram_value = vram_used(vram_nv, vram_handle)
                if vram_value is not None:
                    vram_peak[0] = max(vram_peak[0], vram_value)
                sleep(0.02)

        vram_thread = threading.Thread(target=vram_poll, daemon=True)
        vram_thread.start()

        vram_stage("measurement")
        for vram_pos in range(int(vram_data["runs"])):
            try:
                face_find(
                    vram_image,
                    vram_data["db"],
                    vram_data["detector"],
                    vram_data["recognizer"],
                    vram_data["metric"],
                    bool(vram_data["align"]),
                    bool(vram_data["enforce"]),
                    False,
                )
            except Exception as vram_err:
                if not face_missing(vram_err):
                    raise
                pass

        vram_tfpeak = vram_tfinfo(vram_tf)["peak"]
        vram_final = vram_used(vram_nv, vram_handle)
        vram_stage("done")
    finally:
        vram_stop.set()
        if vram_thread is not None:
            vram_thread.join(timeout=1)
        if vram_nv is not None:
            try:
                vram_nv.nvmlShutdown()
            except Exception:
                pass

    return {
        "detector": vram_data["detector"],
        "recognizer": vram_data["recognizer"],
        "tf_base": vram_tfbase,
        "tf_loaded": vram_tfload,
        "tf_peak": vram_tfpeak,
        "nv_base": vram_base,
        "nv_loaded": vram_loaded,
        "nv_final": vram_final,
        "nv_peak": vram_peak[0] if vram_nv is not None else None,
    }


def vram_main() -> int:
    if len(sys.argv) != 2:
        print("Expected one JSON payload", file=sys.stderr)
        return 2
    try:
        vram_data = json.loads(sys.argv[1])
        vram_result = vram_run(vram_data)
        vram_payload = {"ok": True, "result": vram_result}
    except Exception as vram_err:
        vram_payload = {
            "ok": False,
            "error": " ".join(str(vram_err).split()),
            "type": type(vram_err).__name__,
        }
    print("VRAM_JSON=" + json.dumps(vram_payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(vram_main())
