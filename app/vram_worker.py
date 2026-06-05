import json
import os
import sys
import threading
from time import sleep
from typing import Any, Optional

from app.face import face_find, face_load, face_missing, face_tf
from app.media import med_first


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
    vram_peak = [vram_base or 0]
    vram_stop = threading.Event()

    def vram_poll() -> None:
        while not vram_stop.is_set():
            vram_value = vram_used(vram_nv, vram_handle)
            if vram_value is not None:
                vram_peak[0] = max(vram_peak[0], vram_value)
            sleep(0.02)

    vram_thread = threading.Thread(target=vram_poll, daemon=True)
    vram_thread.start()
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(vram_gpu)
        vram_tf = face_tf(0)
        if not vram_tf.config.list_logical_devices("GPU"):
            raise RuntimeError("TensorFlow did not find a GPU")
        vram_tfbase = vram_tfinfo(vram_tf)["current"]

        face_load(vram_data["detector"], vram_data["recognizer"])
        vram_loaded = vram_used(vram_nv, vram_handle)
        vram_tfload = vram_tfinfo(vram_tf)["current"]
        try:
            vram_tf.config.experimental.reset_memory_stats("GPU:0")
        except Exception:
            pass

        vram_image = med_first(vram_data["input"])
        for vram_pos in range(int(vram_data["runs"]) + 1):
            try:
                face_find(
                    vram_image,
                    vram_data["db"],
                    vram_data["detector"],
                    vram_data["recognizer"],
                    vram_data["metric"],
                    bool(vram_data["align"]),
                    bool(vram_data["enforce"]),
                    vram_pos == 0,
                )
            except ValueError as vram_err:
                if not face_missing(vram_err):
                    raise
                pass

        vram_tfpeak = vram_tfinfo(vram_tf)["peak"]
        vram_final = vram_used(vram_nv, vram_handle)
    finally:
        vram_stop.set()
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
        print("VRAM_JSON=" + json.dumps(vram_result, sort_keys=True))
        return 0
    except Exception as vram_err:
        print(f"VRAM worker failed: {vram_err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(vram_main())
