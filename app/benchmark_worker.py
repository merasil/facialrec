import faulthandler
import importlib.metadata
import json
import os
import platform
import sys
from typing import Any

from app.benchmark import bench_one
from app.media import med_first


def bench_stage(bench_name: str) -> None:
    print(f"BENCH_STAGE={bench_name}", flush=True)


def bench_diagnostics() -> None:
    bench_packages = {}
    for bench_package in (
        "deepface",
        "ultralytics",
        "torch",
        "torchvision",
        "triton",
        "tensorflow",
        "tf-keras",
        "numpy",
        "opencv-python",
        "opencv-python-headless",
        "facenet-pytorch",
        "nvidia-cublas-cu12",
        "nvidia-cuda-cupti-cu12",
        "nvidia-cuda-nvcc-cu12",
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu12",
        "nvidia-cufft-cu12",
        "nvidia-cufile-cu12",
        "nvidia-curand-cu12",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu12",
        "nvidia-cusparselt-cu12",
        "nvidia-nccl-cu12",
        "nvidia-nvjitlink-cu12",
        "nvidia-nvshmem-cu12",
    ):
        try:
            bench_packages[bench_package] = importlib.metadata.version(bench_package)
        except importlib.metadata.PackageNotFoundError:
            bench_packages[bench_package] = "not installed"

    bench_info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        "packages": bench_packages,
    }
    print(
        "BENCH_DIAG=" + json.dumps(bench_info, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def bench_run(bench_data: dict[str, Any]) -> list[str]:
    bench_stage("input loading")
    bench_first = med_first(bench_data["input"])
    return bench_one(
        bench_data["input"],
        bench_data["name"],
        bench_data["db"],
        bench_data["detector"],
        bench_data["recognizer"],
        bench_data["metric"],
        bool(bench_data["align"]),
        bool(bench_data["enforce"]),
        bench_first,
        bench_stage,
    )


def bench_main() -> int:
    faulthandler.enable(all_threads=True)
    if len(sys.argv) != 2:
        print("Expected one JSON payload", file=sys.stderr)
        return 2
    bench_diagnostics()
    try:
        bench_data = json.loads(sys.argv[1])
        bench_row = bench_run(bench_data)
        bench_result = {"ok": True, "row": bench_row}
    except Exception as bench_err:
        bench_result = {
            "ok": False,
            "error": " ".join(str(bench_err).split()),
            "type": type(bench_err).__name__,
        }
    print("BENCH_JSON=" + json.dumps(bench_result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(bench_main())
