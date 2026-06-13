#!/usr/bin/env python3
"""Runtime diagnostics and startup preflight for FacialRec."""

import argparse
import configparser
import faulthandler
import json
import logging
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit


DEFAULT_CONFIG_PATH = "./config/config.ini"
DEFAULT_TIMEOUT = 30
DEFAULT_GPU_TIMEOUT = 300
PROCESS_STOP_TIMEOUT = 10
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
TRUE_VALUES = {"1", "true", "yes", "on", "y", "t"}
FALSE_VALUES = {"0", "false", "no", "off", "n", "f"}


class CheckError(RuntimeError):
    """Raised when a runtime check fails."""


class CheckRunner:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, name: str, function, *args, **kwargs) -> bool:
        print(f"\n--- {name} ---", flush=True)
        try:
            detail = function(*args, **kwargs)
        except Exception as error:
            self.failed += 1
            print(f"[FAIL] {name}: {error}", file=sys.stderr, flush=True)
            return False

        self.passed += 1
        suffix = f": {detail}" if detail else ""
        print(f"[PASS] {name}{suffix}", flush=True)
        return True

    def summary(self) -> bool:
        print(
            f"\nTest summary: {self.passed} passed, {self.failed} failed",
            flush=True,
        )
        return self.failed == 0


def parse_bool(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise CheckError(f"expected a boolean value, got {value!r}")


def required(config: configparser.ConfigParser, section: str, option: str) -> str:
    if not config.has_section(section):
        raise CheckError(f"missing section [{section}]")
    if not config.has_option(section, option):
        raise CheckError(f"missing option {section}.{option}")
    value = config.get(section, option).strip()
    if not value:
        raise CheckError(f"empty option {section}.{option}")
    return value


def require_range(name: str, value: float, minimum: float, maximum: float) -> None:
    if value < minimum or value > maximum:
        raise CheckError(f"{name} must be between {minimum} and {maximum}, got {value}")


def require_minimum(name: str, value: float, minimum: float) -> None:
    if value < minimum:
        raise CheckError(f"{name} must be at least {minimum}, got {value}")


def validate_url(name: str, value: str, schemes: set[str]) -> None:
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in schemes or not parsed.hostname:
        expected = ", ".join(sorted(schemes))
        raise CheckError(f"{name} must be a valid {expected} URL")
    try:
        parsed.port
    except ValueError as error:
        raise CheckError(f"{name} contains an invalid port: {error}") from error


def display_url(value: str) -> str:
    parsed = urlsplit(value)
    if not parsed.password:
        return value
    hostname = parsed.hostname or ""
    if parsed.port:
        hostname = f"{hostname}:{parsed.port}"
    username = f"{parsed.username}:***@" if parsed.username else ""
    return urlunsplit(
        (parsed.scheme, f"{username}{hostname}", parsed.path, parsed.query, parsed.fragment)
    )


def load_settings(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise CheckError(
            f"config file not found: {path}. Create it from config-example.ini"
        )

    config = configparser.ConfigParser()
    try:
        loaded = config.read(path)
    except configparser.Error as error:
        raise CheckError(f"cannot parse {path}: {error}") from error
    if not loaded:
        raise CheckError(f"cannot read config file: {path}")

    stream_url = required(config, "basic", "stream_url")
    stream_url_lowres = config.get("basic", "stream_url_lowres", fallback="").strip()
    push_url = required(config, "basic", "push_url")
    motion_url = required(config, "basic", "motion_url")
    detector_model = required(config, "face_recognition", "detector_model")
    recognition_model = required(config, "face_recognition", "recognition_model")
    metric = required(config, "face_recognition", "metric")
    database_path = required(config, "database", "path")
    alignment = parse_bool(required(config, "face_recognition", "alignment"))
    enforce = parse_bool(required(config, "face_recognition", "enforce"))

    try:
        verbose = config.getint("basic", "verbose", fallback=1)
        clearance = config.getint("thresholds", "clearance")
        last_seen = config.getint("thresholds", "last_seen")
        pretty_sure = config.getfloat("thresholds", "pretty_sure")
        open_door_cooldown = config.getint(
            "thresholds", "open_door_cooldown", fallback=30
        )
        motion_threshold = config.getint("motion", "threshold", fallback=25)
        motion_min_area = config.getfloat(
            "motion", "min_area_percent", fallback=0.2
        )
        motion_cooldown = config.getint(
            "motion", "cooldown_seconds", fallback=5
        )
        background_alpha = config.getfloat(
            "motion", "background_alpha", fallback=0.05
        )
    except (ValueError, configparser.Error) as error:
        raise CheckError(f"invalid numeric config value: {error}") from error

    use_internal_motion = parse_bool(
        config.get("motion", "use_internal", fallback="False")
    )
    stream_resize = parse_bool(
        config.get("basic", "stream_resize", fallback="False")
    )

    validate_url("basic.stream_url", stream_url, {"rtsp", "rtsps"})
    if stream_url_lowres:
        validate_url(
            "basic.stream_url_lowres", stream_url_lowres, {"rtsp", "rtsps"}
        )
    validate_url("basic.push_url", push_url, {"http", "https"})
    if not use_internal_motion:
        if motion_url.lower() == "none":
            raise CheckError(
                "basic.motion_url is required when motion.use_internal is False"
            )
        validate_url("basic.motion_url", motion_url, {"http", "https"})

    require_range("basic.verbose", verbose, 0, 4)
    require_minimum("thresholds.clearance", clearance, 1)
    require_minimum("thresholds.last_seen", last_seen, 1)
    require_range("thresholds.pretty_sure", pretty_sure, 0.0, 1.0)
    require_minimum("thresholds.open_door_cooldown", open_door_cooldown, 0)
    require_range("motion.threshold", motion_threshold, 0, 255)
    require_range("motion.min_area_percent", motion_min_area, 0.0, 100.0)
    require_minimum("motion.cooldown_seconds", motion_cooldown, 0)
    require_range("motion.background_alpha", background_alpha, 0.0, 1.0)

    return {
        "config_path": str(path),
        "stream_url": stream_url,
        "stream_url_lowres": stream_url_lowres,
        "push_url": push_url,
        "motion_url": motion_url,
        "verbose": verbose,
        "detector_model": detector_model,
        "recognition_model": recognition_model,
        "metric": metric,
        "alignment": alignment,
        "enforce": enforce,
        "database_path": database_path,
        "use_internal_motion": use_internal_motion,
        "stream_resize": stream_resize,
        "motion_threshold": motion_threshold,
        "motion_min_area": motion_min_area,
        "motion_cooldown": motion_cooldown,
        "background_alpha": background_alpha,
    }


def identity_image_paths(database_path: Path):
    for folder in sorted(database_path.iterdir()):
        if not folder.is_dir():
            continue
        image_path = next(
            (
                folder / f"{folder.name}{extension}"
                for extension in SUPPORTED_IMAGE_EXTENSIONS
                if (folder / f"{folder.name}{extension}").is_file()
            ),
            None,
        )
        if image_path is not None:
            yield image_path


def check_database(settings: dict[str, Any]) -> str:
    database_path = Path(settings["database_path"])
    if not database_path.is_dir():
        raise CheckError(f"database path does not exist: {database_path}")

    identities = []
    invalid_images = []
    try:
        from PIL import Image
    except ImportError as error:
        raise CheckError("Pillow is not installed") from error

    for image_path in identity_image_paths(database_path):
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as error:
            invalid_images.append(f"{image_path}: {error}")
            continue
        identities.append(image_path.parent.name)

    if invalid_images:
        raise CheckError("invalid database image(s): " + "; ".join(invalid_images))
    if not identities:
        raise CheckError(
            "no identities found; expected db/<name>/<name>.jpg|jpeg|png"
        )
    return f"{len(identities)} valid identities"


def check_stream(url: str, timeout: float) -> str:
    try:
        from lib.streamreader import StreamReader
    except ImportError as error:
        raise CheckError(f"cannot import stream reader: {error}") from error

    stream = StreamReader(url, reconnect_delay=1)
    stream.start()
    deadline = time.monotonic() + timeout
    frame = None
    try:
        while frame is None and time.monotonic() < deadline:
            frame = stream.read(timeout=min(1.0, max(0.1, deadline - time.monotonic())))
    finally:
        stream.stop()

    if frame is None:
        raise CheckError(f"no frame received from {display_url(url)} in {timeout:g}s")
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        raise CheckError("received an invalid frame")
    return f"{display_url(url)} returned {width}x{height}"


def request_motion(settings: dict[str, Any]) -> str:
    try:
        import requests
    except ImportError as error:
        raise CheckError("requests is not installed") from error

    url = settings["motion_url"]
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as error:
        raise CheckError(f"cannot read motion endpoint {display_url(url)}: {error}") from error

    if not isinstance(data, dict) or "val" not in data:
        raise CheckError("motion endpoint JSON must contain a 'val' field")
    if str(data["val"]).upper() not in {"ON", "OFF"}:
        raise CheckError("motion endpoint 'val' must be ON or OFF")
    return f"{display_url(url)} returned {str(data['val']).upper()}"


def check_models(settings: dict[str, Any]) -> str:
    try:
        from include.face_runtime import face_load
    except ImportError as error:
        raise CheckError(f"cannot import face runtime: {error}") from error

    detector = settings["detector_model"]
    recognizer = settings["recognition_model"]
    metric = settings["metric"]
    try:
        deepface = face_load(detector, recognizer)
        threshold = deepface.verification.find_threshold(recognizer, metric)
        sample_image = next(
            identity_image_paths(Path(settings["database_path"])),
            None,
        )
        if sample_image is None:
            raise CheckError("no database image is available for model inference")
        representations = deepface.represent(
            img_path=str(sample_image),
            model_name=recognizer,
            detector_backend=detector,
            enforce_detection=False,
            align=settings["alignment"],
        )
        embedding = representations[0].get("embedding") if representations else None
        if embedding is None or len(embedding) == 0:
            raise CheckError("model inference returned no embedding")
    except Exception as error:
        raise CheckError(
            f"cannot load detector={detector}, recognizer={recognizer}, "
            f"metric={metric}: {error}"
        ) from error
    return (
        f"detector={detector}, recognizer={recognizer}, "
        f"metric={metric}, threshold={threshold:.4f}, "
        f"sample={sample_image.name}"
    )


def check_internal_motion(
    settings: dict[str, Any],
    duration: float,
    stop_after_first_frame: bool = False,
) -> str:
    try:
        from lib.motionchecker import MotionChecker
        from lib.streamreader import StreamReader
    except ImportError as error:
        raise CheckError(f"cannot import motion runtime: {error}") from error

    url = settings["stream_url_lowres"] or settings["stream_url"]
    stream = StreamReader(url, reconnect_delay=1)
    checker = MotionChecker(
        motion_url=None,
        stream_reader=stream,
        use_internal=True,
        threshold=settings["motion_threshold"],
        min_area=settings["motion_min_area"],
        cooldown_seconds=settings["motion_cooldown"],
        verbose=settings["verbose"],
        resize=settings["stream_resize"],
        background_alpha=settings["background_alpha"],
    )
    stream.start()
    checker.start()
    observed_motion = False
    started_at = time.monotonic()
    deadline = time.monotonic() + duration
    try:
        while time.monotonic() < deadline:
            observed_motion = observed_motion or checker.result
            if stop_after_first_frame and checker.prev_frame is not None:
                break
            time.sleep(0.2)
    finally:
        checker.stop()
        stream.stop()

    if checker.prev_frame is None:
        raise CheckError(
            f"internal motion test received no frames from {display_url(url)}"
        )
    elapsed = time.monotonic() - started_at
    state = "motion observed" if observed_motion else "no motion observed"
    return f"processed {elapsed:.1f}s from {display_url(url)}; {state}"


def check_external_motion(settings: dict[str, Any], duration: float) -> str:
    deadline = time.monotonic() + duration
    samples = 0
    observed_states = set()
    while time.monotonic() < deadline or samples == 0:
        detail = request_motion(settings)
        samples += 1
        observed_states.add(detail.rsplit(" ", 1)[-1])
        if time.monotonic() < deadline:
            time.sleep(1)
    return f"{samples} valid samples; states={','.join(sorted(observed_states))}"


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=PROCESS_STOP_TIMEOUT)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=PROCESS_STOP_TIMEOUT)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        print(
            f"Process {process.pid} did not exit after SIGKILL",
            file=sys.stderr,
            flush=True,
        )


def run_command(command: list[str], timeout: float) -> str:
    print(f"$ {' '.join(command)}", flush=True)
    try:
        process = subprocess.Popen(command, start_new_session=True)
    except FileNotFoundError as error:
        raise CheckError(f"command not found: {command[0]}") from error
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        stop_process(process)
        raise CheckError(f"command timed out after {timeout:g}s") from error
    except KeyboardInterrupt:
        stop_process(process)
        raise
    if return_code != 0:
        raise CheckError(f"command exited with status {return_code}")
    return "command completed"


def check_nvidia_smi(timeout: float) -> str:
    return run_command(["nvidia-smi"], timeout)


def check_gpu_environment() -> str:
    return (
        f"Python={sys.version.split()[0]}, "
        f"NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES', '')!r}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}, "
        f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')!r}"
    )


def check_tensorflow_gpu() -> None:
    import tensorflow as tf

    print(f"TensorFlow: {tf.__version__}", flush=True)
    print(
        f"TensorFlow build: {json.dumps(tf.sysconfig.get_build_info(), default=str)}",
        flush=True,
    )
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow GPUs: {gpus}", flush=True)
    if not gpus:
        raise CheckError("TensorFlow cannot see a GPU")
    with tf.device("/GPU:0"):
        value = tf.reduce_sum(tf.random.normal([1024, 1024]))
    print(f"TensorFlow GPU operation: {float(value):.6f}", flush=True)


def check_torch_gpu() -> None:
    import torch

    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"PyTorch CUDA runtime: {torch.version.cuda}", flush=True)
    if not torch.cuda.is_available():
        raise CheckError("PyTorch cannot see a GPU")
    print(f"PyTorch GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"PyTorch capability: {torch.cuda.get_device_capability(0)}", flush=True)
    print(f"PyTorch architectures: {torch.cuda.get_arch_list()}", flush=True)
    with torch.inference_mode():
        left = torch.rand(1024, 1024, device="cuda")
        right = torch.rand(1024, 1024, device="cuda")
        result = (left @ right).sum().item()
        torch.cuda.synchronize()
    print(f"PyTorch GPU operation: {result:.6f}", flush=True)


def run_gpu_child(kind: str) -> int:
    traceback_after = int(os.environ.get("GPU_DIAGNOSTICS_TRACEBACK_AFTER", "120"))
    faulthandler.enable()
    faulthandler.dump_traceback_later(traceback_after, repeat=True)
    try:
        if kind == "tensorflow":
            check_tensorflow_gpu()
        elif kind == "torch":
            check_torch_gpu()
        else:
            raise CheckError(f"unknown GPU child check: {kind}")
    except Exception as error:
        print(f"FAILED: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        faulthandler.cancel_dump_traceback_later()
    return 0


def check_gpu_framework(kind: str, timeout: float) -> str:
    return run_command(
        [sys.executable, "-u", str(Path(__file__).resolve()), "--gpu-child", kind],
        timeout,
    )


def run_base_checks(
    runner: CheckRunner, settings: dict[str, Any], timeout: float
) -> None:
    runner.check("Database", check_database, settings)
    runner.check("Main RTSP stream", check_stream, settings["stream_url"], timeout)
    if settings["stream_url_lowres"]:
        runner.check(
            "Low-resolution RTSP stream",
            check_stream,
            settings["stream_url_lowres"],
            timeout,
        )
    if settings["use_internal_motion"]:
        runner.check(
            "Internal motion pipeline",
            check_internal_motion,
            settings,
            timeout,
            True,
        )
    else:
        runner.check("External motion endpoint", request_motion, settings)
    runner.check("Face detector and recognition model", check_models, settings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FacialRec runtime checks and startup preflight"
    )
    parser.add_argument(
        "--base",
        action="store_true",
        help="check config, database, RTSP, motion endpoint, and face models",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="check nvidia-smi plus TensorFlow and PyTorch GPU operations",
    )
    parser.add_argument(
        "--motion",
        action="store_true",
        help="exercise the configured motion detector for a limited duration",
    )
    parser.add_argument(
        "--all", action="store_true", help="run --base, --gpu, and --motion"
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="start main.py in the same process after all checks pass",
    )
    parser.add_argument(
        "--config-path",
        default=os.environ.get("FACIALREC_CONFIG", DEFAULT_CONFIG_PATH),
        help=f"config file path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("FACIALREC_TEST_TIMEOUT", DEFAULT_TIMEOUT)),
        help=f"RTSP/command timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--gpu-timeout",
        type=float,
        default=float(
            os.environ.get("GPU_DIAGNOSTICS_TIMEOUT", DEFAULT_GPU_TIMEOUT)
        ),
        help=f"per-framework GPU timeout in seconds (default: {DEFAULT_GPU_TIMEOUT})",
    )
    parser.add_argument(
        "--motion-duration",
        type=float,
        default=10.0,
        help="motion test duration in seconds (default: 10)",
    )
    parser.add_argument("--gpu-child", choices=("tensorflow", "torch"), help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu_child:
        return run_gpu_child(args.gpu_child)
    if args.timeout <= 0 or args.gpu_timeout <= 0 or args.motion_duration <= 0:
        print("Timeouts and durations must be greater than zero", file=sys.stderr)
        return 2

    selected = args.base or args.gpu or args.motion or args.all
    run_base = args.base or args.all or not selected
    run_gpu = args.gpu or args.all
    run_motion = args.motion or args.all

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    runner = CheckRunner()
    settings = None

    if run_base or run_motion or args.start:
        try:
            settings = load_settings(args.config_path)
        except Exception as error:
            runner.failed += 1
            print(f"[FAIL] Configuration: {error}", file=sys.stderr, flush=True)
        else:
            runner.passed += 1
            print(
                f"[PASS] Configuration: {args.config_path}",
                flush=True,
            )

    if run_gpu:
        runner.check("GPU environment", check_gpu_environment)
        runner.check("nvidia-smi", check_nvidia_smi, args.timeout)
        runner.check(
            "TensorFlow GPU",
            check_gpu_framework,
            "tensorflow",
            args.gpu_timeout,
        )
        runner.check(
            "PyTorch GPU",
            check_gpu_framework,
            "torch",
            args.gpu_timeout,
        )

    if run_base and settings is not None:
        if args.start and runner.failed:
            print(
                "\n[SKIP] Base checks because a startup prerequisite failed",
                flush=True,
            )
        else:
            run_base_checks(runner, settings, args.timeout)

    if run_motion and settings is not None:
        if args.start and runner.failed:
            print(
                "\n[SKIP] Motion test because a startup prerequisite failed",
                flush=True,
            )
        elif settings["use_internal_motion"]:
            runner.check(
                "Internal motion detection",
                check_internal_motion,
                settings,
                args.motion_duration,
            )
        else:
            runner.check(
                "External motion detection",
                check_external_motion,
                settings,
                args.motion_duration,
            )

    if not runner.summary():
        return 1

    if args.start:
        print("\nAll startup checks passed; starting main.py", flush=True)
        os.environ["FACIALREC_CONFIG"] = args.config_path
        sys.argv = ["main.py"]
        runpy.run_path(str(Path(__file__).with_name("main.py")), run_name="__main__")
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(143))
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nTests interrupted", file=sys.stderr, flush=True)
        raise SystemExit(130)
