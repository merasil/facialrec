import importlib
import logging
import os
from typing import Any


os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("DEEPFACE_LOG_LEVEL", str(logging.ERROR))


class FaceRuntimeError(RuntimeError):
    """Raised when the face runtime cannot be initialized."""


def face_gpu_required() -> bool:
    return os.environ.get("FACIALREC_REQUIRE_GPU", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def face_tf_memory_limit_mb() -> int | None:
    face_limit = os.environ.get("FACIALREC_TF_GPU_MEMORY_LIMIT_MB", "").strip()
    if not face_limit:
        return None
    try:
        face_limit_mb = int(face_limit)
    except ValueError as face_err:
        raise FaceRuntimeError(
            "FACIALREC_TF_GPU_MEMORY_LIMIT_MB must be an integer number of MB"
        ) from face_err
    if face_limit_mb < 0:
        raise FaceRuntimeError(
            "FACIALREC_TF_GPU_MEMORY_LIMIT_MB must be greater than or equal to 0"
        )
    if face_limit_mb == 0:
        return None
    return face_limit_mb


def face_torch_detector(face_detector: str) -> bool:
    face_name = face_detector.lower()
    return face_name.startswith("yolo")


def face_prepare_detector(face_detector: str) -> None:
    face_name = face_detector.lower()
    try:
        if face_name.startswith("yolo"):
            face_module = importlib.import_module("ultralytics")
            getattr(face_module, "YOLO")
    except (AttributeError, ImportError) as face_err:
        raise FaceRuntimeError(
            f"Cannot import backend for detector {face_detector}: {face_err}"
        ) from face_err


def face_api() -> Any:
    try:
        from deepface import DeepFace
    except ImportError as face_err:
        raise FaceRuntimeError(f"Cannot import DeepFace: {face_err}") from face_err
    return DeepFace


def face_tf(face_gpu: int = 0) -> Any:
    try:
        face_tf_module = importlib.import_module("tensorflow")
    except ImportError as face_err:
        raise FaceRuntimeError("TensorFlow is not installed") from face_err

    face_gpus = face_tf_module.config.list_physical_devices("GPU")
    if not face_gpus:
        if face_gpu_required():
            raise FaceRuntimeError(
                "TensorFlow cannot see an NVIDIA GPU. Run "
                "'docker compose -f docker-compose.yml "
                "-f docker-compose.gpu.yml run --rm facialrec "
                "python3 test.py --gpu' "
                "and regenerate the host CDI specification after a GPU change."
            )
        logging.info("No GPU available; using the CPU for face recognition")
        return face_tf_module
    if face_gpu < 0 or face_gpu >= len(face_gpus):
        raise FaceRuntimeError(f"GPU index {face_gpu} is not available")

    face_device = face_gpus[face_gpu]
    face_memory_limit_mb = face_tf_memory_limit_mb()
    try:
        face_tf_module.config.set_visible_devices(face_device, "GPU")
        if face_memory_limit_mb is None:
            face_tf_module.config.experimental.set_memory_growth(face_device, True)
        else:
            face_tf_module.config.set_logical_device_configuration(
                face_device,
                [
                    face_tf_module.config.LogicalDeviceConfiguration(
                        memory_limit=face_memory_limit_mb
                    )
                ],
            )
    except RuntimeError as face_err:
        try:
            face_visible = face_tf_module.config.get_visible_devices("GPU")
            if face_memory_limit_mb is None:
                face_configured = (
                    face_tf_module.config.experimental.get_memory_growth(face_device)
                )
            else:
                face_logical_config = (
                    face_tf_module.config.get_logical_device_configuration(face_device)
                )
                face_configured = (
                    face_logical_config is not None
                    and len(face_logical_config) == 1
                    and face_logical_config[0].memory_limit == face_memory_limit_mb
                )
        except Exception:
            face_visible = []
            face_configured = False
        if face_visible == [face_device] and face_configured:
            return face_tf_module
        raise FaceRuntimeError(f"Cannot configure GPU: {face_err}") from face_err
    if face_memory_limit_mb is None:
        logging.info("Using TensorFlow GPU %d: %s", face_gpu, face_device.name)
    else:
        logging.info(
            "Using TensorFlow GPU %d: %s with %d MB memory limit",
            face_gpu,
            face_device.name,
            face_memory_limit_mb,
        )
    return face_tf_module


def face_load(face_detector: str, face_recognizer: str) -> Any:
    if face_detector.lower() == "fastmtcnn":
        raise FaceRuntimeError(
            "FastMTCNN is not bundled because facenet-pytorch requires an old "
            "PyTorch build without RTX 50-series support. Use retinaface, mtcnn, "
            "or a YOLO detector."
        )

    if face_torch_detector(face_detector):
        face_prepare_detector(face_detector)
        face_deep = face_api()
        face_deep.build_model(model_name=face_detector, task="face_detector")
        face_tf()
    else:
        face_tf()
        face_deep = face_api()
        face_deep.build_model(model_name=face_detector, task="face_detector")

    face_deep.build_model(
        model_name=face_recognizer,
        task="facial_recognition",
    )
    return face_deep
