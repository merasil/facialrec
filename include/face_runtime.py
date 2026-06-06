import importlib
import logging
import os
from typing import Any


os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("DEEPFACE_LOG_LEVEL", str(logging.ERROR))


class FaceRuntimeError(RuntimeError):
    """Raised when the face runtime cannot be initialized."""


def face_torch_detector(face_detector: str) -> bool:
    face_name = face_detector.lower()
    return face_name.startswith("yolo") or face_name == "fastmtcnn"


def face_prepare_detector(face_detector: str) -> None:
    face_name = face_detector.lower()
    try:
        if face_name.startswith("yolo"):
            face_module = importlib.import_module("ultralytics")
            getattr(face_module, "YOLO")
        elif face_name == "fastmtcnn":
            face_module = importlib.import_module("facenet_pytorch")
            getattr(face_module, "MTCNN")
    except (AttributeError, ImportError) as face_err:
        raise FaceRuntimeError(
            f"Cannot import backend for detector {face_detector}: {face_err}"
        ) from face_err


def face_api() -> Any:
    try:
        face_module = importlib.import_module("deepface")
        return getattr(face_module, "DeepFace")
    except (AttributeError, ImportError) as face_err:
        raise FaceRuntimeError("DeepFace is not installed") from face_err


def face_tf(face_gpu: int = 0) -> Any:
    try:
        face_tf_module = importlib.import_module("tensorflow")
    except ImportError as face_err:
        raise FaceRuntimeError("TensorFlow is not installed") from face_err

    face_gpus = face_tf_module.config.list_physical_devices("GPU")
    if not face_gpus:
        return face_tf_module
    if face_gpu < 0 or face_gpu >= len(face_gpus):
        raise FaceRuntimeError(f"GPU index {face_gpu} is not available")

    face_device = face_gpus[face_gpu]
    try:
        face_tf_module.config.set_visible_devices(face_device, "GPU")
        face_tf_module.config.experimental.set_memory_growth(face_device, True)
    except RuntimeError as face_err:
        try:
            face_visible = face_tf_module.config.get_visible_devices("GPU")
            face_growth = face_tf_module.config.experimental.get_memory_growth(
                face_device
            )
        except Exception:
            face_visible = []
            face_growth = False
        if face_visible == [face_device] and face_growth:
            return face_tf_module
        raise FaceRuntimeError(f"Cannot configure GPU: {face_err}") from face_err
    return face_tf_module


def face_load(face_detector: str, face_recognizer: str) -> Any:
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
