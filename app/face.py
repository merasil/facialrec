import logging
import os
from pathlib import Path
from typing import Any


os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("DEEPFACE_LOG_LEVEL", str(logging.ERROR))


class FaceError(RuntimeError):
    """Raised when the face pipeline cannot be initialized."""


def face_api() -> Any:
    try:
        from deepface import DeepFace
    except ImportError as face_err:
        raise FaceError("DeepFace is not installed") from face_err
    return DeepFace


def face_tf(face_gpu: int = 0) -> Any:
    try:
        import tensorflow as tf
    except ImportError as face_err:
        raise FaceError("TensorFlow is not installed") from face_err

    face_gpus = tf.config.list_physical_devices("GPU")
    if not face_gpus:
        return tf
    if face_gpu < 0 or face_gpu >= len(face_gpus):
        raise FaceError(f"GPU index {face_gpu} is not available")
    try:
        tf.config.set_visible_devices(face_gpus[face_gpu], "GPU")
        tf.config.experimental.set_memory_growth(face_gpus[face_gpu], True)
    except RuntimeError as face_err:
        try:
            face_visible = tf.config.get_visible_devices("GPU")
            face_growth = tf.config.experimental.get_memory_growth(
                face_gpus[face_gpu]
            )
        except Exception:
            face_visible = []
            face_growth = False
        if face_visible == [face_gpus[face_gpu]] and face_growth:
            return tf
        raise FaceError(f"Cannot configure GPU: {face_err}") from face_err
    return tf


def face_load(face_detector: str, face_recognizer: str) -> None:
    face_deep = face_api()
    face_deep.build_model(model_name=face_detector, task="face_detector")
    face_deep.build_model(model_name=face_recognizer, task="facial_recognition")


def face_load_detector(face_detector: str) -> None:
    face_deep = face_api()
    face_deep.build_model(model_name=face_detector, task="face_detector")


def face_load_recognizer(face_recognizer: str) -> None:
    face_deep = face_api()
    face_deep.build_model(model_name=face_recognizer, task="facial_recognition")


def face_threshold(face_recognizer: str, face_metric: str) -> float:
    try:
        from deepface.modules.verification import find_threshold
    except ImportError as face_err:
        raise FaceError("DeepFace verification module is unavailable") from face_err
    return float(find_threshold(face_recognizer, face_metric))


def face_detect(
    face_image: Any,
    face_detector: str,
    face_align: bool,
    face_enforce: bool,
) -> bool:
    face_deep = face_api()
    try:
        face_items = face_deep.extract_faces(
            img_path=face_image,
            detector_backend=face_detector,
            enforce_detection=face_enforce,
            align=face_align,
        )
    except Exception as face_err:
        if face_missing(face_err):
            return False
        raise

    if face_enforce:
        return bool(face_items)
    for face_item in face_items:
        face_score = face_item.get("confidence", face_item.get("face_confidence", 0))
        if face_score is None or float(face_score) > 0:
            return True
    return False


def face_missing(face_err: Exception) -> bool:
    face_text = str(face_err).lower()
    face_terms = (
        "face could not be detected",
        "face could not be found",
        "no face",
        "cannot detect face",
        "local variable 'boxes_np'",
        "local variable 'lms_np'",
    )
    return any(face_term in face_text for face_term in face_terms)


def face_find(
    face_image: Any,
    face_db: str,
    face_detector: str,
    face_recognizer: str,
    face_metric: str,
    face_align: bool,
    face_enforce: bool,
    face_refresh: bool,
) -> list[Any]:
    face_deep = face_api()
    return face_deep.find(
        img_path=face_image,
        db_path=face_db,
        detector_backend=face_detector,
        model_name=face_recognizer,
        distance_metric=face_metric,
        align=face_align,
        enforce_detection=face_enforce,
        refresh_database=face_refresh,
        silent=True,
    )


def face_name(face_path: str) -> str:
    return Path(face_path).parent.name


def face_result(face_frames: list[Any], face_expected: str) -> tuple[bool, bool, bool]:
    face_detected = len(face_frames) > 0
    face_any = False
    face_correct = False

    for face_frame in face_frames:
        if face_frame is None or face_frame.empty:
            continue
        face_any = True
        for face_path in face_frame["identity"].tolist():
            if face_name(str(face_path)) == face_expected:
                face_correct = True
                break
        if face_correct:
            break
    return face_detected, face_any, face_correct
