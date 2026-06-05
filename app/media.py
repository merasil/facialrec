from pathlib import Path
from time import monotonic
from typing import Any, Iterator, Optional


MED_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class MedError(RuntimeError):
    """Raised when media input cannot be opened or decoded."""


def med_cv2() -> Any:
    try:
        import cv2
    except ImportError as med_err:
        raise MedError("OpenCV is not installed") from med_err
    return cv2


def med_images(med_path: str) -> list[Path]:
    med_input = Path(med_path)
    if med_input.is_dir():
        return sorted(
            med_file
            for med_file in med_input.iterdir()
            if med_file.is_file() and med_file.suffix.lower() in MED_IMAGES
        )
    if med_input.is_file() and med_input.suffix.lower() in MED_IMAGES:
        return [med_input]
    return []


def med_first(med_path: str) -> Any:
    med_input = Path(med_path)
    med_files = med_images(med_path)
    if med_files:
        return str(med_files[0])
    if med_input.is_dir():
        raise MedError(f"No supported images found in directory: {med_path}")

    med_cv = med_cv2()
    med_cap = med_cv.VideoCapture(med_path)
    if not med_cap.isOpened():
        raise MedError(f"Cannot open media input: {med_path}")
    try:
        med_ok, med_frame = med_cap.read()
        if not med_ok or med_frame is None:
            raise MedError(f"Cannot read a frame from: {med_path}")
        return med_frame
    finally:
        med_cap.release()


def med_iter(med_path: str) -> Iterator[tuple[int, Any]]:
    med_input = Path(med_path)
    med_files = med_images(med_path)
    if med_files:
        for med_pos, med_file in enumerate(med_files, start=1):
            yield med_pos, str(med_file)
        return

    if med_input.is_dir():
        raise MedError(f"No supported images found in directory: {med_path}")
    if not med_input.is_file():
        raise MedError(f"Input is not an image folder or media file: {med_path}")

    med_cv = med_cv2()
    med_cap = med_cv.VideoCapture(med_path)
    if not med_cap.isOpened():
        raise MedError(f"Cannot open video: {med_path}")
    try:
        med_pos = 0
        while True:
            med_ok, med_frame = med_cap.read()
            if not med_ok:
                break
            med_pos += 1
            yield med_pos, med_frame
        if med_pos == 0:
            raise MedError(f"No frames decoded from video: {med_path}")
    finally:
        med_cap.release()


def med_record(
    med_url: str,
    med_path: Path,
    med_duration: float,
    med_fps: float,
) -> int:
    med_cv = med_cv2()
    med_cap = med_cv.VideoCapture(med_url)
    if not med_cap.isOpened():
        raise MedError(f"Cannot open stream: {med_url}")

    try:
        med_ok, med_frame = med_cap.read()
        if not med_ok or med_frame is None:
            raise MedError(f"Cannot read first frame from stream: {med_url}")

        med_height, med_width = med_frame.shape[:2]
        med_src_fps = float(med_cap.get(med_cv.CAP_PROP_FPS))
        med_rate = med_src_fps if med_src_fps > 0 else med_fps
        med_codec = med_cv.VideoWriter_fourcc(*"mp4v")
        med_writer = med_cv.VideoWriter(
            str(med_path),
            med_codec,
            med_rate,
            (med_width, med_height),
        )
        if not med_writer.isOpened():
            raise MedError(f"Cannot create video file: {med_path}")

        try:
            med_count = 0
            med_start = monotonic()
            while monotonic() - med_start < med_duration:
                if med_frame is not None:
                    med_writer.write(med_frame)
                    med_count += 1
                med_ok, med_frame = med_cap.read()
                if not med_ok:
                    break
        finally:
            med_writer.release()
    finally:
        med_cap.release()

    if med_count == 0:
        raise MedError("No frames were recorded")
    return med_count


def med_extract(
    med_video: Path,
    med_folder: Path,
    med_step: int,
    med_filter: Optional[Any] = None,
) -> tuple[int, int]:
    med_cv = med_cv2()
    med_cap = med_cv.VideoCapture(str(med_video))
    if not med_cap.isOpened():
        raise MedError(f"Cannot open recorded video: {med_video}")

    med_seen = 0
    med_saved = 0
    try:
        while True:
            med_ok, med_frame = med_cap.read()
            if not med_ok:
                break
            med_seen += 1
            if (med_seen - 1) % med_step != 0:
                continue
            if med_filter is not None and not med_filter(med_frame):
                continue

            med_saved += 1
            med_path = med_folder / f"{med_saved:06d}.jpg"
            if not med_cv.imwrite(str(med_path), med_frame):
                raise MedError(f"Cannot write JPEG: {med_path}")
    finally:
        med_cap.release()
    return med_seen, med_saved
