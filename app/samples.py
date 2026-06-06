import logging
from pathlib import Path
from typing import Any

from app.config import cfg_get, cfg_get_bool, cfg_get_float, cfg_get_int, cfg_pick
from app.face import face_detect, face_load_detector, face_tf, face_torch_detector
from app.media import med_extract, med_record


def sample_folder(sample_root: str, sample_stamp: int) -> Path:
    sample_base = Path(sample_root)
    sample_path = sample_base / str(sample_stamp)
    sample_pos = 0
    while sample_path.exists():
        sample_pos += 1
        sample_path = sample_base / f"{sample_stamp}-{sample_pos:02d}"
    sample_path.mkdir(parents=True)
    return sample_path


def sample_run(sample_args: Any, sample_cfg: Any, sample_stamp: int) -> int:
    sample_url = cfg_pick(
        sample_args.sample_url,
        cfg_get(sample_cfg, "basic", "stream_url"),
    )
    sample_duration = cfg_pick(
        sample_args.sample_duration,
        cfg_get_float(sample_cfg, "samples", "duration", 10.0),
    )
    sample_step = cfg_pick(
        sample_args.sample_step,
        cfg_get_int(sample_cfg, "samples", "frame_step", 5),
    )
    sample_face = cfg_pick(
        sample_args.sample_face,
        cfg_get_bool(sample_cfg, "samples", "face_only", False),
    )
    sample_keep = cfg_pick(
        sample_args.sample_keep,
        cfg_get_bool(sample_cfg, "samples", "keep_video", False),
    )
    sample_root = cfg_pick(
        sample_args.sample_output,
        cfg_get(sample_cfg, "samples", "output_dir", "output"),
    )
    sample_fps = cfg_pick(
        sample_args.sample_fps,
        cfg_get_float(sample_cfg, "samples", "fallback_fps", 25.0),
    )
    sample_detector = cfg_pick(
        sample_args.face_detector,
        cfg_get(sample_cfg, "face_recognition", "detector_model"),
    )
    sample_align = cfg_pick(
        sample_args.face_align,
        cfg_get_bool(sample_cfg, "face_recognition", "alignment", False),
    )
    sample_enforce = cfg_pick(
        sample_args.face_enforce,
        cfg_get_bool(sample_cfg, "face_recognition", "enforce", True),
    )

    if sample_duration <= 0:
        raise ValueError("--duration must be greater than zero")
    if sample_step <= 0:
        raise ValueError("--frame-step must be greater than zero")
    if sample_fps <= 0:
        raise ValueError("--fallback-fps must be greater than zero")

    sample_path = sample_folder(sample_root, sample_stamp)
    sample_video = sample_path / "capture.mp4"
    logging.info("Recording %.2f seconds from %s", sample_duration, sample_url)
    sample_count = med_record(sample_url, sample_video, sample_duration, sample_fps)

    sample_filter = None
    if sample_face:
        if face_torch_detector(sample_detector):
            face_load_detector(sample_detector)
            face_tf()
        else:
            face_tf()
            face_load_detector(sample_detector)

        def sample_check(sample_frame: Any) -> bool:
            return face_detect(
                sample_frame,
                sample_detector,
                sample_align,
                sample_enforce,
            )

        sample_filter = sample_check

    sample_seen, sample_saved = med_extract(
        sample_video,
        sample_path,
        sample_step,
        sample_filter,
    )
    if not sample_keep:
        sample_video.unlink(missing_ok=True)

    logging.info(
        "Recorded %d frames, decoded %d frames and saved %d JPEG files in %s",
        sample_count,
        sample_seen,
        sample_saved,
        sample_path,
    )
    return 0
