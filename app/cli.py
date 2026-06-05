import argparse
import logging
from time import time
from typing import Any

from app.config import CfgError, cfg_get_int, cfg_load


def cli_bool(cli_parser: Any, cli_name: str, cli_dest: str, cli_help: str) -> None:
    cli_parser.add_argument(
        cli_name,
        dest=cli_dest,
        action=argparse.BooleanOptionalAction,
        default=None,
        help=cli_help,
    )


def cli_config(cli_parser: Any, cli_sub: bool = False) -> None:
    cli_parser.add_argument(
        "--config",
        dest="cfg_path",
        default=argparse.SUPPRESS if cli_sub else "config/config.ini",
        help="INI configuration path",
    )


def cli_face(cli_parser: Any, cli_multi: bool = False) -> None:
    if cli_multi:
        cli_parser.add_argument(
            "--detectors",
            "--detector",
            dest="face_detectors",
            nargs="+",
            default=None,
            help="Face detector models",
        )
        cli_parser.add_argument(
            "--recognizers",
            "--recognizer",
            dest="face_recognizers",
            nargs="+",
            default=None,
            help="Face recognition models",
        )
    else:
        cli_parser.add_argument(
            "--detector",
            dest="face_detector",
            default=None,
            help="Face detector model",
        )
        cli_parser.add_argument(
            "--recognizer",
            dest="face_recognizer",
            default=None,
            help="Face recognition model",
        )
    cli_bool(cli_parser, "--align", "face_align", "Enable face alignment")
    cli_bool(
        cli_parser,
        "--enforce",
        "face_enforce",
        "Require a detected face",
    )


def cli_parser() -> argparse.ArgumentParser:
    cli_root = argparse.ArgumentParser(
        description="Live face recognition and model evaluation tools",
    )
    cli_config(cli_root)
    cli_root.set_defaults(cli_mode=None)

    cli_root.add_argument("--stream-url", dest="live_url", default=None)
    cli_root.add_argument("--stream-lowres", dest="live_low", default=None)
    cli_root.add_argument("--push-url", dest="live_push", default=None)
    cli_root.add_argument("--motion-url", dest="live_motion", default=None)
    cli_root.add_argument("--db", dest="face_db", default=None)
    cli_root.add_argument("--detector", dest="face_detector", default=None)
    cli_root.add_argument("--recognizer", dest="face_recognizer", default=None)
    cli_root.add_argument("--metric", dest="face_metric", default=None)
    cli_bool(cli_root, "--align", "face_align", "Enable face alignment")
    cli_bool(cli_root, "--enforce", "face_enforce", "Require a detected face")
    cli_bool(cli_root, "--stream-resize", "live_resize", "Resize motion frames")
    cli_bool(cli_root, "--internal-motion", "live_internal", "Use internal motion")
    cli_root.add_argument("--verbose", dest="live_verbose", type=int, default=None)
    cli_root.add_argument("--clearance", dest="live_clearance", type=int, default=None)
    cli_root.add_argument("--last-seen", dest="live_last", type=int, default=None)
    cli_root.add_argument("--pretty-sure", dest="live_pretty", type=float, default=None)
    cli_root.add_argument("--door-cooldown", dest="live_door", type=int, default=None)
    cli_root.add_argument(
        "--motion-threshold",
        dest="live_mot_threshold",
        type=int,
        default=None,
    )
    cli_root.add_argument(
        "--motion-area",
        dest="live_mot_area",
        type=float,
        default=None,
    )
    cli_root.add_argument(
        "--motion-cooldown",
        dest="live_mot_cool",
        type=int,
        default=None,
    )
    cli_root.add_argument(
        "--motion-alpha",
        dest="live_mot_alpha",
        type=float,
        default=None,
    )

    cli_subs = cli_root.add_subparsers(dest="cli_mode")

    cli_samples = cli_subs.add_parser(
        "create-test-samples",
        help="Record a short video and save every x-th frame as JPEG",
    )
    cli_config(cli_samples, cli_sub=True)
    cli_samples.add_argument("--stream-url", dest="sample_url", default=None)
    cli_samples.add_argument(
        "--duration",
        dest="sample_duration",
        type=float,
        default=None,
    )
    cli_samples.add_argument(
        "--frame-step",
        dest="sample_step",
        type=int,
        default=None,
    )
    cli_samples.add_argument(
        "--output-dir",
        dest="sample_output",
        default=None,
    )
    cli_samples.add_argument(
        "--fallback-fps",
        dest="sample_fps",
        type=float,
        default=None,
    )
    cli_bool(
        cli_samples,
        "--face-only",
        "sample_face",
        "Save only frames with a detected face",
    )
    cli_bool(
        cli_samples,
        "--keep-video",
        "sample_keep",
        "Keep the recorded MP4 file",
    )
    cli_samples.add_argument(
        "--detector",
        dest="face_detector",
        default=None,
        help="Face detector model",
    )
    cli_bool(cli_samples, "--align", "face_align", "Enable face alignment")
    cli_bool(
        cli_samples,
        "--enforce",
        "face_enforce",
        "Require a detected face",
    )

    cli_bench = cli_subs.add_parser(
        "benchmark",
        help="Benchmark detector, recognizer and metric combinations",
    )
    cli_config(cli_bench, cli_sub=True)
    cli_bench.add_argument("--input", dest="bench_input", required=True)
    cli_bench.add_argument("--name", dest="bench_name", required=True)
    cli_bench.add_argument("--db", dest="face_db", default=None)
    cli_face(cli_bench, cli_multi=True)
    cli_bench.add_argument(
        "--metric",
        dest="face_metrics",
        nargs="+",
        default=None,
        help="Distance metrics",
    )
    cli_bench.add_argument(
        "--output",
        dest="tab_output",
        default=None,
        help="Optional ASCII table file",
    )

    cli_vram = cli_subs.add_parser(
        "vram",
        help="Measure GPU memory for detector and recognizer combinations",
    )
    cli_config(cli_vram, cli_sub=True)
    cli_vram.add_argument("--input", dest="vram_input", required=True)
    cli_vram.add_argument("--db", dest="face_db", default=None)
    cli_face(cli_vram, cli_multi=True)
    cli_vram.add_argument("--metric", dest="face_metric", default=None)
    cli_vram.add_argument("--runs", dest="vram_runs", type=int, default=None)
    cli_vram.add_argument("--gpu", dest="vram_gpu", type=int, default=None)
    cli_vram.add_argument(
        "--output",
        dest="tab_output",
        default=None,
        help="Optional ASCII table file",
    )
    return cli_root


def cli_logging(cli_args: Any, cli_cfg: Any) -> None:
    cli_verbose = getattr(cli_args, "live_verbose", None)
    if cli_verbose is None:
        cli_verbose = cfg_get_int(cli_cfg, "basic", "verbose", 1)
    cli_level = logging.DEBUG if cli_verbose >= 4 else logging.INFO
    logging.basicConfig(
        level=cli_level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cli_run(cli_stamp: int | None = None) -> int:
    cli_args = cli_parser().parse_args()
    try:
        cli_cfg = cfg_load(cli_args.cfg_path)
        cli_logging(cli_args, cli_cfg)
        if cli_args.cli_mode is None:
            from app.live import live_run

            return live_run(cli_args, cli_cfg)
        if cli_args.cli_mode == "create-test-samples":
            from app.samples import sample_run

            return sample_run(cli_args, cli_cfg, cli_stamp or int(time()))
        if cli_args.cli_mode == "benchmark":
            from app.benchmark import bench_run

            return bench_run(cli_args, cli_cfg)
        if cli_args.cli_mode == "vram":
            from app.vram import vram_run

            return vram_run(cli_args, cli_cfg)
        raise ValueError(f"Unknown mode: {cli_args.cli_mode}")
    except (CfgError, OSError, ValueError, RuntimeError) as cli_err:
        logging.error("%s", cli_err)
        return 2
