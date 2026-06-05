#!/usr/bin/env python3

import logging
from time import sleep
from typing import Any

from app.config import cfg_get, cfg_get_float, cfg_get_int, cfg_load
from lib.streamreader import StrReader


class MotTester:
    def __init__(self, mot_url: str, mot_threshold: int = 25, mot_area: float = 0.2):
        self.mot_url = mot_url
        self.mot_threshold = mot_threshold
        self.mot_area = mot_area
        self.mot_start_threshold = mot_threshold
        self.mot_start_area = mot_area
        self.mot_prev = None
        self.mot_paused = False

    def mot_process(self, mot_frame: Any, mot_cv: Any) -> tuple[Any, Any, float]:
        mot_gray = mot_cv.cvtColor(mot_frame, mot_cv.COLOR_BGR2GRAY)
        mot_gray = mot_cv.GaussianBlur(mot_gray, (21, 21), 0)
        if self.mot_prev is None:
            self.mot_prev = mot_gray
            return False, mot_frame, 0.0

        mot_delta = mot_cv.absdiff(self.mot_prev, mot_gray)
        mot_mask = mot_cv.threshold(
            mot_delta,
            self.mot_threshold,
            255,
            mot_cv.THRESH_BINARY,
        )[1]
        mot_mask = mot_cv.dilate(mot_mask, None, iterations=2)
        mot_contours = mot_cv.findContours(
            mot_mask.copy(),
            mot_cv.RETR_EXTERNAL,
            mot_cv.CHAIN_APPROX_SIMPLE,
        )[0]
        mot_pixels = mot_gray.shape[0] * mot_gray.shape[1]
        mot_min = int((self.mot_area / 100.0) * mot_pixels)
        mot_total = 0.0
        mot_found = False
        mot_view = mot_frame.copy()

        for mot_contour in mot_contours:
            mot_size = mot_cv.contourArea(mot_contour)
            mot_total += mot_size
            mot_large = mot_size >= mot_min
            mot_found = mot_found or mot_large
            mot_color = (0, 255, 0) if mot_large else (0, 0, 255)
            mot_cv.drawContours(
                mot_view,
                [mot_contour],
                -1,
                mot_color,
                2 if mot_large else 1,
            )

        self.mot_prev = mot_gray
        return mot_found, mot_view, (mot_total / mot_pixels) * 100.0

    def mot_overlay(
        self,
        mot_frame: Any,
        mot_found: bool,
        mot_percent: float,
        mot_cv: Any,
    ) -> Any:
        mot_layer = mot_frame.copy()
        mot_cv.rectangle(mot_layer, (10, 10), (450, 160), (0, 0, 0), -1)
        mot_cv.addWeighted(mot_layer, 0.6, mot_frame, 0.4, 0, mot_frame)
        mot_color = (0, 255, 0) if mot_found else (0, 0, 255)
        mot_text = "MOTION DETECTED" if mot_found else "No Motion"
        mot_lines = [
            (mot_text, mot_color),
            (f"Pixel Threshold: {self.mot_threshold}", (255, 255, 255)),
            (f"Min Area: {self.mot_area:.2f}%", (255, 255, 255)),
            (f"Motion Area: {mot_percent:.3f}%", (255, 255, 255)),
        ]
        if self.mot_paused:
            mot_lines.append(("PAUSED", (0, 255, 255)))
        for mot_pos, (mot_line, mot_line_color) in enumerate(mot_lines):
            mot_cv.putText(
                mot_frame,
                mot_line,
                (20, 35 + mot_pos * 28),
                mot_cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                mot_line_color,
                2 if mot_pos == 0 else 1,
            )
        return mot_frame

    def mot_run(self) -> int:
        try:
            import cv2
        except ImportError:
            logging.error("OpenCV is not installed")
            return 2

        mot_stream = StrReader(self.mot_url)
        mot_stream.str_start()
        logging.info("Connecting to stream")
        mot_frame = mot_stream.str_read(str_timeout=30)
        if mot_frame is None:
            logging.error("Cannot connect to stream: %s", self.mot_url)
            mot_stream.str_stop()
            return 2

        print(
            "Controls: +/- area, w/s threshold by 5, W/S threshold by 1, "
            "space pause, r reset, q/ESC quit"
        )
        try:
            while True:
                if not self.mot_paused:
                    mot_frame = mot_stream.str_read(str_timeout=1)
                    if mot_frame is not None:
                        mot_found, mot_view, mot_percent = self.mot_process(
                            mot_frame,
                            cv2,
                        )
                        mot_view = self.mot_overlay(
                            mot_view,
                            mot_found,
                            mot_percent,
                            cv2,
                        )
                        cv2.imshow("Motion Test", mot_view)
                else:
                    sleep(0.1)

                mot_key = cv2.waitKey(30) & 0xFF
                if mot_key in (ord("q"), 27):
                    break
                if mot_key in (ord("+"), ord("=")):
                    self.mot_area += 0.05
                elif mot_key in (ord("-"), ord("_")):
                    self.mot_area = max(0.0, self.mot_area - 0.05)
                elif mot_key == ord("w"):
                    self.mot_threshold = min(255, self.mot_threshold + 5)
                elif mot_key == ord("s"):
                    self.mot_threshold = max(0, self.mot_threshold - 5)
                elif mot_key == ord("W"):
                    self.mot_threshold = min(255, self.mot_threshold + 1)
                elif mot_key == ord("S"):
                    self.mot_threshold = max(0, self.mot_threshold - 1)
                elif mot_key == ord(" "):
                    self.mot_paused = not self.mot_paused
                elif mot_key == ord("r"):
                    self.mot_threshold = self.mot_start_threshold
                    self.mot_area = self.mot_start_area
        finally:
            mot_stream.str_stop()
            cv2.destroyAllWindows()

        print(f"threshold = {self.mot_threshold}")
        print(f"min_area_percent = {self.mot_area:.2f}")
        return 0


def mot_main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    try:
        mot_cfg = cfg_load("config/config.ini")
        mot_url = cfg_get(mot_cfg, "basic", "stream_url_lowres", "").strip()
        if not mot_url:
            mot_url = cfg_get(mot_cfg, "basic", "stream_url")
        mot_threshold = cfg_get_int(mot_cfg, "motion", "threshold", 25)
        mot_area = cfg_get_float(mot_cfg, "motion", "min_area_percent", 0.2)
        return MotTester(mot_url, mot_threshold, mot_area).mot_run()
    except ValueError as mot_err:
        logging.error("%s", mot_err)
        return 2


if __name__ == "__main__":
    raise SystemExit(mot_main())
