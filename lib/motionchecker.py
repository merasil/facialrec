import logging
import threading
from time import sleep, time
from typing import Any, Optional

import requests


class MotChecker:
    """Monitor external motion state or detect motion from a stream."""

    def __init__(
        self,
        mot_url: Optional[str],
        mot_stream: Any = None,
        mot_internal: bool = False,
        mot_threshold: int = 25,
        mot_area: float = 0.2,
        mot_cooldown: int = 5,
        mot_verbose: int = 1,
        mot_resize: bool = False,
        mot_alpha: float = 0.05,
    ):
        self.mot_url = mot_url
        self.mot_stream = mot_stream
        self.mot_internal = mot_internal
        self.mot_threshold = mot_threshold
        self.mot_area = mot_area
        self.mot_cooldown = mot_cooldown
        self.mot_verbose = mot_verbose
        self.mot_resize = mot_resize
        self.mot_alpha = mot_alpha
        self.mot_result = False
        self.mot_run = False
        self.mot_session = requests.Session() if not mot_internal else None
        self.mot_event = threading.Event()
        self.mot_thread = None
        self.mot_prev = None
        self.mot_last = None
        self.mot_state = False

    def mot_start(self) -> None:
        if self.mot_run:
            logging.warning("Motion checker already running")
            return
        self.mot_run = True
        self.mot_thread = threading.Thread(target=self.mot_loop, daemon=True)
        self.mot_thread.start()
        logging.info("Motion checker started")

    def mot_loop(self) -> None:
        while self.mot_run:
            if self.mot_internal:
                mot_found = self.mot_check_internal()
            else:
                mot_found = self.mot_check_external()

            if mot_found:
                self.mot_last = time()
            if self.mot_last and time() - self.mot_last <= self.mot_cooldown:
                self.mot_result = True
                self.mot_event.set()
            else:
                self.mot_result = False
                self.mot_event.clear()

            if self.mot_result != self.mot_state:
                if self.mot_verbose >= 3:
                    mot_text = "detected" if self.mot_result else "stopped"
                    logging.info("Motion %s", mot_text)
                self.mot_state = self.mot_result
            sleep(0.1 if self.mot_internal else 1)

    def mot_stop(self) -> None:
        if not self.mot_run:
            return
        self.mot_run = False
        if self.mot_session:
            self.mot_session.close()
        if self.mot_thread and self.mot_thread.is_alive():
            self.mot_thread.join(timeout=2)
        logging.info("Motion checker stopped")

    def mot_check_external(self) -> bool:
        if not self.mot_url or self.mot_url == "None":
            logging.error("External motion URL is not configured")
            return False
        try:
            mot_resp = self.mot_session.get(self.mot_url, timeout=5)
            if mot_resp.status_code not in range(200, 204):
                logging.debug("Motion check returned %s", mot_resp.status_code)
                return False
            return mot_resp.json().get("val") == "ON"
        except requests.exceptions.RequestException as mot_err:
            logging.debug("Motion check failed: %s", mot_err)
            return False
        except Exception as mot_err:
            logging.error("Unexpected motion check error: %s", mot_err)
            return False

    def mot_check_internal(self) -> bool:
        if not self.mot_stream:
            logging.error("Internal motion detection requires a stream")
            return False
        try:
            import cv2

            mot_frame = self.mot_stream.str_read(str_timeout=0.5)
            if mot_frame is None:
                return False
            mot_gray = cv2.cvtColor(mot_frame, cv2.COLOR_BGR2GRAY)
            if self.mot_resize:
                mot_gray = cv2.resize(mot_gray, (320, 240), interpolation=cv2.INTER_AREA)
            mot_gray = cv2.GaussianBlur(mot_gray, (21, 21), 0)
            if self.mot_prev is None:
                self.mot_prev = mot_gray.astype("float")
                return False

            mot_ref = cv2.convertScaleAbs(self.mot_prev)
            mot_delta = cv2.absdiff(mot_ref, mot_gray)
            mot_mask = cv2.threshold(
                mot_delta,
                self.mot_threshold,
                255,
                cv2.THRESH_BINARY,
            )[1]
            mot_mask = cv2.dilate(mot_mask, None, iterations=2)
            mot_contours = cv2.findContours(
                mot_mask.copy(),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )[0]
            mot_pixels = mot_gray.shape[0] * mot_gray.shape[1]
            mot_min = int((self.mot_area / 100.0) * mot_pixels)
            mot_found = any(
                cv2.contourArea(mot_contour) >= mot_min
                for mot_contour in mot_contours
            )
            cv2.accumulateWeighted(mot_gray, self.mot_prev, self.mot_alpha)
            return mot_found
        except Exception as mot_err:
            logging.error("Internal motion detection failed: %s", mot_err)
            return False

    def mot_wait(self, mot_timeout: Optional[float] = None) -> bool:
        return self.mot_event.wait(mot_timeout)

    def mot_clear(self) -> None:
        self.mot_event.clear()

    def __del__(self):
        try:
            self.mot_stop()
        except Exception:
            pass
