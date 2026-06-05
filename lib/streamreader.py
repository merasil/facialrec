import logging
import threading
from queue import Empty, Queue
from time import sleep
from typing import Any, Optional


class StrReader:
    """Continuously read a stream into a small thread-safe frame queue."""

    def __init__(self, str_url: str, str_delay: int = 5, str_size: int = 5):
        self.str_url = str_url
        self.str_delay = str_delay
        self.str_cap = None
        self.str_queue = Queue(maxsize=str_size)
        self.str_run = threading.Event()
        self.str_thread = None

    def str_start(self) -> None:
        if self.str_run.is_set():
            logging.warning("Stream reader already running")
            return
        self.str_run.set()
        self.str_thread = threading.Thread(target=self.str_loop, daemon=True)
        self.str_thread.start()
        logging.info("Stream reader started")

    def str_connect(self) -> bool:
        try:
            import cv2
        except ImportError as str_err:
            logging.error("OpenCV is not installed: %s", str_err)
            sleep(self.str_delay)
            return False

        try:
            if self.str_cap is not None:
                self.str_cap.release()
            self.str_cap = cv2.VideoCapture(self.str_url)
            if not self.str_cap.isOpened():
                logging.error(
                    "Cannot open stream %s. Retrying in %ss",
                    self.str_url,
                    self.str_delay,
                )
                sleep(self.str_delay)
                return False
            return True
        except Exception as str_err:
            logging.error("Error connecting to stream: %s", str_err)
            sleep(self.str_delay)
            return False

    def str_loop(self) -> None:
        while self.str_run.is_set():
            if self.str_cap is None or not self.str_cap.isOpened():
                self.str_connect()
                continue

            str_ok, str_frame = self.str_cap.read()
            if not str_ok:
                logging.error("Failed to read frame. Reconnecting")
                self.str_connect()
                continue

            if self.str_queue.full():
                try:
                    self.str_queue.get_nowait()
                except Empty:
                    pass
            try:
                self.str_queue.put_nowait(str_frame)
            except Exception as str_err:
                logging.error("Frame queue error: %s", str_err)

    def str_read(self, str_timeout: Optional[float] = None) -> Optional[Any]:
        try:
            return self.str_queue.get(timeout=str_timeout)
        except Empty:
            return None

    def str_stop(self) -> None:
        if not self.str_run.is_set():
            return
        self.str_run.clear()
        if self.str_thread and self.str_thread.is_alive():
            self.str_thread.join(timeout=2)
        if self.str_cap:
            try:
                self.str_cap.release()
            except Exception as str_err:
                logging.error("Error releasing capture: %s", str_err)
            self.str_cap = None
        with self.str_queue.mutex:
            self.str_queue.queue.clear()
        logging.info("Stream reader stopped")

    def __del__(self):
        try:
            self.str_stop()
        except Exception:
            pass
