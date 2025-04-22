import threading
import cv2
import time
import logging
from queue import Queue, Empty

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class StreamReader:
    def __init__(self, rtsp_url, reconnect_delay=5, queue_size=5):
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self.capture = None
        self.frame_queue = Queue(maxsize=queue_size)
        self.running = threading.Event()
        self.thread = None

    def start(self):
        if not self.running.is_set():
            self.running.set()
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            logging.info("StreamReader started...")
        else:
            logging.info("StreamReader already running...")

    def _connect(self):
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(self.rtsp_url)
        if not self.capture.isOpened():
            logging.error(f"Cannot open stream {self.rtsp_url}. Retrying in {self.reconnect_delay}s...")
            time.sleep(self.reconnect_delay)

    def _capture_loop(self):
        while self.running.is_set():
            if self.capture is None or not self.capture.isOpened():
                self._connect()
                continue

            success, frame = self.capture.read()
            if not success:
                logging.error("Failed to read frame. Reconnecting...")
                self._connect()
                continue

            try:
                # Wenn die Queue voll ist, das älteste Frame verwerfen
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except Exception as e:
                logging.error(f"Frame-Queue-Fehler: {e}")

    def read(self, timeout=None):
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None

    def stop(self):
        self.running.clear()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.capture:
            self.capture.release()
        # Queue leeren
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
        logging.info("StreamReader gestoppt.")

    def __del__(self):
        self.stop()
