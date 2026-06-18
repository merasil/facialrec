import threading
import cv2
import time
import logging
from queue import Queue, Empty
from typing import Optional
import numpy as np

class StreamReader:
    """
    Thread-safe RTSP stream reader with automatic reconnection

    Continuously reads frames from an RTSP stream in a background thread
    and provides them through a thread-safe queue.
    """

    def __init__(self, rtsp_url: str, reconnect_delay: int = 5, queue_size: int = 5):
        """
        Initialize the stream reader

        Args:
            rtsp_url: RTSP stream URL
            reconnect_delay: Seconds to wait before reconnection attempts
            queue_size: Maximum number of frames to buffer
        """
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self.capture = None
        self.frame_queue = Queue(maxsize=queue_size)
        self.running = threading.Event()
        self.thread = None

    def start(self) -> None:
        """Start the stream reader thread"""
        if not self.running.is_set():
            self.running.set()
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            logging.info("StreamReader started")
        else:
            logging.warning("StreamReader already running")

    def _connect(self) -> bool:
        """
        Connect or reconnect to the RTSP stream

        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.capture is not None:
                self.capture.release()
                self.capture = None

            self.capture = cv2.VideoCapture(self.rtsp_url)
            if not self.capture.isOpened():
                logging.error(f"Cannot open stream {self.rtsp_url}. Retrying in {self.reconnect_delay}s")
                time.sleep(self.reconnect_delay)
                return False
            return True
        except Exception as e:
            logging.error(f"Error connecting to stream: {e}")
            time.sleep(self.reconnect_delay)
            return False

    def _capture_loop(self) -> None:
        """Main capture loop running in background thread"""
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
                # If queue is full, discard oldest frame
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                self.frame_queue.put_nowait(frame)
            except Exception as e:
                logging.error(f"Frame queue error: {e}")

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Read the next frame from the queue

        Args:
            timeout: Maximum time to wait for a frame in seconds

        Returns:
            Frame as numpy array, or None if no frame available
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None

    def stop(self) -> None:
        """Stop the stream reader and clean up resources"""
        if not self.running.is_set():
            return

        self.running.clear()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

        if self.capture:
            try:
                self.capture.release()
            except Exception as e:
                logging.error(f"Error releasing capture: {e}")
            self.capture = None

        # Clear queue
        try:
            with self.frame_queue.mutex:
                self.frame_queue.queue.clear()
        except Exception as e:
            logging.error(f"Error clearing queue: {e}")

        logging.info("StreamReader stopped")

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.stop()
        except Exception:
            pass
