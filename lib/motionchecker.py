import requests
import time
import threading
import logging
from typing import Optional
import cv2
import numpy as np

class MotionChecker:
    """
    Monitor motion detection endpoint in background thread or use internal motion detection

    Continuously polls a motion detection URL and signals when motion is detected,
    or performs internal motion detection using frame differencing.
    """

    def __init__(self, motion_url: Optional[str], stream_reader=None,
                 use_internal: bool = False, threshold: int = 25, min_area: float = 0.2,
                 cooldown_seconds: int = 5):
        """
        Initialize the motion checker

        Args:
            motion_url: URL endpoint to check for motion status (can be None if use_internal=True)
            stream_reader: StreamReader instance for internal motion detection
            use_internal: Use internal motion detection instead of external API
            threshold: Pixel difference threshold for motion detection (0-255)
            min_area: Minimum area as percentage of frame (0.0-100.0) to consider as motion
            cooldown_seconds: Seconds to keep motion active after last detection
        """
        self.motion_url = motion_url
        self.stream_reader = stream_reader
        self.use_internal = use_internal
        self.threshold = threshold
        self.min_area = min_area
        self.cooldown_seconds = cooldown_seconds
        self.result = False
        self.running = False
        self.session = requests.Session() if not use_internal else None
        self.event = threading.Event()
        self.thread = None
        self.prev_frame = None
        self.last_motion_time = None

    def start(self) -> None:
        """Start the motion checker thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            logging.info("MotionChecker started")
        else:
            logging.warning("MotionChecker already running")

    def _update_loop(self) -> None:
        """Main update loop running in background thread"""
        while self.running:
            if self.use_internal:
                motion_detected = self._check_motion_internal()
            else:
                motion_detected = self._check_motion()

            # Update last motion time if motion detected
            if motion_detected:
                self.last_motion_time = time.time()
                logging.debug("Motion detected")

            # Keep motion active if within cooldown period
            if self.last_motion_time:
                time_since_motion = time.time() - self.last_motion_time
                if time_since_motion <= self.cooldown_seconds:
                    self.result = True
                    self.event.set()
                else:
                    self.result = False
                    self.event.clear()
            else:
                self.result = False
                self.event.clear()

            time.sleep(0.1 if self.use_internal else 1)

    def stop(self) -> None:
        """Stop the motion checker and clean up resources"""
        if not self.running:
            return

        self.running = False

        if self.session:
            try:
                self.session.close()
            except Exception as e:
                logging.error(f"Error closing session: {e}")

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

        logging.info("MotionChecker stopped")

    def _check_motion(self) -> bool:
        """
        Check motion status from the configured URL

        Returns:
            True if motion detected, False otherwise
        """
        if not self.motion_url or self.motion_url == "None":
            logging.error("External motion detection enabled but motion_url not configured")
            return False

        try:
            motion_response = self.session.get(self.motion_url, timeout=5)
            if motion_response.status_code not in range(200, 204):
                logging.debug(f"Motion check returned status {motion_response.status_code}")
                return False
            else:
                motion_data = motion_response.json()
                return motion_data.get("val") == "ON"
        except requests.exceptions.RequestException as e:
            logging.debug(f"Motion check failed: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error checking motion: {e}")
            return False

    def _check_motion_internal(self) -> bool:
        """
        Check motion using internal frame differencing

        Returns:
            True if motion detected, False otherwise
        """
        if not self.stream_reader:
            logging.error("Internal motion detection requires stream_reader")
            return False

        try:
            # Get current frame with short timeout
            frame = self.stream_reader.read(timeout=0.5)
            if frame is None:
                return False

            # Convert to grayscale and blur to reduce noise
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # Initialize previous frame on first run
            if self.prev_frame is None:
                self.prev_frame = gray
                return False

            # Calculate absolute difference between frames
            frame_delta = cv2.absdiff(self.prev_frame, gray)
            thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]

            # Dilate to fill gaps
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Calculate minimum area based on frame size
            # min_area is stored as percentage of frame (0.0-100.0)
            frame_area = gray.shape[0] * gray.shape[1]
            min_area_pixels = int((self.min_area / 100.0) * frame_area)

            # Check if any contour is large enough
            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) >= min_area_pixels:
                    motion_detected = True
                    break

            self.prev_frame = gray
            return motion_detected

        except Exception as e:
            logging.error(f"Error in internal motion detection: {e}")
            return False

    def wait_motion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for motion to be detected

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if motion detected, False if timeout
        """
        return self.event.wait(timeout)

    def clear_event(self) -> None:
        """Clear the motion detection event"""
        self.event.clear()

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.stop()
        except Exception:
            pass