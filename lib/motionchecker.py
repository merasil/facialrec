import requests
import time
import threading
import logging
from typing import Optional
import cv2
import numpy as np

MIN_MOTION_DIMENSION = 32


class MotionChecker:
    """
    Monitor motion detection endpoint in background thread or use internal motion detection

    Continuously polls a motion detection URL and signals when motion is detected,
    or performs internal motion detection using frame differencing.
    """

    def __init__(self, motion_url: Optional[str], stream_reader=None,
                 use_internal: bool = False, threshold: int = 25, min_area: float = 0.2,
                 cooldown_seconds: int = 5, verbose: int = 1, resize_factor: int = 1,
                 background_alpha: float = 0.05):
        """
        Initialize the motion checker

        Args:
            motion_url: URL endpoint to check for motion status (can be None if use_internal=True)
            stream_reader: StreamReader instance for internal motion detection
            use_internal: Use internal motion detection instead of external API
            threshold: Pixel difference threshold for motion detection (0-255)
            min_area: Minimum area as percentage of frame (0.0-100.0) to consider as motion
            cooldown_seconds: Seconds to keep motion active after last detection
            verbose: Logging verbosity level (0-4)
            resize_factor: Integer divisor of motion frame width and height (>= 1).
                Both output dimensions must be at least 32 pixels.
            background_alpha: Weight for running background reference (0.0-1.0).
                Lower = slower adaptation, better at catching slow motion.
        """
        self.motion_url = motion_url
        self.stream_reader = stream_reader
        self.use_internal = use_internal
        self.threshold = threshold
        self.min_area = min_area
        self.cooldown_seconds = cooldown_seconds
        self.verbose = verbose
        self.resize_factor = resize_factor
        # Scale spatial filters with the frame; Gaussian kernels must be odd.
        self._blur_size = max(1, 21 // self.resize_factor) | 1
        self._dilation_iterations = (2 + self.resize_factor // 2) // self.resize_factor
        self.background_alpha = background_alpha
        self.result = False
        self.running = False
        self.session = requests.Session() if not use_internal else None
        self.event = threading.Event()
        self.thread = None
        self.prev_frame = None
        self._frame_shape = None
        self.processing_error = None
        self.last_motion_time = None
        self.prev_result = False  # Track previous motion state for transitions

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

            # Detect state transitions
            if self.result != self.prev_result:
                if self.result:
                    if self.verbose >= 3:
                        logging.info("Motion detected - starting face recognition")
                else:
                    if self.verbose >= 3:
                        logging.info("Motion stopped - waiting for next motion")
                self.prev_result = self.result

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

    def _reset_motion_reference(self) -> None:
        """Discard the old scene and any motion held by its cooldown."""
        self.prev_frame = None
        self.last_motion_time = None
        self.result = False
        self.prev_result = False
        self.event.clear()

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

            height, width = frame.shape[:2]
            target_width = width // self.resize_factor
            target_height = height // self.resize_factor
            if min(target_width, target_height) < MIN_MOTION_DIMENSION:
                raise ValueError(
                    f"motion.resize_factor={self.resize_factor} produces "
                    f"{target_width}x{target_height} from {width}x{height}; "
                    f"motion frames must be at least {MIN_MOTION_DIMENSION} pixels "
                    "wide and high. Reduce resize_factor or use a larger stream."
                )

            # Convert to grayscale and blur to reduce noise
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.resize_factor > 1:
                gray = cv2.resize(
                    gray, (target_width, target_height), interpolation=cv2.INTER_AREA
                )
            gray = cv2.GaussianBlur(gray, (self._blur_size, self._blur_size), 0)

            if self._frame_shape != (height, width):
                self._reset_motion_reference()
                self._frame_shape = (height, width)
                logging.info(
                    "Motion detection: %sx%s -> %sx%s, resize_factor=%s",
                    width, height, target_width, target_height, self.resize_factor,
                )
            self.processing_error = None

            # Initialize running background reference on first run
            if self.prev_frame is None:
                self.prev_frame = gray.astype("float")
                return False

            # Compare against slowly adapting background (catches slow motion)
            ref = cv2.convertScaleAbs(self.prev_frame)
            frame_delta = cv2.absdiff(ref, gray)
            thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]

            # Dilate to fill gaps
            if self._dilation_iterations:
                thresh = cv2.dilate(thresh, None, iterations=self._dilation_iterations)

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

            # Update background reference with weighted average
            cv2.accumulateWeighted(gray, self.prev_frame, self.background_alpha)
            return motion_detected

        except Exception as e:
            self._reset_motion_reference()
            error = str(e)
            if error != self.processing_error:
                logging.error("Error in internal motion detection: %s", error)
            self.processing_error = error
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
