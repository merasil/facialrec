import requests
import time
import threading
import logging
from typing import Optional

class MotionChecker:
    """
    Monitor motion detection endpoint in background thread

    Continuously polls a motion detection URL and signals when motion is detected.
    """

    def __init__(self, motion_url: str):
        """
        Initialize the motion checker

        Args:
            motion_url: URL endpoint to check for motion status
        """
        self.motion_url = motion_url
        self.result = False
        self.running = False
        self.session = requests.Session()
        self.event = threading.Event()
        self.thread = None

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
            self._check_motion()
            if self.result:
                logging.info("Motion detected")
                self.event.set()
            else:
                self.event.clear()
            time.sleep(1)

    def stop(self) -> None:
        """Stop the motion checker and clean up resources"""
        if not self.running:
            return

        self.running = False

        try:
            self.session.close()
        except Exception as e:
            logging.error(f"Error closing session: {e}")

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

        logging.info("MotionChecker stopped")

    def _check_motion(self) -> None:
        """Check motion status from the configured URL"""
        try:
            motion_response = self.session.get(self.motion_url, timeout=5)
            if motion_response.status_code not in range(200, 204):
                self.result = False
                logging.debug(f"Motion check returned status {motion_response.status_code}")
            else:
                motion_data = motion_response.json()
                if motion_data.get("val") == "ON":
                    self.result = True
                else:
                    self.result = False
        except requests.exceptions.RequestException as e:
            self.result = False
            logging.debug(f"Motion check failed: {e}")
        except Exception as e:
            self.result = False
            logging.error(f"Unexpected error checking motion: {e}")

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