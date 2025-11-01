#!/usr/bin/env python3
"""
Motion Detection Threshold Testing Tool

Interactive tool to test and calibrate motion detection thresholds.
Displays visual feedback and allows real-time adjustment of parameters.
"""

import cv2
import numpy as np
import configparser
import sys
import os
import logging
from lib.streamreader import StreamReader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class MotionTester:
    def __init__(self, stream_url, initial_threshold=25, initial_min_area=0.2):
        """
        Initialize motion tester

        Args:
            stream_url: RTSP stream URL
            initial_threshold: Pixel difference threshold (0-255)
            initial_min_area: Minimum area as percentage of frame (0.0-100.0)
        """
        self.stream_url = stream_url
        self.threshold = initial_threshold
        self.min_area = initial_min_area
        self.prev_frame = None
        self.paused = False

        print("\n" + "="*60)
        print("Motion Detection Threshold Tester")
        print("="*60)
        print("\nKeyboard Controls:")
        print("  +/-     : Increase/decrease min_area_percent by 0.05")
        print("  w/s     : Increase/decrease pixel threshold by 5")
        print("  W/S     : Increase/decrease pixel threshold by 1")
        print("  space   : Pause/resume")
        print("  r       : Reset to initial values")
        print("  q/ESC   : Quit")
        print("\nStarting values:")
        print(f"  Pixel Threshold: {self.threshold}")
        print(f"  Min Area: {self.min_area}%")
        print("="*60 + "\n")

    def process_frame(self, frame):
        """
        Process frame for motion detection (same algorithm as MotionChecker)

        Returns:
            (motion_detected, thresh_display, frame_with_contours, motion_percent)
        """
        if frame is None:
            return False, None, None, 0.0

        # Convert to grayscale and blur to reduce noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Initialize previous frame on first run
        if self.prev_frame is None:
            self.prev_frame = gray
            return False, None, frame, 0.0

        # Calculate absolute difference between frames
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]

        # Dilate to fill gaps
        thresh_dilated = cv2.dilate(thresh, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(thresh_dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate minimum area based on frame size
        frame_area = gray.shape[0] * gray.shape[1]
        min_area_pixels = int((self.min_area / 100.0) * frame_area)

        # Draw contours and check if any is large enough
        motion_detected = False
        total_motion_area = 0
        frame_with_contours = frame.copy()

        for contour in contours:
            area = cv2.contourArea(contour)
            total_motion_area += area

            if area >= min_area_pixels:
                motion_detected = True
                # Draw large contours in green
                cv2.drawContours(frame_with_contours, [contour], -1, (0, 255, 0), 2)
            else:
                # Draw small contours in red (below threshold)
                cv2.drawContours(frame_with_contours, [contour], -1, (0, 0, 255), 1)

        # Calculate motion as percentage of frame
        motion_percent = (total_motion_area / frame_area) * 100.0

        # Convert threshold image to BGR for display
        thresh_display = cv2.cvtColor(thresh_dilated, cv2.COLOR_GRAY2BGR)

        self.prev_frame = gray
        return motion_detected, thresh_display, frame_with_contours, motion_percent

    def add_overlay_text(self, frame, motion_detected, motion_percent):
        """Add overlay text with current settings and detection status"""
        if frame is None:
            return frame

        overlay = frame.copy()
        height, width = frame.shape[:2]

        # Semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (450, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Status color
        status_color = (0, 255, 0) if motion_detected else (0, 0, 255)
        status_text = "MOTION DETECTED" if motion_detected else "No Motion"

        # Add text
        y_offset = 35
        cv2.putText(frame, status_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        y_offset += 30
        cv2.putText(frame, f"Pixel Threshold: {self.threshold}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        y_offset += 25
        cv2.putText(frame, f"Min Area: {self.min_area:.2f}%", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        y_offset += 25
        cv2.putText(frame, f"Motion Area: {motion_percent:.3f}%", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        y_offset += 25
        if self.paused:
            cv2.putText(frame, "PAUSED", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame

    def run(self):
        """Main loop"""
        # Initialize stream reader
        stream_reader = StreamReader(self.stream_url)
        stream_reader.start()

        print("Connecting to stream...")

        # Wait for first frame
        import time
        timeout = 30
        start_time = time.time()
        frame = None
        while frame is None and time.time() - start_time < timeout:
            frame = stream_reader.read(timeout=1.0)

        if frame is None:
            print(f"ERROR: Could not connect to stream: {self.stream_url}")
            stream_reader.stop()
            return

        print("Stream connected! Starting motion detection test...\n")

        try:
            while True:
                if not self.paused:
                    frame = stream_reader.read(timeout=1.0)
                    if frame is None:
                        continue

                    # Process frame
                    motion_detected, thresh_display, frame_with_contours, motion_percent = self.process_frame(frame)

                    # Add overlay
                    display_frame = self.add_overlay_text(frame_with_contours, motion_detected, motion_percent)

                    # Show frames
                    cv2.imshow('Motion Test - Original with Contours', display_frame)
                    if thresh_display is not None:
                        cv2.imshow('Motion Test - Threshold View', thresh_display)
                else:
                    # Just refresh the display when paused
                    time.sleep(0.1)

                # Handle keyboard input
                key = cv2.waitKey(30) & 0xFF

                if key == ord('q') or key == 27:  # q or ESC
                    break
                elif key == ord('+') or key == ord('='):
                    self.min_area += 0.05
                    print(f"Min Area: {self.min_area:.2f}%")
                elif key == ord('-') or key == ord('_'):
                    self.min_area = max(0.0, self.min_area - 0.05)
                    print(f"Min Area: {self.min_area:.2f}%")
                elif key == ord('w'):
                    self.threshold = min(255, self.threshold + 5)
                    print(f"Pixel Threshold: {self.threshold}")
                elif key == ord('s'):
                    self.threshold = max(0, self.threshold - 5)
                    print(f"Pixel Threshold: {self.threshold}")
                elif key == ord('W'):
                    self.threshold = min(255, self.threshold + 1)
                    print(f"Pixel Threshold: {self.threshold}")
                elif key == ord('S'):
                    self.threshold = max(0, self.threshold - 1)
                    print(f"Pixel Threshold: {self.threshold}")
                elif key == ord(' '):
                    self.paused = not self.paused
                    print("PAUSED" if self.paused else "RESUMED")
                elif key == ord('r'):
                    self.threshold = initial_threshold
                    self.min_area = initial_min_area
                    print(f"Reset - Pixel Threshold: {self.threshold}, Min Area: {self.min_area:.2f}%")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")

        finally:
            # Cleanup
            stream_reader.stop()
            cv2.destroyAllWindows()

            print("\n" + "="*60)
            print("Final values:")
            print(f"  Pixel Threshold: {self.threshold}")
            print(f"  Min Area: {self.min_area:.2f}%")
            print("\nTo apply these values, update config.ini:")
            print(f"  [motion]")
            print(f"  threshold = {self.threshold}")
            print(f"  min_area_percent = {self.min_area:.2f}")
            print("="*60 + "\n")


if __name__ == "__main__":
    # Read config file
    config = configparser.ConfigParser()
    config_path = "./config/config.ini"

    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found at {config_path}", file=sys.stderr)
        print("Please create config.ini based on config-example.ini", file=sys.stderr)
        sys.exit(1)

    config.read(config_path)

    try:
        # Use low-res stream if available, otherwise main stream
        stream_url = config.get("basic", "stream_url_lowres", fallback="").strip()
        if not stream_url:
            stream_url = config["basic"]["stream_url"]

        # Get current motion settings
        initial_threshold = config.getint("motion", "threshold", fallback=25)
        initial_min_area = config.getfloat("motion", "min_area_percent", fallback=0.2)

    except KeyError as e:
        print(f"ERROR: Missing required config key: {e}", file=sys.stderr)
        sys.exit(1)

    # Run motion tester
    tester = MotionTester(stream_url, initial_threshold, initial_min_area)
    tester.run()
