import threading
import cv2
import sys
import time
from datetime import datetime

class StreamReader:
    def __init__(self, rtsp_url, reconnect_delay=5):
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self.capture = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.capture_frames)
            self.thread.start()
            print("---------------------------------------------", file=sys.stderr)
            print("{} INFO: Stream started...".format(datetime.now()), file=sys.stderr)
            print("---------------------------------------------", file=sys.stderr)
        else:
            print("---------------------------------------------", file=sys.stderr)
            print("{} INFO: Stream is already running...".format(datetime.now()), file=sys.stderr)
            print("---------------------------------------------", file=sys.stderr)

    def capture_frames(self):
        while self.running:
            if self.capture is None or not self.capture.isOpened():
                self.connect()
            success, frame = self.capture.read()
            if success:
                with self.lock:
                    self.frame = frame
            else:
                print("---------------------------------------------", file=sys.stderr)
                print("{} ERROR: Failed to read Image...".format(datetime.now()), file=sys.stderr)
                print("---------------------------------------------", file=sys.stderr)
                self.handle_failure()

    def connect(self):
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(self.rtsp_url)
        if not self.capture.isOpened():
            print("---------------------------------------------", file=sys.stderr)
            print("{} ERROR: Can not access Camera! Retrying in {} seconds...".format(datetime.now(), self.reconnect_delay), file=sys.stderr)
            print("---------------------------------------------", file=sys.stderr)
            time.sleep(self.reconnect_delay)

    def handle_failure(self):
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        time.sleep(self.reconnect_delay)

    def read(self):
        with self.lock:
            frame = self.frame
            self.frame = None  # Clear the frame after reading
        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        if self.capture is not None:
            self.capture.release()

    def __del__(self):
        self.stop()