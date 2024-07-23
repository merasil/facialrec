import cv2
import threading
from queue import Queue, Empty
import time
from datetime import datetime
import sys

class StreamReader:
    def __init__(self, rtsp_url, reconnect_delay=5):
        self.rtsp_url = rtsp_url
        self.capture = None
        self.frame_queue = Queue(maxsize=1)
        self.running = False
        self.reconnect_delay = reconnect_delay

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.update, args=())
            self.thread.start()
            print("---------------------------------------------", file=sys.stderr)
            print("{} INFO: Stream started...".format(datetime.now()), file=sys.stderr)
            print("{} INFO: Stream Queuesize: {}...".format(datetime.now(), self.frame_queue.maxsize), file=sys.stderr)
            print("---------------------------------------------", file=sys.stderr)
        else:
            print("---------------------------------------------", file=sys.stderr)
            print("{} INFO: Stream is already running...".format(datetime.now()), file=sys.stderr)
            print("---------------------------------------------", file=sys.stderr)

    def update(self):
        while self.running:
            if self.capture is None or not self.capture.isOpened():
                self.connect()
            ret, frame = self.capture.read()
            if not ret:
                print("---------------------------------------------", file=sys.stderr)
                print("{} ERROR: Failed to read Image...".format(datetime.now()), file=sys.stderr)
                print("---------------------------------------------", file=sys.stderr)
                self.handle_failure()
                continue
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

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
        try:
            frame = self.frame_queue.get_nowait()
        except Empty:
            frame = None
        return frame

    def stop(self):
        self.running = False
        self.thread.join()
        if self.capture is not None:
            self.capture.release()

    def __del__(self):
        self.stop()