import threading
import cv2
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class StreamReader:
    def __init__(self, rtsp_url, reconnect_delay=5):
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self.capture = None
        self.frame = None
        self.running = False

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.capture_frames)
            self.thread.start()
            logging.info("Stream started...")
        else:
            logging.info("Stream is already running...")

    def capture_frames(self):
        while self.running:
            if self.capture is None or not self.capture.isOpened():
                self.connect()
            success, frame = self.capture.read()
            if success:
                self.frame = frame
            else:
                logging.error("Failed to read Image...")
                self.handle_failure()

    def connect(self):
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(self.rtsp_url)
        if not self.capture.isOpened():
            logging.error("Can not access Camera! Retrying in {} seconds...".format(self.reconnect_delay))
            time.sleep(self.reconnect_delay)

    def handle_failure(self):
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        time.sleep(self.reconnect_delay)

    def read(self):
        frame = self.frame
        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        if self.capture is not None:
            self.capture.release()

    def __del__(self):
        self.stop()