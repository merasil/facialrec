import requests
import time
import threading
from datetime import datetime
import sys

class MotionChecker:
    def __init__(self, motion_url):
        self.motion_url = motion_url
        self.result = False
        self.running = False
        self.session = requests.Session()

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.update, args=())
            self.thread.start()
            print("---------------------------------------------", file=sys.stderr)
            print("{} INFO: MotionChecker started...".format(datetime.now()), file=sys.stderr)
            print("---------------------------------------------", file=sys.stderr)
        else:
            print("---------------------------------------------", file=sys.stderr)
            print("{} INFO: MotionChecker is already running...".format(datetime.now()), file=sys.stderr)
            print("---------------------------------------------", file=sys.stderr)

    def update(self):
        while self.running:
            self.check_motion()
            time.sleep(1)

    def check_motion(self):
        try:
            motion_response = self.session.get(self.motion_url, timeout=5)
            if motion_response.status_code not in range(200, 204):
                self.result = False
            else:
                motion_data = motion_response.json()
                if motion_data.get("val") == "ON":
                    self.result = True
                else:
                    self.result = False
        except requests.exceptions.RequestException:
            self.result = False

    def stop(self):
        self.running = False
        self.session.close()
        self.thread.join()

    def __del__(self):
        self.stop()