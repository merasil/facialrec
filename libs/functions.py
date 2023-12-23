import cv2 as cv
import numpy as np
import threading
from deepface import DeepFace
from time import sleep
from datetime import datetime
import sys

error_print_timer = datetime.now()

class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, cameraurl):
        self.cameraurl = cameraurl
        self.camera = cv.VideoCapture(self.cameraurl)
        self.counter = 0
        sleep(5)
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(daemon=True)
        self.start()

    def run(self):
        while True:
            if not self.camera.isOpened() or self.counter == 5:
                print("---------------------------------------------", file=sys.stderr)
                print("{} ERROR: Can not access Camera!".format(datetime.now()), file=sys.stderr)
                print("---------------------------------------------", file=sys.stderr)
                self.camera.release()
                self.camera = cv.VideoCapture(self.cameraurl)
                self.counter = 0
                sleep(5)
            ret, im = self.camera.read()
            if ret:
                self.last_frame = im.copy()
            else:
                print("---------------------------------------------", file=sys.stderr)
                print("{} ERROR: No Image received!".format(datetime.now()), file=sys.stderr)
                print("---------------------------------------------", file=sys.stderr)
                self.counter += 1
                self.last_frame = None
            sleep(0.015)

class MotionDetectionThread(threading.Thread):
    def __init__(self, stream, bgm, bgm_learning_rate, threshold_motion_detection, debug=False):
        self.stream = stream
        self.bgm = bgm
        self.bgm_learning_rate = bgm_learning_rate
        self.threshold_motion_detection = threshold_motion_detection
        self.debug = debug
        self.last_motion_det = datetime.now()
        self.motion = False
        sleep(8)
        super(MotionDetectionThread, self).__init__(daemon=True)
        self.start()
        
    def run(self):
        while True:
            if (datetime.now() - self.last_motion_det).seconds > 30:
                self.motion = False
            img = self.stream.last_frame.copy()
            motionmask = self.bgm.apply(img, self.bgm_learning_rate)
            avg = np.average(cv.threshold(motionmask, 200, 255, cv.THRESH_BINARY)[1])
            if avg < self.threshold_motion_detection:
                sleep(0.2)
                continue
            if self.debug:
                print("---------------------------------------------", file=sys.stderr)
                print("{} SUCCESS: Motion detected: {}! Threshold: {}".format(datetime.now(), avg, self.threshold_motion_detection), file=sys.stderr)
                print("---------------------------------------------", file=sys.stderr)
            self.motion = True
            self.last_motion_det = datetime.now()

class RecordingThread(threading.Thread):
    def __init__(self, stream, duration):
        self.stream = stream
        self.duration = duration
        self.running = False
        self.start_time = datetime.now()
        self.filename = "{}.avi".format(self.start_time)
        self.writer = cv.VideoWriter(self.filename, cv.VideoWriter_fourcc('M','J','P','G'), 10, (2560,1440))
        super(MotionDetectionThread, self).__init__(daemon=True)
    
    def run(self):
        self.running = True
        while (datetime.now() - self.start_time).seconds < 30:
            img = self.stream.last_frame.copy()
            self.writer.write(img)
            sleep(0.1)
        self.running = False
        return 0

def approveface(face, model, metric, threshold_recognizer, threshold_img_cnt, identities, debug=False):
    identity_cnt = {}
    identity_mean = {}
    identity_prob = {}
    identity_ratio = {}
    faces_cnt = 0

    if len(face) == 0:
        return False, "none", 100.0, 100.0

    for index, row in face.iterrows():
        if index == 0 and row[f"{model}_{metric}"] > threshold_recognizer:
            return False, "none", 100.0, 100.0
        if row[f"{model}_{metric}"] < threshold_recognizer:
            for identity in identities:
                if identity in row["identity"]:
                    if identity in identity_cnt:
                        identity_cnt[identity] += 1
                        identity_mean[identity] += float(row[f"{model}_{metric}"])
                    else:
                        identity_cnt[identity] = 1
                        identity_mean[identity] = float(row[f"{model}_{metric}"])
                    faces_cnt += 1
                    break
            continue
        break

    for identity in identity_mean:
        identity_mean[identity] /= identity_cnt[identity]
        identity_prob[identity] = (identity_mean[identity] / identity_cnt[identity])
        identity_ratio[identity] = identity_prob[identity] / (identity_cnt[identity] / faces_cnt)
    face_name = min(identity_prob, key=identity_prob.get)
    face_mean = identity_mean[face_name]
    face_prob = identity_prob[face_name]
    face_ratio = identity_ratio[face_name]
    approved = False
    threshold_face_prob = (threshold_recognizer/threshold_img_cnt)
    
    if identity_cnt[face_name] >= threshold_img_cnt and face_prob <= threshold_face_prob and face_ratio <= threshold_face_prob:
         approved = True
    if debug:
        print("---------------------------------------------", file=sys.stderr)
        print("Threshold probability: {}".format(threshold_face_prob), file=sys.stderr)
        print("Recognized: {} with Mean: {} ({}) and Ratio: {}!\nApproved: {}".format(face_name, face_mean, face_prob, face_ratio, str(approved)), file=sys.stderr)
        print("---------------------------------------------", file=sys.stderr)
        print(face_name, file=sys.stderr)
        print()
    return approved, face_name, face_mean, face_prob

def checkface(face, database="db", model="Facenet512", metric="euclidean_l2", debug=False):
    global error_print_timer
    diff = datetime.now() - error_print_timer
    faces_recognized = []
    for index, row in face.iterrows():
        for identity in database:
            if identity in row["identity"] and row[f"{model}_{metric}"] <= database[identity]["threshold"]:
                faces_recognized.append({"identity":identity, "value":row[f"{model}_{metric}"]})
    if len(faces_recognized) == 1:
        if debug:
            print("---------------------------------------------", file=sys.stderr)
            print("{} SUCCESS: Face found: {} ---> {}! Threshold: {}".format(datetime.now(), faces_recognized[0]["identity"], faces_recognized[0]["value"], database[faces_recognized[0]["identity"]]["threshold"]), file=sys.stderr)
            print("---------------------------------------------", file=sys.stderr)
        return faces_recognized[0]["identity"], faces_recognized[0]["value"]
    else:
        if debug:
            if diff.seconds >= 10:
                if len(faces_recognized) == 0:
                    print("---------------------------------------------", file=sys.stderr)
                    print("{} ERROR: No Face recognized!".format(datetime.now()), file=sys.stderr)
                    print("---------------------------------------------", file=sys.stderr)
                else:
                    print("---------------------------------------------", file=sys.stderr)
                    print("{} ERROR: Found more Faces with Thresholds passed!".format(datetime.now()), file=sys.stderr)
                    print("---------------------------------------------", file=sys.stderr)
                error_print_timer = datetime.now()
        return False, 100