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
            if ret and im is not None:
                self.last_frame = im.copy()
            else:
                print("---------------------------------------------", file=sys.stderr)
                print("{} ERROR: No Image received!".format(datetime.now()), file=sys.stderr)
                print("---------------------------------------------", file=sys.stderr)
                self.counter += 1
            sleep(0.015)

def approveface(face, model, metric, threshold_recognizer, threshold_img_cnt, identities, debug=False):
    identity_cnt = {}
    identity_mean = {}
    identity_prob = {}
    identity_ratio = {}
    faces_cnt = 0

    if len(face) == 0:
        return False, "none", 100.0, 100.0

    for index, row in face.iterrows():
        if index == 0 and row["distance"] > threshold_recognizer:
            return False, "none", 100.0, 100.0
        if row["distance"] < threshold_recognizer:
            for identity in identities:
                if identity in row["identity"]:
                    if identity in identity_cnt:
                        identity_cnt[identity] += 1
                        identity_mean[identity] += float(row["distance"])
                    else:
                        identity_cnt[identity] = 1
                        identity_mean[identity] = float(row["distance"])
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

def checkface(threshold, face, database="db", debug=False):
    global error_print_timer
    diff = datetime.now() - error_print_timer
    faces_recognized = []
    for index, row in face.iterrows():
        if debug:
                print("---------------------------------------------", file=sys.stderr)
                print("{}".format(row), file=sys.stderr)
                print("---------------------------------------------", file=sys.stderr)
        for identity in database:
            if identity in row["identity"]:
                faces_recognized.append({"identity":identity, "value":row["distance"]})
    if len(faces_recognized) == 1:
        if debug:
            print("---------------------------------------------", file=sys.stderr)
            print("{} SUCCESS: Face found: {} ---> {}! Threshold: {}".format(datetime.now(), faces_recognized[0]["identity"], faces_recognized[0]["value"], threshold), file=sys.stderr)
            print("---------------------------------------------", file=sys.stderr)
        return faces_recognized[0]["identity"], faces_recognized[0]["value"]
    else:
        if debug:
            if diff.seconds >= 10:
                if len(faces_recognized) == 0:
                    print("---------------------------------------------", file=sys.stderr)
                    print("{} INFO: No Face recognized!".format(datetime.now()), file=sys.stderr)
                    print("---------------------------------------------", file=sys.stderr)
                else:
                    print("---------------------------------------------", file=sys.stderr)
                    print("{} ERROR: Found more Faces with Thresholds passed!".format(datetime.now()), file=sys.stderr)
                    print("---------------------------------------------", file=sys.stderr)
                error_print_timer = datetime.now()
        return False, 100