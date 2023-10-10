from deepface import DeepFace
import cv2 as cv
from time import sleep
from datetime import datetime
import os
import requests
from random import randint
from libs.functions import *

def resetim(identity_matrix):
    for identity in identity_matrix:
        if identity_matrix[identity]["cnt"] != 0:
            diff = datetime.now() - identity_matrix[identity]["last"]
            if diff.seconds >= threshold_last_seen:
                identity_matrix[identity]["cnt"] = 0

def approveclearance(identity_matrix):
    for identity in identity_matrix:
        if identity_matrix[identity]["cnt"] >= threshold_clearance:
            print("Open Door for {}!".format(identity))
            identity_matrix[identity]["cnt"] = 0
            requests.get("http://kvsi6:8087/set/0_userdata.0.Status_Au%C3%9Fen.EG_FaceRec", {"value":"true"})

models = ["VGG-Face",
        "OpenFace",
        "Facenet",
        "Facenet512",
        "DeepID",
        "ArcFace",
        "SFace",
        "FbDeepFace"]

detectors = ["opencv",
        "ssd",
        "mtcnn",
        "retinaface",
        "mediapipe",
        "yolov8",
        "yunet"]
        
metrics = ["cosine", "euclidean", "euclidean_l2"]

debug = True

model = models[3]
detector = detectors[6]
metric = metrics[2]
db = "db"
threshold_recognizer = 0.98
threshold_clearance = 3
threshold_last_seen = 5
threshold_img_cnt = 1
threshold_last_motion = 30
last_motion = datetime.now()
identity_matrix = {}
identities = set()
for folder in os.scandir(db):
        if folder.is_dir():
                identity_matrix[folder.name] = {"cnt":0, "last":datetime.now()}
                identities.add(folder.name)
print(identities)

cam = cv.VideoCapture("rtsp://viewer:123456@172.23.0.104:554/h264Preview_01_main")
if not cam.isOpened():
    print("Cant open Camera")
    exit()

stream = CameraBufferCleanerThread(cam)
sleep(5)

while True:
    resetim(identity_matrix)
    if stream.last_frame is None:
        if debug:
            print("Couldnt receive Frame. Continuing with next...")
        continue
    try:
        img = stream.last_frame.copy()
        result = DeepFace.find(img_path=img, detector_backend=detector, db_path=db, distance_metric=metric, model_name=model, silent=True)
    except KeyboardInterrupt:
        print("Killing Process...")
        break
    except ValueError:
        # if debug:
        #     print("No Face found. Continuing...")
        continue
    for face in result:
        approved, name, mean, prob = approveface(face, model, metric, threshold_recognizer, threshold_img_cnt, identities, True)
        if approved:
            if debug:
                print("Found", name, "--->", mean)
            identity_matrix[name]["cnt"] += 1
            identity_matrix[name]["last"] = datetime.now()
            approveclearance(identity_matrix)
            continue
        elif name != "none":
            if debug:
                print("Found probably {} but not enough Images recognized to be confident!".format(name))
            continue
        else:
            if debug:
                print("Face not recognized!")
            diff = datetime.now() - last_motion
            if diff.seconds > threshold_last_motion:
                #requests.get("http://kvsi6:8087/set/0_userdata.0.Status_Au%C3%9Fen.EG_MotionDetect", {"value":"true"})
                last_motion = datetime.now()
cam.release()
print("Finished!")