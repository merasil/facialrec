from deepface import DeepFace
import cv2 as cv
import numpy as np
from time import sleep
from datetime import datetime
import os
import sys
import configparser
import tensorflow as tf
from include.functions import *
from lib.streamreader import StreamReader

gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    print("ERROR: Could not find any GPU!")
    
############## Reading Config-File #############
config = configparser.ConfigParser()
config.read("config.ini")

stream_url = config["basic"]["stream_url"]
push_url = config["basic"]["push_url"]
motion_url = config["basic"]["motion_url"]
debug = str2bool(config["basic"]["debug"])

############## Setting up Database #############
path_db = config["database"]["path"]
db = {}
for folder in os.scandir(path_db):
        if folder.is_dir():
            db[folder.name] = {"path":"{}/{}/{}.jpg".format(path_db,folder.name,folder.name), "last_seen":datetime.now(), "cnt":0}

############## Setting up Face Recognition Model #############
model = config["face_recognition"]["model"]
if model == "yunet":
    os.environ["yunet_score_threshold"] = "0.8"
detector = config["face_recognition"]["detector"]
metric = config["face_recognition"]["metric"]
alignment = str2bool(config["face_recognition"]["alignment"])
face_detect_enf = str2bool(config["face_recognition"]["face_detection_enforce"])

############## Setting up Thresholds #############
threshold_model = DeepFace.verification.find_threshold(model, metric)
threshold_clearance = int(config["thresholds"]["clearance"])
threshold_last_seen = int(config["thresholds"]["last_seen"])
threshold_pretty_sure = threshold_model - (threshold_model * float(config["thresholds"]["pretty_sure"]))

############## Settings for Camera (URL, Thread, etc.) #############
stream = StreamReader(stream_url)
stream.start()
sleep(5)
print("---------------------------------------------", file=sys.stderr)
print(db, file=sys.stderr)
print("---------------------------------------------", file=sys.stderr)
print("---------------------------------------------", file=sys.stderr)
print("{} INFO: Loading Model...".format(datetime.now()), file=sys.stderr)
print("---------------------------------------------", file=sys.stderr)
DeepFace.build_model(model)
print("---------------------------------------------", file=sys.stderr)
print("{} INFO: Finished loading Model...".format(datetime.now()), file=sys.stderr)
print("---------------------------------------------", file=sys.stderr)

##############  Starting the Application #############
while True:
    resetDB(db, threshold_last_seen)
    motion = checkMotion(motion_url)
    if not motion:
        if debug:
            print("---------------------------------------------", file=sys.stderr)
            print("{} INFO: No Motion detected. Continuing with next...".format(datetime.now()), file=sys.stderr)
            print("---------------------------------------------", file=sys.stderr)
        sleep(1.0)
        continue
    else:
        frame = stream.read()
        if frame is None:
            if debug:
                print("---------------------------------------------", file=sys.stderr)
                print("{} ERROR: Couldnt receive Frame. Continuing with next...".format(datetime.now()), file=sys.stderr)
                print("---------------------------------------------", file=sys.stderr)
            continue
        else:
            try:
                faces = DeepFace.find(img_path=frame, detector_backend=detector, align=alignment, enforce_detection=face_detect_enf, db_path=path_db, distance_metric=metric, model_name=model, silent=True)
            except KeyboardInterrupt:
                print("Killing Process...")
                stream.stop()
                exit()
            except ValueError as e:
                if debug:
                    print("---------------------------------------------")
                    print("{} ERROR: No Face found! Continuing...".format(datetime.now()), file=sys.stderr)
                    print(e, file=sys.stderr)
                    print("---------------------------------------------")
                continue
            except:
                if debug:
                    print("---------------------------------------------")
                    print("{} ERROR: Unknown Error! Exiting...".format(datetime.now()), file=sys.stderr)
                    print("---------------------------------------------")
                stream.stop()
                exit()
            for face in faces:
                if face.empty == True:
                    if debug:
                        print("---------------------------------------------")
                        print("{} INFO: No Face recognized! Continuing...".format(datetime.now()), file=sys.stderr)
                        print("---------------------------------------------")
                    continue
                else:
                    for identity in db:
                        if identity in face.iloc[0]["identity"]:
                            if debug:
                                print("---------------------------------------------")
                                print("{} INFO: Success! Found Face: {} with Value --> {}".format(datetime.now(), identity, face.iloc[0]["distance"]), file=sys.stderr)
                                print("---------------------------------------------")
                            db[identity]["cnt"] += 1
                            db[identity]["last_seen"] = datetime.now()
                            if face.iloc[0]["distance"] <= threshold_pretty_sure or db[identity]["cnt"] >= threshold_clearance:
                                openDoor(identity, push_url)
                            break