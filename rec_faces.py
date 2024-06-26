from deepface import DeepFace
import cv2 as cv
import numpy as np
from time import sleep
from datetime import datetime
import os
import requests
import configparser
import tensorflow as tf
from libs.functions import *

gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    print("ERROR: Could not find any GPU!")

def str2bool(v):
    return str(v).lower() in ("yes", "y", "true", "t", "1")

def resetim(database):
    for identity in database:
        if database[identity]["cnt"] != 0:
            diff = datetime.now() - database[identity]["last_seen"]
            if diff.seconds >= threshold_last_seen:
                database[identity]["cnt"] = 0

def openDoor(identity, push_url):
    print("Open Door for {}!".format(identity))
    requests.get(push_url, {"value":"true"})

def approveclearance(database, push_url):
    for identity in database:
        if database[identity]["cnt"] >= threshold_clearance:
            openDoor(identity, push_url)
            database[identity]["cnt"] = 0

############## Reading Config-File #############
config = configparser.ConfigParser()
config.read("config.ini")

stream_url = config["basic"]["stream_url"]
push_url = config["basic"]["push_url"]
motion_url = config["basic"]["motion_url"]
debug = str2bool(config["basic"]["debug"])
#debug = False
#if config["basic"]["debug"] == "True":
#    debug = True

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
threshold_pretty_sure = threshold_model - (threshold_model * 0.07)

############## Settings for Camera (URL, Thread, etc.) #############
stream = CameraBufferCleanerThread(stream_url)
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
    # Reseting DB if Face isnt recognized for a period of time...
    resetim(db)
    if stream.last_frame is None:
        if debug:
            print("---------------------------------------------", file=sys.stderr)
            print("{} ERROR: Couldnt receive Frame. Continuing with next...".format(datetime.now()), file=sys.stderr)
            print("---------------------------------------------", file=sys.stderr)
        continue
    motion = requests.get(motion_url)
    if motion.status_code not in range(200, 204):
        print("---------------------------------------------", file=sys.stderr)
        print("{} WARN: Couldnt receive Motion Status. Continuing with Face Detection...".format(datetime.now()), file=sys.stderr)
        print("---------------------------------------------", file=sys.stderr)
    else:
        motion = motion.json()
        if motion["val"] == "OFF":
            if debug:
                print("---------------------------------------------", file=sys.stderr)
                print("{} INFO: No Motion detected. Continuing with next...".format(datetime.now()), file=sys.stderr)
                print("---------------------------------------------", file=sys.stderr)
            sleep(1.0)
            continue
    img = stream.last_frame
    try:
        faces = DeepFace.find(img_path=img, detector_backend=detector, align=alignment, enforce_detection=face_detect_enf, db_path=path_db, distance_metric=metric, model_name=model, silent=not debug)
    except KeyboardInterrupt:
        print("Killing Process...")
        break
    except ValueError as e:
        if debug:
            print("---------------------------------------------")
            print("ERROR: No Face found! Continuing...")
            print(e)
            print("---------------------------------------------")
        continue
    
    # If we got Faces we can check if we know them...
    for face in faces:
        if debug:
                print("---------------------------------------------", file=sys.stderr)
                print("{} INFO: Face detected!".format(datetime.now()), file=sys.stderr)
                print("---------------------------------------------", file=sys.stderr)
        name, value = checkface(threshold=threshold_model, face=face, database=db, debug=True)
        if name:
            db[name]["cnt"] += 1
            db[name]["last_seen"] = datetime.now()
            if value < threshold_pretty_sure:
                openDoor(name, push_url)
            else:
                approveclearance(db, push_url)
            continue
print("Finished!")