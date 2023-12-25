from deepface import DeepFace
import cv2 as cv
import numpy as np
from time import sleep
from datetime import datetime
import os
import requests
import configparser
from libs.functions import *

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
debug = False
if config["basic"]["debug"] == "True":
    debug = True

############## Setting up Database #############
path_db = config["database"]["path"]
db = {}
for folder in os.scandir(path_db):
        if folder.is_dir():
            db[folder.name] = {"path":"{}/{}/{}.jpg".format(path_db,folder.name,folder.name), "threshold":0.0, "last_seen":datetime.now(), "cnt":0}
            if folder.name in config["thresholds"]:
                db[folder.name]["threshold"] = float(config["thresholds"][folder.name])

############## Setting up Face Recognition Model #############
model = config["face_recognition"]["model"]
detector = config["face_recognition"]["detector"]
metric = config["face_recognition"]["metric"]

############## Setting up Thresholds #############              
threshold_clearance = int(config["thresholds"]["clearance"])
threshold_last_seen = int(config["thresholds"]["last_seen"])
threshold_pretty_sure = float(config["thresholds"]["pretty_sure"])

############## Settings for Camera (URL, Thread, etc.) #############
stream = CameraBufferCleanerThread(stream_url)
sleep(5)
print(db)

##############  Starting the Application #############
while True:
    # Reseting DB if Face isnt recognized for a period of time...
    resetim(db)
    if stream.last_frame is None:
        if debug:
            print("---------------------------------------------")
            print("{} ERROR: Couldnt receive Frame. Continuing with next...".format(datetime.now()))
            print("---------------------------------------------")
        continue

    img = stream.last_frame.copy()
    try:
        faces = DeepFace.find(img_path=img, detector_backend=detector, db_path=path_db, distance_metric=metric, model_name=model, silent=True)
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
        name, value = checkface(face=face, database=db, model=model, metric=metric, debug=True)
        if name:
            db[name]["cnt"] += 1
            db[name]["last_seen"] = datetime.now()
            if value < threshold_pretty_sure:
                openDoor(name, push_url)
            else:
                approveclearance(db, push_url)
            continue
print("Finished!")