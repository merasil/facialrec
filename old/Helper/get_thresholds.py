from deepface import DeepFace
import cv2 as cv
from libs.functions import *
from time import sleep
import configparser

def checkThresholds(face, person, values):
    for index, row in face.iterrows():
        metric_value = row[f"{model}_{metric}"]
        if person in row["identity"]:
            if values["low"] > metric_value:
                values["low"] = metric_value
            if values["high"] < metric_value:
                values["high"] = metric_value
            values["mean"].append(metric_value)
        else:
            if values["fp_low"] > metric_value:
                values["fp_low"] = metric_value
            if values["fp_high"] < metric_value:
                values["fp_high"] = metric_value
            values["mean"].append(metric_value)
    if len(values["mean"]):
        mean = sum(values["mean"]) / len(values["mean"])
    else:
        mean = 0
    if len(values["fp_mean"]):
        fp_mean = sum(values["fp_mean"]) / len(values["fp_mean"])
    else:
        fp_mean = 0
    return mean, fp_mean, values["mean"]

##############  Reading Config-File #############
config = configparser.ConfigParser()
config.read("config.ini")

model = config["basic"]["model"]
detector = config["basic"]["detector"]
metric = config["basic"]["metric"]
stream_url = config["basic"]["stream_url"]
push_url = config["basic"]["push_url"]
debug = True

##############  Settings for Camera (URL, Thread, etc.) #############
stream = CameraBufferCleanerThread(stream_url)
sleep(5)

persons = ["bastian", "laura", "max"]
db = "db"

values = {"low":100.0, "high":0.0, "mean":[], "fp":0, "fp_low":100.0, "fp_high":0.0, "fp_mean":[]}

print("Choose the Person you want to test:")
print("0. Bastian")
print("1. Laura")
print("2. Max")
x = input()
person = persons[int(x)]
f_faces = 0
r_faces = 0
sleep(5)

while True:
    if stream.last_frame is None:
       print("Couldnt receive Frame. Continuing with next...")
       continue
    try:
        img = stream.last_frame.copy()
        result = DeepFace.find(img_path=img, detector_backend=detector, db_path=db, distance_metric=metric, model_name=model, silent=True)
    except ValueError:
        #print("No Face found! Continuing...")
        continue
    except KeyboardInterrupt:
        print("Killing Process...")
        break
    except:
        print("Assertion Error! Continuing...")
        continue
    for face in result:
        if face.empty:
            f_faces += 1
        else:
            r_faces += 1
        mean, fp_mean, values["mean"] = checkThresholds(face, person, values)
    print("-----------------------------------")
    print("Person:", person)
    print("Thresholds (low, high, mean):", values["low"], values["high"], mean)
    print("FP Thresholds (low, high, mean):", values["fp_low"], values["fp_high"], fp_mean)
    print("Total Found Faces:", f_faces)
    print("Total Recognized Faces:", r_faces)
    print("-----------------------------------")
    print()
print("Finished...")