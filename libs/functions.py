import cv2 as cv
import threading
from deepface import DeepFace
from time import sleep
import sys

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
                print("ERROR: Can not access Camera!", file=sys.stderr)
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
                print("ERROR: No Image received!", file=sys.stderr)
                print("---------------------------------------------", file=sys.stderr)
                self.counter += 1
                self.last_frame = None
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
    threshold_face_prob = (threshold_recognizer/threshold_img_cnt) #* 0.93
    
    if identity_cnt[face_name] >= threshold_img_cnt and face_prob <= threshold_face_prob and face_ratio <= threshold_face_prob:
         approved = True
    if debug:
        print("---------------------------------------------")
        print("Threshold probability: {}".format(threshold_face_prob))
        print("Recognized: {} with Mean: {} ({}) and Ratio: {}!\nApproved: {}".format(face_name, face_mean, face_prob, face_ratio, str(approved)))
        print("---------------------------------------------")
        print(face_name)
        print()
    return approved, face_name, face_mean, face_prob

def checkface(face, database="db", model="Facenet512", metric="euclidean_l2", debug=False):
    faces_recognized = []
    for index, row in face.iterrows():
        for identity in database:
            if identity in row["identity"] and row[f"{model}_{metric}"] <= database[identity]["threshold"]:
                faces_recognized.append({"identity":identity, "value":row[f"{model}_{metric}"]})
    if len(faces_recognized) == 1:
        if debug:
            print("---------------------------------------------")
            print("SUCCESS: Face found: {} ---> {}! Threshold: {}".format(faces_recognized[0]["identity"], faces_recognized[0]["value"], database[faces_recognized[0]["identity"]]["threshold"]))
            print("---------------------------------------------")
        return faces_recognized[0]["identity"]
    else:
        if debug:
            if len(faces_recognized) == 0:
                print("---------------------------------------------", file=sys.stderr)
                print("ERROR: No Face recognized!", file=sys.stderr)
                print("---------------------------------------------", file=sys.stderr)
            else:
                print("---------------------------------------------", file=sys.stderr)
                print("ERROR: Found more Faces with Thresholds passed!", file=sys.stderr)
                print("---------------------------------------------", file=sys.stderr)
        return False
        
        
        
        
        
        
        
        
        result = DeepFace.verify(face, database[identity]["path"])
        if result["verified"] and result["distance"] <= database[face]["threshold"]:
            if debug:
                print("---------------------------------------------")
                print("Found {} ---> Distance: {}\nThreshold: {}")
                print("---------------------------------------------")
                print()
            faces_recognized.append(face)
    if len(faces_recognized == 1):
        return faces_recognized[0]
    else:
        if debug:
            if not len(faces_recognized):
                print("---------------------------------------------")
                print("ERROR: No Face found!")
                print("---------------------------------------------")
            else:
                print("---------------------------------------------")
                print("ERROR: Found more Faces with Thresholds passed!")
                print("---------------------------------------------")
        return False