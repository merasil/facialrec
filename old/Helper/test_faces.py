from deepface import DeepFace

models = {"VGG-Face":0.283,
        "OpenFace":0.0,
        "Facenet":0.0,
        "Facenet512":0.0,
        "DeepID":0.0,
        "ArcFace":0.0,
        "SFace":0.0}

detectors = ["opencv",
        "ssd",
        "mtcnn",
        "retinaface",
        "mediapipe",
        "yolov8",
        "yunet"]
detector = detectors[6]

metrics = ["cosine", "euclidean", "euclidean_l2"]
metric = metrics[0]

imgs = {"test_face1.jpg":"name", "test_face2.jpg":"name", "test_face3.jpg":"name", "test_face4.jpg":"name2", "test_face5.jpg":"name2", "test_face6.jpg":"name2"}
for model in models:
    c_thr = models[model]
    if c_thr == 0.0:
        print("Skipping {}".format(model))
        print()
        continue
    print("{} with {} ({}):".format(model, metric, c_thr))
    for img in imgs:
        print(img)

        correct = False
        thr = 0.0
        thr_false = 0.0
        nFaces = 0

        result = DeepFace.find(img, "db", distance_metric=metric, detector_backend=detector, model_name=model)
        for face in result:
            if imgs[img] == "none" and face.shape[0] == 0:
                correct = True
            elif len(face) != 0:
                for index, row in face.iterrows():
                    thr_tmp = row[metric]
                    identity = row["identity"]
                    print(identity, ", ", thr_tmp)
                    if index == 0 and imgs[img] == "none" and thr_tmp > c_thr:
                        thr = thr_tmp
                        correct = True
                        break
                    if imgs[img] in identity and thr_tmp < c_thr:
                        correct = True
                        thr = thr_tmp
                        nFaces = index + 1
                        continue
                    thr_false = thr_tmp    
                    break
            if correct:
                if imgs[img] == "none":
                    print("PASSED!\nFace is not in DB or below Threshold. Result is correct!")
                else:
                    print("PASSED!\nRecognized Faces: {}, Threshold: {}".format(nFaces, thr))
            else:
                if imgs[img] == "none":
                    print("FAILED!\nFound Face: {} with Threshold: {}! Should be none!".format(face.iloc[0]["identity"], thr_false))
                else:
                    print("FAILED!\nDidnt find Face or Face was incorrectly recognized!")
        print("---------------------------------------")
    print()
