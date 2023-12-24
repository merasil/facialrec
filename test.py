import threading
from time import sleep
from datetime import datetime
import cv2 as cv
import sys
from libs.functions import *


bgm = cv.createBackgroundSubtractorMOG2()
bgm_learning_rate = -1


stream = CameraBufferCleanerThread("rtsp://viewer:123456@172.23.0.104:554/h264Preview_01_main")
motion = MotionDetectionThread(stream, bgm, bgm_learning_rate, 20, True)
while(True):
    print("zzzz")
    sleep(1)