import threading
from time import sleep
from datetime import datetime
import cv2 as cv
import sys
from libs.functions import *

def recordingThread(state):
    state["running"] = True
    print(state)
    debug = True
    if debug:
        print("---------------------------------------------", file=sys.stderr)
        print("{} SUCCESS: Started Recording Thread...".format(datetime.now()), file=sys.stderr)
        print("---------------------------------------------", file=sys.stderr)
    start_time = datetime.now()
    filename = "{}{}{}_{}{}{}.avi".format(start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    writer = cv.VideoWriter(filename, cv.VideoWriter_fourcc('M','P','4','2'), 10, (2560,1440))
    while (datetime.now() - start_time).seconds < 10:
        try:
            img = state["stream"].last_frame.copy()
        except:
            continue
        #img = cv.resize(img, (1280,720), interpolation= cv.INTER_LINEAR)
        writer.write(img)
        sleep(0.1)
    state["running"] = False
    if debug:
        print("---------------------------------------------", file=sys.stderr)
        print("{} SUCCES: Closed Recording Thread...".format(datetime.now()), file=sys.stderr)
        print("---------------------------------------------", file=sys.stderr)
    return 0

state = [1]
stream = CameraBufferCleanerThread("rtsp://viewer:123456@172.23.0.104:554/h264Preview_01_main")
sleep(5)
state[0] = {"running": False, "stream": stream}
print(state[0])

for s in state:
    x = threading.Thread(target=recordingThread, args=(s, ))
    x.start()
i = 0
while(i < 100):
    print(state[0])
    sleep(0.5)