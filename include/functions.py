import sys
import requests
from datetime import datetime

def str2bool(v):
    return str(v).lower() in ("yes", "y", "true", "t", "1")

def checkMotion(motion_url):
    try:
        motion = requests.get(motion_url)
        if motion.status_code not in range(200, 204):
            return False
        else:
            motion = motion.json()
            if motion["val"] == "ON":
                return True
            return False
    except:
        return False

def openDoor(identity, push_url):
    print("Open Door for {}!".format(identity))
    requests.get(push_url, {"value":"true"})
    
def resetDB(database, threshold):
    for identity in database:
        if database[identity]["cnt"] != 0:
            diff = datetime.now() - database[identity]["last_seen"]
            if diff.seconds >= threshold:
                database[identity]["cnt"] = 0