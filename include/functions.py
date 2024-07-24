import sys
import requests
from datetime import datetime

def str2bool(v):
    return str(v).lower() in ("yes", "y", "true", "t", "1")

def openDoor(identity, push_url):
    print("---------------------------------------------", file=sys.stderr)
    print("{} INFO: Open Door for {}".format(datetime.now(), identity), file=sys.stderr)
    print("---------------------------------------------", file=sys.stderr)
    requests.get(push_url, {"value":"true"})
    
def resetDB(database, threshold):
    for identity in database:
        if database[identity]["cnt"] != 0:
            diff = datetime.now() - database[identity]["last_seen"]
            if diff.seconds >= threshold:
                database[identity]["cnt"] = 0