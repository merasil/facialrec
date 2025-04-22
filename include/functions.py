import sys
import requests
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def str2bool(v):
    return str(v).lower() in ("yes", "y", "true", "t", "1")

def openDoor(identity, push_url):
    logging.info("Open Door for {}".format(identity))
    requests.get(push_url, {"value":"true"})

def resetDB(database, threshold):
    for identity in database:
        if database[identity]["cnt"] != 0:
            diff = datetime.now() - database[identity]["last_seen"]
            if diff.seconds >= threshold:
                database[identity]["cnt"] = 0
