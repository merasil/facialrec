import sys
import requests
import logging
from datetime import datetime
from deepface.commons import distance as dst

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

def calc_distance(src_emb, test_emb, metric="cosine"):
    """Calculate distance between two embeddings using the given metric."""
    if metric == "cosine":
        return dst.findCosineDistance(src_emb, test_emb)
    if metric == "euclidean":
        return dst.findEuclideanDistance(src_emb, test_emb)
    if metric == "euclidean_l2":
        src_norm = dst.l2_normalize(src_emb)
        test_norm = dst.l2_normalize(test_emb)
        return dst.findEuclideanDistance(src_norm, test_norm)
    raise ValueError(f"Unsupported metric: {metric}")

