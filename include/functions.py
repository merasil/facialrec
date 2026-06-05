import logging
from datetime import datetime
from typing import Any

import requests


def door_open(door_name: str, door_url: str, door_verbose: int = 1) -> bool:
    try:
        if door_verbose >= 1:
            logging.info("Opened for %s", door_name)
        door_resp = requests.get(
            door_url,
            params={"value": "true"},
            timeout=5,
        )
        door_resp.raise_for_status()
        return True
    except requests.exceptions.RequestException as door_err:
        logging.error("Failed to open door for %s: %s", door_name, door_err)
        return False


def db_reset(db_data: dict[str, dict[str, Any]], db_threshold: int) -> None:
    for db_name, db_item in db_data.items():
        if db_item["cnt"] == 0:
            continue
        db_diff = datetime.now() - db_item["last_seen"]
        if db_diff.total_seconds() < db_threshold:
            continue
        db_item["cnt"] = 0
        db_item["last_opened"] = None
        logging.debug("Reset counter for %s", db_name)
