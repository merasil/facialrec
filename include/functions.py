import sys
import requests
import logging
from datetime import datetime
from typing import Dict, Any

def str2bool(v: Any) -> bool:
    """Convert string or other value to boolean"""
    return str(v).lower() in ("yes", "y", "true", "t", "1")

def openDoor(identity: str, push_url: str, verbose: int = 1) -> bool:
    """
    Trigger door opening mechanism via HTTP request

    Args:
        identity: Name of the person to open the door for
        push_url: URL to trigger the door mechanism
        verbose: Logging verbosity level (0-4)

    Returns:
        True if request was successful, False otherwise
    """
    try:
        if verbose >= 1:
            logging.info(f"Opened for {identity}")
        response = requests.get(push_url, params={"value": "true"}, timeout=5)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to open door for {identity}: {e}")
        return False

def resetDB(database: Dict[str, Dict[str, Any]], threshold: int) -> None:
    """
    Reset recognition counters for identities that haven't been seen recently

    Args:
        database: Dictionary containing identity data with counters and timestamps
        threshold: Time threshold in seconds after which to reset the counter
    """
    if not database:
        return

    for identity in database:
        if database[identity]["cnt"] != 0:
            diff = datetime.now() - database[identity]["last_seen"]
            if diff.total_seconds() >= threshold:
                database[identity]["cnt"] = 0
                database[identity]["last_opened"] = None
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(f"Reset counter for {identity}")