import json
import requests
import traceback
from utils.helpers import log
import settings


def serialize(obj):
    """Function to convert non-serializable objects to a serializable format."""
    if isinstance(obj, (dict, list, str, int, float, bool, type(None))):
        return obj
    return str(obj) 

def save_sources_log(message_id, sources={}):
    if settings.DEBUG == True:
        log("Skipping save_sources_log")
        return
    """
        Saves the sources log for a given message to a Laravel backend.
        
        This function sends a POST request to the specified Laravel backend URL with the
        message ID and sources data. The sources data is serialized to JSON format before
        being sent. If the request is successful, a success message is logged. If an error
        occurs during the request, an error message is logged along with the exception details.

        Args:
            message_id (str): The ID of the message for which the sources log is being saved.
            sources (dict, optional): A dictionary containing the sources data to be saved.
                Defaults to an empty dictionary.

        Returns:
            None
    """
    try:  
        res = requests.post(
            f"{settings.LARAVEL_BASEURL}/store_message_sources",
            data={
                "message_id": message_id,
                "sources": json.dumps(sources if isinstance(sources, dict) else {}, default=serialize)
            }
        )
        res.raise_for_status()  # Raise exception if response status is not ok
        log(f"Sources log saved successfully for {message_id}")
    except requests.exceptions.RequestException as e:
        log("Error in send_sources_log", e)
    except:
        log("Error in send_sources_log", traceback.format_exc())

