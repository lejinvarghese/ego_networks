# -*- coding: utf-8 -*-
import urllib
from datetime import datetime
from warnings import filterwarnings

from dotenv import load_dotenv

try:
    from utils.custom_logger import CustomLogger

except ModuleNotFoundError:
    from ego_networks.utils.custom_logger import CustomLogger


filterwarnings("ignore")
logger = CustomLogger(__name__)
load_dotenv()
run_time = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")


def split_into_batches(src_list: list, batch_size: int) -> list:
    batches = [
        src_list[x : x + batch_size]
        for x in range(0, len(src_list), batch_size)
    ]

    logger.debug(
        f"Total batches: {len(batches)}, batch size: {len(batches[0])}"
    )
    return batches


def url_exists(location: str) -> bool:
    request = urllib.request.Request(location)
    request.get_method = lambda: "HEAD"
    try:
        response = urllib.request.urlopen(request)
        return True
    except urllib.error.HTTPError:
        return False
