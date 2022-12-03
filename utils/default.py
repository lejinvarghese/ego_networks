# -*- coding: utf-8 -*-
import ast
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


def split_into_batches(src_list: list, batch_size: int):
    batches = [
        src_list[x : x + batch_size]
        for x in range(0, len(src_list), batch_size)
    ]

    logger.info(f"Total batches: {len(batches)}, batch size: {len(batches[0])}")
    return batches


def file_exists(location: str) -> bool:
    request = urllib.request.Request(location)
    request.get_method = lambda: "HEAD"
    try:
        response = urllib.request.urlopen(request)
        return True
    except urllib.error.HTTPError:
        return False


def twitter_profile_image_preprocess(
    image_url: str,
    default_image_url: str = "https://cpraonline.files.wordpress.com/2014/07/new-twitter-logo-vector-200x200.png",
    image_size: int = 200,
):
    image_url = image_url.replace("_normal", f"_{image_size}x{image_size}")
    if file_exists(image_url):
        return image_url
    else:
        return default_image_url