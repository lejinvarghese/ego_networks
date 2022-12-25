# -*- coding: utf-8 -*-

import requests
import re
from warnings import filterwarnings
import pandas as pd

from dotenv import load_dotenv

try:
    from utils.custom_logger import CustomLogger
except ModuleNotFoundError:
    from ego_networks.utils.custom_logger import CustomLogger

load_dotenv()
filterwarnings("ignore")
logger = CustomLogger(__name__)

def get_shelf_data(user_id, shelf, date_key="read"):
    base_shelf = f"https://www.goodreads.com/review/list/{user_id}?per_page=infinite&shelf={shelf}"
    r = requests.get(base_shelf)
    data = pd.read_html(r.text)[-1]
    data["title"] = data["title"].apply(
        lambda x: re.sub(r"^(title )", "", x).lower()
    )
    data["author"] = data["author"].apply(
        lambda x: re.sub(r"(^author )|( \*$)", "", x).lower()
    )
    data["date"] = pd.to_datetime(
        data[date_key].apply(lambda x: re.sub(fr"^(date {date_key} )", "", x)),
        errors="ignore",
    )
    cols = ["title", "author", "date"]
    data = data[cols].sort_values(by="date", ascending=False)
    data["shelf"] = shelf
    return data