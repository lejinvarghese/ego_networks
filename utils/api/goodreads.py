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


def get_book_description(title, author):
    from json import loads

    base_uri = f"https://www.googleapis.com/books/v1/volumes?q=inauthor:{author.lower()}+intitle:{title.lower()}"
    r = requests.get(base_uri)
    try:
        cols = ["kind", "volumeInfo"]
        items = pd.DataFrame(loads(r.text).get("items"))[cols]
        items["rating"] = items["volumeInfo"].apply(
            lambda x: int(x.get("ratingsCount", 0))
        )
        items["description"] = items["volumeInfo"].apply(
            lambda x: x.get("description", "none").lower().strip()
        )
        items = (
            items[items.rating > 0]
            .sort_values(by="rating", ascending=False)
            .head(1)
        )
        desc = items.description.iloc[0]
    except:
        desc = " "
    return title.lower() + " " + re.sub(r"[^a-zA-Z0-9 \n\.]", " ", desc)


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
        data[date_key].apply(lambda x: re.sub(rf"^(date {date_key} )", "", x)),
        errors="ignore",
    )
    data["desc"] = data[["title", "author"]].apply(
        lambda x: get_book_description(x.title, x.author), axis=1
    )
    cols = ["title", "author", "date", "desc"]
    data = data[cols].sort_values(by="date", ascending=False)
    data["shelf"] = shelf
    return data
