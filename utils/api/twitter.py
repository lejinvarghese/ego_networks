# -*- coding: utf-8 -*-
import os
import time
from warnings import filterwarnings

from dotenv import load_dotenv

try:
    from utils.custom_logger import CustomLogger
except ModuleNotFoundError:
    from ego_networks.utils.custom_logger import CustomLogger

load_dotenv()
filterwarnings("ignore")
logger = CustomLogger(__name__)

TWITTER_USERNAME = os.getenv("TWITTER_USERNAME")
TWITTER_API_BEARER_TOKEN = os.getenv("TWITTER_API_BEARER_TOKEN")


def authenticate(api_bearer_token):
    import tweepy

    client = tweepy.Client(api_bearer_token, wait_on_rate_limit=True)
    return client


def get_users_following(
    client, user_id, max_results=1000, total_limit=5000, sleep_time=0.1
):
    import tweepy

    following = []
    for neighbor in tweepy.Paginator(
        client.get_users_following, id=user_id, max_results=max_results
    ).flatten(limit=total_limit):
        time.sleep(sleep_time)
        following.append(neighbor.id)
    logger.info(f"User: {user_id}, Following: {len(following)}")
    return {"user": user_id, "following": following}


def get_users(client, user_fields, user_names=None, user_ids=None):

    if user_ids:
        return client.get_users(
            ids=user_ids,
            user_fields=user_fields,
        ).data
    elif user_names:
        return client.get_users(
            usernames=user_names,
            user_fields=user_fields,
        ).data
    else:
        raise ValueError(
            "Either one of user_names or user_ids should be provided"
        )
