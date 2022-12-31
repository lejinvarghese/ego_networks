# -*- coding: utf-8 -*-
import os
import time
from warnings import filterwarnings

from dotenv import load_dotenv

try:
    from utils.default import url_exists
    from utils.custom_logger import CustomLogger
except ModuleNotFoundError:
    from ego_networks.utils.default import url_exists
    from ego_networks.utils.custom_logger import CustomLogger

load_dotenv()
filterwarnings("ignore")
logger = CustomLogger(__name__)

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


def get_twitter_profile_image(
    user_name: str,
    image_url: str,
    image_size: int = 200,
) -> bool:

    if not (url_exists(image_url)):
        image_url = get_user_profile_image(user_name)
    return image_url.replace("_normal", f"_{image_size}x{image_size}")


def get_user_profile_image(
    user_name: str,
    default_image_url: str = "https://cpraonline.files.wordpress.com/2014/07/new-twitter-logo-vector-200x200.png",
):
    client = authenticate(TWITTER_API_BEARER_TOKEN)
    image_url = get_users(
        client, user_fields=["profile_image_url"], user_names=[user_name]
    )[0].get("profile_image_url")
    if image_url:
        return image_url
    else:
        return default_image_url


def get_engagement(
    client,
    user_id,
    content_type="tweets",
    tweet_fields=[
        "context_annotations",
        "in_reply_to_user_id",
        "public_metrics",
        "entities",
        "created_at",
    ],
):
    if content_type == "tweets":
        return client.get_users_tweets(
            id=user_id, tweet_fields=tweet_fields
        ).data
    elif content_type == "mentions":
        return client.get_users_mentions(
            id=user_id, tweet_fields=tweet_fields
        ).data
    elif content_type == "likes":
        return client.get_liked_tweets(
            id=user_id, tweet_fields=tweet_fields
        ).data
