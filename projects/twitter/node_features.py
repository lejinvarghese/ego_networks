"""
This can serve as the refactoring base for later implementations.
"""

import os
from dotenv import load_dotenv
import time
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import pandas as pd
import numpy as np
import dask.dataframe as dd
import tweepy
import ast
from datetime import datetime

load_dotenv()
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

n_threads = cpu_count() - 1
run_time = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
TWITTER_USERNAME = os.getenv("TWITTER_USERNAME")
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_API_BEARER_TOKEN = os.getenv("TWITTER_API_BEARER_TOKEN")

client = tweepy.Client(TWITTER_API_BEARER_TOKEN, wait_on_rate_limit=True)


def get_node_features(users):
    data = client.get_users(
        ids=users,
        user_fields=["profile_image_url", "username", "public_metrics", "verified"],
    ).data
    return pd.DataFrame(data)


def main():
    user = client.get_user(
        username=TWITTER_USERNAME,
        user_fields=["id"],
    ).data.id

    df = dd.read_csv("data/users_following*.csv").compute()
    df.following = df.following.apply(ast.literal_eval)
    df = df.explode("following")

    df_user = pd.DataFrame({"user": user, "following": df.user.unique()})
    df = pd.concat([df, df_user])

    print(f"Total nodes and edges: {df.shape}")

    existing_users_d1 = list(df.user.unique())
    print(f"Nodes at radius 1: {len(existing_users_d1)}")

    existing_users_d2 = list(df.following.unique())
    print(f"Nodes at radius 2: {len(existing_users_d2)}")

    all_nodes = np.array(list(set(existing_users_d1).union(set(existing_users_d2))))
    all_nodes = all_nodes[~np.isnan(all_nodes)].astype(int)
    print(f"Total nodes upto radius 2: {len(all_nodes)}")

    batch_size = 100
    all_nodes_batches = np.array_split(all_nodes, len(all_nodes) // (batch_size - 1))
    all_nodes_batches = [batch.tolist() for batch in all_nodes_batches]

    print(
        f"Total batches: {np.shape(all_nodes_batches)}, batch size:{np.shape(all_nodes_batches[0])}"
    )

    data = []
    for batch in all_nodes_batches:
        time.sleep(1)
        data.append(get_node_features(batch))

    data = pd.concat(data).reset_index()
    data_public_metrics = pd.json_normalize(data["public_metrics"])
    data = pd.merge(data, data_public_metrics, left_index=True, right_index=True)

    print(data.head())

    data.to_csv(f"data/node_features_{run_time}.csv", index=False)


if __name__ == "__main__":
    main()