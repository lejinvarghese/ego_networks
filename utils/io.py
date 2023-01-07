# -*- coding: utf-8 -*-
import ast
import os
from warnings import filterwarnings
from datetime import datetime

import dask.dataframe as dd
import pandas as pd
from google.cloud import storage

from dotenv import load_dotenv

try:
    from utils.custom_logger import CustomLogger
except ModuleNotFoundError:
    from ego_networks.utils.custom_logger import CustomLogger

load_dotenv()
filterwarnings("ignore")
logger = CustomLogger(__name__)


class DataConfig:
    gcs_client = storage.Client()
    root_dir = os.getenv("CLOUD_STORAGE_BUCKET")
    bucket = gcs_client.get_bucket(root_dir.split("/")[-2:-1][0])
    file_paths = {
        "ties": f"{root_dir}/ties",
        "node_features": f"{root_dir}/features/node",
        "tie_features": f"{root_dir}/features/ties",
        "node_measures": f"{root_dir}/measures/node",
    }


class DataReader(DataConfig):
    def __init__(self, data_type):
        super().__init__()
        self.data_type = data_type

    def run(self):
        try:
            data = dd.read_csv(
                urlpath=f"{self.file_paths.get(self.data_type)}/*.csv",
                dtype={"withheld": "object"},
            ).compute()
            logger.debug(
                f"Read successful: {self.data_type}, shape: {data.shape}"
            )
            return self.__preprocess(data)
        except Exception as error:
            logger.error(f"Read unsuccessful: {self.data_type}, {error}")
            return pd.DataFrame()

    def __preprocess(self, data):
        if self.data_type == "ties":
            data.following = data.following.apply(ast.literal_eval)
            data = data.explode("following")
            return data.dropna().drop_duplicates()
        elif self.data_type == "node_features":
            return (
                data.drop_duplicates(subset="id")
                .set_index("id")
                .drop(columns="witheld", errors="ignore")
            )
        elif self.data_type == "tie_features":
            data = data.rename(
                columns={
                    "user_id": "source",
                    "in_reply_to_user_id": "target",
                    "tweet_id": "weight",
                }
            )
            data["source"] = data["source"].astype(int)
            data["target"] = data["target"].fillna(0.0).astype(int)
            data = (
                data[(data.source != data.target) & (data.target != 0)]
                .groupby(["source", "target"])
                .weight.count()
                .reset_index()
            )
            data["weight"] = data["weight"] + 1
            return data
        else:
            return data


class DataWriter(DataConfig):
    def __init__(self, data, data_type):
        super().__init__()
        self.data = data
        self.data_type = data_type

    def run(self, append=True):
        logger.debug(f"Writing {self.data_type}: {self.data.shape}")

        if append:
            run_time = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
            file_path = f"{self.file_paths.get(self.data_type)}/{self.data_type}_{run_time}"
        else:
            self.__remove()
            file_path = (
                f"{self.file_paths.get(self.data_type)}/{self.data_type}"
            )

        self.data.to_csv(
            f"{file_path}.csv",
            index=False,
        )

    def __remove(self):
        directory_path = "/".join(
            self.file_paths.get(self.data_type).split("/")[3:]
        )
        blobs = self.bucket.list_blobs(prefix=directory_path)
        for blob in blobs:
            # archive
            self.bucket.copy_blob(
                blob,
                self.bucket,
                f"archive/{blob.name}",
            )
            # delete
            blob.delete()
