# -*- coding: utf-8 -*-
import ast
import os
from warnings import filterwarnings
from datetime import datetime

import dask.dataframe as dd
import pandas as pd

from dotenv import load_dotenv

try:
    from utils.custom_logger import CustomLogger
except ModuleNotFoundError:
    from ego_networks.utils.custom_logger import CustomLogger

load_dotenv()
filterwarnings("ignore")
logger = CustomLogger(__name__)


class DataConfig:
    root_dir = os.getenv("CLOUD_STORAGE_BUCKET")
    file_paths = {
        "ties": f"{root_dir}/ties",
        "node_features": f"{root_dir}/features/node",
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
            ).compute()
            logger.info(
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
            return data.dropna()
        elif self.data_type == "node_features":
            return data.drop_duplicates().set_index("id")
        else:
            return data


class DataWriter(DataConfig):
    def __init__(self, data, data_type):
        super().__init__()
        self.data = data
        self.data_type = data_type

    def run(self, append=True):
        logger.info(f"Writing {self.data_type}: {self.data.shape}")
        file_path = self.file_paths.get(self.data_type)
        if append:
            run_time = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
            file_path = f"{file_path}/{self.data_type}_{run_time}"
        else:
            file_path = f"{file_path}/{self.data_type}"

        self.data.to_csv(
            f"{file_path}.csv",
            index=False,
        )
