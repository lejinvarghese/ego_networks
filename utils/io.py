# -*- coding: utf-8 -*-
import ast
import os
from warnings import filterwarnings

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
        "node_measures": f"{root_dir}/measures/node/2",
    }


class DataReader(DataConfig):
    def __init__(self, data_type):
        super().__init__()
        self.data_type = data_type

    def run(self):
        try:
            data = dd.read_parquet(
                path=self.file_paths.get(self.data_type)
            ).compute()
            logger.info(f"Read successful: {self.data_type}")
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

    def run(self, append=True, n_partitions=10):
        logger.info(f"Writing {self.data_type}: {self.data.shape}")
        dd.to_parquet(
            dd.from_pandas(self.data, npartitions=n_partitions),
            path=self.file_paths.get(self.data_type),
            append=append,
            overwrite=not (append),
            ignore_divisions=True,
        )
