# -*- coding: utf-8 -*-
import ast
from datetime import datetime
from warnings import filterwarnings

import dask.dataframe as dd
import pandas as pd

filterwarnings("ignore")
run_time = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")


def split_into_batches(src_list: list, batch_size: int):
    batches = [
        src_list[x : x + batch_size]
        for x in range(0, len(src_list), batch_size)
    ]

    print(f"Total batches: {len(batches)}, batch size: {len(batches[0])}")
    return batches


def read_data(file_path, data_type):
    try:
        if data_type == "ties":
            data = dd.read_csv(f"{file_path}/data/{data_type}*.csv").compute()
            print(f"Read successful: {data_type}")
            data.following = data.following.apply(ast.literal_eval)
            data = data.explode("following")
            return data.dropna()
        elif data_type == "node_features":
            data = dd.read_csv(
                f"{file_path}/data/{data_type}*.csv",
                dtype={"withheld": "object"},
            ).compute()
            print(f"Read successful: {data_type}")
            return data.drop(columns="withheld").drop_duplicates()
        elif data_type == "node_measures":
            data = dd.read_csv(
                f"{file_path}/data/processed/measures/2/node_measures.csv",
            ).compute()
            print(f"Read successful: {data_type}")
            return data
        else:
            raise ValueError(f"Invalid data type: {data_type}")
    except Exception as error:
        print(f"Read unsuccessful: {data_type}, {error}")
        return pd.DataFrame()


def write_data(file_path, data, data_type):
    print(f"Writing {data_type}: {data.shape}")
    data.to_csv(
        f"{file_path}/data/{data_type}_{run_time}.csv",
        index=False,
    )
