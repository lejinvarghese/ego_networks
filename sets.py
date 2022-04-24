import ast
import os

import dask.dataframe as dd
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
CLOUD_STORAGE_BUCKET = os.getenv("CLOUD_STORAGE_BUCKET")

previous_ties = dd.read_csv(
    f"{CLOUD_STORAGE_BUCKET}/data/users_following*.csv"
).compute()
print(previous_ties.head())
print(f"Storage bucket authenticated")
previous_ties.following = previous_ties.following.apply(ast.literal_eval)
previous_ties = previous_ties.explode("following")
previous_ties = previous_ties.dropna()
u = 1120633726478823425
print(previous_ties[previous_ties.user == u])

print(previous_ties.head())
