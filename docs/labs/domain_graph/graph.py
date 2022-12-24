# -*- coding: utf-8 -*-
from typing import List
from warnings import filterwarnings

import numpy as np
import pandas as pd
from networkx import Graph, from_pandas_edgelist
from tensorflow_hub import load as hub_load

# from tensorflow import get_logger as tf_get_logger

# tf_get_logger().setLevel("ERROR")
filterwarnings("ignore")


class DomainGraph:
    def __init__(self, nodes: List[str]):
        self.nodes = nodes
        self.embedding_model = hub_load(
            "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        )

    def __get_embeddings(self, tokens: List[str]) -> np.ndarray:
        return np.array(self.embedding_model(tokens))

    def __get_similarities(
        self, tokens: List[str], embeddings: np.ndarray, max_edges: int = 5
    ) -> pd.DataFrame:
        from faiss import IndexFlatL2, get_num_gpus, index_cpu_to_all_gpus

        print(f"number of gpus >> {get_num_gpus()}")

        cpu_index = IndexFlatL2(embeddings.shape[1])
        gpu_index = index_cpu_to_all_gpus(cpu_index)
        gpu_index.add(embeddings)

        D, I = gpu_index.search(embeddings, max_edges + 1)

        I = pd.DataFrame(
            I,
            columns=["item_" + str(i) for i in range(max_edges + 1)],
            index=tokens,
        )
        D = pd.DataFrame(
            D,
            columns=["score_" + str(i) for i in range(max_edges + 1)],
            index=tokens,
        )
        similar_tokens = I.applymap(lambda x: tokens[x].title()).melt(
            id_vars=["item_0"]
        )
        similar_scores = D.applymap(lambda x: np.round(1 - x, 4)).melt(
            id_vars=["score_0"]
        )

        similar_tokens.columns = ["source", "_rank", "target"]
        similar_tokens["_rank"] = similar_tokens["_rank"].apply(
            lambda x: int(x.split("_")[1])
        )
        similar_scores.columns = ["source", "_rank", "_score"]
        similar_tokens = similar_tokens.merge(
            similar_scores[["_score"]], left_index=True, right_index=True
        )
        similar_tokens.sort_values(
            by=["source", "_rank"], ascending=True, inplace=True
        )
        return similar_tokens[similar_tokens.sim_rank > 0]

    def __create_edges(self):
        domain_embeddings = self.__get_embeddings(tokens=self.nodes)
        similar_tokens = self.__get_similarities(
            embeddings=domain_embeddings, tokens=self.nodes
        )
        similar_tokens["sim_score_min"] = similar_tokens.groupby("source")[
            "sim_score"
        ].transform("min")
        similar_tokens = similar_tokens[
            (similar_tokens.sim_score_min >= -0.15)
            | (similar_tokens.sim_rank <= 2)
        ]
        similar_tokens["sim_score"] = (
            similar_tokens["sim_score"]
            + abs(similar_tokens["sim_score"].min())
            + 0.1
        )
        similar_tokens.drop(columns=["sim_score_min", "sim_rank"], inplace=True)
        similar_tokens.rename(
            columns={
                "sim_score": "weight",
            },
            inplace=True,
        )

        return similar_tokens

    def create(self) -> Graph:

        edges = self.__create_edges()
        print(edges.head(10))
        return from_pandas_edgelist(edges)

    def get_diffusion_grades(self, content):
        return {"Smart Cities": 2, "Intelligence": 4, "Robotics": 6}
