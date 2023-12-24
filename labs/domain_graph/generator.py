# -*- coding: utf-8 -*-
from typing import List
from warnings import filterwarnings

import numpy as np
import pandas as pd
from networkx import Graph, from_pandas_edgelist
from tensorflow_hub import load as hub_load

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

        if get_num_gpus() == 0:
            raise ValueError(
                "No GPUs found. GPU index cannot be used in this instance."
            )

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
        similar_tokens = I.applymap(lambda x: tokens[x]).melt(
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
        return similar_tokens[similar_tokens._rank > 0]

    def __create_edges(self):
        domain_embeddings = self.__get_embeddings(tokens=self.nodes)
        similar_tokens = self.__get_similarities(
            embeddings=domain_embeddings, tokens=self.nodes
        )
        similar_tokens["_min_score"] = similar_tokens.groupby("source")[
            "_score"
        ].transform("min")
        similar_tokens = similar_tokens[
            (similar_tokens._min_score >= -0.15) | (similar_tokens._rank <= 2)
        ]
        similar_tokens["_score"] = (
            similar_tokens["_score"] + abs(similar_tokens["_score"].min()) + 0.1
        )
        similar_tokens.drop(columns=["_min_score", "_rank"], inplace=True)
        similar_tokens.rename(
            columns={
                "_score": "weight",
            },
            inplace=True,
        )
        cols = ["source", "target"]
        similar_tokens[cols] = similar_tokens[cols].applymap(
            lambda x: x.title()
        )
        return similar_tokens

    def create_graph(self) -> Graph:

        edges = self.__create_edges()
        return from_pandas_edgelist(edges)

    def get_diffusion_grades(self, content):
        tokens = [*self.nodes, *content]

        embeddings = self.__get_embeddings(tokens=tokens)
        similar_tokens = self.__get_similarities(
            embeddings=embeddings, tokens=tokens, max_edges=len(tokens)
        )
        similar_tokens = similar_tokens[
            (similar_tokens.source.isin(self.nodes))
        ]
        similar_tokens = similar_tokens[
            (similar_tokens.target.isin(content))
        ].sort_values(by=["_score"], ascending=False)
        similar_tokens = similar_tokens[(similar_tokens._score >= -0.50)]
        similar_tokens["_score"] = (
            similar_tokens["_score"] + abs(similar_tokens["_score"].min()) + 0.1
        )
        similar_tokens["source"] = similar_tokens["source"].apply(
            lambda x: x.title()
        )
        similar_tokens = (
            similar_tokens.groupby("source").target.count().to_dict()
        )
        return {k: 4 * v for k, v in similar_tokens.items()}
