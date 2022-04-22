# -*- coding: utf-8 -*-
import os
import sys

PATH = os.path.abspath(".")
os.environ["KMP_WARNINGS"] = "FALSE"
sys.path.insert(1, PATH)

from utils import timer, draw_graph_interactive
from warnings import filterwarnings
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
from faiss import get_num_gpus, IndexFlatL2, index_cpu_to_all_gpus
import networkx as nx
import community as community_louvain

tf.get_logger().setLevel("ERROR")
filterwarnings("ignore")

N_GPU = get_num_gpus()
print(f"Number of GPUs: {N_GPU}")


@timer
def get_embeddings(model, tokens):
    embeddings = np.array(hub.load(model)(tokens))
    return embeddings


@timer
def get_similarities(embeddings, tokens):
    cpu_index = IndexFlatL2(embeddings.shape[1])
    gpu_index = index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(embeddings)

    top_neighbors = 5 + 1
    D, I = gpu_index.search(embeddings, top_neighbors)

    I = pd.DataFrame(
        I, columns=["item_" + str(i) for i in range(top_neighbors)], index=tokens
    )
    D = pd.DataFrame(
        D, columns=["score_" + str(i) for i in range(top_neighbors)], index=tokens
    )
    similar_tokens = I.applymap(lambda x: tokens[x].title())
    similar_scores = D.applymap(lambda x: np.round(1 - x, 4))

    similar_tokens = similar_tokens.melt(id_vars=["item_0"])
    similar_scores = similar_scores.melt(id_vars=["score_0"])

    similar_tokens.columns = ["ref_item", "sim_rank", "comp_item"]
    similar_tokens["sim_rank"] = similar_tokens["sim_rank"].apply(
        lambda x: int(x.split("_")[1])
    )
    similar_scores.columns = ["ref_item", "sim_rank", "sim_score"]
    similar_tokens = similar_tokens.merge(
        similar_scores[["sim_score"]], left_index=True, right_index=True
    )
    similar_tokens.sort_values(
        by=["ref_item", "sim_rank"], ascending=True, inplace=True
    )
    similar_tokens = similar_tokens[similar_tokens.sim_rank > 0]
    similar_tokens["sim_score_min"] = similar_tokens.groupby("ref_item")[
        "sim_score"
    ].transform("min")
    similar_tokens = similar_tokens[
        (similar_tokens.sim_score_min >= -0.15) | (similar_tokens.sim_rank <= 2)
    ]
    similar_tokens["sim_score"] = (
        similar_tokens["sim_score"] + abs(similar_tokens["sim_score"].min()) + 0.001
    )
    similar_tokens.drop(columns=["sim_score_min", "sim_rank"], inplace=True)

    return similar_tokens


if __name__ == "__main__":
    tokens = [
        "complexity theory",
        "reinforcement learning",
        "deep learning",
        "machine learning",
        "intelligence",
        "space exploration",
        "astronomy",
        "cognitive neuroscience",
        "complex socio-technical systems",
        "smart cities",
        "genetics",
        "diversity",
        "equity",
        "network science",
        "evolutionary theory",
        "optimization algorithms",
        "causality",
        "statistics",
        "recommender systems",
        "chaos theory",
        "innovation",
        "non linear dynamics",
        "artificial life",
        "astrophysics",
        "electronics",
        "internet of things",
        "federated machine learning",
        "marketing",
        "philosophy",
        "psychology",
        "ethics",
        "aesthetics",
        "urban mobility",
        "computational social science",
        "politics",
        "robotics",
        "game theory",
    ]
    models = [
        "https://tfhub.dev/google/universal-sentence-encoder-large/5",
    ]
    results = []

    print(f"Sample tokens: {tokens[:5]}, Total tokens: {len(tokens)}")

    embeddings = get_embeddings(models[0], tokens)

    similar_tokens = get_similarities(embeddings, tokens)
    similar_tokens.rename(
        columns={"ref_item": "source", "comp_item": "target", "sim_score": "weight"},
        inplace=True,
    )
    print(similar_tokens.head(10))

    G = nx.from_pandas_edgelist(similar_tokens, "source", "target", ["weight"])

    size_map = nx.betweenness_centrality(
        G, weight="weight", normalized=True, endpoints=True, seed=42
    )

    color_map = community_louvain.best_partition(G, resolution=0.5)

    fig = draw_graph_interactive(G, color_map=color_map, size_map=size_map, title=" ")
    fig.write_html("domains/output.html")
    fig.write_image("domains/output.png", scale=3.0)
