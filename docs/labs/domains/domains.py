# -*- coding: utf-8 -*-
import os
import sys

ROOT_DIRECTORY = os.path.abspath(".")
os.environ["KMP_WARNINGS"] = "FALSE"
sys.path.insert(1, ROOT_DIRECTORY)

from warnings import filterwarnings

import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from faiss import IndexFlatL2, get_num_gpus, index_cpu_to_all_gpus

try:
    from utils.graph import draw_nx_graph
    from docs.labs.domains.properties import get_graph_properties
except ModuleNotFoundError:
    from ego_networks.utils.graph import draw_nx_graph
    from ego_networks.docs.labs.domains.properties import get_graph_properties

tf.get_logger().setLevel("ERROR")
filterwarnings("ignore")
FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

print(f"number of gpus >> {get_num_gpus()}")


def get_embeddings(model, tokens):
    embeddings = np.array(hub.load(model)(tokens))
    return embeddings


def get_similarities(embeddings, tokens):
    cpu_index = IndexFlatL2(embeddings.shape[1])
    gpu_index = index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(embeddings)

    top_neighbors = 5 + 1
    D, I = gpu_index.search(embeddings, top_neighbors)

    I = pd.DataFrame(
        I,
        columns=["item_" + str(i) for i in range(top_neighbors)],
        index=tokens,
    )
    D = pd.DataFrame(
        D,
        columns=["score_" + str(i) for i in range(top_neighbors)],
        index=tokens,
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
        similar_tokens["sim_score"]
        + abs(similar_tokens["sim_score"].min())
        + 0.001
    )
    similar_tokens.drop(columns=["sim_score_min", "sim_rank"], inplace=True)

    return similar_tokens


def main():
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

    # models = [
    #     "https://tfhub.dev/google/universal-sentence-encoder-large/5",
    # ]
    # print(f"Sample tokens: {tokens[:5]}, Total tokens: {len(tokens)}")

    # embeddings = get_embeddings(models[0], tokens)

    # similar_tokens = get_similarities(embeddings, tokens)
    # similar_tokens.rename(
    #     columns={
    #         "ref_item": "source",
    #         "comp_item": "target",
    #         "sim_score": "weight",
    #     },
    #     inplace=True,
    # )
    # print(similar_tokens.head(10))

    # similar_tokens.to_csv("similar_tokens.csv", index=False)

    similar_tokens = pd.read_csv("similar_tokens.csv")

    G = nx.from_pandas_edgelist(similar_tokens, "source", "target", ["weight"])

    node_colors, node_sizes, edge_colors, edge_sizes = get_graph_properties(G)

    draw_nx_graph(
        G,
        node_color=node_colors,
        node_size=node_sizes,
        font_size=14,
        node_label_font_color="black",
        edge_colors=edge_colors,
        dpi=240,
        figsize=(40, 40),
        width=edge_sizes,
        save=True,
        file_path=f"{FILE_DIRECTORY}/figure.png",
        random_state=34,  # 49
    )

    # pos = nx.spring_layout(G, seed=100)
    # nx.draw(
    #     G,
    #     with_labels=True,
    #     pos=pos,
    #     node_color=list(color_map.values()),
    #     font_weight="bold",
    #     # edge_color=color_map,
    #     # node_size=color_map,
    # )
    # plt.savefig("output_x.png", format="PNG")
    # fig.write_html("domains/output_x.html")
    # fig.write_image("domains/output_x.png", scale=3.0)


if __name__ == "__main__":
    main()
