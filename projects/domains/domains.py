import faiss
import os
import sys
from warnings import filterwarnings
import numpy as np
import pandas as pd

os.environ["KMP_WARNINGS"] = "FALSE"

import tensorflow as tf
import tensorflow_hub as hub

tf.get_logger().setLevel("ERROR")
filterwarnings("ignore")

PROJECT = os.getcwd()

N_GPU = faiss.get_num_gpus()
print("number of GPUs:", N_GPU)
print(hub.__version__, PROJECT)


def get_embeddings(model, tokens):
    embeddings = np.array(hub.load(model)(tokens))
    return embeddings


def get_similarities(embeddings, tokens):
    cpu_index = faiss.IndexFlatL2(embeddings.shape[1])
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(embeddings)

    top_neighbors = 5 + 1
    D, I = gpu_index.search(embeddings, top_neighbors)

    print(np.shape(D), np.shape(I))

    I = pd.DataFrame(I, columns=["item_" + str(i) for i in range(top_neighbors)], index=tokens)
    D = pd.DataFrame(D, columns=["score_" + str(i) for i in range(top_neighbors)], index=tokens)
    similar_tokens = I.applymap(lambda x: tokens[x])
    similar_scores = D.applymap(lambda x: np.round(1 - x, 4))

    print(similar_tokens.head(20), similar_scores.head())

    similar_tokens = similar_tokens.melt(id_vars=["item_0"])
    similar_scores = similar_scores.melt(id_vars=["score_0"])

    similar_tokens.columns = ["ref_item", "sim_rank", "comp_item"]
    similar_tokens["sim_rank"] = similar_tokens["sim_rank"].apply(lambda x: int(x.split("_")[1]))
    similar_scores.columns = ["ref_item", "sim_rank", "sim_score"]
    similar_tokens = similar_tokens.merge(
        similar_scores[["sim_score"]], left_index=True, right_index=True
    )
    similar_tokens.sort_values(by=["ref_item", "sim_rank"], ascending=True, inplace=True)
    similar_tokens = similar_tokens[similar_tokens.sim_rank > 0]
    similar_tokens["sim_score_min"] = similar_tokens.groupby("ref_item")["sim_score"].transform(
        "min"
    )
    similar_tokens = similar_tokens[
        (similar_tokens.sim_score_min > -0.25) | (similar_tokens.sim_rank < 2)
    ]
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
        "polymathy",
        "transdisciplinary research",
        "sociotechnical systems",
        "smart cities",
        "DNA",
        "diversity",
        "equity",
        "network science",
        "evolutionary theory",
        "optimization algorithms",
        "causality",
        "recommender systems",
        "chaos theory",
        "innovation",
        "non linear dynamics",
        "life",
        "astrophysics",
        "electronics",
        "internet of things",
        "federated machine learning",
        "marketing",
        "branding",
        "philosophy",
        "ethics",
        "aesthetics",
        "urban mobility",
        "computational social science",
        "politics",
    ]
    models = [
        "https://tfhub.dev/google/universal-sentence-encoder-large/5",
    ]
    results = []

    print(f"Sample tokens: {tokens[:5]}, Total tokens: {len(tokens)}")

    embeddings = get_embeddings(models[0], tokens)
    print(np.shape(embeddings))

    similar_tokens = get_similarities(embeddings, tokens)
    print(similar_tokens.head(10))