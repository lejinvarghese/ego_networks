import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Alibaba-NLP/gte-modernbert-base"


class DocumentEncoder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode(self, places: dict[str, str]) -> np.ndarray:
        documents = [
            f"{place['name']}. {place['description']}. tags: {', '.join(place['tags'])}"
            for place in places
        ]
        return self.model.encode(documents, normalize_embeddings=True)


class QueryEncoder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode(self, query: str) -> np.ndarray:
        return self.model.encode(query, normalize_embeddings=True)
