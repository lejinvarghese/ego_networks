import faiss
import numpy as np


class Retriever:
    def __init__(self, embeddings: list[np.ndarray], places: list[dict]):
        self.embeddings = embeddings
        self.metadata = {i: place for i, place in enumerate(places)}
        # Use Inner Product index for cosine similarity with pre-normalized vectors
        self.index = faiss.IndexFlatIP(embeddings[0].shape[0])
        self.index.add(np.array(embeddings))

    def retrieve(
        self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.3
    ) -> list[dict]:
        similarities, indices = self.index.search(
            query_embedding.reshape(1, -1), k
        )
        results = []
        rank = 1
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx == -1:
                continue
            place = self.metadata[idx]
            if similarity > threshold:
                results.append(
                    {
                        "rank": rank,
                        "name": place["name"],
                        "similarity": round(float(similarity), 4),
                        "lat": place["lat"],
                        "lon": place["lon"],
                        "id": place["id"],
                    }
                )
                rank += 1
        return results
