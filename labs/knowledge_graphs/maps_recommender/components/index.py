import faiss
import numpy as np
from geopy.distance import geodesic


def geo_filter(
    anchor_coords: tuple[float, float],
    candidates: list[dict[str, float]],
    max_distance_km: float = 1,
) -> list[dict[str, float]]:
    return [
        p
        for p in candidates
        if geodesic(anchor_coords, (p["lat"], p["lon"])).km <= max_distance_km
    ]


class Retriever:
    def __init__(self, place_vectors: list[np.ndarray], place_ids: list[str]):
        self.index = faiss.IndexFlatIP(len(place_vectors[0]))
        self.index.add(np.array(place_vectors).astype("float32"))
        self.id_map = {i: pid for i, pid in enumerate(place_ids)}

    def retrieve(self, query_vector: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        """Retrieve similar items and their scores."""
        D, I = self.index.search(
            np.array([query_vector]).astype("float32"), top_k
        )
        valid_results = [
            (self.id_map[idx], round(float(score), 2))
            for idx, score in zip(I[0], D[0])
            if idx != -1
        ]
        return sorted(valid_results, key=lambda x: x[1], reverse=True)
