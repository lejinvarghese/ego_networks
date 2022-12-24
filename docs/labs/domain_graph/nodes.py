import os
from dotenv import load_dotenv

load_dotenv()

try:
    from src.neighborhoods.goodreads import GoodreadsEgoNeighborhood
except ModuleNotFoundError:
    from ego_networks.src.neighborhoods.goodreads import (
        GoodreadsEgoNeighborhood,
    )
GOODREADS_FOCAL_NODE_ID = os.getenv("GOODREADS_FOCAL_NODE_ID")


def get_content(year=2022):
    node_features, _ = GoodreadsEgoNeighborhood(
        focal_node=GOODREADS_FOCAL_NODE_ID, max_radius=1
    ).update_neighborhood()
    return node_features[node_features.date.dt.year == year]["title"].tolist()


domains = [
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
    "autonomous vehicles",
    "network dynamics",
    "system design",
    "information diffusion",
    "adaptive intelligent systems",
]
content = get_content()