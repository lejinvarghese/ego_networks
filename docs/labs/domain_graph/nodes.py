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


def get_content():
    def get_books(year):
        if year == 2022:
            node_features, _ = GoodreadsEgoNeighborhood(
                focal_node=GOODREADS_FOCAL_NODE_ID, max_radius=1
            ).update_neighborhood()
            return node_features[node_features.date.dt.year == year][
                "desc"
            ].tolist()

    def get_courses(year):
        if year == 2022:
            return [
                "Complete guide to Elasticsearch",
                "Smart cities management of smart urban infrastructures",
            ]

    def get_podcasts(year):
        if year == 2022:
            return [
                "Aviv Bergman on The Evolution of Robustness and Integrating The Disciplines",
                "Grimes: Music, AI, and the Future of Humanity",
                "Liv Boeree: Poker, Game Theory, AI, Simulation, Aliens & Existential Risk",
                "Ariel Ekblaw: Space Colonization and Self-Assembling Space Megastructures",
                "Mark Zuckerberg: Facebook, Metaverse, and Social Media Ethics",
                "Michael Levin: Biology, Life, Aliens, Evolution, Embryogenesis & Xenobots",
                "Richard Haier: IQ Tests, Human Intelligence, and Group Differences",
                "The Science of Creativity & How to Enhance Creative Innovation",
            ]

    def get_projects(year):
        if year == 2022:
            return [
                "Ego networks: Creating complex networks, including Twitter to study information diffusion",
                "Art: Generative AI, Latent Diffusion, and the Future of Art",
                "Spotify: Music genres and playlists",
                "Reinforcement learning: Dynamic programming, Policy and Value Iteration",
            ]

    return [
        *get_books(year=2022),
        *get_courses(year=2022),
        *get_podcasts(year=2022),
        *get_projects(year=2022),
    ]


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
