import click
from components.encoders import DocumentEncoder, QueryEncoder
from components.index import Retriever
from components.planner import greedy_path

from utils import logger

places = [
    {
        "id": "place_123",
        "name": "BarChef",
        "description": "Molecular cocktails, moody lighting, great date spot",
        "tags": ["bar", "cocktails", "romantic", "experimental"],
        "lat": 43.6474749,
        "lon": -79.3795693,
        "category": "bar",
    },
    {
        "id": "place_124",
        "name": "La Palette",
        "description": "French bistro with a modern twist",
        "tags": ["french", "bistro", "modern", "romantic"],
        "lat": 43.6432000,
        "lon": -79.3969600,
        "category": "restaurant",
    },
    {
        "id": "place_125",
        "name": "The Art of Coffee",
        "description": "Artisanal coffee and pastries",
        "tags": ["coffee", "pastries", "artisanal"],
        "lat": 43.6534490,
        "lon": -79.3622397,
        "category": "coffee",
    },
    {
        "id": "place_126",
        "name": "Gourmet Pizza",
        "description": "Italian pizza with a gourmet twist",
        "tags": ["pizza", "gourmet", "italian"],
        "lat": 43.6478,
        "lon": -79.37956,
        "category": "restaurant",
    },
]


@click.command()
@click.option("--query", type=str, default="coffee", help="Query to search for")
@click.option("--distance", type=float, default=2, help="Distance in km")
def main(query: str, distance: float):
    encoder = DocumentEncoder()
    embeddings = encoder.encode(places)
    logger.info(f"Number of documents: {len(embeddings)}")

    retriever = Retriever(embeddings, places)
    query_embedding = QueryEncoder().encode(query)
    results = retriever.retrieve(query_embedding)
    logger.success(f"Results: {results}")

    start = results[0]
    path = greedy_path(start, results[1:], max_km=distance)
    logger.highlight(f"Path: {path['formatted_path']}")
    logger.highlight(f"Total distance: {path['total_distance']} km")


if __name__ == "__main__":
    main()
