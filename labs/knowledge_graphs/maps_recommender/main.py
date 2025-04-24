import click

from components.encoders import DocumentEncoder, QueryEncoder
from components.index import Retriever
from utils import logger

places = [
    {
        "id": "place_123",
        "name": "BarChef",
        "description": "Molecular cocktails, moody lighting, great date spot",
        "tags": ["bar", "cocktails", "romantic", "experimental"],
        "lat": 43.6478,
        "lon": -79.3957,
        "category": "bar",
    },
    {
        "id": "place_124",
        "name": "La Palette",
        "description": "French bistro with a modern twist",
        "tags": ["french", "bistro", "modern", "romantic"],
        "lat": 43.6478,
        "lon": -79.3957,
        "category": "restaurant",
    },
    {
        "id": "place_125",
        "name": "The Art of Coffee",
        "description": "Artisanal coffee and pastries",
        "tags": ["coffee", "pastries", "artisanal"],
    },
    {
        "id": "place_126",
        "name": "Gourmet Pizza",
        "description": "Italian pizza with a gourmet twist",
        "tags": ["pizza", "gourmet", "italian"],
        "lat": 43.6478,
        "lon": -79.3957,
        "category": "restaurant",
    }
]


@click.command()
@click.option("--query", type=str, default="coffee", help="Query to search for")
def main(query: str):
    encoder = DocumentEncoder()
    embeddings = encoder.encode(places)
    logger.info(f"Number of documents: {len(embeddings)}")

    retriever = Retriever(embeddings, [p["name"] for p in places])
    query_embedding = QueryEncoder().encode(query)
    results = retriever.retrieve(query_embedding)
    logger.success(f"Results: {results}")


if __name__ == "__main__":
    main()
