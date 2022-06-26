import os
from warnings import filterwarnings

import pandas as pd
from dotenv import load_dotenv

try:
    from utils.connectors.neo4j import Neo4jConnector
except ModuleNotFoundError:
    from ego_networks.utils.connectors.neo4j import Neo4jConnector

load_dotenv()
filterwarnings("ignore")

NEO4J_HOST = os.getenv("NEO4J_HOST")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

file_path = os.path.dirname(os.path.realpath(__file__))


def main():
    nodes = pd.read_csv(f"{file_path}/data/nodes.csv")
    edges = pd.read_csv(f"{file_path}/data/edges.csv")
    entity_type = "merchants"
    relationship_type = "sold_to"

    neo4j_connector = Neo4jConnector(NEO4J_HOST, NEO4J_USER, NEO4J_PASSWORD)
    neo4j_connector.empty_database()
    neo4j_connector.create_nodes(nodes=nodes, entity_type=entity_type)
    neo4j_connector.create_relationships(
        edges=edges,
        entity_type=entity_type,
        relationship_type=relationship_type,
    )


if __name__ == "__main__":

    main()
