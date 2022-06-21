import pandas as pd
import py2neo
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_HOST = os.getenv("NEO4J_HOST")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


class Neo4jConnector:
    def __init__(self, profile, user, password):
        self.graph = py2neo.Graph(profile, auth=(user, password))
        self.test_connection()

    def test_connection(self):
        try:
            print(
                "Test query to square integers range(1, 3): ",
                self.graph.run(
                    "UNWIND range(1, 3) AS n RETURN n, n * n as n_sq"
                ).data(),
            )
            print("Connection to Neo4j successful")
        except:
            print("Connection to Neo4j unsuccessful")


if __name__ == "__main__":
    nodes = pd.read_csv("data/nodes.csv")
    edges = pd.read_csv("data/edges.csv")

    neo4j_connector = Neo4jConnector(
        profile=NEO4J_HOST, user=NEO4J_USER, password=NEO4J_PASSWORD
    )
