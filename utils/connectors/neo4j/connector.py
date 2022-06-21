import pandas as pd
import py2neo
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_HOST = os.getenv("NEO4J_HOST")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


class Neo4jConnector:
    """
    This connector uses direct Cypher queries though there are helper functions for this library being used.
    """

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

    def empty_database(self):
        self.graph.delete_all()

    def create_nodes(self, nodes=None, entity_type="entity", batch_size=500):
        query = f"""
        UNWIND $node_list as node
        CREATE (e: {entity_type} {{
            id: node.id,
            name: node.name,
            commercial_region_code: node.commercial_region_code
        }})
        """
        print(query)

        for b_start in range(0, len(nodes), batch_size):
            b_end = b_start + batch_size
            records = nodes.iloc[b_start:b_end].to_dict("records")
            tx = self.graph.auto()
            tx.evaluate(query, parameters={"node_list": records})

    def create_relationships(
        self,
        edges=None,
        entity_type="entity",
        relationship_type="connected_to",
        batch_size=500,
    ):
        query = f"""
        UNWIND $edge_list as edge
        MATCH (source: {entity_type} {{id: edge.source}})
        MATCH (target: {entity_type} {{id: edge.target}})

        MERGE (source)-[r: {relationship_type}] -> (target)

        SET r.weight = toInteger(edge.weight)
        """
        print(query)

        for b_start in range(0, len(edges), batch_size):
            b_end = b_start + batch_size
            records = edges.iloc[b_start:b_end].to_dict("records")
            tx = self.graph.auto()
            tx.evaluate(query, parameters={"edge_list": records})


def main():
    nodes = pd.read_csv("data/nodes.csv")
    edges = pd.read_csv("data/edges.csv")
    neo4j_connector = Neo4jConnector(NEO4J_HOST, NEO4J_USER, NEO4J_PASSWORD)
    neo4j_connector.empty_database()
    neo4j_connector.create_nodes(nodes=nodes, entity_type="merchants")
    neo4j_connector.create_relationships(edges=edges, entity_type="merchants")


if __name__ == "__main__":

    main()
