from py2neo import Graph


class Neo4jConnector:
    """
    This connector uses direct Cypher queries though there are helper functions for this library being used.
    """

    def __init__(self, profile, user, password):
        self.__name__ = "Neo4j"
        self.graph = Graph(profile, auth=(user, password))
        self.test_connection()

    def test_connection(self):
        try:
            print(
                "Test query to square integers range(1, 3): ",
                self.graph.run(
                    """
                    UNWIND range(1, 3) AS n
                    RETURN n, n * n AS n_sq
                    """
                ).data(),
            )
            print(f"Connection to {self.__name__} successful")
        except:
            print(f"Connection to {self.__name__} unsuccessful")

    def empty_database(self):
        self.graph.delete_all()

    def create_nodes(self, nodes=None, entity_type="entity", batch_size=1000):

        node_attrs_dict = {
            attr: "node." + attr for attr in nodes.columns.values
        }
        node_attrs_str = ",\n            ".join(
            [
                "%s: %s" % (key, value)
                for (key, value) in node_attrs_dict.items()
            ]
        )

        query = f"""
        UNWIND $node_list as node
        CREATE (e: {entity_type} {{
            {node_attrs_str}
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
        relationship_type="is_connected",
        batch_size=1000,
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
