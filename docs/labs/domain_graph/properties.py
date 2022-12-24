"""Description: Graph properties for the domain graph"""

from community import best_partition
from networkx import Graph, get_edge_attributes, betweenness_centrality

class GraphProperties:
    def __init__(
        self,
        graph: Graph,
        diffusion_grades: dict = None,
        resolution: float = 0.7,
        random_state: int = 34,
    ) -> None:
        self.graph = graph
        self.random_state = random_state
        self.communities = best_partition(
            graph, resolution=resolution, random_state=self.random_state
        )
        self.community_center_colors = {
            "Smart Cities": "mediumseagreen",
            "Intelligence": "gold",
            "Robotics": "cornflowerblue",
            "Machine Learning": "deepskyblue",
            "Evolutionary Theory": "mediumpurple",
            "Network Science": "hotpink",
            "Recommender Systems": "lightcoral",
            "Marketing": "palevioletred",
        }
        self.community_colors = {
            self.communities.get(k): v
            for k, v in self.community_center_colors.items()
        }
        self.diffusion_grades = diffusion_grades

    def __get_node_properties(self) -> tuple:
        node_colors = [
            self.community_colors.get(v, "lightgrey")
            for k, v in self.communities.items()
        ]
        _node_centrality = betweenness_centrality(
            self.graph,
            weight="weight",
            normalized=True,
            endpoints=True,
            seed=self.random_state,
        )
        node_sizes = [5e4 * i for i in list(_node_centrality.values())]
        return node_colors, node_sizes

    def __get_edge_properties(self) -> tuple:
        edge_colors = []
        for e in self.graph.edges():
            u, v = e
            if self.communities.get(u) == self.communities.get(v):
                edge_colors.append(
                    self.community_colors.get(
                        self.communities.get(u), "lightgrey"
                    )
                )
            else:
                edge_colors.append("lightgrey")

        edge_sizes = [
            w * 1e1 for w in get_edge_attributes(self.graph, "weight").values()
        ]
        return edge_colors, edge_sizes

    def __get_line_properties(self):
        line_widths, line_colors = [], []
        for n in self.graph.nodes():
            if n in self.diffusion_grades.keys():
                line_widths.append(self.diffusion_grades.get(n))
                line_colors.append("orange")
            else:
                line_widths.append(0.5)
                line_colors.append(
                    self.community_colors.get(
                        self.communities.get(n), "lightgrey"
                    )
                )
        return line_widths, line_colors

    def create(self) -> tuple:
        node_colors, node_sizes = self.__get_node_properties()
        edge_colors, edge_sizes = self.__get_edge_properties()
        line_widths, line_colors = self.__get_line_properties()

        return (
            node_colors,
            node_sizes,
            edge_colors,
            edge_sizes,
            line_widths,
            line_colors,
        )
