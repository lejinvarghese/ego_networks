# -*- coding: utf-8 -*-
import os
import sys

ROOT_DIRECTORY = os.path.abspath(".")
sys.path.insert(1, ROOT_DIRECTORY)

from warnings import filterwarnings

try:
    from utils.graph import draw_nx_graph
    from docs.labs.domain_graph.properties import (
        GraphProperties,
    )
    from docs.labs.domain_graph.nodes import domains, content
    from docs.labs.domain_graph.generator import DomainGraph
except ModuleNotFoundError:
    from ego_networks.utils.graph import draw_nx_graph
    from ego_networks.docs.labs.domain_graph.properties import GraphProperties
    from ego_networks.docs.labs.domain_graph.nodes import domains, content
    from ego_networks.docs.labs.domain_graph.generator import DomainGraph

filterwarnings("ignore")
os.environ["KMP_WARNINGS"] = "FALSE"
FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def main():

    d_g = DomainGraph(nodes=domains)
    G = d_g.create_graph()
    node_grades = d_g.get_diffusion_grades(content)

    (
        node_colors,
        node_sizes,
        edge_colors,
        edge_sizes,
        line_widths,
        line_colors,
    ) = GraphProperties(graph=G, diffusion_grades=node_grades).create()

    draw_nx_graph(
        G,
        node_color=node_colors,
        node_size=node_sizes,
        line_widths=line_widths,
        line_colors=line_colors,
        font_size=14,
        node_label_font_color="black",
        alpha=0.8,
        edge_colors=edge_colors,
        dpi=240,
        figsize=(40, 40),
        width=edge_sizes,
        save=True,
        file_path=f"{FILE_DIRECTORY}/figure.png",
        random_state=62,
    )


if __name__ == "__main__":
    main()
