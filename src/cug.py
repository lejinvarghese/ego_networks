import cugraph
import cudf

# import a built-in dataset
from cugraph.experimental.datasets import netscience

G = netscience.get_graph(fetch=True)
G = G.to_undirected()

df = cugraph.strongly_connected_components(G)
print(df.head(5))