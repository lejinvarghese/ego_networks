import functools
import time


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"elapsed time: {elapsed_time:0.2f} seconds")
        return value

    return wrapper_timer


@timer
def read_sql(file):
    return str(open(file, "r").read())


@timer
def execute_query(query: str, project: str, return_results: bool = False):
    from google.cloud.bigquery import Client

    client = Client(project=project)
    query_job = client.query(query)
    result = query_job.result()
    if return_results:
        return result.to_dataframe()


@timer
def create_graph_viz(G, color_map=False, size_map=False, title="Graph", directed=False):
    import networkx as nx
    from plotly import colors
    import plotly.graph_objects as go

    pos = nx.spring_layout(G, seed=69)
    palette = colors.qualitative.Plotly
    edge_x = []
    edge_y = []
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    node_sizes = []

    for edge in G.edges():
        x0, y0 = pos.get(edge[0])
        x1, y1 = pos.get(edge[1])
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.2, color="gray"),  # color="gray"
        hoverinfo="text",  # none
        mode="lines",
    )

    edge_trace.text = "text"

    for node in G.nodes():

        x, y = pos.get(node)
        node_x.append(x)
        node_y.append(y)

        node_text.append(node)
        if color_map:
            node_colors.append(
                palette[int(str(abs(hash(color_map.get(node))))[:5]) % 10]
            )
        if size_map:
            node_sizes.append(100 * size_map.get(node))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
    )
    if color_map:
        node_trace.marker.color = node_colors
    if size_map:
        node_trace.marker.size = node_sizes
    else:
        node_trace.marker.size = 10
    node_trace.text = node_text

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            width=600,
            height=600,
            title=title,
            font_size=7,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)",
            hoverlabel=dict(font_size=16, font_family="Open Sans"),
        ),
    )
    if directed:
        for edge in G.edges():
            x0, y0 = pos.get(edge[0])
            x1, y1 = pos.get(edge[1])
            fig.add_annotation(
                x=x0,  # arrows' head
                y=y0,  # arrows' head
                ax=x1,  # arrows' tail
                ay=y1,  # arrows' tail
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                text="",
                showarrow=True,
                arrowhead=1,
                arrowsize=3,
                arrowwidth=0.3,
                arrowcolor="grey",
            )
    return fig


@timer
def get_elasticity(project):
    """
    Gets the price elasticity for an item with a level-level linear mixed model accounting for all groups.
    #https://www.statworx.com/ch/blog/food-for-regression-using-sales-data-to-identify-price-elasticity/
    Usage:
    get_elasticity(project)
    """

    def get_data(project):
        import pandas as pd

        return pd.DataFrame({})

    def train(df, target_var="", group_var="", cols_features=""):
        import statsmodels.api as sm

        df["Intercept"] = 1
        endog = df[target_var].astype(float)  # .apply(lambda x: np.log(1+x))
        exog = df[["Intercept"] + cols_features].astype(
            float
        )  # .apply(lambda x: np.log(1+x))
        exog_re = exog.copy()[cols_features]
        groups = df.reset_index()[group_var]

        model = sm.MixedLM(endog, exog, groups, exog_re=exog_re)
        model = model.fit_regularized(method="l1")
        _coeff = round(model.params[1], 2)
        _pval = round(model.pvalues[1], 2)
        #         _elas = round(_coeff, 2) #for a log-log model
        _elas = round(
            _coeff * df.median()[1] / df.median()[0], 2
        )  # for a level-level model
        if _elas < -1.0:
            _elas_seg = "elastic"
        elif _elas > -1.0:
            _elas_seg = "inelastic"
        else:
            _elas_seg = "unit elastic"
        return model.converged, _coeff, _pval, _elas, _elas_seg

    df = get_data(project)
    return train(df, target_var="qty", group_var="group", cols_features=["price"])
