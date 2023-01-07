import streamlit as st

try:
    from src.controller import Controller
    from utils.api.twitter import get_twitter_profile_image
except ModuleNotFoundError:
    from ego_networks.src.controller import Controller
    from ego_networks.utils.api.twitter import get_twitter_profile_image


@st.cache(
    allow_output_mutation=True,
    persist=True,
    hash_funcs={Controller: id},
    show_spinner=False,
)
def cache_controller():
    return Controller()


def render_header():
    st.set_page_config(
        page_title="🦝",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        "# Current users in your network",
    )
    with open(".streamlit/style.css") as css:
        streamlit_style = f"""
                <style>
                {css.read()}
                MainMenu {{visibility: hidden;}}
                footer {{visibility: hidden;}}
                </style>
                """
        st.markdown(streamlit_style, unsafe_allow_html=True)


engine = cache_controller()
if "network" not in st.session_state:
    st.session_state["network"] = engine.network

try:
    network = st.session_state.get("network")
except:
    st.info("No network loaded yet", icon="🤖")


def render_metrics():

    with st.container():
        cols = st.columns(3, gap="small")
        cols[0].metric(label="Nodes", value=f"{network.G.number_of_nodes()}")
        cols[1].metric(label="Edges", value=f"{network.G.number_of_edges()}")
        cols[2].metric(
            label="Users",
            value=f"{len(network.get_ego_user_attributes())}",
        )


def render_charts(n_users=20):
    import pandas as pd
    import ast
    import plotly.express as px

    metrics = list(
        network.get_ego_user_attributes(attribute="public_metrics").values()
    )[-n_users:]
    df = pd.DataFrame(metrics, columns=["public_metrics"])
    df = pd.DataFrame(df["public_metrics"].apply(ast.literal_eval).tolist())
    df["usernames"] = list(network.get_ego_user_attributes().values())[-n_users:]

    fig = px.scatter(
        df,
        x="following_count",
        y="followers_count",
        size="tweet_count",
        color="usernames",
        hover_name="usernames",
        log_x=True,
        log_y=True,
        size_max=50,
        width=800,
        height=800,
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title="Following",
        yaxis_title="Followers",
    )
    fig.add_vline(x=df.following_count.median(), line_width=0.25)
    fig.add_hline(y=df.followers_count.median(), line_width=0.35)
    st.plotly_chart(
        fig,
    )


def render_users(n_users=20):
    usernames = list(network.get_ego_user_attributes().values())[-n_users:]
    profile_images = list(
        network.get_ego_user_attributes(attribute="profile_image_url").values()
    )[-n_users:]
    n_cols = 4
    cols = st.columns(n_cols)

    for idx, user in enumerate(zip(usernames, profile_images)):
        user_name, profile_image = user[0], user[1]
        try:
            profile_image = get_twitter_profile_image(user_name, profile_image)
            col_idx = idx % n_cols
            with cols[col_idx]:
                st.markdown(
                    f"[![image]({profile_image})](http://twitter.com/{user_name})"
                )
                st.write(f"{idx+1}: **{user_name}**")
        except:
            pass


def main():
    render_header()
    st.markdown("## Metrics")
    render_metrics()
    st.markdown("## Charts")
    render_charts(n_users=1000)
    st.markdown("## Users")
    render_users()


if __name__ == "__main__":
    main()