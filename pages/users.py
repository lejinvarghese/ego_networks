import streamlit as st
from ast import literal_eval
from pandas import DataFrame
from plotly.express import scatter
from plotly.graph_objects import Figure, Pie

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
    st.markdown("## Metrics")

    with st.container():
        cols = st.columns(3, gap="small")
        cols[0].metric(label="Nodes", value=f"{network.G.number_of_nodes()}")
        cols[1].metric(label="Edges", value=f"{network.G.number_of_edges()}")
        cols[2].metric(
            label="Users",
            value=f"{len(network.get_ego_user_attributes())}",
        )


def render_users(n_users=20):
    def form_callback(ratings=None):
        if ratings:
            st.session_state["ratings"] = ratings
        return st.session_state.get("ratings")

    def plot_user_metrics(n_users=1000):

        user_metrics = list(
            network.get_ego_user_attributes(attribute="public_metrics").values()
        )[-n_users:]
        df = DataFrame(user_metrics, columns=["public_metrics"])
        df = DataFrame(df["public_metrics"].apply(literal_eval).tolist())
        df["usernames"] = list(network.get_ego_user_attributes().values())[
            -n_users:
        ]

        fig = scatter(
            df,
            x="following_count",
            y="followers_count",
            size="tweet_count",
            color="usernames",
            hover_name="usernames",
            log_x=True,
            log_y=True,
            size_max=50,
            width=600,
            height=600,
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title="Following",
            yaxis_title="Followers",
        )
        fig.add_vline(x=df.following_count.median(), line_width=0.25)
        fig.add_hline(y=df.followers_count.median(), line_width=0.25)
        st.plotly_chart(fig, use_container_width=True)

    def plot_ratings_distribution():

        try:
            df = DataFrame(form_callback(), columns=["ratings"])
            df = df.groupby("ratings").value_counts().reset_index(name="counts")
            fig = Figure()
            fig.add_trace(
                Pie(
                    labels=df.ratings.to_list(),
                    values=df.counts.to_list(),
                    pull=[0.2, 0.1, 0],
                    hole=0.7,
                ),
            )
            fig.update_layout(
                showlegend=False,
                width=600,
                height=600,
            )
            fig.update_traces(
                textposition="inside",
                textinfo="label+percent",
                marker=dict(
                    colors=[
                        "mediumaquamarine",
                        "lightcoral",
                        "lightgrey",
                    ],
                    line=dict(color="white", width=0.5),
                ),
            )
            st.plotly_chart(fig, use_container_width=True)
        except:
            pass

    st.markdown("## Charts")
    st.markdown("#### Popularity")
    plot_user_metrics()
    st.markdown("#### Rating")
    plot_ratings_distribution()

    st.markdown("## Users")
    usernames = list(network.get_ego_user_attributes().values())[-n_users:]
    profile_images = list(
        network.get_ego_user_attributes(attribute="profile_image_url").values()
    )[-n_users:]

    with st.form(key="form_user_ratings"):
        n_cols = 4
        cols = st.columns(n_cols)
        ratings = []
        for idx, user in enumerate(zip(usernames, profile_images)):
            user_name, profile_image = user[0], user[1]
            try:
                profile_image = get_twitter_profile_image(
                    user_name, profile_image
                )
                col_idx = idx % n_cols
                with cols[col_idx]:
                    st.markdown(
                        f"[![image]({profile_image})](http://twitter.com/{user_name})"
                    )
                    st.write(f"{idx+1}: **{user_name}**")
                    ratings.append(
                        st.selectbox(
                            label=f"user rating",
                            options=[
                                "👾",
                                "🔬",
                                "❤️",
                            ],
                            key=idx,
                            index=1,
                            label_visibility="hidden",
                        )
                    )
            except Exception:
                continue
        st.form_submit_button("Submit", on_click=form_callback, args=(ratings,))
        return ratings


def main():
    render_header()
    render_metrics()
    ratings = render_users(n_users=20)


if __name__ == "__main__":
    main()