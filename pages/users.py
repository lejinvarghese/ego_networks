import streamlit as st
from ast import literal_eval
from pandas import DataFrame, Timestamp
from plotly.express import scatter
from plotly.graph_objects import Figure, Pie

try:
    from src.controller import Controller
    from utils.io import DataWriter
    from utils.api.twitter import get_twitter_profile_image
except ModuleNotFoundError:
    from ego_networks.src.controller import Controller
    from ego_networks.utils.io import DataWriter
    from ego_networks.utils.api.twitter import get_twitter_profile_image


@st.cache(
    allow_output_mutation=True,
    persist=True,
    hash_funcs={Controller: id},
    show_spinner=False,
)
def cache_controller():
    return Controller()


engine = cache_controller()
if "network" not in st.session_state:
    st.session_state["network"] = engine.network

try:
    network = st.session_state.get("network")
except:
    st.info("No network loaded yet", icon="🤖")


def form_callback(ratings=None):
    if ratings:
        st.session_state["ratings"] = ratings
    return st.session_state.get("ratings")


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


def render_summary_metrics():
    st.markdown("## Metrics")

    with st.container():
        cols = st.columns(3, gap="small")
        cols[0].metric(label="Nodes", value=f"{network.G.number_of_nodes()}")
        cols[1].metric(label="Edges", value=f"{network.G.number_of_edges()}")
        cols[2].metric(
            label="Users",
            value=f"{len(network.get_ego_user_attributes())}",
        )


def plot_degree_metrics(n_users=1000):
    st.markdown("## Charts")
    st.markdown("#### In Degree vs Out Degree")

    user_metrics = list(
        network.get_ego_user_attributes(attribute="public_metrics").values()
    )[-n_users:]
    df = DataFrame(user_metrics, columns=["public_metrics"])
    df = DataFrame(df["public_metrics"].apply(literal_eval).tolist())
    df["username"] = list(network.get_ego_user_attributes().values())[-n_users:]

    fig = scatter(
        df,
        x="following_count",
        y="followers_count",
        size="tweet_count",
        color="username",
        hover_name="username",
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


def render_user_ratings(n_users=1000):
    def collect_user_ratings():
        st.markdown("## Users")
        usernames = list(network.get_ego_user_attributes().values())[-n_users:]
        profile_images = list(
            network.get_ego_user_attributes(
                attribute="profile_image_url"
            ).values()
        )[-n_users:]
        selected_users = st.multiselect(
            label="Who would you like to rate?",
            options=["all"] + usernames,
            default=["jack", "elonmusk"],
            help="select 'all' to rate all users, and please unselect 'all' to select individual users",
        )
        if "all" in selected_users:
            s_usernames, s_profile_images = usernames, profile_images
        else:
            s_usernames, s_profile_images = [], []
            for i, v in enumerate(usernames):
                if v in selected_users:
                    s_usernames.append(v)
                    s_profile_images.append(profile_images[i])

        with st.expander("Expand to rate users"):
            with st.form(key="form_user_ratings"):
                n_cols = 4
                cols = st.columns(n_cols)
                ratings = []
                for idx, val in enumerate(zip(s_usernames, s_profile_images)):
                    user_name, profile_image = val[0], val[1]
                    try:
                        profile_image = get_twitter_profile_image(
                            user_name, profile_image
                        )
                    except Exception:
                        pass
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
                                    "⛅",  # new
                                    "🔬",  # neutral
                                    "👾",  # bad
                                    "❤️",  # good
                                ],
                                key=idx,
                                index=0,
                                label_visibility="hidden",
                            )
                        )
                st.form_submit_button(
                    "Submit", on_click=form_callback, args=(ratings,)
                )
        usernames_dict = network.get_ego_user_attributes().items()
        df_ratings = DataFrame(
            [k for k, v in usernames_dict if v in s_usernames],
            columns=["id"],
        )
        df_ratings["username"] = s_usernames
        df_ratings["rating"] = ratings
        df_ratings["event_timestamp"] = Timestamp.now()
        return df_ratings

    def plot_ratings_distribution():

        df = DataFrame(form_callback(), columns=["ratings"])
        df = df.groupby("ratings").value_counts().reset_index(name="counts")
        fig = Figure()
        fig.add_trace(
            Pie(
                labels=df.ratings.to_list(),
                values=df.counts.to_list(),
                pull=[0, 0.2, 0, 0],
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
                    "lightgrey",
                    "mediumaquamarine",
                    "lightcoral",
                    "wheat",
                ],
                line=dict(color="white", width=0.5),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    def plot_rating_metrics(ratings):
        st.markdown("## Ratings")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("#### Table")
            st.dataframe(
                ratings.set_index("id"),
                use_container_width=True,
                width=600,
                height=600,
            )
        with cols[1]:
            st.markdown("#### Distribution")
            plot_ratings_distribution()

    df_ratings = collect_user_ratings()
    plot_rating_metrics(ratings=df_ratings)
    return df_ratings


def main():
    render_header()
    render_summary_metrics()
    plot_degree_metrics()
    ratings = render_user_ratings()
    DataWriter(data=ratings, data_type="node_ratings").run()


if __name__ == "__main__":
    main()
