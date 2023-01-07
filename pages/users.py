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


engine = cache_controller()
if "network" not in st.session_state:
    st.session_state["network"] = engine.network


def render_header():
    st.set_page_config(
        page_title="🦝",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        "# Explore users from your network",
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


def render_summary():
    st.markdown("## Summary")
    try:
        network = st.session_state.get("network")
    except:
        st.info("No network loaded yet", icon="🤖")

    usernames = list(network.get_ego_user_attributes().values())[:20]
    profile_images = list(
        network.get_ego_user_attributes(attribute="profile_image_url").values()
    )[:20]

    with st.container():
        cols = st.columns(4, gap="small")
        cols[0].metric(label="Nodes", value=f"{network.G.number_of_nodes()}")
        cols[1].metric(label="Edges", value=f"{network.G.number_of_edges()}")
        cols[2].metric(
            label="Radius",
            value=f"{network.radius}",
        )
        cols[3].metric(
            label="Following",
            value=f"{len(usernames)}",
        )
        st.markdown("## Users")

        n_cols = 4
        cols = st.columns(n_cols)

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
            except:
                pass


render_header()
render_summary()