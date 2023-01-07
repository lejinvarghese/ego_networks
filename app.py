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
    def rename_pages():
        pages = st.source_util.get_pages("app.py")
        new_page_names = {
            "app": "🏠 home",
            "users": "🦝 users",
        }

        for key, page in pages.items():
            if page.get("page_name") in new_page_names:
                page["page_name"] = new_page_names.get(page.get("page_name"))

    rename_pages()
    st.set_page_config(
        page_title="🏠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("User recommendations")
    st.markdown("Generates user recommendations from your ego network.")
    with open(".streamlit/style.css") as css:
        streamlit_style = f"""
                <style>
                {css.read()}
                MainMenu {{visibility: hidden;}}
                footer {{visibility: hidden;}}
                </style>
                """
        st.markdown(streamlit_style, unsafe_allow_html=True)

    st.image(
        image="https://www.freepnglogos.com/uploads/twitter-logo-png/twitter-logo-vector-png-clipart-1.png",
        width=50,
    )
    st.markdown("## Recommendations")

render_header()
engine = cache_controller()
if "network" not in st.session_state:
    st.session_state["network"] = engine.network

def render_sidebar():
    with st.sidebar:
        st.title("Settings")
        st.write(
            """
            This app helps you generate your own Twitter user recommendations.
            """
        )

        n_recommendations = st.slider(
            label="Select number of recommendations",
            min_value=5,
            max_value=50,
            value=None,
            step=5,
        )

        update_neighborhood = st.selectbox(
            label="Update Neighborhood", options=[False, True]
        )
        update_measures = st.selectbox(
            label="Update Measures",
            options=[False, True],
        )
        recommendation_strategy = st.selectbox(
            label="Recommendation Strategy",
            options=["Diverse", "Connectors", "Influencers"],
        )
        run = st.button("Run")
        return (
            update_neighborhood,
            update_measures,
            recommendation_strategy.lower(),
            n_recommendations,
            run,
        )


def render_recommendations(
    update_neighborhood,
    update_measures,
    recommendation_strategy,
    n_recommendations,
):
    with st.spinner("Please wait"):
        if update_neighborhood:
            engine.update_neighborhood()
        if update_measures:
            engine.update_measures()
        (
            recommended_profile_names,
            recommended_profile_images,
        ) = engine.update_recommendations(
            recommendation_strategy=recommendation_strategy,
            n_recommendations=n_recommendations,
        )

    n_rows = 5
    n_cols = len(recommended_profile_names) // n_rows
    cols = st.columns(n_cols)

    for idx, rec in enumerate(
        zip(recommended_profile_names, recommended_profile_images)
    ):
        user_name, profile_image = rec[0], rec[1]
        profile_image = get_twitter_profile_image(user_name, profile_image)
        col_idx = idx % n_cols
        with cols[col_idx]:
            st.markdown(
                f"[![image]({profile_image})](http://twitter.com/{user_name})"
            )
            st.write(f"{idx+1}: **{user_name}**")


def main():


    (
        update_neighborhood,
        update_measures,
        recommendation_strategy,
        n_recommendations,
        run,
    ) = render_sidebar()

    if run:
        render_recommendations(
            update_neighborhood,
            update_measures,
            recommendation_strategy,
            n_recommendations,
        )


if __name__ == "__main__":
    main()
