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
def engine():
    return Controller()


def render_header():
    st.title("Recommendations from your Ego Network")
    """
    Generates recommendations from the Twitter Ego Network.
    """
    with open(".streamlit/style.css") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

    st.image(
        image="https://www.freepnglogos.com/uploads/twitter-logo-png/twitter-logo-vector-png-clipart-1.png",
        width=50,
    )


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
            engine().update_neighborhood()
        if update_measures:
            engine().update_measures()
        (
            recommended_profile_names,
            recommended_profile_images,
        ) = engine().update_recommendations(
            recommendation_strategy=recommendation_strategy,
            n_recommendations=n_recommendations,
        )

    st.title("**Recommendations**")
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
            st.write(f"{idx+1}: **{user_name}**")
            st.markdown(
                f"[![image]({profile_image})](http://twitter.com/{user_name})"
            )


def main():
    st.set_page_config(layout="wide")
    render_header()

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
