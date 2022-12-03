import streamlit as st

try:
    from src.main import main as engine
    from utils.default import file_exists
except ModuleNotFoundError:
    from ego_networks.src.main import main as engine
    from ego_networks.utils.default import file_exists

DEFAULT_PROFILE_URL = "https://cpraonline.files.wordpress.com/2014/07/new-twitter-logo-vector-200x200.png"
IMAGE_SIZE = 200
N_ROWS = 5


def main():
    st.title("Recommendations from your Ego Network")
    """
    Currently generates recommendations from the Twitter Ego Network.
    """
    with open("style.css") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

    st.image(
        image="https://assets.stickpng.com/images/580b57fcd9996e24bc43c53e.png",
        width=50,
    )

    with st.sidebar:
        st.title("Settings")
        st.write(
            """
            This app helps you to generate your own Twitter recommendations.
            """
        )

        n_recommendations = st.slider(
            label="Select number of recommendations",
            min_value=5,
            max_value=50,
            value=None,
            step=5,
        )

        update_nb_button = st.selectbox(
            label="Update Neighborhood", options=[False, True]
        )
        update_ms_button = st.selectbox(
            label="Update Measures",
            options=[False, True],
        )
        run_button = st.button("Run")

    if run_button:
        with st.spinner("Wait for it..."):
            recommended_profiles, recommended_profile_images = engine(
                k=n_recommendations,
                update_neighborhood=update_nb_button,
                update_measures=update_ms_button,
            )

        st.title("**Recommendations**")
        n_cols = len(recommended_profiles) // N_ROWS
        cols = st.columns(n_cols)

        for idx, rec in enumerate(
            zip(recommended_profiles, recommended_profile_images)
        ):
            user, img = rec[0], rec[1].replace(
                "_normal", f"_{IMAGE_SIZE}x{IMAGE_SIZE}"
            )
            if not (file_exists(img)):
                img = DEFAULT_PROFILE_URL
            col_idx = idx % n_cols
            with cols[col_idx]:
                st.write(f"**{idx+1}: {user}**")
                st.markdown(f"[![image]({img})](http://twitter.com/{user})")


if __name__ == "__main__":
    main()
