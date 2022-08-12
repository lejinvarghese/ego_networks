import streamlit as st

try:
    from src.main import main as engine
except ModuleNotFoundError:
    from ego_networks.src.main import main as engine


def main():
    st.title("Recommendations from your Ego Network")
    """
    Currently generates recommendations from the Twitter Ego Network.
    """

    st.image(
        image="https://assets.stickpng.com/images/580b57fcd9996e24bc43c53e.png",
        width=50,
    )

    with st.sidebar:
        st.title("Note")
        st.write(
            """
            This demo is a work in progress. It is not yet ready for production use.
            """
        )
        st.write(
            """
            This app helps you to generate your own Twitter recommendations.
            """
        )

        k = st.slider(
            label="Select number of recommendations",
            min_value=5,
            max_value=50,
            value=None,
            step=5,
        )

        run_button = st.button("Run")

    if run_button:
        with st.spinner("Wait for it..."):
            recommendations = engine(k=k)

        st.title("**Recommendations**")
        for idx, rec in enumerate(recommendations):
            st.write(f"{idx+1}: {rec}")


if __name__ == "__main__":
    main()
