import streamlit as st


def render_header():
    st.set_page_config(
        page_title="🦝 users",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        "# Explore users from your Ego Network",
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


render_header()