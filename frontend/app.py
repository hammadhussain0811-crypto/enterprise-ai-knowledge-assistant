import requests
import streamlit as st

# Configuration

API_URL = "http://127.0.0.1:8000/api/chat"

# Page configuration

st.set_page_config(
    page_title="Enterprise AI Knowledge Assistant",
    page_icon="🤖",
    layout="centered"
)

# UI

st.title("Enterprise AI Knowledge Assistant")

st.write(
    "Ask questions about the company documents."
)

question = st.text_input(
    "Enter your question:"
)

# Ask question

if st.button("Ask"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching documents and generating answer..."):

            try:

                response = requests.post(
                    API_URL,
                    json={
                        "question": question
                    },
                    timeout=60
                )

                response.raise_for_status()

                data = response.json()

                # Answer
                st.subheader("Answer")

                st.write(data["answer"])

                # Sources
                st.subheader("Sources")

                for source in data["sources"]:

                    st.write(
                        f"📄 {source['source']} — "
                        f"Page {source['page']}"
                    )

            except requests.exceptions.RequestException as e:

                st.error(
                    f"Could not connect to the backend: {e}"
                )