import streamlit as st
import requests
import os
import time
import json
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="Contextual Retrieval",
    page_icon="🦙",
    layout="wide",
)

with open("streamlit_style.css") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

BASE_PATH = os.getenv("BASE_PATH", "")
st.title(f"Chatbot Interface for Drive: {BASE_PATH}")
st.markdown("Implemented in Llama-index 🦙")
st.markdown("Link to the Anthropic [blog post](https://www.anthropic.com/news/contextual-retrieval)")

# Streaming response from API call
# Updated to handle JSON payload returned from the `/rag-chat` endpoint.
# The assistant's answer is streamed token by token while the document
# snippets are stored in ``st.session_state`` for later rendering.
def response_generator(query):
    url = os.getenv("API_URL")
    data = {"query": query}
    with requests.Session() as s:
        resp = s.post(url, json=data)
        resp.raise_for_status()
        payload = resp.json()

    # Save document sources so they can be displayed after streaming
    st.session_state["sources"] = payload.get("sources", [])

    answer = payload.get("answer", "")
    for word in answer.split():
        yield word + " "

# Function for first None request
def fake_data():
    _LOREM_IPSUM = "Hi !!! I am your personal recipe assistant. How can i help you ?"
    for word in _LOREM_IPSUM.split(" "):
        yield word + " "
        time.sleep(0.05)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Saving messages state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if prompt := st.chat_input("Type your questions here !!!"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

# Input passed to API
with st.chat_message("assistant"):
    if prompt is not None:
        response = st.write_stream(response_generator(str(prompt)))
    else:
        response = st.write_stream(fake_data)

    # Display document snippets associated with the response
    if st.session_state.get("sources"):
        st.sidebar.header("Sources")
        for src in st.session_state["sources"]:
            doc_name = os.path.basename(src.get("file", "Document"))
            snippet = src.get("text", "")
            st.sidebar.markdown(f"**{doc_name}**")
            st.sidebar.write(snippet)

st.session_state.messages.append({"role": "assistant", "content": response})
