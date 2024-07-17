# ---
# lambda-test: false
# ---
# ## Demo Streamlit application.
#
# This application is the example from https://docs.streamlit.io/library/get-started/create-an-app.
#
# Streamlit is designed to run its apps as Python scripts, not functions, so we separate the Streamlit
# code into this module, away from the Modal application code.
import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from icml_finder.vectordb import LanceSchema
from lancedb.rerankers import CohereReranker

@st.cache_resource()
def get_reranker():
    return CohereReranker(column="abstract")

st.cache_resource()
def get_vectordb():
    from icml_finder.vectordb import make_vectordb
    hf_hub_download(repo_id="porestar/icml2024_embeddings", filename="icml_sessions.jsonl", local_dir="/root/data", repo_type="dataset")
    return make_vectordb("/root/data/icml_sessions.jsonl", "/root/data/vectordb")

table = get_vectordb()
reranker = get_reranker()


st.title("ICML 2024 Session Finder")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    results = table.search(prompt, query_type="hybrid").limit(10).rerank(normalize='score', reranker=reranker).to_pydantic(LanceSchema)
    print(results)
    for result in results:
        st.code(result.payload.dict())
