# ---
# lambda-test: false
# ---
# ## Demo Streamlit application.
#
# This application is the example from https://docs.streamlit.io/library/get-started/create-an-app.
#
# Streamlit is designed to run its apps as Python scripts, not functions, so we separate the Streamlit
# code into this module, away from the Modal application code.
import streamlit as st
from icml_finder.vectordb import LanceSchema
from lancedb.rerankers import CohereReranker
from icml_finder.vectordb import make_vectordb
import datetime as dt
from dateutil.relativedelta import relativedelta # to add days or years
from collections import defaultdict
from typing import List 

@st.cache_resource()
def get_reranker():
    return CohereReranker(column="abstract")

st.cache_resource()
def get_vectordb():
    return make_vectordb("/icml_data/icml_sessions.jsonl", "/root/data/vectordb")

table = get_vectordb()
reranker = get_reranker()

def group_and_sort_events(events: List[LanceSchema]) -> dict:
    # Group events by the date of their session
    grouped_events = defaultdict(list)
    for event in events:
        if event.payload.time_vienna:
            date_key = event.payload.time_vienna.date()
            grouped_events[date_key].append(event)

    # Sort events within each date group by the time of their session
    for date_key in grouped_events:
        grouped_events[date_key].sort(key=lambda x: x.payload.time_vienna)

    return grouped_events

def make_item(obj: LanceSchema):
    with st.expander(obj.payload.name):
        st.markdown(
            f"""
                ## {obj.payload.name}
                ### {obj.payload.location}
                ### {obj.payload.time_vienna}
                ### Authors: 
                {obj.payload.speakers_authors}
                ### Abstract:
                {obj.payload.abstract}
                ### [Link]({obj.payload.virtualsite_url})
            """
            )

st.title("ICML 2024 Session Finder")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

format = 'MMM DD, YYYY' 
start_date = dt.datetime.now().date() #  I need some range in the past
end_date = dt.datetime.now().date()+relativedelta(days=10)
max_days = end_date-start_date

slider = st.slider('Select date', min_value=start_date, value=(start_date, end_date) ,max_value=end_date, format=format)


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

    results = table.search(prompt, query_type="hybrid").limit(100).rerank(normalize='score', reranker=reranker).to_pydantic(LanceSchema)

    grouped_sorted_events = group_and_sort_events(results)

    for date_key, events_on_date in grouped_sorted_events.items():
        st.markdown(f"## Events on {date_key}:")
        for event in events_on_date:
            make_item(event)
        
