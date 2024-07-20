import streamlit as st
st.set_page_config(layout="wide", page_title="ICML Session Finder")

from icml_finder.vectordb import make_vectordb, LanceSchema
from lancedb.rerankers import CohereReranker
from collections import defaultdict
from typing import List 


if "session_events" not in st.session_state:
    st.session_state["session_events"] = {}

if "prompt" not in st.session_state:
    st.session_state["prompt"] = ""

if "selected_sessions" not in st.session_state:
    st.session_state["selected_sessions"] = defaultdict(set)

@st.cache_resource()
def get_reranker():
    return CohereReranker(column="abstract")


st.cache_resource()
def get_vectordb():
    return make_vectordb("/icml_data/icml_sessions.jsonl", "/root/data/vectordb")

def add_to_selected_sessions(date_key, session: LanceSchema):
    st.session_state.selected_sessions[date_key].add(session)
    

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

chat_area, selection_area, summarization_area = st.columns([0.33, 0.33, 0.33])

with chat_area:
    st.header("ICML Session Search")
    if len(st.session_state.prompt) > 0:
        st.markdown(f"You asked: {st.session_state.prompt}")

    for group, (date_key, events_on_date) in enumerate(st.session_state.session_events.items()):
        
        st.markdown(f"## Events on {date_key}:")
        for event_id, event in enumerate(events_on_date):
            
            col1, col2 = st.columns([0.05, 0.95])
            with col1:
                check_val = st.checkbox("", key=f"event_id_{group}_{event_id}", on_change=add_to_selected_sessions, args=[date_key, event])
            
            with col2:
                make_item(event)

    # React to user input
    if prompt := st.chat_input("What is up?"):
        results = table.search(prompt, query_type="hybrid").limit(100).rerank(normalize='score', reranker=reranker).to_pydantic(LanceSchema)
        st.session_state.session_events = group_and_sort_events(results)
        st.rerun()

        
with selection_area:
    for group, (date_key, events_on_date) in enumerate(st.session_state.selected_sessions.items()):
        
        st.markdown(f"## Events on {date_key}:")
        for event_id, event in enumerate(events_on_date):
            
            col1, col2 = st.columns([0.05, 0.95])
            with col1:
                check_val = st.checkbox("", key=f"event_id_{group}_{event_id}")
            
            with col2:
                make_item(event)