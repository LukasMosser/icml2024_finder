import streamlit as st
st.set_page_config(layout="wide", page_title="ICML Session Finder")

# Container height trick from https://discuss.streamlit.io/t/is-there-a-way-to-set-the-height-of-a-container-or-a-column-to-the-monitor-height-in-px/52504/2
st.markdown('''
            <style>
            .fullHeight {height : 50vh;
                width : 100%}
            </style>''', unsafe_allow_html = True)

from icml_finder.vectordb import make_vectordb, LanceSchema
from lancedb.rerankers import CohereReranker
from collections import defaultdict
from typing import List, Dict
from openai import OpenAI


client = OpenAI()

def rag_prompt(messages: List[Dict[str, str]], query: str, selected_sessions: List[LanceSchema]) -> str:
    PROMPT = """
    The user has selected following <Sessions> from the Internation Conference of Machine Learning 2024 hosted in Vienna, Austria.

    Each session has an <Title>, <Authors>, <Time>, <Location>. and <Abstract>

    You will answer the users <Question> based on the provided selected sessions.

    <Sessions>
    """
    for selected_session in selected_sessions:
        SESSION_PROMPT = f"""
        <Title>
        {selected_session.payload.name}
        <Authors>
        {selected_session.payload.speakers_authors}
        <Time>
        {selected_session.payload.time_vienna}
        <Location>
        {selected_session.payload.location}
        <Abstract>
        {selected_session.payload.abstract}
        """
        PROMPT += SESSION_PROMPT
    PROMPT += f"""
    <Question>{query}
    """
    query_messages = [*messages]
    query_messages.append(
        {"role": "user", "content": PROMPT}
        )
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=query_messages, 
        stream=True
        )
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content

if "session_events" not in st.session_state:
    st.session_state["session_events"] = {}

if "prompt" not in st.session_state:
    st.session_state["prompt"] = ""

if "selected_sessions" not in st.session_state:
    st.session_state["selected_sessions"] = []

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

@st.cache_resource()
def get_reranker():
    return CohereReranker(column="abstract")


st.cache_resource()
def get_vectordb():
    return make_vectordb("/icml_data/icml_sessions.jsonl", "/root/data/vectordb")

def add_to_selected_sessions(date_key, session: LanceSchema):
    st.session_state.selected_sessions.append(session)
    

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
    if prompt := st.chat_input("What is up?"):
        results = table.search(prompt, query_type="hybrid").limit(100).rerank(normalize='score', reranker=reranker).to_pydantic(LanceSchema)
        st.session_state.session_events = group_and_sort_events(results)
        st.rerun()
    
    if len(st.session_state.prompt) > 0:
        with st.chat_message("user"):
            st.write(st.session_state.prompt)

    for group, (date_key, events_on_date) in enumerate(st.session_state.session_events.items()):
        
        st.markdown(f"## Events on {date_key}:")
        for event_id, event in enumerate(events_on_date):
            
            col1, col2 = st.columns([0.05, 0.95])
            with col1:
                check_val = st.checkbox("Checkbox", key=f"event_id_{group}_{event_id}", on_change=add_to_selected_sessions, args=[date_key, event], label_visibility="hidden")
            
            with col2:
                make_item(event)
        
with selection_area:
    st.header("Selected Sessions")
    for event_id, event in enumerate(st.session_state.selected_sessions):
        make_item(event)

with summarization_area:
    st.header("Selected Session Chat")
    
    summary_container = st.container(border=True)
    summary_container.markdown("<iframe scr='linke', class = 'fullHeight'></iframe>", unsafe_allow_html = True)

    with summary_container:
        for message in st.session_state.chat_messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.write(message["content"])
    
    if prompt_summary := st.chat_input("Ask anything about selected sessions"):
        
        with summary_container:
            with st.chat_message("user"):
                st.write(prompt_summary)
        
            with st.chat_message("assistant"):
                written_stream = summary_container.write_stream(
                    rag_prompt(
                        st.session_state.chat_messages, 
                        prompt_summary, 
                        st.session_state.selected_sessions
                        )
                )
        
        st.session_state.chat_messages.append({"role": "user", "content": prompt_summary})
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": written_stream}
        )
        st.rerun()