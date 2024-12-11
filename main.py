# Frontend using streamlit
from backend.core import run_llm
import streamlit as st
from streamlit_chat import message
import os

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

st.header("Programming Documentation Chatbot")

# Model selection dropdown
#llm_model = st.selectbox("Choose the Language Model", ["gpt-3.5-turbo", "gpt-4", "default"])
temperature = st.slider("Set Temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.7)
verbose = st.checkbox("Verbose Mode", value=True)

embedding_option = st.selectbox(
    "Select an Embedding Model",
    ("text-embedding-3-small", "text-embedding-3-large")
)

prompt = st.text_input("Prompt", placeholder = "Enter your query here.")

# Initialization of chat history for context aware responses
if ("chat_answer_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answer_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

# takes sources from links and shows as reference
def create_sources_string(sources_urls: set) -> str:
    if not sources_urls:
        return ""
    sources_list = list(sources_urls)
    sources_list.sort()
    sources_string = "Sources:\n"
    for i, source in enumerate(sources_list):
        url = source.split("https:/")[-1].replace("\\", "/")  # Removes the file path part and formats slashes
        full_url = f"https://{url}"
        sources_string += f"{i+1}. [{full_url}]({full_url})\n"
    return sources_string

if prompt:
    with st.spinner("Generating response..."):
        generate_response = run_llm(
            query = prompt,
            chat_history = st.session_state["chat_history"],
            embedding_model=embedding_option,
            temperature = temperature,
            verbose = verbose
        )
        print(generate_response)

        sources = set(
            [doc.metadata["source"] for doc in generate_response["source_documents"]]
        )

        formatted_response = (
            f"{generate_response['result']} \n\n {create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answer_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generate_response["result"]))

if st.session_state["chat_answer_history"]:
    for i, (user_query, generate_response) in enumerate(zip( st.session_state["user_prompt_history"], st.session_state["chat_answer_history"])):
        message(user_query, is_user=True, key=f"user_{i}")  # Unique key for user messages
        message(generate_response, key=f"response_{i}")