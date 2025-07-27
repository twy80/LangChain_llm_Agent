"""
Manages Streamlit session state variables for the chatbot application.
Handles initialization and reset of conversation history, AI roles,
tool states, and API key validation flags. This module ensures
persistent state management across app reruns.
"""

import streamlit as st


def initialize_session_state_variables() -> None:
    """
    Initialize all the session state variables.
    """

    # Variables for chatbot
    if "history" not in st.session_state:
        st.session_state.history = []
    if "ai_role" not in st.session_state:
        st.session_state.ai_role = 2 * ["You are a helpful assistant."]
    if "prompt_exists" not in st.session_state:
        st.session_state.prompt_exists = False
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    # Variables for tools
    if "tool_names" not in st.session_state:
        st.session_state.tool_names = [[], []]
    if "vector_store_message" not in st.session_state:
        st.session_state.vector_store_message = None
    if "retriever_tool" not in st.session_state:
        st.session_state.retriever_tool = None
    if "show_uploader" not in st.session_state:
        st.session_state.show_uploader = False

    # Variables for API key validation
    if "ready" not in st.session_state:
        st.session_state.ready = False
    if "anthropic_key_validity" not in st.session_state:
        st.session_state.anthropic_key_validity = False
    if "google_key_validity" not in st.session_state:
        st.session_state.google_key_validity = False
    if "google_cse_id_validity" not in st.session_state:
        st.session_state.google_cse_id_validity = False
    if "openai_key_validity" not in st.session_state:
        st.session_state.openai_key_validity = False


def reset_conversation() -> None:
    """
    Reset the session_state variables for resetting the conversation.
    """

    st.session_state.history = []
    st.session_state.ai_role[1] = st.session_state.ai_role[0]
    st.session_state.prompt_exists = False
    st.session_state.vector_store_message = None
    st.session_state.tool_names[1] = st.session_state.tool_names[0]
    st.session_state.retriever_tool = None
    st.session_state.uploader_key = 0


def check_api_keys() -> None:
    """
    Unset this flag to check the validity of the API keys
    """

    st.session_state.ready = False


def show_uploader() -> None:
    """
    Set the flag to show the uploader.
    """

    st.session_state.show_uploader = True
