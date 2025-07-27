"""
API key validation module for various AI services (OpenAI, Anthropic, Google).
Provides functions to validate API keys and CSE ID by making test requests
to respective services and manages their storage in environment variables
and Streamlit session state. Each validation function returns a boolean
indicating the key's validity and updates the application state accordingly.
"""

import os
import requests
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_community import GoogleSearchAPIWrapper


def is_openai_api_key_valid(openai_api_key: str) -> bool:
    """
    Return True if the given OpenAI API key is valid.
    """

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
    }
    try:
        response = requests.get(
            "https://api.openai.com/v1/models", headers=headers
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def is_anthropic_api_key_valid(anthropic_api_key: str) -> bool:
    """
    Return True if the given Anthropic API key is valid.
    """

    headers = {
        "x-api-key": anthropic_api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    payload = {
        "model": "claude-3-5-haiku-latest",
        "max_tokens": 10,
        "messages": [
            {"role": "user", "content": "Hi."}
        ]
    }
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def is_google_api_key_valid(google_api_key: str) -> bool:
    """
    Return True if the given Google API key is valid.
    """

    if not google_api_key:
        return False

    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", google_api_key=google_api_key
    )
    try:
        gemini_llm.invoke("Hello")
    except:
        return False
    else:
        return True


def are_google_api_key_cse_id_valid(
    google_api_key: str, google_cse_id: str
) -> bool:

    """
    Return True if the given Google API key and CSE ID are valid.
    """

    if google_api_key and google_cse_id:
        try:
            search = GoogleSearchAPIWrapper(
                google_api_key=google_api_key,
                google_cse_id=google_cse_id,
                k=1
            )
            result = search.run("test query")
            return (
                not isinstance(result, str) or
                not result.startswith("Google Search Error")
            )
        except:
            return False
    else:
        return False


def check_openai_key(openai_api_key: str) -> None:
    """
    Check if the given OpenAI API key is valid.
    """

    if is_openai_api_key_valid(openai_api_key):
        st.session_state.openai_key_validity = True
        os.environ["OPENAI_API_KEY"] = openai_api_key
    else:
        st.session_state.openai_key_validity = False
        os.environ["OPENAI_API_KEY"] = ""


def check_anthropic_key(anthropic_api_key: str) -> None:
    """
    Check if the given OpenAI API key is valid.
    """

    if is_anthropic_api_key_valid(anthropic_api_key):
        st.session_state.anthropic_key_validity = True
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    else:
        st.session_state.anthropic_key_validity = False
        os.environ["ANTHROPIC_API_KEY"] = ""


def check_google_key(google_api_key: str) -> None:
    """
    Check if the given OpenAI API key is valid.
    """

    if is_google_api_key_valid(google_api_key):
        st.session_state.google_key_validity = True
        os.environ["GOOGLE_API_KEY"] = google_api_key
    else:
        st.session_state.google_key_validity = False
        os.environ["GOOGLE_API_KEY"] = ""


def check_google_cse_id(google_cse_id: str) -> None:
    """
    Check if the given OpenAI API key is valid.
    """

    if (
        st.session_state.google_key_validity and
        are_google_api_key_cse_id_valid(os.environ["GOOGLE_API_KEY"], google_cse_id)
    ):
        st.session_state.google_cse_id_validity = True
        os.environ["GOOGLE_CSE_ID"] = google_cse_id
    else:
        st.session_state.google_cse_id_validity = False
        os.environ["GOOGLE_CSE_ID"] = ""
