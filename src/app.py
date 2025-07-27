"""
A Streamlit web application that provides an interactive chat interface
with various LLM models (GPT-4, Claude, Gemini). Features include:

- Conversation history management (save/load/reset)
- Multiple API key handling (OpenAI, Anthropic, Google)
- Tool integration for enhanced capabilities
- LaTeX equation display support
- Multiple AI roles (general assistant, coding adviser, ...)
- Temperature control for response randomness
- Voice input support

Author: T.-W. Yoon
Date: Mar 2025
"""

import datetime
import json
import os
from typing import Union, Literal

import matplotlib.pyplot as plt
import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent_core import run_agent
from api_validation import (
    check_anthropic_key, check_openai_key, check_google_key, check_google_cse_id
)
from session_state import (
    initialize_session_state_variables, reset_conversation, check_api_keys, show_uploader
)
from tools import set_tools
from utils import (
    audio_to_text, images_to_urls, fig_to_base64, print_char_list,
    load_conversation, serialize_messages, display_text_with_equations
)

def print_conversation(no_of_msgs: Union[Literal["All"], int]) -> None:
    """
    Print the conversation stored in st.session_state.history.
    """

    if no_of_msgs == "All":
        no_of_msgs = len(st.session_state.history)

    for msg in st.session_state.history[-no_of_msgs:]:
        if isinstance(msg, HumanMessage):
            with st.chat_message("human"):
                st.write(msg.content)
        else:
            with st.chat_message("ai"):
                display_text_with_equations(msg.content)

        if urls := msg.additional_kwargs.get("image_urls"):
            for url in urls:
                st.image(url)


def set_prompts() -> None:
    """
    Set chat and agent prompts for tool calling agents
    """

    st.session_state.chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"{st.session_state.ai_role[0]} Your goal is to provide "
                "answers to human inquiries. Should the information not "
                "be available, inform the human explicitly that "
                "the answer could not be found."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    tool_names = print_char_list(st.session_state.tool_names[0])
    st.session_state.agent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"{st.session_state.ai_role[0]} "
                "Your goal is to provide answers to human inquiries.\n\n"
                "You have access to the following tool(s): "
                f"{tool_names}\n\n"
                "When giving your answers, tell the human what your response "
                "is based on and which tools you use. Use Markdown syntax "
                "and include relevant sources, such as links (URLs), following "
                "MLA format. Should the information not be available, inform "
                "the human explicitly that the answer could not be found. "
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )


def create_text(model: str) -> None:
    """
    Take an LLM as input and generate text based on user input
    by calling run_agent().
    """

    # initial system prompts
    general_role = "You are a helpful assistant."
    english_teacher = (
        "You are an English teacher who analyzes texts and corrects "
        "any grammatical issues if necessary."
    )
    translator = (
        "You are a translator who translates English into Korean "
        "and Korean into English."
    )
    coding_adviser = (
        "You are an expert in coding who provides advice on "
        "good coding styles."
    )
    science_assistant = "You are a science assistant."
    roles = (
        general_role, english_teacher, translator,
        coding_adviser, science_assistant
    )

    with st.sidebar:
        st.write("")
        st.write("**Temperature**")
        temperature = st.slider(
            label="Temperature (higher $\Rightarrow$ more random)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            format="%.1f",
            label_visibility="collapsed",
        )
        st.write("")
        st.write("**Messages to Show**")
        no_of_msgs = st.radio(
            label="$\\textsf{Messages to show}$",
            options=("All", 20, 10),
            label_visibility="collapsed",
            horizontal=True,
            index=2,
        )

    st.write("")
    st.write("##### Message to AI")
    st.session_state.ai_role[0] = st.selectbox(
        label="AI's role",
        options=roles,
        index=roles.index(st.session_state.ai_role[1]),
        label_visibility="collapsed",
    )

    if st.session_state.ai_role[0] != st.session_state.ai_role[1]:
        reset_conversation()
        st.rerun()

    st.write("")
    st.write("##### Conversation with AI")

    # Print conversation
    print_conversation(no_of_msgs)

    # Reset, download, or load the conversation
    c1, c2, c3 = st.columns(3)
    c1.button(
        label="$~\:\,\,$Reset$~\:\,\,$",
        on_click=reset_conversation
    )
    current_datetime = datetime.datetime.now()
    datetime_string = current_datetime.strftime("%Y.%m.%d-%H.%M")
    download_file_name = "conversation-" + datetime_string + ".json"
    c2.download_button(
        label="Download",
        data=json.dumps(serialize_messages(st.session_state.history), indent=4),
        file_name=download_file_name,
        mime="application/json",
    )
    c3.button(
        label="$~~\:\,$Load$~~\:\,$",
        on_click=show_uploader,
    )

    if st.session_state.show_uploader and load_conversation():
        st.session_state.show_uploader = False
        st.rerun()

    # Set the agent prompts and tools
    set_prompts()
    tools = set_tools()
    image_urls = []

    # Use your microphone
    st.write("")
    audio_file = st.audio_input(
        label="Speak your query",
        label_visibility="collapsed",
        key="audio_upload_" + str(st.session_state.uploader_key)
    )
    if audio_file:
        query = audio_to_text(audio_file)
        st.session_state.prompt_exists = True

    # Use your keyboard
    text_and_files = st.chat_input(
        placeholder="Enter your query",
        accept_file="multiple",
        file_type=["jpg", "jpeg", "png", "bmp"]
    )
    if text_and_files:
        query = text_and_files.text.strip()
        if text_and_files.files:
            image_urls = images_to_urls(text_and_files.files)
        st.session_state.prompt_exists = True

    if st.session_state.prompt_exists:
        with st.chat_message("human"):
            st.write(query)

        with st.chat_message("ai"):
            generated_text = run_agent(
                query=query,
                model=model,
                tools=tools,
                image_urls=image_urls,
                temperature=temperature
            )
            fig = plt.gcf()
            if fig and fig.get_axes():
                generated_image_url = fig_to_base64(fig)
                st.session_state.history[-1].additional_kwargs["image_urls"] = [
                    generated_image_url
                ]

        st.session_state.prompt_exists = False

        if generated_text is not None:
            st.session_state.uploader_key += 1
            st.rerun()


def show_guide() -> None:
    """
    Show the guide for using the LLM agent app.
    """

    st.info(
        """
        This app presents a tool calling agent. The supported models are Claude-3.5-Haiku,
        Claude-3.5-Sonnet, Claude-Sonnet-4, and Claude-Opus-4 from Anthropic;
        GPT-4o-mini and GPT-4o from OpenAI; and Gemini-2.5-Flash and Gemini-2.5-Pro from Google.
        
        - For the OpenAI models such as 'GPT-4o', your OpenAI API key is needed. You can obtain
          an API key from <https://platform.openai.com/account/api-keys>.

        - For Claude models such as 'Claude-4-Sonnet', your Anthropic API key is needed.
          You can obtain an API key from <https://console.anthropic.com/settings/keys>.

        - For Gemini models such as 'Gemini-2.5-Pro', your Google API key is needed.
          You can obtain an API key from <https://aistudio.google.com/app/apikey>.

        - For searching the internet, use a Google CSE ID obtained from
          <https://programmablesearchengine.google.com/about/> along with
          your Google API key.

        - In addition to the search tool from Google, ArXiv, Wikipedia,
          Retrieval (RAG), and pythonREPL are supported.

        - Temperature can be set by the user.

        - Tracing LLM messages is possible using LangSmith if you download the source code
          and run it on your machine or server.  For this, you need a
          LangChain API key that can be obtained from <https://smith.langchain.com/settings>.

        - When running the code on your machine or server, you can use st.secrets to keep and
          fetch your API keys as environments variables. For such secrets management, see
          <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management>.
        """
    )


def main() -> None:
    """
    Runs a Streamlit web application for text generation using various
    LLM models (GPT, Claude, Gemini) with API key management.
    """

    page_title = "LangChain LLM Agent"
    page_icon = "📚"

    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="centered"
    )

    st.write(f"## {page_icon} $\,${page_title}")

    # Initialize all the session state variables
    initialize_session_state_variables()

    with st.sidebar:
        st.write("")
        st.write("**API Key Selection**")
        choice_api = st.sidebar.radio(
            label="Choice of API",
            options=("Your keys", "My keys"),
            label_visibility="collapsed",
            horizontal=True,
            on_change=check_api_keys
        )
        if choice_api == "My keys":
            st.write("")
            st.write("**Password**")
            user_pin = st.text_input(
                label="Enter password", type="password", label_visibility="collapsed"
            )
            if user_pin == st.secrets["USER_PIN"]:
                anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]
                openai_api_key = st.secrets["OPENAI_API_KEY"]
                google_api_key = st.secrets["GOOGLE_API_KEY"]
                google_cse_id = st.secrets["GOOGLE_CSE_ID"]
                os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
                current_date = datetime.datetime.now().date()
                date_string = str(current_date)
                os.environ["LANGCHAIN_PROJECT"] = "llm_agent_" + date_string
            else:
                anthropic_api_key = ""
                openai_api_key = ""
                google_api_key = ""
                google_cse_id = ""
                st.info("Enter the password")
        else:
            st.write("")
            st.write("**API keys**")
            with st.expander("**Enter your keys**", expanded=False):
                st.write(
                    f"<small>$\:\!$Anthropic API Key</small>",
                    unsafe_allow_html=True
                )
                anthropic_api_key = st.text_input(
                    label="Anthropic API Key",
                    type="password",
                    label_visibility="collapsed",
                    on_change=check_api_keys
                )
                st.write(
                    f"<small>$\:\!$OpenAI API Key</small>",
                    unsafe_allow_html=True
                )
                openai_api_key = st.text_input(
                    label="OpenAI API Key",
                    type="password",
                    label_visibility="collapsed",
                    on_change=check_api_keys
                )
                st.write(
                    f"<small>$\:\!$Google API Key</small>",
                    unsafe_allow_html=True
                )
                google_api_key = st.text_input(
                    label="Google API Key",
                    type="password",
                    label_visibility="collapsed",
                    on_change=check_api_keys
                )
                st.write(
                    f"<small>$\:\!$Google CSE ID</small>",
                    unsafe_allow_html=True
                )
                google_cse_id = st.text_input(
                    label="Google CSE ID",
                    type="password",
                    label_visibility="collapsed",
                    on_change=check_api_keys
                )

        if not st.session_state.ready:
            # Check the validity of the API keys
            check_anthropic_key(anthropic_api_key)
            check_openai_key(openai_api_key)
            check_google_key(google_api_key)
            check_google_cse_id(google_cse_id)

            st.session_state.ready = (
                st.session_state.anthropic_key_validity or
                st.session_state.google_key_validity or
                st.session_state.openai_key_validity
            )

    gpt_models = ("gpt-4o-mini", "gpt-4o")
    claude_models = (
        "claude-3-5-haiku", "claude-3-5-sonnet", "claude-sonnet-4", "claude-opus-4"
    )
    gemini_models = ("gemini-2.5-flash", "gemini-2.5-pro")

    if choice_api == "My keys":
        st.sidebar.write("")
        st.sidebar.write("**LangSmith Tracing**")
        langsmith = st.sidebar.radio(
            label="LangSmith Tracing",
            options=("On", "Off"),
            label_visibility="collapsed",
            index=1,
            horizontal=True
        )
        os.environ["LANGCHAIN_TRACING_V2"] = (
            "true" if langsmith == "On" else "false"
        )
    model_options = ()
    if st.session_state.anthropic_key_validity:
        model_options += claude_models
    if st.session_state.openai_key_validity:
        model_options += gpt_models
    if st.session_state.google_key_validity:
        model_options += gemini_models

    if model_options:
        st.sidebar.write("")
        st.sidebar.write("**Model**")
        model = st.sidebar.radio(
            label="Models",
            options=model_options,
            label_visibility="collapsed",
            index=0,
        )
        if model == "claude-3-5-haiku":
            model += "-latest"
        elif model == "claude-3-5-sonnet":
            model += "-latest"
        elif model == "claude-sonnet-4":
            model += "-20250514"
        elif model == "claude-opus-4":
            model += "-20250514"
        create_text(model)
    else:
        show_guide()

    with st.sidebar:
        st.write("---")
        st.write(
            "<small>**T.-W. Yoon**, Mar. 2024  \n</small>",
            "<small>[OpenAI Assistants](https://assistants.streamlit.app/)  \n</small>",
            "<small>[Multi-Agent Debate](https://multi-agent-debate.streamlit.app/)  \n</small>",
            "<small>[TWY's Playground](https://twy-playground.streamlit.app/)  \n</small>",
            "<small>[Differential equations](https://diff-eqn.streamlit.app/)</small>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
