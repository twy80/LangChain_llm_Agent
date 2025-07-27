"""
Core agent functionality for handling chat interactions with various LLM
providers. Includes classes and functions for streaming responses,
processing queries with images and tools, and managing chat sessions
with OpenAI, Anthropic, and Google AI models.

Key components:
- StreamHandler: Manages real-time response streaming
- get_chat_model: Initializes appropriate chat model based on provider
- process_with_images: Handles queries with image inputs
- process_with_tools: Executes tool-augmented interactions
- run_agent: Main entry point for processing user queries
"""

from typing import Union, List, Any

import streamlit as st
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage, AIMessage
from langchain.tools import Tool
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: Any, **kwargs) -> None:
        new_text = self._extract_text(token)
        if new_text:
            self.text += new_text
            self.container.markdown(self.text)

    def _extract_text(self, token: Any) -> str:
        if isinstance(token, str):
            return token
        elif isinstance(token, list):
            return ''.join(self._extract_text(t) for t in token)
        elif isinstance(token, dict):
            return token.get('text', '')
        else:
            return str(token)


def get_chat_model(
    model: str,
    temperature: float,
    callbacks: List[BaseCallbackHandler]
) -> Union[ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI, None]:

    """
    Get the appropriate chat model based on the given model name.
    """

    model_map = {
        "claude-": ChatAnthropic,
        "gemini-": ChatGoogleGenerativeAI,
        "gpt-": ChatOpenAI
    }
    for prefix, ModelClass in model_map.items():
        if model.startswith(prefix):
            return ModelClass(
                model=model,
                temperature=temperature,
                streaming=True,
                callbacks=callbacks
            )
    return None


def process_with_images(
    llm: Union[ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI],
    message_content: str,
    image_urls: List[str]
) -> str:

    """
    Process the given history query with associated images using a language model.
    """

    content_with_images = (
        [{"type": "text", "text": message_content}] +
        [{"type": "image_url", "image_url": {"url": url}} for url in image_urls]
    )
    message_with_images = [HumanMessage(content=content_with_images)]

    return llm.invoke(message_with_images).content


def process_with_tools(
    llm: Union[ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI],
    tools: List[Tool],
    agent_prompt: str,
    history_query: dict
) -> str:

    """
    Create an AI agent based on the specified agent type and tools,
    then use this agent to process the given history query.
    """

    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, max_iterations=5, verbose=False,
        handle_parsing_errors=True,
    )

    return agent_executor.invoke(history_query)["output"]


def run_agent(
    query: str,
    model: str,
    tools: List[Tool],
    image_urls: List[str],
    temperature: float=1.0
) -> Union[str, None]:

    """
    Generate text based on user queries.

    Args:
        query: User's query
        model: LLM like "gpt-4o"
        tools: list of tools such as Search and Retrieval
        image_urls: List of URLs for images
        temperature: Value between 0 and 2. Defaults to 1.0

    Return:
        generated text

    The chat prompt and message history are stored in
    st.session_state variables.
    """

    try:
        llm = get_chat_model(model, temperature, [StreamHandler(st.empty())])
        if llm is None:
            st.error(f"Unsupported model: {model}", icon="🚨")
            return None
        
        history_query = {
            "chat_history": st.session_state.history, "input": query
        }
        
        message_with_no_image = st.session_state.chat_prompt.invoke(history_query)
        message_content = message_with_no_image.messages[0].content

        if image_urls:
            generated_text = process_with_images(llm, message_content, image_urls)
            human_message = HumanMessage(
                content=query, additional_kwargs={"image_urls": image_urls}
            )
        elif tools:
            generated_text = process_with_tools(
                llm, tools, st.session_state.agent_prompt, history_query
            )
            human_message = HumanMessage(content=query)
        else:
            generated_text = llm.invoke(message_with_no_image).content
            human_message = HumanMessage(content=query)

        if isinstance(generated_text, list):
            generated_text = generated_text[0]["text"]

        st.session_state.history.append(human_message)
        st.session_state.history.append(AIMessage(content=generated_text))

        return generated_text

    except Exception as e:
        st.error(f"An error occurred: {e}", icon="🚨")
        return None
