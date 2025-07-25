"""
LangChain Agents (by T.-W. Yoon, Mar. 2024)
"""

import streamlit as st
import os, base64, requests, datetime, json
import matplotlib.pyplot as plt
from io import BytesIO
from functools import partial
from tempfile import NamedTemporaryFile
from PIL import Image, UnidentifiedImageError
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool, tool
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
# from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.utilities import PythonREPL
from langchain.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel, Field
# The following are for type annotations
from typing import Union, List, Literal, Optional, Dict, Any, Annotated
from matplotlib.figure import Figure
from streamlit.runtime.uploaded_file_manager import UploadedFile


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


def check_api_keys() -> None:
    """
    Unset this flag to check the validity of the API keys
    """

    st.session_state.ready = False


def message_history_to_string(extra_space: bool=True) -> str:
    """
    Return a string of the chat history contained in
    st.session_state.history.
    """

    history_list = []
    for msg in st.session_state.history:
        if isinstance(msg, HumanMessage):
            history_list.append(f"Human: {msg.content}")
        else:
            history_list.append(f"AI: {msg.content}")
    new_lines = "\n\n" if extra_space else "\n"

    return new_lines.join(history_list)


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


def get_vector_store(uploaded_files: List[UploadedFile]) -> Optional[FAISS]:
    """
    Take a list of UploadedFile objects as input,
    and return a FAISS vector store.
    """

    if not uploaded_files:
        return None

    documents = []
    filepaths = []
    loader_map = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader
    }
    try:
        for uploaded_file in uploaded_files:
            # Create a temporary file within the "files/" directory
            with NamedTemporaryFile(dir="files/", delete=False) as file:
                file.write(uploaded_file.getbuffer())
                filepath = file.name
            filepaths.append(filepath)

            file_ext = os.path.splitext(uploaded_file.name.lower())[1]
            loader_class = loader_map.get(file_ext)
            if not loader_class:
                st.error(f"Unsupported file type: {file_ext}", icon="🚨")
                for filepath in filepaths:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                return None

            # Load the document using the selected loader.
            loader = loader_class(filepath)
            documents.extend(loader.load())

        with st.spinner("Vector store in preparation..."):
            # Split the loaded text into smaller chunks for processing.
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                # separators=["\n", "\n\n", "(?<=\. )", "", " "],
            )
            doc = text_splitter.split_documents(documents)
            # Create a FAISS vector database.
            if st.session_state.openai_key_validity:
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-large", dimensions=1536
                )
            else:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001"
                )
            vector_store = FAISS.from_documents(doc, embeddings)
    except Exception as e:
        vector_store = None
        st.error(f"An error occurred: {e}", icon="🚨")
    finally:
        # Ensure the temporary file is deleted after processing
        for filepath in filepaths:
            if os.path.exists(filepath):
                os.remove(filepath)

    return vector_store


def get_retriever() -> None:
    """
    Upload document(s), create a vector store, prepare a retriever tool,
    save the tool to the variable st.session_state.retriever_tool
    """

    st.write("")
    st.write("**Query Document(s)**")
    uploaded_files = st.file_uploader(
        label="Upload an article",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="document_upload_" + str(st.session_state.uploader_key),
    )

    if st.button(label="Create the vector store"):
        # Create the vector store.
        vector_store = get_vector_store(uploaded_files)

        if vector_store is not None:
            uploaded_file_names = [file.name for file in uploaded_files]
            file_names = ", ".join(uploaded_file_names)
            st.session_state.vector_store_message = (
                f"Vector store for :blue[[{file_names}]] is ready!"
            )
            retriever = vector_store.as_retriever()
            st.session_state.retriever_tool = create_retriever_tool(
                retriever,
                name="retriever",
                description=(
                    "Search for information about the uploaded documents. "
                    "For any questions about the documents, you must use "
                    "this tool!"
                ),
            )
            st.session_state.uploader_key += 1


def display_text_with_equations(text: str):
    # Replace inline LaTeX equation delimiters \\( ... \\) with $
    modified_text = text.replace("\\(", "$").replace("\\)", "$")

    # Replace block LaTeX equation delimiters \\[ ... \\] with $$
    modified_text = modified_text.replace("\\[", "$$").replace("\\]", "$$")

    # Use st.markdown to display the formatted text with equations
    st.markdown(modified_text)


def audio_to_text(audio_file: UploadedFile) -> Optional[str]:
    """
    Read audio bytes and return the corresponding text.
    """

    try:
        transcript = OpenAI().audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )
        text = transcript.text
    except Exception as e:
        text = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return text


def image_to_base64(image: Image) -> str:
    """
    Convert an image object from PIL to a base64-encoded image,
    and return the resulting encoded image as a string to be used
    in place of a URL.
    """

    # Convert the image to RGB mode if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Save the image to a BytesIO object
    buffered_image = BytesIO()
    image.save(buffered_image, format="JPEG")

    # Convert BytesIO to bytes and encode to base64
    img_str = base64.b64encode(buffered_image.getvalue())

    # Convert bytes to string
    base64_image = img_str.decode("utf-8")

    return f"data:image/jpeg;base64,{base64_image}"


def shorten_image(image: Image, max_pixels: int=1024) -> Image:
    """
    Take an Image object as input, and shorten the image size
    if the image is greater than max_pixels x max_pixels.
    """

    if max(image.width, image.height) > max_pixels:
        if image.width > image.height:
            new_width, new_height = 1024, image.height * 1024 // image.width
        else:
            new_width, new_height = image.width * 1024 // image.height, 1024

        image = image.resize((new_width, new_height))

    return image


def images_to_urls(uploaded_files: List[UploadedFile]) -> List[str]:
    """
    Convert uploaded image files to base64-encoded images.
    """

    image_urls = []
    try:
        for image_file in uploaded_files:
            image = Image.open(image_file)
            thumbnail = shorten_image(image, 300)
            st.image(thumbnail)
            image = shorten_image(image, 1024)
            image_urls.append(image_to_base64(image))
    except UnidentifiedImageError as e:
        st.error(f"An error occurred: {e}", icon="🚨")

    return image_urls


def fig_to_base64(fig: Figure) -> str:
    """
    Convert a Figure object to a base64-encoded image, and return
    the resulting encoded image to be used in place of a URL.
    """

    with BytesIO() as buffer:
        fig.savefig(buffer, format="JPEG")
        buffer.seek(0)
        image = Image.open(buffer)

        return image_to_base64(image)


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


@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = PythonREPL().run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


def set_tools() -> List[Tool]:
    """
    Set and return the tools for the agent. Tools that can be selected
    are internet_search, arxiv, wikipedia, python_repl, and retrieval.
    For searching the internet, a Google CSE ID is required together
    with a Google API key.
    """

    class MySearchToolInput(BaseModel):
        query: str = Field(description="search query to look up")

    arxiv = load_tools(["arxiv"])[0]
    wikipedia = load_tools(["wikipedia"])[0]
    # python_repl = PythonREPLTool()

    tool_options = ["ArXiv", "Wikipedia", "Python_REPL"]
    tool_dictionary = {
        "ArXiv": arxiv,
        "Wikipedia": wikipedia,
        "Python_REPL": python_repl,
    }

    if st.session_state.openai_key_validity or st.session_state.google_key_validity:
        tool_options.insert(0, "Retrieval")
        tool_dictionary["Retrieval"] = st.session_state.retriever_tool

    if st.session_state.google_key_validity and st.session_state.google_cse_id_validity:
        search = GoogleSearchAPIWrapper()
        internet_search = Tool(
            name="internet_search",
            description=(
                "A search engine for comprehensive, accurate, and trusted results. "
                "Useful for when you need to answer questions about current events. "
                "Input should be a search query."
            ),
            func=partial(search.results, num_results=5),
            args_schema=MySearchToolInput,
        )
        tool_options.insert(0, "Search")
        tool_dictionary["Search"] = internet_search

    st.write("")
    st.write("**Tools**")
    tool_names = st.multiselect(
        label="assistant tools",
        options=tool_options,
        default=st.session_state.tool_names[1],
        label_visibility="collapsed",
    )
    if "Search" not in tool_options:
        st.write(
            "<small>Tools are disabled when images are uploaded and "
            "queried. To search the internet, obtain your Google CSE ID "
            "[here](https://programmablesearchengine.google.com/about/). "
            "Once the valid id is entered, 'Search' will be displayed "
            "in the list of tools.</small>",
            unsafe_allow_html=True,
        )
    else:
        st.write(
            "<small>Tools are disabled when images are uploaded and "
            "queried.</small>",
            unsafe_allow_html=True,
        )
    if "Retrieval" in tool_names:
        # Get the retriever tool and save it to st.session_state.retriever_tool.
        get_retriever()
        if st.session_state.vector_store_message:
            st.write(st.session_state.vector_store_message)

    tools = [
        tool_dictionary[key]
        for key in tool_names if tool_dictionary[key] is not None
    ]
    st.session_state.tool_names[0] = tool_names

    return tools


def print_list(char_list: List[str]) -> str:
    """
    Print a list of characters in a human-readable format.
    """

    if len(char_list) > 2:
        result = ", ".join(char_list[:-1]) + ", and " + char_list[-1]
    elif len(char_list) == 2:
        result = char_list[0] + " and " + char_list[1]
    elif len(char_list) == 1:
        result = char_list[0]
    else:
        result = ""

    return result


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

    tool_names = print_list(st.session_state.tool_names[0])
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


def serialize_messages(
    messages: List[Union[HumanMessage, AIMessage]]
) -> List[Dict]:

    """
    Serialize the list of messages into a list of dicts
    """

    return [msg.model_dump() for msg in messages]


def deserialize_messages(
    serialized_messages: List[Dict]
) -> List[Union[HumanMessage, AIMessage]]:

    """
    Deserialize the list of messages from a list of dicts
    """

    deserialized_messages = []
    for msg in serialized_messages:
        if msg['type'] == 'human':
            deserialized_messages.append(HumanMessage(**msg))
        elif msg['type'] == 'ai':
            deserialized_messages.append(AIMessage(**msg))
    return deserialized_messages


def show_uploader() -> None:
    """
    Set the flag to show the uploader.
    """

    st.session_state.show_uploader = True


def check_conversation_keys(lst: List[Dict[str, Any]]) -> bool:
    """
    Check if all items in the given list are valid conversation entries.
    """

    return all(
        isinstance(item, dict) and
        isinstance(item.get("content"), str) and
        isinstance(item.get("type"), str) and
        isinstance(item.get("additional_kwargs"), dict)
        for item in lst
    )


def load_conversation() -> bool:
    """
    Load the conversation from a JSON file
    """

    st.write("")
    st.write("**Choose a (JSON) conversation file**")
    uploaded_file = st.file_uploader(
        label="Load conversation", type="json", label_visibility="collapsed"
    )
    if uploaded_file:
        try:
            data = json.load(uploaded_file)
            if isinstance(data, list) and check_conversation_keys(data):
                st.session_state.history = deserialize_messages(data)
                return True
            st.error(
                f"The uploaded data does not conform to the expected format.", icon="🚨"
            )
        except Exception as e:
            st.error(f"An error occurred: {e}", icon="🚨")

    return False


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
        **Enter your API Keys in the sidebar**

        - For the OpenAI models such as 'GPT-4o', you can obtain an OpenAI API
          key from https://platform.openai.com/account/api-keys.

        - For Claude models such as 'Claude-4-Sonnet', you can obtain an Anthropic
          API key from https://console.anthropic.com/settings/keys.

        - For Gemini models such as 'Gemini-2.5-Pro', you can obtain a Google API
          key from https://aistudio.google.com/app/apikey.

        - For searching the internet, obtain a Google CSE ID from
          https://programmablesearchengine.google.com/about/ along with your
          Google API key.
        """
    )
    st.image("files/Streamlit_Agent_App.png")
    st.info(
        """
        This app is coded by T.-W. Yoon, a professor of systems theory at
        Korea University. Take a look at some of his other projects:
        - [OpenAI Assistants](https://assistants.streamlit.app/)
        - [Multi-Agent Debate](https://multi-agent-debate.streamlit.app/)
        - [TWY's Playground](https://twy-playground.streamlit.app/)
        - [Differential equations](https://diff-eqn.streamlit.app/)
        """
    )


def agents() -> None:
    """
    Generate text or image by using llm models like "gpt-4o".
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
                st.info("Enter the correct password")
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
    agents()
