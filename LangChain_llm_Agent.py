"""
LangChain Agents (by T.-W. Yoon, Mar. 2024)
"""

import streamlit as st
import os, base64, re, requests, datetime, time
import matplotlib.pyplot as plt
from io import BytesIO
from functools import partial
from tempfile import NamedTemporaryFile
from audio_recorder_streamlit import audio_recorder
from PIL import Image, UnidentifiedImageError
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain_community.utilities import BingSearchAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent
from langchain.agents import create_react_agent
# from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.agents import load_tools
from langchain_experimental.tools import PythonREPLTool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.pydantic_v1 import BaseModel, Field
# The following are for type annotations
from typing import Union, List, Literal, Optional
from matplotlib.figure import Figure
from streamlit.runtime.uploaded_file_manager import UploadedFile
from openai._legacy_response import HttpxBinaryResponseContent


def initialize_session_state_variables() -> None:
    """
    Initialize all the session state variables.
    """

    # Variables for chatbot
    if "ready" not in st.session_state:
        st.session_state.ready = False

    if "openai" not in st.session_state:
        st.session_state.openai = None

    if "history" not in st.session_state:
        st.session_state.history = []

    if "model_type" not in st.session_state:
        st.session_state.model_type = "GPT Models from OpenAI"

    if "agent_type" not in st.session_state:
        st.session_state.agent_type = 2 * ["Tool Calling"]

    if "ai_role" not in st.session_state:
        st.session_state.ai_role = 2 * ["You are a helpful AI assistant."]

    if "prompt_exists" not in st.session_state:
        st.session_state.prompt_exists = False

    if "temperature" not in st.session_state:
        st.session_state.temperature = [0.7, 0.7]

    # Variables for audio and image
    if "audio_bytes" not in st.session_state:
        st.session_state.audio_bytes = None

    if "mic_used" not in st.session_state:
        st.session_state.mic_used = False

    if "audio_response" not in st.session_state:
        st.session_state.audio_response = None

    if "image_url" not in st.session_state:
        st.session_state.image_url = None

    if "image_description" not in st.session_state:
        st.session_state.image_description = None

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    # Variables for tools
    if "tool_names" not in st.session_state:
        st.session_state.tool_names = [[], []]

    if "bing_subscription_validity" not in st.session_state:
        st.session_state.bing_subscription_validity = False

    if "google_cse_id_validity" not in st.session_state:
        st.session_state.google_cse_id_validity = False

    if "vector_store_message" not in st.session_state:
        st.session_state.vector_store_message = None

    if "retriever_tool" not in st.session_state:
        st.session_state.retriever_tool = None


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def is_openai_api_key_valid(openai_api_key: str) -> bool:
    """
    Return True if the given OpenAI API key is valid.
    """

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
    }
    response = requests.get(
        "https://api.openai.com/v1/models", headers=headers
    )

    return response.status_code == 200


def is_bing_subscription_key_valid(bing_subscription_key: str) -> bool:
    """
    Return True if the given Bing subscription key is valid.
    """

    if not bing_subscription_key:
        return False
    try:
        search = BingSearchAPIWrapper(
            bing_subscription_key=bing_subscription_key,
            bing_search_url="https://api.bing.microsoft.com/v7.0/search",
            k=1
        )
        search.run("Where can I get a Bing subscription key?")
    except:
        return False
    else:
        return True


def is_google_api_key_valid(google_api_key: str) -> bool:
    """
    Return True if the given Google API key is valid.
    """

    if not google_api_key:
        return False

    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-pro", google_api_key=google_api_key
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
            search.run("Where can I get a Google CSE ID?")
        except:
            return False
        else:
            return True
    else:
        return False


def check_api_keys() -> None:
    # Unset this flag to check the validity of the OpenAI API key
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


def run_agent(
    query: str,
    model: str,
    tools: List[Tool],
    image_urls: List[str],
    temperature: float=0.7,
    agent_type: Literal["Tool Calling", "ReAct"]="Tool Calling",
) -> str:

    """
    Generate text based on user queries.

    Args:
        query: User's query
        model: LLM like "gpt-3.5-turbo"
        tools: list of tools such as Search and Retrieval
        image_urls: List of URLs for images
        temperature: Value between 0 and 1. Defaults to 0.7
        agent_type: 'Tool Calling' or 'ReAct'

    Return:
        generated text

    The chat prompt and message history are stored in
    st.session_state variables.
    """

    if model.startswith("gpt-"):
        ChatModel = ChatOpenAI
        if image_urls:
            model = "gpt-4o"
    else:
        ChatModel = ChatGoogleGenerativeAI
        if image_urls:
            model = "gemini-pro-vision"

    llm = ChatModel(
        model=model,
        temperature=temperature,
        streaming=True,
        callbacks=[StreamHandler(st.empty())]
    )

    if agent_type == "Tool Calling":
        chat_history = st.session_state.history
    else:
        chat_history = message_history_to_string()

    history_query = {
        "chat_history": chat_history,
        "input": query,
    }
    message_with_no_image = st.session_state.chat_prompt.invoke(history_query)

    try:
        if image_urls:
            content_with_images = (
                [{"type": "text", "text": message_with_no_image.messages[0].content}] +
                [{"type": "image_url", "image_url": {"url": url}} for url in image_urls]
            )
            message_with_images = [HumanMessage(content=content_with_images)]
            generated_text = llm.invoke(message_with_images).content
        elif tools:
            if agent_type == "Tool Calling":
                agent = create_openai_tools_agent(
                    llm, tools, st.session_state.agent_prompt
                )
            else:
                agent = create_react_agent(
                    llm, tools, st.session_state.agent_prompt
                )
            agent_executor = AgentExecutor(
                agent=agent, tools=tools, max_iterations=5, verbose=False,
                handle_parsing_errors=True,
            )
            generated_text = agent_executor.invoke(history_query)["output"]
        else:
            generated_text = llm.invoke(message_with_no_image).content

        if image_urls:
            human_message = HumanMessage(
                content=query, additional_kwargs={"image_urls": image_urls}
            )
        else:
            human_message = HumanMessage(content=query)
        st.session_state.history.append(human_message)
        st.session_state.history.append(AIMessage(content=generated_text))

    except Exception as e:
        generated_text = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return generated_text


def openai_create_image(
    description: str, model: str="dall-e-3", size: str="1024x1024"
) -> Optional[str]:

    """
    Generate image based on user description.

    Args:
        description: User description
        model: Default set to "dall-e-3"
        size: Pixel size of the generated image

    Return:
        URL of the generated image
    """

    try:
        with st.spinner("AI is generating..."):
            response = st.session_state.openai.images.generate(
                model=model,
                prompt=description,
                size=size,
                quality="standard",
                n=1,
            )
        image_url = response.data[0].url
    except Exception as e:
        image_url = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return image_url


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
            if st.session_state.model_type == "GPT Models from OpenAI":
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


def read_audio(audio_bytes: bytes) -> Optional[str]:
    """
    Read audio bytes and return the corresponding text.
    """
    try:
        audio_data = BytesIO(audio_bytes)
        audio_data.name = "recorded_audio.wav"  # dummy name

        transcript = st.session_state.openai.audio.transcriptions.create(
            model="whisper-1", file=audio_data
        )
        text = transcript.text
    except Exception as e:
        text = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return text


def input_from_mic() -> Optional[str]:
    """
    Convert audio input from mic to text and return it.
    If there is no audio input, None is returned.
    """

    time.sleep(0.5)
    audio_bytes = audio_recorder(
        pause_threshold=3.0, text="Speak", icon_size="2x",
        recording_color="#e87070", neutral_color="#6aa36f"        
    )

    if audio_bytes == st.session_state.audio_bytes or audio_bytes is None:
        return None
    else:
        st.session_state.audio_bytes = audio_bytes
        return read_audio(audio_bytes)


def perform_tts(text: str) -> Optional[HttpxBinaryResponseContent]:
    """
    Take text as input, perform text-to-speech (TTS),
    and return an audio_response.
    """

    try:
        with st.spinner("TTS in progress..."):
            audio_response = st.session_state.openai.audio.speech.create(
                model="tts-1",
                voice="fable",
                input=text,
            )
    except Exception as e:
        audio_response = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return audio_response


def play_audio(audio_response: HttpxBinaryResponseContent) -> None:
    """
    Take an audio response (a bytes-like object)
    from TTS as input, and play the audio.
    """

    audio_data = audio_response.read()

    # Encode audio data to base64
    b64 = base64.b64encode(audio_data).decode("utf-8")

    # Create a markdown string to embed the audio player with the base64 source
    md = f"""
        <audio controls autoplay style="width: 100%;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        Your browser does not support the audio element.
        </audio>
        """

    # Use Streamlit to render the audio player
    st.markdown(md, unsafe_allow_html=True)


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


def upload_image_files_return_urls(
    type: List[str]=["jpg", "jpeg", "png", "bmp"]
) -> List[str]:

    """
    Upload image files, convert them to base64-encoded images, and
    return the list of the resulting encoded images to be used
    in place of URLs.
    """

    st.write("")
    st.write("**Query Image(s)**")
    source = st.radio(
        label="Image selection",
        options=("Uploaded", "From URL"),
        horizontal=True,
        label_visibility="collapsed",
    )
    image_urls = []

    if source == "Uploaded":
        uploaded_files = st.file_uploader(
            label="Upload images",
            type=type,
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="image_upload_" + str(st.session_state.uploader_key),
        )
        if uploaded_files:
            try:
                for image_file in uploaded_files:
                    image = Image.open(image_file)
                    thumbnail = shorten_image(image, 300)
                    st.image(thumbnail)
                    image = shorten_image(image, 1024)
                    image_urls.append(image_to_base64(image))
            except UnidentifiedImageError as e:
                st.error(f"An error occurred: {e}", icon="🚨")
    else:
        image_url = st.text_input(
            label="URL of the image",
            label_visibility="collapsed",
            key="image_url_" + str(st.session_state.uploader_key),
        )
        if image_url:
            if is_url(image_url):
                st.image(image_url)
                image_urls = [image_url]
            else:
                st.error("Enter a proper URL", icon="🚨")

    return image_urls


def fig_to_base64(fig: Figure) -> str:
    """
    Convert a Figure object to a base64-encoded image, and return
    the resulting encoded image to be used in place of a URL.
    """

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    image = Image.open(buffer)

    return image_to_base64(image)


def is_url(text: str) -> bool:
    """
    Determine whether text is a URL or not.
    """

    regex = r"(http|https)://([\w_-]+(?:\.[\w_-]+)+)(:\S*)?"
    p = re.compile(regex)
    match = p.match(text)
    if match:
        return True
    else:
        return False


def reset_conversation() -> None:
    """
    Reset the session_state variables for resetting the conversation.
    """

    st.session_state.history = []
    st.session_state.ai_role[1] = st.session_state.ai_role[0]
    st.session_state.prompt_exists = False
    st.session_state.temperature[1] = st.session_state.temperature[0]
    st.session_state.audio_response = None
    st.session_state.vector_store_message = None
    st.session_state.tool_names[1] = st.session_state.tool_names[0]
    st.session_state.agent_type[1] = st.session_state.agent_type[0]
    st.session_state.retriever_tool = None
    st.session_state.uploader_key = 0


def switch_between_apps() -> None:
    """
    Keep the chat settings when switching the mode.
    """

    st.session_state.temperature[1] = st.session_state.temperature[0]
    st.session_state.ai_role[1] = st.session_state.ai_role[0]
    st.session_state.tool_names[1] = st.session_state.tool_names[0]
    st.session_state.agent_type[1] = st.session_state.agent_type[0]


def set_tools() -> List[Tool]:
    """
    Set and return the tools for the agent. Tools that can be selected
    are internet_search, arxiv, wikipedia, python_repl, and retrieval.
    A Bing Subscription Key or Google CSE ID is required for internet_search.
    """

    class MySearchToolInput(BaseModel):
        query: str = Field(description="search query to look up")

    arxiv = load_tools(["arxiv"])[0]
    wikipedia = load_tools(["wikipedia"])[0]
    python_repl = PythonREPLTool()

    tool_options = ["ArXiv", "Wikipedia", "Python_REPL", "Retrieval"]
    tool_dictionary = {
        "ArXiv": arxiv,
        "Wikipedia": wikipedia,
        "Python_REPL": python_repl,
        "Retrieval": st.session_state.retriever_tool
    }

    if st.session_state.bing_subscription_validity:
        search = BingSearchAPIWrapper()
    elif st.session_state.google_cse_id_validity:
        search = GoogleSearchAPIWrapper()
    else:
        search = None

    if search is not None:
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
            "queried. To search the internet, obtain your Bing Subscription "
            "Key [here](https://portal.azure.com/) or Google CSE ID "
            "[here](https://programmablesearchengine.google.com/about/), "
            "and enter it in the sidebar. Once entered, 'Search' will be "
            "displayed in the list of tools. Note also that PythonREPL from "
            "LangChain is still in the experimental phase, so caution is "
            "advised.</small>",
            unsafe_allow_html=True,
        )
    else:
        st.write(
            "<small>Tools are disabled when images are uploaded and "
            "queried. Note also that PythonREPL from LangChain is still "
            "in the experimental phase, so caution is advised.</small>",
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


def set_prompts(agent_type: Literal["Tool Calling", "ReAct"]) -> None:
    """
    Set chat and agent prompts for two different types of agents:
    Tool Calling and ReAct.
    """

    if agent_type == "Tool Calling":
        st.session_state.chat_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"{st.session_state.ai_role[0]} Your goal is to provide "
                "answers to human inquiries. Should the information not "
                "be available, inform the human explicitly that "
                "the answer could not be found."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        st.session_state.agent_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"{st.session_state.ai_role[0]} Your goal is to provide "
                "answers to human inquiries. You should specify the source "
                "of your answers, whether they are based on internet search "
                "results ('internet_search'), scientific articles from "
                "arxiv.org ('arxiv'), Wikipedia documents ('wikipedia'), "
                "uploaded documents ('retriever'), or your general knowledge. "
                "Use Markdown syntax and include relevant sources, such as "
                "links (URLs), following MLA format. Should the information "
                "not be available through internet searches, scientific "
                "articles, Wikipedia documents, uploaded documents, or your "
                "general knowledge, inform the human explicitly that the "
                "answer could not be found. Also, if you use 'python_repl' "
                "for computation, show the Python code that you run."
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
    else:
        st.session_state.chat_prompt = ChatPromptTemplate.from_template(
            f"{st.session_state.ai_role[0]} "
            "Your goal is to provide answers to human inquiries. "
            "Should the information not be available, inform the human "
            "explicitly that the answer could not be found.\n\n"
            "{chat_history}\n\nHuman: {input}\n\n"
            "AI: "
        )
        st.session_state.agent_prompt = ChatPromptTemplate.from_template(
            f"{st.session_state.ai_role[0]} "
            "Your goal is to provide answers to human inquiries. "
            "When giving your answers, tell the human what your response "
            "is based on and which tools you use. Use Markdown syntax "
            "and include relevant sources, such as links (URLs), following "
            "MLA format. Should the information not be available, inform "
            "the human explicitly that the answer could not be found.\n\n"
            "TOOLS:\n"
            "------\n\n"
            "You have access to the following tools:\n\n"
            "{tools}\n\n"
            "To use a tool, please use the following format:\n\n"
            "```\n"
            "Thought: Do I need to use a tool? Yes\n"
            "Action: the action to take, should be one of [{tool_names}]\n"
            "Action Input: the input to the action\n"
            "Observation: the result of the action\n"
            "```\n\n"
            "When you have a response to say to the Human, "
            "or if you do not need to use a tool, you MUST use "
            "the format:\n\n"
            "```\n"
            "Thought: Do I need to use a tool? No\n"
            "Final Answer: [your response here]\n"
            "```\n\n"
            "Begin!\n\n"
            "Previous conversation history:\n\n"
            "{chat_history}\n\n"
            "New input: {input}\n"
            "{agent_scratchpad}"
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

    # Play TTS
    if (
        st.session_state.model_type == "GPT Models from OpenAI"
        and st.session_state.audio_response is not None
    ):
        play_audio(st.session_state.audio_response)
        st.session_state.audio_response = None


def create_text(model: str) -> None:
    """
    Take an LLM as input and generate text based on user input
    by calling run_agent().
    """

    # initial system prompts
    general_role = "You are a helpful AI assistant."
    english_teacher = (
        "You are an AI English teacher who analyzes texts and corrects "
        "any grammatical issues if necessary."
    )
    translator = (
        "You are an AI translator who translates English into Korean "
        "and Korean into English."
    )
    coding_adviser = (
        "You are an AI expert in coding who provides advice on "
        "good coding styles."
    )
    science_assistant = "You are an AI science assistant."
    roles = (
        general_role, english_teacher, translator,
        coding_adviser, science_assistant
    )

    with st.sidebar:
        if st.session_state.model_type == "GPT Models from OpenAI":
            if not model.startswith("gemini-"):
                type_options = ("Tool Calling", "ReAct")
                st.write("")
                st.write("**Agent Type**")
                st.session_state.agent_type[0] = st.sidebar.radio(
                    label="Agent Type",
                    options=type_options,
                    index=type_options.index(st.session_state.agent_type[1]),
                    label_visibility="collapsed",
                )
            st.write("")
            st.write("**Text to Speech**")
            st.session_state.tts = st.radio(
                label="TTS",
                options=("Enabled", "Disabled", "Auto"),
                # horizontal=True,
                index=1,
                label_visibility="collapsed",
            )
        st.write("")
        st.write("**Temperature**")
        st.session_state.temperature[0] = st.slider(
            label="Temperature (higher $\Rightarrow$ more random)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature[1],
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

    # Reset or download the conversation
    left, right = st.columns(2)
    left.button(
        label="$~\:\,\,$Reset$~\:\,\,$",
        on_click=reset_conversation
    )
    right.download_button(
        label="Download",
        data=message_history_to_string(),
        file_name="conversation_with_agent.txt",
        mime="text/plain"
    )

    # Set the agent tools and prompt
    if model.startswith("gemini-"):
        agent_type = "ReAct"
    else:
        agent_type = st.session_state.agent_type[0]

    set_prompts(agent_type)
    tools = set_tools()

    image_urls = []
    with st.sidebar:
        image_urls = upload_image_files_return_urls()

    if st.session_state.model_type == "GPT Models from OpenAI":
        audio_input = input_from_mic()
        if audio_input is not None:
            query = audio_input
            st.session_state.prompt_exists = True
            st.session_state.mic_used = True

    # Use your keyboard
    text_input = st.chat_input(placeholder="Enter your query")

    if text_input:
        query = text_input.strip()
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
                temperature=st.session_state.temperature[0],
                agent_type=agent_type,
            )
            fig = plt.gcf()
            if fig and fig.get_axes():
                generated_image_url = fig_to_base64(fig)
                st.session_state.history[-1].additional_kwargs["image_urls"] = [
                    generated_image_url
                ]
        if (
            st.session_state.model_type == "GPT Models from OpenAI"
            and generated_text is not None
        ):
            # TTS under two conditions
            cond1 = st.session_state.tts == "Enabled"
            cond2 = st.session_state.tts == "Auto" and st.session_state.mic_used
            if cond1 or cond2:
                st.session_state.audio_response = perform_tts(generated_text)
            st.session_state.mic_used = False

        st.session_state.prompt_exists = False

        if generated_text is not None:
            st.session_state.uploader_key += 1
            st.rerun()


def create_image(model: str) -> None:
    """
    Generate image based on user description by calling openai_create_image().
    """

    # Set the image size
    with st.sidebar:
        st.write("")
        st.write("**Pixel size**")
        image_size = st.radio(
            label="$\\hspace{0.1em}\\texttt{Pixel size}$",
            options=("1024x1024", "1792x1024", "1024x1792"),
            # horizontal=True,
            index=0,
            label_visibility="collapsed",
        )

    st.write("")
    st.write("##### Description for your image")

    if st.session_state.image_url is not None:
        st.info(st.session_state.image_description)
        st.image(image=st.session_state.image_url, use_column_width=True)
    
    # Get an image description using the microphone
    if st.session_state.model_type == "GPT Models from OpenAI":
        audio_input = input_from_mic()
        if audio_input is not None:
            st.session_state.image_description = audio_input
            st.session_state.prompt_exists = True

    # Get an image description using the keyboard
    text_input = st.chat_input(
        placeholder="Enter a description for your image",
    )
    if text_input:
        st.session_state.image_description = text_input.strip()
        st.session_state.prompt_exists = True

    if st.session_state.prompt_exists:
        st.session_state.image_url = openai_create_image(
            st.session_state.image_description, model, image_size
        )
        st.session_state.prompt_exists = False
        if st.session_state.image_url is not None:
            st.rerun()


def create_text_image() -> None:
    """
    Generate text or image by using llm models "gpt-3.5-turbo",
    "gpt-4-turbo-preview", "gpt-4-vision-preview", or "dall-e-3",
    """

    st.write("## 📚 LangChain LLM Agent")

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
        )

        if choice_api == "Your keys":
            st.write("")
            st.write("**Model Type**")
            st.session_state.model_type = st.sidebar.radio(
                label="Model type",
                options=(
                    "GPT Models from OpenAI", "Gemini Models from Google"
                ),
                on_change=check_api_keys,
                label_visibility="collapsed",
            )
            st.write("")
            if st.session_state.model_type == "GPT Models from OpenAI":
                validity = "(Verified)" if st.session_state.ready else ""
                st.write(
                    "**OpenAI API Key** ",
                    f"<small>:blue[{validity}]</small>",
                    unsafe_allow_html=True
                )
                openai_api_key = st.text_input(
                    label="OpenAI API Key",
                    type="password",
                    on_change=check_api_keys,
                    label_visibility="collapsed",
                )
                if st.session_state.bing_subscription_validity:
                    validity = "(Verified)"
                else:
                    validity = ""
                st.write(
                    "**Bing Subscription Key** ",
                    f"<small>:blue[{validity}]</small>",
                    unsafe_allow_html=True
                )
                bing_subscription_key = st.text_input(
                    label="Bing Subscription Key",
                    type="password",
                    value="",
                    on_change=check_api_keys,
                    label_visibility="collapsed",
                )
            else:
                validity = "(Verified)" if st.session_state.ready else ""
                st.write(
                    "**Google API Key** ",
                    f"<small>:blue[{validity}]</small>",
                    unsafe_allow_html=True
                )
                google_api_key = st.text_input(
                    label="Google API Key",
                    type="password",
                    on_change=check_api_keys,
                    label_visibility="collapsed",
                )
                if st.session_state.google_cse_id_validity:
                    validity = "(Verified)"
                else:
                    validity = ""
                st.write(
                    "**Google CSE ID** ",
                    f"<small>:blue[{validity}]</small>",
                    unsafe_allow_html=True
                )
                google_cse_id = st.text_input(
                    label="Google CSE ID",
                    type="password",
                    value="",
                    on_change=check_api_keys,
                    label_visibility="collapsed",
                )
            authentication = True
        else:
            openai_api_key = st.secrets["OPENAI_API_KEY"]
            bing_subscription_key = st.secrets["BING_SUBSCRIPTION_KEY"]
            google_api_key = st.secrets["GOOGLE_API_KEY"]
            google_cse_id = st.secrets["GOOGLE_CSE_ID"]
            langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]
            stored_pin = st.secrets["USER_PIN"]
            st.write("**Password**")
            user_pin = st.text_input(
                label="Enter password", type="password", label_visibility="collapsed"
            )
            st.session_state.model_type = "GPT Models from OpenAI"
            authentication = user_pin == stored_pin

        os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"

    if authentication:
        if not st.session_state.ready:
            if choice_api == "My keys":
                os.environ["OPENAI_API_KEY"] = openai_api_key
                os.environ["BING_SUBSCRIPTION_KEY"] = bing_subscription_key
                st.session_state.bing_subscription_validity = True
                st.session_state.openai = OpenAI()
                os.environ["GOOGLE_API_KEY"] = google_api_key
                os.environ["GOOGLE_CSE_ID"] = google_cse_id
                st.session_state.google_cse_id_validity = True
                st.session_state.ready = True
                os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
                current_date = datetime.datetime.now().date()
                date_string = str(current_date)
                os.environ["LANGCHAIN_PROJECT"] = "llm_agent_" + date_string
            else:
                if st.session_state.model_type == "GPT Models from OpenAI":
                    if is_openai_api_key_valid(openai_api_key):
                        os.environ["OPENAI_API_KEY"] = openai_api_key
                        st.session_state.openai = OpenAI()
                        st.session_state.ready = True
                        if is_bing_subscription_key_valid(bing_subscription_key):
                            os.environ["BING_SUBSCRIPTION_KEY"] = bing_subscription_key
                            st.session_state.bing_subscription_validity = True
                        else:
                            st.session_state.bing_subscription_validity = False
                else:
                    if is_google_api_key_valid(google_api_key):
                        os.environ["GOOGLE_API_KEY"] = google_api_key
                        st.session_state.ready = True
                        if are_google_api_key_cse_id_valid(
                            google_api_key, google_cse_id
                        ):
                            os.environ["GOOGLE_CSE_ID"] = google_cse_id
                            st.session_state.google_cse_id_validity = True
                        else:
                            st.session_state.google_cse_id_validity = False

            if st.session_state.ready:
                st.rerun()
            else:
                st.info(
                    """
                    **Enter your OpenAI and Bing Subscription Keys in the sidebar**

                    Get an OpenAI API Key [here](https://platform.openai.com/api-keys)
                    and a Bing Subscription Key [here](https://portal.azure.com/).
                    You can also follow instructions on
                    [this site](https://levelup.gitconnected.com/api-tutorial-how-to-use-bing-web-search-api-in-python-4165d5592a7e)
                    to get your Bing Subscription Key. If you do not plan to search
                    the internet, there is no need to enter your Bing Subscription key.
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
                st.stop()
    else:
        st.info("**Enter the correct password in the sidebar**")
        st.stop()

    with st.sidebar:
        if choice_api == "My keys":
            st.write("")
            st.write("**LangSmith Tracing**")
            langsmith = st.radio(
                label="LangSmith Tracing",
                options=("On", "Off"),
                label_visibility="collapsed",
                index=1,
                horizontal=True
            )
            os.environ["LANGCHAIN_TRACING_V2"] = (
                "True" if langsmith == "On" else "False"
            )
        st.write("")
        st.write("**Model**")
        if choice_api == "My keys":
            model_options=(
                "gpt-3.5-turbo",
                "gpt-4o",
                "gemini-1.0-pro-latest",
                "gemini-1.5-pro-latest",
                "dall-e-3",
            )
        else:
            if st.session_state.model_type == "GPT Models from OpenAI":
                model_options=(
                    "gpt-3.5-turbo",
                    "gpt-4o",
                    "dall-e-3",
                )
            else:
                model_options=(
                    "gemini-1.0-pro-latest",
                    "gemini-1.5-pro-latest",
                )
        model = st.radio(
            label="Models",
            options=model_options,
            label_visibility="collapsed",
            index=1,
            on_change=switch_between_apps,
        )

    if model == "dall-e-3":
        create_image(model)
    else:
        create_text(model)

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
    create_text_image()
