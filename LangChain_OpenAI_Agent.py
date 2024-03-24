"""
LangChain Agents (by T.-W. Yoon, Mar. 2024)
"""

import streamlit as st
import os, base64, re, requests, uuid, datetime
from io import BytesIO
from tempfile import NamedTemporaryFile
from audio_recorder_streamlit import audio_recorder
from PIL import Image, UnidentifiedImageError
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_community.callbacks import StreamlitCallbackHandler
from tavily import TavilyClient


def initialize_session_state_variables():
    """
    Initialize all the session state variables.
    """

    # variables for using OpenAI
    if "ready" not in st.session_state:
        st.session_state.ready = False

    if "openai" not in st.session_state:
        st.session_state.openai = None

    if "message_history" not in st.session_state:
        st.session_state.message_history = ChatMessageHistory()

    # variables for chatbot
    if "ai_role" not in st.session_state:
        st.session_state.ai_role = 2 * ["You are a helpful assistant."]

    if "prompt_exists" not in st.session_state:
        st.session_state.prompt_exists = False

    if "temperature" not in st.session_state:
        st.session_state.temperature = [0.7, 0.7]

    # variables for audio and image
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

    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None

    if "qna" not in st.session_state:
        st.session_state.qna = {"question": "", "answer": ""}

    if "image_source" not in st.session_state:
        st.session_state.image_source = 2 * ["From URL"]

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    # variables for tools
    if "tools" not in st.session_state:
        st.session_state.tools = []

    if "tavily_api_validity" not in st.session_state:
        st.session_state.tavily_api_validity = False

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "vector_store_message" not in st.session_state:
        st.session_state.vector_store_message = None

    if "retriever_tool" not in st.session_state:
        st.session_state.retriever_tool = None


def is_openai_api_key_valid(openai_api_key):
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


def is_tavily_api_key_valid(tavily_api_key):
    """
    Return True if the given Tavily Search API key is valid.
    """

    try:
        tavily = TavilyClient(api_key=tavily_api_key)
        tavily.search(query="where can I get a Tavily search API key?")
    except:
        return False

    return True


def is_langchain_api_key_valid(langchain_api_key):
    """
    Return True if the given LangChain API key is valid.
    """

    run_id = str(uuid.uuid4())

    response = requests.post(
        "https://api.smith.langchain.com/runs",
        json={
            "id": run_id,
            "name": "Test Run",
            "run_type": "chain",
            # "start_time": datetime.datetime.utcnow().isoformat(),
            "inputs": {"text": "Foo"},
        },
        headers={"x-api-key": langchain_api_key},
    )
    requests.patch(
        f"https://api.smith.langchain.com/runs/{run_id}",
        json={
            "outputs": {"my_output": "Bar"},
            # "end_time": datetime.datetime.utcnow().isoformat(),
        },
        headers={"x-api-key": langchain_api_key},
    )

    return response.status_code == 202


def check_api_keys():
    # Unset this flag to check the validity of the OpenAI API key
    st.session_state.ready = False


def run_agent(query, model, tools=[], temperature=0.7):
    """
    Generate text based on user queries.

    Args:
        query (string): User's query
        model (string): LLM like "gpt-3.5-turbo"
        tools (list): sublist of [tavily_search, retrieval, python_repl]
        temperature (float): Value between 0 and 1. Defaults to 0.7

    Return:
        generated text

    The chat prompt and message history are stored in
    st.session_state variables.
    """

    llm = ChatOpenAI(
        temperature=temperature,
        model_name=model,
        streaming=True,
        callbacks=[StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)]
    )
    if tools:
        agent = create_openai_functions_agent(
            llm, tools, st.session_state.agent_prompt
        )
        agent_executor = AgentExecutor(
            agent=agent, tools=tools, max_iterations=5, verbose=False,
            agent_executor_kwargs={"handle_parsing_errors": True},
        )
    else:
        agent_executor = st.session_state.chat_prompt | llm

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: st.session_state.message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    try:
        response = agent_with_chat_history.invoke(
            {"input": query},
            config={"configurable": {"session_id": "chat"}},
        )
        generated_text = response['output'] if tools else response.content
    except Exception as e:
        generated_text = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return generated_text


def openai_create_image(description, model="dall-e-3", size="1024x1024"):
    """
    Generate image based on user description.

    Args:
        description (string): User description
        model (string): Default set to "dall-e-3"
        size (string): Pixel size of the generated image

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


def openai_query_image(image_url, query, model="gpt-4-vision-preview"):
    """
    Answer the user's query about the given image from a URL.

    Args:
        image_url (string): URL of the image
        query (string): the user's query
        model (string): default set to "gpt-4-vision-preview"

    Return:
        text as an answer to the user's query.
    """

    try:
        with st.spinner("AI is thinking..."):
            response = st.session_state.openai.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{query}"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"{image_url}"},
                            },
                        ],
                    },
                ],
                max_tokens=300,
            )
        generated_text = response.choices[0].message.content
    except Exception as e:
        generated_text = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return generated_text


def get_vector_store(uploaded_files):
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
            embeddings = OpenAIEmbeddings()
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


def get_retriever():
    """
    Upload document(s), create a vector store, and return a retriever tool.
    """

    st.write("")
    st.write("##### Document(s) to ask about")
    uploaded_files = st.file_uploader(
        label="Upload an article",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="upload" + str(st.session_state.uploader_key),
    )

    if st.button(label="Create the vector store"):
        # Create the vector store.
        st.session_state.vector_store = get_vector_store(uploaded_files)

        if st.session_state.vector_store is not None:
            uploaded_file_names = [file.name for file in uploaded_files]
            file_names = ", ".join(uploaded_file_names)
            st.session_state.vector_store_message = (
                f"Vector store for :blue[[{file_names}]] is ready!"
            )
            retriever = st.session_state.vector_store.as_retriever()
            retriever_tool = create_retriever_tool(
                retriever,
                name="retriever",
                description=(
                    "Search for information about the uploaded documents. "
                    "For any questions about the documents, you must use "
                    "this tool!"
                ),
            )
            st.session_state.uploader_key += 1
            return retriever_tool
        else:
            return None
    else:
        # Return the retriever_tool already prepared
        return st.session_state.retriever_tool


def display_text_with_equations(text):
    # Replace inline LaTeX equation delimiters \\( ... \\) with $
    modified_text = text.replace("\\(", "$").replace("\\)", "$")

    # Replace block LaTeX equation delimiters \\[ ... \\] with $$
    modified_text = modified_text.replace("\\[", "$$").replace("\\]", "$$")

    # Use st.markdown to display the formatted text with equations
    st.markdown(modified_text)


def read_audio(audio_bytes):
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


def input_from_mic():
    """
    Convert audio input from mic to text and return it.
    If there is no audio input, None is returned.
    """

    audio_bytes = audio_recorder(
        pause_threshold=3.0, text="Speak", icon_size="2x",
        recording_color="#e87070", neutral_color="#6aa36f"        
    )

    if audio_bytes == st.session_state.audio_bytes or audio_bytes is None:
        return None
    else:
        st.session_state.audio_bytes = audio_bytes
        return read_audio(audio_bytes)


def perform_tts(text):
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


def play_audio(audio_response):
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


def image_to_base64(image):
    """
    Convert an image object from PIL to a base64 encoded image,
    and return the resulting encoded image in the form of a URL.
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


def shorten_image(image, max_pixels=1024):
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


def is_url(text):
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


def reset_conversation():
    st.session_state.message_history.clear()
    st.session_state.ai_role[1] = st.session_state.ai_role[0]
    st.session_state.prompt_exists = False
    st.session_state.human_enq = []
    st.session_state.ai_resp = []
    st.session_state.temperature[1] = st.session_state.temperature[0]
    st.session_state.audio_response = None
    st.session_state.vector_store = None
    st.session_state.vector_store_message = None
    st.session_state.tools = []
    st.session_state.retriever_tool = None


def switch_between_apps():
    st.session_state.temperature[1] = st.session_state.temperature[0]
    st.session_state.image_source[1] = st.session_state.image_source[0]
    st.session_state.ai_role[1] = st.session_state.ai_role[0]


def reset_qna_image():
    st.session_state.uploaded_image = None
    st.session_state.qna = {"question": "", "answer": ""}


def create_text(model):
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
    math_assistant = "You are a math assistant."
    roles = (
        general_role, english_teacher, translator, coding_adviser, math_assistant
    )

    with st.sidebar:
        st.write("")
        st.write("**Text to Speech**")
        st.session_state.tts = st.radio(
            label="$\\hspace{0.08em}\\texttt{TTS}$",
            options=("Enabled", "Disabled", "Auto"),
            # horizontal=True,
            index=1,
            label_visibility="collapsed",
        )
        st.write("")
        st.write("**Temperature**")
        st.session_state.temperature[0] = st.slider(
            label="$\\hspace{0.08em}\\texttt{Temperature}\,$ (higher $\Rightarrow$ more random)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature[1],
            step=0.1,
            format="%.1f",
            label_visibility="collapsed",
        )
        st.write("(Higher $\Rightarrow$ More random)")

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
    st.write("**Tools**")
    tool_options = ["tavily_search", "python_repl", "retrieval"]
    selected_tools = st.multiselect(
        label="assistant tools",
        options=tool_options,
        default=st.session_state.tools,
        label_visibility="collapsed",
    )
    if selected_tools != st.session_state.tools:
        st.session_state.tools = selected_tools
        st.rerun()

    if st.session_state.tavily_api_validity:
        tavily_search = TavilySearchResults()
    else:
        tavily_search = None

    python_repl = PythonREPLTool()
    if "python_repl" in selected_tools:
        st.write(
            "<small>PythonREPL from LangChain is still experimental, "
            "and therefore may lead to incorrect results. "
            "Use with caution.</small>",
            unsafe_allow_html=True,
        )

    if "retrieval" in selected_tools:
        st.session_state.retriever_tool = get_retriever()
        if st.session_state.vector_store_message:
            st.write(st.session_state.vector_store_message)

    # Tools to be used with the llm
    tool_dictionary = {
        "tavily_search": tavily_search,
        "python_repl": python_repl,
        "retrieval": st.session_state.retriever_tool
    }
    tools = []
    for tool_name in selected_tools:
        tool = tool_dictionary[tool_name]
        if tool is not None:
            tools.append(tool)

    # Prompts for the agents
    st.session_state.chat_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"{st.session_state.ai_role[0]} Your goal is to provide answers "
            "to human inquiries. Should the information not be available, "
            "please inform the human explicitly that the answer could "
            "not be found."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    st.session_state.agent_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"{st.session_state.ai_role[0]} Your goal is to provide answers "
            "to human inquiries. You must inform the human of the basis of "
            "your answers, i.e., whether your answers are based on search "
            "results from the internet (if 'tavily_search' is used), "
            "information from the uploaded documents (if 'retriever' is "
            "used), or your general knowledge. Use markdown syntax and "
            "include relevant sources, like URLs, following MLA format. "
            "Should the information not be available through internet "
            "searches, the uploaded documents, or your general knowledge, "
            "please inform the human explicitly that the answer could "
            "not be found. Also if you use 'python_repl' to perform "
            "computation, show the Python code that you run. When showing "
            "the Python code, encapsulate the code in Markdown format, "
            "e.g.,\n\n"
            "```python\n"
            "....\n"
            "```"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    st.write("")
    left, right = st.columns([4, 7])
    left.write("##### Conversation with AI")
    right.write("Click on the mic icon and speak, or type text below.")

    # Print conversations
    for message in st.session_state.message_history.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("human"):
                st.write(message.content)
        else:
            with st.chat_message("ai"):
                display_text_with_equations(message.content)

    # Play TTS
    if st.session_state.audio_response is not None:
        play_audio(st.session_state.audio_response)
        st.session_state.audio_response = None

    # Reset the conversation
    st.button(label="Reset the conversation", on_click=reset_conversation)

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
                query,
                model,
                tools=tools,
                temperature=st.session_state.temperature[0],
            )

        if generated_text is not None:
            # TTS under two conditions
            cond1 = st.session_state.tts == "Enabled"
            cond2 = st.session_state.tts == "Auto" and st.session_state.mic_used
            if cond1 or cond2:
                st.session_state.audio_response = perform_tts(generated_text)
            st.session_state.mic_used = False

        st.session_state.prompt_exists = False

        if generated_text is not None:
            st.rerun()


def create_text_with_image(model):
    """
    Respond to the user's query about the image from a URL or uploaded image.
    """

    with st.sidebar:
        sources = ("From URL", "Uploaded")
        st.write("")
        st.write("**Image selection**")
        st.session_state.image_source[0] = st.radio(
            label="Image selection",
            options=sources,
            index=sources.index(st.session_state.image_source[1]),
            label_visibility="collapsed",
        )

    st.write("")
    st.write("##### Image to ask about")
    st.write("")

    if st.session_state.image_source[0] == "From URL":
        # Enter a URL
        st.write("###### :blue[Enter the URL of your image]")

        image_url = st.text_input(
            label="URL of the image", label_visibility="collapsed",
            on_change=reset_qna_image
        )
        if image_url:
            if is_url(image_url):
                st.session_state.uploaded_image = image_url
            else:
                st.error("Enter a proper URL", icon="🚨")
    else:
        # Upload an image file
        st.write("###### :blue[Upload your image]")

        image_file = st.file_uploader(
            label="High resolution images will be resized.",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            on_change=reset_qna_image,
        )
        if image_file is not None:
            # Process the uploaded image file
            try:
                image = Image.open(image_file)
                image = shorten_image(image, 1024)
                st.session_state.uploaded_image = image_to_base64(image)
            except UnidentifiedImageError as e:
                st.error(f"An error occurred: {e}", icon="🚨")

    # Capture the user's query and provide a response if the image is ready
    if st.session_state.uploaded_image:
        st.image(image=st.session_state.uploaded_image, use_column_width=True)

        # Print query & answer
        if st.session_state.qna["question"] and st.session_state.qna["answer"]:
            with st.chat_message("human"):
                st.write(st.session_state.qna["question"])
            with st.chat_message("ai"):
                display_text_with_equations(st.session_state.qna["answer"])

        # Use your microphone
        audio_input = input_from_mic()
        if audio_input is not None:
            st.session_state.qna["question"] = audio_input
            st.session_state.prompt_exists = True

        # Use your keyboard
        text_input = st.chat_input(
            placeholder="Enter your query",
        )
        if text_input:
            st.session_state.qna["question"] = text_input.strip()
            st.session_state.prompt_exists = True

        if st.session_state.prompt_exists:
            generated_text = openai_query_image(
                image_url=st.session_state.uploaded_image,
                query=st.session_state.qna["question"],
                model=model
            )

            st.session_state.prompt_exists = False
            if generated_text is not None:
                st.session_state.qna["answer"] = generated_text
                st.rerun()


def create_image(model):
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


def create_text_image():
    """
    Generate text or image by using llm models "gpt-3.5-turbo-0125",
    "gpt-4-0125-preview", "gpt-4-vision-preview", or "dall-e-3",
    """

    st.write("## 📚 LangChain OpenAI Agent")

    # Initialize all the session state variables
    initialize_session_state_variables()

    with st.sidebar:
        st.write("")
        st.write("**API Key Selection**")
        choice_api = st.sidebar.radio(
            label="$\\hspace{0.25em}\\texttt{Choice of API}$",
            options=("Your keys", "My keys"),
            label_visibility="collapsed",
            horizontal=True,
        )

        if choice_api == "Your keys":
            st.write("**OpenAI API Key**")
            openai_api_key = st.text_input(
                label="$\\textsf{Your OPenAI API Key}$",
                type="password",
                placeholder="sk-",
                on_change=check_api_keys,
                label_visibility="collapsed",
            )
            st.write("**Tavily Search API Key**")
            tavily_api_key = st.text_input(
                label="$\\textsf{Your Tavily API Key}$",
                type="password",
                placeholder="tvly-",
                on_change=check_api_keys,
                label_visibility="collapsed",
            )
            st.write("**LangChain API Key**")
            langchain_api_key = st.text_input(
                label="$\\textsf{Your LangChain API Key}$",
                type="password",
                placeholder="ls__",
                on_change=check_api_keys,
                label_visibility="collapsed",
            )
            authentication = True
        else:
            openai_api_key = st.secrets["OPENAI_API_KEY"]
            tavily_api_key = st.secrets["TAVILY_API_KEY"]
            langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]
            stored_pin = st.secrets["USER_PIN"]
            st.write("**Password**")
            user_pin = st.text_input(
                label="Enter password", type="password", label_visibility="collapsed"
            )
            authentication = user_pin == stored_pin

    if authentication:
        if not st.session_state.ready:
            if is_openai_api_key_valid(openai_api_key):
                os.environ["OPENAI_API_KEY"] = openai_api_key
                st.session_state.openai = OpenAI()
                st.session_state.ready = True

                if is_tavily_api_key_valid(tavily_api_key):
                    os.environ["TAVILY_API_KEY"] = tavily_api_key
                    st.session_state.tavily_api_validity = True
                else:
                    st.session_state.tavily_api_validity = False

                if choice_api == "My keys" or is_langchain_api_key_valid(langchain_api_key):
                    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
                    os.environ["LANGCHAIN_TRACING_V2"] = "true"
                    # os.environ["LANGCHAIN_TRACING_V2"] = "false"
                    current_date = datetime.datetime.now().date()
                    date_string = str(current_date)
                    os.environ["LANGCHAIN_PROJECT"] = "llm_agent_" + date_string
                else:
                    os.environ["LANGCHAIN_TRACING_V2"] = "false"
            else:
                st.info(
                    """
                    **Enter your OpenAI, Tavily Search, and LangChain API keys
                    in the sidebar**

                    Get an OpenAI API key [here](https://platform.openai.com/api-keys),
                    a Tavily Search API key [here](https://app.tavily.com/), and
                    a LangChain API key [here](https://smith.langchain.com/settings).
                    Your LangChain API key is used to trace LLM messages at
                    https://smith.langchain.com.
                    
                    If you do not want to use any search tool, there is no need
                    to enter your Tavily Search API key. Likewise, your LangChain API
                    key is not needed if you are not tracing LLM messages.
                    """
                )
                st.stop()
    else:
        st.info("**Enter the correct password in the sidebar**")
        st.stop()

    with st.sidebar:
        st.write("")
        st.write("**Models**")
        model = st.radio(
            label="$\\textsf{Models}$",
            options=(
                "gpt-3.5-turbo-0125",
                "gpt-4-0125-preview",
                "gpt-4-vision-preview",
                "dall-e-3",
            ),
            label_visibility="collapsed",
            on_change=switch_between_apps,
        )

    if model in ("gpt-3.5-turbo-0125", "gpt-4-0125-preview"):
        create_text(model)
    elif model == "gpt-4-vision-preview":
        create_text_with_image(model)
    else:
        create_image(model)

    with st.sidebar:
        st.write("---")
        st.write(
            "<small>**T.-W. Yoon**, Aug. 2023  \n</small>",
            "<small>[OpenAI Assistants](https://assistants.streamlit.app/)  \n</small>",
            "<small>[TWY's Playground](https://twy-playground.streamlit.app/)  \n</small>",
            "<small>[Differential equations](https://diff-eqn.streamlit.app/)</small>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    create_text_image()
