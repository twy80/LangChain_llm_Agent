"""
ChatGPT & DALL·E using openai API (by T.-W. Yoon, Aug. 2023)
"""

import streamlit as st
import openai
import os, base64, requests, re
from io import BytesIO
from tempfile import NamedTemporaryFile
from audio_recorder_streamlit import audio_recorder
from PIL import Image, UnidentifiedImageError
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage  # AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.callbacks import StreamlitCallbackHandler


def initialize_session_state_variables():
    """
    This function initializes all the session state variables.
    """

    # variables for using OpenAI
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = None

    if "openai" not in st.session_state:
        st.session_state.openai = None

    # variables for chatbot
    if "ai_role" not in st.session_state:
        st.session_state.ai_role = 2 * ["You are a helpful assistant."]

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=st.session_state.ai_role[0])
        ]

    if "prompt_exists" not in st.session_state:
        st.session_state.prompt_exists = False

    if "human_enq" not in st.session_state:
        st.session_state.human_enq = []

    if "ai_resp" not in st.session_state:
        st.session_state.ai_resp = []

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

    # variables for RAG
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "sources" not in st.session_state:
        st.session_state.sources = None

    if "memory" not in st.session_state:
        st.session_state.memory = None


# This is for streaming on Streamlit
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def chat_complete(user_prompt, model="gpt-3.5-turbo", temperature=0.7):
    """
    This function generates text based on user input.

    Args:
        user_prompt (string): User input
        temperature (float): Value between 0 and 1. Defaults to 0.7
        model (string): "gpt-3.5-turbo" or "gpt-4".

    Return:
        generated text

    All the conversations are stored in st.session_state variables.
    """

    openai_llm = ChatOpenAI(
        openai_api_key=st.session_state.openai_api_key,
        temperature=temperature,
        model_name=model,
        streaming=True,
        callbacks=[StreamHandler(st.empty())]
    )

    # Add the user input to the messages
    st.session_state.messages.append(HumanMessage(content=user_prompt))
    try:
        response = openai_llm.invoke(st.session_state.messages)
        generated_text = response.content
    except Exception as e:
        generated_text = None
        st.error(f"An error occurred: {e}", icon="🚨")

    if generated_text is not None:
        # Add the generated output to the messages
        st.session_state.messages.append(response)

    return generated_text


def openai_create_image(description, model="dall-e-3", size="1024x1024"):
    """
    This function generates image based on user description.

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


def openai_query_image_url(image_url, query, model="gpt-4-vision-preview"):
    """
    This function answers the user's query about the given image from a URL.

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


def openai_query_uploaded_image(image_b64, query, model="gpt-4-vision-preview"):
    """
    This function answers the user's query about the uploaded image.

    Args:
        image_b64 (base64 encoded string): base64 encoded image
        query (string): the user's query
        model (string): default set to "gpt-4-vision-preview"

    Return:
        text as an answer to the user's query.
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {st.session_state.openai_api_key}"
    }

    payload = {
        "model": f"{model}",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{query}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    try:
        with st.spinner("AI is thinking..."):
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
        generated_text = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        generated_text = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return generated_text


def get_vector_store(uploaded_files):
    """
    This function takes a list of UploadedFile objects as input,
    and returns a FAISS vector store.
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
                file.write(uploaded_file.read())
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
            embeddings = OpenAIEmbeddings(
                openai_api_key=st.session_state.openai_api_key
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


def document_qna(query, vector_store, model="gpt-3.5-turbo"):
    """
    This function takes a user prompt, a vector store and a GPT model,
    and returns a response on the uploaded document along with sources.
    """

    if vector_store is not None:
        if st.session_state.memory is None:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

        openai_llm = ChatOpenAI(
            openai_api_key=st.session_state.openai_api_key,
            temperature=0,
            model_name=model,
            streaming=True,
            callbacks=[StreamlitCallbackHandler(st.empty())]
        )
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=openai_llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            # retriever=vector_store.as_retriever(search_type="mmr"),
            memory=st.session_state.memory,
            return_source_documents=True
        )

        try:
            # response to the query is given in the form
            # {"question": ..., "chat_history": [...], "answer": ...}.
            response = conversation_chain.invoke({"question": query})
            generated_text = response["answer"]
            source_documents = response["source_documents"]

        except Exception as e:
            generated_text, source_documents = None, None
            st.error(f"An error occurred: {e}", icon="🚨")
    else:
        generated_text, source_documents = None, None

    return generated_text, source_documents


def display_text_with_equations(text):
    # Replace inline LaTeX equation delimiters \\( ... \\) with $
    modified_text = text.replace("\\(", "$").replace("\\)", "$")

    # Replace block LaTeX equation delimiters \\[ ... \\] with $$
    modified_text = modified_text.replace("\\[", "$$").replace("\\]", "$$")

    # Use st.markdown to display the formatted text with equations
    st.markdown(modified_text)


def read_audio(audio_bytes):
    """
    This function reads audio bytes and returns the corresponding text.
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


def perform_tts(text):
    """
    This function takes text as input, performs text-to-speech (TTS),
    and returns an audio_response.
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
    This function takes an audio response (a bytes-like object)
    from TTS as input, and plays the audio.
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
    This function converts an image object from PIL to a base64
    encoded image, and returns the resulting encoded image.
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

    return base64_image


def shorten_image(image, max_pixels=1024):
    """
    This function takes an Image object as input, and shortens the image size
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
    This function determines whether text is a URL or not.
    """

    regex = r"(http|https)://([\w_-]+(?:\.[\w_-]+)+)(:\S*)?"
    p = re.compile(regex)
    match = p.match(text)
    if match:
        return True
    else:
        return False


def reset_conversation():
    st.session_state.messages = [
        SystemMessage(content=st.session_state.ai_role[0])
    ]
    st.session_state.ai_role[1] = st.session_state.ai_role[0]
    st.session_state.prompt_exists = False
    st.session_state.human_enq = []
    st.session_state.ai_resp = []
    st.session_state.temperature[1] = st.session_state.temperature[0]
    st.session_state.audio_response = None
    st.session_state.vector_store = None
    st.session_state.sources = None
    st.session_state.memory = None


def switch_between_apps():
    st.session_state.temperature[1] = st.session_state.temperature[0]
    st.session_state.image_source[1] = st.session_state.image_source[0]
    st.session_state.ai_role[1] = st.session_state.ai_role[0]


def enable_user_input():
    st.session_state.prompt_exists = True


def reset_qna_image():
    st.session_state.uploaded_image = None
    st.session_state.qna = {"question": "", "answer": ""}


def create_text(model):
    """
    This function generates text based on user input
    by calling chat_complete().

    model is set to "gpt-3.5-turbo" or "gpt-4".
    """

    # initial system prompts
    general_role = "You are a helpful assistant."
    english_teacher = "You are an English teacher who analyzes texts and corrects any grammatical issues if necessary."
    translator = "You are a translator who translates English into Korean and Korean into English."
    coding_adviser = "You are an expert in coding who provides advice on good coding styles."
    doc_analyzer = "You are an assistant analyzing the document(s) uploaded."
    roles = (general_role, english_teacher, translator, coding_adviser, doc_analyzer)

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
        # on_change=reset_conversation,
        label_visibility="collapsed",
    )

    if st.session_state.ai_role[0] != st.session_state.ai_role[1]:
        reset_conversation()

    if st.session_state.ai_role[0] == doc_analyzer:
        st.write("")
        left, right = st.columns([4, 7])
        left.write("##### Document(s) to ask about")
        right.write("Temperature is set to 0.")
        uploaded_files = st.file_uploader(
            label="Upload an article",
            type=["txt", "pdf", "docx"],
            accept_multiple_files=True,
            on_change=reset_conversation,
            label_visibility="collapsed",
        )
        button = "Create the vector store"
        if st.button(label=button) and st.session_state.vector_store is None:
            # Create the vector store.
            st.session_state.vector_store = get_vector_store(uploaded_files)

            if st.session_state.vector_store is not None:
                uploaded_file_names = [file.name for file in uploaded_files]
                file_names = ", ".join(uploaded_file_names)
                st.write(f"Vector store for :blue[[{file_names}]] is ready!")

    st.write("")
    left, right = st.columns([4, 7])
    left.write("##### Conversation with AI")
    right.write("Click on the mic icon and speak, or type text below.")

    # Print conversations
    for human, ai in zip(st.session_state.human_enq, st.session_state.ai_resp):
        with st.chat_message("human"):
            st.write(human)
        with st.chat_message("ai"):
            display_text_with_equations(ai)

    if st.session_state.ai_role[0] == doc_analyzer and st.session_state.sources is not None:
        with st.expander("Sources"):
            c1, c2, _ = st.columns(3)
            c1.write("Uploaded document(s):")
            columns = c2.columns(len(st.session_state.sources))
            for index, column in enumerate(columns):
                column.markdown(
                    f"{index + 1}\)",
                    help=st.session_state.sources[index].page_content
                )

    # Play TTS
    if st.session_state.audio_response is not None:
        play_audio(st.session_state.audio_response)
        st.session_state.audio_response = None

    # Reset the conversation
    st.button(label="Reset the conversation", on_click=reset_conversation)

    # Use your keyboard
    user_input = st.chat_input(
        placeholder="Enter your query",
        on_submit=enable_user_input,
        disabled=not uploaded_files if st.session_state.ai_role[0] == doc_analyzer else False
    )

    # Use your microphone
    audio_bytes = audio_recorder(
        pause_threshold=3.0, text="Speak", icon_size="2x",
        recording_color="#e87070", neutral_color="#6aa36f"        
    )

    if audio_bytes != st.session_state.audio_bytes:
        user_prompt = read_audio(audio_bytes)
        st.session_state.audio_bytes = audio_bytes
        if user_prompt is not None:
            st.session_state.prompt_exists = True
            st.session_state.mic_used = True
    elif user_input and st.session_state.prompt_exists:
        user_prompt = user_input.strip()

    if st.session_state.prompt_exists:
        with st.chat_message("human"):
            st.write(user_prompt)

        with st.chat_message("ai"):
            if st.session_state.ai_role[0] == doc_analyzer:
                generated_text, st.session_state.sources = document_qna(
                    user_prompt,
                    vector_store=st.session_state.vector_store,
                    model=model
                )
            else:  # General chatting
                generated_text = chat_complete(
                    user_prompt,
                    temperature=st.session_state.temperature[0],
                    model=model
                )

        if generated_text is not None:
            # TTS under two conditions
            cond1 = st.session_state.tts == "Enabled"
            cond2 = st.session_state.tts == "Auto" and st.session_state.mic_used
            if cond1 or cond2:
                st.session_state.audio_response = perform_tts(generated_text)

            st.session_state.mic_used = False
            st.session_state.human_enq.append(user_prompt)
            st.session_state.ai_resp.append(generated_text)

        st.session_state.prompt_exists = False

        if generated_text is not None:
            st.rerun()


def create_text_with_image(model):
    """
    This function responds to the user's query about the image
    from a URL or uploaded image.
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
                st.session_state.uploaded_image = shorten_image(image, 1024)
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
        audio_bytes = audio_recorder(
            pause_threshold=3.0, text="Speak", icon_size="2x",
            recording_color="#e87070", neutral_color="#6aa36f"        
        )
        if audio_bytes != st.session_state.audio_bytes:
            st.session_state.qna["question"] = read_audio(audio_bytes)
            st.session_state.audio_bytes = audio_bytes
            if st.session_state.qna["question"] is not None:
                st.session_state.prompt_exists = True

        # Use your keyboard
        query = st.chat_input(
            placeholder="Enter your query",
        )
        if query:
            st.session_state.qna["question"] = query
            st.session_state.prompt_exists = True

        if st.session_state.prompt_exists:
            if st.session_state.image_source[0] == "From URL":
                generated_text = openai_query_image_url(
                    image_url=st.session_state.uploaded_image,
                    query=st.session_state.qna["question"],
                    model=model
                )
            else:
                generated_text = openai_query_uploaded_image(
                    image_b64=image_to_base64(st.session_state.uploaded_image),
                    query=st.session_state.qna["question"],
                    model=model
                )

            st.session_state.prompt_exists = False
            if generated_text is not None:
                st.session_state.qna["answer"] = generated_text
                st.rerun()


def create_image(model):
    """
    This function generates image based on user description
    by calling openai_create_image().
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
    audio_bytes = audio_recorder(
        pause_threshold=3.0, text="Speak", icon_size="2x",
        recording_color="#e87070", neutral_color="#6aa36f"        
    )
    if audio_bytes != st.session_state.audio_bytes:
        st.session_state.image_description = read_audio(audio_bytes)
        st.session_state.audio_bytes = audio_bytes
        if st.session_state.image_description is not None:
            st.session_state.prompt_exists = True

    # Get an image description using the keyboard
    text_input = st.chat_input(
        placeholder="Enter a description for your image",
    )
    if text_input:
        st.session_state.image_description = text_input
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
    This main function generates text or image by calling
    openai_create_text() or openai_create_image(), respectively.
    """

    st.write("## 📚 ChatGPT (RAG)$\,$ &$\,$ DALL·E")

    # Initialize all the session state variables
    initialize_session_state_variables()

    with st.sidebar:
        st.write("")
        st.write("**API Key Selection**")
        choice_api = st.sidebar.radio(
            label="$\\hspace{0.25em}\\texttt{Choice of API}$",
            options=("Your key", "My key"),
            label_visibility="collapsed",
            horizontal=True,
        )

        if choice_api == "Your key":
            st.write("**Your API Key**")
            st.session_state.openai_api_key = st.text_input(
                label="$\\hspace{0.25em}\\texttt{Your OpenAI API Key}$",
                type="password",
                placeholder="sk-",
                label_visibility="collapsed",
            )
            authen = False if st.session_state.openai_api_key == "" else True
        else:
            st.session_state.openai_api_key = st.secrets["openai_api_key"]
            stored_pin = st.secrets["user_PIN"]
            st.write("**Password**")
            user_pin = st.text_input(
                label="Enter password", type="password", label_visibility="collapsed"
            )
            authen = user_pin == stored_pin

        st.session_state.openai = openai.OpenAI(
            api_key=st.session_state.openai_api_key
        )

        st.write("")
        st.write("**What to Generate**")
        option = st.radio(
            label="$\\hspace{0.25em}\\texttt{What to generate}$",
            options=(
                "Text (GPT 3.5)", "Text (GPT 4)",
                "Text with Image", "Image (DALL·E 3)"
            ),
            label_visibility="collapsed",
            # horizontal=True,
            on_change=switch_between_apps,
        )

    if authen:
        if option == "Text (GPT 3.5)":
            create_text("gpt-3.5-turbo-1106")
        elif option == "Text (GPT 4)":
            create_text("gpt-4-1106-preview")
        elif option == "Text with Image":
            create_text_with_image("gpt-4-vision-preview")
        else:
            create_image("dall-e-3")
    else:
        st.write("")
        if choice_api == "Your key":
            st.info(
                """
                **Enter your OpenAI API key in the sidebar**

                [Get an OpenAI API key](https://platform.openai.com/api-keys)
                The GPT-4 API can be accessed by those who have made
                a payment of $1 to OpenAI (a strange policy) at the time of
                writing this code.
                """
            )
        else:
            st.info("**Enter the correct password in the sidebar**")

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
