"""
ChatGPT & DALL·E using openai API (by T.-W. Yoon, Aug. 2023)
"""

import streamlit as st
import openai
from audio_recorder_streamlit import audio_recorder
import os, io, base64
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


def openai_create_text(user_prompt, temperature=0.7, model="gpt-3.5-turbo"):
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

    # Add the user input to the prompt
    st.session_state.prompt.append({"role": "user", "content": user_prompt})
    try:
        with st.spinner("AI is thinking..."):
            response = st.session_state.client.chat.completions.create(
                model=model,
                messages=st.session_state.prompt,
                temperature=temperature,
            )
        generated_text = response.choices[0].message.content
    except Exception as e:
        generated_text = None
        st.error(f"An error occurred: {e}", icon="🚨")

    if generated_text:
        # Add the generated output to the prompt
        st.session_state.prompt.append(
            {"role": "assistant", "content": generated_text}
        )

    return generated_text


def openai_create_image(description, size="1024x1024"):
    """
    This function generates image based on user description.

    Args:
        description (string): User description
        size (string): Pixel size of the generated image

    The resulting image is plotted.
    """

    if description:
        try:
            with st.spinner("AI is generating..."):
                response = st.session_state.client.images.generate(
                    model="dall-e-3",
                    prompt=description,
                    size=size,
                    quality="standard",
                    n=1,
                )
            image_url = response.data[0].url
            st.image(image=image_url, use_column_width=True)
        except Exception as e:
            st.error(f"An error occurred: {e}", icon="🚨")

    return None


def get_vector_store(uploaded_file):
    """
    This function takes an UploadedFile object as input,
    and returns a FAISS vector store.
    """

    uploaded_document = "files/uploaded_document"

    if uploaded_file is None:
        return None
    else:
        file_bytes = io.BytesIO(uploaded_file.read())
        with open(uploaded_document, "wb") as f:
            f.write(file_bytes.read())

        # Determine the loader based on the file extension.
        if uploaded_file.name.lower().endswith(".pdf"):
            loader = PyPDFLoader(uploaded_document)
        elif uploaded_file.name.lower().endswith(".txt"):
            loader = TextLoader(uploaded_document)
        elif uploaded_file.name.lower().endswith(".docx"):
            loader = Docx2txtLoader(uploaded_document)
        else:
            st.error("Please load a file in pdf or txt", icon="🚨")
            return None

        # Load the document using the selected loader.
        document = loader.load()

        try:
            with st.spinner("Vector store in preparation..."):
                # Split the loaded text into smaller chunks for processing.
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    # separators=["\n", "\n\n", "(?<=\. )", "", " "],
                )

                doc = text_splitter.split_documents(document)

                # Create a FAISS vector database.
                embeddings = OpenAIEmbeddings(
                    openai_api_key=st.session_state.openai_api_key
                )
                vector_store = FAISS.from_documents(doc, embeddings)
        except Exception as e:
            vector_store = None
            st.error(f"An error occurred: {e}", icon="🚨")

        return vector_store


def get_conversation_chain(vector_store, temperature=0, model="gpt-3.5-turbo"):
    """
    This function takes a vector store, a numerical value between 0 and 1 for
    temperature and a llm model as input, and returns a conversational chain.
    """

    openai_llm = ChatOpenAI(
        openai_api_key=st.session_state.openai_api_key,
        temperature=temperature,
        model_name=model
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True,
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=openai_llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
    )

    return conversation_chain


def openai_doc_answer(user_prompt, conversation):
    """
    This function takes a user prompt and a conversation object as input,
    and generates text on the uploaded document.
    """

    if conversation is not None:
        try:
            with st.spinner("AI is thinking..."):
                # response to the query is given in the form
                # {"question": ..., "chat_history": [...], "answer": ...}.
                response = conversation({"question": user_prompt})
                generated_text = response["answer"]

        except Exception as e:
            generated_text = None
            st.error(f"An error occurred: {e}", icon="🚨")
    else:
        generated_text = None

    return generated_text


def read_audio(audio_bytes):
    """
    This function reads audio bytes and returns the corresponding text.
    """
    try:
        audio_data = io.BytesIO(audio_bytes)
        audio_data.name = "recorded_audio.wav"  # dummy name

        transcript = st.session_state.client.audio.transcriptions.create(
            model="whisper-1", file=audio_data
        )
        text = transcript.text
    except Exception:
        text = None

    return text


def perform_tts(text):
    """
    This function takes text as input, performs text-to-speech (TTS),
    and returns an audio_response.
    """

    try:
        with st.spinner("TTS in progress..."):
            audio_response = st.session_state.client.audio.speech.create(
                model="tts-1",
                voice="shimmer",
                input=text,
            )
    except Exception:
        audio_response = None

    return audio_response


def autoplay_audio(file_path):
    """
    This function takes an audio file as input,
    and automatically plays the audio file.
    """

    # Get the file extension from the file path
    _, ext = os.path.splitext(file_path)

    # Determine the MIME type based on the file extension
    mime_type = f"audio/{ext.lower()[1:]}"  # Remove the leading dot from the extension

    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()

        md = f"""
            <audio controls autoplay style="width: 100%;">
            <source src="data:{mime_type};base64,{b64}" type="{mime_type}">
            </audio>
            """

        st.markdown(md, unsafe_allow_html=True)


def reset_conversation():
    st.session_state.prompt = [
        {"role": "system", "content": st.session_state.prev_ai_role}
    ]
    st.session_state.prompt_exists = False
    st.session_state.human_enq = []
    st.session_state.ai_resp = []
    st.session_state.initial_temp = st.session_state.temp_value
    st.session_state.play_audio = False
    st.session_state.vector_store = None
    st.session_state.conversation = None


def switch_between_apps():
    st.session_state.initial_temp = st.session_state.temp_value


def enable_user_input():
    st.session_state.prompt_exists = True


def create_text(model):
    """
    This function geneates text based on user input
    by calling openai_create_text().

    model is set to "gpt-3.5-turbo" or "gpt-4".
    """

    # Audio file for TTS
    text_audio_file = "files/output_text.wav"

    # initial system prompts
    general_role = "You are a helpful assistant."
    english_teacher = "You are an English teacher who analyzes texts and corrects any grammatical issues if necessary."
    translator = "You are a translator who translates English into Korean and Korean into English."
    coding_adviser = "You are an expert in coding who provides advice on good coding styles."
    doc_analyzer = "You are an assistant analyzing the document uploaded."
    roles = (general_role, english_teacher, translator, coding_adviser, doc_analyzer)

    if "prev_ai_role" not in st.session_state:
        st.session_state.prev_ai_role = general_role

    if "prompt" not in st.session_state:
        st.session_state.prompt = [
            {"role": "system", "content": st.session_state.prev_ai_role}
        ]

    if "prompt_exists" not in st.session_state:
        st.session_state.prompt_exists = False

    if "human_enq" not in st.session_state:
        st.session_state.human_enq = []

    if "ai_resp" not in st.session_state:
        st.session_state.ai_resp = []

    if "initial_temp" not in st.session_state:
        st.session_state.initial_temp = 0.7

    if "prev_audio_bytes" not in st.session_state:
        st.session_state.prev_audio_bytes = None

    if "mic_used" not in st.session_state:
        st.session_state.mic_used = False

    if "play_audio" not in st.session_state:
        st.session_state.play_audio = False

    # session_state variables for RAG
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    with st.sidebar:
        st.write("")
        st.write("**Text to Speech**")
        st.session_state.tts = st.radio(
            label="$\\hspace{0.08em}\\texttt{TTS}$",
            options=("Enabled", "Disabled", "Auto"),
            # horizontal=True,
            index=2,
            label_visibility="collapsed",
        )
        st.write("")
        st.write("**Temperature**")
        st.session_state.temp_value = st.slider(
            label="$\\hspace{0.08em}\\texttt{Temperature}\,$ (higher $\Rightarrow$ more random)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.initial_temp,
            step=0.1,
            format="%.1f",
            label_visibility="collapsed",
        )
        st.write("(Higher $\Rightarrow$ More random)")

    st.write("")
    st.write("##### Message to AI")
    ai_role = st.selectbox(
        label="AI's role",
        options=roles,
        index=roles.index(st.session_state.prev_ai_role),
        label_visibility="collapsed",
    )

    if ai_role != st.session_state.prev_ai_role:
        st.session_state.prev_ai_role = ai_role
        reset_conversation()

    if ai_role == doc_analyzer:
        st.write("")
        left, right = st.columns([4, 7])
        left.write("##### Document to ask about")
        right.write("Temperature is set to 0.")
        uploaded_file = st.file_uploader(
            label="Upload an article",
            type=["txt", "pdf", "docx"],
            accept_multiple_files=False,
            on_change=reset_conversation,
            label_visibility="collapsed",
        )
        if st.session_state.vector_store is None:
            # Create the vector store.
            st.session_state.vector_store = get_vector_store(uploaded_file)

            if st.session_state.vector_store is not None:
                st.write(f"Vector store for :blue[[{uploaded_file.name}]] is ready!")
                # st.session_state.vector_store_ready = True
                # Create the conversation chain.
                st.session_state.conversation = get_conversation_chain(
                    st.session_state.vector_store,
                    temperature=0.0,
                    model=model,
                )

    st.write("")
    left, right = st.columns([4, 7])
    left.write("##### Conversation with AI")
    right.write("Click on the mic icon and speak, or type text below.")

    # Print conversations
    for human, ai in zip(st.session_state.human_enq, st.session_state.ai_resp):
        with st.chat_message("human"):
            st.write(human)
        with st.chat_message("ai"):
            st.write(ai)

    # Play TTS
    if st.session_state.play_audio:
        autoplay_audio(text_audio_file)
        st.session_state.play_audio = False

    # Reset the conversation
    st.button(label="Reset the conversation", on_click=reset_conversation)

    # Use your keyboard
    user_input = st.chat_input(
        placeholder="Enter your query",
        on_submit=enable_user_input,
        disabled=not uploaded_file if ai_role == doc_analyzer else False,
    )

    # Use your microphone
    audio_bytes = audio_recorder(
        pause_threshold=3.0,
        # sample_rate=sr,
        text="Speak",
        recording_color="#e87070",
        neutral_color="#6aa36f",
        # icon_name="user",
        icon_size="2x",
    )

    if audio_bytes != st.session_state.prev_audio_bytes:
        user_prompt = read_audio(audio_bytes)
        if user_prompt is not None:
            st.session_state.prompt_exists = True
            st.session_state.mic_used = True
        st.session_state.prev_audio_bytes = audio_bytes
    elif user_input and st.session_state.prompt_exists:
        user_prompt = user_input.strip()

    if st.session_state.prompt_exists:
        with st.chat_message("human"):
            st.write(user_prompt)

        if ai_role == doc_analyzer:  # RAG (when there is a document uploaded)
            generated_text = openai_doc_answer(
                user_prompt, st.session_state.conversation
            )
        else:  # General chatting
            generated_text = openai_create_text(
                user_prompt, temperature=st.session_state.temp_value, model=model
            )

        if generated_text is not None:
            # with st.chat_message("ai"):
            #     st.write(generated_text)
            # TTS under two conditions
            cond1 = st.session_state.tts == "Enabled"
            cond2 = st.session_state.tts == "Auto" and st.session_state.mic_used
            if cond1 or cond2:
                audio_response = perform_tts(generated_text)
                if audio_response is not None:
                    audio_response.stream_to_file(text_audio_file)
                    st.session_state.play_audio = True

            st.session_state.mic_used = False
            st.session_state.human_enq.append(user_prompt)
            st.session_state.ai_resp.append(generated_text)

        st.session_state.prompt_exists = False

        if generated_text is not None:
            st.rerun()


def create_image():
    """
    This function geneates image based on user description
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

    # Get the image description from the user
    st.write("")
    st.write(f"##### Description for your image")
    description = st.text_area(
        label="$\\hspace{0.1em}\\texttt{Description for your image}\,$ (in $\,$English)",
        # value="",
        label_visibility="collapsed",
    )

    left, _ = st.columns(2)  # To show the results below the button
    left.button(
        label="Generate",
        on_click=openai_create_image(description, image_size)
    )


def openai_create():
    """
    This main function generates text or image by calling
    openai_create_text() or openai_create_image(), respectively.
    """

    if "client" not in st.session_state:
        st.session_state.client = None

    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = None

    st.write("## 🎭 ChatGPT & DALL·E")

    with st.sidebar:
        st.write("")
        st.write("**API Key Selection**")
        choice_api = st.sidebar.radio(
            label="$\\hspace{0.25em}\\texttt{Choic of API}$",
            options=("Your key", "My key"),
            label_visibility="collapsed",
            horizontal=True,
            on_change=reset_conversation,
        )

        if choice_api == "Your key":
            st.write("**Your API Key**")
            st.session_state.openai_api_key = st.text_input(
                label="$\\hspace{0.25em}\\texttt{Your OpenAI API Key}$",
                type="password",
                label_visibility="collapsed",
            )
            # st.write("You can obtain an API key from https://beta.openai.com")
            authen = True
        else:
            st.session_state.openai_api_key = st.secrets["openai_api_key"]
            stored_pin = st.secrets["user_PIN"]
            st.write("**Password**")
            user_pin = st.text_input(
                label="Enter password", type="password", label_visibility="collapsed"
            )
            authen = user_pin == stored_pin

        st.session_state.client = openai.OpenAI(
            api_key=st.session_state.openai_api_key
        )

        st.write("")
        st.write("**What to Generate**")
        option = st.sidebar.radio(
            label="$\\hspace{0.25em}\\texttt{What to generate}$",
            options=("Text (GPT 3.5)", "Text (GPT 4)", "Image (DALL·E 3)"),
            label_visibility="collapsed",
            # horizontal=True,
            on_change=switch_between_apps,
        )

    if not authen:
        st.error("**Incorrect password. Please try again.**", icon="🚨")
    else:
        if option == "Text (GPT 3.5)":
            create_text("gpt-3.5-turbo")
        elif option == "Text (GPT 4)":
            create_text("gpt-4")
        else:
            create_image()

    with st.sidebar:
        st.write("---")
        st.write(
            "<small>**T.-W. Yoon**, Aug. 2023  \n</small>",
            "<small>[TWY's Playground](https://twy-playground.streamlit.app/)  \n</small>",
            "<small>[Differential equations](https://diff-eqn.streamlit.app/)</small>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    openai_create()
