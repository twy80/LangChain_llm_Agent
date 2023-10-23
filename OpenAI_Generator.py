"""
ChatGPT & DALL·E using openai API (by T.-W. Yoon, Aug. 2023)
"""

import openai
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from langdetect import detect
from gtts import gTTS
import base64
# from io import BytesIO
# import clipboard


def openai_create_text(
        user_prompt,
        temperature=0.7,
        model="gpt-3.5-turbo"
    ):
    """
    This function generates text based on user input.

    Args:
        user_prompt (string): User input
        temperature (float): Value between 0 and 2. Defaults to 0.7
        model (string): "gpt-3.5-turbo" or "gpt-4".

    The results are stored in st.session_state variables.
    """

    if user_prompt:
        # Add the user input to the prompt
        st.session_state.prompt.append(
            {"role": "user", "content": user_prompt}
        )
        try:
            with st.spinner("AI is thinking..."):
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=st.session_state.prompt,
                    temperature=temperature,
                    # max_tokens=4096,
                    # top_p=1,
                    # frequency_penalty=0,
                    # presence_penalty=0.6,
                )
            generated_text = response.choices[0].message.content
        except openai.error.OpenAIError as e:
            generated_text = None
            st.error(f"An error occurred: {e}", icon="🚨")

        if generated_text:
            # Add the generated output to the prompt
            st.session_state.prompt.append(
                {"role": "assistant", "content": generated_text}
            )
            st.session_state.generated_text = generated_text
    else:
        st.session_state.generated_text = None
        return None

    return None


def openai_create_image(description, size="512x512"):
    """
    This function generates image based on user description.

    Args:
        description (string): User description
        size (string): Pixel size of the generated image

    The resulting image is plotted.
    """

    if description.strip() == "":
        return None

    try:
        with st.spinner("AI is generating..."):
            response = openai.Image.create(
                prompt=description,
                n=1,
                size=size
            )
        image_url = response['data'][0]['url']
        st.image(
            image=image_url,
            use_column_width=True
        )
    except openai.error.OpenAIError as e:
        st.error(f"An error occurred: {e}", icon="🚨")

    return None


def reset_conversation():
    # to_clipboard = ""
    # for (human, ai) in zip(st.session_state.human_enq, st.session_state.ai_resp):
    #     to_clipboard += "\nHuman: " + human + "\n"
    #     to_clipboard += "\nAI: " + ai + "\n"
    # clipboard.copy(to_clipboard)
    st.session_state.generated_text = None
    st.session_state.prompt = [
        {"role": "system", "content": st.session_state.prev_ai_role}
    ]
    st.session_state.prompt_exists = False
    st.session_state.human_enq = []
    st.session_state.ai_resp = []
    st.session_state.initial_temp = st.session_state.temp_value


def switch_between_apps():
    st.session_state.initial_temp = st.session_state.temp_value
    st.session_state.prev_audio_bytes = None


def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay style="width: 100%;">
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )


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
    close_friend = "You are a close friend of mine."
    roles = (general_role, english_teacher, translator, coding_adviser, close_friend)

    if "generated_text" not in st.session_state:
        st.session_state.generated_text = None

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

    with st.sidebar:
        st.write("")
        st.write("**Text to Speech**")
        st.session_state.tts = st.radio(
            "$\\hspace{0.08em}\\texttt{TTS}$",
            ('Enabled', 'Disabled', 'Auto'),
            # horizontal=True,
            index=2, label_visibility="collapsed"
        )
        st.write("")
        st.write("**Temperature**")
        st.session_state.temp_value = st.slider(
            label="$\\hspace{0.08em}\\texttt{Temperature}\,$ (higher $\Rightarrow$ more random)",
            min_value=0.0, max_value=2.0, value=st.session_state.initial_temp,
            step=0.1, format="%.1f",
            label_visibility="collapsed"
        )
        st.write("(Higher $\Rightarrow$ More random)")

    st.write("")
    st.write("##### Message to AI")
    ai_role = st.selectbox(
        "AI's role", roles, index=roles.index(st.session_state.prev_ai_role),
        label_visibility="collapsed"
    )

    if ai_role != st.session_state.prev_ai_role:
        st.session_state.prev_ai_role = ai_role
        reset_conversation()

    st.write("")
    left, right = st.columns([4, 7])
    left.write("##### Conversations with AI")
    right.write("Click on the mic icon and speak, or type text below.")

    for (human, ai) in zip(st.session_state.human_enq, st.session_state.ai_resp):
        with st.chat_message("human"):
            st.write(human)
        with st.chat_message("ai"):
            st.write(ai)

    # Use your keyboard
    user_input = st.chat_input(placeholder="Enter your query")

    # Use your microphone
    audio_bytes = audio_recorder(
        pause_threshold=2.0,
        # sample_rate=sr,
        text="Speak",
        recording_color="#e87070",
        neutral_color="#6aa36f",
        # icon_name="user",
        icon_size="2x",
    )

    if audio_bytes != st.session_state.prev_audio_bytes:
        try:
            audio_file = "files/recorded_audio.wav"
            with open(audio_file, "wb") as recorded_file:
                recorded_file.write(audio_bytes)
            audio_data = open(audio_file, "rb")

            # audio_data = BytesIO(audio_bytes)
            # audio_data.name = "recorded_audio.wav"

            transcript = openai.Audio.transcribe("whisper-1", audio_data)
            user_prompt = transcript['text']
            st.session_state.prompt_exists = True
            st.session_state.mic_used = True
        except Exception as e:
            st.error(f"An error occurred: {e}", icon="🚨")
        st.session_state.prev_audio_bytes = audio_bytes
    elif user_input:
        user_prompt = user_input.strip()
        st.session_state.prompt_exists = True

    if st.session_state.prompt_exists:
        openai_create_text(
            user_prompt,
            temperature=st.session_state.temp_value,
            model=model
        )
        if st.session_state.generated_text:
            # with st.chat_message("human"):
            #     st.write(user_prompt)
            # with st.chat_message("ai"):
            #     st.write(st.session_state.generated_text)

            # TTS under two conditions
            cond1 = st.session_state.tts == 'Enabled'
            cond2 = st.session_state.tts == 'Auto' and st.session_state.mic_used
            if cond1 or cond2:
                try:
                    with st.spinner("TTS in progress..."):
                        lang = detect(st.session_state.generated_text)
                        tts = gTTS(text=st.session_state.generated_text, lang=lang)
                        tts.save(text_audio_file)
                        # text_audio_file = BytesIO()
                        # tts.write_to_fp(text_audio_file)
                    # autoplay_audio(text_audio_file)
                    # st.audio(text_audio_file.getvalue())
                    st.session_state.play_audio = True
                except Exception as e:
                    st.error(f"An error occurred: {e}", icon="🚨")

            st.session_state.mic_used = False
            st.session_state.human_enq.append(user_prompt)
            st.session_state.ai_resp.append(st.session_state.generated_text)
            # clipboard.copy(st.session_state.generated_text)

        st.session_state.prompt_exists = False
        st.rerun()

    if st.session_state.play_audio:
        autoplay_audio(text_audio_file)
        st.session_state.play_audio = False

    st.button(
        label="Reset",
        on_click=reset_conversation
    )


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
            "$\\hspace{0.1em}\\texttt{Pixel size}$",
            ('256x256', '512x512', '1024x1024'),
            # horizontal=True,
            index=1,
            label_visibility="collapsed"
        )

    # Get the image description from the user
    st.write("")
    st.write(f"##### Description for your image (in English)")
    description = st.text_area(
        label="$\\hspace{0.1em}\\texttt{Description for your image}\,$ (in $\,$English)",
        # value="",
        label_visibility="collapsed"
    )

    left, _ = st.columns(2) # To show the results below the button
    left.button(
        label="Generate",
        on_click=openai_create_image(description, image_size)
    )


def openai_create():
    """
    This main function generates text or image by calling
    openai_create_text() or openai_create_image(), respectively.
    """
    st.write("## 🎭 ChatGPT & DALL·E")

    with st.sidebar:
        st.write("")
        st.write("**Choic of API key**")
        choice_api = st.sidebar.radio(
            "$\\hspace{0.25em}\\texttt{Choic of API}$",
            ('Your key', 'My key'),
            label_visibility="collapsed",
            horizontal=True,
            on_change=reset_conversation
        )

        if choice_api == 'Your key':
            st.write("**Your API Key**")
            openai.api_key = st.text_input(
                label="$\\hspace{0.25em}\\texttt{Your OpenAI API Key}$",
                type="password",
                label_visibility="collapsed"
            )
            # st.write("You can obtain an API key from https://beta.openai.com")
            authen = True
        else:
            openai.api_key = st.secrets["OPENAI_API_KEY"]
            stored_pin = st.secrets["USER_PIN"]
            st.write("**Password**")
            user_pin = st.text_input(
                label="Enter password", type="password", label_visibility="collapsed"
            )
            authen = user_pin == stored_pin

        st.write("")
        st.write("**What to Generate**")
        option = st.sidebar.radio(
            "$\\hspace{0.25em}\\texttt{What to generate}$",
            ('Text (GPT 3.5)', 'Text (GPT 4)', 'Image (DALL·E)'),
            label_visibility="collapsed",
            # horizontal=True,
            on_change=switch_between_apps
        )

    if not authen:
        st.error("**Incorrect password. Please try again.**", icon="🚨")
    else:
        if option == 'Text (GPT 3.5)':
            create_text("gpt-3.5-turbo")
        elif option == 'Text (GPT 4)':
            create_text("gpt-4")
        else:
            create_image()

    with st.sidebar:
        st.write("---")
        st.write(
          "<small>**T.-W. Yoon**, Aug. 2023  \n</small>",
          "<small>[TWY's Playground](https://twy-playground.streamlit.app/)  \n</small>",
          "<small>[Differential equations](https://diff-eqn.streamlit.app/)</small>",
          unsafe_allow_html=True
        )


if __name__ == "__main__":
    openai_create()
