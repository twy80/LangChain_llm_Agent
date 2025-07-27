"""
Utility functions for a Streamlit chat application, providing tools for:
- Message history management and serialization
- Text and equation display formatting
- Audio transcription using Whisper
- Image processing and base64 conversion
- File handling and conversation loading

These utilities support the main chat interface by handling various
data transformations and I/O operations.
"""

import base64
import json
from io import BytesIO
from typing import List, Dict, Union, Optional, Any

import streamlit as st
from PIL import Image, UnidentifiedImageError
from langchain.schema import HumanMessage, AIMessage
from matplotlib.figure import Figure
from openai import OpenAI
from streamlit.runtime.uploaded_file_manager import UploadedFile


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


def print_char_list(char_list: List[str]) -> str:
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
