# [ChatGPT & DALL·E](https://chatgpt-dalle.streamlit.app/)

* This app generates text and images using OpenAI's APIs
  
  - Text outputs are generated using large language models such as "gpt-3.5-turbo",
    "gpt-4-turbo", or "gpt-4-vision-preview", and images are generated using
    "dall-e-3", all from OpenAI.

  - Temperature can be set by the user

  - voice recognition and Text-To-Speech (TTS) functionalities are supported by
    utilizing APIs from OpenAI.

  - Recording of the user's voice is stopped when there is no input for 3 seconds
  
  - RAG (Retrieval Augmented Generation) for an external document is implemented
    by using langchain functions.

* This page is written in python using the Streamlit framework.

* Your OpenAI API key is required to run this code. You can obtain an API key
  from https://platform.openai.com/account/api-keys. If, for some reason, you
  do not want to obtain an API key but would still like to try this code,
  you will need to request a password.

## Usage
[![Exploring the App: A Visual Guide]([files/Streamlit_LLM_App.png](https://youtu.be/GOnGXtYIX0Q))]("Exploring the App: A Visual Guide")
```python
streamlit run OpenAI_Generator.py
```
