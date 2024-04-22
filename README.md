# [LangChain_OpenAI_Agent](https://langchain-openai-agent.streamlit.app/)

* This app generates text and images using OpenAI's APIs and LangChain.
  
  - Your OpenAI API key is required to run this code. You can obtain an API key
    from https://platform.openai.com/account/api-keys.

  - Text outputs are generated using large language models such as "gpt-3.5-turbo",
    "gpt-4-turbo", or "gpt-4-vision-preview", and images are generated using
    "dall-e-3."

  - Temperature can be set by the user

  - Voice recognition and Text-To-Speech (TTS) functionalities are supported.

  - Recording of the user's voice is stopped when there is no input for 3 seconds.
  
  - Supported tools include Tavily Search, ArXiv, Retrieval (RAG), and python_REPL.
    * To use Tavily Search, you need a Tavily API key that can be obtained
      [here](https://app.tavily.com/).
    * PythonREPL from LangChain is still experimental, and therefore caution is
      needed. Users are also advised to choose gpt-4-turbo-preview with Python REPL.

  - Tracing LLM messages is possible using LangSmith if you download the source code
    and run it on your machine or server.  For this, you need a
    LangChain API key that can be obtained [here](https://smith.langchain.com/settings).

* This page is written in Python using the Streamlit framework.

## Usage
```python
streamlit run LangChain_OpenAI_Agent.py
```
[![Exploring the App: A Visual Guide](files/Streamlit_Agent_App.png)](https://youtu.be/ux7ux8YXnMI)
