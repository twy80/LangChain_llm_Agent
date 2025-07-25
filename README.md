# [LangChain_LLM_Agent](https://langchain-llm-agent.streamlit.app/)

* This app presents a tool calling agent. The supported models are claude-3-5-haiku,
  claude-3-5-sonnet, claude-sonnet-4, & claude-opus-4 from Anthropic, gpt-4o-mini
  & gpt-4o from OpenAI, and gemini-2.5-flash & gemini-2.5-pro from Google.
  
  - For the OpenAI models such as 'gpt-4o', your OpenAI API key is needed. You can obtain
    an API key from https://platform.openai.com/account/api-keys.

  - For Claude models such as 'claude-4-sonnet', your Anthropic API key is needed.
    You can obtain an API key from https://console.anthropic.com/settings/keys.

  - For Gemini models such as 'gemini-2.5-pro', your Google API key is needed.
    You can obtain an API key from https://aistudio.google.com/app/apikey.

  - For searching the internet, obtain Google CSE ID
    [here](https://programmablesearchengine.google.com/about/) together with
    your Google API key.

  - Temperature can be set by the user.

  - In addition to the search tool from Google, ArXiv, Wikipedia,
    Retrieval (RAG), and pythonREPL are supported.

  - Tracing LLM messages is possible using LangSmith if you download the source code
    and run it on your machine or server.  For this, you need a
    LangChain API key that can be obtained [here](https://smith.langchain.com/settings).

  - When running the code on your machine or server, you can use st.secrets to keep and
    fetch your API keys as environments variables. For such secrets management, see
    [this page](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

## Usage
```python
streamlit run LangChain_llm_Agent.py
```
[![Exploring the App: A Visual Guide](files/Streamlit_Agent_App.png)](https://youtu.be/6uD480u49lU)
