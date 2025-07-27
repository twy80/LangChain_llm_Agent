# [LangChain_LLM_Agent](https://langchain-llm-agent.streamlit.app/)

This app presents a tool calling agent. The supported models are Claude-3.5-Haiku,
Claude-3.5-Sonnet, Claude-Sonnet-4, and Claude-Opus-4 from Anthropic;
GPT-4o-mini and GPT-4o from OpenAI; and Gemini-2.5-Flash and Gemini-2.5-Pro from Google.
  
- For the OpenAI models such as 'GPT-4o', your OpenAI API key is needed. You can obtain
  an API key from https://platform.openai.com/account/api-keys.

- For Claude models such as 'Claude-4-Sonnet', your Anthropic API key is needed.
  You can obtain an API key from https://console.anthropic.com/settings/keys.

- For Gemini models such as 'Gemini-2.5-Pro', your Google API key is needed.
  You can obtain an API key from https://aistudio.google.com/app/apikey.

- For searching the internet, use a Google CSE ID obtained from
  [here](https://programmablesearchengine.google.com/about/) along with
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

This project uses `uv` for package management.

1.  [Install `uv`](https://github.com/astral-sh/uv).
2.  Run the app with:
```python
uv run streamlit run src/app.py
```
[![Exploring the App: A Visual Guide](files/Streamlit_Agent_App.png)](https://youtu.be/6uD480u49lU)
