"""
Utility script for setting up and managing various AI tools and document
processing capabilities. Provides functionality for:
- Creating vector stores from uploaded documents (PDF, TXT, DOCX)
- Setting up document retrieval tools
- Configuring multiple AI tools including search, ArXiv, Wikipedia, and
  Python REPL
- Managing tool selection and initialization through Streamlit interface

The script integrates with OpenAI and Google AI embeddings, and uses FAISS
for vector storage.
"""

import os, time
from functools import partial
from tempfile import NamedTemporaryFile
from typing import List, Annotated, Optional

import streamlit as st
from langchain.tools import Tool, tool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_experimental.utilities import PythonREPL
from pydantic import BaseModel, Field
from streamlit.runtime.uploaded_file_manager import UploadedFile


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
            if st.session_state.openai_key_validity:
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


@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = PythonREPL().run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


def set_tools() -> List[Tool]:
    """
    Set and return the tools for the agent. Tools that can be selected
    are internet_search, arxiv, wikipedia, python_repl, and retrieval.
    For searching the internet, a Google CSE ID is required together
    with a Google API key.
    """

    class MySearchToolInput(BaseModel):
        query: str = Field(description="search query to look up")

    arxiv = load_tools(["arxiv"])[0]
    wikipedia = load_tools(["wikipedia"])[0]
    # python_repl = PythonREPLTool()

    tool_options = ["ArXiv", "Wikipedia", "Python_REPL"]
    tool_dictionary = {
        "ArXiv": arxiv,
        "Wikipedia": wikipedia,
        "Python_REPL": python_repl,
    }

    if st.session_state.openai_key_validity or st.session_state.google_key_validity:
        tool_options.insert(0, "Retrieval")
        tool_dictionary["Retrieval"] = st.session_state.retriever_tool

    if st.session_state.google_cse_id_validity:
        time.sleep(0.5)
        search = GoogleSearchAPIWrapper()
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
            "queried. To search the internet, obtain your Google CSE ID "
            "[here](https://programmablesearchengine.google.com/about/). "
            "Once the valid id is entered, 'Search' will be displayed "
            "in the list of tools.</small>",
            unsafe_allow_html=True,
        )
    else:
        st.write(
            "<small>Tools are disabled when images are uploaded and "
            "queried.</small>",
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
