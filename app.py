"""
app.py - a simple Streamlit chat interface for the RAG tool.

Each visitor pastes their own Anthropic key and gets their own private set
of documents that lives only for their session. Nothing is shared between
users, and no key is stored.

Run locally with:   streamlit run app.py
"""

import os
import tempfile

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

import rag

load_dotenv()  # lets you optionally keep your own key in a local .env file

st.set_page_config(page_title="Chat with your documents", page_icon="📄")
st.title("📄 Chat with your documents")


# The local embedding model is loaded once and shared (it is read-only).
@st.cache_resource(show_spinner=False)
def get_embedding_fn():
    return embedding_functions.DefaultEmbeddingFunction()


def new_collection():
    """A fresh, in-memory, per-session collection. Not shared, not saved."""
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(
        name="session", embedding_function=get_embedding_fn()
    )


if "collection" not in st.session_state:
    st.session_state.collection = new_collection()
if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources):
    with st.expander("Sources used"):
        if not sources:
            st.write("No sources.")
        for i, c in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {c['source']}**")
            st.write(c["text"])


# ---- Sidebar: key + documents --------------------------------------
with st.sidebar:
    st.header("Your Anthropic key")
    key_input = st.text_input(
        "Paste your API key",
        type="password",
        help="Get one at console.anthropic.com. It is used only for your "
        "session and is not stored.",
    )
    api_key = key_input or os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        st.caption("Key loaded for this session.")
    else:
        st.caption("No key yet. Paste one above to start asking questions.")

    st.divider()
    st.header("Documents")
    uploaded = st.file_uploader(
        "Upload PDF, TXT or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Add to knowledge base", disabled=not uploaded):
        for uf in uploaded:
            suffix = os.path.splitext(uf.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name
            try:
                n = rag.add_document(st.session_state.collection, tmp_path, uf.name)
                st.success(f"Added {uf.name} ({n} chunks).")
            except Exception as e:
                st.error(f"{uf.name}: {e}")
            finally:
                os.remove(tmp_path)

    st.caption(f"Chunks in knowledge base: {st.session_state.collection.count()}")
    if st.button("Clear knowledge base"):
        st.session_state.collection = new_collection()
        st.success("Knowledge base cleared.")
        st.rerun()

# ---- Chat ----------------------------------------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("sources"):
            show_sources(m["sources"])

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not api_key:
            reply, sources = "Please paste your Anthropic API key in the sidebar first.", []
            st.markdown(reply)
        elif st.session_state.collection.count() == 0:
            reply, sources = "Please add at least one document first (use the sidebar).", []
            st.markdown(reply)
        else:
            with st.spinner("Searching the documents and asking Claude..."):
                sources = rag.retrieve(st.session_state.collection, question)
                try:
                    reply = rag.answer(api_key, question, sources)
                except Exception as e:
                    reply = (
                        "Something went wrong calling Claude. Check that your "
                        f"API key is correct and has credit.\n\nDetails: {e}"
                    )
            st.markdown(reply)
            if sources:
                show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )
