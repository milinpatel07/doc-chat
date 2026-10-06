"""
app.py - Streamlit interface for Document chat, a research tool for asking
questions about your own documents.

Developed by Milin Patel.

Two modes:
- Shared (default, e.g. the public link): every visitor brings their own
  key and gets an isolated, in-memory set of documents. Nothing is saved.
- Library mode (LOCAL_MODE=1 in your .env, on your own computer): documents,
  folders, reports and chats are kept between restarts.

Run with:   streamlit run app.py
"""

import csv
import datetime
import hashlib
import html
import io
import json
import os
import random
import tempfile
import time
import uuid

import chromadb
import streamlit as st
import streamlit.components.v1 as components
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

import rag

try:
    import fitz  # PyMuPDF, for page previews
except ImportError:
    fitz = None

load_dotenv()
LOCAL_MODE = os.environ.get("LOCAL_MODE", "").strip() == "1"
LIB_DIR = "library"
FILES_DIR = os.path.join(LIB_DIR, "files")
TEXT_DIR = os.path.join(LIB_DIR, "text")
REPORT_DIR = os.path.join(LIB_DIR, "overviews")
REGISTRY = os.path.join(LIB_DIR, "registry.json")
LOG_DIR = "chat_logs"
UNSORTED = "Unsorted"
LARGE_ONLINE = 15000
REPORT_KINDS = ("overview", "ideas", "mindmap")
VERBS = ["Scrutinizing", "Cross-referencing", "Sifting through the pages",
         "Consulting the index", "Tracing the definitions", "Connecting the ideas"]
VERDICT_STYLE = {"SUPPORTED": ("#067647", "Supported"),
                 "PARTLY SUPPORTED": ("#B54708", "Partly supported"),
                 "CONTRADICTED": ("#B42318", "Contradicted"),
                 "NOT FOUND": ("#6B7280", "Not found in the documents"),
                 "UNCLEAR": ("#6B7280", "Unclear")}

st.set_page_config(page_title="Document chat", page_icon="📄", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [data-testid="stAppViewContainer"], [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    #MainMenu, footer {visibility: hidden;}
    [data-testid="stHeader"] {background: transparent;}
    .block-container {max-width: 820px; padding-top: 2.2rem; padding-bottom: 5rem;}
    .app-title {font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em; color: #1A1D29; margin: 0;}
    .app-sub {color: #6B7280; font-size: 0.98rem; line-height: 1.5; margin: 0.4rem 0 1.2rem; max-width: 62ch;}
    .stButton>button, .stDownloadButton>button {border-radius: 8px; font-weight: 600;}
    [data-testid="stSidebar"] {background: #F7F8FA; border-right: 1px solid #ECEEF2;}
    [data-testid="stSidebar"] h2 {font-size: 0.95rem; font-weight: 700; color: #1A1D29;}
    .status-ok {color: #067647; font-weight: 600; font-size: 0.86rem;}
    .status-wait {color: #B54708; font-weight: 600; font-size: 0.86rem;}
    .getstarted {border: 1px solid #ECEEF2; border-radius: 12px; padding: 1.2rem 1.4rem; background: #FBFBFD;}
    .getstarted h4 {margin: 0 0 0.5rem; font-weight: 700; color: #1A1D29; font-size: 1.02rem;}
    .getstarted ol {margin: 0; padding-left: 1.15rem; color: #374151; line-height: 1.7;}
    .credit {color: #9CA3AF; font-size: 0.8rem;}
    .verdict {font-weight: 700; font-size: 0.9rem;}
    .claim {border-left: 3px solid #E4E7EC; padding: 0.2rem 0 0.2rem 0.8rem; margin: 0.9rem 0 0.3rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---- Storage -------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_fn():
    return embedding_functions.DefaultEmbeddingFunction()


@st.cache_resource(show_spinner=False)
def get_memory_client():
    return chromadb.EphemeralClient()


@st.cache_resource(show_spinner=False)
def get_library_collection():
    for d in (FILES_DIR, TEXT_DIR, REPORT_DIR):
        os.makedirs(d, exist_ok=True)
    client = chromadb.PersistentClient(path=LIB_DIR)
    return client.get_or_create_collection(name="library", embedding_function=get_embedding_fn())


def new_session_collection():
    # A unique name per visitor: in-memory Chroma clients share one store
    # inside a process, so a fixed name would mix different users' documents.
    return get_memory_client().get_or_create_collection(
        name="s" + uuid.uuid4().hex, embedding_function=get_embedding_fn()
    )


def docs_in_collection(col):
    data = col.get(include=["metadatas"])
    sections, pages = {}, {}
    for m in data["metadatas"]:
        src = m.get("source")
        sections[src] = sections.get(src, 0) + 1
        if m.get("page"):
            pages.setdefault(src, set()).add(m["page"])
    folders = {}
    if os.path.exists(REGISTRY):
        with open(REGISTRY, encoding="utf-8") as f:
            folders = json.load(f)
    return {s: {"sections": n, "pages": len(pages.get(s, ())), "folder": folders.get(s, UNSORTED)}
            for s, n in sections.items()}


ss = st.session_state
if "collection" not in ss:
    ss.collection = get_library_collection() if LOCAL_MODE else new_session_collection()
if "docs" not in ss:
    ss.docs = docs_in_collection(ss.collection) if LOCAL_MODE else {}
for key, default in [("messages", []), ("files", {}), ("texts", {}), ("reports", {}),
                     ("kw_version", 0), ("kw_built", -1), ("draft_results", None),
                     ("matrix", None)]:
    ss.setdefault(key, default)
ss.setdefault("log_name", datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))


def save_registry():
    if LOCAL_MODE:
        with open(REGISTRY, "w", encoding="utf-8") as f:
            json.dump({n: d.get("folder", UNSORTED) for n, d in ss.docs.items()}, f, indent=1)


def keyword_index():
    if ss.kw_built != ss.kw_version or "kw_index" not in ss:
        data = ss.collection.get(include=["documents", "metadatas"])
        ss.kw_index = (rag.KeywordIndex(data["ids"], data["documents"], data["metadatas"])
                       if data["ids"] else None)
        ss.kw_built = ss.kw_version
    return ss.kw_index


def _stored(folder, name, ext):
    return os.path.join(folder, hashlib.sha1(name.encode("utf-8")).hexdigest() + ext)


def store_units(name, units):
    ss.texts[name] = units
    if LOCAL_MODE:
        with open(_stored(TEXT_DIR, name, ".json"), "w", encoding="utf-8") as f:
            json.dump(units, f)


def get_units(name):
    """Page texts of a document, in order. Rebuilt from the indexed
    sections for documents added by an older version."""
    if name in ss.texts:
        return ss.texts[name]
    path = _stored(TEXT_DIR, name, ".json")
    if LOCAL_MODE and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            ss.texts[name] = [tuple(u) for u in json.load(f)]
        return ss.texts[name]
    data = ss.collection.get(where={"source": name}, include=["documents", "metadatas"])
    pairs = sorted(zip(data["metadatas"], data["documents"]), key=lambda x: x[0].get("page") or 0)
    return [(m.get("page") or 0, d) for m, d in pairs]


def _report_key(kind, name):
    return name if kind == "overview" else f"{kind}|{name}"   # keeps 0.5 overviews readable


def get_report(kind, name):
    key = _report_key(kind, name)
    if key in ss.reports:
        return ss.reports[key]
    path = _stored(REPORT_DIR, key, ".md")
    if LOCAL_MODE and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            ss.reports[key] = f.read()
        return ss.reports[key]
    return None


def save_report(kind, name, md):
    key = _report_key(kind, name)
    ss.reports[key] = md
    if LOCAL_MODE:
        with open(_stored(REPORT_DIR, key, ".md"), "w", encoding="utf-8") as f:
            f.write(md)


def forget_document(name):
    ss.collection.delete(where={"source": name})
    for store in (ss.docs, ss.files, ss.texts):
        store.pop(name, None)
    for kind in REPORT_KINDS:
        ss.reports.pop(_report_key(kind, name), None)
    if LOCAL_MODE:
        paths = [os.path.join(FILES_DIR, name), _stored(TEXT_DIR, name, ".json")]
        paths += [_stored(REPORT_DIR, _report_key(k, name), ".md") for k in REPORT_KINDS]
        for path in paths:
            if os.path.exists(path):
                os.remove(path)
    save_registry()
    ss.kw_version += 1


def pdf_entry(name):
    if name in ss.files:
        return ss.files[name]
    path = os.path.join(FILES_DIR, name)
    if LOCAL_MODE and os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        ss.files[name] = (data, hashlib.sha1(data).hexdigest())
        return ss.files[name]
    return None


@st.cache_data(show_spinner=False, max_entries=64)
def render_page(file_hash, page, _data):
    doc = fitz.open(stream=_data, filetype="pdf")
    return doc[page - 1].get_pixmap(dpi=110).tobytes("png")


# ---- Chats ---------------------------------------------------------
def source_label(c):
    return f"{c['source']}, p. {c['page']}" if c.get("page") else c["source"]


def transcript_md():
    now = datetime.datetime.now()
    lines = [
        f"# {rag.APP_NAME} transcript", "",
        f"- Tool: {rag.APP_NAME} {rag.APP_VERSION}, developed by Milin Patel",
        f"- Exported: {now:%Y-%m-%d %H:%M}",
        f"- Documents: {', '.join(ss.docs) or 'none'}", "",
        "Answers were written by a language model from the documents listed. "
        "Check every statement against the cited pages before using it.", "",
    ]
    for m in ss.messages:
        if m["role"] == "user":
            lines += ["## Question", "", m["content"], ""]
            continue
        meta = m.get("meta") or {}
        if meta:
            detail = (f", minimum match {meta['min_match']:.2f}"
                      if meta.get("min_match") is not None else "")
            lines += [f"_Answered {meta.get('time', '')} by {meta.get('model', '')} via "
                      f"{meta.get('api', '')}, mode: {meta.get('mode', '')}{detail}_", ""]
        for key in ("note", "read_note"):
            if m.get(key):
                lines += [f"_{m[key]}_", ""]
        lines += [m["content"], ""]
        if m.get("check"):
            lines += [f"_{m['check']['line']}_", ""]
        if m.get("review"):
            lines += ["_Second check: every statement supported._" if m["review"]["ok"]
                      else f"_Second check flagged:_\n{m['review']['details']}", ""]
        if m.get("sources"):
            lines += ["Passages used:", ""]
            for i, c in enumerate(m["sources"], start=1):
                match = f"match {c['score']:.0%}" if c.get("score") is not None else "keyword match"
                lines += [f"- [{i}] {source_label(c)} ({match})", f"  > {' '.join(c['text'].split())}"]
            lines.append("")
    return "\n".join(lines)


def save_chat():
    if LOCAL_MODE and ss.messages:
        os.makedirs(LOG_DIR, exist_ok=True)
        base = os.path.join(LOG_DIR, ss.log_name)
        with open(base + ".md", "w", encoding="utf-8") as f:
            f.write(transcript_md())
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump({"messages": ss.messages}, f)


def past_chats():
    if not (LOCAL_MODE and os.path.isdir(LOG_DIR)):
        return []
    chats = []
    for fname in sorted(os.listdir(LOG_DIR), reverse=True):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(LOG_DIR, fname), encoding="utf-8") as f:
                    msgs = json.load(f).get("messages", [])
                first = next((m["content"] for m in msgs if m["role"] == "user"), "")
                chats.append((fname[:-5], f"{fname[:16].replace('_', ' ')}: {first[:45]}"))
            except Exception:
                continue
    return chats


def open_chat(name):
    with open(os.path.join(LOG_DIR, name + ".json"), encoding="utf-8") as f:
        ss.messages = json.load(f).get("messages", [])
    ss.log_name = name


def show_sources(sources, msg_idx):
    with st.expander(f"Sources used ({len(sources)})"):
        for i, c in enumerate(sources, start=1):
            match = f"match {c['score']:.0%}" if c.get("score") is not None else "keyword match"
            st.markdown(f"**[{i}] {source_label(c)}** ({match})")
            st.write(c["text"])
            if fitz and c.get("page") and c["source"].lower().endswith(".pdf"):
                entry = pdf_entry(c["source"])
                if entry and st.checkbox("Show page", key=f"page_{msg_idx}_{i}"):
                    st.image(render_page(entry[1], c["page"], entry[0]))


def show_checks(m):
    """Automatic quote and citation check, the Precise-mode review, and timing."""
    check, review, meta = m.get("check"), m.get("review"), m.get("meta") or {}
    if check:
        color = "#067647" if check["ok"] else "#B42318"
        mark = "✓" if check["ok"] else "⚠"
        st.markdown(f'<span style="color:{color};font-size:0.86rem;font-weight:600">'
                    f'{mark} {html.escape(check["line"])}</span>', unsafe_allow_html=True)
        if check.get("bad_quotes"):
            with st.expander("Quotes not found word for word"):
                st.caption("These may be paraphrased or altered. Check them on the cited page.")
                for q in check["bad_quotes"]:
                    st.markdown(f"> {q}")
    if review:
        if review["ok"]:
            st.markdown('<span style="color:#067647;font-size:0.86rem;font-weight:600">'
                        '✓ Second check: every statement is supported by the passages</span>',
                        unsafe_allow_html=True)
        else:
            with st.expander("⚠ Second check found statements that may not be supported"):
                st.markdown(review["details"])
    if meta.get("seconds") is not None:
        st.caption(f"{meta.get('mode', '')} mode, answered in {meta['seconds']} s")


# ---- Sidebar -------------------------------------------------------
with st.sidebar:
    st.header("API key")
    key_input = st.text_input(
        "Paste your free NVIDIA key", type="password",
        help="Free at build.nvidia.com, no credit card. Used only for this session. "
        "To skip typing it, save it in a .env file (see START_HERE.md).",
    )
    env_key = os.environ.get("LLM_API_KEY", "")
    api_key = key_input or env_key
    if key_input:
        st.markdown('<span class="status-ok">● Key loaded</span>', unsafe_allow_html=True)
    elif env_key:
        st.markdown('<span class="status-ok">● Using the key saved in .env</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-wait">● No key yet</span>', unsafe_allow_html=True)

    st.divider()
    st.header("Documents")
    if LOCAL_MODE:
        st.caption("Library mode: documents, folders, reports and chats are kept on this computer.")
    folders = sorted({d.get("folder", UNSORTED) for d in ss.docs.values()} | {UNSORTED})
    folder_choice = st.selectbox("Folder for new documents", folders + ["New folder…"])
    if folder_choice == "New folder…":
        folder_choice = st.text_input("Folder name", placeholder="e.g. Standards").strip() or UNSORTED
    uploaded = st.file_uploader("Upload PDF, Word, TXT or MD files",
                                type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    if st.button("Add to knowledge base", type="primary", disabled=not uploaded):
        with st.status("Indexing documents…", expanded=True) as status:
            for uf in uploaded:
                if uf.name in ss.docs:
                    st.write(f"{uf.name} is already added.")
                    continue
                data = uf.getvalue()
                suffix = os.path.splitext(uf.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                bar = st.progress(0.0, text=f"Indexing {uf.name}…")
                try:
                    info = rag.add_document(
                        ss.collection, tmp_path, uf.name,
                        progress=lambda f, n=uf.name: bar.progress(f, text=f"Indexing {n}… {f:.0%}"),
                    )
                    store_units(uf.name, info.pop("units"))
                    info["folder"] = folder_choice
                    ss.docs[uf.name] = info
                    if uf.name.lower().endswith(".pdf"):
                        if LOCAL_MODE:
                            with open(os.path.join(FILES_DIR, uf.name), "wb") as f:
                                f.write(data)
                        else:
                            ss.files[uf.name] = (data, hashlib.sha1(data).hexdigest())
                    pages = f" from {info['pages']} pages" if info["pages"] else ""
                    st.write(f"Added {uf.name} to {folder_choice}: {info['sections']} sections{pages}.")
                except Exception as e:
                    st.write(f"Could not add {uf.name}: {e}")
                finally:
                    os.remove(tmp_path)
                    bar.empty()
            save_registry()
            ss.kw_version += 1
            status.update(label="Indexing finished", state="complete")

    if ss.docs:
        used_folders = sorted({d.get("folder", UNSORTED) for d in ss.docs.values()})
        chosen_folders = (st.multiselect("Search in folders", used_folders, default=used_folders)
                          if len(used_folders) > 1 else used_folders)
        in_folders = [n for n, d in ss.docs.items() if d.get("folder", UNSORTED) in chosen_folders]
        selected = st.multiselect("Search in documents", in_folders, default=in_folders)
        total = sum(d["sections"] for d in ss.docs.values())
        st.caption(f"{len(ss.docs)} document(s), {total} sections indexed")
        if not LOCAL_MODE and total > LARGE_ONLINE:
            st.warning("This is a large collection for the free online version and it may "
                       "slow down. For big libraries, run it on your own computer.")
        with st.expander("Manage documents"):
            target = st.selectbox("Document", list(ss.docs), key="manage_doc")
            new_folder = st.text_input("Move to folder", value=ss.docs[target].get("folder", UNSORTED))
            c1, c2 = st.columns(2)
            if c1.button("Move"):
                ss.docs[target]["folder"] = new_folder.strip() or UNSORTED
                save_registry()
                st.rerun()
            if c2.button("Remove"):
                forget_document(target)
                st.rerun()
    else:
        selected = []
        st.caption("No documents added yet.")

    mode = st.radio(
        "How to answer", ["Auto", "Fast", "Precise", "Thorough"],
        captions=["picks the right mode for each question", "6 passages, one call",
                  "10 passages, answer double-checked", "reads every page, slowest"],
        help="Auto uses Fast for specific questions and Thorough for comparisons, summaries "
        "and complete lists, or the document overviews when they exist. Fast suits specific questions. Precise uses more passages and has the model "
        "check its own answer against them. Thorough goes through every page, for summaries, "
        "complete lists and comparisons. Every answer also gets an automatic check of its "
        "quotes and citations, whatever the mode.",
    )

    with st.expander("Settings"):
        min_sim = st.slider(
            "Minimum match to answer", 0.0, 0.8, rag.MIN_SIMILARITY, 0.05,
            help="Meaning matches weaker than this are ignored. If nothing passes, the app "
            "says the answer is not in the document instead of guessing.",
        )
        tested = []
        if os.path.exists("working_models.txt"):
            with open("working_models.txt", encoding="utf-8") as f:
                tested = [line.strip() for line in f if line.strip()]
        if tested:
            options = tested + ["Other…"]
            start = options.index(rag.DEFAULT_MODEL) if rag.DEFAULT_MODEL in options else 0
            picked = st.selectbox("Model", options, index=start,
                                  help="Models that answered when check_models.py last ran.")
            model = (st.text_input("Model id", value=rag.DEFAULT_MODEL) if picked == "Other…"
                     else picked)
        else:
            model = st.text_input("Model", value=rag.DEFAULT_MODEL,
                                  help="Paste a model id. Run check_models.py to get a dropdown "
                                  "of the models that work with your key.")
        base_url = st.text_input("API address", value=rag.DEFAULT_BASE_URL,
                                 help="Any OpenAI-compatible service. For a fully local, private "
                                 "setup with Ollama, use http://localhost:11434/v1.")

    st.divider()
    st.header("Chats")
    if ss.messages:
        st.download_button("Export this chat (Markdown)", transcript_md(),
                           file_name=f"document-chat_{ss.log_name}.md", mime="text/markdown")
    if st.button("New chat"):
        ss.messages = []
        ss.log_name = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        st.rerun()
    chats = past_chats()
    if chats:
        with st.expander(f"Past chats ({len(chats)})"):
            labels = {name: label for name, label in chats}
            pick = st.selectbox("Chat", list(labels), format_func=labels.get,
                                label_visibility="collapsed")
            if st.button("Open"):
                open_chat(pick)
                st.rerun()
    elif not LOCAL_MODE:
        st.caption("Chats are not saved in this version. Export one to keep it.")

    st.markdown(f'<p class="credit">Developed by Milin Patel, version {rag.APP_VERSION}</p>',
                unsafe_allow_html=True)

local_api = "localhost" in base_url or "127.0.0.1" in base_url


def key_missing():
    if not api_key and not local_api:
        st.warning("Add your API key in the sidebar first.")
        return True
    return False


def progress_bar(text):
    bar = st.progress(0.0, text=text)
    return bar, (lambda f: bar.progress(min(f, 1.0), text=f"{text} {f:.0%}"))


# ---- Main area -----------------------------------------------------
st.markdown('<p class="app-title">Document chat</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-sub">A research assistant for your own documents. Answers come only from '
    'them, with page references and the exact passages shown. If the answer is not there, '
    'it says so.</p>',
    unsafe_allow_html=True,
)
view = st.radio("View", ["Chat", "Tools", "About"], horizontal=True,
                label_visibility="collapsed", key="view",
                help="Switching view while an answer is being written stops that answer.")

# ---- Chat view -----------------------------------------------------
if view == "Chat":
    if not ss.docs and not ss.messages:
        st.markdown(
            """
            <div class="getstarted">
              <h4>Get started</h4>
              <ol>
                <li>Paste your free NVIDIA key in the sidebar, or save it in a .env file.</li>
                <li>Upload a PDF, Word, TXT or MD file and click "Add to knowledge base".</li>
                <li>Ask a question below. Spelling mistakes are fine.</li>
              </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for idx, m in enumerate(ss.messages):
        with st.chat_message(m["role"]):
            for key in ("note", "read_note"):
                if m.get(key):
                    st.caption(m[key])
            st.markdown(m["content"])
            if m["role"] == "assistant":
                show_checks(m)
                if m.get("sources"):
                    show_sources(m["sources"], idx)

    typed = st.chat_input("Ask a question about your documents")
    retry = ss.pop("retry", None)
    if not typed and not retry and ss.messages and ss.messages[-1]["role"] == "user":
        st.info("This question was interrupted before it was answered, probably by a click "
                "elsewhere in the app.")
        if st.button("Answer it now"):
            ss.retry = ss.messages.pop()["content"]
            st.rerun()
    question = typed or retry
    if question:
        earlier = [m["content"] for m in ss.messages if m["role"] == "user"]
        ss.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        sources, note, read_note = [], None, None
        meta = {"time": f"{datetime.datetime.now():%Y-%m-%d %H:%M}", "model": model,
                "api": base_url, "mode": mode,
                "min_match": min_sim if mode != "Thorough" else None}
        check, review, started = None, None, time.time()
        with st.chat_message("assistant"):
            if not api_key and not local_api:
                reply = "Paste your free NVIDIA API key in the sidebar to begin."
                st.markdown(reply)
            elif not ss.docs:
                reply = "Add a document first, using the sidebar."
                st.markdown(reply)
            elif not selected:
                reply = 'Select at least one document under "Search in documents".'
                st.markdown(reply)
            else:
                try:
                    kw = keyword_index()
                    corrected, changes = rag.correct_query(question, kw)
                    if changes:
                        note = f"Searched for: {corrected}"

                    used = mode
                    overviews = {n: get_report("overview", n) for n in selected}
                    if mode == "Auto":
                        if not rag.needs_whole_documents(corrected):
                            used = "Fast"
                        elif all(overviews.values()) and len(selected) > 1:
                            used = "Overviews"
                        else:
                            used = "Thorough"
                    meta["mode"] = used if mode != "Auto" else f"Auto ({used})"
                    meta["min_match"] = min_sim if used in ("Fast", "Precise") else None

                    if used == "Overviews":
                        context = "\n\n".join(f"## Overview of {n}\n{ov}" for n, ov in overviews.items())
                        read_note = ("Answered from the overviews of: " + ", ".join(selected)
                                     + ". For more detail, choose Thorough.")
                        st.caption(read_note)
                        reply = st.write_stream(rag.stream_whole(
                            api_key, corrected, context, model, base_url, earlier[-3:], True))
                    elif used != "Thorough":
                        with st.status(f"{random.choice(VERBS)}…", expanded=False) as status:
                            st.write("Checking spelling against the documents…")
                            if changes:
                                st.write("Corrected: " + ", ".join(f"{a} → {b}" for a, b in changes))
                            query = rag.retrieval_query(corrected, earlier[-1] if earlier else None)
                            st.write(f"Searching {len(selected)} document(s) by meaning and keywords…")
                            sources = rag.retrieve(ss.collection, kw, query, selected, min_sim,
                                                   10 if used == "Precise" else rag.N_RESULTS)
                            if sources:
                                st.write(f"Found {len(sources)} relevant passages. Writing the answer…")
                                status.update(label=f"Found {len(sources)} relevant passages",
                                              state="complete")
                            else:
                                status.update(label="No relevant passages found", state="complete")
                        if note:
                            st.caption(note)
                        if not sources:
                            reply = (f"{rag.NOT_FOUND} No passage matched the question closely "
                                     "enough. Try rephrasing, lower the minimum match in Settings, "
                                     "or switch to Thorough.")
                            st.markdown(reply)
                        else:
                            reply = st.write_stream(rag.stream_answer(
                                api_key, corrected, sources, model, base_url, earlier[-3:]))
                    else:
                        batches = rag.make_batches([(n, get_units(n)) for n in selected])
                        if len(batches) > 12 and not all(overviews.values()):
                            st.caption(f"{len(batches)} parts to read, so this takes a few minutes. "
                                       "Tip: create overviews for these documents under Tools, "
                                       "and comparisons in Auto mode then take seconds.")
                        with st.status(f"{random.choice(VERBS)}…", expanded=True) as status:
                            if changes:
                                st.write("Corrected: " + ", ".join(f"{a} → {b}" for a, b in changes))
                            st.write(f"Reading {len(selected)} document(s) in {len(batches)} part(s)…")
                            if len(batches) == 1:
                                context, from_notes = batches[0], False
                            else:
                                bar, cb = progress_bar("Reading every page…")
                                notes = rag.collect_notes(api_key, corrected, batches, model,
                                                          base_url, progress=cb)
                                bar.empty()
                                context, from_notes = "\n\n".join(notes), True
                                st.write(f"Relevant content found in {len(notes)} of {len(batches)} parts.")
                            status.update(label="Finished reading", state="complete")
                        read_note = f"Read every page of: {', '.join(selected)}"
                        st.caption(read_note)
                        if not context.strip():
                            reply = rag.NOT_FOUND
                            st.markdown(reply)
                        else:
                            reply = st.write_stream(rag.stream_whole(
                                api_key, corrected, context, model, base_url, earlier[-3:], from_notes))

                    if not isinstance(reply, str) or not reply.strip():
                        reply = rag.NOT_FOUND
                    reply += rag.citation_note(reply)

                    if rag.leaked_reasoning(reply):
                        reply += ("\n\n_The model printed its reasoning instead of a clean answer. "
                                  "Ask again, or choose a different model in Settings._")
                    if used in ("Thorough", "Overviews"):
                        result = rag.verify_answer(reply, page_texts=[
                            (n, pg, t) for n in selected for pg, t in get_units(n)])
                    else:
                        result = rag.verify_answer(reply, sources=sources)
                    if result is not None:
                        ok, line = rag.format_check(result)
                        check = {"ok": ok, "line": line, "bad_quotes": result["bad_quotes"]}
                    if used == "Precise" and sources and check is not None:
                        with st.spinner("Double-checking the answer against the passages…"):
                            supported, details = rag.review_answer(api_key, reply, sources,
                                                                   model, base_url)
                        review = {"ok": supported, "details": details}
                except Exception as e:
                    reply = rag.explain_error(e, model)
                    sources = []
                    st.markdown(reply)

        meta["seconds"] = round(time.time() - started)
        ss.messages.append({"role": "assistant", "content": reply, "sources": sources,
                            "note": note, "read_note": read_note, "meta": meta,
                            "check": check, "review": review})
        save_chat()
        st.rerun()

# ---- Tools view ----------------------------------------------------
elif view == "Tools":
    if not ss.docs:
        st.info("Add a document in the sidebar to use the tools.")
    else:
        tool = st.selectbox("Tool", ["Document overview", "Mind map", "Ideas and open questions",
                                     "Check a draft", "Compare documents"])
        stamp = lambda: (f"_Generated {datetime.datetime.now():%Y-%m-%d %H:%M} with {model}. "
                         "Check page references before relying on it._\n\n")

        if tool == "Document overview":
            st.caption("Reads the whole document once and writes its summary, structure and "
                       "defined terms with page references.")
            doc = st.selectbox("Document", list(ss.docs), key="ov_doc")
            report = get_report("overview", doc)
            if report:
                st.markdown(report)
                st.download_button("Download overview", report, key="ov_dl", mime="text/markdown",
                                   file_name=f"overview_{os.path.splitext(doc)[0]}.md")
            if st.button("Recreate overview" if report else "Create overview") and not key_missing():
                bar, cb = progress_bar("Reading the document…")
                try:
                    md = rag.build_overview(api_key, doc, get_units(doc), model, base_url, cb)
                    save_report("overview", doc, stamp() + md)
                    st.rerun()
                except Exception as e:
                    bar.empty()
                    st.error(rag.explain_error(e, model))

        elif tool == "Mind map":
            st.caption("A map of the document's topics, built from its overview. Every branch "
                       "carries a page number so it can be traced back.")
            doc = st.selectbox("Document", list(ss.docs), key="mm_doc")
            outline = get_report("mindmap", doc)
            if outline:
                mermaid = rag.outline_to_mermaid(os.path.splitext(doc)[0], outline)
                components.html(
                    f"""<div class="mermaid">{html.escape(mermaid)}</div>
                    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
                    <script>mermaid.initialize({{startOnLoad: true, theme: 'neutral'}});</script>""",
                    height=620, scrolling=True,
                )
                c1, c2 = st.columns(2)
                c1.download_button("Download outline", outline, key="mm_dl1", mime="text/markdown",
                                   file_name=f"mindmap_{os.path.splitext(doc)[0]}.md")
                c2.download_button("Download Mermaid file", mermaid, key="mm_dl2",
                                   file_name=f"mindmap_{os.path.splitext(doc)[0]}.mmd")
                with st.expander("Outline as text"):
                    st.text(outline)
            if st.button("Recreate mind map" if outline else "Create mind map") and not key_missing():
                bar, cb = progress_bar("Reading the document…")
                try:
                    overview = get_report("overview", doc)
                    if not overview:
                        overview = stamp() + rag.build_overview(api_key, doc, get_units(doc),
                                                                model, base_url, cb)
                        save_report("overview", doc, overview)
                    bar.progress(1.0, text="Drawing the map…")
                    save_report("mindmap", doc, rag.mindmap_outline(api_key, overview, model, base_url))
                    st.rerun()
                except Exception as e:
                    bar.empty()
                    st.error(rag.explain_error(e, model))

        elif tool == "Ideas and open questions":
            st.caption("Open questions, research directions, gaps and assumptions prompted by "
                       "the document. These are suggestions, not statements from the document.")
            doc = st.selectbox("Document", list(ss.docs), key="id_doc")
            report = get_report("ideas", doc)
            if report:
                st.markdown(report)
                st.download_button("Download ideas", report, key="id_dl", mime="text/markdown",
                                   file_name=f"ideas_{os.path.splitext(doc)[0]}.md")
            if st.button("Generate again" if report else "Generate ideas") and not key_missing():
                bar, cb = progress_bar("Reading the document…")
                try:
                    md = rag.build_ideas(api_key, doc, get_units(doc), model, base_url, cb)
                    save_report("ideas", doc, stamp().replace("Check page references",
                                "These are suggestions. Check page references") + md)
                    st.rerun()
                except Exception as e:
                    bar.empty()
                    st.error(rag.explain_error(e, model))

        elif tool == "Check a draft":
            st.caption("Paste a paragraph from your own writing. Each sentence is checked against "
                       "the documents selected in the sidebar: supported, partly supported, "
                       "contradicted, or not found. Up to 12 sentences at a time.")
            draft = st.text_area("Your text", height=180,
                                 placeholder="Paste a paragraph from your paper or thesis…")
            if st.button("Check", disabled=not draft.strip()) and not key_missing():
                claims, left_out = rag.split_claims(draft)
                if not selected:
                    st.warning('Select at least one document under "Search in documents".')
                elif not claims:
                    st.warning("No full sentences found to check.")
                else:
                    kw = keyword_index()
                    bar, cb = progress_bar("Checking each sentence…")
                    results = []
                    try:
                        for i, claim in enumerate(claims):
                            passages = rag.retrieve(ss.collection, kw, claim, selected, min_sim, 4)
                            verdict, reason = rag.check_claim(api_key, claim, passages, model, base_url)
                            results.append({"claim": claim, "verdict": verdict,
                                            "reason": reason, "sources": passages})
                            cb((i + 1) / len(claims))
                        ss.draft_results = {"results": results, "left_out": left_out,
                                            "time": f"{datetime.datetime.now():%Y-%m-%d %H:%M}",
                                            "model": model}
                    except Exception as e:
                        st.error(rag.explain_error(e, model))
                    bar.empty()
            if ss.draft_results:
                res = ss.draft_results
                counts = {}
                for r in res["results"]:
                    counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
                st.markdown("**Result:** " + ", ".join(
                    f"{n} {VERDICT_STYLE.get(v, ('', v))[1].lower()}" for v, n in counts.items()))
                lines = [f"# Draft check", f"_Checked {res['time']} with {res['model']}._", ""]
                for i, r in enumerate(res["results"]):
                    color, label = VERDICT_STYLE.get(r["verdict"], ("#6B7280", r["verdict"]))
                    st.markdown(f'<div class="claim">{html.escape(r["claim"])}</div>'
                                f'<span class="verdict" style="color:{color}">{label}</span>',
                                unsafe_allow_html=True)
                    st.write(r["reason"])
                    if r["sources"]:
                        show_sources(r["sources"], f"draft{i}")
                    lines += [f"- {r['claim']}", f"  - {label}: {r['reason']}"]
                    lines += [f"    - [{j}] {source_label(c)}" for j, c in enumerate(r["sources"], 1)]
                if res["left_out"]:
                    st.caption(f"{res['left_out']} more sentence(s) were not checked. "
                               "Paste them in a second round.")
                st.download_button("Download check (Markdown)", "\n".join(lines),
                                   file_name="draft-check.md", mime="text/markdown")

        elif tool == "Compare documents":
            st.caption("Reads each document and fills the same fields for all of them, with page "
                       "references, like a literature review table.")
            docs = st.multiselect("Documents", list(ss.docs), default=selected or list(ss.docs))
            fields_text = st.text_input(
                "Fields (comma-separated)",
                value="Research question, Method, Data, Main findings, Limitations")
            fields = [f.strip() for f in fields_text.split(",") if f.strip()]
            if st.button("Build table", disabled=not (docs and fields)) and not key_missing():
                rows = []
                try:
                    for n, doc in enumerate(docs, start=1):
                        bar, cb = progress_bar(f"Reading {doc} ({n} of {len(docs)})…")
                        values = rag.extract_fields(api_key, doc, get_units(doc), fields,
                                                    model, base_url, cb)
                        rows.append({"Document": doc, **values})
                        bar.empty()
                    ss.matrix = {"rows": rows, "fields": fields, "model": model,
                                 "time": f"{datetime.datetime.now():%Y-%m-%d %H:%M}"}
                except Exception as e:
                    st.error(rag.explain_error(e, model))
            if ss.matrix:
                st.dataframe(ss.matrix["rows"], use_container_width=True, hide_index=True)
                st.caption(f"Generated {ss.matrix['time']} with {ss.matrix['model']}. "
                           "Check page references before relying on it.")
                buf = io.StringIO()
                writer = csv.DictWriter(buf, fieldnames=["Document"] + ss.matrix["fields"])
                writer.writeheader()
                writer.writerows(ss.matrix["rows"])
                st.download_button("Download table (CSV, opens in Excel)", buf.getvalue().encode("utf-8-sig"),
                                   file_name="comparison.csv", mime="text/csv")

# ---- About view ----------------------------------------------------
else:
    st.markdown(f"""
### About this tool

Document chat {rag.APP_VERSION}, developed by Milin Patel.

A research assistant for asking questions about your own documents: papers, standards,
reports, theses. It answers only from the documents you add, shows the page and the exact
passage behind every statement, and says so plainly when the answer isn't there.

### How it answers

- **Fast and Precise** find the passages that best match your question, by meaning and by
  keywords, with spelling mistakes corrected against the documents' own words. Weak matches
  are discarded. Only the remaining passages go to the language model.
- **Thorough** goes through every page of the selected documents, for summaries,
  complete lists and comparisons.
- The model is told to use only the text it is given, to cite every statement, and to quote
  definitions word for word. Answers without a citation are flagged.

### Speed and accuracy

- **Auto (default):** uses Fast for specific questions, and Thorough for comparisons,
  summaries and complete lists. When every selected document has an overview, comparisons
  are answered from the overviews, which takes seconds.
- **Fast:** 6 passages and one model call, usually seconds.
- **Precise:** 10 passages, then a second pass in which the model checks its own answer
  against them and lists anything unsupported. About twice as long.
- **Thorough:** reads every page. A minute or more for long documents.

Whatever the mode, every answer gets an **automatic check that uses no AI**: each quotation
is searched for word for word in the source text, and each citation is checked to point to
a passage or page that exists. The result is shown under the answer.

### How to verify an answer yourself

1. Look at the automatic check under the answer.
2. Open "Sources used" and read the passages, or use "Show page" to see the original page.
3. For anything you will publish, check it in the original document.

How often the search finds the right page can be measured with eval.py on your own
questions, so accuracy is a number you can state, not a claim.

### Tools

- **Document overview:** summary, structure and defined terms, with pages.
- **Mind map:** the document's topics as a map, every branch with a page number.
- **Ideas and open questions:** suggestions for research, clearly marked as suggestions.
- **Check a draft:** checks each sentence of your own text against the documents.
- **Compare documents:** the same fields filled for several documents, as a table.

### Limits

- A language model can still misread a passage. Always check the cited page before relying
  on an answer. The tool is built to make that check quick, not to replace it.
- The same question can be worded slightly differently each time, and can change if the
  provider replaces the model. Exports record the model and date for that reason.
- Scanned PDFs without a text layer can't be read yet.
- Clicking elsewhere in the app while an answer is being written stops it. The question is
  kept and can be answered with one click.
- Thorough makes about one model call per 15 pages, so long documents take a
  minute or two and use more of the free quota.
- Free models are retired from time to time. If that happens, change the model in Settings.

### Your data

- Reading, indexing and searching happen on the computer running the app.
- Your question and the selected passages are sent to the model provider shown in Settings
  ({base_url}). Free providers may use that data to improve their models, so don't use them
  for licensed, unpublished or confidential documents. For those, run a local model.
- In the shared online version, your documents are kept only for your session, separate
  from everyone else's, and nothing is stored afterwards. Your key is never stored.

### Citing your use of it

If you use answers from this tool in your work, check them against the source and disclose
the use as your institution or journal requires, naming the tool, version, model and date.
The exported chats contain all of these.
""")
