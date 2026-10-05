"""
rag.py - core Retrieval-Augmented Generation logic.

These are stateless helpers. The caller (app.py) owns the vector-database
collection and the API key, so the exact same code works for one local
user and for a shared online app where every visitor brings their own key
and their own private set of documents.
"""

import re
import uuid

import pypdf
from anthropic import Anthropic

# ---- Settings you might change --------------------------------------
MODEL = "claude-sonnet-5-5"   # Claude model that writes the answer.
                              # If you ever get a "model not found" error,
                              # update this one string from the console list.
CHUNK_SIZE = 1000             # characters per chunk (~150-200 words)
CHUNK_OVERLAP = 150           # characters shared between neighbouring chunks
N_RESULTS = 5                 # how many chunks to retrieve per question

SYSTEM_PROMPT = """You answer questions using ONLY the document excerpts provided in the user message.

Rules:
- Use only the information in the excerpts. Do not use outside knowledge.
- If the excerpts do not contain the answer, say exactly: "I couldn't find this in the document." Then, if useful, briefly say what related information the excerpts do contain.
- Cite the excerpts you rely on with their number in square brackets, e.g. [1], [2].
- Be concise and precise. Never invent facts, numbers, citations, or sources.
"""


# ---- Reading files -------------------------------------------------
def read_file(path, filename):
    """Return the plain text of a PDF, TXT or MD file as one string."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        parts = []
        reader = pypdf.PdfReader(path)
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                parts.append(f"[page {page_number}]\n{page_text}")
        return "\n\n".join(parts)
    elif lower.endswith((".txt", ".md")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {filename} (use PDF, TXT or MD).")


# ---- Chunking ------------------------------------------------------
def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks, breaking on nice boundaries
    (paragraph, sentence, line, space) where possible."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:  # not the last chunk: try to end on a clean boundary
            window = text[start:end]
            for sep in ["\n\n", ". ", "\n", " "]:
                pos = window.rfind(sep)
                if pos > chunk_size * 0.5:
                    end = start + pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


# ---- Add a document to a collection --------------------------------
def add_document(collection, path, filename):
    """Read a file, chunk it, and add the chunks to the given collection.
    Returns the number of chunks added."""
    text = read_file(path, filename)
    if not text.strip():
        raise ValueError(
            f"No text could be extracted from {filename}. "
            "If it is a scanned (image-only) PDF, it needs OCR first."
        )
    chunks = split_into_chunks(text)
    ids = [f"{filename}-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


# ---- Retrieve + answer ---------------------------------------------
def retrieve(collection, question, n_results=N_RESULTS):
    """Return the most relevant chunks for a question as a list of dicts."""
    count = collection.count()
    if count == 0:
        return []
    res = collection.query(query_texts=[question], n_results=min(n_results, count))
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return [
        {"text": d, "source": m.get("source"), "chunk": m.get("chunk")}
        for d, m in zip(docs, metas)
    ]


def _build_context(chunks):
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (from {c['source']})\n{c['text']}")
    return "\n\n".join(blocks)


def answer(api_key, question, chunks):
    """Ask Claude to answer using only the given chunks and the user's key."""
    context = _build_context(chunks)
    user_message = f"Document excerpts:\n\n{context}\n\nQuestion: {question}"
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
