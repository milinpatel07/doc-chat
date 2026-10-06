"""
rag.py - core Retrieval-Augmented Generation logic.

Developed by Milin Patel.

Stateless helpers: the caller (app.py) owns the vector collection, the key
and the model settings. Any OpenAI-compatible service works for the answer
step (NVIDIA by default, also Groq, OpenRouter, Gemini, or local Ollama).

Search is hybrid: meaning-based (Chroma embeddings) plus keyword-based
(BM25), merged with reciprocal rank fusion. Spelling in the question is
corrected against the documents' own vocabulary before searching.
"""

import difflib
import hashlib
import math
import re
import time
from collections import Counter

import pypdf
from openai import OpenAI

APP_NAME = "Document chat"
APP_VERSION = "0.8"

# ---- Defaults (model, address and strictness can be changed in the app) ----
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
# Providers retire free models over time. If you get "410 Gone", paste a
# current model id from build.nvidia.com into Settings > Model.

CHUNK_SIZE = 1000        # characters per section
CHUNK_OVERLAP = 150      # characters shared between neighbouring sections
N_RESULTS = 6            # passages sent to the model per question
MIN_SIMILARITY = 0.25    # meaning matches below this are ignored (0 to 1)
BATCH_CHARS = 40000      # text sent per call when reading whole documents

NOT_FOUND = "I couldn't find this in the document."

SYSTEM_PROMPT = f"""You are a careful research assistant. You answer ONLY from the numbered document excerpts in the user message.

Rules:
1. Use only the excerpts. Never use outside knowledge, even if you know the answer.
2. If the excerpts do not answer the question, or the question is not about the documents (greetings, general knowledge, opinions, questions about you), reply with exactly: "{NOT_FOUND}" You may add one short sentence on what related content the excerpts do contain.
3. Put the excerpt number in square brackets after every statement it supports, e.g. [2].
4. When asked for a definition, quote the defining text word for word, then cite it.
5. Never invent facts, numbers, clause numbers, sources or page numbers. If excerpts disagree or are incomplete, say so.
6. The question may contain spelling mistakes; interpret it sensibly.
7. Earlier questions are given only so you understand what a follow-up refers to. Facts must come from the current excerpts.
8. Be concise and precise.
"""

STOPWORDS = set("""a an the and or of to in on for with by is are was were be been being
what which who whom whose when where why how do does did can could should would may
might shall will it its this that these those there their they them as at from into
about than then so such not no yes i you we he she me my our your please give tell
explain define definition describe show list mean means meaning""".split())

COMMON_WORDS = set("""about above after again against also always answer around because
before being below between both bring check clear compare could difference
different does doing done during each either explain first fully found from
give given good great have here into just know least like list make many more
most much must need never only other over please point read really right same
says should show since some still such summary tell than that their them then
there these they thing think this those though three through together under
understand understood until very want were what when where which while whole
will with within without work would write your""".split())

PRONOUNS = {"it", "its", "this", "that", "these", "those", "they", "them", "their", "same"}

TOKEN_RE = re.compile(r"[^\W_]+(?:\.\d+)*")   # keeps clause numbers like 3.30 together


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


# ---- Reading files -------------------------------------------------
def _strip_boilerplate(pages):
    """Remove lines that repeat on many pages (headers, footers, download
    stamps). Digits are ignored when comparing, so page numbers and dates
    don't hide a repeated line."""
    if len(pages) < 4:
        return pages
    norm = lambda line: re.sub(r"\d+", "#", line.strip())
    counts = Counter()
    for _, text in pages:
        counts.update({norm(l) for l in text.splitlines() if len(l.strip()) >= 8})
    limit = max(3, int(0.3 * len(pages)))
    boiler = {l for l, c in counts.items() if c >= limit}
    return [(p, "\n".join(l for l in t.splitlines() if norm(l) not in boiler)) for p, t in pages]


def clean_text(text):
    """Remove layout debris that confuses search and the model: empty table
    rows, long runs of table pipes, separator lines and repeated spaces.
    Converted Markdown files in particular contain a lot of this."""
    lines = []
    for line in text.splitlines():
        if not re.sub(r"[|\-:=_\s.+*#]", "", line):
            continue                                   # pipes, dashes, dots only
        line = re.sub(r"(\|\s*){2,}", "| ", line)      # collapse empty cells
        line = re.sub(r"\.{5,}", " ... ", line)          # table-of-contents dot leaders
        lines.append(line.strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines))


def read_units(path, filename):
    """Return a list of (page_number, text). page_number is 0 when the
    format has no pages (Word, TXT, MD)."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        reader = pypdf.PdfReader(path)
        pages = [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, 1)]
        return [(p, clean_text(t)) for p, t in _strip_boilerplate(pages)]
    if lower.endswith(".docx"):
        import docx  # python-docx
        d = docx.Document(path)
        parts = [p.text for p in d.paragraphs]
        for table in d.tables:
            for row in table.rows:
                parts.append(" | ".join(c.text.strip() for c in row.cells))
        return [(0, clean_text("\n".join(parts)))]
    if lower.endswith((".txt", ".md")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return [(0, clean_text(f.read()))]
    raise ValueError(f"Unsupported file type: {filename}")


# ---- Chunking ------------------------------------------------------
def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
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


# ---- Indexing ------------------------------------------------------
def add_document(collection, path, filename, progress=None):
    """Index a file. IDs are content hashes, so adding the same file twice
    never creates duplicates. progress(fraction) is called while embedding.
    Returns {"sections": n, "pages": p}."""
    units = read_units(path, filename)
    items = {}
    for page, text in units:
        for chunk in split_into_chunks(text):
            cid = hashlib.sha1(f"{filename}|{page}|{chunk}".encode("utf-8")).hexdigest()
            items[cid] = (chunk, page)
    if not items:
        raise ValueError(
            "No text could be extracted. If it is a scanned (image-only) PDF, it needs OCR first."
        )
    ids = list(items)
    step = 200
    for i in range(0, len(ids), step):
        batch = ids[i:i + step]
        collection.upsert(
            ids=batch,
            documents=[items[b][0] for b in batch],
            metadatas=[{"source": filename, "page": items[b][1]} for b in batch],
        )
        if progress:
            progress(min(1.0, (i + len(batch)) / len(ids)))
    pages = sum(1 for p, t in units if p and t.strip())
    return {"sections": len(ids), "pages": pages, "units": units}


# ---- Keyword index (BM25) and spelling ------------------------------
class KeywordIndex:
    """A small BM25 index over all indexed sections, plus the documents'
    vocabulary for spelling correction."""

    def __init__(self, ids, texts, metas):
        self.ids, self.texts, self.metas = ids, texts, metas
        self.tf = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.lengths) / max(1, len(self.lengths))
        self.df = Counter()
        for c in self.tf:
            self.df.update(c.keys())
        self.n = len(texts)
        self.vocab_set = set(self.df)
        self.vocab = sorted(w for w in self.df if len(w) >= 3 and w.isalpha())

    def search(self, query, sources=None, k=12):
        """Return [(id, position)] of the best keyword matches. A section only
        counts if it contains at least one distinctive query term, so common
        words alone never produce a match."""
        terms = [t for t in tokenize(query) if t not in STOPWORDS]
        if not terms or not self.n:
            return []
        rare_limit = max(3, 0.2 * self.n)
        scored = []
        for i, tf in enumerate(self.tf):
            if sources and self.metas[i].get("source") not in sources:
                continue
            s, distinctive = 0.0, False
            for t in terms:
                f = tf.get(t)
                if not f:
                    continue
                df = self.df[t]
                idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
                s += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * self.lengths[i] / self.avg))
                if df <= rare_limit:
                    distinctive = True
            if s > 0 and distinctive:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [(self.ids[i], i) for _, i in scored[:k]]


def correct_query(question, index):
    """Fix spelling against the documents' vocabulary.
    Returns (corrected_question, [(wrong, right), ...])."""
    if index is None or not index.vocab:
        return question, []
    changes = []

    def fix(m):
        word = m.group(0)
        lw = word.lower()
        if (len(lw) < 5 or lw in index.vocab_set or lw in STOPWORDS
                or lw in COMMON_WORDS):
            return word
        match = difflib.get_close_matches(lw, index.vocab, n=1, cutoff=0.85)
        if match:
            changes.append((word, match[0]))
            return match[0]
        return word

    corrected = re.sub(r"[^\W\d_]+", fix, question)
    return corrected, changes


def retrieval_query(question, previous=None):
    """For short or pronoun-based follow-ups, search with the previous
    question too, so "and its examples?" still finds the right passages."""
    words = tokenize(question)
    if previous and (len(words) <= 6 or PRONOUNS & set(words)):
        return f"{previous} {question}"
    return question


# ---- Retrieval -----------------------------------------------------
def retrieve(collection, kw_index, question, sources=None,
             min_similarity=MIN_SIMILARITY, n_results=N_RESULTS):
    """Hybrid search. Meaning matches below min_similarity are dropped;
    keyword matches need a distinctive term. Results are merged by
    reciprocal rank fusion. Similarity is cosine (0 to 1), derived from
    Chroma's squared L2 distance on its normalised embeddings."""
    count = collection.count()
    if count == 0:
        return []
    kwargs = dict(
        query_texts=[question],
        n_results=min(n_results * 2, count),
        include=["documents", "metadatas", "distances"],
    )
    if sources:
        kwargs["where"] = (
            {"source": sources[0]} if len(sources) == 1 else {"source": {"$in": list(sources)}}
        )
    res = collection.query(**kwargs)

    cand, vec_rank, kw_rank = {}, {}, {}
    for r, (cid, doc, meta, dist) in enumerate(
        zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    ):
        vec_rank[cid] = r
        cand[cid] = {"id": cid, "text": doc, "source": meta.get("source"),
                     "page": meta.get("page") or None,
                     "score": max(0.0, 1.0 - dist / 2.0), "keyword": False}

    if kw_index is not None:
        for r, (cid, i) in enumerate(kw_index.search(question, sources)):
            kw_rank[cid] = r
            if cid in cand:
                cand[cid]["keyword"] = True
            else:
                meta = kw_index.metas[i]
                cand[cid] = {"id": cid, "text": kw_index.texts[i], "source": meta.get("source"),
                             "page": meta.get("page") or None, "score": None, "keyword": True}

    kept = [c for c in cand.values()
            if c["keyword"] or (c["score"] is not None and c["score"] >= min_similarity)]
    fused = lambda c: sum(1 / (60 + rank[c["id"]]) for rank in (vec_rank, kw_rank) if c["id"] in rank)
    kept.sort(key=fused, reverse=True)

    out, seen = [], set()
    for c in kept:
        key = c["text"].strip()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out[:n_results]


# ---- Answering -----------------------------------------------------
WHOLE_SYSTEM_PROMPT = f"""You are a careful research assistant. You answer ONLY from the document text in the user message. Each part is marked with its source and page, like [file.pdf, p. 18].

Rules:
1. Use only that text. Never use outside knowledge, even if you know the answer.
2. If the text does not answer the question, or the question is not about the documents, reply with exactly: "{NOT_FOUND}"
3. After every statement, cite the source and page in square brackets, e.g. [file.pdf, p. 18].
4. When asked for a definition, quote the defining text word for word, then cite it.
5. Never invent facts, numbers, clause numbers, sources or page numbers. If the text is incomplete or contradictory, say so.
6. The question may contain spelling mistakes; interpret it sensibly.
7. Be thorough when asked for lists, comparisons or overviews; otherwise be concise.
"""

NOTES_PROMPT = """You collect evidence from document pages for a later answer. Use only the pages given.

Question: {question}

Copy every passage relevant to the question, word for word or very closely, each followed by its source and page in square brackets, e.g. [file.pdf, p. 18]. If nothing on these pages is relevant, reply with exactly: NONE"""

OVERVIEW_NOTES_PROMPT = """You are reading part of a document to help build an overview. Use only the pages given. Write, briefly and with no introduction:
- Main topics, one line each, with the page in square brackets, e.g. [p. 12].
- Defined terms: only terms the text explicitly defines, each with its page, e.g. "triggering condition [p. 17]"."""

OVERVIEW_FINAL_PROMPT = """Combine these notes about one document into an overview. Use only the notes and add nothing that is not in them. Keep page references exactly as given. Use Markdown with these sections:

## Summary
3 to 6 sentences on what the document covers.

## Structure
The main parts in order, with page references.

## Defined terms
Alphabetical list, each term with its page reference."""


def _label(c):
    return f"{c['source']}, p. {c['page']}" if c.get("page") else c["source"]


def _client(api_key, base_url):
    return OpenAI(api_key=api_key or "local", base_url=base_url, timeout=180)


def _strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def _filter_think(pieces):
    """Hide <think>...</think> reasoning blocks some models emit, while
    still streaming everything else as it arrives."""
    full, emitted, tag = "", 0, "<think>"
    for piece in pieces:
        full += piece
        visible = re.sub(r"<think>.*?</think>", "", full, flags=re.S)
        cut = visible.find(tag)
        if cut != -1:
            visible = visible[:cut]
        safe = len(visible)
        tail = visible[-(len(tag) - 1):]
        lt = tail.rfind("<")
        if lt != -1 and tag.startswith(tail[lt:]):
            safe -= len(tail) - lt
        if safe > emitted:
            yield visible[emitted:safe]
            emitted = safe
    visible = re.sub(r"<think>.*?</think>", "", full, flags=re.S)
    cut = visible.find(tag)
    if cut != -1:
        visible = visible[:cut]
    if len(visible) > emitted:
        yield visible[emitted:]


def _extra(model):
    """Reasoning models such as Nemotron print their thinking unless it is
    switched off. Switching it off also makes answers several times faster."""
    if "nemotron" in model.lower() or "qwen3" in model.lower():
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


def _create(client, model, **kwargs):
    """Call the API with reasoning switched off where supported; retry
    without that option if a provider rejects it."""
    extra = _extra(model)
    try:
        return client.chat.completions.create(model=model, extra_body=extra, **kwargs)
    except Exception as e:
        if extra and "400" in str(e):
            return client.chat.completions.create(model=model, **kwargs)
        raise


LEAKED_REASONING = re.compile(
    r"^\s*(here'?s a thinking process|thinking process:|okay, (so |let'?s)|"
    r"let me (think|analy[sz]e)|analy[sz]e (the )?user input)", re.I)


def leaked_reasoning(text):
    """True if a model printed its reasoning instead of an answer."""
    return bool(LEAKED_REASONING.search(text or ""))


def _complete(api_key, model, base_url, system, user, max_tokens=1500):
    """One non-streaming call, retried a few times if the free tier's rate
    limit is hit."""
    client = _client(api_key, base_url)
    for attempt in range(4):
        try:
            r = _create(client, model, temperature=0, max_tokens=max_tokens,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": user}])
            return _strip_think(r.choices[0].message.content)
        except Exception as e:
            if "429" in str(e) and attempt < 3:
                time.sleep(6 * (attempt + 1))
                continue
            raise


def _stream(api_key, model, base_url, system, user, max_tokens=1800):
    stream = _create(_client(api_key, base_url), model, temperature=0,
                     max_tokens=max_tokens, stream=True,
                     messages=[{"role": "system", "content": system},
                               {"role": "user", "content": user}])

    def pieces():
        recent = ""
        for event in stream:
            if event.choices and event.choices[0].delta and event.choices[0].delta.content:
                piece = event.choices[0].delta.content
                recent = (recent + piece)[-400:]
                # Stop a model stuck repeating table pipes or the same short text.
                if re.search(r"(\|\s*){40,}$", recent) or re.search(r"(.{2,20}?)\1{15,}$", recent):
                    return
                yield piece

    yield from _filter_think(pieces())


def _earlier(earlier_questions):
    if not earlier_questions:
        return ""
    return ("Earlier questions in this conversation:\n"
            + "\n".join(f"- {q}" for q in earlier_questions) + "\n\n")


def stream_answer(api_key, question, chunks, model=DEFAULT_MODEL,
                  base_url=DEFAULT_BASE_URL, earlier_questions=None):
    """Passage mode: stream an answer built only from the given passages."""
    context = "\n\n".join(f"[{i}] ({_label(c)})\n{c['text']}" for i, c in enumerate(chunks, 1))
    user = f"{_earlier(earlier_questions)}Document excerpts:\n\n{context}\n\nQuestion: {question}"
    yield from _stream(api_key, model, base_url, SYSTEM_PROMPT, user)


# ---- Whole-document mode -------------------------------------------
def make_batches(named_units, batch_chars=BATCH_CHARS):
    """Group page texts into batches that fit one model call. Every block
    is labelled with its source and page. named_units: [(source, [(page, text)])]."""
    batches, cur, size = [], [], 0
    step = batch_chars - 500
    for source, units in named_units:
        for page, text in units:
            text = (text or "").strip()
            if not text:
                continue
            header = f"[{source}, p. {page}]" if page else f"[{source}]"
            for i in range(0, len(text), step):
                block = f"{header}\n{text[i:i + step]}"
                if cur and size + len(block) > batch_chars:
                    batches.append("\n\n".join(cur))
                    cur, size = [], 0
                cur.append(block)
                size += len(block) + 2
    if cur:
        batches.append("\n\n".join(cur))
    return batches


WHOLE_QUESTION = re.compile(
    r"\b(compar\w*|difference\w*|differ\w*|in common|common points|similarit\w*|"
    r"summar\w*|overview|all (the )?(uploaded )?documents|each document|every document|"
    r"list (all|every)|whole document|entire|overall|main points|key points)\b", re.I)


def needs_whole_documents(question):
    """Questions about whole documents (comparisons, summaries, complete
    lists) can't be answered from a handful of passages."""
    return bool(WHOLE_QUESTION.search(question))


def collect_notes(api_key, question, batches, model=DEFAULT_MODEL,
                  base_url=DEFAULT_BASE_URL, progress=None):
    """Read every batch and keep the passages relevant to the question."""
    notes = []
    for i, batch in enumerate(batches):
        out = _complete(api_key, model, base_url,
                        NOTES_PROMPT.format(question=question), batch, 1500)
        if out and not out.strip().upper().startswith("NONE"):
            notes.append(out)
        if progress:
            progress((i + 1) / len(batches))
    return notes


def stream_whole(api_key, question, context, model=DEFAULT_MODEL,
                 base_url=DEFAULT_BASE_URL, earlier_questions=None, from_notes=False):
    """Whole-document mode: answer from the full text, or from the evidence
    collected across all of it."""
    intro = ("Evidence collected from every page of the documents:" if from_notes
             else "Full document text:")
    user = f"{_earlier(earlier_questions)}{intro}\n\n{context}\n\nQuestion: {question}"
    yield from _stream(api_key, model, base_url, WHOLE_SYSTEM_PROMPT, user, 2500)


IDEAS_NOTES_PROMPT = """You are reading part of a document to help a researcher think ahead. Use only the pages given. List briefly, with no introduction:
- Open questions or limitations the text itself states, each with its page, e.g. [p. 12].
- Points the text leaves vague or undefined, with page.
- Assumptions the text relies on, with page."""

IDEAS_FINAL_PROMPT = """Turn these notes into a brainstorm for a researcher. Everything you write is a suggestion, not a statement from the document. Use Markdown with these sections:

## Open questions
## Possible research directions
## Gaps and vague points
## Assumptions worth testing

Each item is one or two sentences and ends with the page reference(s) from the notes that prompted it. Never invent page references. Give 3 to 7 items per section where the notes allow."""

MINDMAP_PROMPT = """Turn this document overview into a mind map outline. Use only the overview. Output only the outline, nothing else: indented lines, two spaces per level, at most 3 levels, at most 7 items per level, each item under 8 words, ending with its page reference like p. 12. Example:
First main topic p. 3
  A subtopic p. 4
    A detail p. 5"""

CHECK_PROMPT = """You check one claim from a researcher's draft against numbered document excerpts. Use only the excerpts.

Reply in exactly this format:
VERDICT: SUPPORTED, PARTLY SUPPORTED, CONTRADICTED or NOT FOUND
REASON: one sentence, citing excerpt numbers like [2]."""

MATRIX_NOTES_PROMPT = """You extract information from part of a document. Use only the pages given. For each field below, copy what these pages say about it, with the page in square brackets, e.g. [p. 4]. If these pages say nothing about a field, write NONE for it.

Fields:
{fields}

Answer in exactly this format, one line per field:
{template}"""

MATRIX_FINAL_PROMPT = """Combine these notes about one document into one short entry per field. Use only the notes. Each entry is at most two sentences and keeps page references exactly as given. If the notes have nothing for a field, write: Not stated

Answer in exactly this format, one line per field:
{template}"""


def build_report(api_key, source, units, notes_prompt, final_prompt,
                 model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL, progress=None):
    """Read a whole document in parts with notes_prompt, then combine the
    notes with final_prompt."""
    batches = make_batches([(source, units)])
    notes = []
    for i, batch in enumerate(batches):
        notes.append(_complete(api_key, model, base_url, notes_prompt, batch, 1500))
        if progress:
            progress((i + 1) / (len(batches) + 1))
    final = _complete(api_key, model, base_url, final_prompt,
                      "\n\n".join(n for n in notes if n), 3000)
    if progress:
        progress(1.0)
    return final


def build_overview(api_key, source, units, model=DEFAULT_MODEL,
                   base_url=DEFAULT_BASE_URL, progress=None):
    """Summary, structure and defined terms, with page references."""
    return build_report(api_key, source, units, OVERVIEW_NOTES_PROMPT,
                        OVERVIEW_FINAL_PROMPT, model, base_url, progress)


def build_ideas(api_key, source, units, model=DEFAULT_MODEL,
                base_url=DEFAULT_BASE_URL, progress=None):
    """Open questions, research directions, gaps and assumptions."""
    return build_report(api_key, source, units, IDEAS_NOTES_PROMPT,
                        IDEAS_FINAL_PROMPT, model, base_url, progress)


# ---- Mind map ------------------------------------------------------
def mindmap_outline(api_key, overview, model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL):
    return _complete(api_key, model, base_url, MINDMAP_PROMPT, overview, 1500)


def _mm(text):
    """Make text safe as a Mermaid mind map node."""
    return re.sub(r"[()\[\]{}<>\"`#;:]", " ", text).strip() or "item"


def outline_to_mermaid(title, outline):
    """Convert an indented outline into Mermaid mind map syntax."""
    lines = ["mindmap", f"  root(({_mm(title)}))"]
    for raw in outline.replace("\t", "  ").splitlines():
        if not raw.strip() or raw.strip().startswith("```"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        level = min(3, indent // 2 + 1)
        text = raw.strip().lstrip("-*• ").strip()
        lines.append("  " * (level + 1) + _mm(text))
    return "\n".join(lines)


# ---- Draft checking ------------------------------------------------
def split_claims(text, max_claims=12):
    """Split a draft into sentences worth checking. Returns
    (claims, number_left_out)."""
    flat = " ".join(text.split())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])", flat)
    claims = [p.strip() for p in parts if len(p.strip()) >= 25]
    return claims[:max_claims], max(0, len(claims) - max_claims)


def check_claim(api_key, claim, passages, model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL):
    """Returns (verdict, reason) for one claim."""
    if not passages:
        return "NOT FOUND", "No passage in the documents matches this claim."
    context = "\n\n".join(f"[{i}] ({_label(c)})\n{c['text']}" for i, c in enumerate(passages, 1))
    out = _complete(api_key, model, base_url, CHECK_PROMPT,
                    f"Excerpts:\n\n{context}\n\nClaim: {claim}", 400)
    m = re.search(r"VERDICT:\s*(PARTLY SUPPORTED|SUPPORTED|CONTRADICTED|NOT FOUND)", out, re.I)
    verdict = m.group(1).upper() if m else "UNCLEAR"
    r = re.search(r"REASON:\s*(.+)", out, re.S)
    return verdict, (r.group(1).strip() if r else out.strip())


# ---- Comparison table (literature matrix) --------------------------
def parse_fields(text, fields):
    out = {f: "Not stated" for f in fields}
    for line in text.splitlines():
        for f in fields:
            m = re.match(rf"\s*[-*]?\s*\**\s*{re.escape(f)}\s*\**\s*:\s*\**\s*(.+)", line, re.I)
            if m:
                value = m.group(1).strip().strip("*").strip()
                if value and value.upper() != "NONE":
                    out[f] = value
    return out


def extract_fields(api_key, source, units, fields, model=DEFAULT_MODEL,
                   base_url=DEFAULT_BASE_URL, progress=None):
    """Read a document and fill one entry per field, with page references."""
    field_list = "\n".join(f"- {f}" for f in fields)
    template = "\n".join(f"{f}: ..." for f in fields)
    batches = make_batches([(source, units)])
    notes = []
    for i, batch in enumerate(batches):
        notes.append(_complete(api_key, model, base_url,
                               MATRIX_NOTES_PROMPT.format(fields=field_list, template=template),
                               batch, 1500))
        if progress:
            progress((i + 1) / (len(batches) + 1))
    final = _complete(api_key, model, base_url, MATRIX_FINAL_PROMPT.format(template=template),
                      "\n\n".join(notes), 1500)
    if progress:
        progress(1.0)
    return parse_fields(final, fields)


# ---- Verification --------------------------------------------------
REVIEW_PROMPT = """You review an answer against the numbered excerpts it was based on. Use only the excerpts.

If every statement in the answer is supported by the excerpts, reply with exactly: OK
Otherwise list each unsupported or wrong statement on its own line, starting with "- ", followed by a short reason."""


def review_answer(api_key, answer, chunks, model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL):
    """Second pass for Precise mode: the model checks its own answer.
    Returns (all_supported, details)."""
    context = "\n\n".join(f"[{i}] ({_label(c)})\n{c['text']}" for i, c in enumerate(chunks, 1))
    out = _complete(api_key, model, base_url, REVIEW_PROMPT,
                    f"Excerpts:\n\n{context}\n\nAnswer to review:\n{answer}", 800)
    return out.strip().upper().startswith("OK"), out.strip()


def _norm(text):
    """Normalise text for word-for-word comparison: case, quote marks,
    PDF line-break hyphens, punctuation and spacing don't count."""
    text = text.lower().replace("\u00ad", "")
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


QUOTE_RE = re.compile(r'[\u201c\u201e"]([^\u201c\u201d"]{12,600}?)[\u201d\u201c"]')


def verify_answer(answer, sources=None, page_texts=None):
    """Check an answer without any model: are its quotations really in the
    source text, and do its citations point to passages or pages that
    exist? sources: passages (Fast/Precise). page_texts: [(source, page,
    text)] (Thorough). Returns a dict, or None for "not found" answers."""
    if not answer or NOT_FOUND.lower() in answer.lower():
        return None
    texts = [c["text"] for c in sources] if sources else [t for _, _, t in (page_texts or [])]
    corpus = _norm(" ".join(texts))
    res = {"quotes": 0, "quotes_ok": 0, "bad_quotes": [], "cites": 0, "bad_cites": 0}

    for q in QUOTE_RE.findall(answer):
        parts = [x for x in re.split(r"\.\.\.|\u2026", q) if len(x.strip()) >= 8]
        if not parts:
            continue
        res["quotes"] += 1
        if all(_norm(x) in corpus for x in parts):
            res["quotes_ok"] += 1
        else:
            res["bad_quotes"].append(q.strip())

    if sources:
        for m in re.finditer(r"\[(\d+(?:\s*[,\u2013-]\s*\d+)*)\]", answer):
            for n in re.findall(r"\d+", m.group(1)):
                res["cites"] += 1
                if not 1 <= int(n) <= len(sources):
                    res["bad_cites"] += 1
    else:
        pages = {p for _, p, _ in (page_texts or []) if p}
        for m in re.finditer(r"\[[^\[\]]*?p\.\s*(\d+)\]", answer):
            res["cites"] += 1
            if pages and int(m.group(1)) not in pages:
                res["bad_cites"] += 1
    return res


def format_check(res):
    """(all_ok, one-line summary) for display."""
    if res is None:
        return True, ""
    parts = []
    if res["quotes"]:
        parts.append(f"{res['quotes_ok']} of {res['quotes']} quotes found word for word in the sources")
    if res["cites"]:
        if res["bad_cites"]:
            parts.append(f"{res['bad_cites']} of {res['cites']} citations point to passages or pages that don't exist")
        else:
            parts.append(f"all {res['cites']} citations point to real passages or pages")
    if not parts:
        return False, "No quotes or citations to check"
    ok = res["quotes_ok"] == res["quotes"] and not res["bad_cites"]
    return ok, "Automatic check: " + "; ".join(parts)


def citation_note(text):
    """A warning to append when an answer cites nothing."""
    if (text and NOT_FOUND.lower() not in text.lower()
            and not re.search(r"\[\d+\]|p\.\s*\d+", text)):
        return "\n\n_This answer cites nothing. Check it against the documents._"
    return ""


def explain_error(e, model):
    msg = str(e)
    if "410" in msg:
        return (f"The model `{model}` has been retired by the provider. Open Settings in the "
                "sidebar and enter a current model id from build.nvidia.com.")
    if "401" in msg or "403" in msg:
        return "The API key was rejected. Check that it is correct and still active."
    if "404" in msg:
        return f"The model `{model}` was not found. Check the model id in Settings."
    if "429" in msg:
        return "The free tier's rate limit was reached. Wait a minute and ask again."
    if "context" in msg.lower() and "length" in msg.lower():
        return ("The text was too long for this model. Use Search passages mode, or pick a "
                "model with a longer context in Settings.")
    return f"The model could not be reached.\n\nDetails: {msg}"
