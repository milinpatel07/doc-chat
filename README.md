# Document chat

Developed by Milin Patel.

A research tool for asking questions about your own documents and getting
answers that come only from them. It's meant for any researcher, in any
field. You upload PDF, Word, TXT or MD files,
ask a question, and get an answer with page references and the exact
passages it was based on, so you can check it yourself.

To get it running, read START_HERE.md. To put it online, read DEPLOY.md.

## What it does

- Answers only from your documents. If nothing matches the question well
  enough, it says "I couldn't find this in the document" and doesn't ask
  the model at all. Off-topic questions get the same answer.
- Copes with spelling mistakes. Each word in the question is checked
  against the documents' own vocabulary, so typos in technical terms get
  fixed too. The answer shows what it searched for.
- Searches by meaning and by exact keywords, so both loose questions and
  exact terms or clause numbers like "3.30" are found.
- Understands short follow-ups such as "and its examples?".
- Shows each step while it works, then writes the answer as it goes.
- Cites every statement and shows each passage with file name, page and
  match score. "Show page" displays the original PDF page, figures included.
- Flags any answer that cites nothing.
- Strips repeated headers, footers and download stamps from PDFs.
- Never stores the same passage twice.
- Two ways of answering. Fast mode is fast and suits specific
  questions. Thorough mode goes through every page, for summaries,
  complete lists and comparisons.
- Document overview: summary, structure and defined terms of a document,
  with page references, made once and kept.
- Lets you choose which documents to search, and remove single documents.
- Exports the conversation as Markdown, recording the tool version, model,
  date, mode and the full text of every passage used, so any answer can be
  checked later without the tool.
- Library mode (your own computer): documents stay indexed between
  restarts and every chat is saved in chat_logs. See START_HERE.md.

## Speed, accuracy and checking

- Auto (default): picks Fast for specific questions and Thorough for
  comparisons, summaries and complete lists.
- Fast: 6 passages, one model call, usually seconds.
- Precise: 10 passages, plus a second pass in which the model checks its
  own answer and lists anything unsupported. About twice as long.
- Thorough: reads every page, a minute or more for long documents.

Every answer gets an automatic check that uses no AI: quotations are
searched word for word in the source text, and citations are checked to
point to real passages or pages. The result appears under the answer, along
with how long it took. How often the search finds the right page can be
measured with eval.py, so accuracy is a number, not a claim.

## Tools (the Tools view)

- Document overview: summary, structure and defined terms, with pages.
- Mind map: the document's topics as a map, every branch with a page.
- Ideas and open questions: research suggestions, clearly marked as such.
- Check a draft: each sentence of your own writing checked against the
  documents.
- Compare documents: the same fields for several documents in one table,
  exported for Excel.

The About view explains all of this, plus the limits and data handling,
inside the app, so anyone using your link can read it.

## How it works

When you add a document, the text is pulled out page by page, cleaned, and
cut into small overlapping sections. Each section is stored in a local
search index (Chroma) that finds passages by meaning, not just keywords.
This part runs on your machine.

When you ask a question, the closest-matching sections are picked. Weak
matches are thrown away. Only the remaining sections and your question go
to the language model, which is told to answer from them alone, quote
definitions word for word, and cite everything.

## Settings (sidebar, under Settings)

- Minimum match to answer: how close a passage must be before it's used.
  Raise it for stricter answers, lower it if the app refuses questions it
  should be able to answer. The default is 0.25.
- Model: the model that writes answers. Free providers retire models now
  and then. If you see "retired" or "410 Gone", paste a current model id
  from build.nvidia.com here.
- API address: any OpenAI-compatible service. NVIDIA is the default. For
  a fully private setup, run Ollama on your machine and use
  http://localhost:11434/v1.

## Known limits

- Scanned PDFs (images with no text layer) can't be read yet. They need OCR.
- A section never spans two pages, so a definition split across a page
  break may come back in two parts.
- Fast mode mode uses the 6 best-matching passages. For summaries
  or complete lists, switch to Thorough mode.
- Thorough mode makes one model call per roughly 15 pages, so a
  150-page standard takes about 10 calls and a minute or two. It uses more
  of the free quota.
- The same question can give slightly different wording on different days,
  and a different answer if the provider replaces the model. The export
  records the model and date for that reason.
- How much it can hold depends on memory. On your own computer, dozens of
  standards are fine. The free online version suits a handful of large
  PDFs per visitor.

## Data and confidentiality

Indexing and search stay on your machine. What leaves it is your question
and the few selected passages, sent to the model provider. Free tiers may
use that data to improve their models. Don't use them for licensed,
unpublished or confidential documents. For those, use a provider that
contractually doesn't train on inputs, or a local model through Ollama.
Check your institute's rules first.

## Checking the search (for research use)

eval.py tells you how often the search finds the right page. Make a CSV
with the columns question, source and page, using questions whose answer
you've looked up yourself (questions_example.csv shows the format). Then,
in library mode:

    python eval.py questions.csv

Run it again after changing any setting. If the score drops, undo the
change.

## Files

- app.py            the interface
- rag.py            reading, cleaning, chunking, search, answering
- requirements.txt  packages to install
- eval.py           measures search quality on your own question list
- ROADMAP.md        what's done, what's next, and what's left out on purpose
- .streamlit/       theme settings
- .env              optional: your saved key and library mode (never uploaded)
- library/          library mode: saved index and PDFs (never uploaded)
- chat_logs/        library mode: saved chats (never uploaded)
