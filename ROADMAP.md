# Roadmap

Developed by Milin Patel.

The goal is a general document tool any researcher, in any field, can rely
on for years: for theses, literature reviews, papers and standards, and
eventually for whole groups. The
rule behind every decision is that a tool you can trust beats a tool with
many features. Each round has to work and be tested before the next starts.

## Done

0.8 (fixes from real use)
- Model reasoning is switched off for Nemotron and Qwen3 models. It was
  being printed instead of the answer, and made answers several times slower.
- Empty table rows, pipe runs and dot leaders are removed before indexing.
  Converted Markdown files were full of them. Documents added before 0.8
  must be removed and added again to benefit.
- A guard stops the answer if the model starts repeating itself.
- Spelling correction leaves short and common English words alone (it had
  turned "read" into "tread").
- Auto mode, the new default: specific questions use Fast; comparisons,
  summaries and complete lists use Thorough, or the overviews when every
  selected document has one.
- A question interrupted by a click is kept and can be answered with one click.

0.7
- Three modes with the time shown after every answer: Fast, Precise (more
  passages plus a second pass where the model checks its own answer) and
  Thorough (every page).
- Automatic check on every answer, using no AI: each quotation is searched
  word for word in the source text, and each citation is checked to point
  to a passage or page that exists. Shown under the answer and in exports.

0.6
- Three views: Chat, Tools and About.
- About page inside the app: what it does, how it answers, limits, data
  handling, version and credit, readable by anyone using the link.
- Folders for documents, searching by folder, moving and removing documents.
- Chat history in library mode: past chats can be reopened from the sidebar.
- Mind map built from the overview, every branch with a page number,
  downloadable as an outline or a Mermaid file.
- Ideas and open questions per document, clearly marked as suggestions.
- Draft checker: each sentence of your own text marked as supported, partly
  supported, contradicted or not found, with the passages shown.
- Compare documents: the same fields filled for several documents, exported
  as CSV for Excel, like a literature review table.

0.5
- Two ways of answering: search passages (fast, specific questions) and
  read whole documents (every page, for summaries, complete lists and
  comparisons).
- Document overview, made once per document: summary, structure and
  defined terms, all with page references.
- Exports record the tool version, model, provider, date, mode and the full
  text of every passage used, so any answer can be checked again later
  without the tool.
- eval.py: measures how often the search finds the right page, using
  questions you have checked yourself.

0.4 and earlier
- Answers only from the documents, refusing when nothing matches.
- Spelling correction against the documents' own vocabulary.
- Search by meaning and by keywords, including clause numbers.
- Follow-up questions, page references, page previews.
- Library mode with saved documents and chat logs.

## Next

Round 3: research features
- Cross-document glossary: every defined term across all documents, with
  conflicts between definitions flagged.
- Quote collector: save passages with citations into notes while working.
- Export citations to BibTeX or Zotero.
- OCR for scanned PDFs, PowerPoint and LaTeX files, web pages by URL.
- Extend eval.py to measure answer quality, not only page hits.
- Library access from Claude Code and other tools through MCP.

Round 4: running it for a group
- A Docker image for the institute server, with a local model, so
  documents never leave the institute. That also allows licensed
  standards. Needs approval from the institute management and IT first.
- Logins and shared folders for colleagues.

Round 5: a proper web interface
- Only if the tool is used by others. A dedicated web app with chat
  history, inline citations and a better document view. The search and
  answering code stays the same.

## Things deliberately left out

- Generated images. They would be invented, and the point of the tool is
  that nothing is. Page previews show the real figures instead.
- Remembering facts across chats outside the documents. That is how
  unsupported claims get into answers.
- Knowledge graphs or mind maps, unless every link in them can be traced
  to a page.

## Decisions

Recorded so they can be reviewed and changed.

1. General tool, not tied to one field. Field-specific features such as
   extracting "shall" requirements from standards are optional extras for
   later, because they help only some users.
2. Ideas and brainstorming are kept in their own tool, never mixed into chat
   answers, so a suggestion can't be mistaken for something the document
   says.
3. The mind map is built from the overview, not straight from the document,
   so it only contains what the overview already traced to pages.
4. Draft checking covers at most 12 sentences per run, to keep it within
   free-tier limits and keep each check reviewable.
5. Chats and folders are saved only in library mode. The shared online
   version keeps nothing between sessions, on purpose.
6. Views are switched with a radio control rather than tabs, because tabs
   jump back to the first one whenever the app reruns.

7. Accuracy is never stated as a fixed percentage. It depends on the
   documents, questions and model, so it is measured with eval.py instead,
   and every answer is checked automatically.

8. Auto is the default mode, because users can't be expected to know which
   mode a question needs. Fast and Thorough stay available for control.

## How to judge a change

Before and after any change to search settings or code, run
`python eval.py questions.csv` on the same question list. If the score
goes down, the change goes back. Over time, grow the question list from
real questions you've asked and checked.
