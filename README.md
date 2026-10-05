# Chat with your documents

This is a small tool for asking questions about a document and getting
answers that come straight from it. You upload a PDF (or a .txt / .md
file), it reads the thing, and then you ask it questions in a chat box.
The answers are based on what's actually in the document, not on whatever
the model happened to know already, and it shows you the exact bits of
text it used so you can check them.

I kept it deliberately small: five short Python packages, two files of
code, no heavy frameworks. Less to install, less to go wrong.

If you just want to get it running, read START_HERE.md instead. This file
explains what it is and how it works.

## What it does, step by step

When you add a document:
- it pulls the text out of the file
- chops that text into small overlapping pieces ("chunks")
- stores each chunk in a little local database (Chroma) that can later
  find pieces by meaning, not just by matching keywords

The part that turns text into something searchable runs on your own
machine. The document itself is never uploaded anywhere for that.

When you ask a question:
- it finds the handful of chunks that look most relevant to the question
- it sends only those chunks, plus your question, to Claude
- Claude answers using only those chunks, and is told to say "I couldn't
  find this in the document" when the answer isn't in them

So the only thing that leaves your laptop is your question and a few short
excerpts, not the whole file.

## Why you can trust the answers

Two reasons. First, the model is told in plain terms to answer only from
the excerpts it's handed and to admit when something isn't covered.
Second, every answer comes with its source passages shown underneath, so
you can read them and see where the answer came from. If it ever
stretches past what the sources say, you'll see it. That's the whole
reason to build it this way instead of pasting text into a normal chatbot.

## What's in the folder

- app.py             the chat screen you actually use
- rag.py             the engine: reading, chunking, searching, asking Claude
- requirements.txt   the packages to install
- README.md          this file
- START_HERE.md      step by step setup, written for Windows from scratch
- DEPLOY.md          how to put it online and share a link (free)
- .env               optional: lets you save your key locally so you don't retype it

## If you want to change something later

The things worth touching are all at the top of rag.py:
- MODEL is which Claude model writes the answer. If you ever get a
  "model not found" error, that's the line to update.
- CHUNK_SIZE and N_RESULTS control how the text is split and how many
  pieces are pulled per question. The defaults are fine; leave them unless
  you have a reason.

Search quality is good enough out of the box because it runs locally for
free. If you ever want it sharper, Anthropic recommends Voyage AI for
embeddings (model voyage-4). That needs its own key and is the one real
upgrade. You don't need it to get going.

## A note for the institute AI rules

The reading and the database stay on your machine. What goes to Anthropic
is your question plus the few retrieved excerpts, which it needs to write
the answer. Anthropic's commercial API terms say they don't train on API
inputs or outputs by default, but check the current terms and your
institute's rules before pointing this at anything unpublished or
confidential.
