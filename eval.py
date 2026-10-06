"""
eval.py - measure how well the search finds the right page.

Developed by Milin Patel.

You write a small list of questions where you have checked the answer
yourself (the document and the page it is on). This script runs each
question through the same search the app uses and reports how often the
right page is among the passages that would be sent to the model.

It needs no API key and makes no model calls: it tests the search, which
is the part that decides whether an answer can be right at all.

Requirements: library mode (LOCAL_MODE=1), with the documents already
added through the app.

Usage (in the Anaconda Prompt, inside the doc-chat folder):
    python eval.py questions.csv
"""

import csv
import datetime
import sys

import chromadb
from chromadb.utils import embedding_functions

import rag


def main(path):
    client = chromadb.PersistentClient(path="library")
    col = client.get_or_create_collection(
        name="library", embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )
    data = col.get(include=["documents", "metadatas"])
    if not data["ids"]:
        print("The library is empty. Add documents in the app with LOCAL_MODE=1 first.")
        return
    kw = rag.KeywordIndex(data["ids"], data["documents"], data["metadatas"])

    with open(path, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("question", "").strip()]
    if not rows:
        print("No questions found. The file needs the columns: question, source, page")
        return

    found, ranks, report = 0, [], []
    for r in rows:
        question = r["question"].strip()
        source = (r.get("source") or "").strip() or None
        expected = int(r["page"])
        corrected, _ = rag.correct_query(question, kw)
        hits = rag.retrieve(col, kw, corrected, [source] if source else None)
        pages = [h["page"] for h in hits]
        rank = pages.index(expected) + 1 if expected in pages else None
        if rank:
            found += 1
            ranks.append(rank)
        report.append((question, expected, pages, rank))

    print(f"\n{rag.APP_NAME} {rag.APP_VERSION} search evaluation, "
          f"{datetime.datetime.now():%Y-%m-%d %H:%M}")
    print(f"Settings: top {rag.N_RESULTS} passages, minimum match {rag.MIN_SIMILARITY}\n")
    for question, expected, pages, rank in report:
        status = f"found at position {rank}" if rank else "MISSED"
        print(f"- {question}\n    expected p. {expected}, got pages {pages}: {status}")
    print(f"\nRight page found for {found} of {len(rows)} questions "
          f"({found / len(rows):.0%}).")
    if ranks:
        mrr = sum(1 / r for r in ranks) / len(rows)
        print(f"Mean reciprocal rank: {mrr:.2f} (1.00 means always first).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eval.py questions.csv")
    else:
        main(sys.argv[1])
