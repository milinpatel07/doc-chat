"""
check_models.py - find out which chat models actually work with your key.

Developed by Milin Patel.

The model list from the API shows the whole catalog, not what your key can
use. This script sends a tiny test message to every chat model in that list
and keeps the ones that answer. The working ones are saved to
working_models.txt, and the app then offers them in a dropdown.

Usage (Anaconda Prompt, in the doc-chat folder, with docchat active):
    python check_models.py

Takes a few minutes. Run it again now and then, because providers add and
retire models.
"""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

import rag

# Names containing these are not chat models (search, safety filters,
# images, speech, code, translation, document parsing, scoring).
NOT_CHAT = ("embed", "guard", "safety", "parse", "clip", "reward", "translate",
            "detector", "vision", "vlm", "kosmos", "deplot", "fuyu", "neva", "vila",
            "code", "starcoder", "diffusion", "calibration", "cosmos", "omni",
            "topic-control", "retriever", "gemma-2b", "recurrentgemma")


def main():
    load_dotenv()
    key = os.environ.get("LLM_API_KEY", "").strip()
    if not key or "YOUR" in key.upper():
        print("No real key found. Put LLM_API_KEY=nvapi-... in your .env file first.")
        return
    client = OpenAI(api_key=key, base_url=rag.DEFAULT_BASE_URL, timeout=60)
    ids = sorted(m.id for m in client.models.list())
    chat = [i for i in ids if not any(word in i.lower() for word in NOT_CHAT)]
    print(f"{len(ids)} models in the catalog, {len(chat)} look like chat models. Testing each…\n")

    working = []
    for model in chat:
        start = time.time()
        try:
            r = rag._create(client, model, temperature=0, max_tokens=8,
                            messages=[{"role": "user", "content": "Reply with the word OK."}])
            text = (r.choices[0].message.content or "").strip()
            seconds = time.time() - start
            print(f"  works    {model}  ({seconds:.1f} s)")
            working.append(model)
        except Exception as e:
            msg = str(e)
            reason = ("retired" if "410" in msg else "no access" if ("401" in msg or "403" in msg)
                      else "not found" if "404" in msg else "rate limit" if "429" in msg
                      else msg.splitlines()[0][:60])
            print(f"  fails    {model}  ({reason})")
            if "429" in msg:
                time.sleep(10)

    with open("working_models.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(working) + "\n")
    print(f"\n{len(working)} models work with your key. Saved to working_models.txt.")
    print("Restart the app to pick one from the dropdown in Settings.")


if __name__ == "__main__":
    main()
