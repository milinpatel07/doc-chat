# Putting it online so you can share a link

Because each person uses their own free NVIDIA key, there's no secret for you
to hide, which makes this simple. You put the code in a public GitHub repo and let
Streamlit host it for free. About fifteen minutes the first time.

What you end up with: a link like https://doc-chat.streamlit.app that
anyone can open in a browser, on a phone or a laptop, nothing to install.
They paste their own free NVIDIA key (from build.nvidia.com), upload a
document, and ask questions. Each person's documents stay in their own
session and are never shared with anyone else.

## 1. Make a GitHub account

If you don't have one, sign up at github.com. Free.

## 2. Create a repository and upload the files

- On github.com, click the + at the top right, then "New repository".
- Name it doc-chat. Set it to Public. Click "Create repository".
- On the next page, click the "uploading an existing file" link.
- From your D:\doc-chat folder, drag in these files:
  app.py, rag.py, requirements.txt, README.md.
  (You can add START_HERE.md and DEPLOY.md too, they don't hurt.)
- Do NOT upload .env. That's your personal key and it must never go in a
  public repo. The online version doesn't need it anyway, because each
  user types their own key.
- Click "Commit changes" at the bottom.

## 3. Deploy on Streamlit

- Go to share.streamlit.io and sign in with your GitHub account.
- Click "Create app" (or "Deploy an app").
- Pick your doc-chat repository, branch main, and set the main file to
  app.py.
- If there's an advanced option for the Python version, choose 3.11.
- Click Deploy. It builds for a few minutes while it installs the
  packages. The first build is the slow one.
- When it finishes you get your public link. Open it, paste your key,
  test it with a document, then send the link to whoever you want.

## Good to know

- The app goes to sleep after about 12 hours with no visitors and shows a
  "wake it up" button to the next person. One click and it's back. That's
  normal for the free tier.
- Uploaded documents only live for that person's session. Close the tab
  and they're gone. This is a chat-with-a-document tool, not a permanent
  library, and that's also good for privacy.
- When you change the code later, just upload the new file to the same
  GitHub repo (or edit it right on github.com) and Streamlit redeploys by
  itself.
- Memory on the free tier is limited, so very large PDFs (hundreds of
  pages) can be slow or too big. Normal papers and reports are fine.

## If the build fails

Streamlit shows build logs during deploy. Scroll to the last error and
read it. Usually it's one package wanting a specific version. Paste the
error to me and I'll give you the exact fix (often a single pinned line in
requirements.txt).

## A word on your institute's rules

Anyone with the link can use it, and the documents they upload get sent in
pieces to NVIDIA under their own key. Free tiers may use what you
send to improve their models, so this is fine for public or non-sensitive
material only. For confidential or unpublished work, use a provider that
doesn't train on your data (such as paid Anthropic) or run it fully local
with a model like Ollama, and don't use the public link.
