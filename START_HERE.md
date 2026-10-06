# Start here (Windows, from scratch)

You've got the files in your Downloads folder. Here's how to get from that
to a working tool. About ten minutes the first time, quicker after.

## 1. Give the project a proper home

Downloads is a mess and files get lost there, so move the project out.

- Open File Explorer and go to your user folder (This PC > Windows (C:) >
  Users > your name).
- Make a new folder called doc-chat.
- Move all the downloaded files into it: app.py, rag.py, requirements.txt,
  README.md, START_HERE.md, DEPLOY.md, .env.example, .gitignore.

You should end up with one folder, doc-chat, holding all of those.

## 2. Open the right terminal

This is the part that trips people up, so just do this: click the Windows
Start button, type "Anaconda Prompt", and open it.

That's a terminal where conda already works. Use it for everything below.
You do not need Git Bash. Plain PowerShell often can't find conda until
it's been set up, so don't fight with it, just use the Anaconda Prompt.
(VS Code is great for opening and reading the code, but run the commands
here.)

## 3. Go into the folder

In the Anaconda Prompt, type this and press Enter:

    cd %USERPROFILE%\doc-chat

If you left the files in Downloads, it's this instead:

    cd %USERPROFILE%\Downloads\doc-chat

Type dir and press Enter to check. You should see app.py, rag.py and the
rest listed.

## 4. Build the environment and install the packages

Three commands, one at a time. The first makes a clean, separate Python
just for this project so nothing else on your machine is affected. If it
asks you anything, say yes.

    conda create -n docchat python=3.11 -y
    conda activate docchat
    pip install -r requirements.txt

After the activate line you'll see (docchat) at the start of the prompt.
That means you're in the right place. The install takes a couple of
minutes.

## 5. Get your free NVIDIA key

Go to build.nvidia.com, sign in (no credit card needed), open
build.nvidia.com/settings/api-keys, click to create a key, and copy it. It
starts with nvapi-. Treat it like a password: don't paste it into chats or
commit it to GitHub.

You don't have to put it in a file. When the app opens (next step), there's
a box in the sidebar where you paste the key. It's used only for your
session and isn't saved anywhere.

Optional, so you don't retype it every time you run it locally: make a file
called .env in the doc-chat folder with one line, LLM_API_KEY=your-key.
Easiest is to create it inside VS Code, because Windows otherwise likes to
secretly name it .env.txt, which won't work.

## 6. Run it

Back in the Anaconda Prompt, with (docchat) showing:

    streamlit run app.py

Your browser opens with the app. In the sidebar on the left, paste your key
in the key box, then upload a document and click "Add to knowledge base".
Wait until it says it added some chunks, then type a question in the box at
the bottom.

The first document is a bit slow because it downloads the small local
search model once. After that it's fast.

To stop the app, come back to the Anaconda Prompt and press Ctrl+C.

## Next time

You don't repeat all of that. Just open the Anaconda Prompt and run:

    cd %USERPROFILE%\doc-chat
    conda activate docchat
    streamlit run app.py

## If something goes wrong

- "conda is not recognized": you're not in the Anaconda Prompt. Open that
  one specifically (step 2).
- It asks you to paste your key: do that in the sidebar box. If you made a
  .env instead and it's being ignored, check it's really named .env (not
  .env.txt) and is in the doc-chat folder.
- A PDF adds 0 chunks: it's probably a scanned PDF (images, no real text
  layer). Those need OCR, which this tool doesn't do yet.
- The browser didn't open: look in the Anaconda Prompt for a line like
  "Local URL: http://localhost:8501" and paste that into your browser.
