# PDF Q&A agent -- web application edition

This is the same PDF question-answering idea as `agent.py`, reshaped into a
small web app: a FastAPI backend (`web_app.py`) and a single static HTML
page (`static/index.html`) that talks to it. `agent.py` is untouched -- the
two are independent entry points into the same underlying design, and you
can run either one (or both) against the same Chroma server.

## Running it

```bash
# 1. Start the Chroma server (same one agent.py uses)
docker compose up -d

# 2. Install dependencies (adds fastapi/uvicorn/python-multipart on top of
#    what agent.py already needs)
pip install -r requirements.txt

# 3. Set your API key and start the web server
export OPENAI_API_KEY=sk-...
uvicorn web_app:app --reload --port 8001

# 4. Open the app
open http://localhost:8001
```

In the browser: upload a PDF, wait for it to finish indexing, click it in
the sidebar, then ask questions. Uploading the exact same file again (even
renamed) is instant, because it's recognized as already indexed -- see
"Why hash the file contents?" below.

## Why a web app needs a different shape than a CLI script

`agent.py` is a script: it runs top to bottom, does one thing, and exits.
Everything it needs -- the Chroma connection, the embeddings client, the
one PDF you named on the command line, the conversation loop -- lives in
local variables for the duration of that one run.

A web server is a different shape of program: it starts once and then
handles many independent requests, from possibly-different browser tabs,
for as long as it stays up. That changes three things:

1. **Connections become long-lived, shared resources** instead of
   locals. The Chroma client, the embeddings client, and the LLM are each
   created *once*, when the server starts, and every request reuses them.
   FastAPI's `lifespan` hook is where that happens (`web_app.py`, `lifespan()`)
   -- it's the web-server equivalent of the top of `agent.py`'s `main()`.

2. **"Which PDF?" has to travel with every request**, rather than being
   fixed for the whole run. `agent.py` is told the PDF once, as a CLI
   argument. The web app instead identifies each indexed PDF by an id
   (`document_id`) and expects the browser to say which one it means on
   every question.

3. **Conversation history has to be stored somewhere between requests.**
   In `agent.py`, the Python `while True` loop is itself the memory -- the
   messages list lives in a local variable across iterations. An HTTP
   request has no memory of the last one on its own, so the web app keeps
   a small in-memory dictionary (`sessions`, keyed by a `session_id` the
   browser generates and stores in `localStorage`) that plays the same
   role the loop's local variable played in the CLI version.

## Request flow

**Uploading a PDF** (`POST /api/documents`):

```
Browser                     FastAPI (web_app.py)                  Chroma
  |  PDF file (multipart)          |                                  |
  |-------------------------------> |                                  |
  |                                 | hash file content -> document_id |
  |                                 | get_or_create_collection --------> |
  |                                 | <--- already has data? -----------|
  |                                 |   yes -> skip straight to "done"  |
  |                                 |   no  -> PyPDFLoader + splitter   |
  |                                 |          -> OpenAIEmbeddings      |
  |                                 |          -> collection.add -------> |
  |  <-- {document_id, chunks} -----|                                  |
```

**Asking a question** (`POST /api/chat`):

```
Browser                     FastAPI (web_app.py)                  Chroma / OpenAI
  |  {session_id, document_id, question}                            |
  |-------------------------------->|                                |
  |                                 | look up session's message history
  |                                 | (reset if document_id changed)  |
  |                                 | append the new question         |
  |                                 | build a search_pdf tool scoped  |
  |                                 |   to this document's collection |
  |                                 | agent.invoke({"messages": ...}) |
  |                                 |    -> LLM may call search_pdf ---> Chroma query
  |                                 |    -> LLM reads results, answers -> OpenAI chat
  |                                 | append the answer to history    |
  |  <---- {answer} ----------------|                                |
```

## Design decisions worth calling out

**Why hash the file's contents for the document id, instead of using the
filename (like `agent.py` does)?** `agent.py` only ever deals with one
person at a time, at a terminal, so naming a Chroma collection after the
file's *name* is enough. A web app can get the same PDF uploaded by
different people, or the same content under a different filename, or a
different file that happens to share a name. Hashing the bytes
(`collection_id_for`) means "have we indexed *this* content before?" has
one unambiguous answer, and a re-upload of something already indexed is
recognized immediately rather than silently re-embedded (or worse, silently
colliding with an unrelated file of the same name).

**Why is conversation memory in-memory (a plain Python dict), and what
does that cost you?** It's the simplest thing that demonstrates the idea,
and it's genuinely fine for one person running this locally. It does mean:
history is lost if you restart the `uvicorn` process, it isn't shared
across multiple server processes (so it won't survive being deployed
behind a load balancer with more than one worker), and there's no
per-user isolation or authentication -- anyone who can reach the server and
guess/receive a `session_id` can see that conversation. None of that
matters for a local learning project; all of it matters before putting
this in front of real users. The natural next step is a shared, persistent
store (Redis, a database table, or LangGraph's own checkpointer/threads
support) keyed the same way, plus real accounts if more than one person
will use it.

**Why rebuild the agent (and its search tool) on every request instead of
caching one per document?** Building `create_agent(...)` doesn't make any
network calls -- it's just wiring together Python objects (the shared LLM,
a small closure around the collection) -- so it's cheap enough to not
bother caching, and rebuilding it means there's no way for a stale tool to
end up pointing at the wrong document after someone switches which PDF
they're chatting with.

**Why does switching documents mid-session reset the conversation?**
Carrying "the treaty was signed in 1848" as prior context into a
conversation about a completely different PDF would actively mislead the
agent. Since the chat history is really "history of this document,
in this session," changing documents starts a clean one.

## Extending this

- **Persist sessions**: swap the `sessions` dict for Redis or a database
  table, keyed the same way, so history survives a restart and works
  across more than one server process.
- **Multiple documents per question**: build one `search_pdf` tool per
  selected document (or one tool that searches across several
  collections) and pass all of them to `create_agent`.
- **Streaming answers**: `create_agent`'s underlying LangGraph runnable
  supports streaming; wiring that through Server-Sent Events or a
  WebSocket to the frontend would make answers appear incrementally
  instead of all at once.
- **Auth and per-user isolation**: put a real user id in front of
  `session_id` (and maybe `document_id`) so people can't see each other's
  documents or conversations.
- **Delete/expire documents**: there's currently no endpoint to remove an
  indexed PDF's collection; for a shared/public deployment you'd want one,
  plus some retention policy.
