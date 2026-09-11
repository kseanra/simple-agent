# PDF question-answering agent -- Docker + Chroma edition

A variant of the PDF Q&A agent that uses a **real, persistent vector
database** (Chroma) running in a Docker container, instead of an
in-process Python object. Requires Docker Desktop (or any Docker engine)
to be installed and running.

> **Looking for the web app?** This README covers `agent.py`, the
> terminal/CLI version. There's also a browser-based version of the same
> agent -- see [`WEBAPP.md`](./WEBAPP.md) for `web_app.py` and its design.

## Why this instead of the plain in-memory version?

- **Persistence.** Indexed PDFs survive between runs -- ask questions
  today, close your terminal, come back tomorrow, and the same PDF's
  embeddings are still there (no re-embedding, no repeat API cost).
- **Real database semantics.** Multiple collections (one per PDF),
  metadata filtering, and a query API built for this purpose, rather than
  a small in-memory dictionary.
- **No compiled dependencies on your Mac.** The Chroma *server* -- with
  its `onnxruntime` dependency and native build requirements -- runs
  inside the Docker container's Linux environment, where wheel
  availability is excellent regardless of your host OS or its version.
  Your Mac's Python environment only needs `chromadb-client`, a
  lightweight HTTP-only client with a minimal dependency footprint.

## Setup

```bash
# 1. Start the Chroma server (runs in the background)
docker compose up -d

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Set your API key and run
export OPENAI_API_KEY=sk-...
python agent.py path/to/document.pdf
```

Type `exit` to quit. Run it again on the same PDF and you'll see it skip
re-indexing, because the data is still sitting in the Chroma server's
Docker volume.

To stop the server: `docker compose down` (add `-v` to also delete the
persisted data volume).

## How it works, step by step

| Step | What happens | Key pieces |
|---|---|---|
| 1. Load | PDF parsed into one `Document` per page | `PyPDFLoader` |
| 2. Split | Pages broken into ~1000-character overlapping chunks | `RecursiveCharacterTextSplitter` |
| 3. Embed | Chunks converted to vectors via the OpenAI API | `OpenAIEmbeddings` |
| 4. Store & search | Vectors stored in / queried from the Chroma server | `chromadb.HttpClient`, a collection per PDF |
| 5. Retriever tool | The search function is exposed to the agent as a tool | `@tool` decorator |
| 6. Agent | LLM + tool + system prompt, in a call-observe-repeat loop | `create_agent` |

### Why a custom tool instead of `create_retriever_tool`?

The earlier in-memory version used LangChain's `create_retriever_tool`,
which wraps a LangChain `VectorStoreRetriever` object. That object
normally comes from the official `langchain-chroma` integration -- but
that package requires the *full* `chromadb` library as a dependency,
reintroducing the very `onnxruntime`/compiled-binary problem this Docker
setup exists to avoid. So instead, `search_pdf` is a plain Python function
that calls the Chroma HTTP client directly, decorated with `@tool` so the
agent can still call it like any other tool. The docstring on that
function is what the agent's LLM reads to decide when to use it.

### Why one collection per PDF?

Chroma collections are cheap and isolated. Naming each collection after
its source file (`collection_name_for`) means:
- Re-running on the same PDF finds the existing collection and skips
  re-embedding.
- Running on a different PDF gets its own collection -- no risk of one
  document's chunks leaking into another's search results.

## Extending this

- **Multiple documents in one session**: build one tool per collection
  (or one tool that searches across several) and pass a list of tools to
  `create_agent`.
- **Client/server version matching**: `chromadb-client`'s version should
  stay reasonably close to the server image's version (both are pinned
  loosely here via `:latest` and `>=1.0.0` -- for a production setup,
  pin both explicitly and upgrade them together).
- **Remote Chroma server**: since this already talks over HTTP, pointing
  at a Chroma instance running on another machine (or Chroma Cloud) is
  just a matter of changing `CHROMA_HOST`/`CHROMA_PORT`.
- **Conversation memory / page citations**: same suggestions as the
  in-memory version -- see that project's README.
- **A web-based UI**: see [`WEBAPP.md`](./WEBAPP.md) -- a FastAPI +
  browser version of this same agent, so you can ask questions from a
  page instead of a terminal.
