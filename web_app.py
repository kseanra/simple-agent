"""
PDF Question-Answering Agent -- Web application edition
========================================================

Same idea as agent.py (load a PDF, index it, let an LLM agent answer
questions about it), reshaped so a browser drives it over HTTP instead of
a terminal loop. agent.py is untouched -- run it any time you want the
plain CLI version; this is a second, independent entry point that reuses
the same surrounding infrastructure (the Chroma server in docker-compose.yml)
but is its own small application.

Architecture:

  Browser (static/index.html)
     |
     |  POST /api/documents   (upload a PDF)
     |  GET  /api/documents   (list previously indexed PDFs)
     |  POST /api/chat        (ask a question about one of them)
     v
  FastAPI app (this file)
     -> Load & split      (PyPDFLoader + RecursiveCharacterTextSplitter)
     -> Embed              (OpenAIEmbeddings)
     -> Store & search     (Chroma server in Docker -- same one agent.py uses)
     -> Retriever tool      (a small @tool-wrapped function, per request)
     -> Agent (LLM)         (create_agent, LangChain v1's agent API)
     -> Answer

What's different from agent.py, and why:

  - Indexing happens on file upload, not a CLI argument. Each PDF's
    identity is a hash of its *content* (not its filename), so uploading
    the same file twice -- even renamed, even from a different browser --
    reuses the existing collection instead of re-embedding it.

  - Chroma/embeddings/LLM connections are created once at server startup
    (a FastAPI "lifespan" hook) and reused for every request, rather than
    once per script run -- the same objects, just with a longer lifetime
    because the process itself is long-running.

  - Conversation state (the running list of messages) has to live
    *somewhere* between HTTP requests, since each request is otherwise
    stateless. Here it's an in-memory dict keyed by a session id the
    browser generates and keeps. That's the simplest thing that works for
    one server process and is enough to demonstrate the idea -- see the
    "Limitations" section in WEBAPP.md for what a production version would
    do differently (a shared store like Redis, real user accounts, etc.).

Prerequisites:
    docker compose up -d          (starts the Chroma server, see docker-compose.yml)
    export OPENAI_API_KEY=sk-...
    pip install -r requirements.txt
    uvicorn web_app:app --reload --port 8001
    open http://localhost:8001
"""

import hashlib
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain.agents import create_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
SEARCH_K = 4

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a PDF document. "
    "Use the search_pdf tool to find relevant passages before answering. "
    "If the answer isn't in the document, say so clearly instead of "
    "guessing. Keep answers concise and, where useful, mention which part "
    "of the document supports your answer."
)


# ---------------------------------------------------------------------------
# Shared, process-lifetime resources: one Chroma connection, one embeddings
# client, one LLM. Created in the "lifespan" hook below (FastAPI's
# startup/shutdown mechanism) rather than at import time, so the app object
# can be imported/tested without immediately requiring a live Chroma server.
# ---------------------------------------------------------------------------
class AppState:
    chroma_client: Optional[chromadb.HttpClient] = None
    embeddings: Optional[OpenAIEmbeddings] = None
    llm: Optional[ChatOpenAI] = None
    # session_id -> {"document_id": str, "messages": [{"role": ..., "content": ...}, ...]}
    sessions: Dict[str, Dict] = {}


state = AppState()


def connect_to_chroma() -> chromadb.HttpClient:
    """
    Same fail-fast philosophy as agent.py: if the Chroma container isn't
    running, say so clearly rather than surfacing a confusing stack trace
    from deep inside the first request that needs it.
    """
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        client.heartbeat()
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach the Chroma server at {CHROMA_HOST}:{CHROMA_PORT}. "
            f"Start it with: docker compose up -d ({exc})"
        ) from exc
    return client


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(exist_ok=True)
    state.chroma_client = connect_to_chroma()
    state.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    state.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    state.sessions = {}
    yield
    state.chroma_client = None
    state.embeddings = None
    state.llm = None
    state.sessions = {}


app = FastAPI(title="PDF Q&A Web Agent", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Request/response shapes
# ---------------------------------------------------------------------------
class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunks: int
    reused: bool = False


class ChatRequest(BaseModel):
    session_id: str
    document_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str


# ---------------------------------------------------------------------------
# Indexing: same steps as agent.py's load_and_split + get_or_build_collection,
# triggered by an uploaded file's bytes instead of a path on disk, and keyed
# by a content hash instead of a filename so re-uploads (even renamed, even
# from a different visitor) are recognized and never re-embedded.
# ---------------------------------------------------------------------------
def collection_id_for(filename: str, content: bytes) -> str:
    base = re.sub(r"[^a-zA-Z0-9_-]", "_", os.path.splitext(filename)[0]).lower() or "doc"
    digest = hashlib.sha256(content).hexdigest()[:12]
    return f"pdf_{base}_{digest}"


def index_pdf(document_id: str, content: bytes) -> List:
    """Writes the upload to a temp path just long enough for PyPDFLoader to
    read it (it needs a real file path, not bytes), then removes it --
    nothing about the original file is kept around beyond the chunks
    already stored in Chroma."""
    tmp_path = UPLOAD_DIR / f"{document_id}.pdf"
    tmp_path.write_bytes(content)
    try:
        pages = PyPDFLoader(str(tmp_path)).load()
    finally:
        tmp_path.unlink(missing_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(pages)


def get_collection_or_404(document_id: str):
    try:
        return state.chroma_client.get_collection(name=document_id)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"No indexed document with id '{document_id}'. Upload it first.",
        )


# ---------------------------------------------------------------------------
# Agent: identical shape to agent.py's build_search_tool + build_agent, just
# scoped to whichever document's collection the current request names.
# Rebuilding this per request is cheap (no network call, just Python
# closures) and avoids any risk of a stale tool pointing at the wrong
# collection after a session switches documents.
# ---------------------------------------------------------------------------
def build_search_tool(collection):
    @tool
    def search_pdf(query: str) -> str:
        """Search the uploaded PDF document for relevant passages. Use
        this whenever the user asks a question that might be answered by
        the document's contents. Input should be a focused search query,
        not the full user question verbatim."""
        query_vector = state.embeddings.embed_query(query)
        results = collection.query(query_embeddings=[query_vector], n_results=SEARCH_K)
        docs = results.get("documents", [[]])[0]
        if not docs:
            return "No relevant passages found in the document."
        return "\n\n---\n\n".join(docs)

    return search_pdf


def build_agent(tool_fn):
    return create_agent(model=state.llm, tools=[tool_fn], system_prompt=SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def index_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/documents", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    document_id = collection_id_for(file.filename, content)
    collection = state.chroma_client.get_or_create_collection(
        name=document_id, metadata={"source_filename": file.filename}
    )

    if collection.count() > 0:
        return DocumentInfo(
            document_id=document_id,
            filename=file.filename,
            chunks=collection.count(),
            reused=True,
        )

    chunks = index_pdf(document_id, content)
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]
    vectors = state.embeddings.embed_documents(texts)
    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)

    return DocumentInfo(
        document_id=document_id, filename=file.filename, chunks=len(chunks), reused=False
    )


@app.get("/api/documents", response_model=List[DocumentInfo])
def list_documents():
    docs = []
    for summary in state.chroma_client.list_collections():
        name = getattr(summary, "name", summary)
        try:
            collection = state.chroma_client.get_collection(name=name)
        except Exception:
            continue
        meta = collection.metadata or {}
        docs.append(
            DocumentInfo(
                document_id=name,
                filename=meta.get("source_filename", name),
                chunks=collection.count(),
            )
        )
    return docs


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty.")

    collection = get_collection_or_404(req.document_id)

    session = state.sessions.setdefault(
        req.session_id, {"document_id": None, "messages": []}
    )
    if session["document_id"] != req.document_id:
        # Switching which document this session is chatting with starts a
        # fresh conversation -- carrying old messages over would confuse the
        # agent with context about a different PDF.
        session["document_id"] = req.document_id
        session["messages"] = []

    session["messages"].append({"role": "user", "content": req.question})

    tool_fn = build_search_tool(collection)
    agent = build_agent(tool_fn)
    result = agent.invoke({"messages": session["messages"]})
    answer = result["messages"][-1].content

    session["messages"].append({"role": "assistant", "content": answer})
    return ChatResponse(answer=answer)
