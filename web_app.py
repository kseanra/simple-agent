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
     -> Rerank             (cross-encoder over a wider candidate set)
     -> Retriever tool      (a small @tool-wrapped function, per request)
     -> Agent (LLM)         (create_agent, LangChain v1's agent API)
     -> Answer + citations

What's different from agent.py, and why:

  - Indexing happens on file upload, not a CLI argument. Each PDF's
    identity is a hash of its *content* (not its filename), so uploading
    the same file twice -- even renamed, even from a different browser --
    reuses the existing collection instead of re-embedding it.

  - Chroma/embeddings/LLM connections are created once at server startup
    (a FastAPI "lifespan" hook) and reused for every request, rather than
    once per script run -- the same objects, just with a longer lifetime
    because the process itself is long-running.

  - Conversation state (the running list of messages) lives in Redis,
    keyed by a session id the browser generates and keeps. Unlike the
    earlier in-memory dict, this survives a uvicorn restart and is shared
    across multiple server workers, so the app can run behind a load
    balancer. See SessionStore below.

  - Retrieval quality: instead of blindly trusting Chroma's top-k vector
    hits, we pull a wider candidate set and re-rank it with a cross-encoder
    (sentence-transformers), which scores each (question, passage) pair
    jointly rather than by embedding distance alone. The final passages
    also carry their page numbers back to the browser as citations.

Prerequisites:
    docker compose up -d          (starts Chroma AND Redis, see docker-compose.yml)
    export OPENAI_API_KEY=sk-...
    pip install -r requirements.txt
    uvicorn web_app:app --reload --port 8001
    open http://localhost:8001
"""

import hashlib
import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import chromadb
import redis
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

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6380"))
# Sessions expire after this many seconds of inactivity, so the store
# doesn't grow without bound. Refreshed on every read/write.
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", str(7 * 24 * 3600)))

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"

# Chunking is configurable per document type via query params on upload
# (see /api/documents). These are the defaults, tuned for prose PDFs; a
# dense table-heavy or code-heavy PDF often wants smaller chunks.
DEFAULT_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))

# Retrieval + rerank knobs. We fetch CANDIDATE_K passages by vector
# similarity, then the cross-encoder re-scores them and we keep the top
# FINAL_K. Fetching more candidates than we keep is what gives the
# reranker something to improve on.
CANDIDATE_K = int(os.environ.get("CANDIDATE_K", "20"))
FINAL_K = int(os.environ.get("FINAL_K", "4"))

# Cross-encoder used for reranking. Small, CPU-friendly, downloaded once
# and cached under ~/.cache/huggingface on first use.
RERANK_MODEL = os.environ.get(
    "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a PDF document. "
    "Use the search_pdf tool to find relevant passages before answering. "
    "Each passage is prefixed with its page number like '[page 3]'. When "
    "you use information from a passage, cite the page in your answer, e.g. "
    "'(page 3)'. If the answer isn't in the document, say so clearly "
    "instead of guessing. Keep answers concise."
)


# ---------------------------------------------------------------------------
# Redis-backed conversation store. Replaces the old in-memory dict so
# history survives restarts and is shared across workers. Each session is
# one Redis key holding a JSON blob: {"document_id": str, "messages": [...]}.
# ---------------------------------------------------------------------------
class SessionStore:
    def __init__(self, client: "redis.Redis", ttl: int):
        self._r = client
        self._ttl = ttl

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}"

    def load(self, session_id: str) -> Dict:
        raw = self._r.get(self._key(session_id))
        if raw is None:
            return {"document_id": None, "messages": []}
        return json.loads(raw)

    def save(self, session_id: str, session: Dict) -> None:
        # setex refreshes the TTL on every write, so active conversations
        # stay alive and only idle ones expire.
        self._r.setex(self._key(session_id), self._ttl, json.dumps(session))


# ---------------------------------------------------------------------------
# Shared, process-lifetime resources: one Chroma connection, one embeddings
# client, one LLM, one Redis-backed session store, and one lazily-loaded
# reranker. Created in the "lifespan" hook below (FastAPI's
# startup/shutdown mechanism) rather than at import time, so the app object
# can be imported/tested without immediately requiring live services.
# ---------------------------------------------------------------------------
class AppState:
    chroma_client: Optional[chromadb.HttpClient] = None
    embeddings: Optional[OpenAIEmbeddings] = None
    llm: Optional[ChatOpenAI] = None
    sessions: Optional[SessionStore] = None
    _reranker = None
    _reranker_lock = Lock()

    def get_reranker(self):
        """Load the cross-encoder on first use (it pulls in torch and
        downloads model weights, so we don't pay that cost at startup or
        during import). Guarded by a lock so two concurrent requests don't
        both trigger the load."""
        if self._reranker is None:
            with self._reranker_lock:
                if self._reranker is None:
                    from sentence_transformers import CrossEncoder

                    self._reranker = CrossEncoder(RERANK_MODEL)
        return self._reranker


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


def connect_to_redis() -> "redis.Redis":
    """Same fail-fast treatment for Redis as for Chroma."""
    try:
        client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
        )
        client.ping()
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach Redis at {REDIS_HOST}:{REDIS_PORT}. "
            f"Start it with: docker compose up -d ({exc})"
        ) from exc
    return client


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(exist_ok=True)
    state.chroma_client = connect_to_chroma()
    state.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    state.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    state.sessions = SessionStore(connect_to_redis(), SESSION_TTL_SECONDS)
    yield
    state.chroma_client = None
    state.embeddings = None
    state.llm = None
    state.sessions = None


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


class Citation(BaseModel):
    page: Optional[int] = None  # 0-based in PDF metadata; see page_label
    page_label: str  # human-facing label, e.g. "3"
    snippet: str  # short preview of the cited passage


class ChatRequest(BaseModel):
    session_id: str
    document_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = []


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


def index_pdf(
    document_id: str, content: bytes, chunk_size: int, chunk_overlap: int
) -> List:
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
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
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
# Retrieval + rerank. The tool asks Chroma for a WIDE candidate set (by
# embedding similarity), then a cross-encoder re-scores every
# (question, passage) pair and we keep the best FINAL_K. Cross-encoders
# read the query and passage together, so they judge relevance far better
# than embedding distance alone -- at the cost of running the model once
# per candidate, which is why we only rerank a couple dozen, not the whole
# collection.
#
# The tool records which passages it ultimately used (with their page
# numbers) on a small per-request collector, so the chat route can return
# them as citations after the agent finishes.
# ---------------------------------------------------------------------------
def _page_label(metadata: dict) -> tuple[Optional[int], str]:
    """PyPDFLoader stores a 0-based 'page' and often a 'page_label' (the
    label printed on the page, which may differ from its index). Prefer the
    label for humans, fall back to page index + 1."""
    page = metadata.get("page")
    label = metadata.get("page_label")
    if label:
        return page, str(label)
    if isinstance(page, int):
        return page, str(page + 1)
    return None, "?"


def retrieve_and_rerank(collection, query: str) -> List[dict]:
    """Returns up to FINAL_K passages, each as
    {"text": str, "page": Optional[int], "page_label": str}, best first."""
    query_vector = state.embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=CANDIDATE_K,
        include=["documents", "metadatas"],
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0] or [{}] * len(docs)
    if not docs:
        return []

    reranker = state.get_reranker()
    scores = reranker.predict([(query, doc) for doc in docs])
    ranked = sorted(zip(scores, docs, metas), key=lambda t: t[0], reverse=True)

    passages = []
    for _score, doc, meta in ranked[:FINAL_K]:
        page, label = _page_label(meta or {})
        passages.append({"text": doc, "page": page, "page_label": label})
    return passages


def build_search_tool(collection, citation_sink: List[dict]):
    @tool
    def search_pdf(query: str) -> str:
        """Search the uploaded PDF document for relevant passages. Use
        this whenever the user asks a question that might be answered by
        the document's contents. Input should be a focused search query,
        not the full user question verbatim."""
        passages = retrieve_and_rerank(collection, query)
        if not passages:
            return "No relevant passages found in the document."
        # Remember what we surfaced so the route can cite it. Dedup by
        # page label to avoid three citations that all say "page 3".
        seen = {p["page_label"] for p in citation_sink}
        for p in passages:
            if p["page_label"] not in seen:
                citation_sink.append(p)
                seen.add(p["page_label"])
        # Prefix each passage with its page so the LLM can cite inline.
        return "\n\n---\n\n".join(
            f"[page {p['page_label']}] {p['text']}" for p in passages
        )

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
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=400,
            detail="Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size.",
        )

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

    chunks = index_pdf(document_id, content, chunk_size, chunk_overlap)
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

    session = state.sessions.load(req.session_id)
    if session.get("document_id") != req.document_id:
        # Switching which document this session is chatting with starts a
        # fresh conversation -- carrying old messages over would confuse the
        # agent with context about a different PDF.
        session = {"document_id": req.document_id, "messages": []}

    session["messages"].append({"role": "user", "content": req.question})

    # Per-request collector the search tool fills with whatever passages it
    # surfaced; drained into the response as citations below.
    citation_sink: List[dict] = []
    tool_fn = build_search_tool(collection, citation_sink)
    agent = build_agent(tool_fn)
    result = agent.invoke({"messages": session["messages"]})
    answer = result["messages"][-1].content

    session["messages"].append({"role": "assistant", "content": answer})
    state.sessions.save(req.session_id, session)

    citations = [
        Citation(
            page=p["page"],
            page_label=p["page_label"],
            snippet=(p["text"][:200] + "...") if len(p["text"]) > 200 else p["text"],
        )
        for p in citation_sink
    ]
    return ChatResponse(answer=answer, citations=citations)
