"""
PDF Question-Answering Agent -- Dockerized Chroma edition
===========================================================

Same agent as the earlier version, but the vector store is now a real,
persistent database: a Chroma server running in Docker, rather than an
in-process Python object. This is closer to how you'd run this in
production, and it means indexed PDFs survive between runs.

Architecture:

  PDF file
     -> Load & split      (PyPDFLoader + RecursiveCharacterTextSplitter)
     -> Embed             (OpenAIEmbeddings, called directly)
     -> Store & search     (Chroma server in Docker, via chromadb-client over HTTP)
     -> Retriever tool     (a small @tool-wrapped function, see below)
     -> Agent (LLM)        (create_agent, LangChain v1's agent API)
     -> Answer

Why not `langchain-chroma`?
The official LangChain <-> Chroma integration package (`langchain-chroma`)
declares the FULL `chromadb` package as a required dependency -- which
pulls in `onnxruntime` and other compiled binaries that don't always have
wheels for every OS/Python/architecture (this is what caused the earlier
install failures). Since the Chroma *server* now runs inside a Linux
container, we don't need any of that on the Mac side. Instead this script
talks to the server directly using `chromadb-client`, a lightweight
HTTP-only client with a minimal dependency footprint, and wraps the
search as a plain LangChain tool with the `@tool` decorator.

Prerequisites:
    docker compose up -d          (starts the Chroma server, see docker-compose.yml)
    export OPENAI_API_KEY=sk-...
    python agent.py path/to/document.pdf
"""

import os
import re
import sys
import uuid

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent


CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))


# ---------------------------------------------------------------------------
# Step 1-2: Load the PDF and split it into retrievable chunks
# ---------------------------------------------------------------------------
def load_and_split(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    """
    Same as before: PyPDFLoader turns the PDF into one Document per page,
    then RecursiveCharacterTextSplitter breaks those into smaller
    overlapping chunks sized for meaningful embeddings and precise
    retrieval.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(pages)
    print(f"Loaded {len(pages)} page(s), split into {len(chunks)} chunk(s).")
    return chunks


# ---------------------------------------------------------------------------
# Step 3: Connect to the Chroma server and index chunks (if not already done)
# ---------------------------------------------------------------------------
def collection_name_for(pdf_path: str) -> str:
    """Turn a filename into a safe, unique Chroma collection name."""
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    return "pdf_" + re.sub(r"[^a-zA-Z0-9_-]", "_", base).lower()


def connect_to_chroma():
    """
    Connects to the Chroma server over HTTP. This is the only network
    dependency this script has on the vector store -- if the Docker
    container isn't running, this fails fast with a clear message rather
    than a confusing stack trace later.
    """
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        client.heartbeat()
    except Exception as exc:
        print(
            f"Could not reach the Chroma server at {CHROMA_HOST}:{CHROMA_PORT}.\n"
            f"Start it with: docker compose up -d\n"
            f"Underlying error: {exc}"
        )
        sys.exit(1)
    return client


def get_or_build_collection(client, pdf_path: str, embeddings: OpenAIEmbeddings):
    """
    Each PDF gets its own Chroma collection (named after the file), so
    re-running on the same PDF reuses the already-indexed data -- real
    persistence, unlike the in-memory version. Re-running on a different
    PDF just creates a separate collection; nothing is overwritten.
    """
    name = collection_name_for(pdf_path)
    collection = client.get_or_create_collection(name=name)

    if collection.count() > 0:
        print(f"Found existing indexed data for '{name}' ({collection.count()} chunks) -- reusing it.")
        return collection

    print(f"No existing index for '{name}'. Indexing {pdf_path} ...")
    chunks = load_and_split(pdf_path)

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]
    vectors = embeddings.embed_documents(texts)  # one API call embedding all chunks

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
    print(f"Indexed {len(chunks)} chunk(s) into collection '{name}'.")
    return collection


# ---------------------------------------------------------------------------
# Step 4: Wrap Chroma search as a tool the agent can call
# ---------------------------------------------------------------------------
def build_search_tool(collection, embeddings: OpenAIEmbeddings, k: int = 4):
    """
    Because we're talking to Chroma directly (not through a LangChain
    VectorStoreRetriever), the tool is just a plain Python function
    decorated with @tool. The docstring becomes the tool's description --
    exactly what the agent's LLM reads to decide when to call it, so it
    needs to clearly state what the tool is for.
    """
    @tool
    def search_pdf(query: str) -> str:
        """Search the uploaded PDF document for relevant passages. Use
        this whenever the user asks a question that might be answered by
        the document's contents. Input should be a focused search query,
        not the full user question verbatim."""
        query_vector = embeddings.embed_query(query)
        results = collection.query(query_embeddings=[query_vector], n_results=k)
        docs = results.get("documents", [[]])[0]
        if not docs:
            return "No relevant passages found in the document."
        return "\n\n---\n\n".join(docs)

    return search_pdf


# ---------------------------------------------------------------------------
# Step 5: Build the agent
# ---------------------------------------------------------------------------
def build_agent(tool_fn, model: str = "gpt-4o-mini"):
    """
    Identical to the earlier version: create_agent binds an LLM and a
    list of tools into a runnable agent. The agent decides on its own
    whether to call search_pdf, how to phrase the query, and whether to
    call it again before answering.
    """
    llm = ChatOpenAI(model=model, temperature=0)

    agent = create_agent(
        model=llm,
        tools=[tool_fn],
        system_prompt=(
            "You are a helpful assistant answering questions about a PDF "
            "document. Use the search_pdf tool to find relevant passages "
            "before answering. If the answer isn't in the document, say so "
            "clearly instead of guessing. Keep answers concise and, where "
            "useful, mention which part of the document supports your answer."
        ),
    )
    return agent


# ---------------------------------------------------------------------------
# Put it all together: an interactive question-answering loop
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python agent.py path/to/document.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY before running (export OPENAI_API_KEY=sk-...).")
        sys.exit(1)

    client = connect_to_chroma()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    collection = get_or_build_collection(client, pdf_path, embeddings)

    tool_fn = build_search_tool(collection, embeddings)
    agent = build_agent(tool_fn)

    print("\nReady. Ask questions about the PDF (type 'exit' to quit).\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        answer = result["messages"][-1].content
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()
