# PDF question-answering agent (LangChain)

A minimal, runnable example of an agent that loads a PDF and answers
questions about its contents, using LangChain.

## Why "agent" and not just "RAG chain"?

A plain retrieval-augmented chain always searches the vector store once,
whether or not the question needs it. Here the retriever is exposed to the
LLM as a **tool**, and a tool-calling agent decides for itself whether to
search, how to phrase the search query, and whether to search again before
answering. This is a small change in code but a real change in behavior,
and it's the pattern you'd extend later by adding more tools (web search,
a calculator, a second document, an API call).

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...     # any OpenAI-compatible key
python agent.py path/to/document.pdf
```

You'll see an indexing step (loading + splitting + embedding the PDF),
then an interactive prompt. Type `exit` to quit.

## How it works, step by step

| Step | What happens | Key LangChain pieces |
|---|---|---|
| 1. Load | PDF is parsed into one `Document` per page | `PyPDFLoader` |
| 2. Split | Pages are broken into ~1000-character overlapping chunks | `RecursiveCharacterTextSplitter` |
| 3. Embed & store | Each chunk becomes a vector, saved to a local vector DB | `OpenAIEmbeddings`, `Chroma` |
| 4. Retriever tool | The vector store's search is wrapped as a named, described tool | `create_retriever_tool` |
| 5. Agent | LLM + tool + prompt, run in a call-observe-repeat loop | `create_tool_calling_agent`, `AgentExecutor` |

### Why these specific choices

- **Chunk size (1000) / overlap (150)** — big enough to preserve context
  within a chunk, small enough that a single chunk is a focused, relevant
  unit to retrieve. Tune this per document: dense technical PDFs often
  want smaller chunks (500-800); narrative text can go larger.
- **k=4 retrieved chunks** — the number of chunks handed to the LLM per
  search. Too few and the answer misses relevant context; too many and
  you pay for and dilute the prompt with irrelevant text.
- **Tool description matters as much as the code** — the agent's LLM
  decides *when* to call `search_pdf` purely by reading its description.
  Vague descriptions ("searches stuff") produce agents that either never
  search or search unnecessarily.
- **`temperature=0`** on the LLM — for factual Q&A over a document you
  want deterministic, grounded answers, not creative variation.
- **Chroma with `persist_directory`** — saves the embedded index to disk
  so re-running the script on the same PDF skips re-embedding.

## Extending this

- **Conversation memory**: add a `chat_history` placeholder to the prompt
  and pass prior turns into `agent_executor.invoke(...)` so follow-up
  questions ("what about the second point?") resolve correctly.
- **Source citations**: the retrieved chunks carry `metadata["page"]`;
  have the agent or a wrapper surface which page(s) it used.
- **Multiple PDFs**: index each into the same or separate collections and
  either merge retrievers or give the agent one tool per document.
- **Swap the LLM/embeddings provider**: LangChain's interfaces are
  provider-agnostic — swap `ChatOpenAI`/`OpenAIEmbeddings` for Anthropic,
  local models via Ollama, etc., with no change to the rest of the code.
