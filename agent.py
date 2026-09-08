"""
PDF Question-Answering Agent
=============================

A minimal but complete example of a LangChain *agent* (not just a chain)
that can answer questions about the contents of a PDF file.

Architecture (see the diagram in the conversation for the visual version):

  PDF file
     -> Load & split      (PyPDFLoader + RecursiveCharacterTextSplitter)
     -> Embed & store     (OpenAIEmbeddings + Chroma vector store)
     -> Retriever tool     (create_retriever_tool)
     -> Agent (LLM)        (create_tool_calling_agent + AgentExecutor)
     -> Answer

Why an *agent* instead of a simple RetrievalQA chain?
A plain retrieval chain ALWAYS searches the vector store once per question,
even if the question doesn't need it ("hi", "thanks", "what's 2+2?").
An agent is given the retriever as a *tool* and an LLM that decides, on its
own, whether to call that tool, how many times, and with what search query.
This is closer to how a real assistant should behave, and it generalizes:
you can add more tools later (web search, a calculator, another database)
without changing the control flow.

Usage:
    export OPENAI_API_KEY=sk-...
    python agent.py path/to/document.pdf
"""

import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ---------------------------------------------------------------------------
# Step 1-2: Load the PDF and split it into retrievable chunks
# ---------------------------------------------------------------------------
def load_and_split(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    """
    PyPDFLoader turns the PDF into one LangChain Document per page, each
    carrying `page_content` (the text) and `metadata` (e.g. page number).

    We then re-split those pages into smaller overlapping chunks:
      - chunk_size: chunks that are too large produce noisy embeddings and
        blow past what's useful in a prompt; too small loses context.
        ~1000 characters (roughly 150-200 words) is a solid default for
        prose-heavy PDFs.
      - chunk_overlap: a little overlap (150 chars) prevents a sentence
        that straddles a chunk boundary from losing meaning in both halves.
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
# Step 3: Embed the chunks and store them in a local vector database
# ---------------------------------------------------------------------------
def build_vectorstore(chunks, persist_directory: str = "./chroma_db"):
    """
    Each chunk of text is converted into a vector (a list of numbers that
    captures its meaning) by the embeddings model, then stored in Chroma,
    a lightweight local vector database. Later, a question is embedded the
    same way, and Chroma finds the chunks whose vectors are closest to it
    (semantic similarity), rather than requiring exact keyword matches.

    persist_directory lets Chroma save to disk, so you don't have to
    re-embed the PDF every time you run the script.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    return vectorstore


# ---------------------------------------------------------------------------
# Step 4: Wrap the vector store as a tool the agent can call
# ---------------------------------------------------------------------------
def build_retriever_tool(vectorstore, k: int = 4):
    """
    create_retriever_tool packages a retriever as a LangChain Tool with a
    name and a natural-language description. That description is what the
    agent's LLM reads to decide *when* to use this tool -- so it needs to
    clearly state what the tool is for.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    tool = create_retriever_tool(
        retriever,
        name="search_pdf",
        description=(
            "Search the uploaded PDF document for relevant passages. "
            "Use this whenever the user asks a question that might be "
            "answered by the document's contents. Input should be a "
            "focused search query, not the full user question verbatim."
        ),
    )
    return tool


# ---------------------------------------------------------------------------
# Step 5: Build the tool-calling agent
# ---------------------------------------------------------------------------
def build_agent(tool, model: str = "gpt-4o-mini"):
    """
    create_tool_calling_agent binds the LLM, the list of tools, and a
    prompt together. AgentExecutor then runs the actual loop:
      1. Send the question (+ system prompt + tool descriptions) to the LLM.
      2. If the LLM's response is a tool call, execute it and feed the
         result back to the LLM.
      3. Repeat until the LLM responds with a final answer instead of a
         tool call.
    """
    llm = ChatOpenAI(model=model, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant answering questions about a PDF "
         "document. Use the search_pdf tool to find relevant passages "
         "before answering. If the answer isn't in the document, say so "
         "clearly instead of guessing. Keep answers concise and, where "
         "useful, mention which part of the document supports your answer."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, [tool], prompt)
    executor = AgentExecutor(agent=agent, tools=[tool], verbose=True)
    return executor


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

    print(f"Indexing {pdf_path} ...")
    chunks = load_and_split(pdf_path)
    vectorstore = build_vectorstore(chunks)
    tool = build_retriever_tool(vectorstore)
    agent_executor = build_agent(tool)

    print("\nReady. Ask questions about the PDF (type 'exit' to quit).\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = agent_executor.invoke({"input": question})
        print(f"\nAgent: {result['output']}\n")


if __name__ == "__main__":
    main()
