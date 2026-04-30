# PDF Intelligence Assistant — A Production-Ready RAG Chatbot

> A self-contained, customizable **Retrieval-Augmented Generation (RAG)** chatbot that turns any PDF document into a conversational knowledge base. Ask questions in plain English and receive precise, citation-backed answers grounded strictly in your source material.

Built with **LangChain**, **OpenAI GPT**, **ChromaDB**, and **Gradio** — packaged as a single Jupyter notebook that runs anywhere: Google Colab, JupyterLab, VS Code, or local Python.

---

## Table of Contents

- [Why This Project](#why-this-project)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [How It Works](#how-it-works)
- [Customization Guide](#customization-guide)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Why This Project

Most "chat with your PDF" tutorials are toys — they break in production, hallucinate freely, and lose context after two turns. This project is different. It's a **reproducible reference implementation** that solves the real problems teams hit when shipping RAG systems:

- **Hallucination control** through strict prompt engineering and `temperature=0.0`
- **Cost control** through persistent vector storage (no re-embedding on every run)
- **Multi-turn coherence** through windowed conversational memory
- **Verifiability** through automatic page-level source citations
- **Python 3.12 / Gradio 4.x compatibility** with a documented runtime patch for the well-known `gradio_client` ASGI crash

Use this as a starting template for HR assistants, legal research tools, technical documentation bots, customer support agents, internal knowledge bases, or any use case where you need an LLM to answer questions strictly grounded in a private document.

---

## Key Features

| Feature | Description |
| --- | --- |
| **Strict Grounding** | The assistant answers only from your document. Out-of-scope questions trigger a graceful fallback message. |
| **Persistent Vector Store** | ChromaDB embeddings are written to disk once, then reloaded on subsequent runs — no redundant API costs. |
| **Conversational Memory** | A sliding window of the last 5 turns keeps follow-up questions coherent without exceeding token limits. |
| **Source Citations** | Every response is appended with the source page number(s) so users can verify against the original document. |
| **Token-Aware Chunking** | Uses `tiktoken` to chunk by tokens (not characters), giving more predictable retrieval behavior across LLMs. |
| **Polished Web UI** | A `gr.Blocks` Gradio interface with custom CSS, suggested prompts, and a clear-session button. |
| **One-Click Public Sharing** | `share=True` generates a 72-hour public URL via Gradio's tunnel — useful for stakeholder demos. |
| **Python 3.12 Compatibility** | Includes a runtime monkey-patch that neutralizes the Pydantic-v2 / Gradio-4.36 `bool schema` crash. |

---

## Architecture

The system follows the standard RAG two-phase pattern: an offline indexing phase that runs once, and an online query phase that runs on every user request.

```
                   ┌─────────────────────────────────────────────┐
                   │           INDEXING PHASE (Run Once)         │
                   │                                             │
                   │   PDF  →  Chunker  →  Embedder  →  ChromaDB │
                   │  (PyPDF)  (tiktoken) (OpenAI)   (persisted) │
                   └─────────────────────────────────────────────┘
                                                                  
                   ┌─────────────────────────────────────────────┐
                   │       QUERY PHASE (Every User Request)      │
                   │                                             │
                   │   Question → Embed → Vector Search → Top-K  │
                   │       ↓                                     │
                   │   System Prompt + Context + History + Q     │
                   │       ↓                                     │
                   │     OpenAI Chat Completion (gpt-3.5-turbo)  │
                   │       ↓                                     │
                   │   Answer + Page Citations → Gradio Chatbot  │
                   └─────────────────────────────────────────────┘
```

### Architecture Decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Splitter | `RecursiveCharacterTextSplitter` (token-encoded) | Splits on semantic boundaries (paragraph → sentence → word) and counts tokens, not chars |
| Chunk size | 500 tokens, 50-token overlap | Balances retrieval precision against context coherence at chunk boundaries |
| Retrieval depth | Top-K = 4 | Enough context for nuanced answers without flooding the prompt |
| Memory | Last 5 turns (windowed) | Prevents token overflow on long sessions while preserving recent context |
| Temperature | 0.0 | Eliminates creative drift; maximizes factual consistency |
| Vector DB | ChromaDB (local persist dir) | Zero-infra, embeddings written once and reloaded on subsequent runs |
| Citations | Page-level metadata extraction | Every answer is independently verifiable against the source PDF |

---

## Tech Stack

| Library | Version | Role |
| --- | --- | --- |
| `langchain` | 0.2.16 | RAG orchestration framework |
| `langchain-openai` | 0.1.23 | `ChatOpenAI` and `OpenAIEmbeddings` wrappers |
| `langchain-community` | 0.2.16 | `PyPDFLoader` document loader |
| `langchain-chroma` | 0.1.4 | LangChain-native ChromaDB adapter |
| `chromadb` | 0.5.20 | Persistent local vector database |
| `openai` | 1.51.0 | Official OpenAI Python SDK |
| `pypdf` | 4.3.1 | PDF parsing engine |
| `tiktoken` | 0.7.0 | OpenAI's tokenizer for accurate chunking |
| `gradio` | 4.36.1 | Browser-based chat UI |
| `huggingface_hub` | 0.23.4 | Pinned to avoid Gradio dependency conflicts |
| `httpx` | 0.27.2 | Pinned for OpenAI SDK compatibility |

> **Why pinned versions?** RAG stacks are notoriously fragile because LangChain, Gradio, and the OpenAI SDK each release frequently and break each other in subtle ways. The pins above are a known-good combination.

---

## Prerequisites

- **Python 3.10+** (3.12 supported via the included compatibility patch)
- An **OpenAI API key** with access to `gpt-3.5-turbo` and `text-embedding-ada-002` ([get one here](https://platform.openai.com/api-keys))
- A **PDF document** you want to chat with
- ~$0.05–$0.50 in OpenAI credits for the initial embedding pass on a typical 50–200 page document (one-time cost)

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Add your document

Place your PDF in the project root and update the `DOCUMENT_PATH` constant in **Cell 2** of the notebook:

```python
DOCUMENT_PATH = 'your_document.pdf'
```

### 3. Set your OpenAI API key

Either export it as an environment variable:

```bash
export OPENAI_API_KEY='sk-...'
```

Or let the notebook prompt you securely via `getpass()` on first run.

### 4. Run the notebook

Open `rag_chatbot.ipynb` in Jupyter, VS Code, or Google Colab and execute the cells **in order**:

1. **Cell 1** — Installs all dependencies. **Restart the runtime after this cell.**
2. **Cell 2** — Authenticates, loads your PDF, chunks it, and builds the vector database.
3. **Cell 3** — Applies the Python 3.12 compatibility patch and launches the Gradio chat UI.

The UI will be available at `http://localhost:7860` (or via a public `gradio.live` URL if `share=True`).

---

## Configuration Reference

All tuneable parameters live at the top of **Cell 2**. Modify them to match your use case:

```python
LLM_MODEL          = 'gpt-3.5-turbo'             # Reasoning model
EMBEDDING_MODEL    = 'text-embedding-ada-002'    # Vector embedding model
TOKEN_CHUNK_SIZE   = 500                         # Tokens per chunk
TOKEN_OVERLAP      = 50                          # Tokens of overlap between chunks
RETRIEVAL_K        = 4                           # Top-K chunks to retrieve
TEMPERATURE        = 0.0                         # 0.0 = factual, 1.0 = creative
MAX_HISTORY_TURNS  = 5                           # Conversational memory window
CHROMA_PERSIST_DIR = './chroma_store'            # Where vector DB is saved
DOCUMENT_PATH      = 'your_document.pdf'         # Path to your source PDF
```

### Parameter Tuning Guide

| Parameter | When to Increase | When to Decrease |
| --- | --- | --- |
| `TOKEN_CHUNK_SIZE` | Document has long, self-contained sections (e.g., legal contracts) | Document is dense with discrete facts (e.g., FAQs) |
| `TOKEN_OVERLAP` | Concepts often span chunk boundaries | Chunks are already self-contained; you want to save tokens |
| `RETRIEVAL_K` | Questions need synthesis across multiple sections | You want tighter, more focused answers |
| `TEMPERATURE` | You want more conversational, varied responses | You need maximum factual precision (default) |
| `MAX_HISTORY_TURNS` | Users have long, multi-turn investigations | You're hitting token limits |

---

## How It Works

### Indexing Phase (Cell 2)

1. **Load** — `PyPDFLoader` reads each page of the PDF and emits a list of `Document` objects, one per page, with page-number metadata attached.
2. **Chunk** — `RecursiveCharacterTextSplitter.from_tiktoken_encoder()` splits each page into ~500-token chunks, preferring semantic boundaries (paragraphs, then sentences, then words). Tiktoken ensures token counts match what the LLM will actually see.
3. **Embed** — Each chunk is sent to OpenAI's `text-embedding-ada-002` and converted to a 1,536-dimensional vector.
4. **Persist** — All vectors are written to a local ChromaDB directory. Subsequent runs can load directly from disk — no re-embedding needed unless the document changes.

### Query Phase (Cell 3)

1. **Embed the question** — The user's query is converted to the same 1,536-dim embedding space.
2. **Retrieve** — ChromaDB performs cosine-similarity search and returns the top 4 most relevant chunks.
3. **Assemble the prompt** — A structured message list is built: `[system prompt with context]` + `[last 5 turns of history]` + `[current user question]`.
4. **Generate** — The prompt is sent to `gpt-3.5-turbo` with `temperature=0.0` and `max_tokens=1024`.
5. **Cite & return** — Page numbers from the retrieved chunks are appended to the response and shown in the Gradio chat window.

### The System Prompt (the heart of the system)

```python
SYSTEM_PROMPT = (
    "You are the [Your Persona] Assistant.\n"
    "Answer EXCLUSIVELY from the policy context provided below.\n"
    "If the answer is absent, say: This parameter is not defined in the current "
    "document. Please consult [your fallback contact].\n"
    "Be concise, professional, and cite page number(s) at the end.\n\n"
    "Policy Context:\n{context}"
)
```

This is the single most important piece of code in the project. The phrase **"EXCLUSIVELY from the policy context"** is what prevents hallucination. Every other design choice supports this constraint.

---

## Customization Guide

### Swap in your own document

Just change `DOCUMENT_PATH` and re-run Cell 2. The vector DB will rebuild automatically if the persist directory is empty (or you delete it).

### Change the assistant's persona

Edit `SYSTEM_PROMPT` in Cell 3. For example:

```python
# Legal research assistant
SYSTEM_PROMPT = (
    "You are a legal research assistant. Answer questions strictly from the case "
    "law context provided below. Quote relevant passages verbatim. If the answer "
    "is not in the context, respond: 'I cannot find this in the provided cases.'\n\n"
    "Case Law Context:\n{context}"
)
```

```python
# Customer support bot
SYSTEM_PROMPT = (
    "You are a friendly customer support agent. Use only the product documentation "
    "provided to answer questions. If you don't know, route the customer to "
    "support@example.com.\n\nProduct Docs:\n{context}"
)
```

### Switch to GPT-4 or another model

```python
LLM_MODEL = 'gpt-4-turbo'        # More capable, ~10x cost
LLM_MODEL = 'gpt-4o-mini'        # Fast and cheap, similar quality to 3.5
```

### Use a local / open-source LLM

Replace the `OpenAI()` client with any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, Together AI, etc.):

```python
openai_client = OpenAI(
    base_url='http://localhost:11434/v1',  # e.g., Ollama
    api_key='ollama',                       # placeholder
)
LLM_MODEL = 'llama3.1:8b'
```

### Use a different embedding model

```python
# HuggingFace local embeddings (no OpenAI cost)
from langchain_huggingface import HuggingFaceEmbeddings
embedder = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
```

### Customize the UI

The Gradio interface is built with `gr.Blocks` for full layout control. Edit the `CSS` block in Cell 3 to change colors, fonts, or branding. Add or remove components by modifying the layout inside `with gr.Blocks(...) as demo:`.

### Multiple documents

Pass a list of file paths and concatenate the loaded pages before chunking:

```python
documents = ['policy_a.pdf', 'policy_b.pdf', 'handbook.pdf']
all_pages = []
for doc in documents:
    all_pages.extend(PyPDFLoader(doc).load())
chunks = splitter.split_documents(all_pages)
```

---

## Deployment

### Local (Jupyter / VS Code)

Run all cells. The UI launches at `http://localhost:7860`. Set `share=False` in the final `demo.launch()` call to keep it local.

### Google Colab

1. Upload the notebook and your PDF to Colab.
2. Store your API key as a Colab Secret (left sidebar → 🔑 → `OPENAI_API_KEY`).
3. Set `share=True` in `demo.launch()` to get a public `*.gradio.live` URL valid for ~72 hours.

### Hugging Face Spaces

1. Convert the notebook to `app.py`.
2. Create a `requirements.txt` from the pinned versions in Cell 1.
3. Push to a new Hugging Face Space (Gradio SDK).
4. Add `OPENAI_API_KEY` as a repository secret.

### Production Considerations

This notebook is a reference implementation, not a hardened production service. Before shipping to real users, consider:

- **Authentication** — Add Gradio's `auth=` parameter, or front the app with an OAuth proxy.
- **Rate limiting** — Wrap `handle_query` with a token bucket to prevent runaway API costs.
- **Logging & observability** — Log every query, retrieved chunks, and response for quality review.
- **Vector DB scale** — ChromaDB local is fine up to ~100k chunks. Beyond that, consider Pinecone, Weaviate, or Qdrant.
- **Concurrent users** — Increase `max_threads` and `concurrency_limit` in `demo.launch()`, but watch your OpenAI rate limits.
- **Document refresh** — Build a scheduled job that rebuilds the vector store when source documents change.

---

## Troubleshooting

### `TypeError: argument of type 'bool' is not iterable` on Gradio launch

This is the **Python 3.12 + Pydantic v2 + Gradio 4.36.1** ASGI crash. The notebook handles it automatically with a two-layer fix in Cell 3:

1. **Monkey-patch** of `gradio_client.utils._json_schema_to_python_type` to gracefully handle non-dict schema nodes (Pydantic v2 emits literal `True`/`False` as schema values, which Gradio doesn't guard against).
2. **`show_api=False`** in `demo.launch()` to disable the `/info` route entirely so the patched code path is never triggered.

If you still see the error, ensure Cell 3 is run *after* Cell 1 and a runtime restart.

### `FileNotFoundError: 'your_document.pdf' not found`

Update `DOCUMENT_PATH` in Cell 2 to match your actual filename. In Colab, upload via the Files panel (left sidebar). Confirm with `!ls` in a new cell.

### Embeddings cell takes forever / hits rate limits

OpenAI rate-limits new accounts heavily. For very large PDFs (500+ pages), either:
- Upgrade to a higher OpenAI usage tier, or
- Switch to a local embedding model (see [Customization Guide](#use-a-different-embedding-model)).

### "I don't know" responses for questions clearly in the document

Try one or more of:
- Increase `RETRIEVAL_K` from 4 to 6 or 8
- Decrease `TOKEN_CHUNK_SIZE` to 300 (smaller, more focused chunks)
- Increase `TOKEN_OVERLAP` to 100 (more redundancy at boundaries)
- Inspect the retrieved chunks with `vector_db.similarity_search(query, k=4)` to verify the right content is being found

### ChromaDB persistence not working / re-embedding every run

Cell 2 deletes and rebuilds `CHROMA_PERSIST_DIR` on every run by design (`shutil.rmtree`). If you want true caching, replace that block with:

```python
if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
    vector_db = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
else:
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_PERSIST_DIR,
    )
```

---

## Project Structure

```
.
├── README.md                    # You are here
├── rag_chatbot.ipynb            # The main notebook (3 cells)
├── requirements.txt             # Pinned dependency versions
├── .env.example                 # Template for OPENAI_API_KEY
├── .gitignore                   # Excludes .env, chroma_store/, __pycache__/
├── docs/
│   └── architecture.md          # Optional: deeper architecture notes
└── chroma_store/                # Generated on first run (gitignored)
```

---

## Roadmap

Possible enhancements for contributors:

- [ ] Multi-document support with per-document filters
- [ ] Streaming responses (token-by-token) via OpenAI's streaming API
- [ ] Hybrid search (BM25 + vector) for keyword-heavy queries
- [ ] Reranking with `cohere-rerank` or a cross-encoder
- [ ] Optional citation highlighting in the source PDF
- [ ] Dockerfile and Docker Compose for one-command deployment
- [ ] Evaluation harness with `ragas` or `trulens`
- [ ] Support for `.docx`, `.pptx`, `.html`, and Markdown sources

---

## Contributing

Contributions are welcome. Open an issue to discuss substantial changes before submitting a pull request. For small fixes (typos, minor bugs), feel free to PR directly.

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Commit your changes with clear messages
4. Push and open a PR

---

## License

Released under the MIT License. See `LICENSE` for details. You are free to use, modify, and distribute this code for any purpose, including commercial use, provided the copyright notice is retained.

---

## Acknowledgments

- **OpenAI** for the GPT and embedding models
- **LangChain** for the RAG orchestration primitives
- **Chroma** for a delightfully simple vector database
- **Gradio** for making ML demos shareable in five lines of Python
- The broader open-source community whose tutorials, blog posts, and GitHub issues made every design decision in this project less painful

---

> Built as a reusable reference for anyone shipping a document-grounded conversational AI. If this helped you, a ⭐ on the repository is the best thank-you.
