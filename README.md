# RAG Document Assistant

A local retrieval-augmented generation application for asking questions about personal documents without sending document content to a hosted language-model provider.

The project supports document ingestion, semantic retrieval, local LLM inference, and three interfaces: a command-line tool, a FastAPI service, and a Gradio web UI.

## What it demonstrates

- End-to-end RAG pipeline design
- Multi-format document ingestion
- Text and source-code chunking
- Sentence-transformer embeddings
- Persistent vector search with ChromaDB
- Local generation through Ollama
- REST and streaming API design with FastAPI
- CLI and web interfaces built on the same application layer

## Current architecture

```text
Documents
   |
   v
DocumentLoader
   |
   v
TextSplitter / CodeSplitter
   |
   v
Sentence-transformer embeddings
   |
   v
ChromaDB vector store
   |
   v
Top-k retrieval and context assembly
   |
   v
Ollama local LLM
   |
   +--> CLI
   +--> FastAPI
   +--> Gradio UI
```

The `RAGPipeline` class coordinates ingestion, chunking, embedding, storage, retrieval, and answer generation.

## Supported inputs

- PDF
- DOCX
- Markdown
- Plain text
- Source-code files supported by the document loader

## Technology stack

- Python
- Ollama
- ChromaDB
- sentence-transformers
- FastAPI and Uvicorn
- Gradio
- PyPDF2
- python-docx
- Pydantic

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/jadfarhat-cell/document-assistant.git
cd document-assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS or Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and start Ollama

Pull the default model:

```bash
ollama pull llama3.2
```

Make sure the Ollama service is running before submitting queries.

## Usage

### Command-line interface

Ingest one document:

```bash
python cli.py ingest path/to/document.pdf
```

Ingest a directory:

```bash
python cli.py ingest path/to/documents --directory
```

Ask a question:

```bash
python cli.py query "What are the main conclusions?"
```

View system status:

```bash
python cli.py stats
```

Clear indexed documents:

```bash
python cli.py clear
```

### Gradio interface

```bash
python cli.py serve --ui
```

The default Gradio server runs on port `7861`.

### FastAPI service

```bash
python cli.py serve --api
```

Key endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Vector-store and Ollama status |
| `POST` | `/ingest` | Upload and index a document |
| `POST` | `/query` | Query indexed content |
| `DELETE` | `/documents` | Clear indexed documents |

The query endpoint supports both normal responses and server-sent event streaming.

## Configuration

Runtime settings are defined in `src/config.py`, including:

- Ollama model
- ChromaDB collection and persistence directory
- Chunk size and overlap
- Default retrieval count
- API host and port

## Current limitations

This repository is a portfolio project and is not yet a production document platform.

- No authentication or user isolation
- No automated retrieval or answer-quality evaluation suite
- No reranking stage
- No citation-level answer verification
- No automated test suite or CI workflow yet
- Intended primarily for local, single-user use

These limitations are documented intentionally so the public project reflects its current implementation.

## Planned improvements

- Add unit and integration tests
- Add retrieval evaluation using recall-at-k and a small labeled question set
- Add response-groundedness checks
- Add Docker and Docker Compose support
- Add structured logging and latency measurements
- Add optional reranking
- Improve source citations in generated answers

## License

This project is available for learning and portfolio review. Add a formal license before reusing or redistributing the code.
