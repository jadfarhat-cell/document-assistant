# RAG Document Assistant

A local RAG (Retrieval-Augmented Generation) powered document assistant that lets you chat with your documents using a local LLM.

## Features

- Multi-format document ingestion: PDF, DOCX, Markdown, code files, and plain text
- Local LLM: Uses Ollama for private, offline inference
- Vector search: ChromaDB with sentence-transformers embeddings
- Multiple interfaces: Gradio web UI, FastAPI REST API, and CLI

## Requirements

- Python 3.13+
- Ollama with a model installed (default: llama3.2)

## Installation

```bash
git clone https://github.com/jadfarhat-cell/document-assistant.git
cd document-assistant
python -m venv .venv
pip install -r requirements.txt
ollama pull llama3.2
```

## Usage

```bash
python cli.py serve --ui
python cli.py serve --api
python cli.py ingest document.pdf
python cli.py query "What is the main topic?"
```
