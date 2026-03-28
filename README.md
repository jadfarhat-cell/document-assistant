# RAG Document Assistant

A local RAG powered document assistant using Ollama and ChromaDB.

## Features
- Multi-format document ingestion
- Local LLM inference
- Vector search
- REST API

## Installation

```bash
pip install -r requirements.txt
ollama pull llama3.2
```

## Usage

```bash
python cli.py ingest document.pdf
python cli.py query "What is the main topic?"
```

## License

MIT
