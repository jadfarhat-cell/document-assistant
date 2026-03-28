#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="RAG Document Assistant CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("path", help="File or directory path")
    ingest_parser.add_argument("-d", "--directory", action="store_true", help="Treat path as directory")
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of documents to retrieve")
    subparsers.add_parser("stats", help="Show system statistics")
    serve_parser = subparsers.add_parser("serve", help="Start a server")
    serve_group = serve_parser.add_mutually_exclusive_group(required=True)
    serve_group.add_argument("--api", action="store_true", help="Start FastAPI server")
    serve_group.add_argument("--ui", action="store_true", help="Start Gradio UI")
    subparsers.add_parser("clear", help="Clear all indexed documents")
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    from src.rag_pipeline import RAGPipeline
    from src.config import settings
    pipeline = RAGPipeline(
        collection_name=settings.chroma_collection,
        persist_directory=settings.chroma_persist_dir,
        model=settings.ollama_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
    )
    if args.command == "ingest":
        path = Path(args.path)
        if args.directory:
            results = pipeline.ingest_directory(str(path))
            print(f"Ingested {len(results)} files")
        else:
            result = pipeline.ingest_file(str(path))
            print(f"Ingested: {result['filename']}")
    elif args.command == "query":
        result = pipeline.query(args.question, top_k=args.top_k)
        print(result["answer"])
    elif args.command == "stats":
        stats = pipeline.get_stats()
        print(f"Documents indexed: {stats['vector_store']['count']} chunks")
    elif args.command == "serve":
        if args.api:
            import uvicorn
            uvicorn.run("src.api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
        elif args.ui:
            from ui.app import demo, theme, custom_css
            demo.launch(server_name="0.0.0.0", server_port=7861, theme=theme, css=custom_css)
    elif args.command == "clear":
        pipeline.clear()
        print("Cleared all indexed documents.")

if __name__ == "__main__":
    main()
