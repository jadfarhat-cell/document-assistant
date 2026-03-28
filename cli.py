#!/usr/bin/env python3
# Document Assistant CLI
import argparse
from src.rag_pipeline import RAGPipeline
from src.config import settings

def main():
    parser = argparse.ArgumentParser(description='RAG Document Assistant CLI')
    subparsers = parser.add_subparsers(dest='command')
    ingest_parser = subparsers.add_parser('ingest')
    ingest_parser.add_argument('path')
    query_parser = subparsers.add_parser('query')
    query_parser.add_argument('question')
    args = parser.parse_args()
    pipeline = RAGPipeline()
    if args.command == 'ingest':
        result = pipeline.ingest_file(args.path)
        print(f"Ingested: {result['filename']}")
    elif args.command == 'query':
        result = pipeline.query(args.question)
        print(result['answer'])

if __name__ == '__main__':
    main()
