from pathlib import Path
from typing import Optional, Generator

from src.ingestion import DocumentLoader
from src.chunking import TextSplitter
from src.chunking.text_splitter import CodeSplitter
from src.embeddings import Embedder
from src.vectorstore import ChromaStore
from src.retrieval import Retriever
from src.llm import OllamaClient


class RAGPipeline:
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db",
        model: str = "llama3.2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
    ):
        self.loader = DocumentLoader()
        self.text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.code_splitter = CodeSplitter(chunk_size=1000, chunk_overlap=100)
        self.embedder = Embedder()
        self.vector_store = ChromaStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedder=self.embedder,
        )
        self.retriever = Retriever(vector_store=self.vector_store, top_k=top_k)
        self.llm = OllamaClient(model=model)

    def ingest_file(self, file_path: str) -> dict:
        doc = self.loader.load(file_path)
        if doc.metadata.get("type") == "code":
            chunks = self.code_splitter.split_document(doc)
        else:
            chunks = self.text_splitter.split_document(doc)
        self.vector_store.add_chunks(chunks)
        return {
            "filename": doc.metadata["filename"],
            "doc_id": doc.doc_id,
            "chunks": len(chunks),
            "type": doc.metadata.get("type", "unknown"),
        }

    def ingest_directory(self, dir_path: str, recursive: bool = True) -> list[dict]:
        documents = self.loader.load_directory(dir_path, recursive=recursive)
        results = []
        for doc in documents:
            if doc.metadata.get("type") == "code":
                chunks = self.code_splitter.split_document(doc)
            else:
                chunks = self.text_splitter.split_document(doc)
            self.vector_store.add_chunks(chunks)
            results.append({
                "filename": doc.metadata["filename"],
                "doc_id": doc.doc_id,
                "chunks": len(chunks),
                "type": doc.metadata.get("type", "unknown"),
            })
        return results

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        stream: bool = False,
    ) -> dict | Generator[dict, None, None]:
        results = self.retriever.retrieve(question, top_k=top_k)
        context = self.retriever.retrieve_with_context(question, top_k=top_k)
        sources = self.retriever.get_sources(results)
        if stream:
            return self._stream_query(question, context, sources, results)
        else:
            answer = self.llm.generate(question, context, stream=False)
            return {"answer": answer, "sources": sources, "context_chunks": results}

    def _stream_query(
        self,
        question: str,
        context: str,
        sources: list[str],
        results: list[dict],
    ) -> Generator[dict, None, None]:
        full_answer = ""
        for token in self.llm.generate(question, context, stream=True):
            full_answer += token
            yield {"token": token, "partial_answer": full_answer, "sources": sources, "done": False}
        yield {"token": "", "partial_answer": full_answer, "answer": full_answer, "sources": sources, "context_chunks": results, "done": True}

    def get_stats(self) -> dict:
        store_stats = self.vector_store.get_stats()
        ollama_connected = self.llm.check_connection()
        available_models = self.llm.list_models() if ollama_connected else []
        return {
            "vector_store": store_stats,
            "ollama_connected": ollama_connected,
            "available_models": available_models,
            "current_model": self.llm.model,
        }

    def clear(self) -> None:
        self.vector_store.clear()
