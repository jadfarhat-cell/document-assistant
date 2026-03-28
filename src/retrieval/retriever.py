from typing import Optional

from src.vectorstore import ChromaStore


class Retriever:
    def __init__(
        self,
        vector_store: Optional[ChromaStore] = None,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ):
        self.vector_store = vector_store or ChromaStore()
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        k = top_k or self.top_k

        results = self.vector_store.search(
            query=query,
            n_results=k,
            where=filter_metadata,
        )

        if self.score_threshold is not None:
            results = [
                r for r in results
                if r["distance"] <= self.score_threshold
            ]

        for r in results:
            r["score"] = 1 - r["distance"]

        return results

    def retrieve_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> str:
        results = self.retrieve(query, top_k)

        if not results:
            return "No relevant documents found."

        context_parts = []
        for i, result in enumerate(results, 1):
            source = result["metadata"].get("filename", "Unknown")
            content = result["content"]
            score = result.get("score", 0)

            context_parts.append(
                f"[Source {i}: {source} (relevance: {score:.2f})]\n{content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def get_sources(self, results: list[dict]) -> list[str]:
        sources = set()
        for r in results:
            if "source" in r["metadata"]:
                sources.add(r["metadata"]["source"])
            elif "filename" in r["metadata"]:
                sources.add(r["metadata"]["filename"])
        return list(sources)
