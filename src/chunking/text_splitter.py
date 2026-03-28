from dataclasses import dataclass
from typing import Optional
import re

from src.ingestion import Document


@dataclass
class Chunk:
    content: str
    metadata: dict
    chunk_index: int
    doc_id: str


class TextSplitter:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[list[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_document(self, document: Document) -> list[Chunk]:
        text = document.content
        chunks = self._split_text(text)

        return [
            Chunk(
                content=chunk,
                metadata={**document.metadata, "chunk_index": i},
                chunk_index=i,
                doc_id=document.doc_id,
            )
            for i, chunk in enumerate(chunks)
        ]

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.split_document(doc))
        return all_chunks

    def _split_text(self, text: str) -> list[str]:
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        separator = separators[0] if separators else ""
        remaining_separators = separators[1:] if len(separators) > 1 else []

        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        chunks = []
        current_chunk = ""

        for split in splits:
            piece = split if not separator else split + separator

            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += piece
            else:
                if current_chunk:
                    if len(current_chunk) > self.chunk_size and remaining_separators:
                        sub_chunks = self._recursive_split(current_chunk, remaining_separators)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(current_chunk.strip())

                if len(piece) > self.chunk_size and remaining_separators:
                    sub_chunks = self._recursive_split(piece, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = piece

        if current_chunk.strip():
            if len(current_chunk) > self.chunk_size and remaining_separators:
                sub_chunks = self._recursive_split(current_chunk, remaining_separators)
                chunks.extend(sub_chunks)
            else:
                chunks.append(current_chunk.strip())

        return self._add_overlap(chunks)

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return chunks

        overlapped = []
        for i, chunk in enumerate(chunks):
            if i > 0 and self.chunk_overlap > 0:
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-self.chunk_overlap:]
                if overlap_text and not chunk.startswith(overlap_text):
                    chunk = overlap_text + " " + chunk

            overlapped.append(chunk)

        return overlapped


class CodeSplitter(TextSplitter):
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ):
        code_separators = [
            "\nclass ",
            "\ndef ",
            "\n\ndef ",
            "\n\n",
            "\n",
            " ",
            "",
        ]
        super().__init__(chunk_size, chunk_overlap, code_separators)
