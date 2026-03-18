"""
Semantic retriever — the bridge between queries and the vector store.
Handles query processing, retrieval, and result formatting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.embeddings.embed_documents import EmbeddingEngine
from src.embeddings.vector_store import FAISSVectorStore
from src.ingestion.chunker import Chunk
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RetrievalResult:

    chunk: Chunk
    score: float

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def source(self) -> str:
        page = f", page {self.chunk.page}" if self.chunk.page else ""
        return f"{self.chunk.source}{page}"

    def to_dict(self) -> dict:
        return {
            "text": self.chunk.text,
            "source": self.source,
            "score": round(self.score, 4),
            "chunk_id": self.chunk.chunk_id,
        }


class Retriever:

    def __init__(
        self,
        vector_store: FAISSVectorStore = None,
        embedding_engine: EmbeddingEngine = None,
    ):
        self.vector_store = vector_store or FAISSVectorStore()
        self.embedding_engine = embedding_engine or EmbeddingEngine()

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        threshold: float = None,
    ) -> List[RetrievalResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if not self.vector_store.is_loaded:
            self.vector_store.load()

        log.info(f"Retrieving for: '{query[:80]}...' " if len(query) > 80 else f"Retrieving for: '{query}'")
        query_embedding = self.embedding_engine.embed_query(query)

        raw_results = self.vector_store.search(
            query_embedding,
            top_k=top_k or settings.top_k_results,
            threshold=threshold or settings.similarity_threshold,
        )

        results = [RetrievalResult(chunk=chunk, score=score) for chunk, score in raw_results]

        seen_ids = set()
        deduped = []
        for r in results:
            if r.chunk.chunk_id not in seen_ids:
                seen_ids.add(r.chunk.chunk_id)
                deduped.append(r)

        log.info(f"Retrieved {len(deduped)} relevant chunks (top score: {deduped[0].score:.3f})" if deduped else "No relevant chunks found")

        return deduped

    def format_context(self, results: List[RetrievalResult]) -> str:
        if not results:
            return "No relevant context found."

        parts = []
        for i, result in enumerate(results, start=1):
            parts.append(
                f"[Context {i} | Source: {result.source} | Relevance: {result.score:.2f}]\n"
                f"{result.text}"
            )

        return "\n\n---\n\n".join(parts)