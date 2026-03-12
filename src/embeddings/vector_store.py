"""
FAISS-based vector store for semantic retrieval.
Handles building, saving, loading, and querying the index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.ingestion.chunker import Chunk
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


class FAISSVectorStore:

    def __init__(self, index_path: str | Path = None):
        self.index_path = Path(index_path or settings.faiss_index_path)
        self.chunks_path = settings.chunks_file
        self._index = None
        self._chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        try:
            import faiss
        except ImportError:
            raise ImportError("Install FAISS: pip install faiss-cpu")

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        dim = embeddings.shape[1]
        log.info(f"Building FAISS index: {len(chunks)} vectors, dim={dim}")

        self._index = faiss.IndexFlatIP(dim)

        self._index = faiss.IndexIDMap(self._index)
        ids = np.arange(len(chunks), dtype=np.int64)
        self._index.add_with_ids(embeddings, ids)

        self._chunks = chunks
        log.info(f"✓ FAISS index built with {self._index.ntotal} vectors")

    def save(self) -> None:
        try:
            import faiss
        except ImportError:
            raise ImportError("Install FAISS: pip install faiss-cpu")

        if self._index is None:
            raise RuntimeError("No index to save. Call build() first.")

        # Ensure directories exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunks_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_file = Path(f"{self.index_path}.index")
        faiss.write_index(self._index, str(index_file))

        # Save chunk metadata
        chunks_data = [c.to_dict() for c in self._chunks]
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        log.info(f"✓ Saved index to {index_file}")
        log.info(f"✓ Saved {len(chunks_data)} chunks to {self.chunks_path}")

    def load(self) -> None:
        try:
            import faiss
        except ImportError:
            raise ImportError("Install FAISS: pip install faiss-cpu")

        index_file = Path(f"{self.index_path}.index")

        if not index_file.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_file}. "
                "Run: python scripts/build_index.py"
            )

        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at {self.chunks_path}")

        self._index = faiss.read_index(str(index_file))

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        self._chunks = [Chunk.from_dict(c) for c in chunks_data]

        log.info(
            f"✓ Loaded FAISS index: {self._index.ntotal} vectors, "
            f"{len(self._chunks)} chunks"
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
        threshold: float = None,
    ) -> List[Tuple[Chunk, float]]:
        if self._index is None:
            self.load()

        k = top_k or settings.top_k_results
        min_score = threshold or settings.similarity_threshold

        # FAISS expects 2D array
        query_vec = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  
                continue
            if score < min_score:
                continue
            results.append((self._chunks[idx], float(score)))

        return results

    @property
    def is_loaded(self) -> bool:
        return self._index is not None

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0