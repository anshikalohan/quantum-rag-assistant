"""
CLI script to embed chunks and build the FAISS vector index.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --chunks data/processed/chunks.json
    python scripts/build_index.py --rebuild
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.embed_documents import EmbeddingEngine
from src.embeddings.vector_store import FAISSVectorStore
from src.ingestion.chunker import Chunk
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS vector index for the Quantum RAG Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_index.py
  python scripts/build_index.py --rebuild
  python scripts/build_index.py --chunks data/processed/chunks.json
        """,
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default=str(settings.chunks_file),
        help=f"Path to chunks JSON file (default: {settings.chunks_file})",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=str(settings.faiss_index_path),
        help=f"Output path for FAISS index (default: {settings.faiss_index_path})",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild even if index already exists",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size (default: 64)",
    )

    args = parser.parse_args()
    chunks_path = Path(args.chunks)
    index_path = Path(f"{args.index}.index")

    log.info("=" * 60)
    log.info("Quantum RAG Assistant — Building Vector Index")
    log.info("=" * 60)

    # Check if rebuild needed
    if index_path.exists() and not args.rebuild:
        log.info(f"Index already exists at {index_path}")
        log.info("Use --rebuild to force rebuild")
        sys.exit(0)

    # Step 1: Load chunks
    if not chunks_path.exists():
        log.error(f"Chunks file not found: {chunks_path}")
        log.info("Run first: python scripts/ingest.py")
        sys.exit(1)

    log.info(f"Loading chunks from {chunks_path}...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    chunks = [Chunk.from_dict(c) for c in chunks_data]
    texts = [c.text for c in chunks]

    log.info(f"Loaded {len(chunks)} chunks")

    # Step 2: Generate embeddings
    engine = EmbeddingEngine()
    embeddings = engine.embed_texts(texts, batch_size=args.batch_size)

    # Step 3: Build and save FAISS index
    store = FAISSVectorStore(index_path=args.index)
    store._chunks = chunks  # Set chunks directly since we built embeddings
    store.build(chunks, embeddings)
    store.save()

    log.info("")
    log.info("✅ Index built successfully!")
    log.info(f"   Vectors indexed : {store.total_vectors}")
    log.info(f"   Embedding model : {settings.embedding_model}")
    log.info(f"   Index saved to  : {index_path}")
    log.info("")
    log.info("Next step: uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()