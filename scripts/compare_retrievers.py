"""Compare dense, BM25, and hybrid retrieval side by side for one query.

Usage: python scripts/compare_retrievers.py "your question here" [top_k]

No LLM calls -- this only exercises retrieval, so it's free and fast. Useful
for seeing exactly how the three strategies disagree on the same query
before spending anything on generation.
"""

import json
import sys
from pathlib import Path

import numpy as np

from rag_lab.embeddings.local import MiniLMEmbedder
from rag_lab.models import Chunk, RetrievedChunk
from rag_lab.retrieval.bm25 import BM25Retriever
from rag_lab.retrieval.dense_numpy import DenseNumpyRetriever
from rag_lab.retrieval.hybrid import HybridRetriever

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_FOLDER = REPO_ROOT / "data" / "index"


def load_index() -> tuple[list[Chunk], np.ndarray]:
    chunks_path = INDEX_FOLDER / "chunks.json"
    embeddings_path = INDEX_FOLDER / "embeddings.npy"
    if not chunks_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            f"No index found in {INDEX_FOLDER}. Run `python scripts/ingest.py` first."
        )
    chunks = [Chunk(**raw) for raw in json.loads(chunks_path.read_text())]
    embeddings = np.load(embeddings_path)
    return chunks, embeddings


def print_results(label: str, results: list[RetrievedChunk]) -> None:
    print("=" * 70)
    print(label)
    print("=" * 70)
    for i, rc in enumerate(results, start=1):
        source = Path(rc.chunk.source).name
        print(f"[{i}] score={rc.score:.4f}  source={source}")
        print(f"    {rc.chunk.text[:100]}...")
    print()


def main(query: str, top_k: int) -> None:
    chunks, embeddings = load_index()
    embedder = MiniLMEmbedder()

    dense = DenseNumpyRetriever(embedder)
    dense.index(chunks, embeddings=embeddings)

    bm25 = BM25Retriever()
    bm25.index(chunks)

    hybrid = HybridRetriever(dense=DenseNumpyRetriever(embedder), bm25=BM25Retriever())
    hybrid.index(chunks)  # indexes its own internal dense + bm25 copies

    print(f"\nQuery: {query!r}  (top_k={top_k})\n")
    print_results("DENSE (cosine similarity on embeddings)", dense.retrieve(query, top_k))
    print_results("BM25 (keyword overlap)", bm25.retrieve(query, top_k))
    print_results("HYBRID (RRF fusion of dense + BM25)", hybrid.retrieve(query, top_k))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python scripts/compare_retrievers.py "your question here" [top_k]')
        sys.exit(1)
    question = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    main(question, k)
