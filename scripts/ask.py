"""Ask a question against the persisted index.

Usage: python scripts/ask.py "your question here" [top_k]

Loads data/index/chunks.json + embeddings.npy (built by scripts/ingest.py),
runs the full retrieve -> generate pipeline, and prints the answer plus
which chunks were retrieved and their similarity scores.
"""

import json
import sys
from pathlib import Path

import numpy as np

from rag_lab.embeddings.local import MiniLMEmbedder
from rag_lab.generation.anthropic_gen import AnthropicGenerator
from rag_lab.models import Chunk
from rag_lab.pipeline import RagPipeline
from rag_lab.retrieval.dense_numpy import DenseNumpyRetriever

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


def main(query: str, top_k: int) -> None:
    chunks, embeddings = load_index()

    embedder = MiniLMEmbedder()
    retriever = DenseNumpyRetriever()
    retriever.index(chunks, embedder, embeddings=embeddings)
    generator = AnthropicGenerator()
    pipeline = RagPipeline(embedder=embedder, retriever=retriever, generator=generator)

    result = pipeline.answer(query, top_k=top_k)

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result.answer)
    print()
    print("=" * 70)
    print(f"RETRIEVED CHUNKS (top {top_k}, before filtering to what was actually cited)")
    print("=" * 70)
    retrieved = retriever.retrieve(query, top_k=top_k)
    for i, rc in enumerate(retrieved, start=1):
        print(f"[{i}] score={rc.score:.4f} source={rc.chunk.source}")
        print(f"    {rc.chunk.text[:120]}...")
    print()
    print("=" * 70)
    print("CITATIONS ACTUALLY USED IN THE ANSWER")
    print("=" * 70)
    if not result.citations:
        print("(none)")
    for rc in result.citations:
        print(f"- score={rc.score:.4f} source={rc.chunk.source}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python scripts/ask.py "your question here" [top_k]')
        sys.exit(1)
    question = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    main(question, k)
