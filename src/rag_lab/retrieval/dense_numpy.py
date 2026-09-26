"""Brute-force dense vector retrieval: cosine similarity against every indexed chunk.

This is deliberately the simplest possible vector search — no index structure,
just compare the query against every chunk and sort. It's O(n) per query: fine
for a few thousand chunks, unworkable for millions. That gap is exactly what an
ANN index (HNSW, used by Qdrant) exists to close — at the cost of being an
approximate rather than exact search. We build the exact version first so the
"approximate" trade-off means something concrete once Qdrant replaces this.
"""

import numpy as np

from rag_lab.embeddings.base import Embedder
from rag_lab.models import Chunk, RetrievedChunk
from rag_lab.retrieval.base import Retriever


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors.

    YOUR TURN: implement this.

    cosine_similarity(a, b) = dot(a, b) / (||a|| * ||b||)

    Compute the full formula — don't assume a and b are already unit-length,
    even though in this project they will be (both come from a normalizing
    embedder). Computing it defensively means this function stays correct if
    a future embedder doesn't normalize its output.

    Returns a float between -1 and 1 (1 = identical direction, 0 = orthogonal/
    unrelated, -1 = opposite direction).
    """
    return (a@b) / (np.linalg.norm(a) * np.linalg.norm(b))


class DenseNumpyRetriever(Retriever):
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None
        self._embedder: Embedder | None = None

    def index(self, chunks: list[Chunk], embedder: Embedder, embeddings: np.ndarray | None = None) -> None:
        self._chunks = chunks
        self._embedder = embedder
        if embeddings is not None:
            self._embeddings = embeddings
        else:
            self._embeddings = embedder.embed_documents([chunk.text for chunk in chunks])

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the top_k chunks most similar to `query`.

        YOUR TURN: implement this.

        Steps:
        1. Raise ValueError if index() hasn't been called yet (self._embeddings is None).
        2. Embed `query` with self._embedder.embed_query(query).
        3. Score every indexed chunk against the query using cosine_similarity
           (loop over self._embeddings and self._chunks together — no need to
           vectorize for this dataset size).
        4. Sort chunks by score, descending.
        5. Return the top_k as a list of RetrievedChunk(chunk=..., score=...).
        """
        if self._embeddings is None:
            raise ValueError("no embeddings")
        query = self._embedder.embed_query(query)
        result = []
        for i in range(0,len(self._embeddings)):
            angle = cosine_similarity(query,self._embeddings[i])
            result.append([angle,self._chunks[i]])

        result.sort(reverse=True)
        return [RetrievedChunk(chunk=res[1], score=res[0]) for res in result[:top_k]]
