"""Retriever interface: given a query, return the most relevant chunks.

retrieve() takes raw query text, not a precomputed embedding. This matters
for retrievers that don't use embeddings at all — a future BM25Retriever
scores chunks by keyword overlap and never touches an Embedder. Keeping the
outer signature the same means a HybridRetriever can later compose a dense
and a BM25 retriever behind this one interface, and a pipeline can swap
retrievers without caring which kind it's using.
"""

from abc import ABC, abstractmethod

import numpy as np

from rag_lab.embeddings.base import Embedder
from rag_lab.models import Chunk, RetrievedChunk


class Retriever(ABC):
    @abstractmethod
    def index(self, chunks: list[Chunk], embedder: Embedder, embeddings: np.ndarray | None = None) -> None:
        """Store `chunks` so they can later be searched.

        `embedder` is kept for embedding queries at retrieve() time. If
        `embeddings` is given (e.g. loaded from a persisted index), reuse it
        instead of recomputing — otherwise embed `chunks` now.
        """

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the top_k chunks most similar to `query`, ranked by score descending."""
