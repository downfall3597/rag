"""Retriever interface: given a query, return the most relevant chunks.

retrieve() takes raw query text, not a precomputed embedding, and index()
only takes chunks -- neither mentions embeddings. This matters for
retrievers that don't use embeddings at all: a BM25Retriever scores chunks
by keyword overlap and never touches an Embedder. A retriever that *does*
need one (like DenseNumpyRetriever) takes it as a constructor argument
instead, keeping this shared interface provider-agnostic. That's what lets
a HybridRetriever compose a dense and a BM25 retriever behind this one
interface, and a pipeline swap retrievers without caring which kind it's
using.
"""

from abc import ABC, abstractmethod

from rag_lab.models import Chunk, RetrievedChunk


class Retriever(ABC):
    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None:
        """Store `chunks` so they can later be searched."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the top_k chunks most similar to `query`, ranked by score descending."""
