"""Reranker interface: given a query and a candidate list already narrowed
by a Retriever, re-score and re-order them, returning only the best top_k.

A reranker is an optional pipeline step -- it sits between retrieve() and
generate() and touches neither of those interfaces. None, a cross-encoder,
or Jina's API can all be swapped in without changing Retriever, Generator,
or RagPipeline's shape.
"""

from abc import ABC, abstractmethod

from rag_lab.models import RetrievedChunk


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Return the top_k of `candidates`, re-scored and re-ordered for `query`."""
