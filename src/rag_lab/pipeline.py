"""Wires an Embedder + Retriever + Generator (+ optional Reranker) into one
end-to-end RAG pipeline.

Direct constructor injection, no config/factory system — that's deliberately
deferred until there's a real need for one (a results dashboard comparing
many configurations at once). Building a config layer for a handful of
implementations would be premature abstraction with no payoff yet.
"""

from rag_lab.embeddings.base import Embedder
from rag_lab.generation.base import Generator
from rag_lab.models import GenerationResult
from rag_lab.reranking.base import Reranker
from rag_lab.retrieval.base import Retriever


class RagPipeline:
    def __init__(
        self,
        embedder: Embedder,
        retriever: Retriever,
        generator: Generator,
        reranker: Reranker | None = None,
    ):
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker

    def answer(self, query: str, top_k: int = 4, retrieve_k: int | None = None) -> GenerationResult:
        """retrieve_k controls how many candidates the retriever returns before
        an optional reranker narrows them down to top_k. Defaults to top_k when
        there's no reranker (nothing to narrow from), or top_k * 5 when there
        is one (give the reranker a real shortlist to work with, not just the
        final count)."""
        if retrieve_k is None:
            retrieve_k = top_k * 5 if self.reranker else top_k

        candidates = self.retriever.retrieve(query, top_k=retrieve_k)
        if self.reranker:
            candidates = self.reranker.rerank(query, candidates, top_k)
        return self.generator.generate(query, candidates)
