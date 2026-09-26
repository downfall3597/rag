"""Wires an Embedder + Retriever + Generator into one end-to-end RAG pipeline.

Direct constructor injection, no config/factory system — that's deliberately
deferred until there's a second retriever/embedder/generator to choose
between (BM25, hybrid, reranking). Building a config layer for one
implementation each would be premature abstraction with no payoff yet.
"""

from rag_lab.embeddings.base import Embedder
from rag_lab.generation.base import Generator
from rag_lab.models import GenerationResult
from rag_lab.retrieval.base import Retriever


class RagPipeline:
    def __init__(self, embedder: Embedder, retriever: Retriever, generator: Generator):
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator

    def answer(self, query: str, top_k: int = 4) -> GenerationResult:
        retrieved = self.retriever.retrieve(query, top_k=top_k)
        return self.generator.generate(query, retrieved)
