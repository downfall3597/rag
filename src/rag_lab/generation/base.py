"""Generator interface: given a query and its retrieved chunks, produce an answer."""

from abc import ABC, abstractmethod

from rag_lab.models import GenerationResult, RetrievedChunk


class Generator(ABC):
    @abstractmethod
    def generate(self, query: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        """Produce a grounded answer to `query`, citing which of `chunks` it used."""
