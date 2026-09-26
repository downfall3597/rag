"""Test double for Generator — no network call, so tests can run with no API key
and no cost. Records the last call it received so tests can assert on it."""

from rag_lab.generation.base import Generator
from rag_lab.models import GenerationResult, RetrievedChunk


class FakeGenerator(Generator):
    def __init__(self, answer: str = "fake answer"):
        self.answer = answer
        self.last_query: str | None = None
        self.last_chunks: list[RetrievedChunk] | None = None

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        self.last_query = query
        self.last_chunks = chunks
        return GenerationResult(query=query, answer=self.answer, citations=chunks)
