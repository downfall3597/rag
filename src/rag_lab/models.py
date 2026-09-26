"""Shared data contracts used across ingestion, retrieval, and generation."""

from pydantic import BaseModel


class Document(BaseModel):
    """A single source file loaded from disk, before chunking."""

    id: str
    source: str
    text: str


class Chunk(BaseModel):
    """A piece of a Document, sized for embedding."""

    id: str
    document_id: str
    source: str
    text: str
    position: int


class RetrievedChunk(BaseModel):
    """A Chunk returned by a Retriever, with its similarity score."""

    chunk: Chunk
    score: float


class GenerationResult(BaseModel):
    """The final answer produced from a query and its retrieved chunks."""

    query: str
    answer: str
    citations: list[RetrievedChunk]
