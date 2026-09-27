import numpy as np

from rag_lab.embeddings.base import Embedder
from rag_lab.generation.fake import FakeGenerator
from rag_lab.models import Chunk
from rag_lab.pipeline import RagPipeline
from rag_lab.retrieval.dense_numpy import DenseNumpyRetriever


class FakeEmbedder(Embedder):
    """Deterministic embedder for pipeline wiring tests -- no real model needed."""

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.array([self._vectors[text] for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return np.array(self._vectors[text])


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(id=chunk_id, document_id="doc", source="doc.md", text=text, position=0)


def test_pipeline_retrieves_then_generates_and_returns_citations():
    chunks = [_make_chunk("a", "relevant fact"), _make_chunk("b", "irrelevant fact")]
    embedder = FakeEmbedder(
        {
            "relevant fact": [1.0, 0.0],
            "irrelevant fact": [0.0, 1.0],
            "a relevant query": [1.0, 0.0],
        }
    )
    retriever = DenseNumpyRetriever(embedder)
    retriever.index(chunks)
    generator = FakeGenerator(answer="a fake grounded answer [1]")

    pipeline = RagPipeline(embedder=embedder, retriever=retriever, generator=generator)
    result = pipeline.answer("a relevant query", top_k=1)

    assert result.query == "a relevant query"
    assert result.answer == "a fake grounded answer [1]"
    # the pipeline passed exactly the retriever's top-1 result to the generator
    assert len(result.citations) == 1
    assert result.citations[0].chunk.id == "a"
    # the generator received what the retriever actually produced
    assert generator.last_query == "a relevant query"
    assert generator.last_chunks == result.citations
