import numpy as np
import pytest

from rag_lab.embeddings.base import Embedder
from rag_lab.models import Chunk
from rag_lab.retrieval.dense_numpy import DenseNumpyRetriever, cosine_similarity

# --- cosine_similarity: hand-built vectors with a known expected answer. ---


def test_cosine_similarity_of_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_of_opposite_vectors_is_negative_one():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_is_scale_invariant():
    # same direction, different magnitude -- cosine similarity ignores length.
    a = np.array([1.0, 2.0])
    b = np.array([2.0, 4.0])
    assert cosine_similarity(a, b) == pytest.approx(1.0)


# --- DenseNumpyRetriever.retrieve: a fake embedder removes the real model
# from the picture, so this tests only the ranking/top-k logic. ---


class FakeEmbedder(Embedder):
    """Test double: returns pre-set vectors by exact text match, no model needed."""

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.array([self._vectors[text] for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return np.array(self._vectors[text])


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(id=chunk_id, document_id="doc", source="doc.md", text=text, position=0)


def test_retrieve_ranks_by_similarity_and_respects_top_k():
    chunks = [
        _make_chunk("a", "parallel to query"),
        _make_chunk("b", "orthogonal to query"),
        _make_chunk("c", "opposite of query"),
    ]
    embedder = FakeEmbedder(
        {
            "parallel to query": [1.0, 0.0],
            "orthogonal to query": [0.0, 1.0],
            "opposite of query": [-1.0, 0.0],
            "the query": [1.0, 0.0],
        }
    )
    retriever = DenseNumpyRetriever(embedder)
    retriever.index(chunks)

    results = retriever.retrieve("the query", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.id == "a"
    assert results[0].score == pytest.approx(1.0)
    assert results[1].chunk.id == "b"
    assert results[1].score == pytest.approx(0.0)


def test_retrieve_before_index_raises():
    retriever = DenseNumpyRetriever(FakeEmbedder({}))
    with pytest.raises(ValueError):
        retriever.retrieve("anything", top_k=1)
