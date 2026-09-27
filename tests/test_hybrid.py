import numpy as np
import pytest

from rag_lab.embeddings.base import Embedder
from rag_lab.models import Chunk, RetrievedChunk
from rag_lab.retrieval.bm25 import BM25Retriever
from rag_lab.retrieval.dense_numpy import DenseNumpyRetriever
from rag_lab.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


def _rc(chunk_id: str, score: float = 0.0) -> RetrievedChunk:
    chunk = Chunk(id=chunk_id, document_id="doc", source="doc.md", text=f"text {chunk_id}", position=0)
    return RetrievedChunk(chunk=chunk, score=score)


# --- reciprocal_rank_fusion(): pure function, hand-built ranked lists ---


def test_fusion_combines_contributions_from_both_lists():
    list_a = [_rc("1"), _rc("2"), _rc("3")]  # ranks: 1=1st, 2=2nd, 3=3rd
    list_b = [_rc("2"), _rc("3"), _rc("1")]  # ranks: 2=1st, 3=2nd, 1=3rd
    k = 60

    results = reciprocal_rank_fusion([list_a, list_b], k=k, top_k=3)

    expected = {
        "1": 1 / (k + 1) + 1 / (k + 3),
        "2": 1 / (k + 2) + 1 / (k + 1),
        "3": 1 / (k + 3) + 1 / (k + 2),
    }
    by_id = {rc.chunk.id: rc.score for rc in results}
    for chunk_id, expected_score in expected.items():
        assert by_id[chunk_id] == pytest.approx(expected_score)

    # chunk "2" has the highest fused score (rank 1 in B, rank 2 in A)
    assert results[0].chunk.id == "2"


def test_chunk_ranked_in_both_lists_beats_top_rank_in_only_one_list():
    # "only_one" is rank 1 in list_a and absent from list_b.
    # "in_both" is rank 2 in both lists -- weaker individually, but agreed on.
    list_a = [_rc("only_one"), _rc("in_both")]
    list_b = [_rc("something_else"), _rc("in_both")]

    results = reciprocal_rank_fusion([list_a, list_b], k=60, top_k=2)

    assert results[0].chunk.id == "in_both"


def test_fusion_respects_top_k():
    list_a = [_rc("1"), _rc("2"), _rc("3"), _rc("4")]
    results = reciprocal_rank_fusion([list_a], k=60, top_k=2)
    assert len(results) == 2
    assert [rc.chunk.id for rc in results] == ["1", "2"]


# --- HybridRetriever: end-to-end wiring with real dense + BM25 retrievers ---


class FakeEmbedder(Embedder):
    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.array([self._vectors[text] for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return np.array(self._vectors[text])


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(id=chunk_id, document_id="doc", source="doc.md", text=text, position=0)


def test_hybrid_retriever_indexes_and_retrieves_via_both_sub_retrievers():
    chunks = [
        _make_chunk("a", "the cat sat on the mat"),
        _make_chunk("b", "completely unrelated text about weather"),
    ]
    # Give the dense side identical (uninformative) vectors, so any ranking
    # signal in this test comes from BM25's keyword match, proving both
    # sub-retrievers are actually wired into the fused result.
    embedder = FakeEmbedder(
        {
            "the cat sat on the mat": [1.0, 0.0],
            "completely unrelated text about weather": [1.0, 0.0],
            "cat mat": [1.0, 0.0],
        }
    )
    hybrid = HybridRetriever(dense=DenseNumpyRetriever(embedder), bm25=BM25Retriever())
    hybrid.index(chunks)

    results = hybrid.retrieve("cat mat", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.id == "a"
