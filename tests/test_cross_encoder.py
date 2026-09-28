import pytest

from rag_lab.models import Chunk, RetrievedChunk
from rag_lab.reranking.cross_encoder import CrossEncoderReranker, rank_by_score


def _rc(chunk_id: str, text: str, original_score: float = 0.5) -> RetrievedChunk:
    chunk = Chunk(id=chunk_id, document_id="doc", source="doc.md", text=text, position=0)
    return RetrievedChunk(chunk=chunk, score=original_score)


# --- rank_by_score(): pure logic, no model. ---


def test_rank_by_score_orders_descending():
    candidates = [_rc("a", "text a"), _rc("b", "text b"), _rc("c", "text c")]
    scores = [0.1, 0.9, 0.5]  # b > c > a

    result = rank_by_score(candidates, scores, top_k=3)

    assert [rc.chunk.id for rc in result] == ["b", "c", "a"]


def test_rank_by_score_respects_top_k():
    candidates = [_rc("a", "text a"), _rc("b", "text b"), _rc("c", "text c")]
    scores = [0.1, 0.9, 0.5]

    result = rank_by_score(candidates, scores, top_k=2)

    assert len(result) == 2
    assert [rc.chunk.id for rc in result] == ["b", "c"]


def test_rank_by_score_replaces_the_original_score():
    # original_score=0.5 for every candidate on purpose -- if the result
    # still shows 0.5, the cross-encoder score never got applied.
    candidates = [_rc("a", "text a", original_score=0.5), _rc("b", "text b", original_score=0.5)]
    scores = [0.9, 0.2]

    result = rank_by_score(candidates, scores, top_k=2)

    assert result[0].chunk.id == "a"
    assert result[0].score == pytest.approx(0.9)
    assert result[1].score == pytest.approx(0.2)


# --- CrossEncoderReranker: integration test with the real model. ---
# First run downloads the model (~67MB) from Hugging Face Hub.


@pytest.fixture(scope="module")
def reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


def test_reranker_prefers_the_relevant_passage(reranker: CrossEncoderReranker):
    query = "How much paid vacation do employees get?"
    candidates = [
        _rc("relevant", "Full-time employees accrue 15 days of paid vacation per year."),
        _rc("irrelevant", "The office espresso machine is on the third floor."),
    ]

    result = reranker.rerank(query, candidates, top_k=2)

    assert result[0].chunk.id == "relevant"
    assert result[0].score > result[1].score


def test_reranker_handles_empty_candidates(reranker: CrossEncoderReranker):
    assert reranker.rerank("any query", [], top_k=4) == []
