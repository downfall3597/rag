import math

import pytest

from rag_lab.models import Chunk
from rag_lab.retrieval.bm25 import BM25Retriever


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(id=chunk_id, document_id="doc", source="doc.md", text=text, position=0)


# --- idf() ---


@pytest.fixture
def small_corpus_retriever() -> BM25Retriever:
    # N=3. df(cat)=2 (docs 0, 2), df(dog)=1 (doc 1), df(sat)=2 (docs 0, 1).
    chunks = [
        _make_chunk("0", "cat sat on the mat"),
        _make_chunk("1", "dog sat on the mat"),
        _make_chunk("2", "cat cat cat"),
    ]
    retriever = BM25Retriever()
    retriever.index(chunks)
    return retriever


def test_idf_matches_the_bm25_formula(small_corpus_retriever: BM25Retriever):
    expected = math.log((3 - 2 + 0.5) / (2 + 0.5) + 1)
    assert small_corpus_retriever.idf("cat") == pytest.approx(expected)


def test_idf_of_rarer_term_is_higher(small_corpus_retriever: BM25Retriever):
    # "dog" appears in 1 doc, "cat" in 2 -- rarer terms should score higher.
    assert small_corpus_retriever.idf("dog") > small_corpus_retriever.idf("cat")


def test_idf_of_unseen_term_still_computes(small_corpus_retriever: BM25Retriever):
    expected = math.log((3 - 0 + 0.5) / (0 + 0.5) + 1)
    assert small_corpus_retriever.idf("elephant") == pytest.approx(expected)
    assert small_corpus_retriever.idf("elephant") > small_corpus_retriever.idf("dog")


# --- bm25_score(): term frequency saturation (k1) ---


def test_score_increases_with_term_frequency_but_with_diminishing_returns():
    # All three docs are the same length (5 tokens), so the length-
    # normalization term is identical for all of them -- isolating the
    # effect of term frequency alone.
    chunks = [
        _make_chunk("tf1", "cat x x x x"),
        _make_chunk("tf2", "cat cat x x x"),
        _make_chunk("tf5", "cat cat cat cat cat"),
    ]
    retriever = BM25Retriever()
    retriever.index(chunks)

    score_tf1 = retriever.bm25_score(["cat"], 0)
    score_tf2 = retriever.bm25_score(["cat"], 1)
    score_tf5 = retriever.bm25_score(["cat"], 2)

    assert score_tf1 < score_tf2 < score_tf5
    # Diminishing returns: doubling term frequency does not double the score.
    assert (score_tf2 / score_tf1) < 2.0


# --- bm25_score(): document length normalization (b) ---


def test_score_penalizes_the_same_term_frequency_in_a_longer_document():
    chunks = [
        _make_chunk("short", "cat x"),
        _make_chunk("long", "cat x x x x x x x x x"),
    ]
    retriever = BM25Retriever()
    retriever.index(chunks)

    score_short = retriever.bm25_score(["cat"], 0)
    score_long = retriever.bm25_score(["cat"], 1)

    assert score_short > score_long


def test_missing_query_term_contributes_nothing():
    chunks = [_make_chunk("a", "cat cat cat")]
    retriever = BM25Retriever()
    retriever.index(chunks)

    assert retriever.bm25_score(["dog"], 0) == 0.0


# --- retrieve() ---


def test_retrieve_ranks_by_keyword_overlap_and_respects_top_k():
    chunks = [
        _make_chunk("a", "the cat sat on the mat"),
        _make_chunk("b", "the dog sat on the rug"),
        _make_chunk("c", "completely unrelated text about weather"),
    ]
    retriever = BM25Retriever()
    retriever.index(chunks)

    results = retriever.retrieve("cat mat", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.id == "a"
    assert results[0].score > results[1].score


def test_retrieve_before_index_raises():
    retriever = BM25Retriever()
    with pytest.raises(ValueError):
        retriever.retrieve("anything", top_k=1)
