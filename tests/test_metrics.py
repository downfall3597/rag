import math

import pytest

from rag_lab.evaluation.metrics import mrr, ndcg_at_k, recall_at_k

# --- recall_at_k ---


def test_recall_at_k_counts_overlap_with_relevant_set():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d", "z"}  # "z" is relevant but never retrieved at all
    assert recall_at_k(retrieved, relevant, k=4) == pytest.approx(2 / 3)


def test_recall_at_k_only_looks_at_the_first_k_results():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"d"}
    assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(0.0)
    assert recall_at_k(retrieved, relevant, k=4) == pytest.approx(1.0)


def test_recall_at_k_with_empty_relevant_set_is_zero():
    assert recall_at_k(["a", "b"], set(), k=2) == 0.0


# --- mrr ---


def test_mrr_is_one_when_first_relevant_hit_is_rank_one():
    assert mrr(["a", "b", "c"], {"a"}) == pytest.approx(1.0)


def test_mrr_is_reciprocal_of_the_first_hits_rank():
    assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)


def test_mrr_only_counts_the_first_relevant_hit():
    # "b" at rank 3 is also relevant, but "a" at rank 2 is found first.
    assert mrr(["x", "a", "b"], {"a", "b"}) == pytest.approx(1 / 2)


def test_mrr_is_zero_when_nothing_relevant_is_found():
    assert mrr(["x", "y", "z"], {"a"}) == 0.0


# --- ndcg_at_k ---


def test_ndcg_at_k_is_one_for_a_perfect_ranking():
    # Both relevant chunks already occupy the top two positions.
    retrieved = ["a", "b", "x"]
    relevant = {"a", "b"}
    assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_penalizes_relevant_results_ranked_lower():
    retrieved = ["x", "a", "b"]
    relevant = {"a", "b"}

    dcg = 0 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)  # ideal: both relevant chunks ranked first
    expected = dcg / idcg

    assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(expected)
    assert 0 < ndcg_at_k(retrieved, relevant, k=3) < 1.0


def test_ndcg_at_k_is_zero_when_nothing_relevant_is_found():
    assert ndcg_at_k(["x", "y", "z"], {"a"}, k=3) == pytest.approx(0.0)


def test_ndcg_at_k_with_empty_relevant_set_is_zero():
    assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0
