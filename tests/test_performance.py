import pytest

from rag_lab.evaluation.performance import cost_per_1000_queries, percentile

# --- percentile ---


def test_percentile_50_is_the_median_of_an_even_length_list():
    assert percentile([10, 20, 30, 40], 50) == pytest.approx(25.0)


def test_percentile_0_returns_the_minimum():
    assert percentile([5, 1, 9, 3], 0) == pytest.approx(1.0)


def test_percentile_100_returns_the_maximum():
    assert percentile([5, 1, 9, 3], 100) == pytest.approx(9.0)


def test_percentile_95_interpolates_between_two_values():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # n=10, rank = 0.95 * 9 = 8.55 -> between index 8 (value 9) and 9 (value 10)
    assert percentile(values, 95) == pytest.approx(9.55)


def test_percentile_of_a_single_value_list():
    assert percentile([42.0], 50) == pytest.approx(42.0)


def test_percentile_does_not_require_pre_sorted_input():
    assert percentile([30, 10, 40, 20], 50) == pytest.approx(25.0)


# --- cost_per_1000_queries ---


def test_cost_matches_hand_computed_value():
    # (1000/1e6)*1.00 + (500/1e6)*5.00 = 0.001 + 0.0025 = 0.0035 per query
    expected = 0.0035 * 1000
    assert cost_per_1000_queries(1000, 500, "claude-haiku-4-5") == pytest.approx(expected)


def test_cost_increases_with_more_output_tokens():
    low = cost_per_1000_queries(1000, 100, "claude-haiku-4-5")
    high = cost_per_1000_queries(1000, 1000, "claude-haiku-4-5")
    assert high > low


def test_cost_uses_the_specified_models_pricing():
    haiku_cost = cost_per_1000_queries(1000, 500, "claude-haiku-4-5")
    sonnet_cost = cost_per_1000_queries(1000, 500, "claude-sonnet-5")
    # sonnet-5 is priced at 2x haiku-4-5 on both input and output
    assert sonnet_cost == pytest.approx(haiku_cost * 2)
