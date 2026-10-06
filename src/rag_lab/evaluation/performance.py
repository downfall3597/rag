"""Performance & cost metrics: turning raw per-request measurements (latency
in seconds, token counts) into the numbers that actually drive a build-vs-
buy or config-A-vs-config-B decision -- P50/P95 latency and cost per 1,000
queries.

Pricing here is the direct Anthropic API's published per-token rates.
Bedrock is partner-operated with its own separate pricing that can differ
from these numbers -- treat cost figures from a Bedrock run as an
approximation using these rates, not Bedrock's actual bill.
"""

# Price per 1,000,000 tokens (input, output), direct Anthropic API rates.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
}


def percentile(values: list[float], p: float) -> float:
    """The p-th percentile of `values` (0 <= p <= 100), linear interpolation.

    YOUR TURN: implement this.

    This is the same method numpy's default `percentile` uses, spelled out
    by hand:

    1. Sort `values` ascending.
    2. Compute a fractional "rank" position: `rank = (p / 100) * (n - 1)`,
       where `n = len(values)`. Note this is 0-indexed and generally NOT a
       whole number -- e.g. for 10 values and p=95, rank = 0.95 * 9 = 8.55.
    3. Let `lower = floor(rank)` and `upper = ceil(rank)` (both as list
       indices into the sorted values).
    4. Let `frac = rank - lower` (how far between the two indices the exact
       rank falls -- 0.0 means exactly on `lower`, close to 1.0 means
       almost on `upper`).
    5. Return `sorted_values[lower] + frac * (sorted_values[upper] - sorted_values[lower])`
       -- linear interpolation between the two surrounding values.

    Example: values=[10, 20, 30, 40], p=50 -> rank=(0.5)*3=1.5, lower=1,
    upper=2, frac=0.5 -> 20 + 0.5*(30-20) = 25.0 (halfway between the two
    middle values, which is the standard definition of a median on an
    even-length list).

    Why percentiles instead of just the mean/average: a mean can look fine
    while a meaningful fraction of requests are much slower -- P50 (median)
    tells you the typical-case latency, P95 tells you the "unlucky 1-in-20
    request" case, which matters more for user experience than an average
    that one slow outlier barely moves.
    """
    values.sort()
    n = len(values)
    rank = (p / 100) * (n - 1)
    lower = int(rank)
    upper = min(lower + 1, n - 1)
    frac = rank - lower
    if lower == n - 1:
        return values[lower]
    return values[lower] + frac * (values[upper] - values[lower])   


def cost_per_1000_queries(
    mean_input_tokens: float,
    mean_output_tokens: float,
    model: str,
) -> float:
    """Estimated dollar cost to run 1,000 queries at this model's pricing.

    YOUR TURN: implement this.

    Steps:
    1. Look up `(input_price_per_million, output_price_per_million) =
       MODEL_PRICING[model]`.
    2. Compute the cost of ONE query:
       `(mean_input_tokens / 1_000_000) * input_price_per_million
        + (mean_output_tokens / 1_000_000) * output_price_per_million`
    3. Multiply by 1,000 to scale from one query's cost to 1,000 queries'
       cost, and return that.

    `mean_input_tokens`/`mean_output_tokens` are typically the average
    tokens-per-query across a batch of real requests (computed by whoever
    calls this function), not a single request's exact count -- that's why
    they're floats, not ints.
    """
    input_price_per_million, output_price_per_million = MODEL_PRICING[model]
    cost_per_query = (
         (mean_input_tokens / 1_000_000) * input_price_per_million
         + (mean_output_tokens / 1_000_000) * output_price_per_million
      )
    return cost_per_query * 1_000
