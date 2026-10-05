"""Retrieval quality metrics: how good is a ranked list of chunk ids against
a known set of relevant chunk ids for one question?

All three functions score a SINGLE question's result. "Mean" Reciprocal
Rank is mean across questions -- that averaging happens in the eval runner
(scripts/eval.py), not here; these functions each handle one question.

Relevance here is binary (a chunk either is or isn't in relevant_ids) --
our eval set doesn't grade "somewhat relevant" vs. "very relevant", just
"is this one of the chunks that actually answers the question."
"""


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """What fraction of the known-relevant chunks did the top-k retrieved results contain?

    YOUR TURN: implement this.

    recall@k = |retrieved_ids[:k] ∩ relevant_ids| / |relevant_ids|

    In words: look at only the first k ids in retrieved_ids, count how many
    of those are also in relevant_ids, and divide by the total number of
    relevant_ids (not by k -- recall is about how much of the truth you
    found, not how much of your result list was correct).

    Return 0.0 if relevant_ids is empty (nothing to find, avoid dividing by
    zero) -- won't happen in our eval set, but keep the function safe.
    """
    if not relevant_ids:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    relevant_count = len(retrieved_set & relevant_ids)
    return relevant_count / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the FIRST relevant chunk in retrieved_ids, for one question.

    YOUR TURN: implement this.

    Find the smallest 1-indexed position i such that retrieved_ids[i-1] is
    in relevant_ids, and return 1 / i. E.g. if the first relevant chunk
    shows up at position 3, return 1/3 -- putting the right answer at rank 1
    scores 1.0, rank 2 scores 0.5, rank 10 scores 0.1, and so on. If no
    relevant chunk appears anywhere in retrieved_ids, return 0.0.

    Why "reciprocal" rank and not just rank: it keeps the score bounded
    between 0 and 1 and makes the difference between rank 1 and rank 2 (a
    0.5 point swing) much larger than the difference between rank 20 and
    rank 21 (a ~0.002 point swing) -- matching the intuition that whether
    the right answer is 1st or 2nd matters a lot more than whether it's
    20th or 21st.
    """
    for i, chunk_id in enumerate(retrieved_ids):
        if chunk_id in relevant_ids:
            return 1.0 / (i + 1)  # i is 0-indexed, so add 1 for rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k, for one question (binary relevance).

    YOUR TURN: implement this.

    Step 1 -- DCG@k (Discounted Cumulative Gain): for each position i from 1
    to k (1-indexed), let rel_i = 1 if retrieved_ids[i-1] is in relevant_ids,
    else 0. Sum rel_i / log2(i + 1) over those k positions. (Use
    math.log2.) This rewards relevant results, discounted by how far down
    the list they are -- a relevant chunk at position 1 contributes
    1/log2(2) = 1.0, the same chunk at position 10 contributes only
    1/log2(11) ≈ 0.29.

    Step 2 -- IDCG@k (Ideal DCG): the DCG@k you'd get from the best possible
    ordering -- every relevant chunk placed at the very top. That's the
    same formula, but computed as if the first min(len(relevant_ids), k)
    positions were all relevant (rel_i = 1) and the rest were not.

    Step 3 -- nDCG@k = DCG@k / IDCG@k. Return 0.0 if IDCG@k is 0 (no
    relevant chunks to find, avoid dividing by zero).

    Why normalize at all: DCG@k alone grows with how many relevant chunks
    exist, which makes it unusable for comparing across questions with
    different numbers of relevant chunks. Dividing by the best-possible
    score for *this specific question* puts every question on the same
    0-to-1 scale regardless of how many relevant chunks it has.
    """
    import math

    # Step 1: Compute DCG@k
    dcg = 0.0
    for i in range(k):
        if i < len(retrieved_ids) and retrieved_ids[i] in relevant_ids:
            dcg += 1 / math.log2(i + 2)  # i + 2 because log2 is 1-indexed

    # Step 2: Compute IDCG@k
    ideal_relevant_count = min(len(relevant_ids), k)
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal_relevant_count))

    # Step 3: Compute nDCG@k
    if idcg == 0:
        return 0.0
    return dcg / idcg
