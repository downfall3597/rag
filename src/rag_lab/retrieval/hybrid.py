"""Hybrid retrieval: combine a dense retriever and a BM25 retriever so a chunk
either one independently ranks highly still surfaces, covering both
retrieval methods' failure modes (dense misses exact/rare-term matches,
BM25 misses semantically-related chunks with no shared words) at once.

The combination method is Reciprocal Rank Fusion (RRF), scaffolded as its
own function below since it's the concept worth implementing by hand.
HybridRetriever itself is just plumbing: run both retrievers, fuse their
results.
"""

from rag_lab.models import Chunk, RetrievedChunk
from rag_lab.retrieval.base import Retriever

DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_K = 50


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]], k: int, top_k: int
) -> list[RetrievedChunk]:
    """Fuse multiple ranked result lists into one ranking using RRF.

    YOUR TURN: implement this.

    RRF fuses lists using each item's *rank* (1st, 2nd, 3rd, ...), not its
    raw score -- that's the whole point. A dense cosine similarity (-1 to 1)
    and a BM25 score (0 to unbounded) are on completely different scales, so
    summing them directly would let whichever method happens to produce
    bigger numbers dominate the fusion regardless of actual relevance. Rank
    position is scale-free and directly comparable across any retriever.

    For each list in result_lists:
        for each chunk at position i (0-indexed) in that list:
            add 1 / (k + i + 1) to that chunk's running fused score
            (the "+ 1" makes the rank 1-indexed: the top result is rank 1,
            not rank 0 -- this matches the standard RRF formula)

    A chunk that appears in more than one list accumulates a contribution
    from each list it appears in -- that's what lets a chunk both retrievers
    agree on rank above one that only a single retriever found, even if
    neither ranked it first.

    Identify "the same chunk" across lists by `.chunk.id` (RetrievedChunk
    objects from different retrievers are different Python objects even when
    they wrap the same underlying Chunk).

    Return the top_k chunks by fused score, descending, as RetrievedChunk
    objects with `score` set to the fused RRF score -- not either retriever's
    original score, since those aren't meaningful anymore once fused.
    """
    fused_scores: dict[str, float] = {}
    chunks_by_id: dict[str, Chunk] = {}

    for ranked_list in result_lists:
        for i, retrieved in enumerate(ranked_list):
            chunk_id = retrieved.chunk.id
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1 / (k + i + 1)
            chunks_by_id.setdefault(chunk_id, retrieved.chunk)

    ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
    return [
        RetrievedChunk(chunk=chunks_by_id[cid], score=fused_scores[cid])
        for cid in ranked_ids[:top_k]
    ]


class HybridRetriever(Retriever):
    def __init__(
        self,
        dense: Retriever,
        bm25: Retriever,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
    ) -> None:
        self._dense = dense
        self._bm25 = bm25
        self._rrf_k = rrf_k
        self._candidate_k = candidate_k

    def index(self, chunks: list[Chunk]) -> None:
        self._dense.index(chunks)
        self._bm25.index(chunks)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        dense_results = self._dense.retrieve(query, self._candidate_k)
        bm25_results = self._bm25.retrieve(query, self._candidate_k)
        return reciprocal_rank_fusion([dense_results, bm25_results], self._rrf_k, top_k)
