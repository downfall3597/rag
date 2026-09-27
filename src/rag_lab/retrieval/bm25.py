"""BM25 keyword retrieval: score chunks by term overlap with the query, not by
meaning. This is the classic complement to dense retrieval -- it excels at
exact keyword/jargon/ID matches a dense embedder might blur together, and
needs no embedder at all (this is the retriever base.py's docstring meant
when it said "a future BM25Retriever never touches embeddings at all").

Tokenization and index bookkeeping (term frequencies, document frequencies,
document lengths) are scaffolded below -- they're plumbing, not the BM25
concept itself. The two functions actually worth implementing by hand are
idf() and bm25_score(), which is where the real algorithm lives. retrieve()'s
top-k loop is scaffolded too since it's the same sort-and-slice pattern you
already built for DenseNumpyRetriever -- nothing new to practice there.
"""

import math
import re
from collections import Counter

from rag_lab.models import Chunk, RetrievedChunk
from rag_lab.retrieval.base import Retriever

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever(Retriever):
    def __init__(self, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._doc_term_freqs: list[Counter] = []
        self._doc_freqs: dict[str, int] = {}
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._doc_term_freqs = []
        self._doc_freqs = {}
        self._doc_lengths = []

        for chunk in chunks:
            tokens = tokenize(chunk.text)
            self._doc_lengths.append(len(tokens))
            term_freqs = Counter(tokens)
            self._doc_term_freqs.append(term_freqs)
            for term in term_freqs:
                self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1

        self._avg_doc_length = sum(self._doc_lengths) / len(chunks) if chunks else 0.0

    def idf(self, term: str) -> float:
        """Inverse document frequency for `term`: how rare it is across the corpus.

        YOUR TURN: implement this.

        Formula (BM25's IDF, not the plain classic one):
            idf(term) = log((N - df + 0.5) / (df + 0.5) + 1)
        where:
            N  = self._n_docs()  -- total number of indexed chunks
            df = self._doc_freqs.get(term, 0)  -- number of chunks containing `term`

        A term in fewer documents (lower df) should come out with a HIGHER
        idf than a term in more documents -- that's the "rare terms matter
        more" intuition BM25 (and TF-IDF before it) is built on.
        """
        df = self._doc_freqs.get(term, 0)
        return math.log((self._n_docs() - df + 0.5) / (df + 0.5) + 1)

    def bm25_score(self, query_tokens: list[str], doc_index: int) -> float:
        """BM25 relevance score of the chunk at `doc_index` against `query_tokens`.

        YOUR TURN: implement this.

        For each term in query_tokens, add:
            idf(term) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
        where:
            tf      = how many times `term` appears in this document
                      (self._doc_term_freqs[doc_index].get(term, 0); if 0,
                      this term contributes nothing -- skip it)
            doc_len = self._doc_lengths[doc_index]
            avg_doc_len = self._avg_doc_length

        Sum that over every term in query_tokens and return the total.

        Two things this formula is doing on purpose, worth noticing once it
        works: increasing tf increases the score but with diminishing
        returns (the k1 term caps how much repetition helps -- unlike plain
        term counting), and a term match in a document longer than average
        counts for less than the same match in a shorter document (the b
        term -- a long document matching by chance is less impressive than a
        short one matching precisely).
        """
        score = 0.0
        k1 = self.k1
        b = self.b
        doc_len = self._doc_lengths[doc_index]
        avg_doc_len = self._avg_doc_length
        for term in query_tokens:
            tf = self._doc_term_freqs[doc_index].get(term, 0)
            score += self.idf(term) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))

        return score

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if not self._chunks:
            raise ValueError("index() must be called before retrieve()")
        query_tokens = tokenize(query)
        scored = [
            (self.bm25_score(query_tokens, i), chunk) for i, chunk in enumerate(self._chunks)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [RetrievedChunk(chunk=chunk, score=score) for score, chunk in scored[:top_k]]

    def _n_docs(self) -> int:
        return len(self._chunks)
