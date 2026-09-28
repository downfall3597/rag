"""Cross-encoder reranking: feed the query and a candidate's text into the
model TOGETHER, as one input, rather than embedding them separately and
comparing vectors afterward (what dense retrieval does). Letting the model
attend across both texts at once makes it far more accurate at judging
"does this passage actually answer this query" -- but it's much more
expensive per comparison, which is why it only runs over retrieval's
already-narrowed candidate list, never the whole corpus.

Model loading and the query/text pairing for the tokenizer are scaffolded
below (plumbing). rank_by_score is the piece worth implementing by hand: a
small, testable function decoupled from the model itself.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rag_lab.models import RetrievedChunk
from rag_lab.reranking.base import Reranker

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rank_by_score(candidates: list[RetrievedChunk], scores: list[float], top_k: int) -> list[RetrievedChunk]:
    """Pair each candidate with its score, sort descending, keep the top_k.

    YOUR TURN: implement this.

    `scores[i]` is the cross-encoder's relevance score for `candidates[i]` --
    they're the same length and index-aligned (score i belongs to candidate i).

    Return a new list of RetrievedChunk objects for the top_k, in descending
    score order, with `score` set to the cross-encoder score -- not
    `candidates[i].score`, which was the original retriever's score (cosine
    similarity, BM25, or an RRF fusion score). Those aren't the same scale as
    a cross-encoder logit, so don't carry the old score forward -- same
    principle as RRF discarding each retriever's original score once fused.
    """
    scored_candidates = list(zip(candidates, scores))
    scored_candidates.sort(key=lambda pair: pair[1], reverse=True)
    return [
        RetrievedChunk(chunk=candidate.chunk, score=score)
        for candidate, score in scored_candidates[:top_k]
    ]

class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        queries = [query] * len(candidates)
        texts = [c.chunk.text for c in candidates]
        encoded = self.tokenizer(queries, texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**encoded).logits.squeeze(-1)
        scores = logits.tolist()
        return rank_by_score(candidates, scores, top_k)
