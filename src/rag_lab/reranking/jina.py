"""Jina AI's hosted reranking API (https://jina.ai/reranker/). Pure HTTP
plumbing -- there's no algorithm to hand-implement here, unlike the
cross-encoder reranker; the actual model runs entirely on Jina's servers.
Requires a JINA_API_KEY (free tier available at jina.ai).
"""

import os
from typing import Any, Callable

import requests

from rag_lab.models import RetrievedChunk
from rag_lab.reranking.base import Reranker

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
DEFAULT_MODEL = "jina-reranker-v2-base-multilingual"


class JinaReranker(Reranker):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        post_fn: Callable[..., Any] = requests.post,
    ):
        self.api_key = api_key or os.environ["JINA_API_KEY"]
        self.model = model
        self._post_fn = post_fn

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []

        response = self._post_fn(
            JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "query": query,
                "documents": [c.chunk.text for c in candidates],
                "top_n": top_k,
            },
        )
        response.raise_for_status()
        results = response.json()["results"]

        return [
            RetrievedChunk(chunk=candidates[r["index"]].chunk, score=r["relevance_score"])
            for r in results
        ]
