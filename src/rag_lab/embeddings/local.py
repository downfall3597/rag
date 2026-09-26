"""MiniLM embedder, built on raw `transformers` rather than the
`sentence-transformers` wrapper, so the pooling/normalization math below is
explicit instead of hidden inside a library call.

Background: AutoModel gives you one vector per *token* (shape
(batch, seq_len, hidden_dim)), not one vector per sentence. Turning a
sequence of token vectors into a single sentence vector is "pooling."
MiniLM was trained to be pooled by *mean-pooling*: averaging the token
vectors, but only over the real tokens — padding tokens (added so every
sequence in a batch is the same length) must be excluded from the average,
which is what the attention mask is for.
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from rag_lab.embeddings.base import Embedder

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def mean_pool(
    token_embeddings: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Average token_embeddings over the sequence dimension, weighted by attention_mask.

    YOUR TURN: implement this.

    Args:
        token_embeddings: shape (batch_size, seq_len, hidden_dim) — one vector
            per token, per sequence in the batch.
        attention_mask: shape (batch_size, seq_len) — 1 for real tokens, 0 for
            padding tokens.

    Returns:
        shape (batch_size, hidden_dim) — one vector per sequence, computed as
        the mean of that sequence's real-token vectors (padding excluded).

    Hint: broadcast attention_mask to the same shape as token_embeddings,
    multiply, sum over the seq_len dimension, then divide by the number of
    real tokens in each sequence (also from attention_mask) — not by seq_len,
    which would incorrectly count padding.
    """
    token_embeddings = token_embeddings * attention_mask.unsqueeze(-1).float()
    token_embeddings = token_embeddings.sum(1)
    attention_mask = attention_mask.sum(1, keepdim=True)
    return token_embeddings / attention_mask


def l2_normalize(vectors: torch.Tensor) -> torch.Tensor:
    """L2-normalize each row of vectors to unit length.

    YOUR TURN: implement this.

    Args:
        vectors: shape (batch_size, hidden_dim).

    Returns:
        Same shape, where each row has been divided by its own L2 norm, so
        every row now has length 1.

    Why this matters: cosine similarity is dot(a, b) / (||a|| * ||b||). If
    every vector is already unit length, ||a|| = ||b|| = 1, so cosine
    similarity reduces to a plain dot product — which is what the
    DenseNumpyRetriever you'll build next assumes it can rely on for
    performance, but should still compute the full formula defensively.
    """
    for row in vectors:
        norm = torch.norm(row, p=2)
        if norm > 0:
            row /= norm

    return vectors


class MiniLMEmbedder(Embedder):
    def __init__(self, model_name: str = MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            output = self.model(**encoded)
        pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
        normalized = l2_normalize(pooled)
        return normalized.numpy()
