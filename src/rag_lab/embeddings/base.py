"""Embedder interface: turns text into vectors for similarity search.

Split into embed_documents / embed_query (rather than one embed() method)
because some embedding models are asymmetric — they embed a search query
differently from a passage of text (e.g. Voyage/OpenAI/BGE add an
instruction prefix for queries). Our MiniLM model is symmetric, so both
methods do the same thing today, but callers never need to know that.
"""

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of document/chunk texts. Returns shape (len(texts), dim)."""
        

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string. Returns shape (dim,)."""
