"""Env-var-driven factories for picking a retriever/reranker/generator at
runtime. Shared by scripts/ask.py and scripts/eval.py so both exercise the
exact same configuration logic -- an eval score is only meaningful if it's
measuring the same thing you'd actually run.
"""

import os

import numpy as np

from rag_lab.embeddings.base import Embedder
from rag_lab.evaluation.judge import AnswerJudge
from rag_lab.generation.anthropic_gen import AnthropicGenerator
from rag_lab.generation.base import Generator
from rag_lab.generation.bedrock_gen import BedrockGenerator
from rag_lab.models import Chunk
from rag_lab.reranking.base import Reranker
from rag_lab.reranking.cross_encoder import CrossEncoderReranker
from rag_lab.reranking.jina import JinaReranker
from rag_lab.retrieval.base import Retriever
from rag_lab.retrieval.bm25 import BM25Retriever
from rag_lab.retrieval.dense_numpy import DenseNumpyRetriever
from rag_lab.retrieval.hybrid import HybridRetriever


def build_retriever(chunks: list[Chunk], embeddings: np.ndarray, embedder: Embedder) -> Retriever:
    kind = os.environ.get("RAG_RETRIEVER", "dense").lower()
    if kind == "bm25":
        retriever = BM25Retriever()
        retriever.index(chunks)
        return retriever
    if kind == "hybrid":
        retriever = HybridRetriever(dense=DenseNumpyRetriever(embedder), bm25=BM25Retriever())
        retriever.index(chunks)
        return retriever
    retriever = DenseNumpyRetriever(embedder)
    retriever.index(chunks, embeddings=embeddings)
    return retriever


def build_reranker() -> Reranker | None:
    kind = os.environ.get("RAG_RERANKER", "none").lower()
    if kind == "cross_encoder":
        return CrossEncoderReranker()
    if kind == "jina":
        return JinaReranker(model=os.environ.get("RAG_JINA_MODEL", "jina-reranker-v2-base-multilingual"))
    return None


def build_generator() -> Generator:
    if os.environ.get("RAG_GENERATOR", "anthropic").lower() == "bedrock":
        return BedrockGenerator(
            model_id=os.environ.get("RAG_BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
            region=os.environ.get("RAG_BEDROCK_REGION", "us-east-2"),
        )
    return AnthropicGenerator()


def build_judge() -> AnswerJudge:
    """RAG_JUDGE_BACKEND defaults to whatever RAG_GENERATOR is set to, since
    that's the credential set you've already got working -- override it
    independently if you want the judge on a different backend than
    generation."""
    backend = os.environ.get("RAG_JUDGE_BACKEND", os.environ.get("RAG_GENERATOR", "anthropic")).lower()
    if backend == "bedrock":
        return AnswerJudge.from_bedrock(
            model_id=os.environ.get("RAG_JUDGE_MODEL_ID", "us.anthropic.claude-sonnet-5"),
            region=os.environ.get("RAG_BEDROCK_REGION", "us-east-2"),
        )
    return AnswerJudge.from_anthropic(model=os.environ.get("RAG_JUDGE_MODEL", "claude-sonnet-5"))
