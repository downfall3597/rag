"""Ask a question against the persisted index.

Usage: python scripts/ask.py "your question here" [top_k]

Set RAG_RETRIEVER=bm25 or RAG_RETRIEVER=hybrid to use keyword or hybrid
retrieval instead of dense (the default) -- see scripts/compare_retrievers.py
for a free, no-LLM-call side-by-side comparison of all three.

Set RAG_RERANKER=cross_encoder or RAG_RERANKER=jina to rerank retrieval's
candidates before generation (default: no reranking). Jina reads
RAG_JINA_MODEL (default jina-reranker-v2-base-multilingual) and needs a
JINA_API_KEY environment variable.

Set RAG_GENERATOR=bedrock to answer via AWS Bedrock's Converse API instead of
the direct Anthropic API (the default). Bedrock also reads RAG_BEDROCK_REGION
(default us-east-2) and RAG_BEDROCK_MODEL_ID (default
us.anthropic.claude-haiku-4-5-20251001-v1:0 -- a cross-region inference
profile ID, since this model only supports INFERENCE_PROFILE invocation, not
a bare model ID), and uses whatever AWS credentials are already active in
the environment (SSO profile, env vars, IAM role).

Loads data/index/chunks.json + embeddings.npy (built by scripts/ingest.py),
runs the full retrieve -> generate pipeline, and prints the answer plus
which chunks were retrieved and their similarity scores.

Secrets (ANTHROPIC_API_KEY, JINA_API_KEY) are loaded from a .env file in the
repo root if present -- see .env for the expected keys. Real shell env vars
still take precedence over .env, so `JINA_API_KEY=x python scripts/ask.py`
overrides whatever's in .env for that one run.
"""

import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from rag_lab.config import build_generator, build_reranker, build_retriever
from rag_lab.embeddings.local import MiniLMEmbedder
from rag_lab.models import Chunk

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_FOLDER = REPO_ROOT / "data" / "index"


def load_index() -> tuple[list[Chunk], np.ndarray]:
    chunks_path = INDEX_FOLDER / "chunks.json"
    embeddings_path = INDEX_FOLDER / "embeddings.npy"
    if not chunks_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            f"No index found in {INDEX_FOLDER}. Run `python scripts/ingest.py` first."
        )
    chunks = [Chunk(**raw) for raw in json.loads(chunks_path.read_text())]
    embeddings = np.load(embeddings_path)
    return chunks, embeddings


def main(query: str, top_k: int) -> None:
    chunks, embeddings = load_index()

    embedder = MiniLMEmbedder()
    retriever = build_retriever(chunks, embeddings, embedder)
    generator = build_generator()
    reranker = build_reranker()

    # Retrieve and (if configured) rerank exactly once, printing progress as
    # we go, then hand the final candidates straight to the generator -- not
    # going through RagPipeline.answer() here, since that would redo both
    # steps a second time just to get the same intermediate results back for
    # printing. For Jina specifically, a second rerank() call is a second
    # real, paid API request, not just wasted local compute.
    retrieve_k = top_k * 5 if reranker else top_k
    candidates = retriever.retrieve(query, top_k=retrieve_k)

    print("=" * 70)
    label = "RETRIEVED CHUNKS (before reranking)" if reranker else f"RETRIEVED CHUNKS (top {top_k})"
    print(label)
    print("=" * 70)
    for i, rc in enumerate(candidates, start=1):
        print(f"[{i}] score={rc.score:.4f} source={rc.chunk.source}")
        print(f"    {rc.chunk.text[:120]}...")
    print()

    if reranker:
        candidates = reranker.rerank(query, candidates, top_k)
        print("=" * 70)
        print(f"RERANKED CHUNKS (top {top_k})")
        print("=" * 70)
        for i, rc in enumerate(candidates, start=1):
            print(f"[{i}] score={rc.score:.4f} source={rc.chunk.source}")
            print(f"    {rc.chunk.text[:120]}...")
        print()

    result = generator.generate(query, candidates)

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result.answer)
    print()
    print("=" * 70)
    print("CITATIONS ACTUALLY USED IN THE ANSWER")
    print("=" * 70)
    if not result.citations:
        print("(none)")
    for rc in result.citations:
        print(f"- score={rc.score:.4f} source={rc.chunk.source}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python scripts/ask.py "your question here" [top_k]')
        sys.exit(1)
    question = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    main(question, k)
