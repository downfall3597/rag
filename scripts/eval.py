"""Run the eval question set (data/eval/questions.json) against a retrieval
configuration and report Recall@K, MRR, and nDCG@K.

Usage: python scripts/eval.py [top_k]

Retrieval-only -- no LLM calls, so this is free to run repeatedly. Uses the
same RAG_RETRIEVER / RAG_RERANKER env vars as scripts/ask.py (see that
file's docstring), so a reported score always reflects a configuration you
could actually run, e.g.:

    RAG_RETRIEVER=hybrid RAG_RERANKER=cross_encoder python scripts/eval.py

Results are also saved to data/eval/results/<config>.json so different
configurations can be compared later instead of just read off the terminal.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

from rag_lab.config import build_reranker, build_retriever
from rag_lab.embeddings.local import MiniLMEmbedder
from rag_lab.evaluation.metrics import mrr, ndcg_at_k, recall_at_k
from rag_lab.models import Chunk

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_FOLDER = REPO_ROOT / "data" / "index"
EVAL_QUESTIONS_PATH = REPO_ROOT / "data" / "eval" / "questions.json"
EVAL_RESULTS_FOLDER = REPO_ROOT / "data" / "eval" / "results"


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


def config_name() -> str:
    retriever_kind = os.environ.get("RAG_RETRIEVER", "dense").lower()
    reranker_kind = os.environ.get("RAG_RERANKER", "none").lower()
    return f"{retriever_kind}+{reranker_kind}"


def main(top_k: int) -> None:
    chunks, embeddings = load_index()
    questions = json.loads(EVAL_QUESTIONS_PATH.read_text())

    embedder = MiniLMEmbedder()
    retriever = build_retriever(chunks, embeddings, embedder)
    reranker = build_reranker()
    retrieve_k = top_k * 5 if reranker else top_k

    per_question = []
    for q in questions:
        candidates = retriever.retrieve(q["question"], top_k=retrieve_k)
        if reranker:
            candidates = reranker.rerank(q["question"], candidates, top_k)
        retrieved_ids = [rc.chunk.id for rc in candidates]
        relevant_ids = set(q["relevant_chunk_ids"])

        scores = {
            "id": q["id"],
            "question": q["question"],
            "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, top_k),
            "mrr": mrr(retrieved_ids, relevant_ids),
            "ndcg_at_k": ndcg_at_k(retrieved_ids, relevant_ids, top_k),
            "retrieved_ids": retrieved_ids,
            "relevant_ids": sorted(relevant_ids),
        }
        per_question.append(scores)

    def mean(key: str) -> float:
        return sum(s[key] for s in per_question) / len(per_question)

    summary = {
        "config": config_name(),
        "top_k": top_k,
        "num_questions": len(per_question),
        "mean_recall_at_k": mean("recall_at_k"),
        "mean_mrr": mean("mrr"),
        "mean_ndcg_at_k": mean("ndcg_at_k"),
    }

    print("=" * 70)
    print(f"EVAL: {summary['config']}  (top_k={top_k}, {summary['num_questions']} questions)")
    print("=" * 70)
    for s in per_question:
        print(f"[{s['id']}] recall@k={s['recall_at_k']:.2f}  mrr={s['mrr']:.2f}  ndcg@k={s['ndcg_at_k']:.2f}  {s['question']}")
    print()
    print("-" * 70)
    print(f"mean recall@{top_k}: {summary['mean_recall_at_k']:.4f}")
    print(f"mean MRR:       {summary['mean_mrr']:.4f}")
    print(f"mean nDCG@{top_k}:   {summary['mean_ndcg_at_k']:.4f}")

    EVAL_RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    results_path = EVAL_RESULTS_FOLDER / f"{summary['config'].replace('+', '_')}_k{top_k}.json"
    results_path.write_text(json.dumps({"summary": summary, "per_question": per_question}, indent=2))
    print()
    print(f"Saved to {results_path}")


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    main(k)
