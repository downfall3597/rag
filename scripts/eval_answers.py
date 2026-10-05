"""Run the eval question set through the FULL pipeline (retrieve -> rerank
-> generate) and judge each answer's faithfulness, relevance, and
correctness via an LLM judge.

Usage: python scripts/eval_answers.py [top_k]

Unlike scripts/eval.py (retrieval-only, free), this makes real paid LLM
calls -- one generation call AND one judge call per question. Uses the
same RAG_RETRIEVER / RAG_RERANKER / RAG_GENERATOR env vars as
scripts/ask.py. The judge defaults to whatever RAG_GENERATOR is set to
(so e.g. RAG_GENERATOR=bedrock also judges via Bedrock, needing no
ANTHROPIC_API_KEY at all) -- override independently with RAG_JUDGE_BACKEND,
RAG_JUDGE_MODEL (direct API) / RAG_JUDGE_MODEL_ID (Bedrock). Results are
saved to data/eval/results/answers_<config>.json.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from rag_lab.config import build_generator, build_judge, build_reranker, build_retriever
from rag_lab.embeddings.local import MiniLMEmbedder
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
    generator_kind = os.environ.get("RAG_GENERATOR", "anthropic").lower()
    return f"{retriever_kind}+{reranker_kind}+{generator_kind}"


def main(top_k: int) -> None:
    chunks, embeddings = load_index()
    questions = json.loads(EVAL_QUESTIONS_PATH.read_text())

    embedder = MiniLMEmbedder()
    retriever = build_retriever(chunks, embeddings, embedder)
    reranker = build_reranker()
    generator = build_generator()
    judge = build_judge()
    retrieve_k = top_k * 5 if reranker else top_k

    per_question = []
    for q in questions:
        candidates = retriever.retrieve(q["question"], top_k=retrieve_k)
        if reranker:
            candidates = reranker.rerank(q["question"], candidates, top_k)
        result = generator.generate(q["question"], candidates)
        verdict = judge.judge(q["question"], result.answer, q["reference_answer"], result.citations)

        print(f"[{q['id']}] faithfulness={verdict.faithfulness:.2f}  "
              f"relevance={verdict.answer_relevance:.2f}  correctness={verdict.correctness:.2f}")
        print(f"    Q: {q['question']}")
        print(f"    A: {result.answer}")
        print(f"    judge: {verdict.reasoning}")
        print()

        per_question.append({
            "id": q["id"],
            "question": q["question"],
            "answer": result.answer,
            "reference_answer": q["reference_answer"],
            "faithfulness": verdict.faithfulness,
            "answer_relevance": verdict.answer_relevance,
            "correctness": verdict.correctness,
            "reasoning": verdict.reasoning,
        })

    def mean(key: str) -> float:
        return sum(s[key] for s in per_question) / len(per_question)

    summary = {
        "config": config_name(),
        "top_k": top_k,
        "num_questions": len(per_question),
        "mean_faithfulness": mean("faithfulness"),
        "mean_answer_relevance": mean("answer_relevance"),
        "mean_correctness": mean("correctness"),
    }

    print("=" * 70)
    print(f"ANSWER EVAL: {summary['config']}  (top_k={top_k}, {summary['num_questions']} questions)")
    print("=" * 70)
    print(f"mean faithfulness:     {summary['mean_faithfulness']:.4f}")
    print(f"mean answer_relevance: {summary['mean_answer_relevance']:.4f}")
    print(f"mean correctness:      {summary['mean_correctness']:.4f}")

    EVAL_RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    results_path = EVAL_RESULTS_FOLDER / f"answers_{summary['config'].replace('+', '_')}_k{top_k}.json"
    results_path.write_text(json.dumps({"summary": summary, "per_question": per_question}, indent=2))
    print()
    print(f"Saved to {results_path}")


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    main(k)
