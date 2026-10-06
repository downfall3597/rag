"""Benchmark latency, throughput, and (optionally) cost for a pipeline
configuration.

Usage: python scripts/benchmark.py

Env vars (same RAG_RETRIEVER / RAG_RERANKER / RAG_GENERATOR as ask.py):
  RAG_BENCHMARK_FULL=1          Also run generation (retrieve+rerank+generate
                                 end to end) instead of retrieval only.
                                 Makes REAL PAID LLM CALLS -- off by default.
  RAG_BENCHMARK_REQUESTS=20     Total number of requests to fire (cycles
                                 through the eval question set as needed).
  RAG_BENCHMARK_CONCURRENCY=4   How many requests to run at once.

Retrieval-only mode (the default) is free -- no LLM calls, so it's safe to
run repeatedly to compare configurations. Results are saved to
data/eval/results/benchmark_<config>.json.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from rag_lab.config import build_generator, build_reranker, build_retriever  # noqa: E402
from rag_lab.embeddings.local import MiniLMEmbedder  # noqa: E402
from rag_lab.evaluation.performance import cost_per_1000_queries, percentile  # noqa: E402
from rag_lab.models import Chunk  # noqa: E402

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


def config_name(full: bool) -> str:
    retriever_kind = os.environ.get("RAG_RETRIEVER", "dense").lower()
    reranker_kind = os.environ.get("RAG_RERANKER", "none").lower()
    suffix = f"+{os.environ.get('RAG_GENERATOR', 'anthropic').lower()}" if full else ""
    return f"{retriever_kind}+{reranker_kind}{suffix}"


def main() -> None:
    full = os.environ.get("RAG_BENCHMARK_FULL", "0") == "1"
    num_requests = int(os.environ.get("RAG_BENCHMARK_REQUESTS", "20"))
    concurrency = int(os.environ.get("RAG_BENCHMARK_CONCURRENCY", "4"))
    top_k = 4

    chunks, embeddings = load_index()
    questions = json.loads(EVAL_QUESTIONS_PATH.read_text())
    queries = [questions[i % len(questions)]["question"] for i in range(num_requests)]

    embedder = MiniLMEmbedder()
    retriever = build_retriever(chunks, embeddings, embedder)
    reranker = build_reranker()
    retrieve_k = top_k * 5 if reranker else top_k

    generator = build_generator() if full else None

    def run_one(query: str) -> dict:
        start = time.perf_counter()
        candidates = retriever.retrieve(query, top_k=retrieve_k)
        if reranker:
            candidates = reranker.rerank(query, candidates, top_k)
        input_tokens = output_tokens = 0
        if full:
            result = generator.generate(query, candidates)
            if result.usage:
                input_tokens = result.usage.input_tokens
                output_tokens = result.usage.output_tokens
        latency = time.perf_counter() - start
        return {"latency": latency, "input_tokens": input_tokens, "output_tokens": output_tokens}

    print(f"Running {num_requests} requests at concurrency {concurrency}"
          f" ({'full pipeline, REAL LLM CALLS' if full else 'retrieval-only, free'})...")

    batch_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(run_one, queries))
    total_wall_time = time.perf_counter() - batch_start

    latencies = [r["latency"] for r in results]
    throughput_qps = num_requests / total_wall_time

    summary = {
        "config": config_name(full),
        "full_pipeline": full,
        "num_requests": num_requests,
        "concurrency": concurrency,
        "total_wall_time_s": total_wall_time,
        "throughput_qps": throughput_qps,
        "p50_latency_s": percentile(latencies, 50),
        "p95_latency_s": percentile(latencies, 95),
        "mean_latency_s": sum(latencies) / len(latencies),
    }

    if full:
        mean_input = sum(r["input_tokens"] for r in results) / len(results)
        mean_output = sum(r["output_tokens"] for r in results) / len(results)
        model = os.environ.get("RAG_BEDROCK_MODEL_ID" if os.environ.get("RAG_GENERATOR") == "bedrock" else "RAG_MODEL", "claude-haiku-4-5")
        # Pricing is keyed by the direct-API model name regardless of backend
        # (see performance.py's module docstring on Bedrock pricing being an
        # approximation) -- normalize a Bedrock inference-profile id down to
        # its bare model name for the pricing lookup.
        pricing_key = "claude-haiku-4-5" if "haiku-4-5" in model else ("claude-sonnet-5" if "sonnet-5" in model else model)
        summary["mean_input_tokens"] = mean_input
        summary["mean_output_tokens"] = mean_output
        summary["cost_per_1000_queries_usd"] = cost_per_1000_queries(mean_input, mean_output, pricing_key)

    print("=" * 70)
    print(f"BENCHMARK: {summary['config']}")
    print("=" * 70)
    print(f"requests={num_requests}  concurrency={concurrency}  wall_time={total_wall_time:.2f}s")
    print(f"throughput: {throughput_qps:.2f} queries/sec")
    print(f"latency  mean={summary['mean_latency_s']:.3f}s  p50={summary['p50_latency_s']:.3f}s  p95={summary['p95_latency_s']:.3f}s")
    if full:
        print(f"tokens   mean_input={summary['mean_input_tokens']:.1f}  mean_output={summary['mean_output_tokens']:.1f}")
        print(f"cost     ${summary['cost_per_1000_queries_usd']:.4f} per 1,000 queries (approx, see performance.py docstring)")

    EVAL_RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    results_path = EVAL_RESULTS_FOLDER / f"benchmark_{summary['config'].replace('+', '_')}.json"
    results_path.write_text(json.dumps(summary, indent=2))
    print()
    print(f"Saved to {results_path}")


if __name__ == "__main__":
    main()
