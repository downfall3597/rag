"""Grounded generation with citations, using Claude via AWS Bedrock's Converse API.

Uses boto3's bedrock-runtime client and the Converse API -- AWS's unified
request/response shape across any Bedrock-hosted model (Claude, Llama,
Mistral, Titan, ...). Reuses build_prompt/extract_citations from
anthropic_gen.py unchanged: those are plain string/regex logic with no
dependency on which client sends the request, so only the call itself and
the response parsing differ from AnthropicGenerator.
"""

from typing import Any

import boto3

from rag_lab.generation.anthropic_gen import build_prompt, extract_citations
from rag_lab.generation.base import Generator
from rag_lab.models import GenerationResult, RetrievedChunk, TokenUsage

DEFAULT_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_REGION = "us-east-2"


class BedrockGenerator(Generator):
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        region: str = DEFAULT_REGION,
        max_tokens: int = 4096,
        client: Any | None = None,
    ):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self._client = client or boto3.Session(region_name=region).client("bedrock-runtime")

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        prompt = build_prompt(query, chunks)
        response = self._client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": self.max_tokens},
        )
        answer = response["output"]["message"]["content"][0]["text"]
        citations = extract_citations(answer, chunks)
        usage = TokenUsage(
            input_tokens=response["usage"]["inputTokens"],
            output_tokens=response["usage"]["outputTokens"],
        )
        return GenerationResult(query=query, answer=answer, citations=citations, usage=usage)
