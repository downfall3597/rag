"""Grounded generation with citations, using Claude via the Anthropic SDK.

The contract between the two functions below: chunks are numbered [1], [2], ...
in the same order as the `chunks` list passed in — chunks[0] is [1], chunks[1]
is [2], and so on. build_prompt presents them that way to the model;
extract_citations relies on that same numbering to map the model's citation
markers back to the actual RetrievedChunk objects.
"""

import re

import anthropic

from rag_lab.generation.base import Generator
from rag_lab.models import GenerationResult, RetrievedChunk

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """Build the prompt sent to the model.

    YOUR TURN: implement this.

    Requirements:
    - Present each chunk as a numbered source, e.g.:
        [1] <chunk 1's text>
        [2] <chunk 2's text>
      numbered in list order (chunks[0] is [1], chunks[1] is [2], ...).
    - Include the user's `query`.
    - Instruct the model to answer using ONLY the given sources, and to cite
      the source number(s) it used inline in its answer, e.g. "...15 days per
      year [1]." If the sources don't contain the answer, instruct it to say
      so rather than guessing.
    - Return the whole thing as a single string (this becomes the user
      message content sent to the model).

    Why this matters: "stuff the context in and hope" produces answers that
    sound confident but aren't traceable to a source — citations are what
    make an answer verifiable instead of just plausible.
    """
    return f"""
      You are given a query and supporting text that can be used to answer it. Provide answer to the given query.

      RULES:
      1. Strictly use only supporting text given to answer the query.
      2. If the text does not contain any useful information to answer the query, answer with "Insufficient information".
      3. If a you use a supporting text to answer something cite it. e.g. "...15 days per year [1]."
      4. Return answer in a single string.

      Query:{query}
      Supporting text:{[f'[{i+1}] {rc.chunk.text}\n' for i, rc in enumerate(chunks)]}
    """


def extract_citations(answer: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Find which chunks the model actually cited in `answer`.

    YOUR TURN: implement this.

    Requirements:
    - Find citation markers of the form [N] in `answer` (e.g. "[1]", "[2]").
    - Map each N back to chunks[N - 1] (1-indexed, per the numbering
      convention build_prompt uses).
    - Return only the chunks that were actually cited — not all of `chunks`.
    - No duplicates: if [1] appears three times in the answer, chunks[0]
      should appear once in the result.
    - Preserve the order the citations first appear in `answer`.
    - Ignore out-of-range markers gracefully (e.g. "[9]" when there are only
      3 chunks) rather than raising an error — a malformed model response
      shouldn't crash the pipeline.

    Hint: `re.findall(r"\\[(\\d+)\\]", answer)` gets you the marker numbers as
    strings, in order of appearance.
    """
    markers = re.findall(r"\[(\d+)\]", answer)
    seen_indices: set[int] = set()
    result: list[RetrievedChunk] = []
    for marker in markers:
        index = int(marker) - 1
        if index in seen_indices or not (0 <= index < len(chunks)):
            continue
        seen_indices.add(index)
        result.append(chunks[index])
    return result


class AnthropicGenerator(Generator):
    def __init__(self, model: str = DEFAULT_MODEL, client: anthropic.Anthropic | None = None):
        self._client = client or anthropic.Anthropic()
        self._model = model

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        prompt = build_prompt(query, chunks)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.content[0].text
        citations = extract_citations(answer, chunks)
        return GenerationResult(query=query, answer=answer, citations=citations)
