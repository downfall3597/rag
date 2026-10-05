import pytest
from pydantic import ValidationError

from rag_lab.evaluation.judge import build_judge_prompt, parse_judge_response
from rag_lab.models import Chunk, RetrievedChunk


def _rc(chunk_id: str, text: str) -> RetrievedChunk:
    chunk = Chunk(id=chunk_id, document_id="doc", source="doc.md", text=text, position=0)
    return RetrievedChunk(chunk=chunk, score=0.9)


# --- build_judge_prompt ---


def test_build_judge_prompt_includes_all_four_inputs():
    cited = [_rc("a", "Full-time employees accrue 15 days of paid vacation per year.")]
    prompt = build_judge_prompt(
        question="How much vacation do employees get?",
        answer="Employees get 15 days of vacation per year.",
        reference_answer="15 days of paid vacation per year.",
        cited_chunks=cited,
    )
    assert isinstance(prompt, str)
    assert "How much vacation do employees get?" in prompt
    assert "Employees get 15 days of vacation per year." in prompt
    assert "15 days of paid vacation per year." in prompt
    assert "Full-time employees accrue 15 days of paid vacation per year." in prompt


def test_build_judge_prompt_with_no_cited_chunks():
    prompt = build_judge_prompt(
        question="A question?",
        answer="An answer.",
        reference_answer="A reference.",
        cited_chunks=[],
    )
    assert isinstance(prompt, str)
    assert len(prompt) > 0


# --- parse_judge_response ---


def test_parse_judge_response_parses_plain_json():
    text = (
        '{"faithfulness": 0.9, "answer_relevance": 1.0, '
        '"correctness": 0.8, "reasoning": "Mostly grounded and on-topic."}'
    )
    result = parse_judge_response(text)
    assert result.faithfulness == pytest.approx(0.9)
    assert result.answer_relevance == pytest.approx(1.0)
    assert result.correctness == pytest.approx(0.8)
    assert result.reasoning == "Mostly grounded and on-topic."


def test_parse_judge_response_strips_markdown_code_fence():
    text = (
        "```json\n"
        '{"faithfulness": 0.5, "answer_relevance": 0.6, '
        '"correctness": 0.7, "reasoning": "Partially correct."}\n'
        "```"
    )
    result = parse_judge_response(text)
    assert result.faithfulness == pytest.approx(0.5)
    assert result.correctness == pytest.approx(0.7)


def test_parse_judge_response_rejects_out_of_range_scores():
    text = '{"faithfulness": 1.5, "answer_relevance": 1.0, "correctness": 1.0, "reasoning": "x"}'
    with pytest.raises(ValidationError):
        parse_judge_response(text)
