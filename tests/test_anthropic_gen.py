from rag_lab.generation.anthropic_gen import build_prompt, extract_citations
from rag_lab.models import Chunk, RetrievedChunk


def _rc(chunk_id: str, text: str, score: float = 0.9) -> RetrievedChunk:
    chunk = Chunk(id=chunk_id, document_id="doc", source="doc.md", text=text, position=0)
    return RetrievedChunk(chunk=chunk, score=score)


# --- build_prompt ---


def test_build_prompt_includes_query_and_numbered_sources():
    chunks = [_rc("a", "Vacation is 15 days per year."), _rc("b", "Sick leave is uncapped.")]
    prompt = build_prompt("How much vacation do I get?", chunks)

    assert "How much vacation do I get?" in prompt
    assert "[1]" in prompt
    assert "Vacation is 15 days per year." in prompt
    assert "[2]" in prompt
    assert "Sick leave is uncapped." in prompt
    # source 1's text should appear before source 2's, matching chunk order
    assert prompt.index("Vacation is 15 days per year.") < prompt.index("Sick leave is uncapped.")


def test_build_prompt_returns_a_string():
    chunks = [_rc("a", "Some fact.")]
    prompt = build_prompt("A question?", chunks)
    assert isinstance(prompt, str)
    assert len(prompt) > 0


# --- extract_citations ---


def test_extract_citations_returns_only_cited_chunks_in_order():
    chunks = [_rc("a", "fact A"), _rc("b", "fact B"), _rc("c", "fact C")]
    answer = "Fact C is true [3]. Also fact A holds [1]."
    result = extract_citations(answer, chunks)
    assert [rc.chunk.id for rc in result] == ["c", "a"]


def test_extract_citations_skips_uncited_chunks():
    chunks = [_rc("a", "fact A"), _rc("b", "fact B"), _rc("c", "fact C")]
    answer = "Only fact B is relevant [2]."
    result = extract_citations(answer, chunks)
    assert [rc.chunk.id for rc in result] == ["b"]


def test_extract_citations_deduplicates_repeated_markers():
    chunks = [_rc("a", "fact A"), _rc("b", "fact B")]
    answer = "Fact A [1] is true. Fact A [1] again. Fact B [2] too."
    result = extract_citations(answer, chunks)
    assert [rc.chunk.id for rc in result] == ["a", "b"]


def test_extract_citations_ignores_out_of_range_markers():
    chunks = [_rc("a", "fact A")]
    answer = "This cites a real source [1] and a bogus one [9]."
    result = extract_citations(answer, chunks)
    assert [rc.chunk.id for rc in result] == ["a"]


def test_extract_citations_returns_empty_list_when_nothing_cited():
    chunks = [_rc("a", "fact A")]
    answer = "This answer cites nothing at all."
    result = extract_citations(answer, chunks)
    assert result == []
