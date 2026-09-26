import pytest

from rag_lab.ingestion.chunker import chunk_text


def words(text: str) -> list[str]:
    return text.split()


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size=10, overlap=2) == []
    assert chunk_text("   \n  ", chunk_size=10, overlap=2) == []


def test_text_shorter_than_chunk_size_returns_one_chunk():
    text = "one two three"
    result = chunk_text(text, chunk_size=10, overlap=2)
    assert result == ["one two three"]


def test_chunk_size_is_respected_except_for_last_chunk():
    text = " ".join(f"w{i}" for i in range(25))
    result = chunk_text(text, chunk_size=10, overlap=3)
    for chunk in result[:-1]:
        assert len(words(chunk)) == 10
    assert len(words(result[-1])) <= 10


def test_consecutive_chunks_overlap_by_the_requested_word_count():
    text = " ".join(f"w{i}" for i in range(25))
    result = chunk_text(text, chunk_size=10, overlap=3)
    for a, b in zip(result, result[1:]):
        assert words(a)[-3:] == words(b)[:3]


def test_last_chunk_is_kept_even_if_shorter_than_chunk_size():
    text = " ".join(f"w{i}" for i in range(23))
    result = chunk_text(text, chunk_size=10, overlap=2)
    assert words(result[-1]) == [f"w{i}" for i in range(16, 23)]


def test_no_words_are_skipped_or_duplicated_beyond_the_overlap():
    text = " ".join(f"w{i}" for i in range(25))
    result = chunk_text(text, chunk_size=10, overlap=3)
    rebuilt = words(result[0])
    for chunk in result[1:]:
        rebuilt.extend(words(chunk)[3:])
    assert rebuilt == [f"w{i}" for i in range(25)]


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("a b c d e f", chunk_size=5, overlap=5)
