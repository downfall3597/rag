import pytest

from rag_lab.models import Chunk, RetrievedChunk
from rag_lab.reranking.jina import JinaReranker


def _rc(chunk_id: str, text: str) -> RetrievedChunk:
    chunk = Chunk(id=chunk_id, document_id="doc", source="doc.md", text=text, position=0)
    return RetrievedChunk(chunk=chunk, score=0.0)


class FakeResponse:
    def __init__(self, json_data: dict):
        self._json_data = json_data
        self.raised = False

    def raise_for_status(self):
        self.raised = True

    def json(self):
        return self._json_data


class FakePostFn:
    """Records the request it received and returns a canned response."""

    def __init__(self, json_data: dict):
        self.json_data = json_data
        self.last_call: dict | None = None

    def __call__(self, url, headers=None, json=None):
        self.last_call = {"url": url, "headers": headers, "json": json}
        return FakeResponse(self.json_data)


def test_rerank_sends_query_and_documents_and_parses_results_in_returned_order():
    candidates = [_rc("a", "text a"), _rc("b", "text b"), _rc("c", "text c")]
    fake_post = FakePostFn(
        {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.4},
            ]
        }
    )
    reranker = JinaReranker(api_key="test-key", post_fn=fake_post)

    result = reranker.rerank("some query", candidates, top_k=2)

    # request shape
    assert fake_post.last_call["json"]["query"] == "some query"
    assert fake_post.last_call["json"]["documents"] == ["text a", "text b", "text c"]
    assert fake_post.last_call["json"]["top_n"] == 2
    assert fake_post.last_call["headers"]["Authorization"] == "Bearer test-key"

    # response mapped back to the right chunks, in the API's returned order
    assert [rc.chunk.id for rc in result] == ["b", "a"]
    assert result[0].score == pytest.approx(0.9)
    assert result[1].score == pytest.approx(0.4)


def test_rerank_with_no_candidates_skips_the_api_call_entirely():
    fake_post = FakePostFn({"results": []})
    reranker = JinaReranker(api_key="test-key", post_fn=fake_post)

    result = reranker.rerank("some query", [], top_k=4)

    assert result == []
    assert fake_post.last_call is None


def test_uses_explicit_api_key_over_environment(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "env-key")
    reranker = JinaReranker(api_key="explicit-key", post_fn=FakePostFn({"results": []}))
    assert reranker.api_key == "explicit-key"


def test_falls_back_to_environment_variable(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "env-key")
    reranker = JinaReranker(post_fn=FakePostFn({"results": []}))
    assert reranker.api_key == "env-key"
