from rag_lab.generation.bedrock_gen import BedrockGenerator
from rag_lab.models import Chunk, RetrievedChunk


def _rc(chunk_id: str, text: str, score: float = 0.9) -> RetrievedChunk:
    chunk = Chunk(id=chunk_id, document_id="doc", source="doc.md", text=text, position=0)
    return RetrievedChunk(chunk=chunk, score=score)


class FakeBedrockClient:
    """Stands in for boto3's bedrock-runtime client -- no network call."""

    def __init__(self, answer_text: str):
        self.answer_text = answer_text
        self.last_kwargs: dict | None = None

    def converse(self, **kwargs):
        self.last_kwargs = kwargs
        return {
            "output": {"message": {"content": [{"text": self.answer_text}]}},
            "usage": {"inputTokens": 42, "outputTokens": 7},
        }


def test_generate_sends_prompt_and_parses_response_with_citations():
    chunks = [_rc("a", "Vacation is 15 days per year."), _rc("b", "Sick leave is uncapped.")]
    fake_client = FakeBedrockClient(answer_text="You get 15 days per year [1].")

    generator = BedrockGenerator(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0", client=fake_client)
    result = generator.generate("How much vacation do I get?", chunks)

    assert result.answer == "You get 15 days per year [1]."
    assert [rc.chunk.id for rc in result.citations] == ["a"]

    # verify the Converse API request shape
    assert fake_client.last_kwargs["modelId"] == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    sent_messages = fake_client.last_kwargs["messages"]
    assert sent_messages[0]["role"] == "user"
    prompt_text = sent_messages[0]["content"][0]["text"]
    assert "How much vacation do I get?" in prompt_text
    assert "Vacation is 15 days per year." in prompt_text


def test_generate_with_no_citations_in_answer():
    chunks = [_rc("a", "fact A")]
    fake_client = FakeBedrockClient(answer_text="Insufficient information.")

    generator = BedrockGenerator(client=fake_client)
    result = generator.generate("unrelated question?", chunks)

    assert result.answer == "Insufficient information."
    assert result.citations == []
