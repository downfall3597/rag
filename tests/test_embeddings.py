import numpy as np
import pytest
import torch

from rag_lab.embeddings.local import MiniLMEmbedder, l2_normalize, mean_pool

# --- Unit tests for mean_pool and l2_normalize: pure math, no model needed. ---


def test_mean_pool_ignores_padding_tokens():
    # batch of 1, seq_len 3, hidden_dim 2. Third token is padding (mask=0)
    # and has a huge value that must NOT influence the result.
    token_embeddings = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    attention_mask = torch.tensor([[1, 1, 0]])
    result = mean_pool(token_embeddings, attention_mask)
    assert result.shape == (1, 2)
    assert torch.allclose(result, torch.tensor([[2.0, 2.0]]), atol=1e-5)


def test_mean_pool_handles_batch_with_different_real_lengths():
    # sequence 0 has 2 real tokens, sequence 1 has 3 real tokens (no padding).
    token_embeddings = torch.tensor(
        [
            [[2.0, 0.0], [4.0, 0.0], [0.0, 0.0]],  # last token is padding
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],  # all real
        ]
    )
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    result = mean_pool(token_embeddings, attention_mask)
    expected = torch.tensor([[3.0, 0.0], [2.0, 2.0]])
    assert torch.allclose(result, expected, atol=1e-5)


def test_l2_normalize_produces_unit_length_rows():
    vectors = torch.tensor([[3.0, 4.0], [1.0, 0.0], [0.0, -2.0]])
    result = l2_normalize(vectors)
    norms = result.norm(dim=1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


def test_l2_normalize_preserves_direction():
    vectors = torch.tensor([[3.0, 4.0]])
    result = l2_normalize(vectors)
    assert torch.allclose(result, torch.tensor([[0.6, 0.8]]), atol=1e-5)


# --- Integration tests against the real MiniLM model. ---
# First run downloads the model (~90MB) from Hugging Face Hub.

@pytest.fixture(scope="module")
def embedder():
    return MiniLMEmbedder()


def test_embed_documents_returns_unit_vectors_of_expected_shape(embedder):
    vectors = embedder.embed_documents(["hello world", "a much longer sentence about vacation policy"])
    assert vectors.shape == (2, 384)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_embed_query_returns_single_unit_vector(embedder):
    vector = embedder.embed_query("how many vacation days do I get?")
    assert vector.shape == (384,)
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-4)


def test_similar_sentences_score_higher_than_dissimilar_ones(embedder):
    query = embedder.embed_query("How much paid vacation do employees get?")
    relevant = embedder.embed_documents(["Full-time employees accrue 15 days of paid vacation per year."])[0]
    irrelevant = embedder.embed_documents(["The office espresso machine is on the third floor."])[0]

    relevant_score = float(np.dot(query, relevant))
    irrelevant_score = float(np.dot(query, irrelevant))
    assert relevant_score > irrelevant_score
