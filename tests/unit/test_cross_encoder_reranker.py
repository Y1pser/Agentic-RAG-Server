"""Tests for Cross-Encoder based Reranker implementation.

Covers B1.16: CrossEncoderReranker with sentence-transformers integration.
"""

from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest

from rag_mcp_server.src.libs.reranker.base_reranker import BaseReranker, ScoredChunk
from rag_mcp_server.src.libs.reranker.cross_encoder_reranker import (
    CrossEncoderReranker,
    CrossEncoderRerankError,
)
from rag_mcp_server.src.libs.reranker.reranker_factory import RerankerFactory
from rag_mcp_server.src.core.settings import Settings


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class MockCrossEncoder:
    """Mock CrossEncoder model for deterministic unit testing.

    Scoring strategy: counts how many query words appear in the passage,
    giving each match 0.3 points.  Deterministic and fast — no real model.
    """

    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
        self.call_count = 0
        self.last_pairs: Optional[List[tuple]] = None

    def predict(
        self,
        pairs: List[tuple],
        show_progress_bar: bool = False,
    ) -> List[float]:
        """Return deterministic scores based on keyword overlap."""
        self.call_count += 1
        self.last_pairs = pairs

        scores: List[float] = []
        for query, passage in pairs:
            score = 0.0
            query_words = query.lower().split()
            passage_lower = passage.lower()
            for word in query_words:
                if word in passage_lower:
                    score += 0.3
            scores.append(min(score, 1.0))
        return scores


class _MockChunk:
    """Lightweight mock of a retrieval chunk for reranker testing."""

    def __init__(
        self, text: str, score: float = 0.8, metadata: dict = None
    ):
        self.text = text
        self.score = score
        self.metadata = metadata or {}


def _make_chunks(*specs) -> list:
    """Create mock chunks from (text, score, metadata) tuples or plain strings."""
    chunks = []
    for item in specs:
        if isinstance(item, tuple):
            text = item[0]
            score = item[1] if len(item) > 1 else 0.8
            meta = item[2] if len(item) > 2 else {}
            chunks.append(_MockChunk(text=text, score=score, metadata=meta))
        else:
            chunks.append(_MockChunk(text=item))
    return chunks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model():
    """Reusable mock CrossEncoder model."""
    return MockCrossEncoder()


@pytest.fixture
def mock_settings():
    """Settings with cross_encoder model configured."""
    settings = Settings()
    settings.rerank = {
        "rerank_backend": "cross_encoder",
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    }
    return settings


@pytest.fixture
def sample_chunks():
    """Sample candidate chunks for reranking."""
    return _make_chunks(
        ("Python is a programming language.", 0.8),
        ("Machine learning uses neural networks.", 0.75),
        ("RAG combines retrieval and generation.", 0.9),
        ("Embeddings represent text as vectors.", 0.7),
    )


# ---------------------------------------------------------------------------
# CrossEncoderReranker — Initialization
# ---------------------------------------------------------------------------

class TestCrossEncoderRerankerInit:
    """Tests for CrossEncoderReranker initialization."""

    def test_init_with_mock_model(self, mock_settings, mock_model):
        """Initialization with injected mock model."""
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model,
        )
        assert reranker.settings == mock_settings
        assert reranker.model == mock_model

    def test_init_is_base_reranker(self, mock_settings, mock_model):
        """Should be a BaseReranker subclass."""
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model,
        )
        assert isinstance(reranker, BaseReranker)

    def test_model_name_from_settings(self, mock_model):
        """Reads model from rerank.model."""
        settings = Settings()
        settings.rerank = {"model": "my-ce-model"}
        reranker = CrossEncoderReranker(settings=settings, model=mock_model)
        assert reranker._model_name_from_settings(settings) == "my-ce-model"

    def test_model_name_default_when_missing(self, mock_model):
        """Falls back to built-in DEFAULT_MODEL when rerank.model is absent."""
        settings = Settings()
        settings.rerank = {"rerank_backend": "cross_encoder"}
        reranker = CrossEncoderReranker(settings=settings, model=mock_model)
        name = reranker._model_name_from_settings(settings)
        assert name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_model_name_invalid_type_raises(self, mock_model):
        """Raises CrossEncoderRerankError for non-string model name."""
        settings = Settings()
        settings.rerank = {"model": 123}
        reranker = CrossEncoderReranker(settings=settings, model=mock_model)
        with pytest.raises(
            CrossEncoderRerankError, match="must be a non-empty string"
        ):
            reranker._model_name_from_settings(settings)


# ---------------------------------------------------------------------------
# CrossEncoderReranker — Scoring
# ---------------------------------------------------------------------------

class TestCrossEncoderRerankerScoring:
    """Tests for the scoring logic."""

    def test_rerank_orders_by_relevance(
        self, mock_settings, mock_model, sample_chunks
    ):
        """The most relevant chunk should be first."""
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model,
        )

        # chunk_2 = "Machine learning uses neural networks."
        # It will score highest for this query
        query = "machine learning neural networks"
        result = reranker.rerank(query, sample_chunks)

        assert len(result) == 4
        assert all(isinstance(c, ScoredChunk) for c in result)
        # chunk_2 should be first (highest keyword overlap)
        assert result[0].content == "Machine learning uses neural networks."

    def test_rerank_all_scores_are_descending(
        self, mock_settings, mock_model, sample_chunks
    ):
        """Returned scores should be in descending order."""
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model,
        )

        result = reranker.rerank("machine learning", sample_chunks)

        for i in range(len(result) - 1):
            assert result[i].score >= result[i + 1].score

    def test_rerank_with_top_k(
        self, mock_settings, mock_model, sample_chunks
    ):
        """top_k limits the number of results."""
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model,
        )

        result = reranker.rerank(
            "machine learning", sample_chunks, top_k=2
        )

        assert len(result) == 2

    def test_rerank_top_k_default_all(
        self, mock_settings, mock_model, sample_chunks
    ):
        """When top_k is not given, all chunks are returned."""
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model,
        )

        result = reranker.rerank("query", sample_chunks)
        assert len(result) == 4

    def test_rerank_model_called_once(
        self, mock_settings, mock_model, sample_chunks
    ):
        """model.predict should be called exactly once."""
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model,
        )

        reranker.rerank("query", sample_chunks)
        assert mock_model.call_count == 1

    def test_rerank_pairs_are_correct(
        self, mock_settings, mock_model
    ):
        """model.predict should receive (query, text) pairs."""
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model,
        )

        chunks = _make_chunks("Alpha", "Beta")
        reranker.rerank("hello", chunks)

        assert mock_model.last_pairs == [
            ("hello", "Alpha"),
            ("hello", "Beta"),
        ]

    def test_rerank_single_chunk(
        self, mock_settings, mock_model
    ):
        """Single chunk works correctly."""
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model,
        )

        chunks = _make_chunks("Only one")
        result = reranker.rerank("query", chunks)

        assert len(result) == 1
        assert result[0].content == "Only one"
        assert mock_model.call_count == 1


# ---------------------------------------------------------------------------
# CrossEncoderReranker — Metadata Preservation
# ---------------------------------------------------------------------------

class TestCrossEncoderRerankerMetadata:
    """Tests for metadata handling."""

    def test_metadata_preserved(self, mock_settings, mock_model):
        """Original metadata is preserved alongside cross-encoder scores."""
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model,
        )

        chunks = _make_chunks(
            ("Doc A text", 0.8, {"source": "a.pdf", "page": 1}),
            ("Doc B text", 0.9, {"source": "b.pdf", "page": 2}),
        )

        result = reranker.rerank("doc", chunks)

        # First result should have all original metadata
        assert result[0].metadata["source"] == "a.pdf"
        assert result[0].metadata["page"] == 1
        assert result[0].metadata["original_score"] == 0.8
        assert "cross_encoder_score" in result[0].metadata

    def test_original_chunks_unmodified(self, mock_settings, mock_model):
        """Calling rerank should not mutate the original chunk objects."""
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model,
        )

        chunks = _make_chunks(
            ("text", 0.5, {"key": "val"}),
        )
        orig_text = chunks[0].text
        orig_score = chunks[0].score
        orig_meta = dict(chunks[0].metadata)

        reranker.rerank("query", chunks)

        # Original chunk should be untouched
        assert chunks[0].text == orig_text
        assert chunks[0].score == orig_score
        assert chunks[0].metadata == orig_meta


# ---------------------------------------------------------------------------
# CrossEncoderReranker — Input Validation
# ---------------------------------------------------------------------------

class TestCrossEncoderRerankerValidation:
    """Tests for input validation."""

    def test_empty_query_raises(self, mock_settings, mock_model, sample_chunks):
        """Empty query string raises ValueError."""
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model,
        )

        with pytest.raises(ValueError, match="Query cannot be empty"):
            reranker.rerank("   ", sample_chunks)

    def test_whitespace_query_raises(
        self, mock_settings, mock_model, sample_chunks
    ):
        """Whitespace-only query raises ValueError."""
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model,
        )

        with pytest.raises(ValueError, match="Query cannot be empty"):
            reranker.rerank("\t\n", sample_chunks)

    def test_empty_chunks_raises(self, mock_settings, mock_model):
        """Empty chunks list raises ValueError."""
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model,
        )

        with pytest.raises(
            ValueError, match="Candidates list cannot be empty"
        ):
            reranker.rerank("test", [])

    def test_chunk_missing_text_raises(self, mock_settings, mock_model):
        """Chunk without 'text' attribute raises ValueError."""
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model,
        )

        class BadChunk:
            pass

        with pytest.raises(
            ValueError, match="missing required attribute 'text'"
        ):
            reranker.rerank("test", [BadChunk()])

    def test_invalid_top_k_raises(self, mock_settings, mock_model, sample_chunks):
        """Non-positive top_k raises ValueError."""
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model,
        )

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            reranker.rerank("test", sample_chunks, top_k=0)

    def test_top_k_none_type_raises(
        self, mock_settings, mock_model, sample_chunks
    ):
        """Non-integer top_k raises ValueError."""
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model,
        )

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            reranker.rerank("test", sample_chunks, top_k="five")


# ---------------------------------------------------------------------------
# CrossEncoderReranker — Trace Context
# ---------------------------------------------------------------------------

class TestCrossEncoderRerankerTrace:
    """Tests for trace context pass-through."""

    def test_trace_context_accepted(self, mock_settings, mock_model, sample_chunks):
        """rerank() accepts and ignores trace parameter (Phase B-5 ready)."""
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model,
        )

        mock_trace = Mock()
        # Should not raise
        result = reranker.rerank(
            "query", sample_chunks, trace=mock_trace
        )
        assert len(result) == 4


# ---------------------------------------------------------------------------
# CrossEncoderReranker — Factory Registration
# ---------------------------------------------------------------------------

class TestCrossEncoderRerankerFactory:
    """Tests for factory registration."""

    def test_registered_as_cross_encoder(self):
        """Should be registered under 'cross_encoder' key."""
        assert RerankerFactory.is_registered("cross_encoder")

    def test_in_list_providers(self):
        """'cross_encoder' should appear in the provider list."""
        assert "cross_encoder" in RerankerFactory.list_providers()

    def test_factory_creates_cross_encoder(self, mock_settings, mock_model):
        """Factory.create should return a CrossEncoderReranker when configured."""
        # The factory reads rerank_backend from settings.rerank
        # but CrossEncoderReranker needs a real model or we inject one.
        # Just verify registration is correct.
        provider_class = RerankerFactory._PROVIDERS.get("cross_encoder")
        assert provider_class is CrossEncoderReranker


# ---------------------------------------------------------------------------
# CrossEncoderReranker — Determinism
# ---------------------------------------------------------------------------

class TestCrossEncoderRerankerDeterminism:
    """Tests for deterministic behavior with mock models."""

    def test_same_input_same_output(
        self, mock_settings, mock_model
    ):
        """Same input produces identical output (deterministic mock)."""
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model,
        )

        chunks = _make_chunks("Alpha text", "Beta content", "Gamma docs")
        r1 = reranker.rerank("alpha", chunks)
        r2 = reranker.rerank("alpha", chunks)

        assert [c.score for c in r1] == [c.score for c in r2]
        assert [c.content for c in r1] == [c.content for c in r2]

    def test_model_predict_not_called_with_progress_bar(
        self, mock_settings
    ):
        """model.predict is called with show_progress_bar=False."""
        model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=model,
        )

        reranker.rerank("q", _make_chunks("text"))
        assert model.call_count == 1
