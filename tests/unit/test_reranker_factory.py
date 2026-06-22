"""Tests for reranker factory routing and BaseReranker contract.

Covers B1.5: Reranker abstract interface and factory (with NoneReranker fallback).
"""

import pytest
from rag_mcp_server.src.libs.reranker.base_reranker import (
    BaseReranker,
    ScoredChunk,
    NoneReranker,
)
from rag_mcp_server.src.libs.reranker.reranker_factory import RerankerFactory
from rag_mcp_server.src.core.settings import Settings


# ---------------------------------------------------------------------------
# Fake / stub implementations for testing
# ---------------------------------------------------------------------------

class FakeReranker(BaseReranker):
    """Fake reranker that promotes chunks with 'important' in their text."""

    def __init__(self, settings=None, **kwargs):
        self.settings = settings
        self.boost = kwargs.get("boost", 0.2)

    def rerank(self, query, chunks, trace=None, **kwargs):
        reranked = []
        for i, chunk in enumerate(chunks):
            boost = self.boost if "important" in chunk.text.lower() else 0.0
            reranked.append(ScoredChunk(
                content=chunk.text,
                score=chunk.score + boost,
                metadata=chunk.metadata,
            ))
        reranked.sort(key=lambda c: c.score, reverse=True)
        return reranked


class AnotherFakeReranker(BaseReranker):
    """A different fake — reverses order."""

    def __init__(self, settings=None, **kwargs):
        self.settings = settings

    def rerank(self, query, chunks, trace=None, **kwargs):
        reversed_chunks = []
        for i, chunk in enumerate(reversed(chunks)):
            reversed_chunks.append(ScoredChunk(
                content=chunk.text,
                score=float(len(chunks) - i),
                metadata=chunk.metadata,
            ))
        return reversed_chunks


# ---------------------------------------------------------------------------
# Helper — create mock chunks
# ---------------------------------------------------------------------------

class _MockChunk:
    """Lightweight mock of a retrieval chunk for reranker testing."""
    def __init__(self, text: str, score: float = 0.8, metadata: dict = None):
        self.text = text
        self.score = score
        self.metadata = metadata or {}


def _make_chunks(*args) -> list:
    """Create mock chunks from (text, score) pairs or strings."""
    chunks = []
    for item in args:
        if isinstance(item, tuple):
            chunks.append(_MockChunk(text=item[0], score=item[1]))
        else:
            chunks.append(_MockChunk(text=item))
    return chunks


# ---------------------------------------------------------------------------
# ScoredChunk tests
# ---------------------------------------------------------------------------

class TestScoredChunk:
    """Tests for ScoredChunk dataclass."""

    def test_defaults(self):
        """Default metadata is empty dict."""
        sc = ScoredChunk(content="test", score=0.95)
        assert sc.metadata == {}

    def test_fields(self):
        """All fields are accessible."""
        sc = ScoredChunk(content="hello", score=0.87, metadata={"source": "a.pdf"})
        assert sc.content == "hello"
        assert sc.score == 0.87
        assert sc.metadata["source"] == "a.pdf"


# ---------------------------------------------------------------------------
# BaseReranker contract tests
# ---------------------------------------------------------------------------

class TestBaseReranker:
    """Verify BaseReranker enforces its contract."""

    def test_cannot_instantiate_abstract(self):
        """Should not be directly instantiable."""
        with pytest.raises(TypeError):
            BaseReranker()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        """A concrete subclass should work."""
        reranker = FakeReranker()
        chunks = _make_chunks(("normal text", 0.5), ("very important text", 0.6))
        result = reranker.rerank("query", chunks)
        assert len(result) == 2
        assert all(isinstance(c, ScoredChunk) for c in result)
        # The "important" chunk should now be first due to boost
        assert "important" in result[0].content


# ---------------------------------------------------------------------------
# NoneReranker tests (B1.5 special feature)
# ---------------------------------------------------------------------------

class TestNoneReranker:
    """Tests for the NoneReranker pass-through implementation."""

    def test_is_base_reranker(self):
        """NoneReranker should be a BaseReranker subclass."""
        assert issubclass(NoneReranker, BaseReranker)

    def test_pass_through_preserves_chunks(self):
        """Should return chunks unmodified with their original scores."""
        reranker = NoneReranker()
        chunks = _make_chunks(("text A", 0.9), ("text B", 0.7), ("text C", 0.5))
        result = reranker.rerank("query", chunks)

        assert len(result) == 3
        for i, chunk in enumerate(result):
            assert chunk.content == chunks[i].text
            assert chunk.score == chunks[i].score

    def test_empty_list_returns_empty(self):
        """Should handle empty chunk list."""
        reranker = NoneReranker()
        result = reranker.rerank("query", [])
        assert result == []

    def test_scores_are_preserved(self):
        """Scores should not be modified."""
        reranker = NoneReranker()
        chunks = _make_chunks(("text", 0.42))
        result = reranker.rerank("query", chunks)
        assert result[0].score == 0.42


# ---------------------------------------------------------------------------
# RerankerFactory tests
# ---------------------------------------------------------------------------

class TestRerankerFactory:
    """Tests for reranker factory."""

    def setup_method(self):
        RerankerFactory.clear_registry()

    def teardown_method(self):
        RerankerFactory.clear_registry()

    # -- Registration -------------------------------------------------------

    def test_register_provider_adds_to_registry(self):
        RerankerFactory.register_provider("fake", FakeReranker)
        assert "fake" in RerankerFactory._PROVIDERS

    def test_register_non_subclass_raises(self):
        class NotAReranker:
            pass
        with pytest.raises(ValueError, match="must inherit from BaseReranker"):
            RerankerFactory.register_provider("bad", NotAReranker)

    def test_register_is_case_insensitive(self):
        RerankerFactory.register_provider("FAKE", FakeReranker)
        assert RerankerFactory.is_registered("fake")

    # -- Creation -----------------------------------------------------------

    def test_create_returns_reranker_instance(self):
        RerankerFactory.register_provider("fake", FakeReranker)
        settings = Settings()
        settings.rerank = {"rerank_backend": "fake"}

        reranker = RerankerFactory.create(settings)
        assert isinstance(reranker, BaseReranker)

    def test_create_with_none_backend_returns_nonereanker(self):
        """When rerank_backend is 'none', should return NoneReranker."""
        settings = Settings()
        settings.rerank = {"rerank_backend": "none"}

        reranker = RerankerFactory.create(settings)
        assert isinstance(reranker, NoneReranker)

    def test_create_none_reranker_preserves_chunks(self):
        """The NoneReranker from factory should work correctly."""
        settings = Settings()
        settings.rerank = {"rerank_backend": "none"}

        reranker = RerankerFactory.create(settings)
        chunks = _make_chunks(("test", 0.75))
        result = reranker.rerank("q", chunks)
        assert result[0].score == 0.75

    # -- Error handling -----------------------------------------------------

    def test_unknown_provider_raises(self):
        settings = Settings()
        settings.rerank = {"rerank_backend": "unknown"}
        with pytest.raises(ValueError, match="unknown"):
            RerankerFactory.create(settings)

    def test_instantiation_failure_wraps_error(self):
        class BrokenReranker(BaseReranker):
            def __init__(self, settings=None, **kwargs):
                raise RuntimeError("GPU not available")
            def rerank(self, query, chunks, trace=None, **kwargs):
                return []

        RerankerFactory.register_provider("broken", BrokenReranker)
        settings = Settings()
        settings.rerank = {"rerank_backend": "broken"}

        with pytest.raises(RuntimeError, match="Failed to instantiate"):
            RerankerFactory.create(settings)

    # -- Listing & querying ------------------------------------------------

    def test_list_providers_includes_registered(self):
        RerankerFactory.register_provider("b", FakeReranker)
        RerankerFactory.register_provider("a", AnotherFakeReranker)
        assert RerankerFactory.list_providers() == ["a", "b"]

    def test_is_registered_false_for_unknown(self):
        assert not RerankerFactory.is_registered("ghost")

    def test_clear_registry(self):
        RerankerFactory.register_provider("fake", FakeReranker)
        RerankerFactory.clear_registry()
        assert RerankerFactory.list_providers() == []


# ---------------------------------------------------------------------------
# Settings integration
# ---------------------------------------------------------------------------

class TestRerankerSettingsIntegration:
    def test_settings_has_rerank_field(self):
        settings = Settings()
        assert hasattr(settings, "rerank")
        assert isinstance(settings.rerank, dict)
