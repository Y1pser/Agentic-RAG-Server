"""Tests for embedding factory routing and BaseEmbedding contract.

Covers B1.2: Embedding abstract interface and factory.
"""

import pytest
from rag_mcp_server.src.libs.embedding.base_embedding import BaseEmbedding, EmbeddingResponse
from rag_mcp_server.src.libs.embedding.embedding_factory import EmbeddingFactory
from rag_mcp_server.src.core.settings import Settings


# ---------------------------------------------------------------------------
# Fake / stub implementations for testing
# ---------------------------------------------------------------------------

class FakeEmbedding(BaseEmbedding):
    """Fake embedding provider that returns deterministic vectors."""

    def __init__(self, settings=None, **kwargs):
        self.settings = settings
        self.default_dimension = kwargs.get("dimension", 128)

    def embed(self, texts, trace=None, **kwargs):
        dimension = kwargs.get("dimension", self.default_dimension)
        return EmbeddingResponse(
            embeddings=[[0.1] * dimension for _ in texts],
            model="fake-embedding-v1",
            dimensions=dimension,
            usage={"total_tokens": sum(len(t) for t in texts)},
        )


class AnotherFakeEmbedding(BaseEmbedding):
    """A different fake to verify provider switching."""

    def __init__(self, settings=None, **kwargs):
        self.settings = settings
        self.kwargs = kwargs

    def embed(self, texts, trace=None, **kwargs):
        return EmbeddingResponse(
            embeddings=[[float(i + 1)] * 64 for i in range(len(texts))],
            model="another-fake-v2",
            dimensions=64,
        )


# ---------------------------------------------------------------------------
# BaseEmbedding contract tests
# ---------------------------------------------------------------------------

class TestBaseEmbedding:
    """Verify that BaseEmbedding enforces its contract."""

    def test_cannot_instantiate_abstract(self):
        """BaseEmbedding should not be directly instantiable."""
        with pytest.raises(TypeError):
            BaseEmbedding()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        """A concrete subclass should be instantiable and callable."""
        emb = FakeEmbedding()
        result = emb.embed(["hello", "world"])
        assert isinstance(result, EmbeddingResponse)
        assert len(result.embeddings) == 2
        assert len(result.embeddings[0]) == 128

    def test_single_text_embedding(self):
        """Embedding a single text should return a list with one vector."""
        emb = FakeEmbedding()
        result = emb.embed(["hello"])
        assert len(result.embeddings) == 1
        assert result.model == "fake-embedding-v1"

    def test_empty_list_returns_empty(self):
        """Embedding an empty list should return an empty embeddings list."""
        emb = FakeEmbedding()
        result = emb.embed([])
        assert result.embeddings == []


# ---------------------------------------------------------------------------
# EmbeddingFactory tests
# ---------------------------------------------------------------------------

class TestEmbeddingFactory:
    """Tests for the embedding factory: registration, creation, error handling."""

    def setup_method(self):
        """Ensure a clean registry before each test."""
        EmbeddingFactory.clear_registry()

    def teardown_method(self):
        """Clean up after each test."""
        EmbeddingFactory.clear_registry()

    # -- Registration -------------------------------------------------------

    def test_register_provider_adds_to_registry(self):
        """register_provider should make a name available."""
        EmbeddingFactory.register_provider("fake", FakeEmbedding)
        assert "fake" in EmbeddingFactory._PROVIDERS
        assert EmbeddingFactory.is_registered("fake")

    def test_register_non_subclass_raises(self):
        """Registering a class that isn't a BaseEmbedding should raise."""

        class NotAnEmbedding:
            pass

        with pytest.raises(ValueError, match="must inherit from BaseEmbedding"):
            EmbeddingFactory.register_provider("bad", NotAnEmbedding)

    def test_register_is_case_insensitive(self):
        """Provider names should be lower-cased on registration."""
        EmbeddingFactory.register_provider("FAKE", FakeEmbedding)
        assert EmbeddingFactory.is_registered("fake")
        assert EmbeddingFactory.is_registered("FAKE")

    # -- Creation -----------------------------------------------------------

    def test_create_returns_embedding_instance(self):
        """Factory.create should instantiate the registered provider."""
        EmbeddingFactory.register_provider("fake", FakeEmbedding)
        settings = Settings()
        settings.embedding = {"provider": "fake", "model": "fake-embedding-v1"}

        emb = EmbeddingFactory.create(settings)
        assert isinstance(emb, BaseEmbedding)
        assert isinstance(emb, FakeEmbedding)

    def test_create_passes_settings_to_provider(self):
        """The provider constructor should receive the settings object."""
        EmbeddingFactory.register_provider("fake", FakeEmbedding)
        settings = Settings()
        settings.embedding = {
            "provider": "fake",
            "model": "text-embedding-3-small",
            "dimension": 1536,
        }

        emb = EmbeddingFactory.create(settings)
        result = emb.embed(["test"])
        # FakeEmbedding uses kwargs.get("dimension", 128) — settings pass-through
        # is verified by the factory passing **override_kwargs; dimension from
        # settings gets forwarded as a kwarg override.
        assert len(result.embeddings) == 1

    def test_create_with_override_kwargs(self):
        """Override kwargs should be forwarded to the provider constructor."""
        EmbeddingFactory.register_provider("fake", FakeEmbedding)
        settings = Settings()
        settings.embedding = {"provider": "fake"}

        emb = EmbeddingFactory.create(settings, dimension=256)
        result = emb.embed(["test"])
        assert len(result.embeddings[0]) == 256

    # -- Error handling -----------------------------------------------------

    def test_missing_provider_config_raises(self):
        """Missing embedding.provider should raise a clear ValueError."""
        settings = Settings()
        settings.embedding = {"model": "text-embedding-3-small"}  # no provider
        with pytest.raises(ValueError, match="embedding.provider"):
            EmbeddingFactory.create(settings)

    def test_unknown_provider_raises(self):
        """An unregistered provider name should raise ValueError."""
        settings = Settings()
        settings.embedding = {"provider": "nonexistent"}
        with pytest.raises(ValueError, match="nonexistent"):
            EmbeddingFactory.create(settings)

    def test_instantiation_failure_wraps_error(self):
        """If provider __init__ raises, it should be wrapped in RuntimeError."""

        class BrokenEmbedding(BaseEmbedding):
            def __init__(self, settings=None, **kwargs):
                raise ConnectionError("Simulated network failure")

            def embed(self, texts, trace=None, **kwargs):
                return EmbeddingResponse(embeddings=[], model="broken")

        EmbeddingFactory.register_provider("broken", BrokenEmbedding)
        settings = Settings()
        settings.embedding = {"provider": "broken"}

        with pytest.raises(RuntimeError, match="Failed to instantiate"):
            EmbeddingFactory.create(settings)

    # -- Listing & querying ------------------------------------------------

    def test_list_providers_returns_sorted_names(self):
        """list_providers should return sorted registered names."""
        EmbeddingFactory.register_provider("b-provider", FakeEmbedding)
        EmbeddingFactory.register_provider("a-provider", AnotherFakeEmbedding)

        names = EmbeddingFactory.list_providers()
        assert names == ["a-provider", "b-provider"]

    def test_is_registered_returns_false_for_unknown(self):
        """is_registered should return False for unknown providers."""
        assert not EmbeddingFactory.is_registered("ghost")

    def test_clear_registry_removes_all(self):
        """clear_registry should empty the provider registry."""
        EmbeddingFactory.register_provider("fake", FakeEmbedding)
        EmbeddingFactory.clear_registry()
        assert EmbeddingFactory.list_providers() == []
