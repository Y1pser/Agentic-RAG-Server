"""Tests for splitter factory routing and BaseSplitter contract.

Covers B1.3: Splitter abstract interface and factory.
"""

import pytest
from rag_mcp_server.src.libs.splitter.base_splitter import BaseSplitter, SplitChunk
from rag_mcp_server.src.libs.splitter.splitter_factory import SplitterFactory
from rag_mcp_server.src.core.settings import Settings


# ---------------------------------------------------------------------------
# Fake / stub implementations for testing
# ---------------------------------------------------------------------------

class FakeSplitter(BaseSplitter):
    """Fake splitter that splits text by double newlines (paragraphs)."""

    def __init__(self, settings=None, **kwargs):
        self.settings = settings
        self.chunk_size = kwargs.get("chunk_size", 500)

    def split(self, text, trace=None, **kwargs):
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        for i, para in enumerate(paragraphs):
            chunks.append(SplitChunk(
                content=para[:self.chunk_size],
                index=i,
                metadata={"source": "fake", "paragraph": i},
            ))
        return chunks


class AnotherFakeSplitter(BaseSplitter):
    """A different fake to verify provider switching — splits by sentences."""

    def __init__(self, settings=None, **kwargs):
        self.settings = settings

    def split(self, text, trace=None, **kwargs):
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        return [
            SplitChunk(content=s, index=i, metadata={"type": "sentence"})
            for i, s in enumerate(sentences)
        ]


# ---------------------------------------------------------------------------
# BaseSplitter contract tests
# ---------------------------------------------------------------------------

class TestBaseSplitter:
    """Verify that BaseSplitter enforces its contract."""

    def test_cannot_instantiate_abstract(self):
        """BaseSplitter should not be directly instantiable."""
        with pytest.raises(TypeError):
            BaseSplitter()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        """A concrete subclass should be instantiable and callable."""
        splitter = FakeSplitter()
        result = splitter.split("Paragraph one.\n\nParagraph two.")
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(c, SplitChunk) for c in result)
        assert result[0].content == "Paragraph one."
        assert result[1].index == 1

    def test_single_paragraph_returns_one_chunk(self):
        """Text without paragraph breaks should return a single chunk."""
        splitter = FakeSplitter()
        result = splitter.split("Just one paragraph.")
        assert len(result) == 1
        assert result[0].metadata["paragraph"] == 0

    def test_empty_text_returns_empty_list(self):
        """Empty text should return an empty list."""
        splitter = FakeSplitter()
        result = splitter.split("")
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        """Whitespace-only text should return empty list."""
        splitter = FakeSplitter()
        result = splitter.split("   \n\n   ")
        assert result == []

    def test_chunk_size_respected(self):
        """Chunks should not exceed the configured chunk size."""
        splitter = FakeSplitter(chunk_size=10)
        long_text = "A" * 100
        result = splitter.split(long_text)
        for chunk in result:
            assert len(chunk.content) <= 10


# ---------------------------------------------------------------------------
# SplitChunk tests
# ---------------------------------------------------------------------------

class TestSplitChunk:
    """Tests for the SplitChunk dataclass."""

    def test_default_metadata_is_empty_dict(self):
        """SplitChunk metadata should default to an empty dict."""
        chunk = SplitChunk(content="test", index=0)
        assert chunk.metadata == {}

    def test_fields_accessible(self):
        """All fields should be accessible as attributes."""
        chunk = SplitChunk(content="hello", index=3, metadata={"page": 1})
        assert chunk.content == "hello"
        assert chunk.index == 3
        assert chunk.metadata == {"page": 1}


# ---------------------------------------------------------------------------
# SplitterFactory tests
# ---------------------------------------------------------------------------

class TestSplitterFactory:
    """Tests for the splitter factory: registration, creation, error handling."""

    def setup_method(self):
        """Ensure a clean registry before each test."""
        SplitterFactory.clear_registry()

    def teardown_method(self):
        """Clean up after each test."""
        SplitterFactory.clear_registry()

    # -- Registration -------------------------------------------------------

    def test_register_provider_adds_to_registry(self):
        """register_provider should make a name available."""
        SplitterFactory.register_provider("fake", FakeSplitter)
        assert "fake" in SplitterFactory._PROVIDERS
        assert SplitterFactory.is_registered("fake")

    def test_register_non_subclass_raises(self):
        """Registering a non-BaseSplitter class should raise ValueError."""

        class NotASplitter:
            pass

        with pytest.raises(ValueError, match="must inherit from BaseSplitter"):
            SplitterFactory.register_provider("bad", NotASplitter)

    def test_register_is_case_insensitive(self):
        """Provider names should be lower-cased."""
        SplitterFactory.register_provider("FAKE", FakeSplitter)
        assert SplitterFactory.is_registered("fake")
        assert SplitterFactory.is_registered("FAKE")

    # -- Creation -----------------------------------------------------------

    def test_create_returns_splitter_instance(self):
        """Factory.create should instantiate the registered provider."""
        SplitterFactory.register_provider("fake", FakeSplitter)
        settings = Settings()
        settings.splitter = {"provider": "fake"}

        splitter = SplitterFactory.create(settings)
        assert isinstance(splitter, BaseSplitter)
        assert isinstance(splitter, FakeSplitter)

    def test_create_passes_settings_to_provider(self):
        """The provider constructor should receive settings."""
        SplitterFactory.register_provider("fake", FakeSplitter)
        settings = Settings()
        settings.splitter = {"provider": "fake", "chunk_size": 1000}

        splitter = SplitterFactory.create(settings)
        result = splitter.split("Hello\n\nWorld")
        assert len(result) == 2

    def test_create_with_override_kwargs(self):
        """Override kwargs should be forwarded to provider constructor."""
        SplitterFactory.register_provider("fake", FakeSplitter)
        settings = Settings()
        settings.splitter = {"provider": "fake"}

        splitter = SplitterFactory.create(settings, chunk_size=5)
        result = splitter.split("HelloWorldLongText")
        for chunk in result:
            assert len(chunk.content) <= 5

    # -- Error handling -----------------------------------------------------

    def test_missing_provider_config_raises(self):
        """Missing splitter config should raise ValueError."""
        settings = Settings()
        with pytest.raises(ValueError, match="splitter.provider"):
            SplitterFactory.create(settings)

    def test_unknown_provider_raises(self):
        """An unregistered provider name should raise."""
        settings = Settings()
        settings.splitter = {"provider": "nonexistent"}
        with pytest.raises(ValueError, match="nonexistent"):
            SplitterFactory.create(settings)

    def test_instantiation_failure_wraps_error(self):
        """Init failures should wrap in RuntimeError."""

        class BrokenSplitter(BaseSplitter):
            def __init__(self, settings=None, **kwargs):
                raise ConnectionError("Simulated failure")

            def split(self, text, trace=None, **kwargs):
                return []

        SplitterFactory.register_provider("broken", BrokenSplitter)
        settings = Settings()
        settings.splitter = {"provider": "broken"}

        with pytest.raises(RuntimeError, match="Failed to instantiate"):
            SplitterFactory.create(settings)

    # -- Listing & querying ------------------------------------------------

    def test_list_providers_returns_sorted_names(self):
        """list_providers should return sorted names."""
        SplitterFactory.register_provider("b-splitter", FakeSplitter)
        SplitterFactory.register_provider("a-splitter", AnotherFakeSplitter)

        names = SplitterFactory.list_providers()
        assert names == ["a-splitter", "b-splitter"]

    def test_is_registered_returns_false_for_unknown(self):
        """is_registered should be False for unknowns."""
        assert not SplitterFactory.is_registered("ghost")

    def test_clear_registry_removes_all(self):
        """clear_registry should empty the registry."""
        SplitterFactory.register_provider("fake", FakeSplitter)
        SplitterFactory.clear_registry()
        assert SplitterFactory.list_providers() == []


# ---------------------------------------------------------------------------
# Settings integration
# ---------------------------------------------------------------------------

class TestSplitterSettingsIntegration:
    """Integration of splitter with Settings."""

    def test_settings_has_splitter_field(self):
        """Settings should expose a splitter config section."""
        settings = Settings()
        assert hasattr(settings, "splitter")
        assert isinstance(settings.splitter, dict)
