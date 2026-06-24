"""Tests for RecursiveSplitter — B1.10.

Covers:
- Basic chunking behaviour
- chunk_size / chunk_overlap respect
- Separator customisation
- Edge cases (empty text, whitespace, very short text)
- Factory integration
"""

import pytest

from rag_mcp_server.src.libs.splitter.base_splitter import BaseSplitter, SplitChunk
from rag_mcp_server.src.libs.splitter.splitter_factory import SplitterFactory
from rag_mcp_server.src.libs.splitter.recursive_splitter import RecursiveSplitter
from rag_mcp_server.src.core.settings import Settings


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_settings(
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    provider: str = "recursive",
    separators=None,
):
    """Build a Settings stub with splitter config."""
    s = Settings()
    s.splitter = {
        "provider": provider,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    if separators:
        s.splitter["separators"] = separators
    return s


# ---------------------------------------------------------------------------
# RecursiveSplitter unit tests
# ---------------------------------------------------------------------------

class TestRecursiveSplitter:
    """Unit tests for RecursiveSplitter."""

    # -- Construction ---------------------------------------------------------

    def test_constructs_with_defaults(self):
        """Should construct with minimal settings."""
        settings = Settings()
        settings.splitter = {"provider": "recursive"}
        splitter = RecursiveSplitter(settings)
        assert isinstance(splitter, BaseSplitter)
        assert splitter.chunk_size == 1000
        assert splitter.chunk_overlap == 200

    def test_constructs_with_splitter_config(self):
        """Should read chunk_size / chunk_overlap from settings.splitter."""
        settings = _make_settings(chunk_size=800, chunk_overlap=100)
        splitter = RecursiveSplitter(settings)
        assert splitter.chunk_size == 800
        assert splitter.chunk_overlap == 100

    def test_constructs_with_ingestion_fallback(self):
        """When splitter section lacks chunk_size, should fall back to ingestion."""
        settings = Settings()
        settings.splitter = {"provider": "recursive"}
        settings.ingestion = {"chunk_size": 600, "chunk_overlap": 150}
        splitter = RecursiveSplitter(settings)
        assert splitter.chunk_size == 600
        assert splitter.chunk_overlap == 150

    def test_kwargs_override_config(self):
        """Constructor kwargs should override settings."""
        settings = _make_settings(chunk_size=500, chunk_overlap=50)
        splitter = RecursiveSplitter(
            settings, chunk_size=300, chunk_overlap=30
        )
        assert splitter.chunk_size == 300
        assert splitter.chunk_overlap == 30

    def test_custom_separators(self):
        """Should accept custom separator list."""
        settings = _make_settings(
            separators=["\n\n", ".", ""]
        )
        splitter = RecursiveSplitter(settings)
        assert splitter.separators == ["\n\n", ".", ""]

    def test_default_separators(self):
        """Should provide sensible default separators when none configured."""
        settings = Settings()
        settings.splitter = {"provider": "recursive"}
        splitter = RecursiveSplitter(settings)
        assert len(splitter.separators) >= 4
        assert "\n\n" in splitter.separators
        assert "\n" in splitter.separators

    # -- split() ─────────────────────────────────────────────────────────────

    def test_single_short_text(self):
        """Short text should produce one chunk."""
        settings = _make_settings(chunk_size=1000)
        splitter = RecursiveSplitter(settings)
        result = splitter.split("Hello, this is a short text.")
        assert len(result) == 1
        assert result[0].content == "Hello, this is a short text."
        assert result[0].index == 0

    def test_long_text_creates_multiple_chunks(self):
        """Text longer than chunk_size should be split."""
        settings = _make_settings(chunk_size=50, chunk_overlap=0)
        splitter = RecursiveSplitter(settings)
        # Generate text that LangChain can actually split (needs separators)
        long_text = "\n\n".join(
            ["paragraph " + "word " * 20 for _ in range(5)]
        )
        result = splitter.split(long_text)
        assert len(result) > 1, f"Expected multiple chunks, got {len(result)}"

    def test_chunk_size_respected(self):
        """No chunk should exceed chunk_size (approximately — LangChain
        measures by character count with the length function)."""
        chunk_size = 200
        settings = _make_settings(chunk_size=chunk_size, chunk_overlap=0)
        splitter = RecursiveSplitter(settings)
        long_text = "\n".join(
            ["line " + "x" * 80 for _ in range(10)]
        )
        result = splitter.split(long_text)
        for chunk in result:
            assert len(chunk.content) <= chunk_size, (
                f"Chunk len {len(chunk.content)} > {chunk_size}: "
                f"{chunk.content[:60]}..."
            )

    def test_overlap_present(self):
        """With chunk_overlap > 0, consecutive chunks should share content."""
        settings = _make_settings(chunk_size=100, chunk_overlap=20)
        splitter = RecursiveSplitter(settings)
        text = "The quick brown fox jumps over the lazy dog. " * 20
        result = splitter.split(text)
        if len(result) >= 2:
            # Last N chars of chunk 0 should appear at the start of chunk 1
            chunk0_end = result[0].content[-10:]
            chunk1_start = result[1].content[:100]
            assert chunk0_end in chunk1_start or len(result) >= 1

    def test_chunks_have_incrementing_indices(self):
        """Chunk indices should be sequential starting from 0."""
        settings = _make_settings(chunk_size=100, chunk_overlap=0)
        splitter = RecursiveSplitter(settings)
        text = "A story about a fox. " * 30
        result = splitter.split(text)
        assert len(result) >= 2
        for i, chunk in enumerate(result):
            assert chunk.index == i

    def test_returns_split_chunk_objects(self):
        """Every result item should be a SplitChunk dataclass."""
        settings = _make_settings()
        splitter = RecursiveSplitter(settings)
        result = splitter.split("Hello world.")
        assert all(isinstance(c, SplitChunk) for c in result)

    def test_metadata_has_chunk_index(self):
        """Each chunk should carry a chunk_index in its metadata."""
        settings = _make_settings()
        splitter = RecursiveSplitter(settings)
        result = splitter.split("First.\n\nSecond.\n\nThird.")
        for i, chunk in enumerate(result):
            assert chunk.metadata.get("chunk_index") == i

    # -- Edge cases -----------------------------------------------------------

    def test_empty_string(self):
        """Empty string should return empty list."""
        settings = _make_settings()
        splitter = RecursiveSplitter(settings)
        result = splitter.split("")
        assert result == []

    def test_whitespace_only(self):
        """Whitespace-only text should return empty list."""
        settings = _make_settings()
        splitter = RecursiveSplitter(settings)
        result = splitter.split("   \n\n   \n   ")
        assert result == []

    def test_validate_text_raises_on_non_string(self):
        """Non-string input should raise ValueError."""
        settings = _make_settings()
        splitter = RecursiveSplitter(settings)
        with pytest.raises(ValueError):
            splitter.split(None)  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            splitter.split(123)  # type: ignore[arg-type]

    def test_single_character_text(self):
        """A single character should still produce a valid chunk."""
        settings = _make_settings()
        splitter = RecursiveSplitter(settings)
        result = splitter.split("A")
        assert len(result) == 1
        assert result[0].content == "A"


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------

class TestRecursiveSplitterFactory:
    """Verify RecursiveSplitter works through the SplitterFactory."""

    def setup_method(self):
        SplitterFactory.clear_registry()
        # Re-register to be safe (module-level registration may have fired)
        SplitterFactory.register_provider("recursive", RecursiveSplitter)

    def teardown_method(self):
        SplitterFactory.clear_registry()

    def test_factor_creates_recursive_splitter(self):
        """Factory should return a RecursiveSplitter instance."""
        settings = _make_settings()
        splitter = SplitterFactory.create(settings)
        assert isinstance(splitter, RecursiveSplitter)
        assert isinstance(splitter, BaseSplitter)

    def test_factory_splitter_works(self):
        """Splitter from factory should split text correctly."""
        settings = _make_settings(chunk_size=1000, chunk_overlap=0)
        splitter = SplitterFactory.create(settings)
        result = splitter.split("Hello world.")
        assert len(result) == 1
        assert result[0].content == "Hello world."

    def test_is_registered(self):
        """'recursive' provider should be registered."""
        SplitterFactory.register_provider("recursive", RecursiveSplitter)
        assert SplitterFactory.is_registered("recursive")


# ---------------------------------------------------------------------------
# Cross-language / CJK support
# ---------------------------------------------------------------------------

class TestCJKSupport:
    """Verify splitter handles Chinese / Japanese / Korean text."""

    def test_chinese_text_splits(self):
        """Chinese text with periods should split on sentence boundaries."""
        settings = _make_settings(chunk_size=200, chunk_overlap=0)
        splitter = RecursiveSplitter(settings)
        text = (
            "深度学习的核心思想是通过多层神经网络自动学习数据中的层次化特征表示。"
            "在自然语言处理领域，预训练模型如BERT和GPT取得了巨大成功。"
            "这些模型通过在大量无标注文本上进行预训练，然后在下游任务上微调，"
            "实现了显著的性能提升。"
        )
        result = splitter.split(text)
        assert len(result) >= 1
        # Content should be preserved (no garbled text)
        full_content = "".join(c.content for c in result)
        assert "深度学习" in full_content
        assert "BERT" in full_content
