"""Tests for Vision LLM factory routing and BaseVisionLLM contract.

Covers B1.13: Vision LLM abstract interface and factory integration.
"""

import base64
import re
from pathlib import Path

import pytest
from rag_mcp_server.src.libs.vision.base_vision_llm import BaseVisionLLM
from rag_mcp_server.src.libs.vision.vision_factory import VisionLLMFactory
from rag_mcp_server.src.core.settings import Settings


# ---------------------------------------------------------------------------
# Fake / stub implementations for testing
# ---------------------------------------------------------------------------

class FakeVisionLLM(BaseVisionLLM):
    """Fake Vision LLM that returns a fixed description."""

    def __init__(self, settings=None, **kwargs):
        self.settings = settings
        self.prefix = kwargs.get("prefix", "Image shows")

    def describe(self, image, trace=None, **kwargs):
        data = self._resolve_image_bytes(image)
        size = len(data)
        detail = kwargs.get("detail", "auto")
        return f"{self.prefix} an image of {size} bytes (detail={detail})"


class AnotherFakeVisionLLM(BaseVisionLLM):
    """A different fake — always returns the same caption."""

    def __init__(self, settings=None, **kwargs):
        self.settings = settings

    def describe(self, image, trace=None, **kwargs):
        self._resolve_image_bytes(image)
        return "A photograph of something interesting"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png_bytes(size: int = 100) -> bytes:
    """Create minimal fake PNG bytes for testing."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * max(0, size - 8)


# ---------------------------------------------------------------------------
# BaseVisionLLM contract tests
# ---------------------------------------------------------------------------

class TestBaseVisionLLM:
    """Verify BaseVisionLLM enforces its contract."""

    def test_cannot_instantiate_abstract(self):
        """Should not be directly instantiable."""
        with pytest.raises(TypeError):
            BaseVisionLLM()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        """A concrete subclass should work."""
        vision = FakeVisionLLM()
        result = vision.describe(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x00")
        assert isinstance(result, str)
        assert "Image shows" in result

    def test_describe_returns_string(self):
        """describe() must return a string."""
        vision = FakeVisionLLM()
        result = vision.describe(b"fake-image-data")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _resolve_image_bytes tests
# ---------------------------------------------------------------------------

class TestResolveImageBytes:
    """Tests for the _resolve_image_bytes helper on BaseVisionLLM."""

    def test_bytes_passthrough(self):
        """Bytes input should pass through unchanged."""
        vision = FakeVisionLLM()
        data = b"hello image bytes"
        result = vision._resolve_image_bytes(data)
        assert result == data

    def test_bytes_empty_raises(self):
        """Empty bytes should raise ValueError."""
        vision = FakeVisionLLM()
        with pytest.raises(ValueError, match="cannot be empty"):
            vision._resolve_image_bytes(b"")

    def test_path_resolves(self, tmp_path):
        """Path input should read file contents."""
        vision = FakeVisionLLM()
        img = tmp_path / "test.png"
        img.write_bytes(b"path image data")
        result = vision._resolve_image_bytes(img)
        assert result == b"path image data"

    def test_str_path_resolves(self, tmp_path):
        """String path input should read file contents."""
        vision = FakeVisionLLM()
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"file bytes here")
        result = vision._resolve_image_bytes(str(img))
        assert result == b"file bytes here"

    def test_path_not_found_raises(self, tmp_path):
        """Non-existent path should raise FileNotFoundError."""
        vision = FakeVisionLLM()
        missing = tmp_path / "does_not_exist.png"
        with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
            vision._resolve_image_bytes(missing)

    def test_str_path_not_found_raises(self, tmp_path):
        """Non-existent string path should raise FileNotFoundError."""
        vision = FakeVisionLLM()
        missing = tmp_path / "nope.jpg"
        with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
            vision._resolve_image_bytes(str(missing))

    def test_empty_file_raises(self, tmp_path):
        """Empty file should raise ValueError."""
        vision = FakeVisionLLM()
        empty = tmp_path / "empty.png"
        empty.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            vision._resolve_image_bytes(empty)

    def test_base64_decodes(self):
        """Base64 string should be decoded to bytes."""
        vision = FakeVisionLLM()
        raw = b"some binary image data!!"
        encoded = base64.b64encode(raw).decode("ascii")
        result = vision._resolve_image_bytes(encoded)
        assert result == raw

    def test_base64_data_uri_decodes(self):
        """Base64 with data URI prefix should be decoded."""
        vision = FakeVisionLLM()
        raw = b"more binary data here"
        encoded = base64.b64encode(raw).decode("ascii")
        data_uri = f"data:image/png;base64,{encoded}"
        result = vision._resolve_image_bytes(data_uri)
        assert result == raw

    def test_invalid_base64_raises(self):
        """Invalid base64 should raise ValueError."""
        vision = FakeVisionLLM()
        # String long enough to trigger base64 path but with invalid chars
        with pytest.raises(ValueError, match="Failed to decode"):
            vision._resolve_image_bytes("!!!!not-valid-base64-data-long-enough-for-heuristic!!!!")

    def test_malformed_data_uri_raises(self):
        """Data URI without comma should raise ValueError."""
        vision = FakeVisionLLM()
        with pytest.raises(ValueError, match="Malformed data URI"):
            vision._resolve_image_bytes("data:image/png;base64")


# ---------------------------------------------------------------------------
# VisionLLMFactory tests
# ---------------------------------------------------------------------------

class TestVisionLLMFactory:
    """Tests for Vision LLM factory."""

    def setup_method(self):
        VisionLLMFactory.clear_registry()

    def teardown_method(self):
        VisionLLMFactory.clear_registry()

    # -- Registration -------------------------------------------------------

    def test_register_provider_adds_to_registry(self):
        VisionLLMFactory.register_provider("fake", FakeVisionLLM)
        assert "fake" in VisionLLMFactory._PROVIDERS

    def test_register_non_subclass_raises(self):
        class NotAVisionLLM:
            pass

        with pytest.raises(ValueError, match="must inherit from BaseVisionLLM"):
            VisionLLMFactory.register_provider("bad", NotAVisionLLM)

    def test_register_is_case_insensitive(self):
        VisionLLMFactory.register_provider("FAKE", FakeVisionLLM)
        assert VisionLLMFactory.is_registered("fake")

    # -- Creation via vision config -----------------------------------------

    def test_create_returns_vision_llm_instance(self):
        VisionLLMFactory.register_provider("fake", FakeVisionLLM)
        settings = Settings()
        settings.vision = {"provider": "fake"}

        vision = VisionLLMFactory.create(settings)
        assert isinstance(vision, BaseVisionLLM)

    def test_create_with_override_kwargs(self):
        VisionLLMFactory.register_provider("fake", FakeVisionLLM)
        settings = Settings()
        settings.vision = {"provider": "fake"}

        vision = VisionLLMFactory.create(settings, prefix="Photo of")
        result = vision.describe(b"test")
        assert result.startswith("Photo of")

    # -- Fallback to llm.vision_provider ------------------------------------

    def test_create_falls_back_to_llm_vision_provider(self):
        """When no settings.vision, should use settings.llm.vision_provider."""
        VisionLLMFactory.register_provider("azure", AnotherFakeVisionLLM)
        settings = Settings()
        # No settings.vision key
        settings.llm = {"provider": "openai", "vision_provider": "azure"}

        vision = VisionLLMFactory.create(settings)
        assert isinstance(vision, BaseVisionLLM)
        result = vision.describe(b"test")
        assert "photograph" in result.lower()

    # -- Error handling -----------------------------------------------------

    def test_missing_provider_config_raises(self):
        """When no vision config at all, should raise ValueError."""
        settings = Settings()
        settings.llm = {"provider": "openai"}  # No vision_provider
        with pytest.raises(ValueError, match="Missing required"):
            VisionLLMFactory.create(settings)

    def test_unknown_provider_raises(self):
        settings = Settings()
        settings.vision = {"provider": "nonexistent"}
        with pytest.raises(ValueError, match="nonexistent"):
            VisionLLMFactory.create(settings)

    def test_instantiation_failure_wraps_error(self):
        class BrokenVisionLLM(BaseVisionLLM):
            def __init__(self, settings=None, **kwargs):
                raise RuntimeError("GPU not available")

            def describe(self, image, trace=None, **kwargs):
                return ""

        VisionLLMFactory.register_provider("broken", BrokenVisionLLM)
        settings = Settings()
        settings.vision = {"provider": "broken"}

        with pytest.raises(RuntimeError, match="Failed to instantiate"):
            VisionLLMFactory.create(settings)

    # -- Listing & querying -------------------------------------------------

    def test_list_providers_includes_registered(self):
        VisionLLMFactory.register_provider("b", FakeVisionLLM)
        VisionLLMFactory.register_provider("a", AnotherFakeVisionLLM)
        assert VisionLLMFactory.list_providers() == ["a", "b"]

    def test_is_registered_false_for_unknown(self):
        assert not VisionLLMFactory.is_registered("ghost")

    def test_clear_registry(self):
        VisionLLMFactory.register_provider("fake", FakeVisionLLM)
        VisionLLMFactory.clear_registry()
        assert VisionLLMFactory.list_providers() == []


# ---------------------------------------------------------------------------
# Settings integration
# ---------------------------------------------------------------------------

class TestVisionSettingsIntegration:
    def test_settings_has_vision_field(self):
        """Settings dataclass must declare a vision field for factory use."""
        settings = Settings()
        assert hasattr(settings, "vision")
        assert isinstance(settings.vision, dict)

    def test_settings_vision_field_is_writable(self):
        """Vision config dict should be assignable."""
        settings = Settings()
        settings.vision = {"provider": "azure"}
        assert settings.vision["provider"] == "azure"


# ---------------------------------------------------------------------------
# describe() provider-style usage tests
# ---------------------------------------------------------------------------

class TestDescribeProviderStyle:
    """End-to-end tests simulating how ImageCaptioner (B2.7) will use Vision LLM."""

    def test_describe_from_bytes(self):
        """ImageCaptioner receives raw image bytes from PDF extraction."""
        vision = FakeVisionLLM()
        png_data = _make_png_bytes(200)
        result = vision.describe(png_data)
        assert "200 bytes" in result

    def test_describe_from_path(self, tmp_path):
        """ImageCaptioner receives a file path from ImageStorage."""
        img = tmp_path / "extracted_image.png"
        img.write_bytes(b"pretend this is a real PNG from PDF")

        vision = FakeVisionLLM()
        result = vision.describe(img)
        assert "bytes" in result

    def test_describe_passes_detail_kwarg(self):
        """Provider-specific kwargs like 'detail' should be forwarded."""
        vision = FakeVisionLLM()
        result = vision.describe(b"data", detail="high")
        assert "detail=high" in result

    def test_describe_default_detail(self):
        """Default detail level is used when not specified."""
        vision = FakeVisionLLM()
        result = vision.describe(b"data")
        assert "detail=auto" in result
