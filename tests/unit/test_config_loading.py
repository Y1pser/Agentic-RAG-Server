"""Test configuration loading and validation."""

import pytest
import tempfile
from pathlib import Path
from rag_mcp_server.src.core.settings import Settings, load_settings, validate_settings


class TestSettingsDataclass:
    """Unit tests for Settings dataclass."""

    def test_settings_has_required_fields(self):
        """Settings should expose all top-level config sections."""
        settings = Settings()
        assert hasattr(settings, "llm")
        assert hasattr(settings, "embedding")
        assert hasattr(settings, "vector_store")
        assert hasattr(settings, "retrieval")
        assert hasattr(settings, "observability")


class TestLoadSettings:
    """Tests for load_settings function."""

    def test_loads_valid_yaml(self):
        """Should parse a valid YAML file into a Settings object."""
        yaml_content = """
llm:
  provider: openai
  model: gpt-4o
  api_key: test-key
embedding:
  provider: openai
  model: text-embedding-3-small
vector_store:
  backend: chroma
retrieval:
  sparse_backend: bm25
  fusion_algorithm: rrf
  rerank_backend: none
observability:
  enabled: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            settings = load_settings(f.name)
        Path(f.name).unlink()
        assert settings.llm["provider"] == "openai"
        assert settings.vector_store["backend"] == "chroma"

    def test_missing_file_raises(self):
        """Should raise FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError):
            load_settings("/nonexistent/path.yaml")


class TestValidateSettings:
    """Tests for settings validation."""

    def test_missing_llm_provider_raises(self):
        """Should raise ValueError when llm.provider is missing."""
        settings = Settings()
        settings.llm = {"model": "gpt-4o"}  # no provider
        with pytest.raises(ValueError, match="llm.provider"):
            validate_settings(settings)

    def test_missing_embedding_provider_raises(self):
        """Should raise ValueError when embedding.provider is missing."""
        settings = Settings()
        settings.llm = {"provider": "openai"}
        settings.embedding = {"model": "text-embedding-3-small"}  # no provider
        with pytest.raises(ValueError, match="embedding.provider"):
            validate_settings(settings)

    def test_valid_settings_pass(self):
        """Should not raise for valid settings."""
        settings = Settings()
        settings.llm = {"provider": "openai"}
        settings.embedding = {"provider": "openai"}
        settings.vector_store = {"backend": "chroma"}
        validate_settings(settings)  # should not raise
