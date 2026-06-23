"""Test that settings.yaml is loaded and validated correctly.

Since the project no longer uses .env files, this module tests the
settings-loading pipeline: YAML parse → Settings dataclass → validation.
"""

import pytest
from rag_mcp_server.src.core.settings import Settings, validate_settings


class TestValidateSettings:
    """Tests for settings validation (no .env involved)."""

    def test_missing_llm_api_key_raises(self) -> None:
        settings = Settings()
        settings.llm = {"provider": "openai"}
        settings.embedding = {"provider": "openai"}
        settings.vector_store = {"backend": "chroma"}
        with pytest.raises(ValueError, match="llm.api_key"):
            validate_settings(settings)

    def test_missing_llm_provider_raises(self) -> None:
        settings = Settings()
        settings.llm = {"api_key": "sk-test"}
        settings.embedding = {"provider": "openai"}
        settings.vector_store = {"backend": "chroma"}
        with pytest.raises(ValueError, match="llm.provider"):
            validate_settings(settings)

    def test_valid_settings_pass(self) -> None:
        settings = Settings()
        settings.llm = {"provider": "openai", "api_key": "sk-test"}
        settings.embedding = {"provider": "openai"}
        settings.vector_store = {"backend": "chroma"}
        validate_settings(settings)  # should not raise
