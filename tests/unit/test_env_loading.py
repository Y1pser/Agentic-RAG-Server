"""Test environment variable loading via python-dotenv."""

from rag_mcp_server.src.core.settings import apply_env_overrides, Settings


class TestEnvOverrides:
    """Tests for environment variable overrides on settings."""

    def test_env_var_overrides_llm_api_key(self, monkeypatch):
        """Environment variables should override settings values."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-test-key")
        settings = Settings()
        settings.llm = {"provider": "openai", "model": "gpt-4o"}

        apply_env_overrides(settings)

        assert settings.llm["api_key"] == "sk-env-test-key"

    def test_env_var_missing_does_not_crash(self):
        """Missing env vars should not cause errors."""
        settings = Settings()
        settings.llm = {"provider": "openai"}
        apply_env_overrides(settings)  # should not raise

    def test_multiple_providers_get_keys(self, monkeypatch):
        """Multiple search API keys should all be picked up."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        monkeypatch.setenv("SERPAPI_API_KEY", "serp-test")
        settings = Settings()
        settings.agent = {"web_search": {"backend": "tavily"}}

        apply_env_overrides(settings)

        assert settings.agent["web_search"].get("tavily_api_key") == "tvly-test"
        assert settings.agent["web_search"].get("serpapi_api_key") == "serp-test"
