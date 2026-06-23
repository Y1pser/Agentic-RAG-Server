"""Unit tests for OpenAILLM, AzureLLM, and DeepSeekLLM providers.

All external HTTP calls are mocked so the suite runs offline and
deterministically.  Each provider is tested for initialisation, message
validation, successful chat completion, error propagation, and factory
registration.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from rag_mcp_server.src.core.settings import Settings
from rag_mcp_server.src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from rag_mcp_server.src.libs.llm.llm_factory import LLMFactory


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_mock_response(
    content: str = "Hello from mock!",
    model: str = "mock-model",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    """Build a mock ``openai`` chat-completion response object."""
    mock = MagicMock()
    mock.model = model
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    mock.usage.prompt_tokens = prompt_tokens
    mock.usage.completion_tokens = completion_tokens
    mock.usage.total_tokens = prompt_tokens + completion_tokens
    mock.to_dict.return_value = {"mock": True, "model": model}
    return mock


def _valid_messages() -> List[Message]:
    return [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="What is the capital of France?"),
    ]


def _reload_provider_modules() -> None:
    """Force re-import of provider modules so auto-registration fires again.

    Must be called after ``LLMFactory.clear_registry()`` to re-populate
    the factory with the provider classes.
    """
    import rag_mcp_server.src.libs.llm.openai_llm
    import rag_mcp_server.src.libs.llm.azure_llm
    import rag_mcp_server.src.libs.llm.deepseek_llm

    importlib.reload(rag_mcp_server.src.libs.llm.openai_llm)
    importlib.reload(rag_mcp_server.src.libs.llm.azure_llm)
    importlib.reload(rag_mcp_server.src.libs.llm.deepseek_llm)


@pytest.fixture(autouse=True)
def _clean_factory() -> None:
    """Ensure the factory registry is clean before each test, then re-register
    all concrete providers so every test sees a populated registry."""
    LLMFactory.clear_registry()
    _reload_provider_modules()
    yield
    LLMFactory.clear_registry()


# ═══════════════════════════════════════════════════════════════════════
# OpenAILLM
# ═══════════════════════════════════════════════════════════════════════


class TestOpenAILLM:
    """Tests for the standard OpenAI provider."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        import rag_mcp_server.src.libs.llm.openai_llm as mod

        self.provider_cls = mod.OpenAILLM

    # ── registration ────────────────────────────────────────────────

    def test_auto_registered_with_factory(self) -> None:
        assert LLMFactory.is_registered("openai")
        assert LLMFactory.is_registered("OPENAI")
        assert "openai" in LLMFactory.list_providers()

    # ── initialisation ──────────────────────────────────────────────

    def test_init_with_settings_api_key(self) -> None:
        settings = Settings(llm={"provider": "openai", "api_key": "sk-test"})
        with patch("openai.OpenAI") as mock_client:
            self.provider_cls(settings)
            mock_client.assert_called_once_with(api_key="sk-test")

    def test_init_with_env_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        settings = Settings(llm={"provider": "openai"})
        with patch("openai.OpenAI") as mock_client:
            self.provider_cls(settings)
            mock_client.assert_called_once_with(api_key="sk-env")

    def test_init_with_base_url(self) -> None:
        settings = Settings(
            llm={
                "provider": "openai",
                "api_key": "sk-test",
                "base_url": "https://proxy.example.com/v1",
            }
        )
        with patch("openai.OpenAI") as mock_client:
            self.provider_cls(settings)
            mock_client.assert_called_once_with(
                api_key="sk-test", base_url="https://proxy.example.com/v1"
            )

    def test_init_raises_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = Settings(llm={"provider": "openai"})
        with pytest.raises(ValueError, match="API key not found"):
            self.provider_cls(settings)

    def test_init_kwargs_override_model(self) -> None:
        settings = Settings(
            llm={"provider": "openai", "api_key": "sk-test", "model": "gpt-4o"}
        )
        with patch("openai.OpenAI"):
            llm = self.provider_cls(settings, model="gpt-4o-mini")
        assert llm.model == "gpt-4o-mini"

    def test_init_extra_kwargs_become_default_params(self) -> None:
        settings = Settings(llm={"provider": "openai", "api_key": "sk-test"})
        with patch("openai.OpenAI"):
            llm = self.provider_cls(settings, temperature=0.3, max_tokens=256)
        assert llm.default_params == {"temperature": 0.3, "max_tokens": 256}

    # ── chat ────────────────────────────────────────────────────────

    def test_chat_success(self) -> None:
        settings = Settings(llm={"provider": "openai", "api_key": "sk-test"})
        mock_resp = _make_mock_response("Paris", "gpt-4o")
        with patch("openai.OpenAI") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_resp
            llm = self.provider_cls(settings)
        result = llm.chat(_valid_messages())
        assert isinstance(result, ChatResponse)
        assert result.content == "Paris"
        assert result.model == "gpt-4o"
        assert result.usage is not None
        assert result.usage["total_tokens"] == 15
        assert result.raw_response is not None

    def test_chat_merges_default_and_call_params(self) -> None:
        settings = Settings(llm={"provider": "openai", "api_key": "sk-test"})
        mock_resp = _make_mock_response()
        with patch("openai.OpenAI") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_resp
            llm = self.provider_cls(settings, temperature=0.2)
        llm.chat(_valid_messages(), max_tokens=100)
        call_kwargs = (
            mock_client.return_value.chat.completions.create.call_args.kwargs
        )
        assert call_kwargs["temperature"] == 0.2  # default param
        assert call_kwargs["max_tokens"] == 100  # call-time override

    def test_chat_raises_on_api_failure(self) -> None:
        settings = Settings(llm={"provider": "openai", "api_key": "sk-test"})
        with patch("openai.OpenAI") as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = (
                ConnectionError("timeout")
            )
            llm = self.provider_cls(settings)
        with pytest.raises(RuntimeError, match="OpenAI API call failed"):
            llm.chat(_valid_messages())

    def test_chat_raises_on_invalid_messages(self) -> None:
        settings = Settings(llm={"provider": "openai", "api_key": "sk-test"})
        with patch("openai.OpenAI"):
            llm = self.provider_cls(settings)
        with pytest.raises(ValueError):
            llm.chat([])

    # ── factory integration ─────────────────────────────────────────

    def test_factory_create_openai(self) -> None:
        settings = Settings(llm={"provider": "openai", "api_key": "sk-test"})
        with patch("openai.OpenAI"):
            llm = LLMFactory.create(settings)
        assert isinstance(llm, self.provider_cls)


# ═══════════════════════════════════════════════════════════════════════
# AzureLLM
# ═══════════════════════════════════════════════════════════════════════


class TestAzureLLM:
    """Tests for the Azure OpenAI provider."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        import rag_mcp_server.src.libs.llm.azure_llm as mod

        self.provider_cls = mod.AzureLLM

    def test_auto_registered_with_factory(self) -> None:
        assert LLMFactory.is_registered("azure")

    def test_init_with_settings(self) -> None:
        settings = Settings(
            llm={
                "provider": "azure",
                "api_key": "sk-azure",
                "endpoint": "https://example.openai.azure.com",
            }
        )
        with patch("openai.AzureOpenAI") as mock_client:
            self.provider_cls(settings)
            mock_client.assert_called_once_with(
                api_key="sk-azure",
                azure_endpoint="https://example.openai.azure.com",
                api_version="2024-02-15-preview",  # default
            )

    def test_init_with_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "sk-env")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
        settings = Settings(llm={"provider": "azure"})
        with patch("openai.AzureOpenAI") as mock_client:
            self.provider_cls(settings)
            mock_client.assert_called_once_with(
                api_key="sk-env",
                azure_endpoint="https://env.openai.azure.com",
                api_version="2024-06-01",
            )

    def test_init_raises_without_endpoint(self) -> None:
        settings = Settings(llm={"provider": "azure", "api_key": "sk-test"})
        with pytest.raises(ValueError, match="endpoint not found"):
            self.provider_cls(settings)

    def test_init_raises_without_api_key(self) -> None:
        settings = Settings(
            llm={
                "provider": "azure",
                "endpoint": "https://example.openai.azure.com",
            }
        )
        with pytest.raises(ValueError, match="API key not found"):
            self.provider_cls(settings)

    def test_chat_success(self) -> None:
        settings = Settings(
            llm={
                "provider": "azure",
                "api_key": "sk-azure",
                "endpoint": "https://example.openai.azure.com",
                "model": "gpt-4o-deployment",
            }
        )
        mock_resp = _make_mock_response("Azure response", "gpt-4o")
        with patch("openai.AzureOpenAI") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_resp
            llm = self.provider_cls(settings)
        result = llm.chat(_valid_messages())
        assert result.content == "Azure response"
        assert result.model == "gpt-4o"

    def test_factory_create_azure(self) -> None:
        settings = Settings(
            llm={
                "provider": "azure",
                "api_key": "sk-azure",
                "endpoint": "https://example.openai.azure.com",
            }
        )
        with patch("openai.AzureOpenAI"):
            llm = LLMFactory.create(settings)
        assert isinstance(llm, self.provider_cls)


# ═══════════════════════════════════════════════════════════════════════
# DeepSeekLLM
# ═══════════════════════════════════════════════════════════════════════


class TestDeepSeekLLM:
    """Tests for the DeepSeek provider."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        import rag_mcp_server.src.libs.llm.deepseek_llm as mod

        self.provider_cls = mod.DeepSeekLLM

    def test_auto_registered_with_factory(self) -> None:
        assert LLMFactory.is_registered("deepseek")

    def test_init_defaults_to_deepseek_base_url(self) -> None:
        settings = Settings(llm={"provider": "deepseek", "api_key": "sk-ds"})
        with patch("openai.OpenAI") as mock_client:
            self.provider_cls(settings)
            mock_client.assert_called_once_with(
                api_key="sk-ds", base_url="https://api.deepseek.com/v1"
            )

    def test_init_falls_back_to_openai_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fallback")
        settings = Settings(llm={"provider": "deepseek"})
        with patch("openai.OpenAI") as mock_client:
            self.provider_cls(settings)
            mock_client.assert_called_once_with(
                api_key="sk-openai-fallback",
                base_url="https://api.deepseek.com/v1",
            )

    def test_init_prefers_deepseek_key_over_openai_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
        settings = Settings(llm={"provider": "deepseek"})
        with patch("openai.OpenAI") as mock_client:
            self.provider_cls(settings)
            mock_client.assert_called_once_with(
                api_key="sk-ds", base_url="https://api.deepseek.com/v1"
            )

    def test_init_raises_without_any_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = Settings(llm={"provider": "deepseek"})
        with pytest.raises(ValueError, match="API key not found"):
            self.provider_cls(settings)

    def test_init_default_model_is_deepseek_chat(self) -> None:
        settings = Settings(llm={"provider": "deepseek", "api_key": "sk-ds"})
        with patch("openai.OpenAI"):
            llm = self.provider_cls(settings)
        assert llm.model == "deepseek-chat"

    def test_chat_success(self) -> None:
        settings = Settings(llm={"provider": "deepseek", "api_key": "sk-ds"})
        mock_resp = _make_mock_response("DeepSeek says hello", "deepseek-chat")
        with patch("openai.OpenAI") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_resp
            llm = self.provider_cls(settings)
        result = llm.chat(_valid_messages())
        assert result.content == "DeepSeek says hello"
        assert result.model == "deepseek-chat"

    def test_factory_create_deepseek(self) -> None:
        settings = Settings(llm={"provider": "deepseek", "api_key": "sk-ds"})
        with patch("openai.OpenAI"):
            llm = LLMFactory.create(settings)
        assert isinstance(llm, self.provider_cls)


# ═══════════════════════════════════════════════════════════════════════
# Factory multi-provider registration
# ═══════════════════════════════════════════════════════════════════════


class TestMultiProviderRegistration:
    """All three providers should coexist in the factory registry."""

    def test_all_three_registered(self) -> None:
        providers = LLMFactory.list_providers()
        assert "azure" in providers
        assert "deepseek" in providers
        assert "openai" in providers

    def test_factory_resolves_correct_class_per_provider(self) -> None:
        import rag_mcp_server.src.libs.llm.openai_llm as oai
        import rag_mcp_server.src.libs.llm.azure_llm as az
        import rag_mcp_server.src.libs.llm.deepseek_llm as ds

        with patch("openai.OpenAI"):
            llm_o = LLMFactory.create(
                Settings(llm={"provider": "openai", "api_key": "sk"})
            )
            assert isinstance(llm_o, oai.OpenAILLM)

        with patch("openai.AzureOpenAI"):
            llm_a = LLMFactory.create(
                Settings(
                    llm={
                        "provider": "azure",
                        "api_key": "sk",
                        "endpoint": "https://x.openai.azure.com",
                    }
                )
            )
            assert isinstance(llm_a, az.AzureLLM)

        with patch("openai.OpenAI"):
            llm_d = LLMFactory.create(
                Settings(llm={"provider": "deepseek", "api_key": "sk"})
            )
            assert isinstance(llm_d, ds.DeepSeekLLM)
