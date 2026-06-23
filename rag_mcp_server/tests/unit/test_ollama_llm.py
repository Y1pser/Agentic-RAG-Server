"""Unit tests for OllamaLLM provider.

All external HTTP calls are mocked via ``httpx.Client``.  Configuration is
passed through ``Settings.llm`` dict — no env vars or .env file needed.

IMPORTANT: Because OllamaLLM.chat() creates an ``httpx.Client`` internally
(rather than holding a persistent client), the ``patch("httpx.Client")``
context manager MUST wrap the ``chat()`` call as well — not just the
constructor.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from rag_mcp_server.src.core.settings import Settings
from rag_mcp_server.src.libs.llm.base_llm import ChatResponse, Message
from rag_mcp_server.src.libs.llm.llm_factory import LLMFactory

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_ollama_response(
    content: str = "Hello from Ollama!",
    model: str = "llama3",
    status_code: int = 200,
    prompt_eval_count: int = 15,
    eval_count: int = 25,
) -> MagicMock:
    """Build a mock ``httpx`` response matching Ollama's /api/chat format."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {
        "model": model,
        "created_at": "2026-01-28T12:00:00.000000Z",
        "message": {
            "role": "assistant",
            "content": content,
        },
        "done": True,
        "done_reason": "stop",
        "total_duration": 1234567890,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }
    return resp


def _make_error_response(
    status_code: int = 400,
    error_message: str = "model not found",
) -> MagicMock:
    """Build a mock error HTTP response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"error": error_message}
    resp.text = f"Error: {error_message}"
    return resp


def _valid_messages() -> List[Message]:
    return [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="What is the capital of France?"),
    ]


def _reload_ollama_module() -> None:
    """Force re-import so auto-registration fires again."""
    import rag_mcp_server.src.libs.llm.ollama_llm as mod

    importlib.reload(mod)


# ═══════════════════════════════════════════════════════════════════════
# Pytest fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _clean_factory() -> None:
    """Ensure the factory registry is clean before each test."""
    LLMFactory.clear_registry()
    _reload_ollama_module()
    yield
    LLMFactory.clear_registry()


@pytest.fixture
def provider_cls():
    """Return the OllamaLLM class (auto-registered)."""
    import rag_mcp_server.src.libs.llm.ollama_llm as mod

    return mod.OllamaLLM


# ═══════════════════════════════════════════════════════════════════════
# Factory Registration
# ═══════════════════════════════════════════════════════════════════════


class TestOllamaFactoryRegistration:
    """Tests for Ollama provider registration with LLMFactory."""

    def test_auto_registered_with_factory(self, provider_cls) -> None:
        assert LLMFactory.is_registered("ollama")

    def test_case_insensitive_registration(self, provider_cls) -> None:
        assert LLMFactory.is_registered("OLLAMA")
        assert LLMFactory.is_registered("Ollama")

    def test_factory_create_ollama(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama", "model": "llama3"})
        with patch("httpx.Client"):
            llm = LLMFactory.create(settings)
        assert isinstance(llm, provider_cls)


# ═══════════════════════════════════════════════════════════════════════
# Initialisation
# ═══════════════════════════════════════════════════════════════════════


class TestOllamaInit:
    """Tests for OllamaLLM initialisation (no HTTP calls)."""

    def test_init_default_base_url(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama"})
        llm = provider_cls(settings)
        assert llm.base_url == "http://localhost:11434"

    def test_init_custom_base_url_from_settings(self, provider_cls) -> None:
        settings = Settings(
            llm={
                "provider": "ollama",
                "base_url": "http://192.168.1.100:11434",
            }
        )
        llm = provider_cls(settings)
        assert llm.base_url == "http://192.168.1.100:11434"

    def test_init_kwarg_overrides_base_url(self, provider_cls) -> None:
        settings = Settings(
            llm={
                "provider": "ollama",
                "base_url": "http://settings:11434",
            }
        )
        llm = provider_cls(settings, base_url="http://kwarg:11434")
        assert llm.base_url == "http://kwarg:11434"

    def test_init_base_url_from_env(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama"})
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://env:11434"}):
            llm = provider_cls(settings)
        assert llm.base_url == "http://env:11434"

    def test_init_explicit_overrides_env(self, provider_cls) -> None:
        settings = Settings(
            llm={"provider": "ollama", "base_url": "http://settings:11434"}
        )
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://env:11434"}):
            llm = provider_cls(settings)
        assert llm.base_url == "http://settings:11434"

    def test_init_strips_trailing_slash(self, provider_cls) -> None:
        settings = Settings(
            llm={"provider": "ollama", "base_url": "http://localhost:11434/"}
        )
        llm = provider_cls(settings)
        assert llm.base_url == "http://localhost:11434"

    def test_init_model_from_settings(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama", "model": "mistral"})
        llm = provider_cls(settings)
        assert llm.model == "mistral"

    def test_init_kwarg_overrides_model(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama", "model": "llama3"})
        llm = provider_cls(settings, model="codellama")
        assert llm.model == "codellama"

    def test_init_default_model(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama"})
        llm = provider_cls(settings)
        assert llm.model == "llama3"

    def test_init_temperature_from_settings(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama", "temperature": 0.5})
        llm = provider_cls(settings)
        assert llm.default_temperature == 0.5

    def test_init_default_temperature(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama"})
        llm = provider_cls(settings)
        assert llm.default_temperature == 0.7

    def test_init_max_tokens_from_settings(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama", "max_tokens": 4096})
        llm = provider_cls(settings)
        assert llm.default_max_tokens == 4096

    def test_init_default_max_tokens(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama"})
        llm = provider_cls(settings)
        assert llm.default_max_tokens == 2048

    def test_init_custom_timeout(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama", "timeout": 300})
        llm = provider_cls(settings)
        assert llm.timeout == 300.0

    def test_init_default_timeout(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama"})
        llm = provider_cls(settings)
        assert llm.timeout == 120.0


# ═══════════════════════════════════════════════════════════════════════
# Chat — success paths  (mock MUST wrap chat() call)
# ═══════════════════════════════════════════════════════════════════════


class TestOllamaChat:
    """Tests for OllamaLLM.chat() with mocked HTTP."""

    def test_chat_success(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama", "model": "llama3"})
        llm = provider_cls(settings)
        mock_resp = _make_ollama_response("Hello from Ollama!", "llama3")

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_resp
            )
            result = llm.chat(_valid_messages())

        assert result.content == "Hello from Ollama!"
        assert result.model == "llama3"

    def test_chat_returns_usage_stats(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama", "model": "llama3"})
        llm = provider_cls(settings)
        mock_resp = _make_ollama_response(prompt_eval_count=20, eval_count=50)

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_resp
            )
            result = llm.chat(_valid_messages())

        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 20
        assert result.usage["completion_tokens"] == 50
        assert result.usage["total_tokens"] == 70

    def test_chat_preserves_raw_response(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama", "model": "llama3"})
        llm = provider_cls(settings)
        mock_resp = _make_ollama_response()

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_resp
            )
            result = llm.chat(_valid_messages())

        assert result.raw_response is not None
        assert "model" in result.raw_response
        assert "message" in result.raw_response

    def test_chat_calls_correct_endpoint(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama", "model": "llama3"})
        llm = provider_cls(settings)
        mock_resp = _make_ollama_response()

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_resp
            )
            llm.chat([Message(role="user", content="Hello")])

            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            assert call_args.args[0] == "http://localhost:11434/api/chat"

    def test_chat_uses_stream_false(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama", "model": "llama3"})
        llm = provider_cls(settings)
        mock_resp = _make_ollama_response()

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_resp
            )
            llm.chat([Message(role="user", content="Hello")])

            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            payload = call_args.kwargs["json"]
            assert payload["stream"] is False

    def test_chat_sets_num_predict(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama", "max_tokens": 512})
        llm = provider_cls(settings)
        mock_resp = _make_ollama_response()

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_resp
            )
            llm.chat([Message(role="user", content="Hello")])

            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            payload = call_args.kwargs["json"]
            assert payload["options"]["num_predict"] == 512

    def test_chat_with_system_message(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama", "model": "llama3"})
        llm = provider_cls(settings)
        mock_resp = _make_ollama_response("I understand the context.")

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_resp
            )
            result = llm.chat(_valid_messages())

        assert result.content == "I understand the context."

    def test_chat_model_override(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama", "model": "llama3"})
        llm = provider_cls(settings)
        mock_resp = _make_ollama_response(model="mistral")

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_resp
            )
            result = llm.chat(_valid_messages(), model="mistral")

        assert result.model == "mistral"

    def test_chat_temperature_override(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama"})
        llm = provider_cls(settings)
        mock_resp = _make_ollama_response()

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_resp
            )
            llm.chat([Message(role="user", content="Hello")], temperature=0.1)

            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            assert call_args.kwargs["json"]["options"]["temperature"] == 0.1

    def test_chat_max_tokens_override(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama"})
        llm = provider_cls(settings)
        mock_resp = _make_ollama_response()

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_resp
            )
            llm.chat([Message(role="user", content="Hello")], max_tokens=100)

            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            assert call_args.kwargs["json"]["options"]["num_predict"] == 100


# ═══════════════════════════════════════════════════════════════════════
# Validation  (no HTTP needed — validation runs first)
# ═══════════════════════════════════════════════════════════════════════


class TestOllamaValidation:
    """Tests for input validation."""

    def test_chat_empty_messages_raises(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama"})
        llm = provider_cls(settings)
        with pytest.raises(ValueError, match="cannot be empty"):
            llm.chat([])

    def test_chat_invalid_role_raises(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama"})
        llm = provider_cls(settings)
        with pytest.raises(ValueError, match="invalid role"):
            llm.chat([Message(role="invalid", content="Hello")])

    def test_chat_empty_content_raises(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama"})
        llm = provider_cls(settings)
        with pytest.raises(ValueError, match="empty content"):
            llm.chat([Message(role="user", content="  ")])


# ═══════════════════════════════════════════════════════════════════════
# Error Handling  (mock MUST wrap chat() call + pytest.raises)
# ═══════════════════════════════════════════════════════════════════════


class TestOllamaErrorHandling:
    """Tests for error handling scenarios."""

    def test_api_error_response(self, provider_cls) -> None:
        settings = Settings(llm={"provider": "ollama", "model": "llama3"})
        llm = provider_cls(settings)
        mock_resp = _make_error_response(404, "model not found")

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_resp
            )
            with pytest.raises(RuntimeError, match="HTTP 404"):
                llm.chat(_valid_messages())

    def test_timeout_error(self, provider_cls) -> None:
        import httpx

        settings = Settings(llm={"provider": "ollama", "model": "llama3"})
        llm = provider_cls(settings)

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.side_effect = (
                httpx.TimeoutException("timeout")
            )
            with pytest.raises(RuntimeError, match="timed out"):
                llm.chat(_valid_messages())

    def test_connection_error(self, provider_cls) -> None:
        import httpx

        settings = Settings(llm={"provider": "ollama", "model": "llama3"})
        llm = provider_cls(settings)

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.side_effect = (
                httpx.ConnectError("Connection refused")
            )
            with pytest.raises(RuntimeError, match="Connection failed"):
                llm.chat(_valid_messages())

    def test_request_error(self, provider_cls) -> None:
        import httpx

        settings = Settings(llm={"provider": "ollama", "model": "llama3"})
        llm = provider_cls(settings)

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.side_effect = (
                httpx.RequestError("Network error")
            )
            with pytest.raises(RuntimeError, match="Request failed"):
                llm.chat(_valid_messages())

    def test_error_does_not_leak_base_url(self, provider_cls) -> None:
        settings = Settings(
            llm={
                "provider": "ollama",
                "model": "llama3",
                "base_url": "http://secret-server:11434",
            }
        )
        llm = provider_cls(settings)
        mock_resp = _make_error_response(500, "internal error")

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_resp
            )
            with pytest.raises(RuntimeError) as exc_info:
                llm.chat(_valid_messages())
        assert "secret-server" not in str(exc_info.value)

    def test_unexpected_response_format(self, provider_cls) -> None:
        """Response with neither 'message' nor 'response' key."""
        settings = Settings(llm={"provider": "ollama", "model": "llama3"})
        llm = provider_cls(settings)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"unexpected": "format"}

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = resp
            with pytest.raises(RuntimeError, match="Unexpected response format"):
                llm.chat(_valid_messages())


# ═══════════════════════════════════════════════════════════════════════
# Factory multi-provider coexistence
# ═══════════════════════════════════════════════════════════════════════


class TestOllamaCoexistence:
    """Ollama should coexist with other providers in the factory registry."""

    @pytest.fixture(autouse=True)
    def _register_all(self) -> None:
        """Load all provider modules so the factory is populated."""
        LLMFactory.clear_registry()

        # Import all modules first, then reload them
        import rag_mcp_server.src.libs.llm.openai_llm as _oai
        import rag_mcp_server.src.libs.llm.azure_llm as _az
        import rag_mcp_server.src.libs.llm.deepseek_llm as _ds
        import rag_mcp_server.src.libs.llm.ollama_llm as _ol

        importlib.reload(_oai)
        importlib.reload(_az)
        importlib.reload(_ds)
        importlib.reload(_ol)
        yield
        LLMFactory.clear_registry()

    def test_all_four_providers_registered(self) -> None:
        providers = LLMFactory.list_providers()
        assert "azure" in providers
        assert "deepseek" in providers
        assert "ollama" in providers
        assert "openai" in providers

    def test_factory_creates_ollama(self) -> None:
        import rag_mcp_server.src.libs.llm.ollama_llm as mod

        settings = Settings(llm={"provider": "ollama", "model": "llama3"})
        with patch("httpx.Client"):
            llm = LLMFactory.create(settings)
        assert isinstance(llm, mod.OllamaLLM)
