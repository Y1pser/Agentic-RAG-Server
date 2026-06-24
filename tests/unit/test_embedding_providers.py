"""Unit tests for OpenAIEmbedding and AzureEmbedding providers.

All external HTTP calls are mocked.  Configuration is passed through
``Settings.embedding`` dict — no env vars or .env file needed.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from rag_mcp_server.src.core.settings import Settings
from rag_mcp_server.src.libs.embedding.base_embedding import (
    BaseEmbedding,
    EmbeddingResponse,
)
from rag_mcp_server.src.libs.embedding.embedding_factory import EmbeddingFactory


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_mock_embedding_response(
    texts: List[str],
    model: str = "mock-embedding-model",
    dimension: int = 1536,
    total_tokens: int = 0,
) -> MagicMock:
    """Build a mock ``openai`` embeddings response object.

    Creates one embedding per input text, each a list of ``dimension`` floats.
    """
    mock = MagicMock()
    mock.model = model

    data_entries = []
    for i, text in enumerate(texts):
        entry = MagicMock()
        entry.index = i
        entry.embedding = [0.1 * (i + 1)] * dimension
        data_entries.append(entry)

    mock.data = data_entries
    mock.usage.total_tokens = total_tokens or sum(len(t) for t in texts)
    mock.to_dict.return_value = {"mock": True, "model": model}
    return mock


def _reload_provider_modules() -> None:
    """Force re-import of embedding provider modules so auto-registration fires."""
    import rag_mcp_server.src.libs.embedding.openai_embedding
    import rag_mcp_server.src.libs.embedding.azure_embedding

    importlib.reload(rag_mcp_server.src.libs.embedding.openai_embedding)
    importlib.reload(rag_mcp_server.src.libs.embedding.azure_embedding)


@pytest.fixture(autouse=True)
def _clean_factory() -> None:
    """Ensure the factory registry is clean before each test."""
    EmbeddingFactory.clear_registry()
    _reload_provider_modules()
    yield
    EmbeddingFactory.clear_registry()


# ═══════════════════════════════════════════════════════════════════════════
# OpenAIEmbedding
# ═══════════════════════════════════════════════════════════════════════════


class TestOpenAIEmbedding:
    """Tests for the standard OpenAI embedding provider."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        import rag_mcp_server.src.libs.embedding.openai_embedding as mod

        self.provider_cls = mod.OpenAIEmbedding

    # ── registration ────────────────────────────────────────────────────

    def test_auto_registered_with_factory(self) -> None:
        assert EmbeddingFactory.is_registered("openai")

    # ── initialisation ──────────────────────────────────────────────────

    def test_init_with_api_key_in_settings(self) -> None:
        settings = Settings(
            embedding={"provider": "openai", "api_key": "sk-test"}
        )
        with patch("openai.OpenAI") as mock_client:
            self.provider_cls(settings)
            mock_client.assert_called_once_with(api_key="sk-test")

    def test_init_with_base_url(self) -> None:
        settings = Settings(
            embedding={
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

    def test_init_raises_without_api_key(self) -> None:
        settings = Settings(embedding={"provider": "openai"})
        with pytest.raises(ValueError, match="API key not found"):
            self.provider_cls(settings)

    def test_init_kwargs_override_api_key(self) -> None:
        settings = Settings(
            embedding={"provider": "openai", "api_key": "sk-from-settings"}
        )
        with patch("openai.OpenAI") as mock_client:
            self.provider_cls(settings, api_key="sk-from-kwargs")
            mock_client.assert_called_once_with(api_key="sk-from-kwargs")

    def test_init_kwargs_override_model(self) -> None:
        settings = Settings(
            embedding={
                "provider": "openai",
                "api_key": "sk-test",
                "model": "text-embedding-3-small",
            }
        )
        with patch("openai.OpenAI"):
            emb = self.provider_cls(settings, model="text-embedding-3-large")
        assert emb.model == "text-embedding-3-large"

    def test_init_default_model(self) -> None:
        settings = Settings(
            embedding={"provider": "openai", "api_key": "sk-test"}
        )
        with patch("openai.OpenAI"):
            emb = self.provider_cls(settings)
        assert emb.model == "text-embedding-3-small"

    def test_init_dimensions_from_settings(self) -> None:
        settings = Settings(
            embedding={
                "provider": "openai",
                "api_key": "sk-test",
                "dimensions": 1536,
            }
        )
        with patch("openai.OpenAI"):
            emb = self.provider_cls(settings)
        assert emb.default_dimensions == 1536

    def test_init_dimensions_from_kwargs_overrides_settings(self) -> None:
        settings = Settings(
            embedding={
                "provider": "openai",
                "api_key": "sk-test",
                "dimensions": 1536,
            }
        )
        with patch("openai.OpenAI"):
            emb = self.provider_cls(settings, dimensions=256)
        assert emb.default_dimensions == 256

    def test_init_dimensions_none_by_default(self) -> None:
        settings = Settings(
            embedding={"provider": "openai", "api_key": "sk-test"}
        )
        with patch("openai.OpenAI"):
            emb = self.provider_cls(settings)
        assert emb.default_dimensions is None

    def test_init_extra_kwargs_become_default_params(self) -> None:
        settings = Settings(
            embedding={"provider": "openai", "api_key": "sk-test"}
        )
        with patch("openai.OpenAI"):
            emb = self.provider_cls(settings, user="test-user")
        assert emb.default_params == {"user": "test-user"}

    # ── embed ───────────────────────────────────────────────────────────

    def test_embed_single_text(self) -> None:
        settings = Settings(
            embedding={"provider": "openai", "api_key": "sk-test"}
        )
        mock_resp = _make_mock_embedding_response(["hello"])
        with patch("openai.OpenAI") as mock_client:
            mock_client.return_value.embeddings.create.return_value = mock_resp
            emb = self.provider_cls(settings)
        result = emb.embed(["hello"])
        assert isinstance(result, EmbeddingResponse)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 1536
        assert result.model == "mock-embedding-model"

    def test_embed_multiple_texts(self) -> None:
        settings = Settings(
            embedding={"provider": "openai", "api_key": "sk-test"}
        )
        texts = ["hello", "world", "test"]
        mock_resp = _make_mock_embedding_response(texts)
        with patch("openai.OpenAI") as mock_client:
            mock_client.return_value.embeddings.create.return_value = mock_resp
            emb = self.provider_cls(settings)
        result = emb.embed(texts)
        assert len(result.embeddings) == 3

    def test_embed_preserves_order(self) -> None:
        """Ensure embeddings are returned in the same order as input texts."""
        settings = Settings(
            embedding={"provider": "openai", "api_key": "sk-test"}
        )
        texts = ["first", "second", "third"]
        mock_resp = _make_mock_embedding_response(texts)
        with patch("openai.OpenAI") as mock_client:
            mock_client.return_value.embeddings.create.return_value = mock_resp
            emb = self.provider_cls(settings)
        result = emb.embed(texts)
        # first embedding should have value 0.1, third should have 0.3
        assert result.embeddings[0][0] == pytest.approx(0.1)
        assert result.embeddings[1][0] == pytest.approx(0.2)
        assert result.embeddings[2][0] == pytest.approx(0.3)

    def test_embed_empty_list(self) -> None:
        settings = Settings(
            embedding={"provider": "openai", "api_key": "sk-test"}
        )
        with patch("openai.OpenAI"):
            emb = self.provider_cls(settings)
        result = emb.embed([])
        assert result.embeddings == []
        assert result.model == "text-embedding-3-small"

    def test_embed_with_dimensions_override(self) -> None:
        settings = Settings(
            embedding={"provider": "openai", "api_key": "sk-test"}
        )
        mock_resp = _make_mock_embedding_response(
            ["hello"], dimension=256
        )
        with patch("openai.OpenAI") as mock_client:
            mock_client.return_value.embeddings.create.return_value = mock_resp
            emb = self.provider_cls(settings)
        result = emb.embed(["hello"], dimensions=256)
        assert len(result.embeddings[0]) == 256

    def test_embed_merges_default_and_call_params(self) -> None:
        settings = Settings(
            embedding={"provider": "openai", "api_key": "sk-test"}
        )
        mock_resp = _make_mock_embedding_response(["hello"])
        with patch("openai.OpenAI") as mock_client:
            mock_client.return_value.embeddings.create.return_value = mock_resp
            emb = self.provider_cls(settings, user="default-user")
        emb.embed(["hello"], dimensions=512)
        call_kwargs = (
            mock_client.return_value.embeddings.create.call_args.kwargs
        )
        assert call_kwargs["user"] == "default-user"
        assert call_kwargs["dimensions"] == 512

    def test_embed_includes_usage_when_available(self) -> None:
        settings = Settings(
            embedding={"provider": "openai", "api_key": "sk-test"}
        )
        mock_resp = _make_mock_embedding_response(
            ["hello", "world"], total_tokens=10
        )
        with patch("openai.OpenAI") as mock_client:
            mock_client.return_value.embeddings.create.return_value = mock_resp
            emb = self.provider_cls(settings)
        result = emb.embed(["hello", "world"])
        assert result.usage == {"total_tokens": 10}

    def test_embed_raises_on_api_failure(self) -> None:
        settings = Settings(
            embedding={"provider": "openai", "api_key": "sk-test"}
        )
        with patch("openai.OpenAI") as mock_client:
            mock_client.return_value.embeddings.create.side_effect = (
                ConnectionError("timeout")
            )
            emb = self.provider_cls(settings)
        with pytest.raises(RuntimeError, match="OpenAI Embedding API call failed"):
            emb.embed(["hello"])

    def test_embed_raises_on_invalid_texts(self) -> None:
        settings = Settings(
            embedding={"provider": "openai", "api_key": "sk-test"}
        )
        with patch("openai.OpenAI"):
            emb = self.provider_cls(settings)
        with pytest.raises(ValueError):
            emb.embed(["valid", 123, "also valid"])  # type: ignore[list-item]

    # ── factory integration ─────────────────────────────────────────────

    def test_factory_create_openai(self) -> None:
        settings = Settings(
            embedding={"provider": "openai", "api_key": "sk-test"}
        )
        with patch("openai.OpenAI"):
            emb = EmbeddingFactory.create(settings)
        assert isinstance(emb, self.provider_cls)


# ═══════════════════════════════════════════════════════════════════════════
# AzureEmbedding
# ═══════════════════════════════════════════════════════════════════════════


class TestAzureEmbedding:
    """Tests for the Azure OpenAI embedding provider."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        import rag_mcp_server.src.libs.embedding.azure_embedding as mod

        self.provider_cls = mod.AzureEmbedding

    def test_auto_registered_with_factory(self) -> None:
        assert EmbeddingFactory.is_registered("azure")

    def test_init_with_settings(self) -> None:
        settings = Settings(
            embedding={
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
                api_version="2024-02-15-preview",
            )

    def test_init_with_custom_api_version(self) -> None:
        settings = Settings(
            embedding={
                "provider": "azure",
                "api_key": "sk-azure",
                "endpoint": "https://example.openai.azure.com",
                "api_version": "2024-06-01",
            }
        )
        with patch("openai.AzureOpenAI") as mock_client:
            self.provider_cls(settings)
            mock_client.assert_called_once_with(
                api_key="sk-azure",
                azure_endpoint="https://example.openai.azure.com",
                api_version="2024-06-01",
            )

    def test_init_raises_without_endpoint(self) -> None:
        settings = Settings(
            embedding={"provider": "azure", "api_key": "sk-test"}
        )
        with pytest.raises(ValueError, match="endpoint not found"):
            self.provider_cls(settings)

    def test_init_raises_without_api_key(self) -> None:
        settings = Settings(
            embedding={
                "provider": "azure",
                "endpoint": "https://example.openai.azure.com",
            }
        )
        with pytest.raises(ValueError, match="API key not found"):
            self.provider_cls(settings)

    def test_init_default_model(self) -> None:
        settings = Settings(
            embedding={
                "provider": "azure",
                "api_key": "sk-azure",
                "endpoint": "https://example.openai.azure.com",
            }
        )
        with patch("openai.AzureOpenAI"):
            emb = self.provider_cls(settings)
        assert emb.model == "text-embedding-3-small"

    def test_init_dimensions_from_settings(self) -> None:
        settings = Settings(
            embedding={
                "provider": "azure",
                "api_key": "sk-azure",
                "endpoint": "https://example.openai.azure.com",
                "dimensions": 3072,
            }
        )
        with patch("openai.AzureOpenAI"):
            emb = self.provider_cls(settings)
        assert emb.default_dimensions == 3072

    def test_init_kwargs_override_endpoint(self) -> None:
        settings = Settings(
            embedding={
                "provider": "azure",
                "api_key": "sk-from-settings",
                "endpoint": "https://from-settings.openai.azure.com",
            }
        )
        with patch("openai.AzureOpenAI") as mock_client:
            self.provider_cls(
                settings,
                endpoint="https://from-kwargs.openai.azure.com",
            )
            mock_client.assert_called_once_with(
                api_key="sk-from-settings",
                azure_endpoint="https://from-kwargs.openai.azure.com",
                api_version="2024-02-15-preview",
            )

    def test_init_kwargs_override_api_version(self) -> None:
        settings = Settings(
            embedding={
                "provider": "azure",
                "api_key": "sk-azure",
                "endpoint": "https://example.openai.azure.com",
            }
        )
        with patch("openai.AzureOpenAI") as mock_client:
            self.provider_cls(settings, api_version="2025-01-01-preview")
            mock_client.assert_called_once_with(
                api_key="sk-azure",
                azure_endpoint="https://example.openai.azure.com",
                api_version="2025-01-01-preview",
            )

    # ── embed ───────────────────────────────────────────────────────────

    def test_embed_success(self) -> None:
        settings = Settings(
            embedding={
                "provider": "azure",
                "api_key": "sk-azure",
                "endpoint": "https://example.openai.azure.com",
                "model": "text-embedding-3-small-deploy",
            }
        )
        mock_resp = _make_mock_embedding_response(["Azure embed test"])
        with patch("openai.AzureOpenAI") as mock_client:
            mock_client.return_value.embeddings.create.return_value = mock_resp
            emb = self.provider_cls(settings)
        result = emb.embed(["Azure embed test"])
        assert isinstance(result, EmbeddingResponse)
        assert len(result.embeddings) == 1
        assert result.model == "mock-embedding-model"

    def test_embed_multiple_texts(self) -> None:
        settings = Settings(
            embedding={
                "provider": "azure",
                "api_key": "sk-azure",
                "endpoint": "https://example.openai.azure.com",
            }
        )
        texts = ["alpha", "beta", "gamma"]
        mock_resp = _make_mock_embedding_response(texts)
        with patch("openai.AzureOpenAI") as mock_client:
            mock_client.return_value.embeddings.create.return_value = mock_resp
            emb = self.provider_cls(settings)
        result = emb.embed(texts)
        assert len(result.embeddings) == 3

    def test_embed_empty_list(self) -> None:
        settings = Settings(
            embedding={
                "provider": "azure",
                "api_key": "sk-azure",
                "endpoint": "https://example.openai.azure.com",
            }
        )
        with patch("openai.AzureOpenAI"):
            emb = self.provider_cls(settings)
        result = emb.embed([])
        assert result.embeddings == []

    def test_embed_raises_on_api_failure(self) -> None:
        settings = Settings(
            embedding={
                "provider": "azure",
                "api_key": "sk-azure",
                "endpoint": "https://example.openai.azure.com",
            }
        )
        with patch("openai.AzureOpenAI") as mock_client:
            mock_client.return_value.embeddings.create.side_effect = (
                ConnectionError("timeout")
            )
            emb = self.provider_cls(settings)
        with pytest.raises(
            RuntimeError, match="Azure OpenAI Embedding API call failed"
        ):
            emb.embed(["hello"])

    # ── factory integration ─────────────────────────────────────────────

    def test_factory_create_azure(self) -> None:
        settings = Settings(
            embedding={
                "provider": "azure",
                "api_key": "sk-azure",
                "endpoint": "https://example.openai.azure.com",
            }
        )
        with patch("openai.AzureOpenAI"):
            emb = EmbeddingFactory.create(settings)
        assert isinstance(emb, self.provider_cls)


# ═══════════════════════════════════════════════════════════════════════════
# Factory multi-provider registration
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiProviderRegistration:
    """Both embedding providers should coexist in the factory registry."""

    def test_both_registered(self) -> None:
        providers = EmbeddingFactory.list_providers()
        assert "azure" in providers
        assert "openai" in providers

    def test_factory_resolves_correct_class_per_provider(self) -> None:
        import rag_mcp_server.src.libs.embedding.openai_embedding as oai
        import rag_mcp_server.src.libs.embedding.azure_embedding as az

        with patch("openai.OpenAI"):
            emb_o = EmbeddingFactory.create(
                Settings(embedding={"provider": "openai", "api_key": "sk"})
            )
            assert isinstance(emb_o, oai.OpenAIEmbedding)

        with patch("openai.AzureOpenAI"):
            emb_a = EmbeddingFactory.create(
                Settings(
                    embedding={
                        "provider": "azure",
                        "api_key": "sk",
                        "endpoint": "https://x.openai.azure.com",
                    }
                )
            )
            assert isinstance(emb_a, az.AzureEmbedding)
