"""Azure OpenAI Embedding provider implementation.

Uses the ``AzureOpenAI`` client from the ``openai`` SDK.  All configuration
is read from ``settings.embedding`` (no env-var fallback).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag_mcp_server.src.core.settings import Settings
from rag_mcp_server.src.libs.embedding.base_embedding import (
    BaseEmbedding,
    EmbeddingResponse,
)
from rag_mcp_server.src.libs.embedding.embedding_factory import EmbeddingFactory


class AzureEmbedding(BaseEmbedding):
    """Embedding provider backed by Azure OpenAI Service.

    All configuration is read from the ``embedding`` section of
    ``settings.yaml``:

    .. code-block:: yaml

        embedding:
          provider: azure
          model: text-embedding-3-small   # deployment name
          api_key: "your-azure-key"
          endpoint: "https://YOUR_RESOURCE.openai.azure.com"
          api_version: "2024-02-15-preview"
          dimensions: 1536                 # optional
    """

    _DEFAULT_API_VERSION = "2024-02-15-preview"

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        """Initialise the Azure OpenAI embedding client.

        Args:
            settings: Application settings.
            **kwargs: Overrides / extra embedding defaults.  Recognised
                keys: ``model``, ``api_key``, ``endpoint``, ``api_version``,
                ``dimensions``.
        """
        from openai import AzureOpenAI

        self.settings = settings

        # ── resolve model (deployment name) ─────────────────────────────
        self.model: str = kwargs.pop(
            "model", settings.embedding.get("model", "text-embedding-3-small")
        )

        # ── resolve credentials ─────────────────────────────────────────
        api_key: Optional[str] = (
            kwargs.pop("api_key", None)
            or settings.embedding.get("api_key")
        )
        endpoint: Optional[str] = (
            kwargs.pop("endpoint", None)
            or settings.embedding.get("endpoint")
        )
        api_version: str = (
            kwargs.pop("api_version", None)
            or settings.embedding.get("api_version")
            or self._DEFAULT_API_VERSION
        )

        if not api_key:
            raise ValueError(
                "Azure OpenAI API key not found. Specify api_key in the "
                "embedding section of config/settings.yaml."
            )
        if not endpoint:
            raise ValueError(
                "Azure OpenAI endpoint not found. Specify endpoint in the "
                "embedding section of config/settings.yaml."
            )

        # Resolve default dimensions
        self.default_dimensions: Optional[int] = kwargs.pop("dimensions", None)
        if self.default_dimensions is None:
            dims_from_settings = settings.embedding.get("dimensions")
            if dims_from_settings is not None:
                self.default_dimensions = int(dims_from_settings)

        self.default_params: Dict[str, Any] = kwargs
        self.client: AzureOpenAI = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )

    # ── public interface ──────────────────────────────────────────────────

    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Generate embeddings for a list of input texts via Azure OpenAI.

        Args:
            texts: List of text strings to embed.
            trace: Optional TraceContext for observability (reserved for Phase B-5).
            **kwargs: Provider-specific parameters (dimensions, user, etc.).

        Returns:
            EmbeddingResponse containing the embedding vectors and metadata.

        Raises:
            ValueError: If texts list is empty or contains invalid entries.
            RuntimeError: If the Azure OpenAI API call fails.
        """
        self.validate_texts(texts)
        if not texts:
            return EmbeddingResponse(
                embeddings=[],
                model=self.model,
                dimensions=self.default_dimensions or 0,
            )

        # Merge dimensions: kwargs > constructor default > settings default
        params: Dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "input": texts,
        }

        dims = kwargs.pop("dimensions", self.default_dimensions)
        if dims is not None:
            params["dimensions"] = dims

        params.update(self.default_params)
        params.update(kwargs)

        try:
            response = self.client.embeddings.create(**params)
        except Exception as exc:
            raise RuntimeError(
                f"Azure OpenAI Embedding API call failed: {exc}"
            ) from exc

        # Sort by index to preserve input order
        sorted_data = sorted(response.data, key=lambda d: d.index)
        embeddings = [list(d.embedding) for d in sorted_data]

        actual_dimensions = len(embeddings[0]) if embeddings else 0

        return EmbeddingResponse(
            embeddings=embeddings,
            model=response.model or self.model,
            dimensions=actual_dimensions,
            usage=(
                {"total_tokens": response.usage.total_tokens}
                if response.usage
                else None
            ),
        )


# ── auto-register with factory ─────────────────────────────────────────────
EmbeddingFactory.register_provider("azure", AzureEmbedding)
