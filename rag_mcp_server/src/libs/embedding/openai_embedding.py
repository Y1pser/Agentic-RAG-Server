"""OpenAI Embedding provider implementation.

Provides a standard OpenAI embeddings backend using the ``openai`` SDK.
All configuration (api_key, base_url, model, dimensions) is read from
``settings.embedding``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag_mcp_server.src.core.settings import Settings
from rag_mcp_server.src.libs.embedding.base_embedding import (
    BaseEmbedding,
    EmbeddingResponse,
)
from rag_mcp_server.src.libs.embedding.embedding_factory import EmbeddingFactory


class OpenAIEmbedding(BaseEmbedding):
    """Embedding provider backed by the standard OpenAI API.

    All configuration is read from the ``embedding`` section of
    ``settings.yaml``:

    .. code-block:: yaml

        embedding:
          provider: openai
          model: text-embedding-3-small
          api_key: "sk-..."
          base_url: "https://api.openai.com/v1"   # optional
          dimensions: 1536                          # optional

    Extra keyword arguments passed to the constructor become default
    embedding parameters (e.g. ``dimensions``).
    """

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        """Initialise the OpenAI embedding client.

        Args:
            settings: Application settings.
            **kwargs: Overrides / extra embedding defaults.  Recognised
                keys: ``model``, ``api_key``, ``base_url``, ``dimensions``.
        """
        from openai import OpenAI

        self.settings = settings

        # ── resolve model ──────────────────────────────────────────────
        self.model: str = kwargs.pop(
            "model", settings.embedding.get("model", "text-embedding-3-small")
        )

        # ── resolve credentials ─────────────────────────────────────────
        api_key: Optional[str] = (
            kwargs.pop("api_key", None)
            or settings.embedding.get("api_key")
        )
        base_url: Optional[str] = (
            kwargs.pop("base_url", None)
            or settings.embedding.get("base_url")
        )

        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Specify api_key in the embedding "
                "section of config/settings.yaml."
            )

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        # Resolve default dimensions
        self.default_dimensions: Optional[int] = kwargs.pop("dimensions", None)
        if self.default_dimensions is None:
            dims_from_settings = settings.embedding.get("dimensions")
            if dims_from_settings is not None:
                self.default_dimensions = int(dims_from_settings)

        # Extra kwargs become default embedding parameters
        self.default_params: Dict[str, Any] = kwargs
        self.client: OpenAI = OpenAI(**client_kwargs)

    # ── public interface ──────────────────────────────────────────────────

    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Generate embeddings for a list of input texts via OpenAI.

        Args:
            texts: List of text strings to embed.
            trace: Optional TraceContext for observability (reserved for Phase B-5).
            **kwargs: Provider-specific parameters (dimensions, user, etc.).

        Returns:
            EmbeddingResponse containing the embedding vectors and metadata.

        Raises:
            ValueError: If texts list is empty or contains invalid entries.
            RuntimeError: If the OpenAI API call fails.
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
            raise RuntimeError(f"OpenAI Embedding API call failed: {exc}") from exc

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
EmbeddingFactory.register_provider("openai", OpenAIEmbedding)
