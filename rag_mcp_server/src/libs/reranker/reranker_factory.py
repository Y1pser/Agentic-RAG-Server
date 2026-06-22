"""Factory for creating reranker provider instances.

This module implements the Factory Pattern to instantiate the appropriate
reranker backend based on configuration, enabling configuration-driven
selection of different reranking strategies without code changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rag_mcp_server.src.libs.reranker.base_reranker import BaseReranker, NoneReranker

if TYPE_CHECKING:
    from rag_mcp_server.src.core.settings import Settings


class RerankerFactory:
    """Factory for creating reranker provider instances.

    This factory reads the reranker backend configuration from settings and
    instantiates the corresponding reranker implementation. Concrete provider
    classes register themselves with the factory at import time.

    Special case: When rerank_backend is 'none', returns a NoneReranker
    pass-through without needing registration.

    Design Principles Applied:
    - Factory Pattern: Centralizes object creation logic.
    - Config-Driven: Backend selection based on settings.yaml.
    - Fail-Fast: Raises clear errors for unknown backends.
    - Graceful Fallback: Built-in NoneReranker for 'none' config.
    """

    # Registry of supported reranker backends
    _PROVIDERS: dict[str, type[BaseReranker]] = {}

    @classmethod
    def register_provider(
        cls, name: str, provider_class: type[BaseReranker]
    ) -> None:
        """Register a new reranker backend implementation.

        Args:
            name: The backend identifier (e.g., 'llm', 'cross-encoder').
            provider_class: The BaseReranker subclass implementing the backend.

        Raises:
            ValueError: If provider_class doesn't inherit from BaseReranker.
        """
        if not issubclass(provider_class, BaseReranker):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit "
                f"from BaseReranker"
            )
        cls._PROVIDERS[name.lower()] = provider_class

    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseReranker:
        """Create a reranker instance based on configuration.

        Reads settings.rerank.rerank_backend to determine which backend
        to instantiate. Returns a NoneReranker (pass-through) when the
        backend is 'none' or not specified.

        Args:
            settings: The application settings containing reranker config.
            **override_kwargs: Optional parameters to override config values.

        Returns:
            An instance of the configured reranker backend.

        Raises:
            ValueError: If the configured backend is not supported.
            RuntimeError: If backend instantiation fails.

        Example:
            >>> settings = load_settings('config/settings.yaml')
            >>> reranker = RerankerFactory.create(settings)
            >>> reranked = reranker.rerank(query, chunks)
        """
        rerank_cfg = getattr(settings, "rerank", None) or {}
        backend_name = rerank_cfg.get("rerank_backend", "none")

        # NoneReranker is the built-in fallback — no registration needed
        if backend_name.lower() == "none":
            return NoneReranker(settings=settings, **override_kwargs)

        provider_class = cls._PROVIDERS.get(backend_name.lower())

        if provider_class is None:
            available = (
                ", ".join(sorted(cls._PROVIDERS.keys()))
                if cls._PROVIDERS
                else "none"
            )
            raise ValueError(
                f"Unsupported reranker backend: '{backend_name}'. "
                f"Available backends: {available}."
            )

        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate reranker backend "
                f"'{backend_name}': {e}"
            ) from e

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered backend names.

        Returns:
            Sorted list of available backend identifiers.
        """
        return sorted(cls._PROVIDERS.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a backend is registered.

        Args:
            name: The backend identifier to check.

        Returns:
            True if the backend is registered, False otherwise.
        """
        return name.lower() in cls._PROVIDERS

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered backends. Useful for testing."""
        cls._PROVIDERS.clear()
