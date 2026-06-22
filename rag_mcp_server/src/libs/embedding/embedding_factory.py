"""Factory for creating embedding provider instances.

This module implements the Factory Pattern to instantiate the appropriate
embedding provider based on configuration, enabling configuration-driven
selection of different backends without code changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rag_mcp_server.src.libs.embedding.base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from rag_mcp_server.src.core.settings import Settings


class EmbeddingFactory:
    """Factory for creating embedding provider instances.

    This factory reads the provider configuration from settings and instantiates
    the corresponding embedding implementation. Concrete provider classes
    register themselves with the factory at import time.

    Design Principles Applied:
    - Factory Pattern: Centralizes object creation logic.
    - Config-Driven: Provider selection based on settings.yaml.
    - Fail-Fast: Raises clear errors for unknown providers.
    """

    # Registry of supported embedding providers
    # Populated by concrete implementations calling register_provider()
    _PROVIDERS: dict[str, type[BaseEmbedding]] = {}

    @classmethod
    def register_provider(
        cls, name: str, provider_class: type[BaseEmbedding]
    ) -> None:
        """Register a new embedding provider implementation.

        This method allows provider implementations to register themselves
        with the factory, supporting extensibility.

        Args:
            name: The provider identifier (e.g., 'openai', 'azure').
            provider_class: The BaseEmbedding subclass implementing the provider.

        Raises:
            ValueError: If provider_class doesn't inherit from BaseEmbedding.
        """
        if not issubclass(provider_class, BaseEmbedding):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit "
                f"from BaseEmbedding"
            )
        cls._PROVIDERS[name.lower()] = provider_class

    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseEmbedding:
        """Create an embedding instance based on configuration.

        Args:
            settings: The application settings containing embedding configuration.
            **override_kwargs: Optional parameters to override config values.

        Returns:
            An instance of the configured embedding provider.

        Raises:
            ValueError: If the configured provider is not supported.
            RuntimeError: If provider instantiation fails.

        Example:
            >>> settings = load_settings('config/settings.yaml')
            >>> emb = EmbeddingFactory.create(settings)
            >>> response = emb.embed(['hello', 'world'])
        """
        provider_name = settings.embedding.get("provider", "")
        if not provider_name:
            raise ValueError(
                "Missing required configuration: embedding.provider. "
                "Please ensure 'embedding.provider' is specified in settings.yaml"
            )

        provider_class = cls._PROVIDERS.get(provider_name.lower())

        if provider_class is None:
            available = (
                ", ".join(sorted(cls._PROVIDERS.keys()))
                if cls._PROVIDERS
                else "none"
            )
            raise ValueError(
                f"Unsupported embedding provider: '{provider_name}'. "
                f"Available providers: {available}."
            )

        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate embedding provider '{provider_name}': {e}"
            ) from e

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names.

        Returns:
            Sorted list of available provider identifiers.
        """
        return sorted(cls._PROVIDERS.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a provider is registered.

        Args:
            name: The provider identifier to check.

        Returns:
            True if the provider is registered, False otherwise.
        """
        return name.lower() in cls._PROVIDERS

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered providers. Useful for testing."""
        cls._PROVIDERS.clear()
