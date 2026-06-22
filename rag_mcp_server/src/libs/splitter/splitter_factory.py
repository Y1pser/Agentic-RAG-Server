"""Factory for creating text splitter provider instances.

This module implements the Factory Pattern to instantiate the appropriate
splitter provider based on configuration, enabling configuration-driven
selection of different chunking strategies without code changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rag_mcp_server.src.libs.splitter.base_splitter import BaseSplitter

if TYPE_CHECKING:
    from rag_mcp_server.src.core.settings import Settings


class SplitterFactory:
    """Factory for creating text splitter provider instances.

    This factory reads the provider configuration from settings and instantiates
    the corresponding splitter implementation. Concrete provider classes register
    themselves with the factory at import time.

    Design Principles Applied:
    - Factory Pattern: Centralizes object creation logic.
    - Config-Driven: Provider selection based on settings.yaml.
    - Fail-Fast: Raises clear errors for unknown providers.
    """

    # Registry of supported splitter providers
    # Populated by concrete implementations calling register_provider()
    _PROVIDERS: dict[str, type[BaseSplitter]] = {}

    @classmethod
    def register_provider(
        cls, name: str, provider_class: type[BaseSplitter]
    ) -> None:
        """Register a new splitter provider implementation.

        This method allows provider implementations to register themselves
        with the factory, supporting extensibility.

        Args:
            name: The provider identifier (e.g., 'recursive', 'semantic', 'fixed').
            provider_class: The BaseSplitter subclass implementing the provider.

        Raises:
            ValueError: If provider_class doesn't inherit from BaseSplitter.
        """
        if not issubclass(provider_class, BaseSplitter):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit "
                f"from BaseSplitter"
            )
        cls._PROVIDERS[name.lower()] = provider_class

    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseSplitter:
        """Create a splitter instance based on configuration.

        Args:
            settings: The application settings containing splitter configuration.
            **override_kwargs: Optional parameters to override config values.

        Returns:
            An instance of the configured splitter provider.

        Raises:
            ValueError: If the configured provider is not supported or
                splitter config is missing.
            RuntimeError: If provider instantiation fails.

        Example:
            >>> settings = load_settings('config/settings.yaml')
            >>> splitter = SplitterFactory.create(settings)
            >>> chunks = splitter.split(long_document_text)
        """
        splitter_cfg = getattr(settings, "splitter", None)
        if not splitter_cfg:
            raise ValueError(
                "Missing required configuration: splitter.provider. "
                "Please ensure 'splitter.provider' is specified in settings.yaml"
            )

        provider_name = splitter_cfg.get("provider", "")
        if not provider_name:
            raise ValueError(
                "Missing required configuration: splitter.provider. "
                "Please ensure 'splitter.provider' is specified in settings.yaml"
            )

        provider_class = cls._PROVIDERS.get(provider_name.lower())

        if provider_class is None:
            available = (
                ", ".join(sorted(cls._PROVIDERS.keys()))
                if cls._PROVIDERS
                else "none"
            )
            raise ValueError(
                f"Unsupported splitter provider: '{provider_name}'. "
                f"Available providers: {available}."
            )

        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate splitter provider '{provider_name}': {e}"
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
