"""Factory for creating LLM provider instances.

This module implements the Factory Pattern to instantiate the appropriate
LLM provider based on configuration, enabling configuration-driven selection
of different backends without code changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rag_mcp_server.src.libs.llm.base_llm import BaseLLM

if TYPE_CHECKING:
    from rag_mcp_server.src.core.settings import Settings


class LLMFactory:
    """Factory for creating LLM provider instances.

    This factory reads the provider configuration from settings and instantiates
    the corresponding LLM implementation. Concrete provider classes register
    themselves with the factory at import time.

    Design Principles Applied:
    - Factory Pattern: Centralizes object creation logic.
    - Config-Driven: Provider selection based on settings.yaml.
    - Fail-Fast: Raises clear errors for unknown providers.
    """

    # Registry of supported LLM providers
    # Populated by concrete implementations calling register_provider()
    _PROVIDERS: dict[str, type[BaseLLM]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseLLM]) -> None:
        """Register a new LLM provider implementation.

        This method allows provider implementations to register themselves
        with the factory, supporting extensibility.

        Args:
            name: The provider identifier (e.g., 'openai', 'azure', 'ollama').
            provider_class: The BaseLLM subclass implementing the provider.

        Raises:
            ValueError: If provider_class doesn't inherit from BaseLLM.
        """
        if not issubclass(provider_class, BaseLLM):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseLLM"
            )
        cls._PROVIDERS[name.lower()] = provider_class

    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseLLM:
        """Create an LLM instance based on configuration.

        Args:
            settings: The application settings containing LLM configuration.
            **override_kwargs: Optional parameters to override config values.

        Returns:
            An instance of the configured LLM provider.

        Raises:
            ValueError: If the configured provider is not supported.
            RuntimeError: If provider instantiation fails.

        Example:
            >>> settings = load_settings('config/settings.yaml')
            >>> llm = LLMFactory.create(settings)
            >>> response = llm.chat([Message(role='user', content='Hello')])
        """
        provider_name = settings.llm.get("provider", "")
        if not provider_name:
            raise ValueError(
                "Missing required configuration: llm.provider. "
                "Please ensure 'llm.provider' is specified in settings.yaml"
            )

        provider_class = cls._PROVIDERS.get(provider_name.lower())

        if provider_class is None:
            available = (
                ", ".join(sorted(cls._PROVIDERS.keys()))
                if cls._PROVIDERS
                else "none"
            )
            raise ValueError(
                f"Unsupported LLM provider: '{provider_name}'. "
                f"Available providers: {available}."
            )

        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate LLM provider '{provider_name}': {e}"
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
