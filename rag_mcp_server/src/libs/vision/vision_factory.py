"""Factory for creating Vision LLM provider instances.

This module implements the Factory Pattern to instantiate the appropriate
Vision LLM provider based on configuration, enabling configuration-driven
selection of different backends without code changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rag_mcp_server.src.libs.vision.base_vision_llm import BaseVisionLLM

if TYPE_CHECKING:
    from rag_mcp_server.src.core.settings import Settings


class VisionLLMFactory:
    """Factory for creating Vision LLM provider instances.

    This factory reads the provider configuration from settings and instantiates
    the corresponding Vision LLM implementation. Concrete provider classes
    register themselves with the factory at import time.

    Design Principles Applied:
    - Factory Pattern: Centralizes object creation logic.
    - Config-Driven: Provider selection based on settings.yaml.
    - Fail-Fast: Raises clear errors for unknown providers.
    """

    # Registry of supported Vision LLM providers
    # Populated by concrete implementations calling register_provider()
    _PROVIDERS: dict[str, type[BaseVisionLLM]] = {}

    @classmethod
    def register_provider(
        cls, name: str, provider_class: type[BaseVisionLLM]
    ) -> None:
        """Register a new Vision LLM provider implementation.

        This method allows provider implementations to register themselves
        with the factory, supporting extensibility.

        Args:
            name: The provider identifier (e.g., 'azure', 'openai').
            provider_class: The BaseVisionLLM subclass implementing the provider.

        Raises:
            ValueError: If provider_class doesn't inherit from BaseVisionLLM.
        """
        if not issubclass(provider_class, BaseVisionLLM):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit "
                f"from BaseVisionLLM"
            )
        cls._PROVIDERS[name.lower()] = provider_class

    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseVisionLLM:
        """Create a Vision LLM instance based on configuration.

        Reads ``vision.provider`` from settings to determine which backend
        to instantiate.

        Args:
            settings: The application settings containing vision configuration.
            **override_kwargs: Optional parameters to override config values.

        Returns:
            An instance of the configured Vision LLM provider.

        Raises:
            ValueError: If the configured provider is not supported.
            RuntimeError: If provider instantiation fails.

        Example:
            >>> settings = load_settings('config/settings.yaml')
            >>> vision_llm = VisionLLMFactory.create(settings)
            >>> description = vision_llm.describe('path/to/image.png')
        """
        # Vision LLM lives under the 'vision' config key; fall back to 'llm'
        # if no dedicated vision section exists.
        vision_config = getattr(settings, "vision", None)
        if vision_config and vision_config.get("provider"):
            provider_name = vision_config["provider"]
        else:
            # Fallback: use same provider as text LLM (but with vision model)
            provider_name = settings.llm.get("vision_provider", "")

        if not provider_name:
            raise ValueError(
                "Missing required configuration: vision.provider. "
                "Please ensure 'vision.provider' is specified in settings.yaml "
                "or 'llm.vision_provider' is set as a fallback."
            )

        provider_class = cls._PROVIDERS.get(provider_name.lower())

        if provider_class is None:
            available = (
                ", ".join(sorted(cls._PROVIDERS.keys()))
                if cls._PROVIDERS
                else "none"
            )
            raise ValueError(
                f"Unsupported Vision LLM provider: '{provider_name}'. "
                f"Available providers: {available}."
            )

        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Vision LLM provider "
                f"'{provider_name}': {e}"
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
