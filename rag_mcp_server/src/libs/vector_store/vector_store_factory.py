"""Factory for creating vector store provider instances.

This module implements the Factory Pattern to instantiate the appropriate
vector database backend based on configuration, enabling configuration-driven
selection of different stores without code changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rag_mcp_server.src.libs.vector_store.base_vector_store import BaseVectorStore

if TYPE_CHECKING:
    from rag_mcp_server.src.core.settings import Settings


class VectorStoreFactory:
    """Factory for creating vector store provider instances.

    This factory reads the backend configuration from settings and instantiates
    the corresponding vector store implementation. Concrete provider classes
    register themselves with the factory at import time.

    Design Principles Applied:
    - Factory Pattern: Centralizes object creation logic.
    - Config-Driven: Backend selection based on settings.yaml.
    - Fail-Fast: Raises clear errors for unknown backends.

    Note:
        The config key for vector store is 'backend' (not 'provider'),
        matching the original DEV_SPEC convention.
    """

    # Registry of supported vector store backends
    # Populated by concrete implementations calling register_provider()
    _PROVIDERS: dict[str, type[BaseVectorStore]] = {}

    @classmethod
    def register_provider(
        cls, name: str, provider_class: type[BaseVectorStore]
    ) -> None:
        """Register a new vector store backend implementation.

        Args:
            name: The backend identifier (e.g., 'chroma', 'faiss', 'pinecone').
            provider_class: The BaseVectorStore subclass implementing the backend.

        Raises:
            ValueError: If provider_class doesn't inherit from BaseVectorStore.
        """
        if not issubclass(provider_class, BaseVectorStore):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit "
                f"from BaseVectorStore"
            )
        cls._PROVIDERS[name.lower()] = provider_class

    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseVectorStore:
        """Create a vector store instance based on configuration.

        Args:
            settings: The application settings containing vector_store configuration.
            **override_kwargs: Optional parameters to override config values.

        Returns:
            An instance of the configured vector store backend.

        Raises:
            ValueError: If the configured backend is not supported or config
                is missing.
            RuntimeError: If backend instantiation fails.

        Example:
            >>> settings = load_settings('config/settings.yaml')
            >>> store = VectorStoreFactory.create(settings)
            >>> ids = store.add([VectorRecord(embedding=[0.1, 0.2])])
        """
        vs_cfg = getattr(settings, "vector_store", None)
        if not vs_cfg:
            raise ValueError(
                "Missing required configuration: vector_store.backend. "
                "Please ensure 'vector_store.backend' is specified in settings.yaml"
            )

        backend_name = vs_cfg.get("backend", "")
        if not backend_name:
            raise ValueError(
                "Missing required configuration: vector_store.backend. "
                "Please ensure 'vector_store.backend' is specified in settings.yaml"
            )

        provider_class = cls._PROVIDERS.get(backend_name.lower())

        if provider_class is None:
            available = (
                ", ".join(sorted(cls._PROVIDERS.keys()))
                if cls._PROVIDERS
                else "none"
            )
            raise ValueError(
                f"Unsupported vector store backend: '{backend_name}'. "
                f"Available backends: {available}."
            )

        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate vector store backend "
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
