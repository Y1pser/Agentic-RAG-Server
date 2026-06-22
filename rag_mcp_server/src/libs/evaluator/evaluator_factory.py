"""Factory for creating evaluator provider instances.

This module implements the Factory Pattern to instantiate the appropriate
evaluator backend based on an explicit backend parameter, enabling
configuration-driven selection of different evaluation frameworks
without code changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rag_mcp_server.src.libs.evaluator.base_evaluator import BaseEvaluator

if TYPE_CHECKING:
    from rag_mcp_server.src.core.settings import Settings


class EvaluatorFactory:
    """Factory for creating evaluator provider instances.

    This factory instantiates the appropriate evaluator based on an explicit
    backend name passed at creation time (unlike other factories that read
    the provider from settings). This allows multiple evaluators to coexist
    and be used in parallel (e.g., CompositeEvaluator).

    Design Principles Applied:
    - Factory Pattern: Centralizes object creation logic.
    - Config-Driven: Backend selection via explicit parameter + settings.
    - Fail-Fast: Raises clear errors for unknown backends.
    - Multi-Backend Friendly: Explicit backend param supports parallel
      evaluation by CompositeEvaluator.
    """

    # Registry of supported evaluator backends
    _PROVIDERS: dict[str, type[BaseEvaluator]] = {}

    @classmethod
    def register_provider(
        cls, name: str, provider_class: type[BaseEvaluator]
    ) -> None:
        """Register a new evaluator backend implementation.

        Args:
            name: The backend identifier (e.g., 'ragas', 'custom', 'truelens').
            provider_class: The BaseEvaluator subclass implementing the backend.

        Raises:
            ValueError: If provider_class doesn't inherit from BaseEvaluator.
        """
        if not issubclass(provider_class, BaseEvaluator):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit "
                f"from BaseEvaluator"
            )
        cls._PROVIDERS[name.lower()] = provider_class

    @classmethod
    def create(
        cls,
        settings: Settings,
        backend: str | None = None,
        **override_kwargs: Any,
    ) -> BaseEvaluator:
        """Create an evaluator instance for a specific backend.

        Unlike other factories that read the provider name from settings,
        EvaluatorFactory accepts an explicit backend parameter. This design
        enables the CompositeEvaluator (B7.2) to instantiate multiple
        evaluators in parallel.

        Args:
            settings: The application settings containing evaluation config.
            backend: The evaluator backend to use (e.g., 'ragas', 'custom').
                If not provided, reads from settings.evaluation.backends[0].
            **override_kwargs: Optional parameters to override config values.

        Returns:
            An instance of the requested evaluator backend.

        Raises:
            ValueError: If no backend is specified or it's not supported.
            RuntimeError: If backend instantiation fails.

        Example:
            >>> settings = load_settings('config/settings.yaml')
            >>> evaluator = EvaluatorFactory.create(settings, backend='ragas')
            >>> result = evaluator.evaluate(question, contexts, answer)
        """
        if not backend:
            # Fall back to settings if no explicit backend
            eval_cfg = getattr(settings, "evaluation", None) or {}
            backends = eval_cfg.get("backends", [])
            if not backends:
                raise ValueError(
                    "No evaluator backend specified. Pass backend=... or set "
                    "evaluation.backends in settings.yaml"
                )
            backend = backends[0]

        provider_class = cls._PROVIDERS.get(backend.lower())

        if provider_class is None:
            available = (
                ", ".join(sorted(cls._PROVIDERS.keys()))
                if cls._PROVIDERS
                else "none"
            )
            raise ValueError(
                f"Unsupported evaluator backend: '{backend}'. "
                f"Available backends: {available}."
            )

        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate evaluator backend "
                f"'{backend}': {e}"
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
