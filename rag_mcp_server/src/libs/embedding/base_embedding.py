"""Abstract base class for embedding providers.

This module defines the pluggable interface for embedding model providers,
enabling seamless switching between different backends (OpenAI, Azure, etc.)
through configuration-driven instantiation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EmbeddingResponse:
    """Response from an embedding model.

    Attributes:
        embeddings: List of embedding vectors, one per input text. Each vector
            is a list of floats whose length equals the model dimension.
        model: The model identifier that generated the embeddings.
        dimensions: Number of dimensions in each embedding vector.
        usage: Optional token usage statistics (total_tokens).
    """

    embeddings: List[List[float]]
    model: str
    dimensions: Optional[int] = None
    usage: Optional[Dict[str, int]] = None

    def __post_init__(self) -> None:
        """Derive dimensions from the first embedding if not explicitly set."""
        if self.dimensions is None:
            if self.embeddings:
                self.dimensions = len(self.embeddings[0])
            else:
                self.dimensions = 0


class BaseEmbedding(ABC):
    """Abstract base class for embedding model providers.

    All embedding implementations must inherit from this class and implement
    the embed() method. This ensures a consistent interface across different
    providers (OpenAI, Azure, sentence-transformers, etc.).

    Design Principles Applied:
    - Pluggable: Subclasses can be swapped without changing upstream code.
    - Observable: Accepts optional TraceContext for observability integration.
    - Config-Driven: Instances are created via factory based on settings.
    """

    @abstractmethod
    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Generate embeddings for a list of input texts.

        Args:
            texts: List of text strings to embed.
            trace: Optional TraceContext for observability (reserved for Phase B-5).
            **kwargs: Provider-specific parameters (dimension, batch_size, etc.).

        Returns:
            EmbeddingResponse containing the embedding vectors and metadata.

        Raises:
            ValueError: If texts list is empty or contains invalid entries.
            RuntimeError: If the embedding provider call fails.
        """
        ...

    def validate_texts(self, texts: List[str]) -> None:
        """Validate the input text list.

        Args:
            texts: List of text strings to validate.

        Raises:
            ValueError: If texts list is empty or contains non-string entries.
        """
        if not isinstance(texts, list):
            raise ValueError("texts must be a list of strings")
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(
                    f"Item at index {i} is not a string (got {type(text).__name__})"
                )
