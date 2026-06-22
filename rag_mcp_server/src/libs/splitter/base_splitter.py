"""Abstract base class for document splitter providers.

This module defines the pluggable interface for text splitting providers,
enabling the ingestion pipeline to swap between different chunking strategies
(recursive, semantic, fixed-size, etc.) through configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SplitChunk:
    """Represents a single chunk of text produced by a splitter.

    Attributes:
        content: The text content of this chunk.
        index: The zero-based position of this chunk within the split output.
        metadata: Optional metadata associated with this chunk (e.g., page
            number, section heading, character offset).
    """

    content: str
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSplitter(ABC):
    """Abstract base class for text splitter providers.

    All splitter implementations must inherit from this class and implement
    the split() method. This ensures a consistent interface across different
    chunking strategies (RecursiveCharacterTextSplitter, semantic splitters,
    fixed-size splitters, etc.).

    Design Principles Applied:
    - Pluggable: Subclasses can be swapped without changing upstream code.
    - Observable: Accepts optional TraceContext for observability integration.
    - Config-Driven: Instances are created via factory based on settings.
    """

    @abstractmethod
    def split(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[SplitChunk]:
        """Split a document text into chunks.

        Args:
            text: The full text content of the document to split.
            trace: Optional TraceContext for observability (reserved for Phase B-5).
            **kwargs: Provider-specific parameters (chunk_size, chunk_overlap,
                separators, etc.).

        Returns:
            A list of SplitChunk objects representing the chunked document.

        Raises:
            ValueError: If text is empty or invalid.
            RuntimeError: If the splitting operation fails.
        """
        ...

    def validate_text(self, text: str) -> None:
        """Validate input text before splitting.

        Args:
            text: The text to validate.

        Raises:
            ValueError: If text is not a string.
        """
        if not isinstance(text, str):
            raise ValueError(
                f"text must be a string (got {type(text).__name__})"
            )
