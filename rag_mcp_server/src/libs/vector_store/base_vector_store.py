"""Abstract base class for vector store providers.

This module defines the pluggable interface for vector database backends,
enabling seamless switching between different stores (Chroma, Pinecone, etc.)
through configuration-driven instantiation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VectorRecord:
    """Represents a record stored in a vector database.

    Attributes:
        embedding: The dense vector embedding for this record.
        id: A unique identifier. If None, the store should generate one.
        metadata: Optional metadata key-value pairs (source, page, etc.).
        text: Optional text content associated with this vector.
    """

    embedding: List[float]
    id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    text: Optional[str] = None


@dataclass
class QueryResult:
    """Result returned from a vector similarity query.

    Attributes:
        id: The identifier of the matching record.
        score: The similarity score (higher = more similar).
        metadata: Optional metadata from the matched record.
    """

    id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseVectorStore(ABC):
    """Abstract base class for vector database providers.

    All vector store implementations must inherit from this class and implement
    add(), get(), query(), and delete() methods. This ensures a consistent
    interface across different backends (Chroma, Pinecone, FAISS, etc.).

    Design Principles Applied:
    - Pluggable: Subclasses can be swapped without changing upstream code.
    - Observable: Each method accepts an optional TraceContext for observability.
    - Config-Driven: Instances are created via factory based on settings.
    """

    @abstractmethod
    def add(
        self,
        records: List[VectorRecord],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add vector records to the store.

        Args:
            records: List of VectorRecord objects to store. If a record's id
                is None, the store should auto-generate one.
            trace: Optional TraceContext for observability (reserved for Phase B-5).
            **kwargs: Provider-specific parameters.

        Returns:
            List of record IDs in the same order as the input records.

        Raises:
            ValueError: If records list is empty.
            RuntimeError: If the store operation fails.
        """
        ...

    @abstractmethod
    def get(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[VectorRecord]:
        """Retrieve vector records by their IDs.

        Args:
            ids: List of record identifiers to fetch.
            trace: Optional TraceContext for observability (reserved for Phase B-5).
            **kwargs: Provider-specific parameters.

        Returns:
            List of VectorRecord objects (only for IDs that exist).

        Raises:
            RuntimeError: If the store operation fails.
        """
        ...

    @abstractmethod
    def query(
        self,
        embedding: List[float],
        top_k: int = 5,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[QueryResult]:
        """Query the store for records most similar to a given embedding.

        Args:
            embedding: The query vector to compare against stored records.
            top_k: Maximum number of results to return.
            trace: Optional TraceContext for observability (reserved for Phase B-5).
            **kwargs: Provider-specific parameters (filters, distance metric, etc.).

        Returns:
            List of QueryResult objects sorted by descending similarity score.

        Raises:
            ValueError: If embedding is empty.
            RuntimeError: If the query operation fails.
        """
        ...

    @abstractmethod
    def delete(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> int:
        """Delete vector records by their IDs.

        Args:
            ids: List of record identifiers to delete.
            trace: Optional TraceContext for observability (reserved for Phase B-5).
            **kwargs: Provider-specific parameters.

        Returns:
            Number of records actually removed.

        Raises:
            RuntimeError: If the delete operation fails.
        """
        ...

    def validate_embedding(self, embedding: List[float]) -> None:
        """Validate an embedding vector.

        Args:
            embedding: The embedding vector to validate.

        Raises:
            ValueError: If embedding is empty or contains non-float values.
        """
        if not embedding:
            raise ValueError("embedding must not be empty")
        if not isinstance(embedding, list):
            raise ValueError(
                f"embedding must be a list of floats (got {type(embedding).__name__})"
            )
