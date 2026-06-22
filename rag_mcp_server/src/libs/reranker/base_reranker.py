"""Abstract base class for reranker providers.

This module defines the pluggable interface for reranking providers,
enabling the retrieval pipeline to swap between different reranking
strategies (LLM-based, cross-encoder, None pass-through) through
configuration-driven instantiation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # ScoredChunk is self-referencing only in the interface


@dataclass
class ScoredChunk:
    """A chunk with a reranking score and content.

    Attributes:
        content: The text content of the chunk.
        score: The reranking relevance score (higher = more relevant).
        metadata: Optional metadata (source document, page, original score, etc.).
    """

    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseReranker(ABC):
    """Abstract base class for reranker providers.

    All reranker implementations must inherit from this class and implement
    the rerank() method. The reranker takes a query and a list of candidate
    chunks, and returns them reordered by relevance.

    Design Principles Applied:
    - Pluggable: Subclasses can be swapped without changing upstream code.
    - Observable: Accepts optional TraceContext for observability integration.
    - Config-Driven: Instances are created via factory based on settings.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List[Any],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[ScoredChunk]:
        """Rerank candidate chunks by relevance to the query.

        Args:
            query: The search query string.
            chunks: List of candidate chunks. Each chunk must have at minimum
                a 'text' (str) and 'score' (float) attribute, plus an optional
                'metadata' (dict) attribute.
            trace: Optional TraceContext for observability (reserved for Phase B-5).
            **kwargs: Provider-specific parameters.

        Returns:
            List of ScoredChunk objects sorted by descending score.

        Raises:
            ValueError: If chunks list is empty.
            RuntimeError: If the reranking operation fails.
        """
        ...


class NoneReranker(BaseReranker):
    """Pass-through reranker that preserves original chunk order and scores.

    This is the default / fallback reranker used when no reranking backend
    is configured (rerank_backend: none). It wraps the input chunks as
    ScoredChunk objects without modifying their scores.
    """

    def __init__(self, settings: Any = None, **kwargs: Any):
        """Initialize the NoneReranker.

        Args:
            settings: Application settings (accepted for factory compatibility,
                but unused by this pass-through implementation).
            **kwargs: Additional parameters (ignored).
        """
        pass

    def rerank(
        self,
        query: str,
        chunks: List[Any],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[ScoredChunk]:
        """Return chunks unchanged with their original scores.

        Args:
            query: The search query (ignored — no reranking logic).
            chunks: Candidate chunks to pass through.
            trace: Optional TraceContext (ignored).
            **kwargs: Ignored.

        Returns:
            List of ScoredChunk wrapping each input chunk unchanged.
        """
        result = []
        for chunk in chunks:
            score = getattr(chunk, "score", 0.0)
            metadata = getattr(chunk, "metadata", {}) or {}
            result.append(ScoredChunk(
                content=getattr(chunk, "text", ""),
                score=score,
                metadata=dict(metadata),
            ))
        return result
