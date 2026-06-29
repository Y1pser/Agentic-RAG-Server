"""Cross-Encoder based Reranker implementation.

This module implements reranking using Cross-Encoder models that directly
score (query, passage) pairs via sentence-transformers. Cross-Encoders provide
more accurate relevance scores than bi-encoder or LLM-based approaches,
at the cost of higher computational load (one inference per pair, no caching).

Design Principles Applied:
- Pluggable: Swappable with other reranker implementations via factory.
- Config-Driven: Model name and parameters come from settings.yaml.
- Observable: Supports optional TraceContext for observability integration.
- Lazy Loading: sentence-transformers is imported only when a model is loaded,
  so the module can be imported without the dependency installed.
- Testable: Accepts an injected model for deterministic unit testing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from rag_mcp_server.src.libs.reranker.base_reranker import BaseReranker, ScoredChunk
from rag_mcp_server.src.libs.reranker.reranker_factory import RerankerFactory

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder based reranker for scoring query-passage pairs.

    This implementation uses Cross-Encoder models (e.g., ms-marco-MiniLM-L-6-v2
    from sentence-transformers) that directly encode and score (query, passage)
    pairs, providing more accurate relevance scores than bi-encoder approaches.

    Design Principles Applied:
    - Pluggable: Can be swapped with other reranker implementations via factory.
    - Config-Driven: Model name and parameters come from settings.yaml.
    - Observable: Supports TraceContext for monitoring (Phase B-5 integration).
    - Testable: Accepts injected model for deterministic unit testing.
    - Lazy: sentence-transformers imported only when a real model is loaded.
    """

    # Default model — small (80 MB), fast, runs on CPU
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        settings: Any,
        model: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Cross-Encoder Reranker.

        Args:
            settings: Application settings containing rerank configuration.
            model: Optional pre-initialized CrossEncoder model. If None, creates
                from ``settings.rerank.model``. Used for testing to inject mocks.
            **kwargs: Additional provider-specific parameters (reserved).

        Raises:
            CrossEncoderRerankError: If model initialization fails.
        """
        self.settings = settings

        if model is not None:
            self.model = model
        else:
            model_name = self._model_name_from_settings(settings)
            self.model = self._load_model(model_name)

    # ── config resolution ─────────────────────────────────────────────────

    def _model_name_from_settings(self, settings: Any) -> str:
        """Extract model name from settings.

        Reads ``settings.rerank.model``, falling back to the built-in default.

        Args:
            settings: Application settings.

        Returns:
            Model name string (never empty).

        Raises:
            CrossEncoderRerankError: If the resolved model name is invalid.
        """
        rerank_cfg = getattr(settings, "rerank", None) or {}
        model_name = rerank_cfg.get("model") or self.DEFAULT_MODEL

        if not model_name or not isinstance(model_name, str):
            raise CrossEncoderRerankError(
                "rerank.model must be a non-empty string, "
                f"got: {type(model_name).__name__}"
            )
        return model_name

    @staticmethod
    def _load_model(model_name: str) -> Any:
        """Load a Cross-Encoder model via sentence-transformers.

        Import is deferred so the module can be imported without
        sentence-transformers installed (e.g., in environments that
        only use the NoneReranker or LLMReranker).

        Args:
            model_name: HuggingFace model identifier or local path.

        Returns:
            Initialized ``CrossEncoder`` instance.

        Raises:
            CrossEncoderRerankError: If sentence-transformers is not installed
                or model loading fails.
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise CrossEncoderRerankError(
                "sentence-transformers is required for Cross-Encoder reranking. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        try:
            logger.info("Loading Cross-Encoder model: %s", model_name)
            model = CrossEncoder(model_name)
            logger.info("Cross-Encoder model loaded: %s", model_name)
            return model
        except Exception as exc:
            raise CrossEncoderRerankError(
                f"Failed to load Cross-Encoder model '{model_name}': {exc}"
            ) from exc

    # ── public interface ──────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        chunks: List[Any],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[ScoredChunk]:
        """Rerank candidate chunks using Cross-Encoder scoring.

        Args:
            query: The search query string.
            chunks: List of candidate chunks. Each must have a ``text`` (str)
                attribute. Optional: ``score`` (float), ``metadata`` (dict).
            trace: Optional TraceContext for observability (Phase B-5).
            **kwargs: Additional parameters:
                - ``top_k`` (int): Limit output to top N results (default: all).

        Returns:
            List of ScoredChunk sorted by Cross-Encoder relevance score
            (descending). Each chunk's ``score`` is the model prediction,
            and ``metadata`` preserves original fields plus ``original_score``
            and ``cross_encoder_score``.

        Raises:
            ValueError: If query is empty or chunks list is empty.
            CrossEncoderRerankError: If scoring fails.
        """
        # Validate inputs
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")
        if not chunks:
            raise ValueError("Candidates list cannot be empty")
        for i, chunk in enumerate(chunks):
            if not hasattr(chunk, "text"):
                raise ValueError(
                    f"Chunk at index {i} missing required attribute 'text'"
                )

        # Resolve top_k
        top_k = kwargs.get("top_k", len(chunks))
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        # Build (query, passage) pairs
        pairs: List[tuple] = [
            (query, getattr(c, "text", "")) for c in chunks
        ]

        # Score all pairs
        try:
            raw_scores = self.model.predict(
                pairs,
                show_progress_bar=False,
            )
            # Convert numpy array to list
            if hasattr(raw_scores, "tolist"):
                scores: List[float] = raw_scores.tolist()
            else:
                scores = list(raw_scores)
        except Exception as exc:
            raise CrossEncoderRerankError(
                f"Cross-Encoder scoring failed: {exc}"
            ) from exc

        # Build ScoredChunk results
        scored: List[ScoredChunk] = []
        for chunk, score in zip(chunks, scores):
            orig_score = getattr(chunk, "score", 0.0)
            orig_meta = getattr(chunk, "metadata", {}) or {}
            scored.append(
                ScoredChunk(
                    content=getattr(chunk, "text", ""),
                    score=float(score),
                    metadata={
                        **dict(orig_meta),
                        "original_score": orig_score,
                        "cross_encoder_score": float(score),
                    },
                )
            )

        # Sort by score descending, apply top_k
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]


class CrossEncoderRerankError(RuntimeError):
    """Raised when Cross-Encoder reranking fails."""


# ── auto-register with factory ─────────────────────────────────────────
RerankerFactory.register_provider("cross_encoder", CrossEncoderReranker)
