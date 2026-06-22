"""Abstract base class for RAG evaluator providers.

This module defines the pluggable interface for evaluation backends,
enabling the pipeline to assess retrieval and generation quality using
different frameworks (Ragas, custom metrics, etc.) through
configuration-driven instantiation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationResult:
    """Result of a RAG quality evaluation.

    Attributes:
        metrics: Named metric scores (e.g., faithfulness, relevancy, recall).
            Values are floats in [0.0, 1.0] range.
        overall_score: Aggregated overall quality score in [0.0, 1.0].
        details: Optional additional details (latency, model used, warnings).
    """

    metrics: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class BaseEvaluator(ABC):
    """Abstract base class for RAG evaluator providers.

    All evaluator implementations must inherit from this class and implement
    the evaluate() method. This enables pluggable evaluation backends
    (Ragas, TruLens, custom scoring, etc.) for measuring retrieval and
    generation quality.

    Design Principles Applied:
    - Pluggable: Subclasses can be swapped without changing upstream code.
    - Observable: Accepts optional TraceContext for observability integration.
    - Config-Driven: Instances are created via factory based on settings.
    """

    @abstractmethod
    def evaluate(
        self,
        question: str,
        contexts: List[str],
        answer: str,
        ground_truth: Optional[str] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate the quality of a RAG answer.

        Args:
            question: The user's original question.
            contexts: List of retrieved context strings used to generate
                the answer.
            answer: The generated answer to evaluate.
            ground_truth: Optional reference answer for comparison metrics
                (e.g., recall, exact match).
            trace: Optional TraceContext for observability (reserved for
                Phase B-5).
            **kwargs: Provider-specific parameters.

        Returns:
            EvaluationResult with metric scores, overall score, and details.

        Raises:
            ValueError: If required inputs are empty.
            RuntimeError: If the evaluation operation fails.
        """
        ...

    def validate_inputs(
        self,
        question: str,
        contexts: List[str],
        answer: str,
    ) -> None:
        """Validate evaluation inputs.

        Args:
            question: The user's question.
            contexts: Retrieved contexts.
            answer: Generated answer.

        Raises:
            ValueError: If any required input is empty.
        """
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        if not contexts:
            raise ValueError("contexts must not be empty")
        if not answer or not answer.strip():
            raise ValueError("answer must not be empty")
