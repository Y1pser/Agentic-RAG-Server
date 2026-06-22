"""Tests for evaluator factory routing and BaseEvaluator contract.

Covers B1.6: Evaluator abstract interface and factory.
"""

import pytest
from rag_mcp_server.src.libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
)
from rag_mcp_server.src.libs.evaluator.evaluator_factory import EvaluatorFactory
from rag_mcp_server.src.core.settings import Settings


# ---------------------------------------------------------------------------
# Fake / stub implementations for testing
# ---------------------------------------------------------------------------

class FakeEvaluator(BaseEvaluator):
    """Fake evaluator that returns a fixed high-quality score."""

    def __init__(self, settings=None, **kwargs):
        self.settings = settings
        self.default_score = kwargs.get("default_score", 0.9)

    def evaluate(self, question, contexts, answer,
                 ground_truth=None, trace=None, **kwargs):
        return EvaluationResult(
            metrics={
                "faithfulness": self.default_score,
                "relevancy": self.default_score,
                "recall": 0.85,
            },
            overall_score=self.default_score,
            details={"fake": True, "contexts_count": len(contexts)},
        )


class AnotherFakeEvaluator(BaseEvaluator):
    """A different fake — strict grader."""

    def __init__(self, settings=None, **kwargs):
        self.settings = settings

    def evaluate(self, question, contexts, answer,
                 ground_truth=None, trace=None, **kwargs):
        return EvaluationResult(
            metrics={"accuracy": 0.45, "completeness": 0.50},
            overall_score=0.475,
        )


# ---------------------------------------------------------------------------
# EvaluationResult tests
# ---------------------------------------------------------------------------

class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_defaults(self):
        """Defaults should be empty dict / 0.0."""
        result = EvaluationResult()
        assert result.metrics == {}
        assert result.overall_score == 0.0
        assert result.details == {}

    def test_full_result(self):
        """All fields should be accessible."""
        result = EvaluationResult(
            metrics={"faithfulness": 0.95, "relevancy": 0.88},
            overall_score=0.915,
            details={"model": "gpt-4o", "latency_ms": 1200},
        )
        assert result.metrics["faithfulness"] == 0.95
        assert result.overall_score == 0.915
        assert result.details["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# BaseEvaluator contract tests
# ---------------------------------------------------------------------------

class TestBaseEvaluator:
    """Verify BaseEvaluator enforces its contract."""

    def test_cannot_instantiate_abstract(self):
        """BaseEvaluator should not be directly instantiable."""
        with pytest.raises(TypeError):
            BaseEvaluator()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        """A concrete subclass should be instantiable and callable."""
        evaluator = FakeEvaluator()
        result = evaluator.evaluate(
            question="What is RAG?",
            contexts=["RAG is Retrieval-Augmented Generation."],
            answer="RAG combines retrieval with generation.",
        )
        assert isinstance(result, EvaluationResult)
        assert result.overall_score == 0.9
        assert "faithfulness" in result.metrics
        assert result.details["contexts_count"] == 1

    def test_evaluate_with_ground_truth(self):
        """Should accept an optional ground_truth parameter."""
        evaluator = FakeEvaluator()
        result = evaluator.evaluate(
            question="Q?",
            contexts=["ctx"],
            answer="ans",
            ground_truth="expected answer",
        )
        assert result.overall_score == 0.9


# ---------------------------------------------------------------------------
# EvaluatorFactory tests
# ---------------------------------------------------------------------------

class TestEvaluatorFactory:
    """Tests for evaluator factory."""

    def setup_method(self):
        EvaluatorFactory.clear_registry()

    def teardown_method(self):
        EvaluatorFactory.clear_registry()

    # -- Registration -------------------------------------------------------

    def test_register_provider_adds_to_registry(self):
        EvaluatorFactory.register_provider("fake", FakeEvaluator)
        assert "fake" in EvaluatorFactory._PROVIDERS

    def test_register_non_subclass_raises(self):
        class NotAnEvaluator:
            pass
        with pytest.raises(ValueError, match="must inherit from BaseEvaluator"):
            EvaluatorFactory.register_provider("bad", NotAnEvaluator)

    def test_register_is_case_insensitive(self):
        EvaluatorFactory.register_provider("FAKE", FakeEvaluator)
        assert EvaluatorFactory.is_registered("fake")

    # -- Creation -----------------------------------------------------------

    def test_create_returns_evaluator_instance(self):
        EvaluatorFactory.register_provider("fake", FakeEvaluator)
        settings = Settings()
        settings.evaluation = {"backends": ["fake"]}

        evaluator = EvaluatorFactory.create(settings, backend="fake")
        assert isinstance(evaluator, BaseEvaluator)
        assert isinstance(evaluator, FakeEvaluator)

    def test_create_with_override_kwargs(self):
        EvaluatorFactory.register_provider("fake", FakeEvaluator)
        settings = Settings()
        settings.evaluation = {"backends": ["fake"]}

        evaluator = EvaluatorFactory.create(settings, backend="fake", default_score=0.5)
        result = evaluator.evaluate("q?", ["c"], "a")
        assert result.overall_score == 0.5

    # -- Error handling -----------------------------------------------------

    def test_missing_backend_raises(self):
        settings = Settings()
        with pytest.raises(ValueError, match="backend"):
            EvaluatorFactory.create(settings)

    def test_unknown_provider_raises(self):
        settings = Settings()
        with pytest.raises(ValueError, match="unknown"):
            EvaluatorFactory.create(settings, backend="unknown")

    def test_instantiation_failure_wraps_error(self):
        class BrokenEvaluator(BaseEvaluator):
            def __init__(self, settings=None, **kwargs):
                raise ValueError("Invalid config")
            def evaluate(self, question, contexts, answer,
                         ground_truth=None, trace=None, **kwargs):
                return EvaluationResult()

        EvaluatorFactory.register_provider("broken", BrokenEvaluator)
        settings = Settings()

        with pytest.raises(RuntimeError, match="Failed to instantiate"):
            EvaluatorFactory.create(settings, backend="broken")

    # -- Listing & querying ------------------------------------------------

    def test_list_providers_returns_sorted_names(self):
        EvaluatorFactory.register_provider("ragas", FakeEvaluator)
        EvaluatorFactory.register_provider("custom", AnotherFakeEvaluator)
        assert EvaluatorFactory.list_providers() == ["custom", "ragas"]

    def test_is_registered_false_for_unknown(self):
        assert not EvaluatorFactory.is_registered("deep-eval")

    def test_clear_registry(self):
        EvaluatorFactory.register_provider("fake", FakeEvaluator)
        EvaluatorFactory.clear_registry()
        assert EvaluatorFactory.list_providers() == []


# ---------------------------------------------------------------------------
# Settings integration
# ---------------------------------------------------------------------------

class TestEvaluatorSettingsIntegration:
    def test_settings_has_evaluation_field(self):
        settings = Settings()
        assert hasattr(settings, "evaluation")
        assert isinstance(settings.evaluation, dict)
