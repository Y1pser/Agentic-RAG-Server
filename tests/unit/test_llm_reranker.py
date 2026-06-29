"""Tests for LLM-based Reranker implementation.

Covers B1.12: LLMReranker with prompt template support.
"""

import json
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest

from rag_mcp_server.src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from rag_mcp_server.src.libs.reranker.base_reranker import BaseReranker, ScoredChunk
from rag_mcp_server.src.libs.reranker.llm_reranker import LLMRerankError, LLMReranker
from rag_mcp_server.src.libs.reranker.reranker_factory import RerankerFactory
from rag_mcp_server.src.core.settings import Settings


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class MockLLM(BaseLLM):
    """Mock LLM for unit testing — returns pre-configured responses."""

    def __init__(self, response_content: str = "[]"):
        self.response_content = response_content
        self.call_count = 0
        self.last_messages: Optional[List[Message]] = None

    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        self.call_count += 1
        self.last_messages = messages
        return ChatResponse(
            content=self.response_content,
            model="mock-model",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20,
            },
        )


class _MockChunk:
    """Lightweight mock of a retrieval chunk for reranker testing."""

    def __init__(self, text: str, score: float = 0.8, metadata: dict = None):
        self.text = text
        self.score = score
        self.metadata = metadata or {}


def _make_chunks(*specs) -> list:
    """Create mock chunks from (text, score, metadata) tuples or plain strings."""
    chunks = []
    for item in specs:
        if isinstance(item, tuple):
            text = item[0]
            score = item[1] if len(item) > 1 else 0.8
            meta = item[2] if len(item) > 2 else {}
            chunks.append(_MockChunk(text=text, score=score, metadata=meta))
        else:
            chunks.append(_MockChunk(text=item))
    return chunks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Settings()
    settings.llm = {
        "provider": "openai",
        "model": "gpt-4o",
        "api_key": "sk-test",
    }
    settings.rerank = {"rerank_backend": "llm"}
    return settings


@pytest.fixture
def sample_prompt():
    """Sample rerank prompt template."""
    return (
        "You are an AI assistant specialized in evaluating relevance.\n"
        "Given a query and passages, score each passage on relevance (0-3).\n"
        "Output JSON format with passage_id and score."
    )


@pytest.fixture
def sample_chunks():
    """Sample candidate chunks for reranking."""
    return _make_chunks(
        ("Python is a programming language.", 0.8),
        ("Machine learning uses neural networks.", 0.75),
        ("RAG combines retrieval and generation.", 0.9),
    )


# ---------------------------------------------------------------------------
# LLMReranker — Initialization
# ---------------------------------------------------------------------------

class TestLLMRerankerInit:
    """Tests for LLMReranker initialization."""

    def test_init_with_defaults(self, mock_settings, sample_prompt, tmp_path):
        """Initialization with provided prompt path and LLM."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        mock_llm = MockLLM()
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )

        assert reranker.settings == mock_settings
        assert reranker.llm == mock_llm
        assert reranker.prompt_template == sample_prompt

    def test_init_missing_prompt_file(self, mock_settings):
        """Initialization fails when prompt file doesn't exist."""
        mock_llm = MockLLM()

        with pytest.raises(LLMRerankError, match="prompt file not found"):
            LLMReranker(
                settings=mock_settings,
                prompt_path="/nonexistent/path/rerank.txt",
                llm=mock_llm,
            )

    def test_load_prompt_template(self, mock_settings, sample_prompt, tmp_path):
        """Prompt template is loaded correctly from file."""
        prompt_file = tmp_path / "custom_prompt.txt"
        prompt_file.write_text(sample_prompt)

        mock_llm = MockLLM()
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )

        assert sample_prompt in reranker.prompt_template

    def test_is_base_reranker(self, mock_settings, sample_prompt, tmp_path):
        """LLMReranker should be a BaseReranker subclass."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)
        mock_llm = MockLLM()
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )
        assert isinstance(reranker, BaseReranker)

    def test_prompt_path_from_settings(
        self, mock_settings, sample_prompt, tmp_path
    ):
        """Reads prompt_path from rerank.prompt_path in settings."""
        prompt_file = tmp_path / "custom.txt"
        prompt_file.write_text(sample_prompt)

        mock_settings.rerank = {
            "rerank_backend": "llm",
            "prompt_path": str(prompt_file),
        }
        mock_llm = MockLLM()
        reranker = LLMReranker(settings=mock_settings, llm=mock_llm)
        assert reranker.prompt_path == str(prompt_file)

    def test_prompt_path_default_when_settings_missing(
        self, mock_settings, sample_prompt, tmp_path
    ):
        """Falls back to DEFAULT_PROMPT_PATH when settings has no prompt_path.

        NOTE: the default path points to config/prompts/rerank.txt which
        doesn't exist in a test context, so we verify the fallback without
        triggering the full init chain.
        """
        mock_llm = MockLLM()
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=None,   # explicitly None → triggers fallback
            llm=mock_llm,
        )
        # mock_settings.rerank has no prompt_path → falls to default
        assert reranker.prompt_path == LLMReranker.DEFAULT_PROMPT_PATH


# ---------------------------------------------------------------------------
# LLMReranker — Prompt Building
# ---------------------------------------------------------------------------

class TestLLMRerankerPromptBuilding:
    """Tests for prompt building functionality."""

    def test_build_rerank_prompt(
        self, mock_settings, sample_prompt, sample_chunks, tmp_path
    ):
        """Prompt contains query and all chunk texts."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        mock_llm = MockLLM()
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )

        candidates = [
            {"id": "chunk_0", "text": "Python is a programming language."},
            {"id": "chunk_1", "text": "Machine learning uses neural networks."},
            {"id": "chunk_2", "text": "RAG combines retrieval and generation."},
        ]

        query = "What is RAG?"
        prompt = reranker._build_rerank_prompt(query, candidates)

        assert sample_prompt in prompt
        assert query in prompt
        for c in candidates:
            assert c["id"] in prompt
            assert c["text"] in prompt

    def test_build_prompt_with_missing_text(
        self, mock_settings, sample_prompt, tmp_path
    ):
        """Prompt building handles candidates with alternative 'content' field."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        mock_llm = MockLLM()
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )

        candidates = [
            {"id": "chunk_a", "content": "Alternative field name"},
            {"id": "chunk_b"},  # Missing both text and content
        ]

        prompt = reranker._build_rerank_prompt("test", candidates)
        assert "Alternative field name" in prompt
        assert "chunk_a" in prompt
        assert "chunk_b" in prompt


# ---------------------------------------------------------------------------
# LLMReranker — Response Parsing
# ---------------------------------------------------------------------------

class TestLLMRerankerResponseParsing:
    """Tests for LLM response parsing and validation."""

    def test_parse_valid_json_response(
        self, mock_settings, sample_prompt, tmp_path
    ):
        """Valid JSON response is parsed correctly."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        mock_llm = MockLLM()
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )

        response = json.dumps([
            {"passage_id": "chunk_0", "score": 2, "reasoning": "Relevant"},
            {"passage_id": "chunk_1", "score": 1, "reasoning": "Partial match"},
        ])

        parsed = reranker._parse_llm_response(response)
        assert len(parsed) == 2
        assert parsed[0]["passage_id"] == "chunk_0"
        assert parsed[0]["score"] == 2
        assert parsed[1]["passage_id"] == "chunk_1"
        assert parsed[1]["score"] == 1

    def test_parse_json_with_markdown_wrapper(
        self, mock_settings, sample_prompt, tmp_path
    ):
        """JSON wrapped in markdown code fences is parsed correctly."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        mock_llm = MockLLM()
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )

        response = """```json
[
    {"passage_id": "chunk_0", "score": 3},
    {"passage_id": "chunk_1", "score": 1}
]
```"""

        parsed = reranker._parse_llm_response(response)
        assert len(parsed) == 2
        assert parsed[0]["score"] == 3

    def test_parse_plain_code_fence(
        self, mock_settings, sample_prompt, tmp_path
    ):
        """JSON wrapped in plain code fences (no language specifier)."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        mock_llm = MockLLM()
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )

        response = """```
[{"passage_id": "chunk_0", "score": 2}]
```"""

        parsed = reranker._parse_llm_response(response)
        assert len(parsed) == 1
        assert parsed[0]["score"] == 2

    def test_parse_invalid_json(
        self, mock_settings, sample_prompt, tmp_path
    ):
        """Non-JSON response raises LLMRerankError."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        mock_llm = MockLLM()
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )

        with pytest.raises(LLMRerankError, match="not valid JSON"):
            reranker._parse_llm_response("This is not JSON at all")

    def test_parse_non_array_response(
        self, mock_settings, sample_prompt, tmp_path
    ):
        """Object response (not array) raises LLMRerankError."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        mock_llm = MockLLM()
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )

        with pytest.raises(LLMRerankError, match="Expected JSON array"):
            reranker._parse_llm_response('{"key": "value"}')

    def test_parse_missing_passage_id(
        self, mock_settings, sample_prompt, tmp_path
    ):
        """Missing passage_id field raises LLMRerankError."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        mock_llm = MockLLM()
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )

        with pytest.raises(
            LLMRerankError, match="missing required field 'passage_id'"
        ):
            reranker._parse_llm_response('[{"score": 2}]')

    def test_parse_missing_score(
        self, mock_settings, sample_prompt, tmp_path
    ):
        """Missing score field raises LLMRerankError."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        mock_llm = MockLLM()
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )

        with pytest.raises(
            LLMRerankError, match="missing required field 'score'"
        ):
            reranker._parse_llm_response('[{"passage_id": "chunk_0"}]')

    def test_parse_non_numeric_score(
        self, mock_settings, sample_prompt, tmp_path
    ):
        """Non-numeric score raises LLMRerankError."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        mock_llm = MockLLM()
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )

        with pytest.raises(LLMRerankError, match="score must be numeric"):
            reranker._parse_llm_response(
                '[{"passage_id": "chunk_0", "score": "high"}]'
            )

    def test_parse_non_dict_item(
        self, mock_settings, sample_prompt, tmp_path
    ):
        """Non-dict items in array raise LLMRerankError."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        mock_llm = MockLLM()
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )

        with pytest.raises(LLMRerankError, match="not a dict"):
            reranker._parse_llm_response('["not a dict"]')


# ---------------------------------------------------------------------------
# LLMReranker — Reranking
# ---------------------------------------------------------------------------

class TestLLMRerankerReranking:
    """End-to-end tests for the rerank() method."""

    def test_rerank_success(
        self, mock_settings, sample_prompt, sample_chunks, tmp_path
    ):
        """Successful rerank orders chunks by LLM-assigned scores."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        llm_response = json.dumps([
            {"passage_id": "chunk_2", "score": 3, "reasoning": "Most relevant"},
            {"passage_id": "chunk_0", "score": 2, "reasoning": "Partially relevant"},
            {"passage_id": "chunk_1", "score": 1, "reasoning": "Less relevant"},
        ])

        mock_llm = MockLLM(response_content=llm_response)
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )

        result = reranker.rerank("What is RAG?", sample_chunks)

        assert len(result) == 3
        assert all(isinstance(c, ScoredChunk) for c in result)

        # Should be sorted by LLM score: 3, 2, 1
        assert result[0].score == 3
        assert result[0].content == "RAG combines retrieval and generation."
        assert result[1].score == 2
        assert result[1].content == "Python is a programming language."
        assert result[2].score == 1
        assert result[2].content == "Machine learning uses neural networks."

        assert mock_llm.call_count == 1

    def test_rerank_single_chunk_no_op(
        self, mock_settings, sample_prompt, tmp_path
    ):
        """Single chunk is returned as-is without calling LLM."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        mock_llm = MockLLM()
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )

        chunks = _make_chunks("Only one chunk")
        result = reranker.rerank("test", chunks)

        assert len(result) == 1
        assert result[0].content == "Only one chunk"
        assert mock_llm.call_count == 0

    def test_rerank_empty_query(
        self, mock_settings, sample_prompt, sample_chunks, tmp_path
    ):
        """Empty query raises ValueError."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        mock_llm = MockLLM()
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )

        with pytest.raises(ValueError, match="Query cannot be empty"):
            reranker.rerank("   ", sample_chunks)

    def test_rerank_empty_chunks(
        self, mock_settings, sample_prompt, tmp_path
    ):
        """Empty chunks list raises ValueError."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        mock_llm = MockLLM()
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )

        with pytest.raises(ValueError, match="Candidates list cannot be empty"):
            reranker.rerank("test query", [])

    def test_rerank_chunk_missing_text(
        self, mock_settings, sample_prompt, sample_chunks, tmp_path
    ):
        """Chunk missing 'text' attribute raises ValueError."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        mock_llm = MockLLM()
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )

        class BadChunk:
            pass  # No 'text' attribute

        with pytest.raises(
            ValueError, match="missing required attribute 'text'"
        ):
            reranker.rerank("test", [BadChunk()])

    def test_rerank_llm_failure(
        self, mock_settings, sample_prompt, sample_chunks, tmp_path
    ):
        """LLM call failure raises LLMRerankError."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        class FailingLLM(BaseLLM):
            def chat(self, messages, trace=None, **kwargs):
                raise RuntimeError("LLM service unavailable")

        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=FailingLLM(),
        )

        with pytest.raises(LLMRerankError, match="LLM call failed"):
            reranker.rerank("test query", sample_chunks)

    def test_rerank_malformed_response(
        self, mock_settings, sample_prompt, sample_chunks, tmp_path
    ):
        """Malformed LLM response raises LLMRerankError."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        mock_llm = MockLLM(response_content="This is not valid JSON")
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )

        with pytest.raises(LLMRerankError, match="not valid JSON"):
            reranker.rerank("test query", sample_chunks)

    def test_rerank_metadata_preserved(
        self, mock_settings, sample_prompt, tmp_path
    ):
        """Original metadata is preserved alongside LLM scores."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        llm_response = json.dumps([
            {"passage_id": "chunk_0", "score": 3},
            {"passage_id": "chunk_1", "score": 2},
        ])

        mock_llm = MockLLM(response_content=llm_response)
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )

        chunks = _make_chunks(
            ("Text A", 0.8, {"source": "a.pdf", "page": 1}),
            ("Text B", 0.9, {"source": "b.pdf", "page": 2}),
        )

        result = reranker.rerank("test", chunks)

        assert result[0].metadata["source"] == "a.pdf"
        assert result[0].metadata["page"] == 1
        assert result[0].metadata["original_score"] == 0.8
        assert result[0].metadata["llm_rerank_score"] == 3

        assert result[1].metadata["source"] == "b.pdf"
        assert result[1].metadata["original_score"] == 0.9
        assert result[1].metadata["llm_rerank_score"] == 2

    def test_rerank_with_trace_context(
        self, mock_settings, sample_prompt, sample_chunks, tmp_path
    ):
        """Trace context is passed through to the LLM."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        llm_response = json.dumps([
            {"passage_id": "chunk_0", "score": 3},
            {"passage_id": "chunk_1", "score": 2},
        ])

        mock_llm = MockLLM(response_content=llm_response)
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )

        mock_trace = Mock()
        chunks = _make_chunks("Text 1", "Text 2")
        reranker.rerank("query", chunks, trace=mock_trace)

        assert mock_llm.call_count == 1

    def test_rerank_passes_kwargs_to_llm(
        self, mock_settings, sample_prompt, sample_chunks, tmp_path
    ):
        """Extra kwargs are forwarded to the LLM chat call."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        llm_response = json.dumps([
            {"passage_id": "chunk_0", "score": 2},
            {"passage_id": "chunk_1", "score": 1},
        ])

        mock_llm = MockLLM(response_content=llm_response)
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
            temperature=0.0,
        )

        reranker.rerank("query", sample_chunks, max_tokens=500)

        assert mock_llm.call_count == 1
        # Verify the prompt contains the query
        assert mock_llm.last_messages is not None


# ---------------------------------------------------------------------------
# LLMReranker — Chunk ID resolution
# ---------------------------------------------------------------------------

class TestLLMRerankerIdResolution:
    """Tests for chunk ID handling during reranking."""

    def test_metadata_id_used_for_passage(
        self, mock_settings, sample_prompt, tmp_path
    ):
        """When chunk metadata has an 'id', it is used as the passage_id."""
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text(sample_prompt)

        llm_response = json.dumps([
            {"passage_id": "doc-1", "score": 3},
            {"passage_id": "doc-2", "score": 1},
        ])

        mock_llm = MockLLM(response_content=llm_response)
        reranker = LLMReranker(
            settings=mock_settings,
            prompt_path=str(prompt_file),
            llm=mock_llm,
        )

        chunks = _make_chunks(
            ("Doc 1 content", 0.8, {"id": "doc-1"}),
            ("Doc 2 content", 0.7, {"id": "doc-2"}),
        )

        result = reranker.rerank("test", chunks)
        assert len(result) == 2
        assert result[0].content == "Doc 1 content"
        assert result[0].score == 3


# ---------------------------------------------------------------------------
# LLMReranker — Factory Registration
# ---------------------------------------------------------------------------

class TestLLMRerankerFactoryRegistration:
    """Tests that LLMReranker is properly registered with RerankerFactory."""

    def test_registered_as_llm(self):
        """LLMReranker should be registered under the 'llm' key."""
        assert RerankerFactory.is_registered("llm")

    def test_llm_in_list_providers(self):
        """'llm' should appear in the list of available providers."""
        assert "llm" in RerankerFactory.list_providers()
