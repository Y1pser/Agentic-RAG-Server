"""LLM-based Reranker implementation.

This module implements reranking using Large Language Models to evaluate
the relevance of candidate passages to a given query. It reads prompts from
a configurable file and structures LLM outputs for downstream processing.

Design Principles Applied:
- Pluggable: Swappable with other reranker implementations via factory.
- Config-Driven: Prompt file path and LLM settings come from configuration.
- Observable: Supports optional TraceContext for observability integration.
- Structured Output: Validates LLM output against expected JSON schema.
- Fail-Fast: Raises clear errors for malformed responses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_mcp_server.src.libs.llm.base_llm import Message
from rag_mcp_server.src.libs.llm.llm_factory import LLMFactory
from rag_mcp_server.src.libs.reranker.base_reranker import BaseReranker, ScoredChunk
from rag_mcp_server.src.libs.reranker.reranker_factory import RerankerFactory


class LLMReranker(BaseReranker):
    """LLM-based reranker using structured prompts.

    This implementation leverages an LLM to score and rerank candidate chunks
    based on their relevance to a query. It reads the reranking prompt from
    a configurable file and expects structured JSON output from the LLM.

    Design Principles Applied:
    - Pluggable: Can be swapped with other reranker implementations via factory.
    - Config-Driven: Prompt file path and LLM settings come from configuration.
    - Observable: Supports TraceContext for monitoring (Phase B-5 integration).
    - Structured Output: Validates LLM output against expected schema.
    - Fail-Fast: Raises clear LLMRerankError for invalid responses.
    """

    # Configuration key paths
    DEFAULT_PROMPT_PATH = "config/prompts/rerank.txt"

    def __init__(
        self,
        settings: Any,
        prompt_path: Optional[str] = None,
        llm: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the LLM Reranker.

        Args:
            settings: Application settings containing LLM and rerank configuration.
            prompt_path: Optional path to rerank prompt file. Resolution order:
                explicit arg > ``settings.rerank.llm_rerank.prompt_path`` >
                ``config/prompts/rerank.txt`` (default). Injectable for testing.
            llm: Optional LLM instance. If None, creates via LLMFactory from settings.
                Injectable for testing with mock LLMs.
            **kwargs: Additional provider-specific parameters passed through
                to the LLM chat call (e.g., temperature, max_tokens).

        Raises:
            LLMRerankError: If the prompt file cannot be loaded.
        """
        self.settings = settings
        self.llm = llm or LLMFactory.create(settings)

        # Resolve prompt_path: explicit arg > settings > default
        if prompt_path:
            self.prompt_path = prompt_path
        else:
            rerank_cfg = getattr(settings, "rerank", None) or {}
            self.prompt_path = (
                rerank_cfg.get("prompt_path")
                or self.DEFAULT_PROMPT_PATH
            )
        self.default_kwargs = kwargs

        # Load prompt template
        self.prompt_template = self._load_prompt_template(self.prompt_path)

    # ── prompt loading ────────────────────────────────────────────────────

    def _load_prompt_template(self, path: str) -> str:
        """Load the rerank prompt template from a file.

        Args:
            path: Path to the prompt template file.

        Returns:
            The prompt template as a string.

        Raises:
            LLMRerankError: If the prompt file cannot be found or read.
        """
        prompt_file = Path(path)
        if not prompt_file.exists():
            raise LLMRerankError(
                f"Rerank prompt file not found: {path}"
            )
        try:
            return prompt_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise LLMRerankError(
                f"Failed to read rerank prompt file {path}: {exc}"
            ) from exc

    # ── prompt building ───────────────────────────────────────────────────

    def _build_rerank_prompt(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> str:
        """Build the reranking prompt with query and candidate texts.

        Args:
            query: The user query string.
            candidates: List of candidate records to rerank. Each is a dict
                with ``id`` and ``text`` keys.

        Returns:
            Formatted prompt string ready for the LLM.
        """
        candidates_text = []
        for i, candidate in enumerate(candidates):
            passage_id = candidate.get("id", f"passage_{i}")
            text = candidate.get("text", candidate.get("content", ""))
            candidates_text.append(f"Passage ID: {passage_id}\nText: {text}\n")

        candidates_str = "\n".join(candidates_text)

        return (
            f"{self.prompt_template}\n\n"
            f"Query: {query}\n\n"
            f"Passages:\n{candidates_str}\n\n"
            "Output your response as a JSON array of objects, one per passage."
        )

    # ── response parsing ──────────────────────────────────────────────────

    def _parse_llm_response(
        self, response_text: str
    ) -> List[Dict[str, Any]]:
        """Parse and validate the LLM response.

        Expects a JSON array of objects, each with ``passage_id`` and
        ``score`` fields.  Handles markdown-wrapped JSON.

        Args:
            response_text: Raw text response from the LLM.

        Returns:
            List of parsed ranking records.

        Raises:
            LLMRerankError: If the response is not valid JSON, not an array,
                or has missing/invalid fields.
        """
        text = response_text.strip()

        # Strip markdown code fences
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Parse JSON
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMRerankError(
                f"LLM response is not valid JSON: {exc}\n"
                f"Response: {response_text[:200]}"
            ) from exc

        # Validate top-level structure
        if not isinstance(parsed, list):
            raise LLMRerankError(
                f"Expected JSON array, got {type(parsed).__name__}. "
                f"Response: {response_text[:200]}"
            )

        # Validate each item
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise LLMRerankError(
                    f"Item {i} is not a dict (type: {type(item).__name__})"
                )
            if "passage_id" not in item:
                raise LLMRerankError(
                    f"Item {i} missing required field 'passage_id'"
                )
            if "score" not in item:
                raise LLMRerankError(
                    f"Item {i} missing required field 'score'"
                )

            score = item["score"]
            if not isinstance(score, (int, float)):
                raise LLMRerankError(
                    f"Item {i} score must be numeric, got "
                    f"{type(score).__name__}: {score}"
                )

        return parsed

    # ── public interface ──────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        chunks: List[Any],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[ScoredChunk]:
        """Rerank candidate chunks by relevance to the query using an LLM.

        Args:
            query: The search query string.
            chunks: List of candidate chunks. Each chunk must have at minimum
                a ``text`` (str) attribute.  Optional: ``score`` (float),
                ``metadata`` (dict).
            trace: Optional TraceContext for observability (Phase B-5).
            **kwargs: Generation parameters forwarded to the LLM chat call
                (e.g., temperature, max_tokens).

        Returns:
            List of ScoredChunk sorted by LLM-assigned relevance score descending.

        Raises:
            ValueError: If query is empty or chunks list is empty.
            LLMRerankError: If the LLM call fails or the response is malformed.
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

        # Single chunk — no need to call LLM
        if len(chunks) == 1:
            chunk = chunks[0]
            return [
                ScoredChunk(
                    content=getattr(chunk, "text", ""),
                    score=getattr(chunk, "score", 0.0),
                    metadata=dict(getattr(chunk, "metadata", {}) or {}),
                )
            ]

        # Build candidate dicts from chunk objects
        candidates: List[Dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            chunk_id = (
                getattr(chunk, "metadata", None)
                and chunk.metadata.get("id")
            ) or f"chunk_{i}"
            candidates.append({
                "id": chunk_id,
                "text": getattr(chunk, "text", ""),
            })

        # Build prompt and call LLM
        try:
            prompt = self._build_rerank_prompt(query, candidates)
        except Exception as exc:
            raise LLMRerankError(
                f"Failed to build rerank prompt: {exc}"
            ) from exc

        try:
            messages = [Message(role="user", content=prompt)]
            merged_kwargs = {**self.default_kwargs, **kwargs}
            response = self.llm.chat(messages, trace=trace, **merged_kwargs)
            response_text = response.content
        except LLMRerankError:
            raise
        except Exception as exc:
            raise LLMRerankError(
                f"LLM call failed during reranking: {exc}"
            ) from exc

        # Parse LLM response
        try:
            parsed_results = self._parse_llm_response(response_text)
        except LLMRerankError:
            raise
        except Exception as exc:
            raise LLMRerankError(
                f"Failed to parse LLM rerank response: {exc}"
            ) from exc

        # Map results back to ScoredChunk
        try:
            reranked = self._map_to_scored_chunks(
                parsed_results, candidates, chunks
            )
        except Exception as exc:
            raise LLMRerankError(
                f"Failed to map LLM results to chunks: {exc}"
            ) from exc

        return reranked

    # ── result mapping ────────────────────────────────────────────────────

    def _map_to_scored_chunks(
        self,
        parsed_results: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        chunks: List[Any],
    ) -> List[ScoredChunk]:
        """Map LLM-scored results back to ScoredChunk objects.

        Args:
            parsed_results: Parsed LLM output with ``passage_id`` and ``score``.
            candidates: Candidate dicts used to build the prompt.
            chunks: Original chunk objects.

        Returns:
            List of ScoredChunk sorted by LLM score descending.
        """
        # Build id → index mapping
        id_to_index: Dict[str, int] = {}
        for i, candidate in enumerate(candidates):
            passage_id = candidate.get("id", f"chunk_{i}")
            id_to_index[passage_id] = i

        # Match LLM results to original chunks
        scored: List[ScoredChunk] = []
        for result in parsed_results:
            passage_id = result["passage_id"]
            llm_score = float(result["score"])

            if passage_id in id_to_index:
                idx = id_to_index[passage_id]
                chunk = chunks[idx]
                scored.append(
                    ScoredChunk(
                        content=getattr(chunk, "text", ""),
                        score=llm_score,
                        metadata=dict(
                            getattr(chunk, "metadata", {}) or {},
                            **{
                                "original_score": getattr(chunk, "score", 0.0),
                                "llm_rerank_score": llm_score,
                            },
                        ),
                    )
                )

        # Sort by LLM score descending
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored


class LLMRerankError(RuntimeError):
    """Raised when LLM reranking fails."""


# ── auto-register with factory ─────────────────────────────────────────
RerankerFactory.register_provider("llm", LLMReranker)
