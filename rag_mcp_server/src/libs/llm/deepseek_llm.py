"""DeepSeek LLM provider implementation.

DeepSeek exposes an OpenAI-compatible API, so this provider re-uses the
standard ``openai.OpenAI`` client pointed at ``https://api.deepseek.com/v1``
(or a custom base URL from config).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from rag_mcp_server.src.core.settings import Settings
from rag_mcp_server.src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from rag_mcp_server.src.libs.llm.llm_factory import LLMFactory

# Default base URL for the DeepSeek API.
_DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


class DeepSeekLLM(BaseLLM):
    """LLM provider backed by the DeepSeek API.

    Because DeepSeek is OpenAI-compatible, this implementation is a thin
    wrapper around ``openai.OpenAI`` with a pre-configured base URL.
    API key resolution order:

    1. ``api_key`` in ``settings.llm`` or constructor kwargs
    2. ``DEEPSEEK_API_KEY`` environment variable
    3. ``OPENAI_API_KEY`` environment variable (shared fallback)

    Design Principles Applied
    -------------------------
    - **Pluggable** – drop-in replacement for any other BaseLLM.
    - **Config-Driven** – base URL and credentials come from ``Settings``
      or env vars.
    - **Observable** – trace context argument reserved for Phase B-5.
    """

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        """Initialise the DeepSeek client.

        Args:
            settings: Application settings.  Expected keys in
                ``settings.llm``: ``model``, ``api_key``, ``base_url``
                (all optional when env vars are set).
            **kwargs: Overrides / extra generation defaults.  Recognised
                keys: ``model``, ``api_key``, ``base_url``.  Everything
                else is treated as a default generation parameter.
        """
        from openai import OpenAI

        self.settings = settings

        # ── resolve model ──────────────────────────────────────────
        self.model: str = kwargs.pop(
            "model", settings.llm.get("model", "deepseek-chat")
        )

        # ── resolve credentials ─────────────────────────────────────
        api_key: Optional[str] = (
            kwargs.pop("api_key", None)
            or settings.llm.get("api_key")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        base_url: Optional[str] = (
            kwargs.pop("base_url", None)
            or settings.llm.get("base_url")
            or os.getenv("OPENAI_BASE_URL")
            or _DEFAULT_BASE_URL
        )

        if not api_key:
            raise ValueError(
                "DeepSeek API key not found. Set DEEPSEEK_API_KEY or "
                "OPENAI_API_KEY in .env, or specify api_key in the llm "
                "section of settings.yaml."
            )

        # Extra kwargs become default generation parameters
        self.default_params: Dict[str, Any] = kwargs
        self.client: OpenAI = OpenAI(api_key=api_key, base_url=base_url)

    # ── public interface ──────────────────────────────────────────────

    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a conversation to DeepSeek and return the response.

        Args:
            messages: Conversation history (system / user / assistant).
            trace: Reserved for observability (Phase B-5).
            **kwargs: Per-call overrides merged on top of ``default_params``.

        Returns:
            ``ChatResponse`` with the completion content and usage metadata.

        Raises:
            ValueError: If ``messages`` fails validation.
            RuntimeError: If the DeepSeek API call fails.
        """
        self.validate_messages(messages)

        openai_messages: List[Dict[str, str]] = [
            {"role": m.role, "content": m.content} for m in messages
        ]

        params: Dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": openai_messages,
        }
        params.update(self.default_params)
        params.update(kwargs)

        try:
            response = self.client.chat.completions.create(**params)
        except Exception as exc:
            raise RuntimeError(f"DeepSeek API call failed: {exc}") from exc

        choice = response.choices[0]
        return ChatResponse(
            content=choice.message.content or "",
            model=response.model or self.model,
            usage=(
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                if response.usage
                else None
            ),
            raw_response=response.to_dict() if hasattr(response, "to_dict") else None,
        )


# ── auto-register with factory ─────────────────────────────────────────
LLMFactory.register_provider("deepseek", DeepSeekLLM)
