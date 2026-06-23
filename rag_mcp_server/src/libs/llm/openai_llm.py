"""OpenAI LLM provider implementation.

Provides a standard OpenAI chat-completion backend using the `openai` SDK.
Configuration is driven by ``settings.yaml`` / env vars, matching the
BaseLLM interface so it can be swapped transparently via LLMFactory.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from rag_mcp_server.src.core.settings import Settings
from rag_mcp_server.src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from rag_mcp_server.src.libs.llm.llm_factory import LLMFactory


class OpenAILLM(BaseLLM):
    """LLM provider backed by the standard OpenAI API.

    Reads ``api_key`` and an optional ``base_url`` from the settings ``llm``
    section, falling back to the ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL``
    environment variables.  Extra keyword arguments passed to the constructor
    are forwarded as default generation parameters (e.g. ``temperature``).

    Design Principles Applied
    -------------------------
    - **Pluggable** – same ``chat()`` contract as every other BaseLLM.
    - **Config-Driven** – key material and model selection come from
      ``Settings``, not hard-coded.
    - **Observable** – accepts an optional trace context (reserved for B-5).
    """

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        """Initialise the OpenAI client.

        Args:
            settings: Application settings.  Expected keys in
                ``settings.llm``: ``model``, ``api_key``, ``base_url``
                (the last two are optional when the corresponding env
                vars are set).
            **kwargs: Overrides / extra generation defaults.  Recognised
                keys: ``model``, ``api_key``, ``base_url`` — anything else
                becomes a default parameter sent to the Chat Completion API.
        """
        from openai import OpenAI

        self.settings = settings

        # ── resolve model ──────────────────────────────────────────
        self.model: str = kwargs.pop("model", settings.llm.get("model", "gpt-4o"))

        # ── resolve credentials ─────────────────────────────────────
        api_key: Optional[str] = (
            kwargs.pop("api_key", None)
            or settings.llm.get("api_key")
            or os.getenv("OPENAI_API_KEY")
        )
        base_url: Optional[str] = (
            kwargs.pop("base_url", None)
            or settings.llm.get("base_url")
            or os.getenv("OPENAI_BASE_URL")
        )

        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in .env "
                "or specify api_key in the llm section of settings.yaml."
            )

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        # Extra kwargs become default generation parameters (temperature, etc.)
        self.default_params: Dict[str, Any] = kwargs
        self.client: OpenAI = OpenAI(**client_kwargs)

    # ── public interface ──────────────────────────────────────────────

    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a conversation to OpenAI and return the response.

        Args:
            messages: Conversation history (system / user / assistant).
            trace: Reserved for observability (Phase B-5).
            **kwargs: Per-call overrides that are merged on top of
                ``default_params``.

        Returns:
            ``ChatResponse`` with the completion content and usage metadata.

        Raises:
            ValueError: If ``messages`` fails validation.
            RuntimeError: If the OpenAI API call fails.
        """
        self.validate_messages(messages)

        openai_messages: List[Dict[str, str]] = [
            {"role": m.role, "content": m.content} for m in messages
        ]

        # Build API parameters: defaults → per-call overrides
        params: Dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": openai_messages,
        }
        params.update(self.default_params)
        params.update(kwargs)

        try:
            response = self.client.chat.completions.create(**params)
        except Exception as exc:
            raise RuntimeError(f"OpenAI API call failed: {exc}") from exc

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
LLMFactory.register_provider("openai", OpenAILLM)
