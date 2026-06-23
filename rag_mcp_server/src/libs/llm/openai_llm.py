"""OpenAI LLM provider implementation.

Provides a standard OpenAI chat-completion backend using the `openai` SDK.
All configuration (api_key, base_url, model, …) is read from ``settings.llm``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag_mcp_server.src.core.settings import Settings
from rag_mcp_server.src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from rag_mcp_server.src.libs.llm.llm_factory import LLMFactory


class OpenAILLM(BaseLLM):
    """LLM provider backed by the standard OpenAI API.

    All configuration is read from the ``llm`` section of ``settings.yaml``:

    .. code-block:: yaml

        llm:
          provider: openai
          model: gpt-4o
          api_key: "sk-..."
          base_url: "https://api.openai.com/v1"   # optional

    Extra keyword arguments passed to the constructor become default
    generation parameters (e.g. ``temperature``).
    """

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        """Initialise the OpenAI client.

        Args:
            settings: Application settings.
            **kwargs: Overrides / extra generation defaults.  Recognised
                keys: ``model``, ``api_key``, ``base_url``.
        """
        from openai import OpenAI

        self.settings = settings

        # ── resolve model ──────────────────────────────────────────
        self.model: str = kwargs.pop("model", settings.llm.get("model", "gpt-4o"))

        # ── resolve credentials ─────────────────────────────────────
        api_key: Optional[str] = (
            kwargs.pop("api_key", None)
            or settings.llm.get("api_key")
        )
        base_url: Optional[str] = (
            kwargs.pop("base_url", None)
            or settings.llm.get("base_url")
        )

        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Specify api_key in the llm "
                "section of config/settings.yaml."
            )

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        # Extra kwargs become default generation parameters
        self.default_params: Dict[str, Any] = kwargs
        self.client: OpenAI = OpenAI(**client_kwargs)

    # ── public interface ──────────────────────────────────────────────

    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a conversation to OpenAI and return the response."""
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
