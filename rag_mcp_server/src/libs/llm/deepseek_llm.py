"""DeepSeek LLM provider implementation.

DeepSeek exposes an OpenAI-compatible API — this provider is a thin wrapper
around ``openai.OpenAI`` pointed at DeepSeek's base URL.  All configuration
is read from ``settings.llm``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag_mcp_server.src.core.settings import Settings
from rag_mcp_server.src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from rag_mcp_server.src.libs.llm.llm_factory import LLMFactory

_DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


class DeepSeekLLM(BaseLLM):
    """LLM provider backed by the DeepSeek API.

    All configuration is read from the ``llm`` section of ``settings.yaml``:

    .. code-block:: yaml

        llm:
          provider: deepseek
          model: deepseek-chat
          api_key: "sk-your-deepseek-key"
          base_url: "https://api.deepseek.com/v1"   # optional
    """

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        """Initialise the DeepSeek client.

        Args:
            settings: Application settings.
            **kwargs: Overrides / extra generation defaults.  Recognised
                keys: ``model``, ``api_key``, ``base_url``.
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
        )
        base_url: Optional[str] = (
            kwargs.pop("base_url", None)
            or settings.llm.get("base_url")
            or _DEFAULT_BASE_URL
        )

        if not api_key:
            raise ValueError(
                "DeepSeek API key not found. Specify api_key in the llm "
                "section of config/settings.yaml."
            )

        self.default_params: Dict[str, Any] = kwargs
        self.client: OpenAI = OpenAI(api_key=api_key, base_url=base_url)

    # ── public interface ──────────────────────────────────────────────

    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a conversation to DeepSeek and return the response."""
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
