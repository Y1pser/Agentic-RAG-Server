"""Azure OpenAI LLM provider implementation.

Uses the ``AzureOpenAI`` client from the ``openai`` SDK so the chat()
interface is identical to the standard OpenAI provider, but the
connection is routed through an Azure OpenAI Service resource.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from rag_mcp_server.src.core.settings import Settings
from rag_mcp_server.src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from rag_mcp_server.src.libs.llm.llm_factory import LLMFactory


class AzureLLM(BaseLLM):
    """LLM provider backed by Azure OpenAI Service.

    Requires ``api_key``, ``endpoint``, and ``api_version``.  These can be
    supplied in ``settings.yaml`` under ``llm`` or via the well-known
    environment variables:

    * ``AZURE_OPENAI_API_KEY``
    * ``AZURE_OPENAI_ENDPOINT``
    * ``AZURE_OPENAI_API_VERSION``

    Design Principles Applied
    -------------------------
    - **Pluggable** – drop-in replacement for any other BaseLLM.
    - **Config-Driven** – all connection details come from ``Settings`` /
      environment, never from hard-coded strings.
    - **Observable** – trace context argument reserved for Phase B-5.
    """

    # Default API version when nothing is configured explicitly.
    _DEFAULT_API_VERSION = "2024-02-15-preview"

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        """Initialise the Azure OpenAI client.

        Args:
            settings: Application settings.  Expected keys in
                ``settings.llm``: ``model``, ``api_key``, ``endpoint``,
                ``api_version`` (all optional when env vars are set).
            **kwargs: Overrides / extra generation defaults.  Recognised
                keys: ``model``, ``api_key``, ``endpoint``, ``api_version``.
                Anything else is treated as a default generation parameter.
        """
        from openai import AzureOpenAI

        self.settings = settings

        # ── resolve model (deployment name) ─────────────────────────
        self.model: str = kwargs.pop("model", settings.llm.get("model", "gpt-4o"))

        # ── resolve credentials ─────────────────────────────────────
        api_key: Optional[str] = (
            kwargs.pop("api_key", None)
            or settings.llm.get("api_key")
            or os.getenv("AZURE_OPENAI_API_KEY")
        )
        endpoint: Optional[str] = (
            kwargs.pop("endpoint", None)
            or settings.llm.get("endpoint")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        api_version: str = (
            kwargs.pop("api_version", None)
            or settings.llm.get("api_version")
            or os.getenv("AZURE_OPENAI_API_VERSION")
            or self._DEFAULT_API_VERSION
        )

        if not api_key:
            raise ValueError(
                "Azure OpenAI API key not found. Set AZURE_OPENAI_API_KEY in "
                ".env or specify api_key in the llm section of settings.yaml."
            )
        if not endpoint:
            raise ValueError(
                "Azure OpenAI endpoint not found. Set AZURE_OPENAI_ENDPOINT "
                "in .env or specify endpoint in the llm section of settings.yaml."
            )

        # Extra kwargs become default generation parameters
        self.default_params: Dict[str, Any] = kwargs
        self.client: AzureOpenAI = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )

    # ── public interface ──────────────────────────────────────────────

    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a conversation to Azure OpenAI and return the response.

        Args:
            messages: Conversation history (system / user / assistant).
            trace: Reserved for observability (Phase B-5).
            **kwargs: Per-call overrides merged on top of ``default_params``.

        Returns:
            ``ChatResponse`` with the completion content and usage metadata.

        Raises:
            ValueError: If ``messages`` fails validation.
            RuntimeError: If the Azure OpenAI API call fails.
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
            raise RuntimeError(f"Azure OpenAI API call failed: {exc}") from exc

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
LLMFactory.register_provider("azure", AzureLLM)
