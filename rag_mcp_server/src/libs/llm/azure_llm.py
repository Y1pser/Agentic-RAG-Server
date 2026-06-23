"""Azure OpenAI LLM provider implementation.

Uses the ``AzureOpenAI`` client from the ``openai`` SDK.  All configuration
is read from ``settings.llm`` (no env-var fallback).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag_mcp_server.src.core.settings import Settings
from rag_mcp_server.src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from rag_mcp_server.src.libs.llm.llm_factory import LLMFactory


class AzureLLM(BaseLLM):
    """LLM provider backed by Azure OpenAI Service.

    All configuration is read from the ``llm`` section of ``settings.yaml``:

    .. code-block:: yaml

        llm:
          provider: azure
          model: gpt-4o              # deployment name
          api_key: "your-azure-key"
          endpoint: "https://YOUR_RESOURCE.openai.azure.com"
          api_version: "2024-02-15-preview"
    """

    _DEFAULT_API_VERSION = "2024-02-15-preview"

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        """Initialise the Azure OpenAI client.

        Args:
            settings: Application settings.
            **kwargs: Overrides / extra generation defaults.  Recognised
                keys: ``model``, ``api_key``, ``endpoint``, ``api_version``.
        """
        from openai import AzureOpenAI

        self.settings = settings

        # ── resolve model (deployment name) ─────────────────────────
        self.model: str = kwargs.pop("model", settings.llm.get("model", "gpt-4o"))

        # ── resolve credentials ─────────────────────────────────────
        api_key: Optional[str] = (
            kwargs.pop("api_key", None)
            or settings.llm.get("api_key")
        )
        endpoint: Optional[str] = (
            kwargs.pop("endpoint", None)
            or settings.llm.get("endpoint")
        )
        api_version: str = (
            kwargs.pop("api_version", None)
            or settings.llm.get("api_version")
            or self._DEFAULT_API_VERSION
        )

        if not api_key:
            raise ValueError(
                "Azure OpenAI API key not found. Specify api_key in the llm "
                "section of config/settings.yaml."
            )
        if not endpoint:
            raise ValueError(
                "Azure OpenAI endpoint not found. Specify endpoint in the llm "
                "section of config/settings.yaml."
            )

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
        """Send a conversation to Azure OpenAI and return the response."""
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
