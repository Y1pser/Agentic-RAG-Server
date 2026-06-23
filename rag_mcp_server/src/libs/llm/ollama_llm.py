"""Ollama LLM provider implementation.

Provides a chat-completion backend for locally running Ollama instances
using the httpx HTTP client.  All configuration is read from ``settings.llm``.

Ollama's /api/chat endpoint uses a different request/response format than
OpenAI, so this provider does NOT use the ``openai`` SDK.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from rag_mcp_server.src.core.settings import Settings
from rag_mcp_server.src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from rag_mcp_server.src.libs.llm.llm_factory import LLMFactory


class OllamaLLM(BaseLLM):
    """LLM provider backed by a local Ollama instance.

    All configuration is read from the ``llm`` section of ``settings.yaml``:

    .. code-block:: yaml

        llm:
          provider: ollama
          model: llama3
          base_url: "http://localhost:11434"    # optional
          temperature: 0.7                      # optional
          max_tokens: 2048                      # optional

    The ``base_url`` is resolved in this order:
    1. Explicit ``base_url`` kwarg passed to the constructor.
    2. ``settings.llm.base_url``.
    3. ``OLLAMA_BASE_URL`` environment variable.
    4. Default ``http://localhost:11434``.
    """

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 120.0  # generous timeout for local inference

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        """Initialise the Ollama client.

        Args:
            settings: Application settings.
            **kwargs: Overrides / extra generation defaults.  Recognised
                keys: ``model``, ``base_url``, ``timeout``.
        """
        self.settings = settings

        # ── resolve model ──────────────────────────────────────────
        self.model: str = kwargs.pop("model", settings.llm.get("model", "llama3"))

        # ── resolve base URL (explicit > settings > env > default) ─
        self.base_url: str = (
            kwargs.pop("base_url", None)
            or settings.llm.get("base_url")
            or os.environ.get("OLLAMA_BASE_URL")
            or self.DEFAULT_BASE_URL
        ).rstrip("/")

        # ── resolve timeout ─────────────────────────────────────────
        self.timeout: float = float(
            kwargs.pop("timeout", None)
            or settings.llm.get("timeout", self.DEFAULT_TIMEOUT)
        )

        # ── default generation parameters ───────────────────────────
        self.default_temperature: float = float(
            kwargs.pop("temperature", settings.llm.get("temperature", 0.7))
        )
        self.default_max_tokens: int = int(
            kwargs.pop("max_tokens", settings.llm.get("max_tokens", 2048))
        )

        # Remaining kwargs stored for future extensibility
        self.extra_kwargs = kwargs

    # ── public interface ──────────────────────────────────────────────

    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a conversation to Ollama and return the response.

        Ollama uses ``num_predict`` (not ``max_tokens``) and wraps
        generation parameters in an ``options`` object.
        """
        self.validate_messages(messages)

        # ── build API payload ──────────────────────────────────────
        model = kwargs.pop("model", self.model)
        temperature = kwargs.pop("temperature", self.default_temperature)
        max_tokens = kwargs.pop("max_tokens", self.default_max_tokens)

        api_messages: List[Dict[str, str]] = [
            {"role": m.role, "content": m.content} for m in messages
        ]

        payload: Dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        # Merge any remaining kwargs into options (extensibility)
        if kwargs:
            payload["options"].update(kwargs)

        # ── call API ───────────────────────────────────────────────
        import httpx

        url = f"{self.base_url}/api/chat"
        headers = {"Content-Type": "application/json"}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException:
            raise RuntimeError(
                f"[Ollama] Request timed out after {self.timeout} seconds. "
                "Consider increasing the timeout for larger models."
            )
        except httpx.ConnectError:
            raise RuntimeError(
                "[Ollama] Connection failed. Ensure Ollama is running "
                "locally. Start it with 'ollama serve'."
            )
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"[Ollama] Request failed: {type(exc).__name__}: {exc}"
            ) from exc

        # ── handle HTTP errors ─────────────────────────────────────
        if response.status_code != 200:
            error_detail = self._parse_error(response)
            raise RuntimeError(
                f"[Ollama] API error (HTTP {response.status_code}): {error_detail}"
            )

        # ── parse response ─────────────────────────────────────────
        data: Dict[str, Any] = response.json()

        # Ollama /api/chat returns {"message": {"role": ..., "content": ...}}
        if "message" in data:
            content = data["message"]["content"]
        elif "response" in data:
            # Legacy /api/generate fallback
            content = data["response"]
        else:
            raise RuntimeError(
                "[Ollama] Unexpected response format: "
                "missing 'message' or 'response' key"
            )

        # Build usage stats when available
        usage: Optional[Dict[str, int]] = None
        if "eval_count" in data or "prompt_eval_count" in data:
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

        return ChatResponse(
            content=content,
            model=data.get("model", model),
            usage=usage,
            raw_response=data,
        )

    # ── internal helpers ───────────────────────────────────────────────

    @staticmethod
    def _parse_error(response: Any) -> str:
        """Extract a human-readable error message from an HTTP response.

        The message never leaks internal URLs or credentials.
        """
        try:
            error_data = response.json()
            if "error" in error_data:
                return str(error_data["error"])
            return response.text[:200] if response.text else "Unknown error"
        except Exception:
            return response.text[:200] if response.text else "Unknown error"


# ── auto-register with factory ─────────────────────────────────────────
LLMFactory.register_provider("ollama", OllamaLLM)
