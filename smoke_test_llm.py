"""Real-API smoke test for B1.7 LLM providers.

Usage:
    1. Fill in real API keys in config/settings.yaml first
    2. python smoke_test_llm.py

This script reads config/settings.yaml directly (no .env file).
Each provider is tested independently; failures in one don't block the others.
"""

import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_mcp_server.src.core.settings import Settings, load_settings
from rag_mcp_server.src.libs.llm.base_llm import Message


def test_provider(provider_name: str, settings_kwargs: dict, label: str) -> bool:
    """Test a single provider with a real API call.

    Creates a Settings object with the given provider + credentials
    (all from settings_kwargs), then calls the LLM via the factory.
    """
    print(f"\n{'='*60}")
    print(f"Testing {label} ({provider_name}) ...")
    print(f"{'='*60}")

    try:
        from rag_mcp_server.src.libs.llm.llm_factory import LLMFactory

        settings = Settings(llm={"provider": provider_name, **settings_kwargs})
        llm = LLMFactory.create(settings)
        print(f"  [OK] Instance created: {type(llm).__name__}")

        messages = [
            Message(role="system", content="Reply in one short sentence."),
            Message(role="user", content="Say hello and tell me what model you are."),
        ]
        resp = llm.chat(messages)
        print(f"  [OK] Response: {resp.content}")
        print(f"  [OK] Model: {resp.model}")
        print(f"  [OK] Tokens: {resp.usage}")
        print(f"  [PASS] {label}")
        return True

    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False


def main():
    # Load full settings for convenience (fallback credentials)
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config", "settings.yaml"
    )
    base = load_settings(config_path) if os.path.exists(config_path) else Settings()

    results: dict[str, bool] = {}

    # ── DeepSeek ────────────────────────────────────────────────────
    results["DeepSeek"] = test_provider(
        "deepseek",
        {
            "model": base.llm.get("model", "deepseek-chat"),
            "api_key": base.llm.get("api_key", ""),
            "base_url": base.llm.get("base_url", "https://api.deepseek.com/v1"),
        },
        "DeepSeek",
    )

    # ── OpenAI ──────────────────────────────────────────────────────
    results["OpenAI"] = test_provider(
        "openai",
        {
            "model": base.llm.get("model", "gpt-4o-mini"),
            "api_key": base.llm.get("api_key", ""),
        },
        "OpenAI",
    )

    # ── Azure ───────────────────────────────────────────────────────
    results["Azure"] = test_provider(
        "azure",
        {
            "model": base.llm.get("model", "gpt-4o"),
            "api_key": base.llm.get("api_key", ""),
            "endpoint": base.llm.get("endpoint", ""),
        },
        "Azure OpenAI",
    )

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, ok in results.items():
        status = "[PASS]" if ok else "[FAIL] (check config/settings.yaml)"
        print(f"  {name}: {status}")

    total = sum(results.values())
    print(f"\n{total}/{len(results)} providers working.")


if __name__ == "__main__":
    main()
