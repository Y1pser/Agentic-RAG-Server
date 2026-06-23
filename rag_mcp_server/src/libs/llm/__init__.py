"""
LLM Module.

This package contains LLM client abstractions and implementations:
- Base LLM class (text-only)
- LLM Factory
- OpenAI-compatible provider implementations (B1.7)
- Ollama provider (B1.8, TBD)
"""

from rag_mcp_server.src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from rag_mcp_server.src.libs.llm.llm_factory import LLMFactory

# Provider implementations — importing them auto-registers each with LLMFactory.
# Keep these after the base imports so the registry is ready.
from rag_mcp_server.src.libs.llm.openai_llm import OpenAILLM  # noqa: F401
from rag_mcp_server.src.libs.llm.azure_llm import AzureLLM  # noqa: F401
from rag_mcp_server.src.libs.llm.deepseek_llm import DeepSeekLLM  # noqa: F401

__all__ = [
    # Base classes
    "BaseLLM",
    # Data types
    "ChatResponse",
    "Message",
    # Factory
    "LLMFactory",
    # Provider implementations
    "OpenAILLM",
    "AzureLLM",
    "DeepSeekLLM",
]
