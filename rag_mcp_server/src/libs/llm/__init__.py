"""
LLM Module.

This package contains LLM client abstractions and implementations:
- Base LLM class (text-only)
- LLM Factory
- Provider implementations (to be added in B1.7-B1.8)
"""

from rag_mcp_server.src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from rag_mcp_server.src.libs.llm.llm_factory import LLMFactory

__all__ = [
    # Base classes
    "BaseLLM",
    # Data types
    "ChatResponse",
    "Message",
    # Factory
    "LLMFactory",
]
