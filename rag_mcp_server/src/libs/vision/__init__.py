"""
Vision LLM Module.

This package contains Vision LLM abstractions and implementations:
- Base Vision LLM class (image-to-text)
- Vision LLM Factory
- Provider implementations (Azure Vision LLM, etc.)
"""

from rag_mcp_server.src.libs.vision.base_vision_llm import BaseVisionLLM
from rag_mcp_server.src.libs.vision.vision_factory import VisionLLMFactory

# Provider implementations — importing them auto-registers each with VisionLLMFactory.
# Keep these after the base imports so the registry is ready.
# (B1.14: from rag_mcp_server.src.libs.vision.azure_vision_llm import AzureVisionLLM)

__all__ = [
    # Base classes
    "BaseVisionLLM",
    # Factory
    "VisionLLMFactory",
]
