"""Reranker abstraction layer.

Provides:
- BaseReranker abstract interface with ScoredChunk dataclass
- NoneReranker pass-through fallback
- LLMReranker implementation (B1.12)
- CrossEncoderReranker implementation (B1.16)
- RerankerFactory for config-driven instantiation
"""

from rag_mcp_server.src.libs.reranker.base_reranker import (
    BaseReranker,
    NoneReranker,
    ScoredChunk,
)
from rag_mcp_server.src.libs.reranker.reranker_factory import RerankerFactory

# Import implementations to auto-register with RerankerFactory
from rag_mcp_server.src.libs.reranker.llm_reranker import (  # noqa: F401
    LLMReranker,
    LLMRerankError,
)
from rag_mcp_server.src.libs.reranker.cross_encoder_reranker import (  # noqa: F401
    CrossEncoderReranker,
    CrossEncoderRerankError,
)

__all__ = [
    "BaseReranker",
    "NoneReranker",
    "ScoredChunk",
    "RerankerFactory",
    "LLMReranker",
    "LLMRerankError",
    "CrossEncoderReranker",
    "CrossEncoderRerankError",
]
