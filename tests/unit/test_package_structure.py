"""Test that all packages in the project skeleton are importable."""

import importlib
import pytest


# All packages that should be importable after A1
EXPECTED_PACKAGES = [
    # Root
    "main",
    # RAG MCP Server
    "rag_mcp_server",
    "rag_mcp_server.src",
    "rag_mcp_server.src.core",
    "rag_mcp_server.src.libs",
    "rag_mcp_server.src.libs.llm",
    "rag_mcp_server.src.libs.embedding",
    "rag_mcp_server.src.libs.splitter",
    "rag_mcp_server.src.libs.vector_store",
    "rag_mcp_server.src.libs.reranker",
    "rag_mcp_server.src.libs.vision",
    "rag_mcp_server.src.ingestion",
    "rag_mcp_server.src.ingestion.transforms",
    "rag_mcp_server.src.ingestion.encoders",
    "rag_mcp_server.src.retrieval",
    "rag_mcp_server.src.mcp_server",
    "rag_mcp_server.src.mcp_server.tools",
    "rag_mcp_server.src.observability",
    "rag_mcp_server.src.observability.dashboard",
    # Agent Layer
    "agent_layer",
    "agent_layer.core",
    "agent_layer.tools",
    "agent_layer.versions",
]


@pytest.mark.parametrize("package_name", EXPECTED_PACKAGES)
def test_package_importable(package_name: str):
    """Each package in the skeleton should be importable."""
    mod = importlib.import_module(package_name)
    assert mod is not None
    assert hasattr(mod, "__doc__"), f"{package_name} missing __doc__"
