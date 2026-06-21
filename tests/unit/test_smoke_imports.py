"""Verify all key packages are importable — quick smoke test."""


def test_can_import_rag_mcp_server():
    """The RAG MCP server package should be importable."""
    import rag_mcp_server
    assert rag_mcp_server is not None


def test_can_import_agent_layer():
    """The agent layer package should be importable."""
    import agent_layer
    assert agent_layer is not None


def test_can_import_main():
    """The main entry point should be importable."""
    from main import main
    assert callable(main)
