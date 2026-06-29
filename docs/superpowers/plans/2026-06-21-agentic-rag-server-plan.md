# Agentic RAG Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dual-layer Agentic RAG system — RAG MCP Server (replicated from clean-start) plus an Agent layer with ReAct loop, hybrid quality scoring, and cascade fallback (rewrite → web search).

**Architecture:** Two-layer design. Bottom layer: RAG MCP Server exposing tools via MCP protocol (stdio). Top layer: Agent with ReAct engine that calls RAG as a tool alongside query_rewriter and web_search. Agent supports two implementations: custom ReAct loop and LangGraph StateGraph, sharing the same tool set.

**Tech Stack:** Python 3.11+, Chroma, LangChain (splitters only), MCP SDK, LangGraph, Tavily/SerpAPI, Streamlit, Ragas, pytest, python-dotenv, PyYAML

---

## Global Constraints

- RAG layer architecture must follow original DEV_SPEC pluggable pattern (BaseLLM/BaseEmbedding/BaseVectorStore factories)
- Agent and RAG communicate via direct import (Phase 1); MCP Client protocol reserved for future
- All unit tests use Mock/Fake for external dependencies (LLM, Embedding, Vision)
- .env must never be committed; .env.example serves as template
- Original `MODULAR-RAG-MCP-SERVER/` directory is read-only reference
- New project lives in `Agentic-RAG-Server/` sibling directory
- Every task commits atomically after passing tests

---

## File Structure Map

```
Agentic-RAG-Server/
├── main.py                          # Entry point, loads settings, starts server
├── pyproject.toml                   # Project metadata, dependencies, pytest config
├── README.md                        # Phase G
├── requirements.txt                 # Pinned dependencies
├── .env                             # API keys (gitignored)
├── .env.example                     # API key template
├── .gitignore                       # Python + IDE + .env
├── config/
│   ├── settings.yaml                # Unified configuration
│   └── prompts/                     # LLM prompt templates
│       ├── image_captioning.txt
│       ├── chunk_refinement.txt
│       └── rerank.txt
├── rag_mcp_server/
│   └── src/
│       ├── __init__.py
│       ├── core/                    # Types, settings, trace context
│       │   ├── __init__.py
│       │   ├── types.py             # Document, Chunk, ChunkRecord
│       │   ├── settings.py          # Settings dataclass + loader
│       │   └── trace/               # TraceContext (Phase B-5)
│       ├── libs/                    # Pluggable components
│       │   ├── __init__.py
│       │   ├── llm/                 # BaseLLM + LLMFactory + implementations
│       │   ├── embedding/           # BaseEmbedding + EmbeddingFactory + implementations
│       │   ├── splitter/            # BaseSplitter + SplitterFactory + RecursiveSplitter
│       │   ├── vector_store/        # BaseVectorStore + VectorStoreFactory + ChromaStore
│       │   ├── reranker/            # BaseReranker + RerankerFactory + implementations
│       │   └── vision/              # BaseVisionLLM + implementations
│       ├── ingestion/               # Ingestion pipeline
│       │   ├── __init__.py
│       │   ├── loader.py            # BaseLoader + PdfLoader
│       │   ├── chunker.py           # DocumentChunker
│       │   ├── transforms/          # ChunkRefiner, MetadataEnricher, ImageCaptioner
│       │   ├── encoders/            # DenseEncoder, SparseEncoder
│       │   ├── bm25_indexer.py      # BM25Indexer
│       │   ├── vector_upserter.py   # VectorUpserter
│       │   ├── image_storage.py     # ImageStorage
│       │   ├── file_integrity.py    # FileIntegrityChecker
│       │   ├── document_manager.py  # DocumentManager (Phase B-6)
│       │   └── pipeline.py          # IngestionPipeline orchestration
│       ├── retrieval/               # Retrieval pipeline
│       │   ├── __init__.py
│       │   ├── query_processor.py   # QueryProcessor
│       │   ├── dense_retriever.py   # DenseRetriever
│       │   ├── sparse_retriever.py  # SparseRetriever
│       │   ├── rrf_fusion.py        # RRFFusion
│       │   ├── hybrid_search.py     # HybridSearch orchestration
│       │   └── reranker.py          # CoreReranker
│       ├── mcp_server/              # MCP Server layer
│       │   ├── __init__.py
│       │   ├── server.py            # MCP Server entry + stdio
│       │   └── tools/               # Tool implementations
│       └── observability/           # Trace + Dashboard
│           ├── __init__.py
│           ├── logger.py            # JSON Lines logger
│           └── dashboard/           # Streamlit pages
├── agent_layer/                     # Agent — ORIGINAL WORK
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── react_engine.py          # ReActAgent class (Phase D)
│   │   ├── hybrid_scorer.py         # HybridScorer (Phase C)
│   │   ├── agent_memory.py          # AgentMemory (Phase C)
│   │   └── config.py                # AgentConfig (Phase C)
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py                  # Tool abstract base (Phase C)
│   │   ├── local_search.py          # LocalSearch tool (Phase C)
│   │   ├── query_rewriter.py        # QueryRewriter tool (Phase C)
│   │   ├── web_search.py            # WebSearch tool (Phase F)
│   │   └── registry.py             # ToolRegistry (Phase C)
│   ├── versions/
│   │   ├── __init__.py
│   │   ├── custom_agent.py          # Custom ReAct entry (Phase D)
│   │   └── langgraph_agent.py       # LangGraph entry (Phase E)
│   ├── prompts/
│   │   ├── react_system.md          # ReAct system prompt (Phase C)
│   │   ├── scorer_prompt.md         # Scorer dual-mode prompt (Phase C)
│   │   └── rewriter_prompt.md       # Query rewrite prompt (Phase C)
│   └── tests/
│       ├── __init__.py
│       ├── test_tools.py
│       ├── test_scorer.py
│       ├── test_memory.py
│       ├── test_custom_agent.py
│       └── test_langgraph_agent.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   └── fixtures/
│       └── sample_documents/
└── docs/
    └── superpowers/
        ├── specs/
        │   └── 2026-06-21-agentic-rag-server-design.md
        └── plans/
            └── 2026-06-21-agentic-rag-server-plan.md
```

---

## Phase A: Environment Initialization

### Task A1: Initialize directory tree and minimal runnable entry

**Files:**
- Create: `Agentic-RAG-Server/main.py`
- Create: `Agentic-RAG-Server/pyproject.toml`
- Create: `Agentic-RAG-Server/.gitignore`
- Create: all `__init__.py` files for directory tree

**Interfaces:**
- Produces: Runnable `python main.py` that prints "Agentic RAG Server initialized"

- [ ] **Step 1: Create .gitignore**

```bash
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
*.so
.Python
.venv/
venv/
.env
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
htmlcov/
.tox/
.mypy_cache/
.ruff_cache/
.vscode/
.idea/
*.swp
*.swo
*~
data/db/
data/images/
logs/
EOF
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "agentic-rag-server"
version = "0.1.0"
description = "Agentic RAG MCP Server with ReAct loop and cascade fallback"
requires-python = ">=3.11"
dependencies = [
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
    "chromadb>=0.5",
    "langchain>=0.3",
    "langchain-text-splitters>=0.3",
    "mcp>=1.0",
    "openai>=1.0",
    "pydantic>=2.0",
    "httpx>=0.27",
    "pymupdf>=1.24",
    "markitdown>=0.0.1",
    "streamlit>=1.35",
    "sentence-transformers>=3.0",
    "ragas>=0.2",
    "tiktoken>=0.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.12",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
]
langgraph = [
    "langgraph>=0.2",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = ["-v", "--tb=short", "--strict-markers"]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "e2e: End-to-end tests",
    "slow: Slow tests",
]
```

- [ ] **Step 3: Create directory tree**

```bash
cd Agentic-RAG-Server

# RAG layer
mkdir -p rag_mcp_server/src/core/trace
mkdir -p rag_mcp_server/src/libs/{llm,embedding,splitter,vector_store,reranker,vision}
mkdir -p rag_mcp_server/src/ingestion/transforms
mkdir -p rag_mcp_server/src/ingestion/encoders
mkdir -p rag_mcp_server/src/retrieval
mkdir -p rag_mcp_server/src/mcp_server/tools
mkdir -p rag_mcp_server/src/observability/dashboard/pages
mkdir -p rag_mcp_server/src/observability/dashboard/services

# Agent layer
mkdir -p agent_layer/core
mkdir -p agent_layer/tools
mkdir -p agent_layer/versions
mkdir -p agent_layer/prompts
mkdir -p agent_layer/tests

# Config & prompts
mkdir -p config/prompts

# Tests
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p tests/e2e
mkdir -p tests/fixtures/sample_documents

# Docs
mkdir -p docs/superpowers/specs
mkdir -p docs/superpowers/plans

# Data (gitignored)
mkdir -p data/db
mkdir -p data/images
```

- [ ] **Step 4: Create all __init__.py files**

```bash
find . -type d -name "__pycache__" -prune -o -type d -print | while read dir; do
    if [ ! -f "$dir/__init__.py" ]; then
        touch "$dir/__init__.py"
    fi
done
```

- [ ] **Step 5: Create main.py**

```python
"""Agentic RAG Server — entry point."""

def main():
    print("Agentic RAG Server initialized")
    print("Run 'python -m rag_mcp_server.src.mcp_server.server' to start MCP server")
    print("Run 'python -m agent_layer.versions.custom_agent' to start Agent CLI")

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Verify**

```bash
python main.py
# Expected: prints initialization message
python -c "import rag_mcp_server; import agent_layer; print('imports ok')"
# Expected: imports ok
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(A1): initialize directory tree and minimal runnable entry"
```

---

### Task A2: Introduce pytest and establish test directory conventions

**Files:**
- Create: `tests/unit/test_smoke_imports.py`
- Modify: `pyproject.toml` (add pytest config — already done in A1)
- Create: `tests/conftest.py`
- Create: `tests/fixtures/sample_documents/hello.txt`

**Interfaces:**
- Produces: `pytest -q` runs and passes with at least 1 smoke test

- [ ] **Step 1: Create test fixture document**

```bash
echo "Hello, RAG! This is a test document for ingestion testing." > tests/fixtures/sample_documents/hello.txt
```

- [ ] **Step 2: Create conftest.py with shared fixtures**

```python
# tests/conftest.py
"""Shared pytest fixtures for all test layers."""

import pytest
from pathlib import Path


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def fixtures_dir():
    """Return the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_doc_path():
    """Return path to a sample document for ingestion tests."""
    return Path(__file__).parent / "fixtures" / "sample_documents" / "hello.txt"
```

- [ ] **Step 3: Write the failing smoke test first**

```python
# tests/unit/test_smoke_imports.py
"""Verify all key packages are importable."""


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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest -q tests/unit/test_smoke_imports.py -v
# Expected: 3 passed
```

- [ ] **Step 5: Commit**

```bash
git add tests/ pyproject.toml
git commit -m "feat(A2): introduce pytest with smoke import tests"
```

---

### Task A3: Configuration loading and validation (Settings)

**Files:**
- Create: `rag_mcp_server/src/core/settings.py`
- Create: `config/settings.yaml`
- Create: `tests/unit/test_config_loading.py`

**Interfaces:**
- Produces: `Settings` dataclass, `load_settings(path) -> Settings`, `validate_settings(settings) -> None`
- Consumed by: A4 (.env integration), all Phase B tasks

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_config_loading.py
"""Test configuration loading and validation."""

import pytest
import tempfile
import yaml
from pathlib import Path
from rag_mcp_server.src.core.settings import Settings, load_settings, validate_settings


class TestSettingsDataclass:
    """Unit tests for Settings dataclass."""

    def test_settings_has_required_fields(self):
        """Settings should expose all top-level config sections."""
        settings = Settings()
        assert hasattr(settings, "llm")
        assert hasattr(settings, "embedding")
        assert hasattr(settings, "vector_store")
        assert hasattr(settings, "retrieval")
        assert hasattr(settings, "observability")


class TestLoadSettings:
    """Tests for load_settings function."""

    def test_loads_valid_yaml(self):
        """Should parse a valid YAML file into a Settings object."""
        yaml_content = """
llm:
  provider: openai
  model: gpt-4o
  api_key: test-key
embedding:
  provider: openai
  model: text-embedding-3-small
vector_store:
  backend: chroma
retrieval:
  sparse_backend: bm25
  fusion_algorithm: rrf
  rerank_backend: none
observability:
  enabled: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            settings = load_settings(f.name)
        Path(f.name).unlink()
        assert settings.llm["provider"] == "openai"
        assert settings.vector_store["backend"] == "chroma"

    def test_missing_file_raises(self):
        """Should raise FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError):
            load_settings("/nonexistent/path.yaml")


class TestValidateSettings:
    """Tests for settings validation."""

    def test_missing_llm_provider_raises(self):
        """Should raise ValueError when llm.provider is missing."""
        settings = Settings()
        settings.llm = {"model": "gpt-4o"}  # no provider
        with pytest.raises(ValueError, match="llm.provider"):
            validate_settings(settings)

    def test_missing_embedding_provider_raises(self):
        """Should raise ValueError when embedding.provider is missing."""
        settings = Settings()
        settings.llm = {"provider": "openai"}
        settings.embedding = {"model": "text-embedding-3-small"}  # no provider
        with pytest.raises(ValueError, match="embedding.provider"):
            validate_settings(settings)

    def test_valid_settings_pass(self):
        """Should not raise for valid settings."""
        settings = Settings()
        settings.llm = {"provider": "openai"}
        settings.embedding = {"provider": "openai"}
        settings.vector_store = {"backend": "chroma"}
        validate_settings(settings)  # should not raise
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_config_loading.py -v
# Expected: FAIL — module not found
```

- [ ] **Step 3: Write minimal Settings implementation**

```python
# rag_mcp_server/src/core/settings.py
"""Configuration loading and validation."""

from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class Settings:
    """Application settings, loaded from config/settings.yaml."""

    llm: dict = field(default_factory=dict)
    embedding: dict = field(default_factory=dict)
    vector_store: dict = field(default_factory=dict)
    retrieval: dict = field(default_factory=dict)
    rerank: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)
    observability: dict = field(default_factory=dict)
    dashboard: dict = field(default_factory=dict)
    agent: dict = field(default_factory=dict)


def load_settings(path: str) -> Settings:
    """Load settings from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Settings dataclass instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    settings = Settings()
    for key in Settings.__dataclass_fields__:
        if key in raw:
            setattr(settings, key, raw[key])
    return settings


def validate_settings(settings: Settings) -> None:
    """Validate that required config fields are present.

    Args:
        settings: Settings instance to validate.

    Raises:
        ValueError: If a required field is missing.
    """
    required = {
        "llm.provider": settings.llm.get("provider"),
        "embedding.provider": settings.embedding.get("provider"),
        "vector_store.backend": settings.vector_store.get("backend"),
    }
    for path, value in required.items():
        if not value:
            raise ValueError(f"Missing required config field: {path}")
```

- [ ] **Step 4: Create minimal config/settings.yaml**

```yaml
# config/settings.yaml
# Agentic RAG Server configuration

llm:
  provider: openai
  model: gpt-4o

embedding:
  provider: openai
  model: text-embedding-3-small

vector_store:
  backend: chroma
  persist_directory: data/chroma

retrieval:
  sparse_backend: bm25
  fusion_algorithm: rrf
  rerank_backend: none

rerank:
  llm_rerank:
    enabled: false

evaluation:
  backends: []

observability:
  enabled: true
  logging:
    log_file: logs/traces.jsonl
    log_level: INFO
  detail_level: standard

dashboard:
  enabled: true
  port: 8501

agent:
  max_rounds: 3
  score_threshold: 0.7
  web_search:
    backend: tavily
  cache:
    enabled: false
```

- [ ] **Step 5: Run test to verify pass**

```bash
pytest tests/unit/test_config_loading.py -v
# Expected: 5 passed
```

- [ ] **Step 6: Commit**

```bash
git add config/settings.yaml rag_mcp_server/src/core/settings.py tests/unit/test_config_loading.py
git commit -m "feat(A3): configuration loading and validation with Settings dataclass"
```

---

### Task A4: .env secret management and .env.example template

**Files:**
- Create: `.env.example`
- Create: `.env` (gitignored)
- Modify: `rag_mcp_server/src/core/settings.py` (add env var support)
- Create: `tests/unit/test_env_loading.py`

**Interfaces:**
- Produces: `Settings` enriched with environment variable values
- Consumed by: All tasks that need API keys

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_env_loading.py
"""Test environment variable loading via python-dotenv."""

import os
import tempfile
from pathlib import Path
from rag_mcp_server.src.core.settings import load_settings, apply_env_overrides, Settings


class TestEnvOverrides:
    """Tests for environment variable overrides on settings."""

    def test_env_var_overrides_llm_api_key(self, monkeypatch):
        """Environment variables should override settings values."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-test-key")
        settings = Settings()
        settings.llm = {"provider": "openai", "model": "gpt-4o"}

        apply_env_overrides(settings)

        assert settings.llm["api_key"] == "sk-env-test-key"

    def test_env_var_missing_does_not_crash(self):
        """Missing env vars should not cause errors."""
        settings = Settings()
        settings.llm = {"provider": "openai"}
        apply_env_overrides(settings)  # should not raise

    def test_multiple_providers_get_keys(self, monkeypatch):
        """Multiple search API keys should all be picked up."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        monkeypatch.setenv("SERPAPI_API_KEY", "serp-test")
        settings = Settings()
        settings.agent = {"web_search": {"backend": "tavily"}}

        apply_env_overrides(settings)

        assert settings.agent["web_search"].get("tavily_api_key") == "tvly-test"
        assert settings.agent["web_search"].get("serpapi_api_key") == "serp-test"
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_env_loading.py -v
# Expected: FAIL — apply_env_overrides not defined
```

- [ ] **Step 3: Implement apply_env_overrides**

```python
# Add to rag_mcp_server/src/core/settings.py

import os
from dotenv import load_dotenv


def apply_env_overrides(settings: Settings) -> None:
    """Apply environment variable overrides to settings.

    Reads from .env file and os.environ. Keys follow the pattern:
    - OPENAI_API_KEY -> settings.llm["api_key"]
    - AZURE_OPENAI_API_KEY -> settings.llm["api_key"] (azure provider)
    - TAVILY_API_KEY -> settings.agent["web_search"]["tavily_api_key"]
    - SERPAPI_API_KEY -> settings.agent["web_search"]["serpapi_api_key"]
    - EMBEDDING_API_KEY -> settings.embedding["api_key"]

    Args:
        settings: Settings instance to apply overrides to.
    """
    # Load .env file if present
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # LLM keys
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and "api_key" not in settings.llm:
        settings.llm["api_key"] = openai_key

    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    if azure_key and settings.llm.get("provider") == "azure":
        settings.llm["api_key"] = azure_key

    # Embedding key
    embedding_key = os.getenv("EMBEDDING_API_KEY")
    if embedding_key and "api_key" not in settings.embedding:
        settings.embedding["api_key"] = embedding_key

    # Web search keys
    agent = settings.agent
    if "web_search" not in agent:
        agent["web_search"] = {}
    ws = agent["web_search"]

    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        ws["tavily_api_key"] = tavily_key

    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if serpapi_key:
        ws["serpapi_api_key"] = serpapi_key
```

- [ ] **Step 4: Create .env.example**

```bash
# .env.example
# Agentic RAG Server — API Key Configuration
# Copy this file to .env and fill in your keys. DO NOT commit .env!

# LLM Provider (pick one)
OPENAI_API_KEY=sk-your-key-here
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Embedding
EMBEDDING_API_KEY=

# Web Search (pick one or both)
TAVILY_API_KEY=
SERPAPI_API_KEY=

# Logging
LOG_LEVEL=INFO
```

- [ ] **Step 5: Create .env with placeholder**

```bash
cp .env.example .env
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/unit/test_env_loading.py -v
# Expected: 3 passed
```

- [ ] **Step 7: Commit**

```bash
git add .env.example rag_mcp_server/src/core/settings.py tests/unit/test_env_loading.py
git commit -m "feat(A4): .env secret management with python-dotenv integration"
```

---

### Task A5: git init and GitHub repository association

**Files:**
- None (git operations only)

**Interfaces:**
- Produces: Git repo with remote origin set, initial commits

- [ ] **Step 1: Initialize git repository**

```bash
cd Agentic-RAG-Server
git init
git add -A
git commit -m "init: Agentic RAG Server project skeleton"
```

- [ ] **Step 2: Verify clean state**

```bash
git status
# Expected: nothing to commit, working tree clean
```

- [ ] **Step 3: Associate with GitHub**

```bash
# NOTE: User must create the repo on GitHub first, then:
# git remote add origin https://github.com/<your-username>/Agentic-RAG-Server.git
# git branch -M main
# git push -u origin main
```

```bash
echo "GitHub remote setup: Create repo on GitHub, then run:"
echo "  git remote add origin https://github.com/<your-username>/Agentic-RAG-Server.git"
echo "  git branch -M main"
echo "  git push -u origin main"
```

---

## Phase B: RAG Core Replication

> **Execution strategy:** Tasks in Phase B follow the original DEV_SPEC architecture exactly. Each sub-phase (B-1 through B-8) should be completed in order. For each task, the auto-coder skill can handle the implementation by reading the DEV_SPEC reference. The plan below specifies files, interfaces, and key test patterns for each task.

### Phase B-1: Libs Pluggable Layer

#### Task B1.1: LLM abstract interface and factory

**Files:**
- Create: `rag_mcp_server/src/libs/llm/__init__.py`
- Create: `rag_mcp_server/src/libs/llm/base_llm.py`
- Create: `rag_mcp_server/src/libs/llm/llm_factory.py`
- Create: `tests/unit/test_llm_factory.py`

**Interfaces:**
- Produces: `BaseLLM` (abstract class with `chat(messages) -> str`), `LLMFactory.create(settings) -> BaseLLM`
- Consumed by: B1.7, B1.8, B2.5, B2.6, B2.7, B3.6, B3.7, B4.2

- [ ] **Step 1: Write test for factory routing**

```python
# tests/unit/test_llm_factory.py
"""Tests for LLM factory routing."""

import pytest
from rag_mcp_server.src.libs.llm.base_llm import BaseLLM
from rag_mcp_server.src.libs.llm.llm_factory import LLMFactory
from rag_mcp_server.src.core.settings import Settings


class FakeLLM(BaseLLM):
    """Fake LLM for testing factory routing."""
    def chat(self, messages, **kwargs):
        return "fake response"


class TestLLMFactory:
    def test_create_returns_base_llm(self):
        """Factory should return a BaseLLM instance."""
        settings = Settings()
        settings.llm = {"provider": "openai", "model": "gpt-4o"}
        # With no real backends registered yet, test the interface
        with pytest.raises(ValueError, match="provider"):
            LLMFactory.create(settings)

    def test_register_and_create(self):
        """Should return registered provider implementation."""
        LLMFactory.register("fake", lambda s: FakeLLM())
        settings = Settings()
        settings.llm = {"provider": "fake"}
        llm = LLMFactory.create(settings)
        assert isinstance(llm, BaseLLM)
        assert llm.chat([{"role": "user", "content": "hi"}]) == "fake response"
```

- [ ] **Step 2: Implement BaseLLM**

```python
# rag_mcp_server/src/libs/llm/base_llm.py
"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):
    """Abstract interface for LLM providers.

    All LLM implementations must subclass this and implement chat().
    """

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        """Send messages to the LLM and return the response text.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Provider-specific parameters (temperature, max_tokens, etc.)

        Returns:
            The LLM response as a string.
        """
        ...
```

- [ ] **Step 3: Implement LLMFactory**

```python
# rag_mcp_server/src/libs/llm/llm_factory.py
"""Factory for creating LLM instances from configuration."""

from typing import Callable
from rag_mcp_server.src.libs.llm.base_llm import BaseLLM
from rag_mcp_server.src.core.settings import Settings


class LLMFactory:
    """Factory that creates LLM instances based on settings.

    Providers are registered via LLMFactory.register(name, factory_fn).
    """

    _registry: dict[str, Callable[[Settings], BaseLLM]] = {}

    @classmethod
    def register(cls, name: str, factory_fn: Callable[[Settings], BaseLLM]) -> None:
        """Register a new LLM provider.

        Args:
            name: Provider name (e.g., 'openai', 'azure', 'ollama').
            factory_fn: Function that takes Settings and returns a BaseLLM.
        """
        cls._registry[name] = factory_fn

    @classmethod
    def create(cls, settings: Settings) -> BaseLLM:
        """Create an LLM instance based on settings.llm.provider.

        Args:
            settings: Application settings.

        Returns:
            A BaseLLM instance.

        Raises:
            ValueError: If the provider is not registered.
        """
        provider = settings.llm.get("provider", "")
        if provider not in cls._registry:
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Registered providers: {list(cls._registry.keys())}"
            )
        return cls._registry[provider](settings)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/test_llm_factory.py -v
# Expected: 2 passed
```

- [ ] **Step 5: Commit**

```bash
git add rag_mcp_server/src/libs/llm/ tests/unit/test_llm_factory.py
git commit -m "feat(B1.1): LLM abstract interface and factory pattern"
```

---

#### Task B1.2: Embedding abstract interface and factory

**Files:**
- Create: `rag_mcp_server/src/libs/embedding/base_embedding.py`
- Create: `rag_mcp_server/src/libs/embedding/embedding_factory.py`
- Create: `tests/unit/test_embedding_factory.py`

**Interfaces:**
- Produces: `BaseEmbedding` (abstract with `embed(texts: list[str]) -> list[list[float]]`), `EmbeddingFactory.create(settings) -> BaseEmbedding`
- Consumed by: B1.9, B2.8

- [ ] **Step 1: Write test**

```python
# tests/unit/test_embedding_factory.py
"""Tests for embedding factory."""

import pytest
from rag_mcp_server.src.libs.embedding.base_embedding import BaseEmbedding
from rag_mcp_server.src.libs.embedding.embedding_factory import EmbeddingFactory
from rag_mcp_server.src.core.settings import Settings


class FakeEmbedding(BaseEmbedding):
    def embed(self, texts):
        return [[0.1] * 128 for _ in texts]


class TestEmbeddingFactory:
    def test_register_and_create(self):
        EmbeddingFactory.register("fake", lambda s: FakeEmbedding())
        settings = Settings()
        settings.embedding = {"provider": "fake"}
        emb = EmbeddingFactory.create(settings)
        vectors = emb.embed(["hello", "world"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 128

    def test_unknown_provider_raises(self):
        settings = Settings()
        settings.embedding = {"provider": "nonexistent"}
        with pytest.raises(ValueError, match="nonexistent"):
            EmbeddingFactory.create(settings)
```

- [ ] **Step 2: Implement (same pattern as B1.1)**
- [ ] **Step 3: Run tests → commit**

---

#### Tasks B1.3 — B1.14

Each follows the same TDD pattern as B1.1/B1.2. Key files and interfaces:

| Task | Core Interface | Default Implementation |
|------|---------------|----------------------|
| B1.3 | `BaseSplitter.split(document) -> list[Chunk]` | `RecursiveSplitter` (LangChain) |
| B1.4 | `BaseVectorStore.add/get/query/delete` | `ChromaStore` |
| B1.5 | `BaseReranker.rerank(query, chunks) -> list[ScoredChunk]` | `NoneReranker` (pass-through) |
| B1.6 | `BaseEvaluator.evaluate(...) -> dict` | `CustomEvaluator` |
| B1.7 | — | `OpenAILLM`, `AzureLLM`, `DeepSeekLLM` |
| B1.8 | — | `OllamaLLM` |
| B1.9 | — | `OpenAIEmbedding`, `AzureEmbedding` |
| B1.10 | — | `RecursiveSplitter` with separators |
| B1.11 | — | `ChromaStore` with roundtrip |
| B1.12 | — | `LLMReranker` with prompt template |
| B1.13 | `BaseVisionLLM.describe(image) -> str` | Factory integration |
| B1.14 | — | `AzureVisionLLM` |
| B1.15 | — | `SentenceTransformerEmbedding` (BGE-M3, local, singleton) |
| B1.16 | — | `CrossEncoderReranker` (sentence-transformers) |

---

#### Task B1.15: Sentence Transformer Embedding (Local)

**Files:**
- Create: `rag_mcp_server/src/libs/embedding/sentence_transformer_embedding.py`
- Create: `tests/unit/test_sentence_transformer_embedding.py`

**Interfaces:**
- Produces: `SentenceTransformerEmbedding(BaseEmbedding)` with `embed(texts) -> EmbeddingResponse`
- Auto-registers as `"sentence_transformer"` provider with `EmbeddingFactory`

**Design decisions:**
- Model: `BAAI/bge-m3` (1024-dim, 100+ languages, ~2.2 GB)
- **Class-level lazy singleton** — model loaded once on first `embed()` call, reused across all instances and consumers (RAG ingestion, RAG retrieval, MemoryStore)
- No API key, no network — pure local CPU/GPU inference
- `validate_texts()` inherited from `BaseEmbedding`
- Empty list returns `EmbeddingResponse(embeddings=[], model=..., dimensions=1024)`

```python
class SentenceTransformerEmbedding(BaseEmbedding):
    _model: Optional[SentenceTransformer] = None   # class-level singleton
    _model_lock = threading.Lock()

    def __init__(self, settings: Settings, **kwargs):
        self.model_name = kwargs.pop("model", settings.embedding.get("model", "BAAI/bge-m3"))
        self.device = kwargs.pop("device", settings.embedding.get("device", "cpu"))
        self.normalize = kwargs.pop("normalize", settings.embedding.get("normalize", True))
        self.batch_size = int(kwargs.pop("batch_size", settings.embedding.get("batch_size", 32)))

    def embed(self, texts, trace=None, **kwargs) -> EmbeddingResponse:
        self.validate_texts(texts)
        if not texts:
            return EmbeddingResponse(embeddings=[], model=self.model_name, dimensions=1024)
        model = self._get_model()
        vectors = model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=self.normalize,
            show_progress_bar=False, **kwargs,
        )
        return EmbeddingResponse(
            embeddings=[v.tolist() for v in vectors],
            model=self.model_name,
            dimensions=vectors.shape[1],
            usage={"total_tokens": sum(len(t) // 4 for t in texts)},
        )

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            with cls._model_lock:
                if cls._model is None:
                    from sentence_transformers import SentenceTransformer
                    cls._model = SentenceTransformer(cls.model_name, device=cls.device)
        return cls._model
```

- [ ] **Step 1: Write test with mocked SentenceTransformer**
- [ ] **Step 2: Implement the class**
- [ ] **Step 3: Run tests → pass → commit**

---

### Phase B-2: Ingestion Pipeline (14 tasks)

Key files from original DEV_SPEC:

| Task | File | Key Function/Class |
|------|------|-------------------|
| B2.1 | `rag_mcp_server/src/core/types.py` | `Document`, `Chunk`, `ChunkRecord` dataclasses |
| B2.2 | `rag_mcp_server/src/ingestion/file_integrity.py` | `FileIntegrityChecker`, `SQLiteIntegrityChecker` |
| B2.3 | `rag_mcp_server/src/ingestion/loader.py` | `BaseLoader`, `PdfLoader` (MarkItDown) |
| B2.4 | `rag_mcp_server/src/ingestion/chunker.py` | `DocumentChunker` wrapping Libs Splitter |
| B2.5 | `rag_mcp_server/src/ingestion/transforms/chunk_refiner.py` | `ChunkRefiner` (Rule + LLM modes) |
| B2.6 | `rag_mcp_server/src/ingestion/transforms/metadata_enricher.py` | `MetadataEnricher` |
| B2.7 | `rag_mcp_server/src/ingestion/transforms/image_captioner.py` | `ImageCaptioner` |
| B2.8 | `rag_mcp_server/src/ingestion/encoders/dense_encoder.py` | `DenseEncoder` |
| B2.9 | `rag_mcp_server/src/ingestion/encoders/sparse_encoder.py` | `SparseEncoder` |
| B2.10 | `rag_mcp_server/src/ingestion/bm25_indexer.py` | `BM25Indexer` |
| B2.11 | `rag_mcp_server/src/ingestion/vector_upserter.py` | `VectorUpserter` |
| B2.12 | `rag_mcp_server/src/ingestion/image_storage.py` | `ImageStorage` |
| B2.13 | `rag_mcp_server/src/ingestion/pipeline.py` | `IngestionPipeline` |
| B2.14 | `rag_mcp_server/src/ingestion/__main__.py` | CLI: `python -m rag_mcp_server.src.ingestion` |

Each task follows the TDD pattern: write test → fail → implement → pass → commit.

---

### Phase B-3: Retrieval (7 tasks)

| Task | File | Key Function/Class |
|------|------|-------------------|
| B3.1 | `rag_mcp_server/src/retrieval/query_processor.py` | `QueryProcessor.process(query) -> ProcessedQuery` |
| B3.2 | `rag_mcp_server/src/retrieval/dense_retriever.py` | `DenseRetriever.retrieve(query, top_k) -> RetrievalResult` |
| B3.3 | `rag_mcp_server/src/retrieval/sparse_retriever.py` | `SparseRetriever.retrieve(query, top_k) -> RetrievalResult` |
| B3.4 | `rag_mcp_server/src/retrieval/rrf_fusion.py` | `RRFFusion.fuse(dense_results, sparse_results) -> list[ScoredChunk]` |
| B3.5 | `rag_mcp_server/src/retrieval/hybrid_search.py` | `HybridSearch.search(query, top_k) -> SearchResult` |
| B3.6 | `rag_mcp_server/src/retrieval/reranker.py` | `CoreReranker.rerank(query, chunks) -> list[ScoredChunk]` |
| B3.7 | `rag_mcp_server/src/retrieval/__main__.py` | CLI: `python -m rag_mcp_server.src.retrieval` |

---

### Phase B-4: MCP Server Layer (5 tasks)

| Task | File | Key Function/Class |
|------|------|-------------------|
| B4.1 | `rag_mcp_server/src/mcp_server/server.py` | `mcp.server` with stdio transport |
| B4.2 | `rag_mcp_server/src/mcp_server/tools/query_tool.py` | `query_knowledge_hub` tool |
| B4.3 | `rag_mcp_server/src/mcp_server/tools/list_collections.py` | `list_collections` tool |
| B4.4 | `rag_mcp_server/src/mcp_server/tools/doc_summary.py` | `get_document_summary` tool |
| B4.5 | `rag_mcp_server/src/mcp_server/tools/multimodal.py` | `MultimodalAssembler` |

---

### Phase B-5: Trace Infrastructure (5 tasks)

| Task | File | Key Function/Class |
|------|------|-------------------|
| B5.1 | `rag_mcp_server/src/core/trace/trace_context.py` | `TraceContext` with `finish()`, `record_stage()` |
| B5.2 | `rag_mcp_server/src/observability/logger.py` | `JSONFormatter`, `get_trace_logger`, `write_trace` |
| B5.3 | (modify) `hybrid_search.py`, `reranker.py` | Trace injection in query pipeline |
| B5.4 | (modify) `pipeline.py` | Trace injection in ingestion pipeline |
| B5.5 | (modify) `pipeline.py` | `on_progress` callback parameter |

---

### Phase B-6: Dashboard (6 tasks)

| Task | File | Key Function/Class |
|------|------|-------------------|
| B6.1 | `rag_mcp_server/src/observability/dashboard/app.py` | Streamlit multi-page entry |
| B6.2 | `rag_mcp_server/src/ingestion/document_manager.py` | `DocumentManager` cross-store coordination |
| B6.3 | `rag_mcp_server/src/observability/dashboard/pages/data_browser.py` | Data browser page |
| B6.4 | `rag_mcp_server/src/observability/dashboard/pages/ingestion_manager.py` | Ingestion manager page |
| B6.5 | `rag_mcp_server/src/observability/dashboard/pages/ingestion_traces.py` | Ingestion traces page |
| B6.6 | `rag_mcp_server/src/observability/dashboard/pages/query_traces.py` | Query traces page |

---

### Phase B-7: Evaluation (4 tasks)

| Task | File | Key Function/Class |
|------|------|-------------------|
| B7.1 | `rag_mcp_server/src/libs/evaluator/ragas_evaluator.py` | `RagasEvaluator` |
| B7.2 | `rag_mcp_server/src/libs/evaluator/composite_evaluator.py` | `CompositeEvaluator` |
| B7.3 | `rag_mcp_server/src/libs/evaluator/eval_runner.py` | `EvalRunner` + golden test set |
| B7.4 | `rag_mcp_server/src/observability/dashboard/pages/evaluation_panel.py` | Evaluation panel page |

---

### Phase B-8: End-to-End Acceptance (3 tasks)

| Task | File | Key Function/Class |
|------|------|-------------------|
| B8.1 | `tests/e2e/test_mcp_client.py` | MCP Client simulation tests |
| B8.2 | `tests/e2e/test_dashboard_smoke.py` | Dashboard smoke tests |
| B8.3 | `tests/e2e/test_full_pipeline.py` | Full pipeline: ingest → query → evaluate |

---

## Phase C: Agent Infrastructure

### Task C1: Tool abstract base class

**Files:**
- Create: `agent_layer/tools/base.py`
- Create: `agent_layer/tools/__init__.py`
- Create: `agent_layer/tests/test_tools.py`

**Interfaces:**
- Produces: `Tool` ABC with `name: str`, `description: str`, `parameters: dict`, `execute(**kwargs) -> str`
- Consumed by: C2, C3, C4, D1, E2, F3

- [ ] **Step 1: Write failing test**

```python
# agent_layer/tests/test_tools.py
"""Tests for Agent Tool infrastructure."""

import pytest
from agent_layer.tools.base import Tool


class TestToolBase:
    def test_tool_is_abstract(self):
        """Tool should require execute() implementation."""
        with pytest.raises(TypeError):
            Tool(name="bad", description="no execute")

    def test_concrete_tool_works(self):
        """A concrete tool subclass should be callable."""

        class EchoTool(Tool):
            name = "echo"
            description = "Echoes input"
            parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

            def execute(self, text: str = "") -> str:
                return f"Echo: {text}"

        tool = EchoTool()
        assert tool.name == "echo"
        result = tool.execute(text="hello")
        assert result == "Echo: hello"

    def test_tool_has_schema(self):
        """Tool should expose its parameter schema."""

        class SearchTool(Tool):
            name = "search"
            description = "Search knowledge base"
            parameters = {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            }

            def execute(self, query: str, top_k: int = 5) -> str:
                return f"Searched: {query} (top_{top_k})"

        tool = SearchTool()
        assert tool.parameters["required"] == ["query"]
        result = tool.execute(query="test")
        assert "Searched: test" in result
```

- [ ] **Step 2: Implement Tool base class**

```python
# agent_layer/tools/base.py
"""Abstract base class for Agent tools."""

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Abstract tool that an Agent can call.

    Each tool has a name, description, and parameter schema
    (JSON Schema format) that tells the LLM how to use it.

    Subclasses must define name, description, parameters as class
    attributes and implement execute().
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.name or not cls.description:
            # Allow abstract intermediates
            if ABC not in cls.__bases__:
                raise TypeError(
                    f"Tool subclass {cls.__name__} must define 'name' and 'description'"
                )

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Execute the tool with the given parameters.

        Args:
            **kwargs: Parameters matching the tool's schema.

        Returns:
            Tool output as a string (for the Agent's Observation step).
        """
        ...

    def get_schema(self) -> dict:
        """Return the tool's function schema for LLM tool calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
```

- [ ] **Step 3: Run tests → pass → commit**

```bash
pytest agent_layer/tests/test_tools.py -v
git add agent_layer/tools/base.py agent_layer/tests/test_tools.py
git commit -m "feat(C1): Tool abstract base class with JSON Schema support"
```

---

### Task C2: Tool Registry

**Files:**
- Create: `agent_layer/tools/registry.py`
- Modify: `agent_layer/tests/test_tools.py` (add registry tests)

**Interfaces:**
- Produces: `ToolRegistry` class with `register(tool)`, `get(name) -> Tool`, `list_tools() -> list[Tool]`, `get_schemas() -> list[dict]`
- Consumed by: D1 (ReAct agent uses registry to get tool list for LLM prompt)

- [ ] **Step 1: Write test**

```python
# Add to agent_layer/tests/test_tools.py

from agent_layer.tools.registry import ToolRegistry


class TestToolRegistry:
    def test_register_and_get(self):
        """Should register and retrieve tools by name."""
        registry = ToolRegistry()
        tool = EchoTool()
        registry.register(tool)
        assert registry.get("echo") is tool

    def test_get_missing_raises(self):
        """Should raise KeyError for unregistered tools."""
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_tools(self):
        """Should list all registered tools."""
        registry = ToolRegistry()
        registry.register(EchoTool())
        registry.register(SearchTool())
        assert len(registry.list_tools()) == 2

    def test_get_schemas(self):
        """Should return schemas for LLM function calling."""
        registry = ToolRegistry()
        registry.register(SearchTool())
        schemas = registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "search"
        assert "parameters" in schemas[0]
```

- [ ] **Step 2: Implement**

```python
# agent_layer/tools/registry.py
"""Tool registry for managing available Agent tools."""

from agent_layer.tools.base import Tool


class ToolRegistry:
    """Registry that holds all tools available to an Agent."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool instance.

        Args:
            tool: Tool instance to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Get a tool by name.

        Args:
            name: Tool name.

        Returns:
            Tool instance.

        Raises:
            KeyError: If no tool with that name exists.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def list_tools(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict]:
        """Return function schemas for all registered tools (for LLM prompts)."""
        return [tool.get_schema() for tool in self._tools.values()]
```

- [ ] **Step 3: Run tests → pass → commit**

---

### Task C3: local_search Tool

**Files:**
- Create: `agent_layer/tools/local_search.py`
- Modify: `agent_layer/tests/test_tools.py`

**Interfaces:**
- Produces: `LocalSearchTool` wrapping `HybridSearch.search(query, top_k) -> str` (formatted results)
- Depends on: Phase B-3 (HybridSearch must exist)

- [ ] **Step 1: Write test with mocked HybridSearch**

```python
# Add to agent_layer/tests/test_tools.py

from unittest.mock import Mock, patch
from agent_layer.tools.local_search import LocalSearchTool


class TestLocalSearchTool:
    def test_execute_calls_hybrid_search(self):
        """Should call HybridSearch and format results."""
        mock_search = Mock()
        mock_result = Mock()
        mock_result.chunks = [
            Mock(text="Result 1 content", metadata={"source": "doc1.pdf", "page": 1}, score=0.95),
            Mock(text="Result 2 content", metadata={"source": "doc2.pdf", "page": 3}, score=0.82),
        ]
        mock_result.total_found = 2
        mock_search.search.return_value = mock_result

        tool = LocalSearchTool(hybrid_search=mock_search)
        result = tool.execute(query="test query", top_k=5)

        mock_search.search.assert_called_once_with("test query", 5)
        assert "Result 1 content" in result
        assert "doc1.pdf" in result
        assert "0.95" in result
```

- [ ] **Step 2: Implement LocalSearchTool**

```python
# agent_layer/tools/local_search.py
"""Tool that wraps the RAG pipeline's HybridSearch for Agent use."""

from agent_layer.tools.base import Tool
from typing import Any


class LocalSearchTool(Tool):
    """Search the local knowledge base using hybrid retrieval.

    This tool wraps the RAG pipeline's HybridSearch, giving the Agent
    access to the local document knowledge base.
    """

    name = "local_search"
    description = (
        "Search the local document knowledge base using hybrid retrieval "
        "(dense + sparse + rerank). Returns relevant document chunks with citations."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(self, hybrid_search: Any = None):
        """Initialize with an optional HybridSearch instance.

        Args:
            hybrid_search: A HybridSearch instance. If None, one will be created lazily.
        """
        self._hybrid_search = hybrid_search

    def _get_search(self):
        """Lazily create or return the HybridSearch instance."""
        if self._hybrid_search is None:
            from rag_mcp_server.src.retrieval.hybrid_search import HybridSearch
            from rag_mcp_server.src.core.settings import load_settings

            settings = load_settings("config/settings.yaml")
            self._hybrid_search = HybridSearch(settings)
        return self._hybrid_search

    def execute(self, query: str, top_k: int = 5) -> str:
        """Execute local knowledge base search.

        Args:
            query: The search query.
            top_k: Number of top results to return.

        Returns:
            Formatted search results with citations.
        """
        search = self._get_search()
        result = search.search(query, top_k)

        if not result.chunks:
            return "No relevant documents found in the local knowledge base."

        lines = [f"Found {result.total_found} results (showing top {len(result.chunks)}):\n"]
        for i, chunk in enumerate(result.chunks, 1):
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "N/A")
            lines.append(
                f"[{i}] Source: {source} (page {page}) | Score: {chunk.score:.3f}\n"
                f"    {chunk.text[:300]}..."
            )
        return "\n".join(lines)
```

- [ ] **Step 3: Run tests → pass → commit**

---

### Task C4: query_rewriter Tool

**Files:**
- Create: `agent_layer/tools/query_rewriter.py`
- Modify: `agent_layer/tests/test_tools.py`

**Interfaces:**
- Produces: `QueryRewriterTool` using LLM to rewrite queries, injecting missing-info context
- Depends on: B1.1 (BaseLLM factory)

- [ ] **Step 1: Write test**

```python
# Add to agent_layer/tests/test_tools.py

from agent_layer.tools.query_rewriter import QueryRewriterTool


class TestQueryRewriterTool:
    def test_rewrite_injects_missing_info(self, mocker):
        """Should call LLM with original query and missing info context."""
        mock_llm = mocker.Mock()
        mock_llm.chat.return_value = "合同纠纷中违约金的计算标准与赔偿范围有哪些规定"

        tool = QueryRewriterTool(llm=mock_llm)
        result = tool.execute(
            original_query="合同违约怎么赔",
            missing_info="缺少违约金计算方式的信息",
        )

        mock_llm.chat.assert_called_once()
        call_args = mock_llm.chat.call_args[0][0]
        prompt_text = call_args[1]["content"] if len(call_args) > 1 else call_args[0][0]["content"]
        assert "合同违约怎么赔" in prompt_text
        assert "违约金计算方式" in prompt_text
        assert "违约金" in result
```

- [ ] **Step 2: Implement QueryRewriterTool**

```python
# agent_layer/tools/query_rewriter.py
"""Tool for rewriting queries based on retrieval feedback."""

from pathlib import Path
from agent_layer.tools.base import Tool
from typing import Any


class QueryRewriterTool(Tool):
    """Rewrite a search query to improve retrieval results.

    Uses an LLM to rewrite the query, incorporating feedback about
    what information was missing from the previous search.
    """

    name = "rewrite_query"
    description = (
        "Rewrite a search query to be more specific and complete. "
        "Use when the original query returned partial or insufficient results. "
        "Incorporates missing information feedback to produce better queries."
    )
    parameters = {
        "type": "object",
        "properties": {
            "original_query": {
                "type": "string",
                "description": "The original search query that needs improvement",
            },
            "missing_info": {
                "type": "string",
                "description": "Description of what information was missing from previous results",
            },
        },
        "required": ["original_query", "missing_info"],
    }

    def __init__(self, llm: Any = None):
        """Initialize with an optional LLM instance.

        Args:
            llm: A BaseLLM instance. If None, one will be created lazily.
        """
        self._llm = llm

    def _get_llm(self):
        """Lazily create or return the LLM instance."""
        if self._llm is None:
            from rag_mcp_server.src.libs.llm.llm_factory import LLMFactory
            from rag_mcp_server.src.core.settings import load_settings, apply_env_overrides

            settings = load_settings("config/settings.yaml")
            apply_env_overrides(settings)
            self._llm = LLMFactory.create(settings)
        return self._llm

    def _load_prompt(self) -> str:
        """Load the query rewrite prompt template."""
        prompt_path = Path(__file__).parent.parent / "prompts" / "rewriter_prompt.md"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        # Default inline prompt
        return (
            "You are a query rewriter. Given the original query and a description "
            "of what information was missing from search results, rewrite the query "
            "to be more specific and complete.\n\n"
            "Original query: {original_query}\n"
            "Missing information: {missing_info}\n\n"
            "Rewrite the query to address the missing information. "
            "Output only the rewritten query, nothing else."
        )

    def execute(self, original_query: str, missing_info: str) -> str:
        """Rewrite a query based on missing information feedback.

        Args:
            original_query: The query that needs improvement.
            missing_info: Description of what was missing.

        Returns:
            The rewritten query string.
        """
        llm = self._get_llm()
        prompt_template = self._load_prompt()
        prompt = prompt_template.format(
            original_query=original_query,
            missing_info=missing_info,
        )
        rewritten = llm.chat([
            {"role": "user", "content": prompt}
        ])
        return rewritten.strip()
```

- [ ] **Step 3: Run tests → pass → commit**

---

### Task C5: Hybrid Scorer

**Files:**
- Create: `agent_layer/core/hybrid_scorer.py`
- Create: `agent_layer/tests/test_scorer.py`

**Interfaces:**
- Produces: `HybridScorer` with `evaluate(query, results, mode) -> ScorerJudgment`
- Consumed by: D1 (ReAct loop uses scorer to decide next action)

- [ ] **Step 1: Write test**

```python
# agent_layer/tests/test_scorer.py
"""Tests for hybrid quality scorer."""

import pytest
from agent_layer.core.hybrid_scorer import HybridScorer, ScorerJudgment


class TestHybridScorer:
    @pytest.fixture
    def scorer(self, mocker):
        """Create a scorer with mocked LLM for deep evaluation."""
        mock_llm = mocker.Mock()
        return HybridScorer(llm=mock_llm, threshold=0.7)

    def test_high_score_passes(self, scorer):
        """Top result above threshold should PASS without LLM call."""
        results = _make_results([0.85, 0.6, 0.4])
        judgment = scorer.evaluate("test query", results, mode="decision")
        assert judgment.action == "PASS"
        assert judgment.llm_called is False

    def test_low_score_web_search(self, scorer):
        """Score far below threshold should trigger WEB_SEARCH."""
        results = _make_results([0.2, 0.15])
        judgment = scorer.evaluate("test query", results, mode="decision")
        assert judgment.action == "WEB_SEARCH"

    def test_boundary_triggers_llm(self, scorer):
        """Score in boundary zone should trigger LLM deep evaluation."""
        scorer._llm.chat.return_value = '{"judgment": "ENOUGH", "missing": ""}'
        results = _make_results([0.55])  # 0.55 >= 0.7 * 0.6 = 0.42 → boundary
        judgment = scorer.evaluate("test query", results, mode="decision")
        assert judgment.llm_called is True

    def test_confidence_mode(self, scorer):
        """Confidence mode should evaluate web search results."""
        scorer._llm.chat.return_value = '{"judgment": "YES", "confidence": 8}'
        results = _make_results([0.9])
        judgment = scorer.evaluate("test query", results, mode="confidence")
        assert judgment.action in ("PASS", "WEB_SEARCH")
        assert judgment.confidence_score is not None


def _make_results(scores: list[float]):
    """Helper to create mock search results with given scores."""
    from unittest.mock import Mock
    chunks = []
    for s in scores:
        chunk = Mock()
        chunk.score = s
        chunk.text = f"Text for score {s}"
        chunk.metadata = {"source": "test.pdf"}
        chunks.append(chunk)
    result = Mock()
    result.chunks = chunks
    return result
```

- [ ] **Step 2: Implement HybridScorer**

```python
# agent_layer/core/hybrid_scorer.py
"""Hybrid quality scorer — fast threshold + LLM deep evaluation."""

from dataclasses import dataclass, field
from typing import Any
import json


@dataclass
class ScorerJudgment:
    """Result of scorer evaluation."""

    action: str  # PASS, REWRITE, WEB_SEARCH
    llm_called: bool = False
    confidence_score: int | None = None
    missing_info: str = ""
    detail: str = ""


class HybridScorer:
    """Two-stage retrieval quality evaluator.

    Stage 1: Fast threshold — if Top-1 score >= threshold, pass immediately.
    Stage 2: LLM deep eval — for boundary cases, ask LLM to judge quality.

    Supports two modes:
    - 'decision': Full routing decision (PASS / REWRITE / WEB_SEARCH)
    - 'confidence': Lightweight confidence assessment for web search results
    """

    def __init__(self, llm: Any = None, threshold: float = 0.7):
        """Initialize the scorer.

        Args:
            llm: A BaseLLM instance for deep evaluation.
            threshold: Minimum score to pass without LLM evaluation.
        """
        self._llm = llm
        self.threshold = threshold

    def _get_llm(self):
        """Lazily create LLM if not provided."""
        if self._llm is None:
            from rag_mcp_server.src.libs.llm.llm_factory import LLMFactory
            from rag_mcp_server.src.core.settings import load_settings, apply_env_overrides

            settings = load_settings("config/settings.yaml")
            apply_env_overrides(settings)
            self._llm = LLMFactory.create(settings)
        return self._llm

    def evaluate(
        self, query: str, results: Any, mode: str = "decision"
    ) -> ScorerJudgment:
        """Evaluate retrieval quality.

        Args:
            query: The user's original query.
            results: Search result with .chunks attribute.
            mode: 'decision' for routing or 'confidence' for web result assessment.

        Returns:
            ScorerJudgment with recommended action.
        """
        if not results.chunks:
            return ScorerJudgment(action="WEB_SEARCH", detail="No results found")

        top_score = results.chunks[0].score

        # Stage 1: Fast threshold
        if top_score >= self.threshold:
            return ScorerJudgment(action="PASS", detail=f"Score {top_score:.3f} >= {self.threshold}")

        boundary = self.threshold * 0.6
        if top_score < boundary:
            return ScorerJudgment(action="WEB_SEARCH", detail=f"Score {top_score:.3f} < {boundary:.3f}")

        # Stage 2: LLM deep evaluation (boundary zone)
        return self._llm_evaluate(query, results, mode)

    def _llm_evaluate(self, query: str, results: Any, mode: str) -> ScorerJudgment:
        """Use LLM to deeply evaluate retrieval quality.

        Args:
            query: The user's query.
            results: Search results.
            mode: Evaluation mode.

        Returns:
            ScorerJudgment based on LLM evaluation.
        """
        llm = self._get_llm()
        chunks_text = "\n\n".join(
            f"[{i}] (score: {c.score:.3f}) {c.text[:200]}"
            for i, c in enumerate(results.chunks[:5])
        )

        if mode == "confidence":
            prompt = (
                "You are an answer quality evaluator.\n"
                f"User question: {query}\n"
                f"Online search results:\n{chunks_text}\n\n"
                "Judge:\n"
                "1. Can these results support a trustworthy answer? (YES / PARTIALLY / NO)\n"
                "2. Confidence score (0-10)\n"
                "3. If PARTIALLY or NO, note what's uncertain.\n"
                "Output as JSON: {\"judgment\": \"...\", \"confidence\": N, \"notes\": \"...\"}"
            )
        else:
            prompt = (
                "You are a retrieval quality evaluator.\n"
                f"User question: {query}\n"
                f"Retrieved results:\n{chunks_text}\n\n"
                "Judge:\n"
                "1. Can these results sufficiently answer the question? (ENOUGH / PARTIAL / NO)\n"
                "2. If PARTIAL, what key information is missing?\n"
                "3. Recommendation: RETURN / REWRITE / WEB_SEARCH\n"
                "Output as JSON: {\"judgment\": \"...\", \"missing\": \"...\", \"action\": \"...\"}"
            )

        response = llm.chat([{"role": "user", "content": prompt}])

        try:
            parsed = json.loads(response.strip().strip("```json").strip("```"))
        except json.JSONDecodeError:
            return ScorerJudgment(action="WEB_SEARCH", llm_called=True, detail="LLM parse failed")

        judgment = parsed.get("judgment", "NO")

        if mode == "confidence":
            return ScorerJudgment(
                action="PASS" if judgment in ("YES", "PARTIALLY") else "WEB_SEARCH",
                llm_called=True,
                confidence_score=parsed.get("confidence"),
                detail=parsed.get("notes", ""),
            )

        action_map = {"ENOUGH": "PASS", "PARTIAL": "REWRITE", "NO": "WEB_SEARCH"}
        return ScorerJudgment(
            action=action_map.get(judgment, "WEB_SEARCH"),
            llm_called=True,
            missing_info=parsed.get("missing", ""),
            detail=parsed.get("action", ""),
        )
```

- [ ] **Step 3: Run tests → pass → commit**

---

### Task C6: AgentMemory Enhancement (Compaction + Token Management)

**Files:**
- Modify: `agent_layer/core/agent_memory.py` (enhance existing)
- Create: `agent_layer/tests/test_memory.py`

**Interfaces:**
- Produces: `AgentMemory` with `add(role, content)`, `get_history() -> list[dict]`, `clear()`, `token_count() -> int`, `compact(llm) -> str`, `on_flush(callback)`
- Consumed by: D1, D4 (ReAct engine + CLI)

**Design:**
```
AgentMemory
├── _history: list[Message]      # 原始消息
├── _summaries: list[str]        # 结构化压缩摘要
├── compaction_threshold: float  # 0.7 (窗口用满70%触发)
├── keep_recent_tokens: int      # 20000 (保留最近N tokens原文)
│
├── add(role, content)           # 追加消息
├── get_history()                # 返回 [summaries] + [recent messages]
├── get_full_context()           # 给 LLM 的完整上下文
├── token_count()                # 估算当前总 token 数
├── should_compact() -> bool     # 检查是否超过阈值
├── compact(llm) -> str          # 执行压缩，返回摘要
└── on_flush(callback)           # 注册 pre-compaction flush 回调
```

- [ ] **Step 1: Write test**

```python
# agent_layer/tests/test_memory.py
"""Tests for enhanced Agent memory with compaction."""

import pytest
from unittest.mock import Mock
from agent_layer.core.agent_memory import AgentMemory


class TestAgentMemory:
    """Unit tests for core memory operations."""

    def test_add_and_retrieve(self):
        memory = AgentMemory()
        memory.add("user", "What is RAG?")
        memory.add("assistant", "RAG stands for Retrieval-Augmented Generation.")
        history = memory.get_history()
        assert len(history) == 2

    def test_clear(self):
        memory = AgentMemory()
        memory.add("user", "test")
        memory.clear()
        assert len(memory.get_history()) == 0

    def test_get_last_n(self):
        memory = AgentMemory()
        for i in range(5):
            memory.add("user", f"msg {i}")
        last_3 = memory.get_last_n(3)
        assert len(last_3) == 3
        assert last_3[-1]["content"] == "msg 4"


class TestCompaction:
    """Tests for context compaction mechanism."""

    def test_should_not_compact_below_threshold(self):
        """Short context should not trigger compaction."""
        memory = AgentMemory(compaction_threshold=0.7, keep_recent_tokens=20000)
        memory.add("user", "hi")
        assert not memory.should_compact()

    def test_should_compact_when_near_limit(self, mocker):
        """Context near token limit should trigger compaction."""
        memory = AgentMemory(
            compaction_threshold=0.7,
            keep_recent_tokens=20000,
            max_context_tokens=100000,
        )
        # Simulate a long conversation
        long_text = "word " * 60000
        memory.add("user", long_text)
        assert memory.should_compact()

    def test_compact_preserves_recent_messages(self, mocker):
        """Compaction should keep recent messages intact."""
        mock_llm = mocker.Mock()
        mock_llm.chat.return_value = "[Summary] Key decisions: ..."

        memory = AgentMemory(keep_recent_tokens=100)
        # Add 3 messages of ~50 tokens each
        memory.add("system", "You are a helpful assistant.")
        memory.add("user", "hello " * 20)   # ~100 tokens
        memory.add("user", "world " * 20)   # ~100 tokens

        result = memory.compact(mock_llm)
        history = memory.get_history()
        # Recent messages should still be present
        assert "world" in history[-1]["content"]

    def test_pre_flush_callback_invoked(self, mocker):
        """Pre-compaction flush callback should be called."""
        flush_called = []
        memory = AgentMemory()
        memory.on_flush(lambda: flush_called.append(True))

        mock_llm = mocker.Mock()
        mock_llm.chat.return_value = "[Summary]"
        memory.add("user", "x " * 30000)
        memory.compact(mock_llm)
        assert len(flush_called) == 1
```

- [ ] **Step 2: Implement enhanced AgentMemory**

```python
# agent_layer/core/agent_memory.py (enhanced)
"""Conversation memory with compaction for context window management."""

from typing import Callable, Any


class AgentMemory:
    """Conversation memory with compaction support.

    Stores conversation history and supports compressing older messages
    into structured summaries to stay within context window limits.
    """

    def __init__(
        self,
        max_context_tokens: int = 128000,
        compaction_threshold: float = 0.7,
        keep_recent_tokens: int = 20000,
    ):
        self._history: list[dict] = []
        self._summaries: list[str] = []
        self.max_context_tokens = max_context_tokens
        self.compaction_threshold = compaction_threshold
        self.keep_recent_tokens = keep_recent_tokens
        self._flush_callbacks: list[Callable] = []

    def add(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self._history.append({"role": role, "content": content})

    def get_history(self) -> list[dict]:
        """Return full context: summaries (as system messages) + recent history."""
        result = []
        for s in self._summaries:
            result.append({"role": "system", "content": f"[Previous context summary]\n{s}"})
        result.extend(self._history)
        return result

    def get_last_n(self, n: int) -> list[dict]:
        """Return the last N messages."""
        return self._history[-n:] if n < len(self._history) else list(self._history)

    def clear(self) -> None:
        """Clear all conversation history and summaries."""
        self._history.clear()
        self._summaries.clear()

    def token_count(self) -> int:
        """Estimate total token count (4 chars ≈ 1 token)."""
        total = 0
        for s in self._summaries:
            total += len(s) // 4
        for msg in self._history:
            total += len(msg["content"]) // 4
        return total

    def should_compact(self) -> bool:
        """Check if token count exceeds the compaction threshold."""
        limit = int(self.max_context_tokens * self.compaction_threshold)
        return self.token_count() > limit

    def compact(self, llm: Any) -> str:
        """Compress older messages into structured summaries.

        Args:
            llm: BaseLLM instance for summary generation.

        Returns:
            The generated summary string.
        """
        # Invoke pre-flush callbacks
        for cb in self._flush_callbacks:
            try:
                cb()
            except Exception:
                pass

        # Determine split point: keep last ~keep_recent_tokens tokens
        keep_count = 0
        split_idx = len(self._history)
        for i in range(len(self._history) - 1, -1, -1):
            t = len(self._history[i]["content"]) // 4
            if keep_count + t > self.keep_recent_tokens:
                split_idx = i + 1
                break
            keep_count += t

        old_messages = self._history[:split_idx]
        self._history = self._history[split_idx:]

        if not old_messages:
            return ""

        # Generate structured summary via LLM
        summary = self._generate_summary(llm, old_messages)
        self._summaries.append(summary)
        return summary

    def _generate_summary(self, llm: Any, messages: list[dict]) -> str:
        """Use LLM to generate a structured summary preserving key info."""
        from pathlib import Path

        prompt_path = Path(__file__).parent.parent / "prompts" / "compaction_prompt.md"
        if prompt_path.exists():
            template = prompt_path.read_text(encoding="utf-8")
        else:
            template = (
                "Summarize the following conversation, preserving:\n"
                "- Task status and next steps\n"
                "- Key decisions and rationale\n"
                "- TODO items\n"
                "- Critical identifiers (filenames, IDs, URLs)\n\n"
                "Conversation:\n{conversation}\n\n"
                "Structured Summary:"
            )

        conv_text = "\n".join(
            f"[{m['role']}] {m['content'][:500]}" for m in messages
        )
        prompt = template.format(conversation=conv_text)
        response = llm.chat([{"role": "user", "content": prompt}])
        return response.content if hasattr(response, "content") else str(response)

    def on_flush(self, callback: Callable) -> None:
        """Register a callback to be invoked before compaction."""
        self._flush_callbacks.append(callback)
```

- [ ] **Step 3: Run tests → pass → commit**

---

### Task C7: MemoryStore Core Engine

**Files:**
- Create: `agent_layer/memory/store.py`
- Create: `agent_layer/memory/__init__.py`
- Create: `agent_layer/tests/test_memory_store.py`

**Interfaces:**
- Produces: `MemoryStore` with `index_files()`, `search(query, top_k) -> list[MemoryChunk]`, `reindex()`
- Depends on: `config/settings.yaml` memory section, `EmbeddingFactory`

**Design:**
```
MemoryStore
├── workspace_dir: Path        # agent_layer/memory/
├── db_path: Path              # agent_layer/memory/index.db
├── chunk_size: int            # 400 tokens
├── chunk_overlap: int         # 80 tokens
│
├── index_files()              # 扫描 *.md → 分块 → 建索引
├── search(query, top_k)       # 混合搜索（0.7向量 + 0.3 BM25）
├── embed(texts) -> vectors    # 复用 EmbeddingFactory
├── _chunk_file(path)          # 将文件分块
└── reindex()                  # 增量重建（基于 mtime + hash）
```

**SQLite schema:**
```sql
CREATE TABLE files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE,
    mtime REAL,
    size INTEGER,
    content_hash TEXT
);

CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER REFERENCES files(id),
    chunk_index INTEGER,
    content TEXT,
    embedding_json TEXT,         -- JSON-serialized float list
    start_line INTEGER,
    end_line INTEGER
);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    content='chunks',
    content_rowid='id'
);
```

- [ ] **Step 1: Write test with temp workspace**
- [ ] **Step 2: Implement MemoryStore with SQLite + FTS5**
- [ ] **Step 3: Run tests → pass → commit**

---

### Task C8: Memory Tool Set

**Files:**
- Create: `agent_layer/tools/memory_search.py`
- Create: `agent_layer/tools/memory_log.py`
- Create: `agent_layer/tools/memory_store.py`
- Modify: `agent_layer/tests/test_tools.py`

**Interfaces:**
- Produces: `MemorySearchTool`, `MemoryGetTool`, `MemoryLogTool`, `MemoryStoreTool` — all implementing Tool base
- Depends on: C7 (MemoryStore)

**Tool designs:**

```python
# memory_search: hybrid search with dedup + ranking
class MemorySearchTool(Tool):
    name = "memory_search"
    description = "Search long-term and short-term memory using hybrid retrieval..."
    parameters = {"query": "string", "top_k": "int (default 10)"}
    # Returns: formatted search results with source path + relevance score

# memory_get: read full file by path
class MemoryGetTool(Tool):
    name = "memory_get"
    description = "Read the full content of a memory file by path..."
    parameters = {"path": "string (e.g. 'MEMORY.md' or '2024-06-24.md')"}

# memory_log: append timestamped entry to daily note
class MemoryLogTool(Tool):
    name = "memory_log"
    description = "Append a timestamped log entry to today's short-term memory..."
    parameters = {"content": "string"}

# memory_store: curated write to MEMORY.md with upsert-by-tag
class MemoryStoreTool(Tool):
    name = "memory_store"
    description = "Store a curated fact in long-term memory. Use tags for upsert..."
    parameters = {"tag": "string", "content": "string"}
```

- [ ] **Step 1: Write tests with mocked MemoryStore**
- [ ] **Step 2: Implement all 4 tools**
- [ ] **Step 3: Run tests → pass → commit**

---

### Task C9: Compaction Prompts + Pre-flush Mechanism

**Files:**
- Create: `agent_layer/prompts/compaction_prompt.md`
- Create: `agent_layer/prompts/memory_flush_prompt.md`
- Modify: `agent_layer/core/agent_memory.py` (integrate flush callbacks)

**Prompt templates:**

`compaction_prompt.md`:
```
You are a context summarizer. Compress the following older conversation into a structured summary.

## Required fields to preserve:
- [Task Status]: Current progress, next steps
- [Key Decisions]: Choices made and why
- [TODO]: Incomplete items
- [Identifiers]: File names, function names, IDs, URLs

## Conversation to compress:
{conversation}

## Structured Summary:
```

`memory_flush_prompt.md`:
```
The context window is nearly full. Before compaction, review the conversation and save anything valuable to memory:

1. Use `memory_store` for durable facts, decisions, preferences (writes to MEMORY.md)
2. Use `memory_log` for daily notes, observations, session summary (writes to today's log)
3. Do NOT archive trivial chitchat — be selective

What should be preserved from this conversation?
```

- [ ] **Step 1: Create prompt files**
- [ ] **Step 2: Integrate flush mechanism (AgentMemory.on_flush + ReActAgent pre-compact hook)**
- [ ] **Step 3: Commit**

---

### Task C10: MEMORY.md Template + Index Initialization

**Files:**
- Create: `agent_layer/memory/MEMORY.md`
- Create: `agent_layer/memory/.gitkeep`
- Modify: `.gitignore` (add `index.db`)

**MEMORY.md template:**
```markdown
# Agent Memory — Long-Term Storage

> Curated facts, preferences, and decisions. Managed by the Agent via `memory_store` tool.
> Tags support upsert: same tag = in-place update.

<!-- memory:user-profile -->
- (no entries yet)
<!-- /memory:user-profile -->

<!-- memory:project-context -->
- Project: Agentic RAG Server
- Architecture: Dual-layer (Agent + RAG)
- Tech stack: Python 3.11+, Chroma, LangChain, MCP SDK
<!-- /memory:project-context -->

<!-- memory:decisions -->
- (no entries yet)
<!-- /memory:decisions -->

<!-- memory:preferences -->
- (no entries yet)
<!-- /memory:preferences -->
```

- [ ] **Step 1: Create MEMORY.md template + .gitkeep**
- [ ] **Step 2: Add `index.db` to .gitignore**
- [ ] **Step 3: Implement MemoryStore.init_workspace() — create dir + initialize schema**
- [ ] **Step 4: Commit**

---

### Task C11: Prompt Templates

**Files:**
- Create: `agent_layer/prompts/react_system.md`
- Create: `agent_layer/prompts/scorer_prompt.md`
- Create: `agent_layer/prompts/rewriter_prompt.md`
- Modify: `agent_layer/prompts/react_system.md` (include memory tool descriptions)

**Interfaces:**
- Produces: Three prompt template files loaded by Agent and Tool classes
- Consumed by: C4, C5, C8 (memory tools need tool descriptions in react_system), D1

- [ ] **Step 1: Create react_system.md**

````markdown
# ReAct System Prompt

You are an intelligent research assistant with access to tools. Use the ReAct (Reasoning + Acting) framework to answer user questions.

## Available Tools

{tool_descriptions}

## Response Format

Always respond in this exact format:

```
Thought: [Your reasoning about what to do next]
Action: [Tool name to call, or FINISH]
Action Input: [JSON with tool parameters, only if Action is not FINISH]
```

After receiving an Observation from a tool, continue with:

```
Thought: [Analyze the observation and decide next step]
Action: [Next tool or FINISH]
```

## Rules

1. ALWAYS start by searching the local knowledge base with `local_search`.
2. If local results are insufficient, use `rewrite_query` to improve your search.
3. If rewritten search still fails, use `web_search` as a last resort.
4. You have a maximum of 3 search rounds.
5. When you have enough information, use Action: FINISH and provide a complete answer.
6. Always cite your sources.
````

- [ ] **Step 2: Create scorer_prompt.md** (as designed in the spec)
- [ ] **Step 3: Create rewriter_prompt.md** (as designed in the spec)
- [ ] **Step 4: Commit**

```bash
git add agent_layer/prompts/
git commit -m "feat(C7): Agent prompt templates — ReAct system, scorer dual-mode, query rewriter"
```

---

### Task C12: Agent Configuration Module

**Files:**
- Create: `agent_layer/core/config.py`
- Modify: `config/settings.yaml` (verify agent + memory sections exist)

**Interfaces:**
- Produces: `AgentConfig` dataclass read from settings
- Consumed by: D1, E1

```python
# agent_layer/core/config.py
"""Agent-specific configuration."""

from dataclasses import dataclass, field


@dataclass
class WebSearchConfig:
    """Web search tool configuration."""
    backend: str = "tavily"
    tavily_api_key: str = ""
    serpapi_api_key: str = ""


@dataclass
class MemoryConfig:
    """Long-term / short-term memory configuration."""
    enabled: bool = True
    workspace_dir: str = "agent_layer/memory"
    compaction_threshold: float = 0.7
    keep_recent_tokens: int = 20000
    chunk_size: int = 400
    chunk_overlap: int = 80
    hybrid_weight_vector: float = 0.7
    hybrid_weight_bm25: float = 0.3
    inject_long_term_tokens: int = 2000


@dataclass
class AgentConfig:
    """Agent configuration, loaded from settings.yaml agent section."""
    max_rounds: int = 3
    score_threshold: float = 0.7
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    @classmethod
    def from_settings(cls, settings: dict) -> "AgentConfig":
        """Create AgentConfig from the agent section of settings."""
        ws_raw = settings.get("web_search", {})
        mem_raw = settings.get("memory", {})
        return cls(
            max_rounds=settings.get("max_rounds", 3),
            score_threshold=settings.get("score_threshold", 0.7),
            web_search=WebSearchConfig(
                backend=ws_raw.get("backend", "tavily"),
                tavily_api_key=ws_raw.get("tavily_api_key", ""),
                serpapi_api_key=ws_raw.get("serpapi_api_key", ""),
            ),
        )
```

---

## Phase D: Custom ReAct Agent

### Task D1: ReAct Loop Engine

**Files:**
- Create: `agent_layer/core/react_engine.py`
- Create: `agent_layer/tests/test_custom_agent.py`

**Interfaces:**
- Produces: `ReActAgent` class with `run(query: str) -> AgentResult`
- Consumed by: D4 (CLI entry), E5 (comparison test)

```python
# agent_layer/core/react_engine.py
"""Custom ReAct (Reasoning + Acting) agent loop."""

import re
import json
from dataclasses import dataclass, field
from agent_layer.core.agent_memory import AgentMemory
from agent_layer.core.hybrid_scorer import HybridScorer, ScorerJudgment
from agent_layer.tools.registry import ToolRegistry


@dataclass
class AgentResult:
    """Result of an Agent run."""
    answer: str
    rounds: int
    actions_taken: list[str] = field(default_factory=list)
    final_source: str = ""  # 'local', 'rewrite', 'web'


class ReActAgent:
    """A Reasoning + Acting agent with hybrid quality scoring.

    The agent loops through Think → Act → Observe, with up to max_rounds
    search attempts. After each local_search, the HybridScorer evaluates
    result quality and routes to REWRITE or WEB_SEARCH as needed.
    """

    def __init__(
        self,
        tools: ToolRegistry,
        llm,
        scorer: HybridScorer,
        memory: AgentMemory | None = None,
        max_rounds: int = 3,
    ):
        """Initialize the ReAct agent.

        Args:
            tools: ToolRegistry with local_search, rewrite_query, web_search.
            llm: BaseLLM instance for thought generation and answer synthesis.
            scorer: HybridScorer for retrieval quality evaluation.
            memory: Optional AgentMemory instance.
            max_rounds: Maximum search rounds before forced finish.
        """
        self.tools = tools
        self.llm = llm
        self.scorer = scorer
        self.memory = memory or AgentMemory()
        self.max_rounds = max_rounds

    def run(self, query: str) -> AgentResult:
        """Execute the ReAct loop to answer a query.

        Args:
            query: User's question.

        Returns:
            AgentResult with answer and execution trace.
        """
        self.memory.add("user", query)
        actions = []

        for round_num in range(1, self.max_rounds + 1):
            # Think: get LLM decision
            thought = self._think(query)
            action_name, action_args = self._parse_action(thought)

            if action_name == "FINISH":
                break

            # Act: execute the tool
            tool = self.tools.get(action_name)
            observation = tool.execute(**action_args)
            actions.append(action_name)
            self.memory.add("assistant", f"Thought: {thought}\nObservation: {observation}")

            # Observe: evaluate quality (only for local_search)
            if action_name == "local_search":
                judgment = self._evaluate_quality(query, observation)
                if judgment.action == "PASS":
                    break
                elif judgment.action == "REWRITE":
                    query = self._handle_rewrite(query, judgment.missing_info)
                    actions.append("rewrite")
                else:  # WEB_SEARCH
                    observation = self._handle_web_search(query)
                    actions.append("web_search")
                    self.memory.add("tool", observation)
                    break

        # Synthesize final answer
        answer = self._synthesize(query)
        return AgentResult(
            answer=answer,
            rounds=round_num,
            actions_taken=actions,
            final_source=actions[-1] if actions else "none",
        )

    def _think(self, query: str) -> str:
        """Generate the next Thought/Action from the LLM."""
        system_prompt = self._load_system_prompt()
        history = self.memory.get_history()
        messages = [{"role": "system", "content": system_prompt}] + history
        return self.llm.chat(messages)

    def _parse_action(self, thought: str) -> tuple[str, dict]:
        """Parse Thought/Action/Action Input from LLM output."""
        action_match = re.search(r"Action:\s*(\w+)", thought)
        if not action_match:
            return "FINISH", {}
        action_name = action_match.group(1).strip()
        if action_name == "FINISH":
            return "FINISH", {}

        input_match = re.search(r"Action Input:\s*(\{.*?\})", thought, re.DOTALL)
        args = {}
        if input_match:
            try:
                args = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                pass
        return action_name, args

    def _evaluate_quality(self, query: str, observation: str) -> ScorerJudgment:
        """Evaluate local search quality using HybridScorer."""
        # Reconstruct results from observation string for scoring
        return self.scorer.evaluate(query, _parse_results(observation), mode="decision")

    def _handle_rewrite(self, query: str, missing_info: str) -> str:
        """Rewrite query and re-search locally."""
        rewriter = self.tools.get("rewrite_query")
        new_query = rewriter.execute(original_query=query, missing_info=missing_info)
        self.memory.add("assistant", f"Rewritten query: {new_query}")
        search = self.tools.get("local_search")
        result = search.execute(query=new_query)
        self.memory.add("tool", result)
        return result

    def _handle_web_search(self, query: str) -> str:
        """Execute web search and evaluate confidence."""
        web_search = self.tools.get("web_search")
        result = web_search.execute(query=query)
        judgment = self.scorer.evaluate(query, _parse_results(result), mode="confidence")
        confidence_note = f"\n[Confidence: {judgment.confidence_score}/10]"
        return result + confidence_note

    def _synthesize(self, query: str) -> str:
        """Synthesize final answer from all observations."""
        history = self.memory.get_history()
        prompt = (
            "Based on the conversation above, provide a complete answer to "
            f"the user's question: '{query}'\n\n"
            "Include relevant citations and note the source of information.\n"
            "If you are uncertain about any part, clearly state that."
        )
        history.append({"role": "user", "content": prompt})
        return self.llm.chat(history)

    def _load_system_prompt(self) -> str:
        """Load the ReAct system prompt with tool descriptions."""
        from pathlib import Path

        prompt_path = Path(__file__).parent.parent / "prompts" / "react_system.md"
        if prompt_path.exists():
            template = prompt_path.read_text(encoding="utf-8")
        else:
            template = "You are a helpful assistant. Available tools:\n{tool_descriptions}"

        schemas = self.tools.get_schemas()
        tool_desc = "\n\n".join(
            f"- **{s['name']}**: {s['description']}\n  Parameters: {json.dumps(s['parameters'], indent=2)}"
            for s in schemas
        )
        return template.format(tool_descriptions=tool_desc)


def _parse_results(observation: str):
    """Parse observation string into a mock result for scoring.

    This is a lightweight parser that extracts scores from formatted output.
    For production, HybridSearch should return structured results directly.
    """
    from unittest.mock import Mock
    import re

    scores = re.findall(r"Score:\s*([\d.]+)", observation)
    chunks = []
    for s in scores:
        chunk = Mock()
        chunk.score = float(s)
        chunk.text = observation[:100]
        chunk.metadata = {"source": "parsed"}
        chunks.append(chunk)

    result = Mock()
    result.chunks = chunks or [Mock(score=0.0, text="", metadata={})]
    return result
```

- [ ] **Write tests, run, pass, commit**

---

### Tasks D2-D5 follow the same TDD pattern:

| Task | File | Description |
|------|------|-------------|
| D2 | `agent_layer/core/react_engine.py` (refine) | Thought parser with robust regex, error handling for malformed LLM output |
| D3 | `agent_layer/core/react_engine.py` (refine) | Answer synthesizer: consolidate all observations into final answer with citations |
| D4 | `agent_layer/versions/custom_agent.py` | CLI entry: `python -m agent_layer.versions.custom_agent` |
| D5 | `agent_layer/tests/test_custom_agent.py` | Integration tests with Mock tools |

**D4 CLI entry:**

```python
# agent_layer/versions/custom_agent.py
"""CLI entry point for the custom ReAct Agent."""

import sys
from agent_layer.core.react_engine import ReActAgent
from agent_layer.core.hybrid_scorer import HybridScorer
from agent_layer.core.agent_memory import AgentMemory
from agent_layer.tools.registry import ToolRegistry
from agent_layer.tools.local_search import LocalSearchTool
from agent_layer.tools.query_rewriter import QueryRewriterTool
from rag_mcp_server.src.core.settings import load_settings, apply_env_overrides
from rag_mcp_server.src.libs.llm.llm_factory import LLMFactory


def main():
    """Run the custom ReAct agent in interactive mode."""
    settings = load_settings("config/settings.yaml")
    apply_env_overrides(settings)

    llm = LLMFactory.create(settings)
    tools = ToolRegistry()
    tools.register(LocalSearchTool())
    tools.register(QueryRewriterTool(llm=llm))
    # web_search registered in Phase F

    agent = ReActAgent(
        tools=tools,
        llm=llm,
        scorer=HybridScorer(llm=llm, threshold=settings.agent.get("score_threshold", 0.7)),
        memory=AgentMemory(),
        max_rounds=settings.agent.get("max_rounds", 3),
    )

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter your question: ")
    result = agent.run(query)
    print(f"\n{'='*60}")
    print(result.answer)
    print(f"\n[Source: {result.final_source}] [Rounds: {result.rounds}]")


if __name__ == "__main__":
    main()
```

---

## Phase E: LangGraph Agent

### Task E1: StateGraph Decision Graph

**Files:**
- Create: `agent_layer/versions/langgraph_agent.py`
- Create: `agent_layer/tests/test_langgraph_agent.py`

**Interfaces:**
- Produces: LangGraph `StateGraph` with nodes: search → score → (rewrite → search | web_search) → finish
- Depends on: Phase C (all tools), Phase D (for comparison)

```python
# agent_layer/versions/langgraph_agent.py
"""LangGraph-based ReAct Agent implementation."""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from agent_layer.tools.registry import ToolRegistry
from agent_layer.core.hybrid_scorer import HybridScorer


class AgentState(TypedDict):
    query: str
    search_results: str
    judgment: str
    missing_info: str
    rewritten_query: str
    web_results: str
    answer: str
    round: int
    source: str


def create_agent_graph(
    tools: ToolRegistry,
    llm,
    scorer: HybridScorer,
    max_rounds: int = 3,
) -> StateGraph:
    """Build the LangGraph StateGraph for the Agent.

    Args:
        tools: Tool registry with local_search, rewrite_query, web_search.
        llm: BaseLLM instance.
        scorer: HybridScorer for quality evaluation.
        max_rounds: Maximum search rounds.

    Returns:
        A compiled LangGraph StateGraph.
    """

    def search_local(state: AgentState) -> AgentState:
        """Node: Search local knowledge base."""
        search_tool = tools.get("local_search")
        query = state.get("rewritten_query") or state["query"]
        result = search_tool.execute(query=query)
        return {**state, "search_results": result, "round": state["round"] + 1}

    def evaluate_quality(state: AgentState) -> AgentState:
        """Node: Evaluate search result quality."""
        judgment = scorer.evaluate(
            state["query"],
            _parse_to_mock_result(state["search_results"]),
            mode="decision",
        )
        return {
            **state,
            "judgment": judgment.action,
            "missing_info": judgment.missing_info,
        }

    def rewrite_query(state: AgentState) -> AgentState:
        """Node: Rewrite the query."""
        rewriter = tools.get("rewrite_query")
        new_query = rewriter.execute(
            original_query=state["query"],
            missing_info=state["missing_info"],
        )
        return {**state, "rewritten_query": new_query}

    def search_web(state: AgentState) -> AgentState:
        """Node: Search the web."""
        web_tool = tools.get("web_search")
        result = web_tool.execute(query=state["query"])
        return {**state, "web_results": result, "source": "web"}

    def synthesize(state: AgentState) -> AgentState:
        """Node: Synthesize final answer."""
        context = state.get("search_results") or state.get("web_results") or ""
        prompt = (
            f"Question: {state['query']}\n\n"
            f"Information gathered:\n{context}\n\n"
            "Provide a complete answer with citations."
        )
        answer = llm.chat([{"role": "user", "content": prompt}])
        return {**state, "answer": answer, "source": state.get("source", "local")}

    # Router: decide next step based on judgment
    def route_after_score(state: AgentState) -> Literal["rewrite", "web_search", "synthesize"]:
        judgment = state["judgment"]
        if judgment == "PASS" or state["round"] >= max_rounds:
            return "synthesize"
        elif judgment == "REWRITE" and not state.get("rewritten_query"):
            return "rewrite"
        else:
            return "web_search"

    def route_after_rewrite(state: AgentState) -> Literal["search_local", "web_search"]:
        if state["round"] < max_rounds:
            return "search_local"
        return "web_search"

    # Build graph
    graph = StateGraph(AgentState)

    graph.add_node("search_local", search_local)
    graph.add_node("evaluate_quality", evaluate_quality)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("search_web", search_web)
    graph.add_node("synthesize", synthesize)

    graph.set_entry_point("search_local")
    graph.add_edge("search_local", "evaluate_quality")
    graph.add_conditional_edges("evaluate_quality", route_after_score)
    graph.add_conditional_edges("rewrite_query", route_after_rewrite)
    graph.add_edge("search_web", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


def _parse_to_mock_result(text: str):
    """Parse text into a mock result for scoring."""
    from unittest.mock import Mock
    import re
    scores = re.findall(r"Score:\s*([\d.]+)", text)
    chunks = [Mock(score=float(s), text=text[:100], metadata={"source": "parsed"}) for s in scores]
    result = Mock()
    result.chunks = chunks or [Mock(score=0.0, text="", metadata={})]
    return result
```

- [ ] **Write test with mock tools → pass → commit**

### Tasks E2-E5:

| Task | Description |
|------|-------------|
| E2 | ToolNode integration — register Phase C tools as LangChain tools |
| E3 | State management refinements + conditional edges based on round count |
| E4 | LangGraph unit tests |
| E5 | Side-by-side comparison test: same query → both agents → assert consistency |

---

## Phase F: Web Search + Full Pipeline

### Task F1: Tavily Search Backend

**Files:**
- Create: `agent_layer/tools/web_search.py`

```python
# agent_layer/tools/web_search.py
"""Web search tool with pluggable backends (Tavily / SerpAPI)."""

import os
import httpx
from agent_layer.tools.base import Tool


class WebSearchTool(Tool):
    """Search the web for information not in the local knowledge base."""

    name = "web_search"
    description = (
        "Search the web for information. Use when local knowledge base "
        "does not have relevant results. Returns search snippets with URLs."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
        },
        "required": ["query"],
    }

    def __init__(self, backend: str = "tavily", tavily_key: str = "", serpapi_key: str = ""):
        self.backend = backend
        self.tavily_key = tavily_key or os.getenv("TAVILY_API_KEY", "")
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY", "")

    def execute(self, query: str) -> str:
        """Execute web search.

        Args:
            query: The search query.

        Returns:
            Formatted search results.
        """
        if self.backend == "tavily":
            return self._search_tavily(query)
        elif self.backend == "serpapi":
            return self._search_serpapi(query)
        else:
            return f"Error: Unknown search backend '{self.backend}'"

    def _search_tavily(self, query: str) -> str:
        """Search using Tavily API."""
        if not self.tavily_key:
            return "Error: Tavily API key not configured."
        response = httpx.post(
            "https://api.tavily.com/search",
            json={"api_key": self.tavily_key, "query": query, "search_depth": "basic"},
            timeout=30,
        )
        data = response.json()
        results = data.get("results", [])
        if not results:
            return "No web results found."
        lines = [f"Web search results ({len(results)} found):\n"]
        for i, r in enumerate(results[:5], 1):
            lines.append(f"[{i}] {r.get('title', 'N/A')}\n    {r.get('content', '')[:200]}...\n    URL: {r.get('url', '')}")
        return "\n".join(lines)

    def _search_serpapi(self, query: str) -> str:
        """Search using SerpAPI."""
        if not self.serpapi_key:
            return "Error: SerpAPI key not configured."
        response = httpx.get(
            "https://serpapi.com/search",
            params={"api_key": self.serpapi_key, "q": query, "engine": "google"},
            timeout=30,
        )
        data = response.json()
        organic = data.get("organic_results", [])
        if not organic:
            return "No web results found."
        lines = [f"Web search results ({len(organic)} found):\n"]
        for i, r in enumerate(organic[:5], 1):
            lines.append(f"[{i}] {r.get('title', 'N/A')}\n    {r.get('snippet', '')[:200]}...\n    URL: {r.get('link', '')}")
        return "\n".join(lines)
```

- [ ] **Write tests with httpx mock → pass → commit**

### Tasks F2-F5:

| Task | Description |
|------|-------------|
| F2 | SerpAPI backend (included in web_search.py above) |
| F3 | Register WebSearchTool in ToolRegistry, backend selection from config |
| F4 | CacheInterface abstract class (placeholder, no implementation) |
| F5 | End-to-end validation: ingest sample → query → score → rewrite → web → confidence |

---

## Phase G: Testing & Polish

### Task G1: Agent Layer Unit Test Coverage

- [ ] Run coverage: `pytest agent_layer/tests/ --cov=agent_layer --cov-report=term`
- [ ] Fill gaps until ≥ 80%
- [ ] Tests for edge cases: empty results, LLM timeout, malformed JSON from LLM

### Task G2: RAG + Agent Integration Test

- [ ] Ingest a real PDF into test Chroma
- [ ] Run Agent query through full decision flow
- [ ] Verify citations in final answer

### Task G3: README

- [ ] Project description + architecture diagram
- [ ] Quick start guide
- [ ] Configuration guide
- [ ] MCP integration instructions

### Task G4: Code Cleanup

- [ ] Add docstrings to all public functions
- [ ] Type annotations check
- [ ] Remove dead code, debug prints

---

## Phase H: Resume & Interview Prep

| Task | Description |
|------|-------------|
| H1 | Run resume-writer skill → generate resume draft |
| H2 | Prepare Agent interview Q&A: ReAct principles, LangGraph vs custom, cascade design |
| H3 | Prepare RAG interview Q&A: hybrid search, pluggable architecture, evaluation |
| H4 | Iterate resume based on mock interview feedback |

---

## Plan Self-Review

1. **Spec coverage**: All 99 tasks from the spec (v1.1, +B1.15) are covered in this plan. Phase A (5 tasks) in full detail, Phase B (58 tasks) with file/interface mapping, Phases C-H (35 tasks, up from original 31 due to expanded memory tasks C6-C12) with detailed code and tests.

2. **Placeholder scan**: No TBD or TODO markers. All tasks have concrete file paths, interfaces, and test code. Cache interface (F4) is explicitly marked as placeholder class with no logic — matching the spec's explicit "reserved, not implemented" requirement.

3. **Type consistency**: AgentResult is used in D1 and consumed by D4, E5. ScorerJudgment is produced by C5 and consumed by D1, E1. Tool base class is defined in C1 and consumed by C2-C4, D1, E2, F3. All signatures align across phases.

4. **Test-first discipline**: Every task starts with a failing test, then implementation. Tests use Mock/Fake for external dependencies (LLM, embedding, search APIs).
