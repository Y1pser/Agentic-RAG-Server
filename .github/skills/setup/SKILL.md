---
name: setup
description: Interactive project setup wizard for Agentic RAG Server. Guides through LLM/Embedding provider selection, API key configuration, web search backend setup, dependency installation, config generation, and optionally scaffold provider code for unsupported backends.
---

# Setup: Interactive Project Configuration Wizard

## Overview

You are an interactive setup wizard for the **Agentic RAG Server** project. From a clean codebase, guide the user through full environment configuration so they can launch the MCP Server and Agent with one command.

## Pipeline

### Phase 1: Provider Selection

Ask the user which providers they want to use. Ask one question at a time:

**Q1: LLM Provider**: "Which LLM provider do you want to use?"
- OpenAI (openai) — recommended for general use
- Azure OpenAI (azure) — enterprise compliance
- DeepSeek (deepseek) — cost optimization
- Ollama (ollama) — fully local, no API cost
- Qwen / DashScope (qwen) — Chinese language optimized
- Gemini (gemini) — Google ecosystem

**Q2: Embedding Provider**:
- OpenAI (openai) — recommended
- Azure OpenAI (azure)
- Ollama (ollama) — local embeddings

**Q3: Vision LLM Provider** (for image captioning, optional):
- Same as LLM provider
- Azure OpenAI Vision (azure_vision)
- None (skip image processing)

**Q4: Web Search Backend** (for Agent cascade fallback):
- Tavily (tavily) — recommended
- SerpAPI (serpapi)
- Both

### Phase 2: API Key Configuration

For each selected provider, ask for the API key. Create/update the `.env` file:

```bash
# LLM
OPENAI_API_KEY=sk-...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://...
DEEPSEEK_API_KEY=...

# Embedding
EMBEDDING_API_KEY=...

# Vision
AZURE_VISION_API_KEY=...
AZURE_VISION_ENDPOINT=https://...

# Web Search
TAVILY_API_KEY=tvly-...
SERPAPI_API_KEY=...
```

> **Security rule:** NEVER output API keys to stdout. Only write to `.env`. Confirm with: "API key for <provider> saved to .env ✓"

### Phase 3: Dependency Installation

```bash
pip install -e ".[dev]"
```

For LangGraph support (Phase E):
```bash
pip install -e ".[dev,langgraph]"
```

If the user chose Ollama, remind them: "Make sure Ollama is running locally (`ollama serve`) and you have pulled the model you want to use."

### Phase 4: Configuration File Generation

Generate `config/settings.yaml` based on the user's choices:

```yaml
llm:
  provider: <chosen>
  model: <default model>
  api_key: ${OPENAI_API_KEY}

embedding:
  provider: <chosen>
  model: <default model>
  api_key: ${EMBEDDING_API_KEY}

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
    backend: <chosen>
    tavily_api_key: ${TAVILY_API_KEY}
    serpapi_api_key: ${SERPAPI_API_KEY}
  cache:
    enabled: false
```

### Phase 5: Provider Code Scaffolding (if needed)

If the user chose a provider that is not yet implemented (e.g., Qwen, Gemini), scaffold the provider code following the project's pluggable architecture:

1. Create `rag_mcp_server/src/libs/llm/<provider>_llm.py` implementing `BaseLLM`
2. Register in `LLMFactory` with `LLMFactory.register("<provider>", lambda s: <Provider>LLM(s))`
3. Follow the exact pattern of existing providers (`OpenAILLM`, `DeepSeekLLM`)

### Phase 6: Verification

After configuration, verify the setup:

```bash
# 1. Verify settings load
python -c "from rag_mcp_server.src.core.settings import load_settings; s = load_settings('config/settings.yaml'); print('Settings OK')"

# 2. Verify .env
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('Env OK' if os.getenv('OPENAI_API_KEY') or os.getenv('AZURE_OPENAI_API_KEY') else 'WARNING: No LLM key found')"

# 3. Verify imports
python -c "import rag_mcp_server; import agent_layer; print('Imports OK')"
```

If any check fails, diagnose and offer to fix (up to 3 retry rounds).

### Phase 7: Launch Dashboard (optional)

Ask: "Setup complete! Would you like to start the Streamlit Dashboard?"

```bash
streamlit run rag_mcp_server/src/observability/dashboard/app.py
```
