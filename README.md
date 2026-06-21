# Agentic RAG Server

A modular RAG system with **Agent-driven autonomous decision-making**.

When local retrieval falls short, the Agent automatically triggers Query Rewriting or Web Search as cascade fallback — exposing tools via the MCP protocol.

## Architecture

```
External MCP Client (Copilot / Claude Desktop)
        │
        ▼
┌──────────────────────────┐
│    RAG MCP Server         │  ← Replicated from clean-start
│    Ingestion + Hybrid     │
│    Search + MCP Tools     │
└──────────┬───────────────┘
           │ direct import
           ▼
┌──────────────────────────┐
│    Agent Layer            │  ← Original work
│    ReAct Engine + Hybrid  │
│    Scorer + Cascade       │
└──────────────────────────┘
```

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Run
python main.py
```

## Project Status

🚧 **Phase A** — Environment initialization in progress.

See `docs/superpowers/specs/` for the full design document and 94-task plan.
