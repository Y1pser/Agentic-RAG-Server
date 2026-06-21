---
name: auto-coder
description: Autonomous spec-driven development agent for Agentic RAG Server. Syncs the design spec into chapter-based reference files, identifies the next pending task from the progress tracking table, implements code following spec architecture and patterns, runs tests with up to 3 auto-fix rounds, and persists progress with atomic commits. Use when user says "auto code", "自动开发", "自动写代码", "auto dev", "一键开发", or wants fully automated spec-to-code workflow.
---

# Auto Coder

One trigger completes **read spec → find task → code → test → persist progress**.

Optional modifiers: append a task ID (e.g. `auto code B2.3`) to target a specific task, or `--no-commit` to skip git commit.

This project has two layers: **Phase B** replicates the original RAG core from `../MODULAR-RAG-MCP-SERVER/` (clean-start, write from scratch referencing original code), while **Phase C-F** are original Agent-layer work (design from spec, no reference code).

---

## Pipeline

```
Sync Spec → Find Task → Implement → Test (≤3 fix rounds) → Persist
```

Pause only at the end for commit confirmation. Run everything else autonomously.

> **⚠️ CRITICAL: Activate `.venv` before ANY `python`/`pytest` command (idempotent, re-run if unsure).**
> - **Windows**: `.\.venv\Scripts\Activate.ps1`
> - **macOS/Linux**: `source .venv/bin/activate`

## Reference Map

All files under `.github/skills/auto-coder/references/`:

| File | Content | When to Read |
|------|---------|-------------|
| `01-overview.md` | Project positioning, goals, core differentiators vs original | First task or when needing project-level context |
| `02-architecture.md` | Two-layer architecture (Agent + RAG), data flow, component responsibilities | When creating/modifying core modules |
| `03-agent-design.md` | ReAct loop, hybrid scorer (two-stage), cascade fallback, query rewrite | Phase C-F tasks (everything Agent-layer) |
| `04-two-versions.md` | LangGraph vs custom ReAct strategy, core class structures | Phase D-E tasks (both Agent versions) |
| `05-directory.md` | Full directory tree, boundary rules (allowed/forbidden per directory) | Every task — file placement reference |
| `06-schedule.md` | Task schedule, progress tracking table, overall progress, tech stack appendix | Every cycle (Sync Spec step) |

---

### 1. Sync Spec

```bash
python .github/skills/auto-coder/scripts/sync_spec.py
```

Then read the schedule file to get task statuses:
- Read `.github/skills/auto-coder/references/06-schedule.md`

Task markers:

| Marker | Status |
|--------|--------|
| `[ ]` / `⬜` | Not started |
| `[~]` / `🔶` / `(进行中)` | In progress |
| `[x]` / `✅` / `(已完成)` | Completed |

---

### 2. Find Task

Pick the first `IN_PROGRESS` task, then the first `NOT_STARTED`. If user specified a task ID, use that directly.

Quick-check predecessor artifacts exist (file-level only). On mismatch, log a warning and continue — only stop if the target task itself is blocked.

---

### 3. Implement

1. **Read relevant spec** from `.github/skills/auto-coder/references/`:
   - Architecture & data flow: `02-architecture.md`
   - Agent design details: `03-agent-design.md` (Phase C-F tasks)
   - Two-version strategy: `04-two-versions.md` (Phase D-E tasks)
   - Directory & boundary rules: `05-directory.md`

2. **Determine task layer and directory:**

| Task Prefix | Layer | Root Directory | Nature |
|------------|-------|---------------|--------|
| A* | Project root | `Agentic-RAG-Server/` | Environment init, config, skeleton |
| B* | RAG Layer | `rag_mcp_server/` | Replication — write from scratch, reference original |
| C* | Agent Layer | `agent_layer/` | Agent infrastructure (original work) |
| D* | Agent Layer | `agent_layer/` | Custom ReAct engine (original work) |
| E* | Agent Layer | `agent_layer/` | LangGraph version (original work) |
| F* | Agent Layer | `agent_layer/` | Web search + full pipeline integration |
| G* | Project-wide | Both layers | Testing, coverage, documentation |
| H* | Skills | No code | Resume & interview prep (use other skills) |

3. **Extract** from the task row: task name, notes column, which files to create/modify.

4. **Plan** files to create/modify before writing any code.

5. **Code** — project-specific rules:
   - Treat spec as single source of truth
   - Phase B (RAG replication): reference original code in `../MODULAR-RAG-MCP-SERVER/` for structure and logic, but write from clean-start — do NOT copy-paste. Follow the pluggable pattern: `Base*` abstract → `*Factory` registry → default `Concrete*` implementation.
   - Phase C-F (Agent original): design from spec, no reference code. Follow the patterns defined in `03-agent-design.md` and `04-two-versions.md`.
   - Use `config/settings.yaml` values, never hardcode API keys or endpoints
   - Match existing codebase patterns and style (type hints, docstrings, naming)

6. **Write tests** alongside code:
   - RAG layer tests: `rag_mcp_server/tests/unit/` or `tests/unit/`
   - Agent layer tests: `agent_layer/tests/`
   - Mock external deps (LLM, Embedding, VectorStore, Vision) in unit tests
   - Only integration tests touch real backends

7. **Self-review** before running tests: verify all planned files exist, imports resolve, and boundary rules are respected.

---

### 4. Test & Auto-Fix

```

Round 0..2:
  Run pytest on relevant test file
  If pass → go to step 5
  If fail → analyze error, apply fix, re-run

Round 3 still failing → STOP, show failure report to user
```

---

### 5. Persist

1. **Update the spec file** at `docs/superpowers/specs/2026-06-21-agentic-rag-server-design.md`:
   - Change task marker `[ ]` → `[x]` (or `[~]` → `[x]`)
   - Add completion date (today, format: YYYY-MM-DD) in the date column
   - Add a brief note if relevant in the notes column
   - Update the **overall progress table** (section `### 📈 总体进度`): increment completed count + update percentage

2. **Re-sync**: `python .github/skills/auto-coder/scripts/sync_spec.py --force`

3. **Show summary & ask**:

```
✅ [B1.1] LLM 抽象接口与工厂 — done
   Files: rag_mcp_server/src/libs/base_llm.py, rag_mcp_server/src/libs/llm_factory.py
   Tests: 6/6 passed
   Commit: feat(libs): [B1.1] implement BaseLLM + LLMFactory

   "commit" → git add + commit
   "skip"   → end
   "next"   → commit + start next task
```

On "next", loop back to step 1 and start the next task.

---

## Project Context

- **Project:** Agentic RAG Server — two-layer architecture (Agent brain + RAG tool)
- **Spec:** `docs/superpowers/specs/2026-06-21-agentic-rag-server-design.md`
- **Plan:** `docs/superpowers/plans/2026-06-21-agentic-rag-server-plan.md`
- **Original reference:** `../MODULAR-RAG-MCP-SERVER/` (read-only, Phase B only)
- **Total tasks:** 94 across 8 phases (A-H)
- **Key difference from original:** Agent layer adds autonomous decision-making (ReAct + hybrid scoring + cascade fallback) that the original RAG server doesn't have
