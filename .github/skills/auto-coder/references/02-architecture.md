## 2. 整体架构

### 架构图

```
┌─────────────────────────────────────────────────────┐
│                  外部 MCP Client                      │
│         Copilot / Claude Desktop / Codex             │
└──────────────────────┬──────────────────────────────┘
                       │ MCP Protocol (stdio)
                       ▼
┌─────────────────────────────────────────────────────┐
│              你的 RAG MCP Server                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │  Ingestion Pipeline (PDF→Chunk→Embed→Upsert)    │ │
│  │  Hybrid Search (Dense + Sparse + RRF + Rerank)  │ │
│  │  Tools: query_knowledge_hub / list_collections  │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────┬──────────────────────┬───────────────┘
               │ 直接 import          │ MCP Protocol
               ▼                      ▼
┌──────────────────────────┐   未来扩展
│   你的 Agent 层 (大脑)    │
│                          │
│  ┌─────────────────────┐ │
│  │  ReAct 循环引擎      │ │
│  │  Think→Act→Observe  │ │
│  └─────────┬───────────┘ │
│            │              │
│  ┌─────────▼───────────┐ │
│  │  Tool Set            │ │
│  │  ├─ local_search ───┼─┼──→ RAG Pipeline
│  │  ├─ rewrite_query ──┼─┼──→ LLM
│  │  └─ web_search ─────┼─┼──→ Search API
│  └─────────────────────┘ │
│            │              │
│  ┌─────────▼───────────┐ │
│  │  混合判分器           │ │
│  │  快速阈值 + LLM深判  │ │
│  └─────────────────────┘ │
└──────────────────────────┘
```

### 两层职责

| 层 | 职责 | 来源 |
|----|------|------|
| **RAG MCP Server** | 知识摄取 + 混合检索，通过 MCP 暴露工具 | 复现 clean-start（AI 辅助），架构不修改 |
| **Agent 层** | ReAct 决策 + 质量判分 + 级联降级 | **原创区**，从零设计 |

### Agent 调用 RAG 的方式

**Phase 1（当前阶段）**：直接 import
- Agent 和 RAG 在同一个 Python 项目内
- Agent 直接 import RAG 的检索函数，不涉及 MCP 客户端代码
- MCP Server 依旧对外暴露，供 Copilot/Claude 等外部 Client 使用

**Phase 2（进阶）**：MCP Client 方式
- Agent 以子进程启动 RAG MCP Server，通过 stdio JSON-RPC 调用
- 松耦合，Agent 和 RAG 独立进程

### 数据流（一次查询）

```
用户提问 → Agent 接收
  │
  ▼
Think: 我需要本地检索
  │
  ▼
Act: 调用 local_search(query) → 走 RAG Pipeline
  │
  ▼
Observe: 混合判分器 (scorer_prompt.md, mode: decision)
  │
  ├─ [PASS] → 直接返回答案
  │
  ├─ [PARTIAL] → Think: 信息不够全，改写试试
  │   └─ Act: rewrite_query(query + 缺失信息) → local_search(新 query)
  │       └─ Observe: 二次判分
  │           └─ 还是不行
  │               └─ Think: 本地没有，联网搜
  │                   └─ Act: web_search(query)
  │                       └─ Observe: 轻量信心评估 (scorer_prompt.md, mode: confidence)
  │                           └─ 标注信心等级，返回结果
  │
  └─ [NO] → 直接 Act: web_search(query)
      └─ Observe: 轻量信心评估 → 返回结果
```

### 联网搜索后端选择

支持 Tavily 和 SerpAPI，通过 `settings.yaml` 配置切换：

```yaml
agent:
  web_search:
    backend: tavily  # tavily | serpapi
    tavily_api_key: ${TAVILY_API_KEY}
    serpapi_api_key: ${SERPAPI_API_KEY}
  max_rounds: 3
  cache:
    enabled: false  # 预留接口，待实现
```

---
