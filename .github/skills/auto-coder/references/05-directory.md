## 5. 目录结构

```
D:\Codex_workspace\559-3\
├── MODULAR-RAG-MCP-SERVER\        ← 原始项目，只读不碰
│
└── Agentic-RAG-Server\            ← 你的新项目
    │
    ├── rag_mcp_server\            ← 复现部分（从 clean-start 用 AI 写）
    │   ├── src/
    │   │   ├── core/              # 类型定义
    │   │   ├── ingestion/         # 数据摄取管线
    │   │   ├── retrieval/         # 混合检索 + 重排
    │   │   ├── mcp_server/        # MCP 工具暴露
    │   │   ├── libs/              # 可插拔组件
    │   │   └── observability/     # 追踪 + Dashboard
    │   ├── config/
    │   ├── tests/
    │   └── data/
    │
    ├── agent_layer\               ← 原创区
    │   ├── core/
    │   │   ├── react_engine.py    # 自研 ReAct 循环
    │   │   ├── hybrid_scorer.py   # 混合判分器
    │   │   └── agent_memory.py    # Agent 记忆管理
    │   ├── tools/
    │   │   ├── base.py            # Tool 抽象基类
    │   │   ├── local_search.py    # 本地 RAG 检索工具
    │   │   ├── query_rewriter.py  # Query 改写工具
    │   │   ├── web_search.py      # 联网搜索（Tavily / SerpAPI）
    │   │   └── registry.py        # Tool 注册中心
    │   ├── versions/
    │   │   ├── langgraph_agent.py # LangGraph 版 Agent
    │   │   └── custom_agent.py    # 自研 ReAct 版 Agent
    │   ├── prompts/
    │   │   ├── react_system.md    # ReAct System Prompt
    │   │   ├── scorer_prompt.md   # 判分 Prompt（decision + confidence）
    │   │   └── rewriter_prompt.md # 改写 Prompt
    │   └── tests/
    │
    ├── docs/
    │   └── superpowers/
    │       └── specs/             # 设计文档
    │
    ├── config/
    │   └── settings.yaml          # 统一配置
    │
    ├── .env                       # API Keys（gitignore）
    ├── .env.example               # 模板
    ├── .gitignore
    ├── README.md
    └── requirements.txt
```

### 边界规则

| 目录 | 允许 | 禁止 |
|------|------|------|
| `MODULAR-RAG-MCP-SERVER/` | 参考代码、读文档 | 任何修改 |
| `rag_mcp_server/` | 复现 clean-start 代码 | 架构改动需对齐 DEV_SPEC |
| `agent_layer/` | 原创，自由发挥 | 无 |

---
