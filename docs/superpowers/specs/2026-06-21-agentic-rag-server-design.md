# Agentic RAG Server — 设计文档

> 版本：1.0 | 日期：2026-06-21

---

## 目录

- [1. 项目定位与目标](#1-项目定位与目标)
- [2. 整体架构](#2-整体架构)
- [3. Agent ReAct 循环设计](#3-agent-react-循环设计)
- [4. 两版 Agent 策略](#4-两版-agent-策略)
- [5. 目录结构](#5-目录结构)
- [6. 项目排期与进度跟踪](#6-项目排期与进度跟踪)

---

## 1. 项目定位与目标

### 项目名

`Agentic-RAG-Server`

### 一句话描述

一个具有 Agent 自主决策能力的模块化 RAG 系统。本地检索不达预期时，Agent 自动触发 Query 改写或联网搜索进行级联兜底，通过 MCP 协议对外暴露工具接口。

### 使用者背景

- 校招，RAG + Agent 双方向
- 西电本科 CS + 港城大硕士 CS，Python 熟练（LeetCode ~200题），RAG/LLM 理论知识扎实，缺实战经验
- 时间：1-2 个月

### 核心差异点（与原项目 Modular RAG MCP Server 的区分）

| 维度 | 原项目 | 本项目 |
|------|--------|--------|
| 检索模式 | 被动响应：收 Query → 搜 → 返回 | **主动决策**：收 Query → 搜 → 自评 → 改写/联网/返回 |
| 架构层次 | 单一 Server 层 | **双层架构**：Agent 层（大脑）+ RAG 层（工具） |
| 检索质量 | 依赖检索策略本身 | **混合判分**：快速阈值 + LLM 深判 |
| 知识覆盖 | 仅本地知识库 | 本地优先 + 联网兜底（缓存接口预留） |
| Agent 实现 | 无 | **双版本**：LangGraph 版 + 自研 ReAct 版 |

---

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

## 3. Agent ReAct 循环设计

### 决策流程图

```
              ┌──────────────┐
              │  接收 Query   │
              └──────┬───────┘
                     ▼
              ┌──────────────┐
         ┌───→│ local_search │
         │    └──────┬───────┘
         │           ▼
         │    ┌──────────────┐
         │    │  混合判分器   │
         │    └──┬───┬───┬──┘
         │       │   │   │
         │     PASS PARTIAL NO
         │       │   │   │
         │       ▼   │   ▼
         │  ┌──────┐ │  直接联网
         │  │返回  │ │
         │  └──────┘ │
         │           ▼
         │    ┌──────────────┐
         │    │ 阈值自适应    │  ← 边界→下调阈值，下次更快
         │    └──────┬───────┘
         │           ▼
         │    ┌──────────────┐
         │    │ LLM 深判     │
         │    └──┬───┬───────┘
         │       │   │
         │    PARTIAL NO
         │       │   │
         │       ▼   ▼
         │   改写查询  联网搜索
         │       │       │
         │       ▼       │
         │  再搜本地     │
         │       │       │
         │   还不满意   │
         │       │       │
         └───不满足──────┘
                 │
                 ▼
            ┌──────────┐
            │ 返回结果  │
            │ + 缓存入口│
            └──────────┘
```

### 关键设计细节

#### 混合判分器 (HybridScorer)

两段式判分：

1. **第一关：快速阈值**
   - Top-1 分数 ≥ threshold → PASS，直接返回
   - Top-1 分数 ≥ threshold × 0.6 → 边界，走 LLM 深判
   - Top-1 分数 < threshold × 0.6 → NO，直接联网

2. **第二关：LLM 深判** (scorer_prompt.md, mode: decision)
   - 判断：检索结果能否充分回答？
   - ENOUGH → PASS
   - PARTIAL → REWRITE（会告知缺失什么信息）
   - NO → WEB_SEARCH

3. **阈值自适应**
   - 边界判为 PASS 时，自动下调阈值（下次同类型直接过）
   - 避免频繁触发 LLM 深判，降低延迟和成本

#### LLM 深判 Prompt 双模式

**mode: decision**（路由判断，本地搜索后使用）
```
你是一个检索质量评估器。
用户问题：{query}
检索到的结果：{results}
请判断：
1. 这些结果能否充分回答用户问题？（ENOUGH / PARTIAL / NO）
2. 如果 PARTIAL，缺少什么关键信息？
3. 建议：直接回答 / 改写查询重搜 / 需要联网搜索
```

**mode: confidence**（信心评估，联网搜索后使用）
```
你是一个答案质量评估器。
用户问题：{query}
在线搜索结果：{web_results}
请判断：
1. 搜索结果能否支撑一个可信回答？（YES / PARTIALLY / NO）
2. 信心分（0-10）
3. 如果 PARTIALLY 或 NO，标注不确定的部分
仅用于评估，不需要建议下一步行动。
```

#### Query 改写策略

- LLM 驱动改写
- 改写时注入深判发现的「信息缺口」作为 Prompt 上下文
- 改写方向取决于缺口类型：缺关键词 → 扩展同义词；缺细节 → 具体化查询

#### 最大重试

- 最多 3 轮：原查询 → 改写重搜 → 联网搜索
- 第 3 轮后无论结果如何都返回

#### 缓存接口（预留，不实现）

```python
class CacheInterface:
    def should_cache(self, result: AgentResult) -> bool: ...
    def cache(self, query: str, result: AgentResult) -> None: ...
    def lookup(self, query: str) -> AgentResult | None: ...
```

---

## 4. 两版 Agent 策略

| | LangGraph 版 | 自研 ReAct 版 |
|---|---|---|
| **目标** | 快速出活，工程化 | 展示底层理解 |
| **实现方式** | LangGraph StateGraph + ToolNode | 纯 Python 循环 + Prompt 模板 |
| **Tool 抽象** | LangChain `@tool` 装饰器 | 自写 `Tool` 基类 |
| **状态管理** | LangGraph `State` dict | 自写 `AgentMemory` 类 |
| **Tool 集** | 同一套（local_search, rewrite_query, web_search） | 同一套 |

### 自研版核心结构

```python
class ReActAgent:
    def __init__(self, tools: list[Tool], memory: AgentMemory, scorer: HybridScorer):
        self.tools = {t.name: t for t in tools}
        self.memory = memory
        self.scorer = scorer
        self.max_rounds = 3

    def run(self, query: str) -> AgentResult:
        self.memory.add("user", query)
        for round in range(self.max_rounds):
            thought = self.think()               # LLM 决策下一步
            action, args = self.parse(thought)   # 解析动作
            if action == "FINISH":
                break
            observation = self.tools[action].execute(**args)
            quality = self.scorer.evaluate(query, observation)
            self.memory.add("assistant",
                f"Thought: {thought}\nObservation: {observation}")
        return self.synthesize()                 # LLM 汇总最终答案
```

---

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

## 6. 项目排期与进度跟踪

> **排期原则**
> 
> - **严格对齐本 Spec 架构分层与目录结构**：以第 5 节目录树为"交付清单"，每一步都要在文件系统上产生可见变化。
> - **先打通主闭环，再补齐默认实现**：优先做"可跑通的端到端路径"，再完善各模块的默认后端实现。
> - **外部依赖可替换/可 Mock**：LLM/Embedding/Vision/VectorStore 的真实调用在单元测试中一律用 Fake/Mock，集成测试再开真实后端（可选）。
> - **RAG 层与原 DEV_SPEC 对齐**：Phase B 的子任务结构参照原项目 DEV_SPEC 的 9 阶段划分，确保复现完整性。
> - **Agent 层原创**：Phase C-F 为原创工作，从零设计并实现。

### 阶段总览

| 阶段 | 目标 |
|------|------|
| **Phase A** | 环境初始化：工程骨架 + 测试基座 + 配置系统 |
| **Phase B** | RAG 核心复现：完整复现原项目全部功能 |
| **Phase C** | Agent 基础设施：Tool 集 + 判分器 + Prompt 模板 |
| **Phase D** | 自研 ReAct 版：Think→Act→Observe 循环引擎 |
| **Phase E** | LangGraph 版：StateGraph 实现，与自研版对照 |
| **Phase F** | 联网搜索 + 全链路：Tavily/SerpAPI + 级联兜底 |
| **Phase G** | 测试 + 打磨：Agent 层测试 + 集成测试 + 文档 |
| **Phase H** | 简历 + 面试准备：resume-writer + 面试追问 |

---

### 📊 进度跟踪表

> **状态说明**：`[ ]` 未开始 | `[~]` 进行中 | `[x]` 已完成
> 
> **更新时间**：每完成一个子任务后更新对应状态

#### Phase A：环境初始化

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| A1 | 初始化目录树与最小可运行入口 | [x] | 2026-06-21 | 创建完整目录骨架(rag_mcp_server + agent_layer 双层)、main.py、pyproject.toml、.gitignore、.env.example、requirements.txt、README.md、所有 __init__.py、23 个冒烟测试通过 |
| A2 | 引入 pytest 并建立测试目录约定 | [x] | 2026-06-21 | pytest 配置(pyproject.toml)、conftest.py 共享 fixtures、sample_documents/hello.txt 测试用文档、29 个冒烟测试全部通过 |
| A3 | 配置加载与校验（Settings） | [x] | 2026-06-21 | `Settings` dataclass(9 配置域)、`load_settings()` YAML解析、`validate_settings()` 必填字段校验(llm/embedding/vector_store)、6 个单元测试通过 |
| A4 | .env 密钥管理与 .env.example | [x] | 2026-06-21 | `apply_env_overrides()` 支持 OPENAI/AZURE/TAVILY/SERPAPI/EMBEDDING 密钥注入、`.env.example` 模板(已在A1创建)、python-dotenv 可选集成、3 个测试通过 |
| A5 | git init + GitHub 仓库关联 | [x] | 2026-06-21 | git init + 5 次提交 + GitHub 远程已建立，已 push 到 https://github.com/Y1pser/Agentic-RAG-Server |

#### Phase B：RAG 核心复现

##### B-1：Libs 可插拔层（Factory + Base 接口 + 默认实现）

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| B1.1 | LLM 抽象接口与工厂 | [x] | 2026-06-22 | `BaseLLM` + `LLMFactory`，支持按配置选择 provider |
| B1.2 | Embedding 抽象接口与工厂 | [x] | 2026-06-22 | `BaseEmbedding` + `EmbeddingFactory`，支持批量 embed |
| B1.3 | Splitter 抽象接口与工厂 | [x] | 2026-06-22 | `BaseSplitter` + `SplitterFactory` |
| B1.4 | VectorStore 抽象接口与工厂 | [x] | 2026-06-22 | `BaseVectorStore` + `VectorStoreFactory` |
| B1.5 | Reranker 抽象接口与工厂（含 None 回退） | [ ] | — | `BaseReranker` + `RerankerFactory` + `NoneReranker` |
| B1.6 | Evaluator 抽象接口与工厂 | [ ] | — | `BaseEvaluator` + `EvaluatorFactory` |
| B1.7 | OpenAI-Compatible LLM 实现 | [ ] | — | `OpenAILLM` + `AzureLLM` + `DeepSeekLLM` |
| B1.8 | Ollama LLM 实现 | [ ] | — | `OllamaLLM` |
| B1.9 | OpenAI & Azure Embedding 实现 | [ ] | — | `OpenAIEmbedding` + `AzureEmbedding` |
| B1.10 | Recursive Splitter 默认实现 | [ ] | — | `RecursiveSplitter`（LangChain 集成）|
| B1.11 | ChromaStore 默认实现 | [ ] | — | `ChromaStore` + roundtrip 验证 |
| B1.12 | LLM Reranker 实现 | [ ] | — | `LLMReranker` + Prompt 模板支持 |
| B1.13 | Vision LLM 抽象接口与工厂集成 | [ ] | — | `BaseVisionLLM` + `LLMFactory` 扩展 |
| B1.14 | Azure Vision LLM 实现 | [ ] | — | `AzureVisionLLM` + 图片压缩 |

##### B-2：Ingestion Pipeline（PDF → Chunk → Embedding → Upsert）

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| B2.1 | 定义核心数据类型/契约 | [ ] | — | `Document` / `Chunk` / `ChunkRecord` |
| B2.2 | 文件完整性检查（SHA256） | [ ] | — | `FileIntegrityChecker` + SQLite 存储 |
| B2.3 | Loader 抽象基类与 PDF Loader | [ ] | — | `BaseLoader` + `PdfLoader`（MarkItDown）+ 图片提取 |
| B2.4 | Splitter 集成（调用 Libs） | [ ] | — | `DocumentChunker` 包装 Libs Splitter |
| B2.5 | ChunkRefiner（LLM 去噪 + 重组） | [ ] | — | `ChunkRefiner`（Rule + LLM 两种模式）|
| B2.6 | MetadataEnricher（标题/摘要/标签） | [ ] | — | `MetadataEnricher`（Rule + LLM 两种模式）|
| B2.7 | ImageCaptioner（Vision LLM 图片描述） | [ ] | — | `ImageCaptioner` + Vision LLM 集成 |
| B2.8 | DenseEncoder（批量向量化） | [ ] | — | 批量编码 + 差量计算（Content Hash 去重）|
| B2.9 | SparseEncoder（BM25 稀疏编码） | [ ] | — | 词频统计 + 语料库 IDF 计算 |
| B2.10 | BM25Indexer（倒排索引 + IDF 持久化） | [ ] | — | 倒排索引构建 + pickle 持久化 |
| B2.11 | VectorUpserter（幂等 upsert） | [ ] | — | 稳定 `chunk_id` 生成 + 幂等写入 |
| B2.12 | ImageStorage（图片存储 + SQLite 索引） | [ ] | — | 本地文件存储 + `image_id` 索引 |
| B2.13 | Pipeline 编排（MVP 串起来） | [ ] | — | 完整 Load→Split→Transform→Embed→Upsert 流程 |
| B2.14 | 脚本入口 `ingest.py` | [ ] | — | CLI 摄取脚本 + 文件发现 + skip 已处理 |

##### B-3：Retrieval（Dense + Sparse + RRF + Rerank）

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| B3.1 | QueryProcessor（关键词提取 + filters） | [ ] | — | `ProcessedQuery` 类型 + 停用词过滤 + filter 语法 |
| B3.2 | DenseRetriever（调用 VectorStore.query） | [ ] | — | `RetrievalResult` 类型 + 依赖注入 |
| B3.3 | SparseRetriever（BM25 查询） | [ ] | — | BM25 倒排索引查询 |
| B3.4 | RRF Fusion | [ ] | — | `RRFFusion` 类 + k 参数可配置 + 加权融合 |
| B3.5 | HybridSearch 编排 | [ ] | — | 并行检索 + 优雅降级 + 元数据过滤 |
| B3.6 | Reranker（Core 层编排 + Fallback） | [ ] | — | `CoreReranker` + LLM Reranker 集成 + Fallback 机制 |
| B3.7 | 脚本入口 `query.py` | [ ] | — | CLI 查询入口 + verbose 输出 |

##### B-4：MCP Server 层与 Tools

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| B4.1 | MCP Server 入口与 Stdio 约束 | [ ] | — | `server.py` 使用官方 MCP SDK + stdio transport |
| B4.2 | `query_knowledge_hub` Tool | [ ] | — | 主检索工具 + `ResponseBuilder` + `CitationGenerator` |
| B4.3 | `list_collections` Tool | [ ] | — | 集合列表 + `CollectionInfo` |
| B4.4 | `get_document_summary` Tool | [ ] | — | 文档摘要获取 + 错误处理 |
| B4.5 | 多模态返回组装（Text + Image） | [ ] | — | `MultimodalAssembler` + base64 编码 + MIME 检测 |

##### B-5：Trace 基础设施与打点

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| B5.1 | TraceContext 增强 | [ ] | — | `finish()` + 耗时统计 + `trace_type` + `to_dict()` |
| B5.2 | 结构化日志 logger（JSON Lines） | [ ] | — | `JSONFormatter` + `get_trace_logger` + `write_trace` |
| B5.3 | 在 Query 链路打点 | [ ] | — | HybridSearch + CoreReranker trace 注入（5 阶段）|
| B5.4 | 在 Ingestion 链路打点 | [ ] | — | Pipeline 五阶段 trace 注入 |
| B5.5 | Pipeline 进度回调 (on_progress) | [ ] | — | `on_progress` 回调（6 阶段通知）|

##### B-6：可视化管理平台 Dashboard

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| B6.1 | Dashboard 基础架构与系统总览页 | [ ] | — | `app.py` 多页面导航 + `overview.py` + `ConfigService` |
| B6.2 | DocumentManager 实现 | [ ] | — | 跨存储（ChromaStore + BM25 + ImageStorage + IntegrityChecker）协调操作 |
| B6.3 | 数据浏览器页面 | [ ] | — | `DataService` + 文档列表 + Chunk 内容展示 |
| B6.4 | Ingestion 管理页面 | [ ] | — | 文件上传 + `IngestionPipeline` 集成 + 实时进度条 |
| B6.5 | Ingestion 追踪页面 | [ ] | — | `TraceService` 读取 traces.jsonl + 阶段时间线 |
| B6.6 | Query 追踪页面 | [ ] | — | Query trace 过滤 + 检索结果展示 + rerank 对比 |

##### B-7：评估体系

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| B7.1 | RagasEvaluator 实现 | [ ] | — | Ragas 框架集成 + Faithfulness / Relevancy / Recall |
| B7.2 | CompositeEvaluator 实现 | [ ] | — | 多评估器并行执行 + 结果汇总 |
| B7.3 | EvalRunner + Golden Test Set | [ ] | — | 评估运行器 + 标准测试集 JSON 格式 |
| B7.4 | 评估面板页面 | [ ] | — | Dashboard 评估页面 + 历史趋势对比 |

##### B-8：端到端验收与文档收口

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| B8.1 | E2E：MCP Client 侧调用模拟 | [ ] | — | JSON-RPC 请求模拟 + 完整工具调用链路 |
| B8.2 | E2E：Dashboard 冒烟测试 | [ ] | — | 6 个页面冒烟测试 + AppTest 框架 |
| B8.3 | 全链路 E2E 验收 | [ ] | — | ingest → query → evaluate 完整脚本验证 |

#### Phase C：Agent 基础设施

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| C1 | Tool 抽象基类 | [ ] | — | `Tool` 基类：`name` / `description` / `parameters` / `execute()` |
| C2 | Tool 注册中心 | [ ] | — | `ToolRegistry`：按名称查找 + 生成 Tool 描述列表（供 LLM Prompt 使用）|
| C3 | local_search Tool | [ ] | — | 包装 RAG Pipeline 的 `HybridSearch.search()` |
| C4 | query_rewriter Tool | [ ] | — | LLM 驱动的 Query 改写，注入深判反馈的「信息缺口」|
| C5 | 混合判分器（HybridScorer） | [ ] | — | 快速阈值 + LLM 深判双模式（decision / confidence）+ 阈值自适应 |
| C6 | Agent 记忆管理（AgentMemory） | [ ] | — | 对话历史存储 + 上下文窗口管理 + Token 估算 |
| C7 | Prompt 模板文件 | [ ] | — | `react_system.md` / `scorer_prompt.md` / `rewriter_prompt.md` |
| C8 | Agent 配置模块 | [ ] | — | `settings.yaml` 中 agent 配置节 + `AgentConfig` dataclass |

#### Phase D：自研 ReAct 版

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| D1 | ReAct 循环引擎 | [ ] | — | `ReActAgent` 类：Think → Act → Observe 主循环 + 最大 3 轮限制 |
| D2 | Thought 解析器 | [ ] | — | 解析 LLM 输出的 Thought/Action/Action Input 结构 |
| D3 | 答案合成器（Synthesizer） | [ ] | — | 汇总所有 Observe 结果，生成最终答案（含引用标注）|
| D4 | CLI 入口 | [ ] | — | `python -m agent_layer.versions.custom_agent` 可交互式问答 |
| D5 | 自研版单元测试 | [ ] | — | ReAct 循环各环节独立测试 + Mock Tool 集成测试 |

#### Phase E：LangGraph 版

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| E1 | StateGraph 决策图定义 | [ ] | — | `StateGraph` 节点：search → score → rewrite → web_search → finish |
| E2 | ToolNode 集成 | [ ] | — | 复用 Phase C Tool 集，注册为 LangChain Tool |
| E3 | State 管理与条件路由 | [ ] | — | `AgentState` 定义 + 条件边（基于混合判分结果路由）|
| E4 | LangGraph 版单元测试 | [ ] | — | StateGraph 节点独立测试 + 完整流程测试 |
| E5 | 两版对比验证 | [ ] | — | 同一 Query 输入，验证两版 Agent 输出一致性（允许合理差异）|

#### Phase F：联网搜索 + 全链路

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| F1 | Tavily 搜索后端实现 | [ ] | — | `TavilySearch` 类 + API 封装 + 错误处理 |
| F2 | SerpAPI 搜索后端实现 | [ ] | — | `SerpAPISearch` 类 + API 封装 + 错误处理 |
| F3 | Web Search Tool 注册 | [ ] | — | `web_search.py` 统一接口 + `settings.yaml` 后端切换 |
| F4 | 缓存接口预留 | [ ] | — | `CacheInterface` 抽象类（不实现逻辑，仅占位 + 文档注释）|
| F5 | 全链路端到端验证 | [ ] | — | 本地搜索 → 判分 → 改写 → 联网 → confidence 评估，完整跑通 |

#### Phase G：测试 + 打磨

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| G1 | Agent 层单元测试补齐 | [ ] | — | 所有 Tool / Scorer / Memory 单元测试覆盖率 ≥ 80% |
| G2 | RAG + Agent 集成测试 | [ ] | — | 真实 PDF 文档摄入 → 检索 → Agent 决策全链路 |
| G3 | README 完善 | [ ] | — | 项目介绍 + 快速开始 + 架构说明 + 配置指南 + MCP 集成说明 |
| G4 | 代码清理与注释补全 | [ ] | — | 关键模块 docstring + 类型标注补全 |

#### Phase H：简历 + 面试准备

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| H1 | 调用 resume-writer skill 生成简历初稿 | [ ] | — | 结合项目技术亮点 + 用户背景定制 |
| H2 | 面试追问准备（Agent 方向） | [ ] | — | ReAct 原理、LangGraph vs 自研对比、级联降级设计决策 |
| H3 | 面试追问准备（RAG 方向） | [ ] | — | 混合检索原理、可插拔架构设计、评估体系、MCP 协议 |
| H4 | 简历迭代优化 | [ ] | — | 结合面试反馈持续修改 |

---

### 📈 总体进度

| 阶段 | 总任务数 | 已完成 | 进度 |
|------|---------|--------|------|
| Phase A | 5 | 5 | 100% |
| Phase B-1 | 14 | 4 | 29% |
| Phase B-2 | 14 | 0 | 0% |
| Phase B-3 | 7 | 0 | 0% |
| Phase B-4 | 5 | 0 | 0% |
| Phase B-5 | 5 | 0 | 0% |
| Phase B-6 | 6 | 0 | 0% |
| Phase B-7 | 4 | 0 | 0% |
| Phase B-8 | 3 | 0 | 0% |
| Phase C | 8 | 0 | 0% |
| Phase D | 5 | 0 | 0% |
| Phase E | 5 | 0 | 0% |
| Phase F | 5 | 0 | 0% |
| Phase G | 4 | 0 | 0% |
| Phase H | 4 | 0 | 0% |
| **总计** | **94** | **9** | **9.6%** |

---

## 技术选型汇总

| 组件 | 选型 | 说明 |
|------|------|------|
| RAG 框架 | 自研 Pipeline（参考 clean-start） | 复用原项目可插拔架构 |
| 向量库 | Chroma | 嵌入式，零部署 |
| Embedding | OpenAI / Azure | 通过工厂模式可切换 |
| LLM | Azure OpenAI / OpenAI | 通过工厂模式可切换 |
| MCP SDK | Python `mcp` | 官方 SDK，stdio transport |
| Agent 框架 | 自研 + LangGraph 双版本 | 面试对比叙事 |
| 联网搜索 | Tavily + SerpAPI | 可切换后端 |
| 配置管理 | YAML + .env | python-dotenv 加载 |
| Dashboard | Streamlit | 六页管理平台（复现） |
| 评估 | Ragas + Custom | 可插拔评估 |

---

## 自审查

- [x] 无 TBD / TODO 占位
- [x] 架构与功能描述一致
- [x] 范围适中，94 个子任务覆盖 8 个阶段，可在 1-2 个月内完成
- [x] 无歧义需求，所有判定标准已明确
- [x] 缓存接口标注「预留不实现」，避免范围蔓延
- [x] Legal Splitter 已明确砍掉
- [x] .env 密钥管理方案已确定
- [x] 进度跟踪表已建立，与 DEV_SPEC 格式对齐，每个子任务功能单一、可独立验收
- [x] 每个阶段包含明确的验证标准
