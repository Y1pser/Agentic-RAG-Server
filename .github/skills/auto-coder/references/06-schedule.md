## 6. Agent 长短期记忆设计

> 参考：OpenClaw builtin memory engine — 三层记忆模型：上下文窗口（Compaction）→ 短期笔记（YYYY-MM-DD.md）→ 长期记忆（MEMORY.md），SQLite + FTS5 + 向量混合搜索索引。

### 6.1 三层记忆模型

```
┌──────────────────────────────────────────────────┐
│  长期记忆 (MEMORY.md)                             │
│  持久化：用户偏好、重要决策、知识沉淀                │
│  跨周/月，Agent 通过 memory_store 主动策展         │
│  每次会话自动注入（有 token 预算，默认 ~2K tokens） │
├──────────────────────────────────────────────────┤
│  短期笔记 (memory/YYYY-MM-DD.md)                  │
│  持久化：按天滚动的追加式日志                      │
│  跨会话保留，仅通过 memory_search 按需检索        │
│  不自动注入上下文（避免 token 浪费）              │
├──────────────────────────────────────────────────┤
│  上下文窗口 (AgentMemory 对话历史)                 │
│  易失：当前会话的 Think/Act/Observe               │
│  Compaction 机制防止溢出                          │
│  会话结束即释放                                   │
└──────────────────────────────────────────────────┘
```

### 6.2 Compaction 机制（上下文窗口管理）

**触发条件**：当前上下文 token 数超过模型窗口上限的 70%（可配置，默认 `compaction_threshold: 0.7`）。

**压缩策略**：
- 保留最近 ~N 条原始消息（可配置 `keep_recent_tokens`，默认约 20K tokens）
- 更早的消息用 LLM 分块生成结构化摘要替代
- 摘要强制保留字段：

```
[任务状态]  当前任务的进度、下一步
[关键决策]  做了什么选择及理由
[TODO]      尚未完成的事项
[关键标识]  文件名、函数名、ID、URL 等精确引用
```

- 压缩后上下文结构：`[系统提示] + [摘要块]* + [最近原文]`
- Compaction 由 `AgentMemory.compact()` 方法执行，在每一轮 Think 之前检查并触发

**Pre-compaction flush**：压缩前，静默触发一轮 Agent 回合：
- 提示模型："当前上下文即将超限，请检查对话，把值得长期保留的信息写入记忆。"
- Agent 可以调用 `memory_store` / `memory_log` 沉淀关键内容
- Flush 回合结束后才执行压缩

### 6.3 长期记忆存储

**文件布局**：

```
agent_layer/memory/
├── MEMORY.md                  # 策展过的持久记忆
├── YYYY-MM-DD.md              # 每日短期笔记（追加式日志）
├── .gitkeep                   # 保持目录在 git 中
└── index.db                   # SQLite 混合索引（gitignored）
```

**MEMORY.md**：
- 内容：用户偏好、重要决策、知识沉淀、项目级别的事实
- 写操作：Agent 通过 `memory_store` 工具写入，支持 upsert-by-tag
- 读操作：每次会话启动时自动注入（有 token 预算控制，默认 ~2K tokens）
- 维护：Agent 定期去重、清理过时条目

**memory/YYYY-MM-DD.md**：
- 内容：追加式日志、任务记录、观察、会话摘要
- 写操作：Agent 通过 `memory_log` 工具追加
- 读操作：始终通过 `memory_search` 按需检索，不自动注入上下文（一天的日志可能很长，自动注入浪费 token）
- 生命周期：自然滚动，不需手动清理

### 6.4 混合搜索索引

对标 OpenClaw builtin 引擎：**SQLite + FTS5（BM25 关键词搜索）+ 向量嵌入（余弦相似度语义搜索）**。

**索引结构**（`index.db`）：

| 表 | 用途 |
|---|---|
| `files` | 跟踪 mtime、size、content hash，用于增量索引 |
| `chunks` | 文本块（~400 tokens，80 token 重叠）+ 嵌入向量（JSON） |
| `chunks_fts` | FTS5 全文索引，BM25 关键词搜索 |

**检索流程**：

```
memory_search(query, top_k=10)
  │
  ├─→ 向量搜索：embed(query) → 余弦相似度 → top_k*2 候选
  ├─→ BM25 搜索：FTS5 → 分数归一化（1/(1+rank)）→ top_k*2 候选
  │
  └─→ 混合融合（权重 0.7 向量 + 0.3 BM25）
       └─→ 去重 → 排序 → 返回 top_k
```

**Embedding 提供方**：复用项目已有的 Embedding 工厂（`EmbeddingFactory.create(settings)`），和 RAG 层共享配置。

**增量索引**：
- 文件变更时（mtime/hash 变化），触发增量重建
- 由 `MemoryStore.reindex()` 方法执行，Agent 启动时调用一次

### 6.5 记忆工具集

Agent 通过 Tool 基类（C1）暴露四个记忆操作工具：

| 工具 | 对应文件 | 功能 |
|------|---------|------|
| `memory_search` | 按需检索 | 混合搜索 `MEMORY.md` + `*.md`，返回相关片段（含来源路径 + 分数） |
| `memory_get` | 按路径读取 | 返回指定文件全文（如 `MEMORY.md` 或 `YYYY-MM-DD.md`） |
| `memory_log` | 追加到短期笔记 | 追加一条日志到今日 `YYYY-MM-DD.md`，自动带时间戳 |
| `memory_store` | 写入长期记忆 | 策展式写入 `MEMORY.md`，按 tag 分组，支持 upsert-by-tag |

**memory_store upsert 格式**（MEMORY.md 中）：

```markdown
<!-- memory:user-preference -->
- 用户偏好 Python 语言，惯用 pytest 做测试
- 代码风格偏好 type hints + docstring
<!-- /memory:user-preference -->
```

相同 tag 的内容会被原地替换，避免重复写入。

### 6.6 沉淀桥接（三层之间的衔接）

| 时机 | 动作 | 方向 |
|------|------|------|
| **Pre-compaction flush** | 上下文超限前，静默回合提示 Agent 写入 `memory_log` / `memory_store` | 上下文 → 短期笔记 / 长期记忆 |
| **Agent 主动策展** | 任务中发现重要信息，调用 `memory_store` 写入 MEMORY.md | 上下文 → 长期记忆 |
| **会话结束提醒** | 会话退出时，提示 Agent 做回顾，关键产出写入今日笔记 | 上下文 → 短期笔记 |

> **不做** OpenClaw 的完整 Dreaming（Light→Deep→REM 后台自动沉淀），那是进阶版范围。Agent 主动驱动沉淀，不设后台定时任务。

### 6.7 配置节

`settings.yaml` 中的 `memory` 配置节：

```yaml
memory:
  enabled: true
  workspace_dir: "agent_layer/memory"
  # Compaction
  compaction_threshold: 0.7        # 上下文使用率触发压缩
  keep_recent_tokens: 20000        # 压缩时保留最近 N tokens 原文
  # 索引
  chunk_size: 400                  # 索引分块大小（tokens）
  chunk_overlap: 80                # 分块重叠大小
  hybrid_weight_vector: 0.7        # 混合搜索向量权重
  hybrid_weight_bm25: 0.3          # 混合搜索 BM25 权重
  # 自动注入
  inject_long_term_tokens: 2000    # 每次会话注入 MEMORY.md 的 token 预算
  # 短期笔记不自动注入，一律通过 memory_search 按需检索
```

---
