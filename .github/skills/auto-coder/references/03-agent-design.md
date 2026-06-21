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
