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
