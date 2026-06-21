---
name: resume-writer
description: Generates customized resume project experience entries based on the Agentic RAG Server project. Uses a triangle model of writing principles + project highlights + user profile to produce targeted, quantified resume content for AI/ML interview positions.
---

# Resume Writer: Agentic RAG Server Resume Generator

## Overview

Generate a customized resume project experience entry based on the **Agentic RAG Server** project. The output follows the triangle model: **writing principles + project highlights + user profile = customized resume**.

## Process

### Step 1: Profile Collection

Ask the user 4 questions, ONE AT A TIME:

**Q1: Target Role**
- RAG Engineer (RAG 工程师)
- AI Agent Developer (Agent 开发工程师)
- LLM Application Engineer (大模型应用开发工程师)
- Full-Stack AI Engineer (全栈 AI 工程师)
- Backend Engineer → AI Transition (后端转 AI)
- Other (请说明)

**Q2: Business Context**
- What domain/industry are you targeting? (金融 / 法律 / 医疗 / 电商 / 通用 / 其他)
- Any specific business scenario you want to highlight?

**Q3: Technical Focus**
Which technical highlights do you want to emphasize? (Select 3-5):
- A. Agent + RAG dual-layer architecture
- B. Self-implemented ReAct engine (& LangGraph comparison)
- C. Hybrid quality scoring (fast threshold + LLM deep eval)
- D. Cascade fallback (local → rewrite → web search)
- E. Pluggable architecture (LLM/Embedding/Reranker/VectorStore)
- F. MCP protocol integration
- G. Multimodal image processing (Image Captioning)
- H. Evaluation framework (Ragas + Golden Test Set)
- I. Streamlit Dashboard with full-chain tracing
- J. Skill-driven development (auto-coder + qa-tester + setup)

**Q4: Special Requirements**
- Any metrics you want to emphasize?
- Any specific angle for your resume?
- Target language: 中文 only, or English + 中文?

### Step 2: Highlight Mapping

Based on the user's target role, map the selected highlights to resume bullet points. Here are the 10 project highlights:

| # | Highlight | Bullet Point Template |
|---|-----------|----------------------|
| 1 | **Agent + RAG 双层架构** | 设计 Agent + RAG 双层架构，Agent 层负责 ReAct 决策与工具调度，RAG 层通过 MCP 协议暴露标准化检索接口，实现"大脑指挥手脚"的松耦合设计 |
| 2 | **自研 ReAct 引擎 + LangGraph 对照** | 自研 ReAct 循环引擎（Think→Act→Observe），同时实现 LangGraph StateGraph 版本，双版本共享 Tool 集，对比验证工程化框架与自研方案的取舍 |
| 3 | **混合判分器** | 设计两段式检索质量判分器：快速阈值初筛 + LLM 深判兜底，边界情况自动下调阈值实现自适应，兼顾检索延迟与判分精度 |
| 4 | **级联降级策略** | 实现本地检索→Query 改写→联网搜索三级级联兜底，LLM 驱动的 Query 改写注入信息缺口反馈，联网结果附带信心评分标注 |
| 5 | **全链路可插拔架构** | LLM / Embedding / Reranker / VectorStore / WebSearch 五大模块均定义抽象接口 + 工厂模式，通过 YAML 配置一键切换后端，支持 OpenAI / Azure / DeepSeek / Ollama 等 5+ Provider 零代码迁移 |
| 6 | **MCP 协议集成** | 遵循 Model Context Protocol 标准，通过 Stdio Transport 暴露 `query_knowledge_hub` 等工具接口，可直接对接 Copilot / Claude Desktop 等 MCP Client |
| 7 | **多模态文档处理** | 采用 Image-to-Text 策略，集成 Vision LLM 自动生成图片描述并缝合进 Chunk，复用纯文本 RAG 链路实现"搜文字出图" |
| 8 | **混合检索 + 重排** | 实现 BM25 + Dense Embedding 双路召回 + RRF 融合 + Cross-Encoder/LLM Rerank 精排，平衡查全率与查准率 |
| 9 | **评估体系 + Dashboard** | 集成 Ragas 评估框架，建立 Golden Test Set 回归测试机制；搭建 Streamlit 六页面管理平台，覆盖数据浏览、Ingestion/Query 全链路追踪、评估面板 |
| 10 | **Skill 驱动全流程开发** | 编写 DEV_SPEC 规格文档驱动 auto-coder 自动编码、qa-tester 自动测试与修复、setup 一键环境配置，94 个子任务全量交付 |

> **Highlight #10 (Skill 驱动全流程) is recommended for ALL roles.** It demonstrates cutting-edge AI-assisted development methodology and is a strong interview talking point.

### Step 3: Resume Generation

Generate a resume entry following the **four-paragraph structure**:

1. **背景 (Background)**: 1-2 sentences. What problem exists? Why is this project needed?
2. **目标 (Objective)**: 1 sentence. What did you aim to achieve? Include quantified targets.
3. **过程 (Process)**: 4-6 bullet points. Each bullet: action verb + technical detail + quantification. Use the selected highlights.
4. **结果 (Results)**: 2-3 sentences. Quantified outcomes. Metrics should be defensible in interviews.

**Bullet point rules:**
- Start with action verbs: 设计 / 实现 / 构建 / 自研 / 集成 / 搭建
- Include technical specifics: algorithm names, protocol names, architecture patterns
- Quantify: X+ providers, Y test cases, Z% improvement
- Each bullet = one coherent technical achievement

### Step 4: Interview Follow-up Prediction

After the resume, generate 3-5 predicted interview questions:

```markdown
## 面试追问预测

1. **ReAct 引擎相关问题**
   - "你的 ReAct 循环和 LangGraph 的有什么本质区别？"
   - "为什么选择自研而不是直接用 LangGraph？"
   - 参考回答要点：控制力 vs 开发效率、Prompt 可定制性、面试展示深度

2. **级联降级相关问题**
   - "本地检索、改写、联网搜索的切换条件是什么？"
   - "阈值怎么选的？如何避免无限循环？"
   - 参考回答要点：混合判分两段式设计、3 轮硬上限、自适应阈值

3. **架构设计相关问题**
   - "Agent 层和 RAG 层为什么要分开？"
   - "可插拔架构具体怎么实现的？"
   - 参考回答要点：松耦合、工厂模式、配置驱动、工具复用

4. **评估与质量相关问题**
   - "怎么评估检索质量？怎么评估 Agent 决策质量？"
   - 参考回答要点：Ragas + Golden Test Set、混合判分器、对比实验

5. **Skill 驱动开发相关问题**
   - "Skill 是什么？你是怎么用 Skill 做开发的？"
   - 参考回答要点：Spec 驱动、auto-coder 自动编码、qa-tester 自动测试、94 子任务全量交付
```

## Output Format

```markdown
**<项目名>** | <时间范围> | <角色>

**背景**：...

**目标**：...

**过程**：
- ...
- ...

**结果**：...

**技术栈**：...

---

## 面试追问预测
...
```
