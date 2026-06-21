---
name: skill-creator
description: Guide for creating effective agent skills in the Agentic RAG Server project. Covers skill anatomy, progressive disclosure design, and the 5-step creation process.
---

# Skill Creator: Build Custom Skills

## Overview

Guide for creating new skills or updating existing ones in the **Agentic RAG Server** project. Skills extend the agent's capabilities with specialized knowledge, workflows, or tool integrations.

## Skill Anatomy

Every skill lives in `.github/skills/<skill-name>/`:

```
.github/skills/<skill-name>/
├── SKILL.md              # Required: skill definition (YAML frontmatter + body)
├── scripts/              # Optional: executable scripts
│   └── helper.py
└── references/           # Optional: additional reference files
    └── guide.md
```

### SKILL.md Structure

```markdown
---
name: my-skill
description: One-line summary — used to decide relevance during recall
---

# Skill Title

## Overview
What this skill does, when to use it.

## Pipeline / Process
Step-by-step instructions for the agent.

## Hard Gates (if any)
Non-negotiable rules the agent MUST follow.
```

## Progressive Disclosure

Skills use a 3-level loading model:

| Level | What | When |
|-------|------|------|
| **Metadata** | `name` + `description` in frontmatter | Always in context |
| **Body** | Full SKILL.md content | Loaded on trigger |
| **Resources** | `scripts/` + `references/` files | Loaded on demand |

## Creation Process

### Step 1: Understand with Examples

Look at existing skills in `.github/skills/`:
- `auto-coder/SKILL.md` — complex pipeline with scripts
- `setup/SKILL.md` — interactive wizard
- `resume-writer/SKILL.md` — content generation

### Step 2: Plan Content

Answer these questions:
1. What's the trigger? (keywords the user says)
2. What's the output? (code, config, document, report?)
3. What rules must the agent follow? (hard gates)
4. What references does the agent need? (project-specific knowledge)

### Step 3: Initialize with Script

```bash
python .github/skills/skill-creator/scripts/init_skill.py <skill-name> --path .github/skills
```

This creates the directory skeleton with SKILL.md template.

### Step 4: Write the SKILL.md

- **Frontmatter**: name (kebab-case), description (one line, actionable)
- **Overview**: 1-2 paragraphs on purpose and trigger
- **Pipeline**: Step-by-step. Use checkboxes (`- [ ]`) for task lists
- **Hard Gates**: Rules the agent MUST follow (use bold, CAPS for emphasis)
- **Code blocks**: Always include exact commands, file paths, expected output

### Step 5: Iterate

Test the skill by triggering it. Observe what the agent does. Refine:
- Is the trigger clear? Does the agent know when to use it?
- Are instructions unambiguous? Does the agent follow the pipeline?
- Are file paths correct? Do scripts work?

## Quality Checklist

- [ ] Frontmatter has `name` and `description`
- [ ] Description is one line, describes what the skill DOES
- [ ] Pipeline steps are concrete (exact commands, file paths)
- [ ] Hard gates are clearly marked
- [ ] No TBD / TODO placeholders
- [ ] Scripts (if any) are tested and working
