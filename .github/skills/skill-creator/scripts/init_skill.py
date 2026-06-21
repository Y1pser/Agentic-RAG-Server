"""Initialize a new skill directory skeleton.

Usage: python init_skill.py <skill-name> --path .github/skills
"""

import sys
from pathlib import Path


SKILL_TEMPLATE = """---
name: {skill_name}
description: [One-line description of what this skill does — used to decide relevance during recall]
---

# {skill_title}

## Overview

[What this skill does and when to use it. 1-2 paragraphs.]

## Pipeline

### Step 1: [First Step]

- [ ] [Action item with exact command or code]

```bash
echo "Hello"
```

### Step 2: [Second Step]

- [ ] [Action item]

## Hard Gates

- **You MUST [rule].**
- **You MUST NOT [rule].**
"""


def main():
    if len(sys.argv) < 2:
        print("Usage: python init_skill.py <skill-name> [--path <dir>]")
        sys.exit(1)

    skill_name = sys.argv[1]
    base_dir = Path.cwd()

    # Parse --path
    if "--path" in sys.argv:
        idx = sys.argv.index("--path")
        base_dir = Path(sys.argv[idx + 1])

    skill_dir = base_dir / skill_name
    if skill_dir.exists():
        print(f"ERROR: Directory already exists: {skill_dir}")
        sys.exit(1)

    skill_dir.mkdir(parents=True)
    (skill_dir / "scripts").mkdir()
    (skill_dir / "references").mkdir()

    skill_title = " ".join(w.capitalize() for w in skill_name.split("-"))
    content = SKILL_TEMPLATE.format(skill_name=skill_name, skill_title=skill_title)

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(content, encoding="utf-8")

    print(f"✓ Created skill skeleton at {skill_dir}")
    print(f"  ├── SKILL.md")
    print(f"  ├── scripts/")
    print(f"  └── references/")
    print(f"\nNext: Edit {skill_md} to define your skill's behavior.")


if __name__ == "__main__":
    main()
