---
name: package
description: Cleans and packages the Agentic RAG Server project for distribution. Removes caches, virtual environments, build artifacts, data, logs, and sanitizes API keys.
---

# Package: Clean & Package Project for Distribution

## Overview

Clean the **Agentic RAG Server** project so it's ready for distribution — GitHub push, ZIP delivery, or interviewer review.

## Pipeline

### Step 1: Dry-Run

Show what will be removed/cleaned:

```bash
echo "=== Files to be removed ==="
find . -type d -name "__pycache__" -o -name ".pytest_cache" -o -name "*.egg-info" | head -20
echo ""
echo "=== Data directories (if --keep-data not set) ==="
ls -la data/db/ data/images/ logs/ 2>/dev/null || echo "(empty or not found)"
echo ""
echo "=== .env file ==="
ls -la .env 2>/dev/null && echo "WILL BE EXCLUDED (never packaged)"
```

### Step 2: Clean

Remove all non-essential files:

```bash
# Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# Virtual environment
rm -rf .venv/ venv/

# Build artifacts
rm -rf dist/ build/ *.egg-info/

# Data (unless --keep-data)
rm -rf data/db/* data/images/*

# Logs
rm -rf logs/

# Coverage
rm -rf htmlcov/ .coverage

# IDE
rm -rf .vscode/ .idea/

# OS
find . -type f -name ".DS_Store" -delete 2>/dev/null
find . -type f -name "Thumbs.db" -delete 2>/dev/null
```

### Step 3: Sanitize API Keys

If `config/settings.yaml` contains actual API keys, replace with placeholders:

```bash
# Replace actual keys with env var references
sed -i 's/api_key: .*/api_key: ${OPENAI_API_KEY}/' config/settings.yaml
sed -i 's/tavily_api_key: .*/tavily_api_key: ${TAVILY_API_KEY}/' config/settings.yaml
```

### Step 4: Verify

```bash
# Check no keys leaked
grep -r "sk-" . --include="*.yaml" --include="*.py" 2>/dev/null && echo "WARNING: API keys found!" || echo "✓ No API keys found"

# Check .env is gitignored
git check-ignore .env && echo "✓ .env is gitignored" || echo "WARNING: .env not gitignored"

# Check clean state
echo ""
echo "Package ready! Files remaining:"
find . -type f -not -path './.git/*' | wc -l
```

## Options

| Flag | Effect |
|------|--------|
| `--keep-data` | Preserve `data/db/` and `data/images/` (ingested documents) |
| `--no-sanitize` | Skip API key masking in config |
