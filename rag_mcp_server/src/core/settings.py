"""Configuration loading and validation."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """Application settings, loaded from config/settings.yaml."""

    llm: dict = field(default_factory=dict)
    embedding: dict = field(default_factory=dict)
    vector_store: dict = field(default_factory=dict)
    retrieval: dict = field(default_factory=dict)
    rerank: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)
    observability: dict = field(default_factory=dict)
    dashboard: dict = field(default_factory=dict)
    agent: dict = field(default_factory=dict)


def load_settings(path: str) -> Settings:
    """Load settings from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Settings dataclass instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for config loading. "
            "Install it with: pip install pyyaml"
        )

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    settings = Settings()
    for key in Settings.__dataclass_fields__:
        if key in raw:
            setattr(settings, key, raw[key])
    return settings


def validate_settings(settings: Settings) -> None:
    """Validate that required config fields are present.

    Args:
        settings: Settings instance to validate.

    Raises:
        ValueError: If a required field is missing.
    """
    required = {
        "llm.provider": settings.llm.get("provider"),
        "embedding.provider": settings.embedding.get("provider"),
        "vector_store.backend": settings.vector_store.get("backend"),
    }
    for field_path, value in required.items():
        if not value:
            raise ValueError(f"Missing required config field: {field_path}")


def apply_env_overrides(settings: Settings) -> None:
    """Apply environment variable overrides to settings.

    Reads from .env file and os.environ. Keys follow the pattern:
    - OPENAI_API_KEY -> settings.llm["api_key"]
    - AZURE_OPENAI_API_KEY -> settings.llm["api_key"] (azure provider)
    - TAVILY_API_KEY -> settings.agent["web_search"]["tavily_api_key"]
    - SERPAPI_API_KEY -> settings.agent["web_search"]["serpapi_api_key"]
    - EMBEDDING_API_KEY -> settings.embedding["api_key"]

    Args:
        settings: Settings instance to apply overrides to.
    """
    # Load .env file if present
    try:
        from dotenv import load_dotenv
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass  # python-dotenv is optional

    # LLM keys
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and "api_key" not in settings.llm:
        settings.llm["api_key"] = openai_key

    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    if azure_key and settings.llm.get("provider") == "azure":
        settings.llm["api_key"] = azure_key

    # Embedding key
    embedding_key = os.getenv("EMBEDDING_API_KEY")
    if embedding_key and "api_key" not in settings.embedding:
        settings.embedding["api_key"] = embedding_key

    # Web search keys
    agent = settings.agent
    if "web_search" not in agent:
        agent["web_search"] = {}
    ws = agent["web_search"]

    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        ws["tavily_api_key"] = tavily_key

    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if serpapi_key:
        ws["serpapi_api_key"] = serpapi_key
