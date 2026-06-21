"""Configuration loading and validation."""

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
