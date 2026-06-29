"""Configuration loading and validation.

All settings (including API keys) live in ``config/settings.yaml``.
There is no .env file — keys are read directly from the YAML config.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """Application settings, loaded from config/settings.yaml."""

    llm: dict = field(default_factory=dict)
    embedding: dict = field(default_factory=dict)
    splitter: dict = field(default_factory=dict)
    vector_store: dict = field(default_factory=dict)
    retrieval: dict = field(default_factory=dict)
    rerank: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)
    observability: dict = field(default_factory=dict)
    dashboard: dict = field(default_factory=dict)
    agent: dict = field(default_factory=dict)
    ingestion: dict = field(default_factory=dict)
    vision: dict = field(default_factory=dict)


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

    raw = _load_yaml(config_path)

    # Merge local overrides if present (same directory, gitignored)
    local_path = config_path.parent / "settings.local.yaml"
    if local_path.exists():
        local_raw = _load_yaml(local_path)
        _deep_merge(raw, local_raw)

    settings = Settings()
    for key in Settings.__dataclass_fields__:
        if key in raw:
            setattr(settings, key, raw[key])
    return settings


def _load_yaml(path: Path) -> dict:
    """Load a single YAML file, returning a dict (empty if file is empty)."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge ``override`` into ``base`` in-place.

    Dict values are merged recursively; everything else is overwritten.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def validate_settings(settings: Settings) -> None:
    """Validate that required config fields are present.

    Args:
        settings: Settings instance to validate.

    Raises:
        ValueError: If a required field is missing.
    """
    required = {
        "llm.provider": settings.llm.get("provider"),
        "llm.api_key": settings.llm.get("api_key"),
        "embedding.provider": settings.embedding.get("provider"),
        "vector_store.backend": settings.vector_store.get("backend"),
    }
    for field_path, value in required.items():
        if not value:
            raise ValueError(f"Missing required config field: {field_path}")
