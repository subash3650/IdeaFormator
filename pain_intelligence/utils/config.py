"""Configuration loader using PyYAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If config file is malformed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config


def get_nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Retrieve a nested config value by key chain.

    Args:
        config: Configuration dictionary.
        *keys: Sequence of keys to traverse.
        default: Default value if key path not found.

    Returns:
        The value at the key path, or default.
    """
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current
