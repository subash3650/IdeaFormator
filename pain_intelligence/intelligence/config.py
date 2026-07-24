"""Configuration loader for the Knowledge Extraction Engine."""

from __future__ import annotations

from typing import Any

from pain_intelligence.utils.config import load_config as _load_config, get_nested


DEFAULT_CONFIG: dict[str, Any] = {
    "intelligence": {
        "output_dir": "reports",
        "debug": False,
        "sampling": {
            "enabled": False,
            "max_documents": 100000,
            "strategy": "stratified",
            "strata_column": "platform",
        },
        "confidence": {
            "exact_dictionary_match": 0.95,
            "pattern_match": 0.85,
            "fuzzy_match": 0.65,
            "heuristic": 0.55,
            "statistical": 0.45,
            "evidence_base": 0.70,
            "signal_threshold": 0.80,
        },
        "extraction": {
            "ngram_min_frequency": 5,
            "ngram_max_features": 100,
            "keyword_max_features": 100,
            "entity_min_mentions": 3,
            "pattern_min_confidence": 0.5,
        },
        "evidence": {
            "min_observation_count": 3,
            "min_document_count": 3,
            "top_snippets_count": 5,
        },
        "signals": {
            "min_document_count": 10,
            "max_avg_rating": 3.0,
            "min_confidence": 0.7,
        },
        "visualization": {
            "theme": "plotly_dark",
            "width": 1200,
            "height": 600,
        },
        "dashboard": {
            "host": "localhost",
            "port": 8501,
        },
    },
}


def load_intelligence_config(config_path: str | None = None) -> dict[str, Any]:
    """Load intelligence config, merging with system config and defaults."""
    merged = dict(DEFAULT_CONFIG)

    if config_path:
        try:
            sys_cfg = _load_config(config_path)
            intel_cfg = sys_cfg.get("intelligence", {})
            merged["intelligence"].update(intel_cfg)
        except Exception:
            pass

    return merged


def get_intelligence_setting(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Get a nested intelligence config setting."""
    intel = config.get("intelligence", config)
    current: Any = intel
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
            if current is None:
                return default
        else:
            return default
    return current if current is not None else default