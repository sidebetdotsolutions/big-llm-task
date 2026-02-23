"""Default configuration and per-job config merging."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from .credentials import AllCredentials
from .models import JobConfig, ProviderConfig, RetryConfig

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "providers": [
        {
            "name": "anthropic",
            "model": "claude-sonnet-4-6",
            "max_tokens": 16384,
        },
        {
            "name": "openrouter",
            "model": "google/gemini-3.1-pro-preview",
            "max_tokens": 16384,
        },
        {
            "name": "bedrock",
            "model": "us.meta.llama4-scout-17b-instruct-v1:0",
            "max_tokens": 8192,
        },
    ],
    "temperature": 1.0,
    "retry": {
        "max_attempts_per_provider": 3,
        "initial_backoff_seconds": 2,
        "backoff_multiplier": 2,
        "max_backoff_seconds": 30,
    },
    "stream_to_disk": True,
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into base, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_job_config_yaml(job_dir: Path) -> dict[str, Any]:
    """Load per-job config.yaml if it exists."""
    config_path = job_dir / "config.yaml"
    if not config_path.is_file():
        return {}
    try:
        text = config_path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except (yaml.YAMLError, OSError) as exc:
        logger.warning("Failed to load job config %s: %s", config_path, exc)
        return {}


def _apply_max_tokens_shorthand(merged: dict) -> dict:
    """If top-level max_tokens is set, apply it to all providers that don't override it."""
    top_level = merged.pop("max_tokens", None)
    if top_level is not None and "providers" in merged:
        for p in merged["providers"]:
            if "max_tokens" not in p:
                p["max_tokens"] = top_level
    return merged


def load_config(job_dir: Path, credentials: AllCredentials) -> JobConfig:
    """Load and merge configuration, filtering unavailable providers."""
    job_overrides = _load_job_config_yaml(job_dir)
    merged = _deep_merge(DEFAULTS, job_overrides)
    merged = _apply_max_tokens_shorthand(merged)

    # Build provider configs
    raw_providers = merged.pop("providers", [])
    providers: list[ProviderConfig] = []
    for p in raw_providers:
        pc = ProviderConfig(**p)
        if credentials.is_available(pc.name):
            providers.append(pc)
        else:
            logger.info("Provider %s skipped — credentials unavailable.", pc.name)

    if not providers:
        logger.warning("No providers available after credential filtering.")

    # Build retry config
    retry_data = merged.pop("retry", {})
    retry = RetryConfig(**retry_data)

    return JobConfig(
        providers=providers,
        temperature=merged.get("temperature", 1.0),
        retry=retry,
        stream_to_disk=merged.get("stream_to_disk", True),
    )
