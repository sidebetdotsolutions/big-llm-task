"""Credential loader — parses YAML frontmatter from credentials.md files."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

VAULT_ROOT = Path.home() / "empire" / "vault" / "credentials"

PROVIDER_FILES = {
    "anthropic": "anthropic/credentials.md",
    "openrouter": "openrouter/credentials.md",
    "bedrock": "aws/credentials.md",
}


@dataclass
class ProviderCredentials:
    api_key: Optional[str] = None
    available: bool = False


@dataclass
class AllCredentials:
    providers: dict[str, ProviderCredentials] = field(default_factory=dict)

    def is_available(self, provider_name: str) -> bool:
        cred = self.providers.get(provider_name)
        return cred is not None and cred.available


def _parse_frontmatter(text: str) -> dict:
    """Extract YAML from the first pair of --- fences."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}
    end = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = i
            break
    if end is None:
        return {}
    yaml_text = "\n".join(lines[1:end])
    try:
        return yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse YAML frontmatter: %s", exc)
        return {}


def load_credentials() -> AllCredentials:
    """Load credentials for all providers from the vault."""
    all_creds = AllCredentials()

    for provider_name, rel_path in PROVIDER_FILES.items():
        cred_path = VAULT_ROOT / rel_path
        cred = ProviderCredentials()

        if not cred_path.is_file():
            logger.debug("Credential file not found for %s: %s", provider_name, cred_path)
            all_creds.providers[provider_name] = cred
            continue

        try:
            text = cred_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read credential file for %s: %s", provider_name, exc)
            all_creds.providers[provider_name] = cred
            continue

        frontmatter = _parse_frontmatter(text)

        if provider_name == "bedrock":
            # For AWS/Bedrock, file existence is enough — auth via environment.
            cred.available = True
            logger.debug("AWS credentials file found; Bedrock marked available.")
        else:
            api_key = frontmatter.get("api_key", "")
            if api_key:
                cred.api_key = str(api_key)
                cred.available = True
                logger.debug("Loaded API key for %s.", provider_name)
            else:
                logger.debug("No api_key found in frontmatter for %s.", provider_name)

        all_creds.providers[provider_name] = cred

    return all_creds
