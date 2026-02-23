"""Input markdown assembly — reads and concatenates job input files."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

SEPARATOR = "\n\n---\n\n"


@dataclass
class AssembledInput:
    user_message: str
    system_prompt: Optional[str]
    input_files: list[str]
    total_characters: int


def _load_manifest(input_dir: Path) -> Optional[list[str]]:
    """Load manifest.yaml if present, returning ordered list of filenames."""
    manifest_path = input_dir / "manifest.yaml"
    if not manifest_path.is_file():
        return None
    try:
        text = manifest_path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if isinstance(data, dict) and "files" in data:
            return data["files"]
        logger.warning("manifest.yaml missing 'files' key; ignoring manifest.")
        return None
    except (yaml.YAMLError, OSError) as exc:
        logger.warning("Failed to load manifest.yaml: %s", exc)
        return None


def assemble_input(job_dir: Path) -> AssembledInput:
    """Assemble the input for an LLM call from a job directory.

    Raises ValueError if the input directory is invalid.
    """
    input_dir = job_dir / "input"
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # System prompt
    system_path = input_dir / "system.md"
    system_prompt: Optional[str] = None
    if system_path.is_file():
        system_prompt = system_path.read_text(encoding="utf-8")
        logger.debug("Loaded system prompt (%d chars).", len(system_prompt))

    # Determine file ordering
    manifest = _load_manifest(input_dir)
    if manifest is not None:
        filenames = manifest
        for fn in filenames:
            fp = input_dir / fn
            if not fp.is_file():
                raise ValueError(f"File listed in manifest.yaml not found: {fn}")
    else:
        # Glob *.md, exclude system.md, sort alphabetically
        md_files = sorted(input_dir.glob("*.md"))
        filenames = [
            f.name for f in md_files if f.name != "system.md"
        ]

    if not filenames:
        raise ValueError(
            f"No input .md files found (besides system.md) in {input_dir}"
        )

    # Concatenate
    sections: list[str] = []
    for fn in filenames:
        fp = input_dir / fn
        content = fp.read_text(encoding="utf-8")
        sections.append(f"<!-- source: {fn} -->\n{content}")

    user_message = SEPARATOR.join(sections)
    total_chars = len(user_message)

    logger.info(
        "Assembled %d input files, %d total characters. System prompt: %s.",
        len(filenames),
        total_chars,
        "yes" if system_prompt else "no",
    )

    return AssembledInput(
        user_message=user_message,
        system_prompt=system_prompt,
        input_files=filenames,
        total_characters=total_chars,
    )
