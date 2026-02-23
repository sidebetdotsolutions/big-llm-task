"""Core job execution logic with provider fallback and retry."""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from botocore.exceptions import BotoCoreError, ClientError

from .concatenator import AssembledInput, assemble_input
from .config import load_config
from .credentials import AllCredentials, load_credentials
from .models import (
    AttemptRecord,
    JobConfig,
    JobStatus,
    LLMResponse,
    MetadataFile,
    ProviderConfig,
    StatusFile,
)
from .providers.anthropic import AnthropicProvider
from .providers.base import BaseProvider
from .providers.bedrock import BedrockProvider
from .providers.openrouter import OpenRouterProvider

logger = logging.getLogger(__name__)

JOBS_ROOT = Path(__file__).resolve().parent.parent / "jobs"

# Errors that mean "skip to next provider" rather than retry
NON_RETRYABLE_STATUS_CODES = {401, 403, 400, 413}


def _write_status(job_dir: Path, status: StatusFile) -> None:
    path = job_dir / "status.json"
    path.write_text(
        status.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _write_metadata(job_dir: Path, metadata: MetadataFile) -> None:
    path = job_dir / "output" / "metadata.json"
    path.write_text(
        metadata.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _build_provider(
    pc: ProviderConfig, credentials: AllCredentials
) -> BaseProvider:
    """Instantiate the correct provider from config + credentials."""
    cred = credentials.providers.get(pc.name)
    if pc.name == "anthropic":
        return AnthropicProvider(api_key=cred.api_key, model=pc.model)
    elif pc.name == "openrouter":
        return OpenRouterProvider(api_key=cred.api_key, model=pc.model)
    elif pc.name == "bedrock":
        return BedrockProvider(model=pc.model)
    else:
        raise ValueError(f"Unknown provider: {pc.name}")


def _is_retryable(exc: Exception) -> bool:
    """Determine if an exception is retryable."""
    # httpx status errors
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code in NON_RETRYABLE_STATUS_CODES:
            return False
        if code in {429, 500, 502, 503, 529}:
            return True
        return False

    # Anthropic SDK errors
    import anthropic

    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.InternalServerError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code in NON_RETRYABLE_STATUS_CODES:
            return False
        if exc.status_code in {429, 500, 502, 503, 529}:
            return True
        return False

    # Boto / AWS errors
    if isinstance(exc, ClientError):
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code in {"ThrottlingException", "ServiceUnavailableException"}:
            return True
        return False
    if isinstance(exc, BotoCoreError):
        return True

    # Connection-level errors
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException, ConnectionError, TimeoutError)):
        return True

    return False


def _get_retry_after(exc: Exception) -> float | None:
    """Extract Retry-After header value if present."""
    if isinstance(exc, httpx.HTTPStatusError):
        retry_after = exc.response.headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
    return None


def _compute_backoff(attempt: int, config: JobConfig) -> float:
    base = min(
        config.retry.initial_backoff_seconds
        * (config.retry.backoff_multiplier ** (attempt - 1)),
        config.retry.max_backoff_seconds,
    )
    return base + random.uniform(0, 1)


def run_job(
    job_id: str,
    provider_override: str | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    """Execute a job end-to-end.

    Raises SystemExit(1) on failure.
    """
    job_dir = JOBS_ROOT / job_id

    # --- 1. VALIDATE ---
    if not job_dir.is_dir():
        raise SystemExit(f"Job directory not found: {job_dir}")

    # Check for running job
    status_path = job_dir / "status.json"
    if status_path.is_file():
        try:
            current_status = StatusFile.model_validate_json(
                status_path.read_text(encoding="utf-8")
            )
            if current_status.status == JobStatus.RUNNING and not force:
                raise SystemExit(
                    f"Job {job_id} is currently running. Use --force to override."
                )
        except (ValueError, json.JSONDecodeError):
            pass  # Corrupted status file, proceed anyway

    # Ensure output dir exists
    output_dir = job_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # Write initial status
    status = StatusFile(status=JobStatus.PENDING)
    _write_status(job_dir, status)

    # --- 2. LOAD CONFIG ---
    credentials = load_credentials()
    config = load_config(job_dir, credentials)

    if provider_override:
        config.providers = [
            p for p in config.providers if p.name == provider_override
        ]
        if not config.providers:
            raise SystemExit(
                f"Provider '{provider_override}' not available (no credentials or not in config)."
            )

    if not config.providers:
        raise SystemExit("No providers available. Check credentials.")

    logger.info(
        "Config loaded: %d provider(s), temperature=%.1f, max_tokens varies per provider.",
        len(config.providers),
        config.temperature,
    )

    # --- 3. ASSEMBLE INPUT ---
    assembled: AssembledInput = assemble_input(job_dir)
    logger.info(
        "Input: %d files, %d chars. System prompt: %s.",
        len(assembled.input_files),
        assembled.total_characters,
        "yes" if assembled.system_prompt else "no",
    )

    if dry_run:
        print(f"[DRY RUN] Job: {job_id}")
        print(f"  Providers: {[p.name for p in config.providers]}")
        print(f"  Input files: {assembled.input_files}")
        print(f"  Input chars: {assembled.total_characters}")
        print(f"  System prompt: {'yes' if assembled.system_prompt else 'no'}")
        print(f"  Temperature: {config.temperature}")
        return

    # --- 4. EXECUTE ---
    status.status = JobStatus.RUNNING
    status.started_at = datetime.now(timezone.utc)
    _write_status(job_dir, status)

    response_path = output_dir / "response.md"
    final_response: LLMResponse | None = None

    for pc in config.providers:
        provider = _build_provider(pc, credentials)

        if not provider.validate():
            logger.warning("Provider %s failed validation, skipping.", pc.name)
            status.attempts.append(
                AttemptRecord(
                    provider=pc.name,
                    attempt=0,
                    result="error",
                    error="Validation failed",
                )
            )
            continue

        for attempt in range(1, config.retry.max_attempts_per_provider + 1):
            logger.info(
                "Trying %s (attempt %d/%d, model=%s, max_tokens=%d)...",
                pc.name,
                attempt,
                config.retry.max_attempts_per_provider,
                pc.model,
                pc.max_tokens,
            )

            try:
                gen = provider.stream(
                    user_message=assembled.user_message,
                    system_prompt=assembled.system_prompt,
                    temperature=config.temperature,
                    max_tokens=pc.max_tokens,
                )

                # Stream to disk
                with open(response_path, "w", encoding="utf-8") as f:
                    chunk_count = 0
                    chars_received = 0
                    last_log_time = time.monotonic()

                    try:
                        while True:
                            chunk = next(gen)
                            f.write(chunk)
                            if config.stream_to_disk:
                                f.flush()
                            chunk_count += 1
                            chars_received += len(chunk)

                            now = time.monotonic()
                            if now - last_log_time >= 10:
                                logger.info(
                                    "  Streaming: %d chunks, %d chars received...",
                                    chunk_count,
                                    chars_received,
                                )
                                last_log_time = now
                    except StopIteration as stop:
                        final_response = stop.value

                if final_response is None:
                    raise RuntimeError("Provider stream ended without returning LLMResponse.")

                # Success
                attempt_record = AttemptRecord(
                    provider=pc.name,
                    attempt=attempt,
                    result="success",
                    latency_seconds=final_response.latency_seconds,
                )
                status.attempts.append(attempt_record)
                status.status = JobStatus.COMPLETED
                status.completed_at = datetime.now(timezone.utc)
                status.provider = pc.name
                status.model = final_response.model
                _write_status(job_dir, status)

                metadata = MetadataFile(
                    provider=final_response.provider,
                    model=final_response.model,
                    input_tokens=final_response.input_tokens,
                    output_tokens=final_response.output_tokens,
                    stop_reason=final_response.stop_reason,
                    latency_seconds=final_response.latency_seconds,
                    input_files=assembled.input_files,
                    input_characters=assembled.total_characters,
                    system_prompt_used=assembled.system_prompt is not None,
                    temperature=config.temperature,
                    max_tokens=pc.max_tokens,
                    timestamp=datetime.now(timezone.utc),
                )
                _write_metadata(job_dir, metadata)

                logger.info(
                    "Job %s completed: provider=%s, model=%s, "
                    "input_tokens=%d, output_tokens=%d, latency=%.1fs.",
                    job_id,
                    final_response.provider,
                    final_response.model,
                    final_response.input_tokens,
                    final_response.output_tokens,
                    final_response.latency_seconds,
                )
                return

            except Exception as exc:
                error_str = str(exc)
                logger.warning(
                    "Provider %s attempt %d failed: %s",
                    pc.name,
                    attempt,
                    error_str,
                )
                status.attempts.append(
                    AttemptRecord(
                        provider=pc.name,
                        attempt=attempt,
                        result="error",
                        error=error_str,
                    )
                )

                if not _is_retryable(exc):
                    logger.info(
                        "Non-retryable error from %s; moving to next provider.",
                        pc.name,
                    )
                    break

                if attempt < config.retry.max_attempts_per_provider:
                    retry_after = _get_retry_after(exc)
                    if retry_after:
                        sleep_time = retry_after
                    else:
                        sleep_time = _compute_backoff(attempt, config)
                    logger.info("Retrying in %.1f seconds...", sleep_time)
                    time.sleep(sleep_time)

    # All providers exhausted
    status.status = JobStatus.FAILED
    status.completed_at = datetime.now(timezone.utc)
    status.error = "All providers exhausted"
    _write_status(job_dir, status)

    logger.error("Job %s failed: all providers exhausted.", job_id)
    raise SystemExit(1)
