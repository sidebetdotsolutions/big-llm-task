
# big-llm-task: Architecture & Build Spec

## Overview

A Python CLI utility that executes large-context LLM queries using a job-directory convention. Jobs are self-contained directories with markdown inputs, optional configuration, and structured outputs. The utility supports three providers with automatic fallback: Anthropic → OpenRouter → AWS Bedrock.

---

## Directory Layout

### Project Root

```
/home/jlm/empire/holdings/sidebetsolutions/projects/big-llm-task/
  big_llm_task/
    __init__.py
    cli.py              # Typer CLI entrypoint
    runner.py           # Core job execution logic
    providers/
      __init__.py
      base.py           # Abstract provider interface
      anthropic.py      # Anthropic Messages API
      openrouter.py     # OpenRouter (OpenAI-compatible)
      bedrock.py        # AWS Bedrock Converse API
    credentials.py      # Credential loader (YAML frontmatter parser)
    config.py           # Default config + per-job config merging
    models.py           # Pydantic models for status, config, etc.
    concatenator.py     # Input markdown assembly
  jobs/                 # Job directories live here
  pyproject.toml
  README.md
```

### Job Directory Structure

```
jobs/{job_id}/
  input/
    system.md           # Optional. If present, used as the system prompt.
    *.md                # All other .md files concatenated as the user message.
    manifest.yaml       # Optional. If present, specifies explicit file ordering.
  output/
    response.md         # LLM response text
    metadata.json       # Token counts, model used, latency, cost estimate
  config.yaml           # Optional per-job overrides
  status.json           # Job lifecycle state
```

---

## Credentials

### Location

```
~/empire/vault/credentials/{provider}/credentials.md
```

Provider directory names (exact):
- `anthropic/credentials.md`
- `openrouter/credentials.md`
- `aws/credentials.md`

### Frontmatter Format

**Anthropic:**
```yaml
---
api_key: "sk-ant-..."
---
```

**OpenRouter:**
```yaml
---
api_key: "sk-or-..."
---
```

**AWS:**
```yaml
---
aws_access_key_id: ""
aws_secret_access_key: ""
---
```

### Credential Loading (`credentials.py`)

- Parse YAML between the first pair of `---` fences in each file.
- For Anthropic and OpenRouter: extract `api_key` string.
- For AWS: ignore the file contents entirely. AWS auth is handled by the environment (IAM role / env vars / ~/.aws). The loader only needs to confirm the file exists as a signal that AWS is configured. Hardcode region to `us-east-1`.
- If a credential file is missing or the key is empty, mark that provider as unavailable (skip in fallback chain, don't error).

---

## Configuration

### Global Defaults (`config.py`)

```python
DEFAULTS = {
    "providers": [
        {
            "name": "anthropic",
            "model": "claude-sonnet-4-6-20250514",
            "max_tokens": 16384,
        },
        {
            "name": "openrouter",
            "model": "google/gemini-3.1-pro-preview",
            "max_tokens": 16384,
        },
        {
            "name": "bedrock",
            "model": "us.meta.llama4-scout-17b-16e-instruct-v1:0",
            "max_tokens": 16384,
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
```

### Per-Job Overrides (`jobs/{job_id}/config.yaml`)

Any key from the defaults can be overridden. Example:

```yaml
temperature: 0.3
max_tokens: 32768
providers:
  - name: anthropic
    model: claude-sonnet-4-6-20250514
# Only use Anthropic for this job, no fallback
```

### Config Resolution

1. Load global defaults.
2. If `jobs/{job_id}/config.yaml` exists, deep-merge it over defaults.
3. Filter out providers whose credentials are unavailable.
4. Result is the effective config for the run.

---

## Input Assembly (`concatenator.py`)

### Ordering

1. If `input/manifest.yaml` exists, it contains an ordered list of filenames:
   ```yaml
   files:
     - context.md
     - examples.md
     - question.md
   ```
   Use this order exactly. Error if a listed file doesn't exist.

2. If no manifest, glob `input/*.md`, exclude `system.md`, sort alphabetically by filename.

### Concatenation

- Read each file's full contents.
- Join with `\n\n---\n\n` as a separator between files.
- Prepend a comment header to each section: `<!-- source: {filename} -->` so the LLM (and humans reviewing output) can see provenance.
- The concatenated result becomes the single user message content.

### System Prompt

- If `input/system.md` exists, its full contents become the system prompt.
- If absent, no system prompt is sent (or an empty string, depending on provider requirements).

---

## Provider Interface (`providers/base.py`)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator

@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str  # "end_turn", "max_tokens", "error"
    latency_seconds: float
    provider: str

class BaseProvider(ABC):
    @abstractmethod
    def validate(self) -> bool:
        """Check if credentials and config are valid. Called before run."""
        ...

    @abstractmethod
    def stream(
        self,
        user_message: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
    ) -> Generator[str, None, LLMResponse]:
        """
        Yield text chunks as they arrive.
        Return the final LLMResponse with full metadata when the generator completes.
        The generator's return value is captured via StopIteration.value.
        """
        ...
```

### Anthropic Provider (`providers/anthropic.py`)

- Use the `anthropic` Python SDK (`pip install anthropic`).
- Instantiate with the API key from credentials.
- Call `client.messages.stream(...)` or `client.messages.create(..., stream=True)`.
- Model: from config (default `claude-sonnet-4-6-20250514`).
- Map `stop_reason` from Anthropic's response.
- Extract `usage.input_tokens` and `usage.output_tokens` from the final message.

### OpenRouter Provider (`providers/openrouter.py`)

- Use `httpx` (async not required; sync streaming is fine).
- Endpoint: `https://openrouter.ai/api/v1/chat/completions`
- Headers: `Authorization: Bearer {api_key}`, `Content-Type: application/json`, plus recommended `HTTP-Referer` and `X-Title` headers.
- Send as OpenAI-compatible chat completion with `stream: true`.
- Parse SSE stream, yield content deltas.
- Extract token usage from the final `[DONE]` chunk or the `usage` field if present (OpenRouter sometimes returns it in a final chunk; if not, estimate from text length and log a warning).

### Bedrock Provider (`providers/bedrock.py`)

- Use `boto3` with the Converse API (`client.converse_stream()`).
- Region: `us-east-1` (hardcoded).
- Model ID from config (default `us.meta.llama4-scout-17b-16e-instruct-v1:0`).
- The Converse API normalizes the request/response format across models.
- Parse the streaming response event stream, yield text chunks from `contentBlockDelta` events.
- Extract token usage from the `metadata` event at the end of the stream.
- No explicit credentials passed to boto3 — it picks up from environment.

---

## Job Runner (`runner.py`)

### Execution Flow

```
run_job(job_id) -> None:

1.  VALIDATE
    - Confirm jobs/{job_id}/ exists
    - Confirm input/ has at least one .md file (besides system.md)
    - Create output/ if it doesn't exist
    - Write status.json: {"status": "pending", "started_at": null}

2.  LOAD CONFIG
    - Merge global defaults with per-job config.yaml
    - Filter provider list to those with valid credentials

3.  ASSEMBLE INPUT
    - Load system prompt from input/system.md (if exists)
    - Concatenate input markdown files per ordering rules
    - Log total input character count

4.  EXECUTE (with fallback)
    - Write status.json: {"status": "running", "started_at": ISO8601}
    - For each provider in the ordered list:
        - For attempt 1..max_attempts_per_provider:
            - Open output/response.md for writing (truncate)
            - Call provider.stream(...)
            - Write chunks to output/response.md as they arrive
            - On success:
                - Finalize output/response.md
                - Write output/metadata.json (see schema below)
                - Write status.json: {"status": "completed", ...}
                - Return
            - On retryable error (5xx, rate limit, timeout, connection error):
                - Log warning with attempt number
                - Sleep with exponential backoff
                - Continue to next attempt
            - On non-retryable error (4xx auth, 4xx bad request, provider-specific):
                - Log error
                - Break to next provider
    - If all providers exhausted:
        - Write status.json: {"status": "failed", "error": "...", ...}
        - Write partial output/response.md if any content was received
        - Exit with code 1

5.  CLEANUP
    - Log summary: provider used, tokens, latency, status
```

### Error Classification

| Error Type | Action |
|---|---|
| HTTP 429 (rate limit) | Retry with backoff (respect `Retry-After` header if present) |
| HTTP 500, 502, 503, 529 | Retry with backoff |
| Connection timeout / reset | Retry with backoff |
| HTTP 401, 403 | Skip to next provider immediately |
| HTTP 400 (bad request) | Skip to next provider (likely model/format issue) |
| HTTP 413 (payload too large) | Skip to next provider (context window exceeded) |
| Stream interrupted mid-response | Retain partial output, retry from scratch (not resumable) |

### Retry Backoff

```
sleep_seconds = min(
    initial_backoff * (multiplier ** (attempt - 1)),
    max_backoff
) + random.uniform(0, 1)  # jitter
```

---

## Output Schemas

### `status.json`

```json
{
  "status": "completed",
  "started_at": "2026-02-23T10:00:00Z",
  "completed_at": "2026-02-23T10:01:23Z",
  "provider": "anthropic",
  "model": "claude-sonnet-4-6-20250514",
  "attempts": [
    {
      "provider": "anthropic",
      "attempt": 1,
      "result": "success",
      "latency_seconds": 83.2
    }
  ],
  "error": null
}
```

For failed jobs:

```json
{
  "status": "failed",
  "started_at": "2026-02-23T10:00:00Z",
  "completed_at": "2026-02-23T10:02:45Z",
  "provider": null,
  "model": null,
  "attempts": [
    {"provider": "anthropic", "attempt": 1, "result": "error", "error": "401 Unauthorized"},
    {"provider": "openrouter", "attempt": 1, "result": "error", "error": "503 Service Unavailable"},
    {"provider": "openrouter", "attempt": 2, "result": "error", "error": "503 Service Unavailable"},
    {"provider": "openrouter", "attempt": 3, "result": "error", "error": "503 Service Unavailable"},
    {"provider": "bedrock", "attempt": 1, "result": "error", "error": "ThrottlingException"}
  ],
  "error": "All providers exhausted"
}
```

### `output/metadata.json`

```json
{
  "provider": "anthropic",
  "model": "claude-sonnet-4-6-20250514",
  "input_tokens": 245000,
  "output_tokens": 4821,
  "stop_reason": "end_turn",
  "latency_seconds": 83.2,
  "input_files": ["context.md", "examples.md", "question.md"],
  "input_characters": 890432,
  "system_prompt_used": true,
  "temperature": 1.0,
  "max_tokens": 16384,
  "timestamp": "2026-02-23T10:01:23Z"
}
```

---

## CLI Interface (`cli.py`)

Use **Typer** for the CLI framework.

### Commands

```bash
# Scaffold a new job directory
big-llm-task new <job_id>
  Creates: jobs/{job_id}/input/  (empty, with a placeholder README)
  Creates: jobs/{job_id}/status.json  (status: "draft")
  Optional: --from <other_job_id>  (copies input/ and config.yaml from another job)

# Run a job
big-llm-task run <job_id>
  Executes the full runner flow.
  Flags:
    --provider <name>       Override: use only this provider (skip fallback)
    --dry-run               Assemble input + validate config, print summary, don't call LLM
    --verbose / -v          Detailed logging to stderr

# Check job status
big-llm-task status <job_id>
  Pretty-prints status.json with color.
  If running, show elapsed time.

# List all jobs
big-llm-task list
  Table of all job dirs with status, timestamp, model used, token counts.
  Flags:
    --status <filter>       Filter by status (draft/pending/running/completed/failed)

# Retry a failed job
big-llm-task retry <job_id>
  Clears status.json and output/, then runs again.
  Equivalent to: reset + run.
```

### Entry Point

In `pyproject.toml`:

```toml
[project.scripts]
big-llm-task = "big_llm_task.cli:app"
```

---

## Dependencies

```toml
[project]
name = "big-llm-task"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.40",
    "boto3>=1.35",
    "httpx>=0.27",
    "typer>=0.12",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "rich>=13.0",
]
```

No async required. All providers use synchronous streaming. `rich` is for CLI output formatting (tables, colored status, progress).

---

## Logging

- Use Python's `logging` module, not print statements.
- Default level: `INFO` (to stderr).
- `--verbose` sets level to `DEBUG`.
- Log format: `%(asctime)s %(levelname)s %(name)s: %(message)s`
- Key log events:
  - Job start, config loaded, provider selected
  - Input assembly: file count, total chars, system prompt presence
  - Provider attempt: which provider, which attempt number
  - Stream progress: log every ~10 seconds with chunk count / chars received so far
  - Provider result: success/failure, latency, tokens
  - Fallback: why previous provider failed, which is next
  - Job complete: final summary line

---

## Edge Cases & Design Decisions

1. **Empty input directory**: Error immediately. A job with no input is invalid.

2. **Only system.md in input/**: Error. There must be at least one non-system .md file. The system prompt alone is not a query.

3. **Very large inputs**: Don't load everything into memory at once for concatenation — but in practice, even 1M tokens of text is ~4MB, which is fine in memory. No need for streaming file reads.

4. **Partial output on failure**: If streaming was in progress and the connection drops, keep whatever was written to `response.md`. The `status.json` will show `failed` and `metadata.json` won't exist, so it's clear the output is partial. Don't delete partial output — it may still be useful.

5. **Concurrent runs of the same job**: Don't handle this. If `status.json` shows `running`, the `run` command should warn and exit (unless `--force` is passed). Simple file-based locking via status check, no flock needed.

6. **Non-.md files in input/**: Ignore them silently. Only `*.md` files are concatenated. This lets users drop reference files (images, PDFs) in the dir without breaking things.

7. **Config validation**: Use Pydantic models to validate both global defaults and per-job config. Fail early with clear error messages if config is malformed.

8. **Model name drift**: Model strings will change over time. The defaults in `config.py` are just defaults — users should be able to override via `config.yaml` without touching code. Consider also supporting a `~/.big-llm-task/defaults.yaml` for user-level default overrides (stretch goal, not MVP).

---

## Implementation Order (suggested for Claude Code)

1. `models.py` — Pydantic models for config, status, metadata, LLMResponse
2. `credentials.py` — Frontmatter parser + credential loader
3. `config.py` — Default config + merge logic
4. `concatenator.py` — Input file assembly
5. `providers/base.py` — Abstract interface
6. `providers/anthropic.py` — Primary provider
7. `providers/openrouter.py` — First fallback
8. `providers/bedrock.py` — Second fallback
9. `runner.py` — Core orchestration
10. `cli.py` — Typer commands
11. `pyproject.toml` — Packaging + entry point
12. Integration test with a real job directory (manual)

Each step is independently testable. The provider implementations are the most complex pieces; the rest is plumbing.

