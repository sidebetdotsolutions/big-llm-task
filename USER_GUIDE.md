# big-llm-task User Guide

## Dependencies

```bash
pip install anthropic boto3 httpx typer pyyaml pydantic rich
```

## Credentials

Place credential files in `~/empire/vault/credentials/`:

```
anthropic/credentials.md    # --- api_key: "sk-ant-..." ---
openrouter/credentials.md   # --- api_key: "sk-or-..." ---
aws/credentials.md          # --- (any frontmatter; auth via ~/.aws) ---
```

Missing or empty credentials skip that provider automatically.

## Quick Start

Run from the project root:

```bash
# Create a job
python -m big_llm_task new my-job

# Add your prompt
echo "Summarize the key themes in this text." > jobs/my-job/input/question.md

# Run it
python -m big_llm_task run my-job
```

Output lands in `jobs/my-job/output/response.md`.

## Commands

| Command | Description |
|---|---|
| `python -m big_llm_task new <job_id>` | Scaffold a new job directory |
| `python -m big_llm_task run <job_id>` | Execute a job |
| `python -m big_llm_task status <job_id>` | Show job status |
| `python -m big_llm_task list` | List all jobs |
| `python -m big_llm_task retry <job_id>` | Clear output and re-run a failed job |

### Run Flags

- `--provider <name>` — Use only this provider (anthropic, openrouter, bedrock)
- `--dry-run` — Validate and print summary without calling the LLM
- `--force` — Run even if job status shows "running"
- `-v / --verbose` — Detailed logging

### Other Flags

- `python -m big_llm_task new <id> --from <other_id>` — Copy input and config from another job
- `python -m big_llm_task list --status <filter>` — Filter by status (draft/pending/running/completed/failed)

## Job Directory Structure

```
jobs/my-job/
  input/
    system.md         # Optional system prompt
    *.md              # User message files (concatenated alphabetically)
    manifest.yaml     # Optional explicit file ordering
  config.yaml         # Optional per-job config overrides
  output/
    response.md       # LLM response
    metadata.json     # Tokens, latency, model info
  status.json         # Job lifecycle state
```

## Input File Ordering

Files are concatenated alphabetically by default. To control order, create `input/manifest.yaml`:

```yaml
files:
  - context.md
  - examples.md
  - question.md
```

## Per-Job Config

Override defaults by adding `config.yaml` to the job directory:

```yaml
temperature: 0.3
providers:
  - name: anthropic
    model: claude-sonnet-4-6
    max_tokens: 32768
```

## Provider Fallback

By default, providers are tried in order: Anthropic → OpenRouter → Bedrock. Each gets up to 3 retry attempts on transient errors before moving to the next.
