"""Typer CLI entrypoint for big-llm-task."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .models import JobStatus, StatusFile
from .runner import JOBS_ROOT, run_job

app = typer.Typer(
    name="big-llm-task",
    help="Execute large-context LLM queries using a job-directory convention.",
    add_completion=False,
)

console = Console(stderr=True)

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format=LOG_FORMAT, level=level, force=True)


def _read_status(job_dir: Path) -> StatusFile | None:
    status_path = job_dir / "status.json"
    if not status_path.is_file():
        return None
    try:
        return StatusFile.model_validate_json(
            status_path.read_text(encoding="utf-8")
        )
    except (ValueError, json.JSONDecodeError):
        return None


def _read_metadata(job_dir: Path) -> dict | None:
    meta_path = job_dir / "output" / "metadata.json"
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (ValueError, json.JSONDecodeError):
        return None


# --- Commands ---


@app.command()
def new(
    job_id: str = typer.Argument(..., help="Unique job identifier"),
    from_job: Optional[str] = typer.Option(
        None, "--from", help="Copy input/ and config.yaml from another job"
    ),
) -> None:
    """Scaffold a new job directory."""
    job_dir = JOBS_ROOT / job_id

    if job_dir.exists():
        console.print(f"[red]Job '{job_id}' already exists.[/red]")
        raise typer.Exit(1)

    if from_job:
        source_dir = JOBS_ROOT / from_job
        if not source_dir.is_dir():
            console.print(f"[red]Source job '{from_job}' not found.[/red]")
            raise typer.Exit(1)

        job_dir.mkdir(parents=True)
        # Copy input/
        source_input = source_dir / "input"
        if source_input.is_dir():
            shutil.copytree(source_input, job_dir / "input")
        else:
            (job_dir / "input").mkdir()

        # Copy config.yaml
        source_config = source_dir / "config.yaml"
        if source_config.is_file():
            shutil.copy2(source_config, job_dir / "config.yaml")
    else:
        job_dir.mkdir(parents=True)
        input_dir = job_dir / "input"
        input_dir.mkdir()

    # Write initial status
    status = StatusFile(status=JobStatus.DRAFT)
    (job_dir / "status.json").write_text(
        status.model_dump_json(indent=2), encoding="utf-8"
    )

    console.print(f"[green]Created job:[/green] {job_dir}")


@app.command(name="run")
def run_cmd(
    job_id: str = typer.Argument(..., help="Job ID to run"),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Use only this provider (skip fallback)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate and print summary without calling LLM"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Detailed logging"),
    force: bool = typer.Option(
        False, "--force", help="Run even if job is currently marked as running"
    ),
) -> None:
    """Run a job."""
    _setup_logging(verbose)
    run_job(
        job_id=job_id,
        provider_override=provider,
        dry_run=dry_run,
        force=force,
    )


@app.command()
def status(
    job_id: str = typer.Argument(..., help="Job ID to check"),
) -> None:
    """Check job status."""
    job_dir = JOBS_ROOT / job_id
    if not job_dir.is_dir():
        console.print(f"[red]Job '{job_id}' not found.[/red]")
        raise typer.Exit(1)

    st = _read_status(job_dir)
    if st is None:
        console.print(f"[yellow]No status.json found for job '{job_id}'.[/yellow]")
        raise typer.Exit(1)

    color_map = {
        JobStatus.DRAFT: "dim",
        JobStatus.PENDING: "cyan",
        JobStatus.RUNNING: "yellow",
        JobStatus.COMPLETED: "green",
        JobStatus.FAILED: "red",
    }
    color = color_map.get(st.status, "white")

    console.print(f"[bold]Job:[/bold] {job_id}")
    console.print(f"[bold]Status:[/bold] [{color}]{st.status.value}[/{color}]")

    if st.started_at:
        console.print(f"[bold]Started:[/bold] {st.started_at.isoformat()}")

    if st.status == JobStatus.RUNNING and st.started_at:
        elapsed = datetime.now(timezone.utc) - st.started_at
        console.print(f"[bold]Elapsed:[/bold] {elapsed}")

    if st.completed_at:
        console.print(f"[bold]Completed:[/bold] {st.completed_at.isoformat()}")

    if st.provider:
        console.print(f"[bold]Provider:[/bold] {st.provider}")
    if st.model:
        console.print(f"[bold]Model:[/bold] {st.model}")

    if st.error:
        console.print(f"[bold]Error:[/bold] [red]{st.error}[/red]")

    if st.attempts:
        console.print(f"\n[bold]Attempts ({len(st.attempts)}):[/bold]")
        for a in st.attempts:
            result_color = "green" if a.result == "success" else "red"
            line = f"  {a.provider} #{a.attempt}: [{result_color}]{a.result}[/{result_color}]"
            if a.latency_seconds is not None:
                line += f" ({a.latency_seconds:.1f}s)"
            if a.error:
                line += f" — {a.error}"
            console.print(line)


@app.command(name="list")
def list_cmd(
    status_filter: Optional[str] = typer.Option(
        None, "--status", help="Filter by status (draft/pending/running/completed/failed)"
    ),
) -> None:
    """List all jobs."""
    if not JOBS_ROOT.is_dir():
        console.print("[yellow]No jobs directory found.[/yellow]")
        return

    job_dirs = sorted(
        [d for d in JOBS_ROOT.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    if not job_dirs:
        console.print("[yellow]No jobs found.[/yellow]")
        return

    table = Table(title="Jobs")
    table.add_column("Job ID", style="bold")
    table.add_column("Status")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Tokens (in/out)")
    table.add_column("Timestamp")

    color_map = {
        "draft": "dim",
        "pending": "cyan",
        "running": "yellow",
        "completed": "green",
        "failed": "red",
    }

    for job_dir in job_dirs:
        st = _read_status(job_dir)
        if st is None:
            continue

        status_val = st.status.value
        if status_filter and status_val != status_filter:
            continue

        color = color_map.get(status_val, "white")
        meta = _read_metadata(job_dir)

        provider_str = st.provider or "-"
        model_str = st.model or "-"
        tokens_str = "-"
        ts_str = "-"

        if meta:
            tokens_str = f"{meta.get('input_tokens', 0):,}/{meta.get('output_tokens', 0):,}"
            ts_str = meta.get("timestamp", "-")

        if not ts_str or ts_str == "-":
            if st.completed_at:
                ts_str = st.completed_at.isoformat()
            elif st.started_at:
                ts_str = st.started_at.isoformat()

        table.add_row(
            job_dir.name,
            f"[{color}]{status_val}[/{color}]",
            provider_str,
            model_str,
            tokens_str,
            str(ts_str),
        )

    console.print(table)


@app.command()
def retry(
    job_id: str = typer.Argument(..., help="Job ID to retry"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Detailed logging"),
) -> None:
    """Retry a failed job (clears status and output, then runs again)."""
    _setup_logging(verbose)

    job_dir = JOBS_ROOT / job_id
    if not job_dir.is_dir():
        console.print(f"[red]Job '{job_id}' not found.[/red]")
        raise typer.Exit(1)

    # Clear output
    output_dir = job_dir / "output"
    if output_dir.is_dir():
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    # Reset status
    status = StatusFile(status=JobStatus.PENDING)
    (job_dir / "status.json").write_text(
        status.model_dump_json(indent=2), encoding="utf-8"
    )

    console.print(f"[yellow]Retrying job {job_id}...[/yellow]")
    run_job(job_id=job_id)


if __name__ == "__main__":
    app()
