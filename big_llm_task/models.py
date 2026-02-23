"""Pydantic models for config, status, metadata, and LLM responses."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --- LLM Response ---


class LLMResponse(BaseModel):
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str  # "end_turn", "max_tokens", "error"
    latency_seconds: float
    provider: str


# --- Configuration ---


class ProviderConfig(BaseModel):
    name: str
    model: str
    max_tokens: int = 16384


class RetryConfig(BaseModel):
    max_attempts_per_provider: int = 3
    initial_backoff_seconds: float = 2
    backoff_multiplier: float = 2
    max_backoff_seconds: float = 30


class JobConfig(BaseModel):
    providers: list[ProviderConfig] = Field(default_factory=list)
    temperature: float = 1.0
    retry: RetryConfig = Field(default_factory=RetryConfig)
    stream_to_disk: bool = True


# --- Status ---


class JobStatus(str, Enum):
    DRAFT = "draft"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AttemptRecord(BaseModel):
    provider: str
    attempt: int
    result: str  # "success" or "error"
    latency_seconds: Optional[float] = None
    error: Optional[str] = None


class StatusFile(BaseModel):
    status: JobStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    attempts: list[AttemptRecord] = Field(default_factory=list)
    error: Optional[str] = None


# --- Metadata ---


class MetadataFile(BaseModel):
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    latency_seconds: float
    input_files: list[str]
    input_characters: int
    system_prompt_used: bool
    temperature: float
    max_tokens: int
    timestamp: datetime
