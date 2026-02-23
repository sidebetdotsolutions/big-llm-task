"""Abstract provider interface for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generator

from ..models import LLMResponse


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
        """Yield text chunks as they arrive.

        Return the final LLMResponse with full metadata when the generator
        completes. The generator's return value is captured via
        StopIteration.value.
        """
        ...
