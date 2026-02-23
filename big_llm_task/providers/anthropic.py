"""Anthropic Messages API provider."""

from __future__ import annotations

import logging
import time
from typing import Generator

import anthropic

from ..models import LLMResponse
from .base import BaseProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self._client: anthropic.Anthropic | None = None

    def validate(self) -> bool:
        try:
            self._client = anthropic.Anthropic(api_key=self.api_key)
            return True
        except Exception as exc:
            logger.warning("Anthropic validation failed: %s", exc)
            return False

    def stream(
        self,
        user_message: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
    ) -> Generator[str, None, LLMResponse]:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": user_message}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        start = time.monotonic()
        collected_text: list[str] = []

        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                collected_text.append(text)
                yield text

            # Get the final message with metadata
            response = stream.get_final_message()

        latency = time.monotonic() - start

        # Map stop reason
        stop_reason_map = {
            "end_turn": "end_turn",
            "max_tokens": "max_tokens",
            "stop_sequence": "end_turn",
        }
        stop_reason = stop_reason_map.get(response.stop_reason, response.stop_reason)

        return LLMResponse(
            text="".join(collected_text),
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=stop_reason,
            latency_seconds=latency,
            provider="anthropic",
        )
