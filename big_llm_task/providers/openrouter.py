"""OpenRouter provider (OpenAI-compatible chat completions)."""

from __future__ import annotations

import json
import logging
import time
from typing import Generator

import httpx

from ..models import LLMResponse
from .base import BaseProvider

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterProvider(BaseProvider):
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def validate(self) -> bool:
        return bool(self.api_key)

    def stream(
        self,
        user_message: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
    ) -> Generator[str, None, LLMResponse]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/sidebetsolutions/big-llm-task",
            "X-Title": "big-llm-task",
        }

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        start = time.monotonic()
        collected_text: list[str] = []
        input_tokens = 0
        output_tokens = 0
        stop_reason = "end_turn"
        model_used = self.model

        with httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
            with client.stream("POST", OPENROUTER_URL, headers=headers, json=payload) as resp:
                resp.raise_for_status()

                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Extract model name if present
                    if "model" in chunk:
                        model_used = chunk["model"]

                    # Extract token usage if present
                    if "usage" in chunk:
                        usage = chunk["usage"]
                        input_tokens = usage.get("prompt_tokens", input_tokens)
                        output_tokens = usage.get("completion_tokens", output_tokens)

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    if content:
                        collected_text.append(content)
                        yield content

                    # Check finish reason
                    finish = choice.get("finish_reason")
                    if finish:
                        if finish == "length":
                            stop_reason = "max_tokens"
                        else:
                            stop_reason = "end_turn"

        latency = time.monotonic() - start
        full_text = "".join(collected_text)

        # Estimate tokens if not provided
        if input_tokens == 0:
            input_tokens = len(user_message) // 4
            logger.warning(
                "OpenRouter did not return input token count; estimated %d from text length.",
                input_tokens,
            )
        if output_tokens == 0:
            output_tokens = len(full_text) // 4
            logger.warning(
                "OpenRouter did not return output token count; estimated %d from text length.",
                output_tokens,
            )

        return LLMResponse(
            text=full_text,
            model=model_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
            latency_seconds=latency,
            provider="openrouter",
        )
