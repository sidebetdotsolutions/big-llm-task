"""AWS Bedrock Converse API provider."""

from __future__ import annotations

import logging
import time
from typing import Generator

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from ..models import LLMResponse
from .base import BaseProvider

logger = logging.getLogger(__name__)

BEDROCK_REGION = "us-east-1"


class BedrockProvider(BaseProvider):
    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None

    def validate(self) -> bool:
        try:
            self._client = boto3.client(
                "bedrock-runtime", region_name=BEDROCK_REGION
            )
            return True
        except (BotoCoreError, ClientError) as exc:
            logger.warning("Bedrock validation failed: %s", exc)
            return False

    def stream(
        self,
        user_message: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
    ) -> Generator[str, None, LLMResponse]:
        if self._client is None:
            self._client = boto3.client(
                "bedrock-runtime", region_name=BEDROCK_REGION
            )

        messages = [
            {
                "role": "user",
                "content": [{"text": user_message}],
            }
        ]

        kwargs: dict = {
            "modelId": self.model,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }

        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]

        start = time.monotonic()
        collected_text: list[str] = []
        input_tokens = 0
        output_tokens = 0
        stop_reason = "end_turn"

        response = self._client.converse_stream(**kwargs)

        for event in response["stream"]:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                text = delta.get("text", "")
                if text:
                    collected_text.append(text)
                    yield text

            elif "metadata" in event:
                metadata = event["metadata"]
                usage = metadata.get("usage", {})
                input_tokens = usage.get("inputTokens", 0)
                output_tokens = usage.get("outputTokens", 0)

            elif "messageStop" in event:
                reason = event["messageStop"].get("stopReason", "end_turn")
                if reason == "max_tokens":
                    stop_reason = "max_tokens"
                else:
                    stop_reason = "end_turn"

        latency = time.monotonic() - start

        return LLMResponse(
            text="".join(collected_text),
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
            latency_seconds=latency,
            provider="bedrock",
        )
