"""Dual-backend LLM client for the Analyst agent."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod

from src.analyst.prompts import (
    SYSTEM_PROMPT,
    format_few_shot_messages,
    format_user_prompt,
)
from src.analyst.schema import TradeSignal

logger = logging.getLogger(__name__)

# Neutral fallback signal used when all retries are exhausted
FALLBACK_SIGNAL = TradeSignal(
    reasoning="API error — returning neutral fallback signal.",
    decision="hold",
)


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def call(self, system_prompt: str, messages: list[dict]) -> str:
        """Send messages to the LLM and return the raw string response."""
        ...


class OllamaBackend(LLMBackend):
    """Local Llama 8B via ollama Python library."""

    def __init__(self, model: str = "llama3.1:8b", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def call(self, system_prompt: str, messages: list[dict]) -> str:
        import ollama

        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)

        response = ollama.chat(
            model=self.model,
            messages=full_messages,
            options={"temperature": self.temperature},
            format="json",  # ollama JSON mode
        )
        return response["message"]["content"]


class ClaudeBackend(LLMBackend):
    """Anthropic Claude API with prompt caching."""

    def __init__(
        self, model: str = "claude-sonnet-4-20250514", temperature: float = 0.0
    ):
        import anthropic

        self.client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
        self.model = model
        self.temperature = temperature

    def call(self, system_prompt: str, messages: list[dict]) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},  # prompt caching
                }
            ],
            messages=messages,
        )
        return response.content[0].text


class AzureOpenAIBackend(LLMBackend):
    """Azure OpenAI GPT-5 backend."""

    DEFAULT_ENDPOINT = (
        "https://sas-hackathon-sep.openai.azure.com/openai/deployments/"
        "gpt-5-chat/chat/completions"
    )
    DEFAULT_API_VERSION = "2024-12-01-preview"

    def __init__(
        self,
        model: str = "gpt-5-chat",
        temperature: float = 0.0,
        api_key: str | None = None,
        endpoint: str | None = None,
        api_version: str | None = None,
    ):
        import os

        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "AZURE_OPENAI_API_KEY must be set in .env or passed explicitly."
            )
        self.endpoint = endpoint or os.environ.get(
            "AZURE_OPENAI_ENDPOINT", self.DEFAULT_ENDPOINT
        )
        self.api_version = api_version or os.environ.get(
            "AZURE_OPENAI_API_VERSION", self.DEFAULT_API_VERSION
        )

    def call(self, system_prompt: str, messages: list[dict]) -> str:
        import requests

        url = f"{self.endpoint}?api-version={self.api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": full_messages,
            "temperature": self.temperature,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


class AnalystClient:
    """Main Analyst client. Wraps an LLM backend and returns validated
    TradeSignal objects.

    Usage:
        client = AnalystClient(backend=OllamaBackend())
        signal = client.analyze("AAPL beats earnings", "AAPL", "2024-01-15")
    """

    def __init__(
        self,
        backend: LLMBackend,
        max_retries: int = 3,
        include_few_shot: bool = True,
    ):
        self.backend = backend
        self.max_retries = max_retries
        self.include_few_shot = include_few_shot

    def analyze(self, headline: str, ticker: str, date: str) -> TradeSignal:
        """Analyze a single headline and return a validated TradeSignal.

        Retries up to max_retries times on API or parsing failures.
        Returns FALLBACK_SIGNAL (hold) after all retries exhausted.
        """
        messages: list[dict] = []
        if self.include_few_shot:
            messages.extend(format_few_shot_messages())
        messages.append(
            {"role": "user", "content": format_user_prompt(headline, ticker, date)}
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                raw_response = self.backend.call(SYSTEM_PROMPT, messages)
                parsed = json.loads(raw_response)
                signal = TradeSignal(**parsed)
                return signal
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Attempt {attempt}/{self.max_retries}: JSON parse error: {e}"
                )
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt}/{self.max_retries}: Error: {e}"
                )

            if attempt < self.max_retries:
                time.sleep(1.0 * attempt)  # linear backoff

        logger.error(
            f"All {self.max_retries} retries exhausted for headline: "
            f"{headline[:80]}..."
        )
        return FALLBACK_SIGNAL
