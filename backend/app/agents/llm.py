"""LLM access behind a small interface so providers (and test fakes) can be swapped.

Pick the provider with LLM_PROVIDER: "gemini" (default) or "anthropic".
"""

import inspect
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import anthropic
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"

DEFAULT_MODELS = {
    "gemini": "gemini-3.8-flash",
    "anthropic": "claude-opus-5-5",
}


class LLMError(Exception):
    """Generation failed; the message is safe to show the user."""


@dataclass
class Usage:
    """Billable tokens of one call, filled in by `stream_answer` when passed in.

    `output_tokens` includes thinking tokens, which providers bill as output.
    `cached_input_tokens` is the part of `input_tokens` served from the prompt cache.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0


class LLM(Protocol):
    def stream_answer(
        self, system: str, messages: list[dict], usage: Usage | None = None
    ) -> AsyncIterator[str]:
        """Yield answer text as it is generated. If `usage` is given, fill it in at the end."""
        ...


class AnthropicLLM:
    def __init__(
        self,
        client: Any,
        model: str,
        max_tokens: int = 16000,
        effort: str = "",
        fallbacks: bool = True,
    ):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.fallbacks = fallbacks

    def _request(self, system: str, messages: list[dict]) -> dict:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            # The system prompt never changes, so it caches across requests.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
            "thinking": {"type": "adaptive"},
        }
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        if self.fallbacks:
            # If the model declines (safety classifier), the API re-runs the
            # request on a fallback model inside the same call.
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    async def stream_answer(
        self, system: str, messages: list[dict], usage: Usage | None = None
    ) -> AsyncIterator[str]:
        try:
            async with self.client.beta.messages.stream(**self._request(system, messages)) as s:
                async for text in s.text_stream:
                    yield text
                final = await s.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise LLMError("The LLM API key is missing or invalid.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("The LLM is rate limited. Please try again shortly.") from exc
        except anthropic.APIStatusError as exc:
            logger.error("LLM API error %s: %s", exc.status_code, exc.message)
            raise LLMError("The LLM request failed.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the LLM API.") from exc
        except TypeError as exc:
            # The SDK raises TypeError("Could not resolve authentication method...")
            # when no API key or other credential is configured at all.
            if "authentication" in str(exc):
                raise LLMError("The LLM API key is missing or invalid.") from exc
            raise

        if usage is not None and (u := getattr(final, "usage", None)) is not None:
            cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
            cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
            usage.input_tokens = (u.input_tokens or 0) + cache_write + cache_read
            usage.cached_input_tokens = cache_read
            usage.output_tokens = u.output_tokens or 0
        if final.stop_reason == "refusal":
            raise LLMError("The model declined to answer this request.")
        if final.stop_reason == "max_tokens":
            logger.warning("Answer truncated at max_tokens=%d", self.max_tokens)


_GEMINI_BLOCKED = {
    genai_types.FinishReason.SAFETY,
    genai_types.FinishReason.BLOCKLIST,
    genai_types.FinishReason.PROHIBITED_CONTENT,
    genai_types.FinishReason.SPII,
    genai_types.FinishReason.RECITATION,
}


class GeminiLLM:
    """Google Gemini via the google-genai SDK."""

    def __init__(
        self,
        model: str,
        api_key: str = "",
        max_tokens: int = 16000,
        thinking_level: str = "",
        client: Any = None,
    ):
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.thinking_level = thinking_level
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                # An empty key lets the SDK read GEMINI_API_KEY / GOOGLE_API_KEY itself.
                self._client = genai.Client(api_key=self.api_key or None)
            except ValueError as exc:  # raised when no API key is configured at all
                raise LLMError("The LLM API key is missing or invalid.") from exc
        return self._client

    def _request(self, system: str, messages: list[dict]) -> dict:
        contents = [
            genai_types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[genai_types.Part(text=m["content"])],
            )
            for m in messages
        ]
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=self.max_tokens,
            thinking_config=(
                genai_types.ThinkingConfig(thinking_level=self.thinking_level)
                if self.thinking_level
                else None
            ),
        )
        return {"model": self.model, "contents": contents, "config": config}

    async def stream_answer(
        self, system: str, messages: list[dict], usage: Usage | None = None
    ) -> AsyncIterator[str]:
        client = self._get_client()
        try:
            stream = client.aio.models.generate_content_stream(**self._request(system, messages))
            if inspect.isawaitable(stream):
                stream = await stream
            async for chunk in stream:
                # Each chunk carries running totals; the last one is final.
                if usage is not None and (meta := getattr(chunk, "usage_metadata", None)):
                    usage.input_tokens = meta.prompt_token_count or 0
                    usage.cached_input_tokens = meta.cached_content_token_count or 0
                    usage.output_tokens = (meta.candidates_token_count or 0) + (
                        meta.thoughts_token_count or 0
                    )
                feedback = getattr(chunk, "prompt_feedback", None)
                if feedback is not None and feedback.block_reason:
                    raise LLMError("The model declined to answer this request.")
                if chunk.text:
                    yield chunk.text
                finish = chunk.candidates[0].finish_reason if chunk.candidates else None
                if finish in _GEMINI_BLOCKED:
                    raise LLMError("The model declined to answer this request.")
                if finish == genai_types.FinishReason.MAX_TOKENS:
                    logger.warning("Answer truncated at max_output_tokens=%d", self.max_tokens)
        except genai_errors.ClientError as exc:
            if exc.code in (401, 403) or "API key" in str(exc):
                raise LLMError("The LLM API key is missing or invalid.") from exc
            if exc.code == 429:
                raise LLMError("The LLM is rate limited. Please try again shortly.") from exc
            logger.error("Gemini API error %s: %s", exc.code, exc)
            raise LLMError("The LLM request failed.") from exc
        except genai_errors.ServerError as exc:
            logger.error("Gemini server error %s: %s", exc.code, exc)
            raise LLMError("The LLM request failed.") from exc


def create_llm(s: Settings) -> LLM:
    provider = s.llm_provider.lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"Unknown LLM_PROVIDER: {s.llm_provider!r}")
    model = s.llm_model or DEFAULT_MODELS[provider]
    if provider == "gemini":
        return GeminiLLM(
            model=model,
            api_key=s.gemini_api_key,
            max_tokens=s.llm_max_tokens,
            thinking_level=s.llm_effort,
        )
    # An empty key lets the SDK fall back to its other credential sources.
    client = anthropic.AsyncAnthropic(api_key=s.anthropic_api_key or None)
    return AnthropicLLM(
        client,
        model=model,
        max_tokens=s.llm_max_tokens,
        effort=s.llm_effort,
        fallbacks=s.llm_fallbacks,
    )


@lru_cache
def get_llm() -> LLM:
    return create_llm(get_settings())
