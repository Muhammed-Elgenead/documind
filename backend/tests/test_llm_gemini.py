"""GeminiLLM and provider selection against a fake google-genai client (no network)."""

from types import SimpleNamespace

import pytest
from google.genai import errors, types

from app.agents.llm import AnthropicLLM, GeminiLLM, LLMError, Usage, create_llm
from app.core.config import Settings

STOP = types.FinishReason.STOP


def chunk(text, finish=None, block_reason=None, usage=None):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=finish)],
        prompt_feedback=SimpleNamespace(block_reason=block_reason) if block_reason else None,
        usage_metadata=usage,
    )


def meta(prompt, candidates, thoughts=None, cached=None):
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        thoughts_token_count=thoughts,
        cached_content_token_count=cached,
    )


class FakeGenaiClient:
    """Mimics client.aio.models.generate_content_stream (an awaitable -> async iterator)."""

    def __init__(self, chunks=(), error=None):
        self.calls: list[dict] = []
        outer = self

        class _Models:
            async def generate_content_stream(self, **kwargs):
                outer.calls.append(kwargs)
                if error:
                    raise error

                async def gen():
                    for c in chunks:
                        yield c

                return gen()

        self.aio = SimpleNamespace(models=_Models())


async def collect(llm, messages=None):
    msgs = messages or [{"role": "user", "content": "q"}]
    return [t async for t in llm.stream_answer("sys prompt", msgs)]


async def test_streams_text_and_builds_request():
    client = FakeGenaiClient([chunk("Hel"), chunk("lo [1]."), chunk("", STOP)])
    llm = GeminiLLM("gemini-3.8-flash", max_tokens=999, client=client)
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    assert await collect(llm, history) == ["Hel", "lo [1]."]

    (req,) = client.calls
    assert req["model"] == "gemini-3.8-flash"
    assert [c.role for c in req["contents"]] == ["user", "model", "user"]
    assert [c.parts[0].text for c in req["contents"]] == ["q1", "a1", "q2"]
    assert req["config"].system_instruction == "sys prompt"
    assert req["config"].max_output_tokens == 999
    assert req["config"].thinking_config is None


async def test_thinking_level_passed_through():
    client = FakeGenaiClient([chunk("x", STOP)])
    await collect(GeminiLLM("m", thinking_level="low", client=client))
    assert client.calls[0]["config"].thinking_config.thinking_level == types.ThinkingLevel.LOW


@pytest.mark.parametrize(
    "bad",
    [chunk("", types.FinishReason.SAFETY), chunk("", block_reason=types.BlockedReason.SAFETY)],
)
async def test_safety_block_raises(bad):
    llm = GeminiLLM("m", client=FakeGenaiClient([chunk("partial "), bad]))
    with pytest.raises(LLMError, match="declined"):
        await collect(llm)


def _client_error(code, message):
    return errors.ClientError(code, {"error": {"code": code, "message": message}})


@pytest.mark.parametrize(
    ("error", "match"),
    [
        (_client_error(400, "API key not valid. Please pass a valid API key."), "API key"),
        (_client_error(403, "Permission denied"), "API key"),
        (_client_error(429, "Resource exhausted"), "rate limited"),
        (_client_error(400, "Invalid argument"), "request failed"),
        (errors.ServerError(503, {"error": {"code": 503, "message": "Overloaded"}}), "failed"),
    ],
)
async def test_api_errors_become_llm_errors(error, match):
    llm = GeminiLLM("m", client=FakeGenaiClient(error=error))
    with pytest.raises(LLMError, match=match):
        await collect(llm)


async def test_missing_key_becomes_llm_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(LLMError, match="API key"):
        await collect(GeminiLLM("m", api_key=""))


def test_create_llm_selects_provider_and_default_model():
    gem = create_llm(Settings(llm_provider="gemini", llm_model=""))
    assert isinstance(gem, GeminiLLM) and gem.model == "gemini-3.8-flash"
    custom = create_llm(Settings(llm_provider="GEMINI", llm_model="gemini-3.6-flash"))
    assert custom.model == "gemini-3.6-flash"
    claude = create_llm(Settings(llm_provider="anthropic", llm_model="", anthropic_api_key="k"))
    assert isinstance(claude, AnthropicLLM) and claude.model == "claude-opus-5-5"
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        create_llm(Settings(llm_provider="openai"))


async def test_usage_uses_last_totals_and_counts_thinking_as_output():
    client = FakeGenaiClient(
        [chunk("a", usage=meta(900, 3)), chunk("b"), chunk("", STOP, usage=meta(900, 12, 30, 600))]
    )
    usage = Usage()
    llm = GeminiLLM("m", client=client)
    _ = [t async for t in llm.stream_answer("s", [{"role": "user", "content": "q"}], usage)]
    assert usage == Usage(input_tokens=900, output_tokens=42, cached_input_tokens=600)
