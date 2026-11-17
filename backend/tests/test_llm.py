"""AnthropicLLM against a fake SDK client (no network)."""

from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from app.agents.llm import FALLBACK_BETA, AnthropicLLM, LLMError, Usage


class FakeStream:
    def __init__(self, chunks, stop_reason="end_turn", error=None, usage=None):
        self._chunks = chunks
        self._stop_reason = stop_reason
        self._error = error
        self._usage = usage

    async def __aenter__(self):
        if self._error:
            raise self._error
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    async def text_stream(self):
        for c in self._chunks:
            yield c

    async def get_final_message(self):
        return SimpleNamespace(stop_reason=self._stop_reason, usage=self._usage)


class FakeClient:
    def __init__(self, stream: FakeStream):
        self.calls: list[dict] = []
        outer = self

        class _Messages:
            def stream(self, **kwargs):
                outer.calls.append(kwargs)
                return stream

        self.beta = SimpleNamespace(messages=_Messages())


async def _collect(llm, system="sys", messages=None):
    return [
        t async for t in llm.stream_answer(system, messages or [{"role": "user", "content": "q"}])
    ]


async def test_streams_text_and_builds_request():
    client = FakeClient(FakeStream(["Hel", "lo [1]."]))
    llm = AnthropicLLM(client, model="claude-opus-5-5", max_tokens=1234, effort="low")
    assert await _collect(llm) == ["Hel", "lo [1]."]

    (req,) = client.calls
    assert req["model"] == "claude-opus-5-5"
    assert req["max_tokens"] == 1234
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"] == {"effort": "low"}
    assert req["system"][0]["text"] == "sys"
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["betas"] == [FALLBACK_BETA]
    assert req["fallbacks"] == "default"
    assert "temperature" not in req


async def test_optional_params_omitted():
    client = FakeClient(FakeStream(["x"]))
    await _collect(AnthropicLLM(client, model="m", effort="", fallbacks=False))
    (req,) = client.calls
    assert "output_config" not in req
    assert "betas" not in req and "fallbacks" not in req


async def test_refusal_raises():
    llm = AnthropicLLM(FakeClient(FakeStream([], stop_reason="refusal")), model="m")
    with pytest.raises(LLMError, match="declined"):
        await _collect(llm)


def _status_error(cls, code):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("err", response=httpx2.Response(code, request=req), body=None)


@pytest.mark.parametrize(
    ("error", "match"),
    [
        (_status_error(anthropic.AuthenticationError, 401), "API key"),
        (_status_error(anthropic.RateLimitError, 429), "rate limited"),
        (_status_error(anthropic.InternalServerError, 500), "request failed"),
    ],
)
async def test_api_errors_become_llm_errors(error, match):
    llm = AnthropicLLM(FakeClient(FakeStream([], error=error)), model="m")
    with pytest.raises(LLMError, match=match):
        await _collect(llm)


async def test_missing_credentials_become_llm_error():
    err = TypeError("Could not resolve authentication method. Expected one of api_key, ...")
    llm = AnthropicLLM(FakeClient(FakeStream([], error=err)), model="m")
    with pytest.raises(LLMError, match="API key"):
        await _collect(llm)


async def test_unrelated_type_errors_propagate():
    llm = AnthropicLLM(FakeClient(FakeStream([], error=TypeError("bug"))), model="m")
    with pytest.raises(TypeError, match="bug"):
        await _collect(llm)


async def test_usage_is_reported_including_cache_tokens():
    u = SimpleNamespace(
        input_tokens=100,
        cache_creation_input_tokens=20,
        cache_read_input_tokens=500,
        output_tokens=42,
    )
    llm = AnthropicLLM(FakeClient(FakeStream(["a"], usage=u)), model="m")
    usage = Usage()
    _ = [t async for t in llm.stream_answer("s", [{"role": "user", "content": "q"}], usage)]
    assert usage == Usage(input_tokens=620, output_tokens=42, cached_input_tokens=500)
