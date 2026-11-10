import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from filing_analyst.config import Settings
from filing_analyst.llm import (
    DailyBudgetExceeded,
    LLMClient,
    LLMError,
    MinuteLimiter,
    backoff_seconds,
)

FAKE_KEY = "test-key-123"


class Risk(BaseModel):
    category: str
    severity: int


class FakeClock:
    """Fake time: sleep() moves the clock forward instantly and remembers each wait."""

    def __init__(self) -> None:
        self.t = 0.0
        self.waits: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.t += seconds


def ok_body(content: str, tokens_in: int = 50, tokens_out: int = 10) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
    }


def make_client(tmp_path: Path, responses: list[httpx.Response], **overrides) -> tuple:
    """Client wired to a fake server that returns `responses` in order."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return responses[len(calls) - 1]

    settings = Settings(
        _env_file=None,
        sec_user_agent="Test test@example.com",
        llm_api_key=FAKE_KEY,
        data_dir=tmp_path,
        **overrides,
    )
    clock = FakeClock()
    client = LLMClient(settings, httpx.MockTransport(handler), clock.sleep, clock.now)
    return client, calls, clock


def log_lines(client: LLMClient) -> list[dict]:
    return [json.loads(line) for line in client.log.path.read_text().splitlines()]


MESSAGES = [{"role": "user", "content": "hi"}]


def test_backoff_doubles_and_respects_retry_after() -> None:
    assert [backoff_seconds(a, None, jitter=0) for a in range(4)] == [1, 2, 4, 8]
    assert backoff_seconds(0, "10", jitter=0) == 10
    assert backoff_seconds(1, None, jitter=0.5) == 2.5


def test_429_is_retried_after_waiting(tmp_path: Path) -> None:
    client, calls, clock = make_client(
        tmp_path,
        [
            httpx.Response(429, headers={"retry-after": "3"}),
            httpx.Response(200, json=ok_body("hello")),
        ],
    )
    assert client.complete(MESSAGES, task="t") == "hello"
    assert len(calls) == 2
    assert any(w >= 3 for w in clock.waits)
    assert log_lines(client)[-1]["retries"] == 1


def test_second_identical_call_comes_from_cache(tmp_path: Path) -> None:
    client, calls, _ = make_client(tmp_path, [httpx.Response(200, json=ok_body("hello"))])
    client.complete(MESSAGES, task="t")
    assert client.complete(MESSAGES, task="t") == "hello"
    assert len(calls) == 1
    first, second = log_lines(client)
    assert (first["cached"], second["cached"]) == (False, True)
    assert second["tokens_in"] == 0


def test_bad_json_is_rejected_then_reasked(tmp_path: Path) -> None:
    client, calls, _ = make_client(
        tmp_path,
        [
            httpx.Response(200, json=ok_body('{"category": "supply", "severity": "high"}')),
            httpx.Response(200, json=ok_body('{"category": "supply", "severity": 4}')),
        ],
    )
    risk = client.complete_json(MESSAGES, task="t", response_model=Risk)
    assert risk == Risk(category="supply", severity=4)
    assert len(calls) == 2
    line = log_lines(client)[-1]
    assert line["success"] is False  # not valid on the first try
    assert line["retries"] == 1
    assert line["tokens_in"] == 100  # both attempts are counted
    assert line["error"].startswith("severity: ")  # names the field, not just "1 error"


def test_gives_up_when_output_stays_invalid(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path, [httpx.Response(200, json=ok_body("not json"))] * 3)
    with pytest.raises(LLMError, match="no valid answer"):
        client.complete_json(MESSAGES, task="t", response_model=Risk)
    assert log_lines(client)[-1]["success"] is False
    assert not list(client.settings.llm_cache_dir.glob("*.json"))  # bad output is never cached


def test_401_is_not_retried_and_key_is_not_leaked(tmp_path: Path) -> None:
    client, calls, _ = make_client(tmp_path, [httpx.Response(401, json={"error": "bad key"})])
    with pytest.raises(LLMError, match="HTTP 401") as err:
        client.complete(MESSAGES, task="t")
    assert len(calls) == 1
    assert FAKE_KEY not in str(err.value)
    line = log_lines(client)[-1]  # failed calls are logged too, so success rates stay honest
    assert line["success"] is False
    assert "HTTP 401" in line["error"]


# What Groq strict mode really returns when the model breaks the schema (Nov 10 full run).
GROQ_SCHEMA_REJECTION = {
    "error": {
        "message": "Generated JSON does not match the expected schema. Please adjust your "
        "prompt. See 'failed_generation' for more details. Error: jsonschema: '/category' "
        "does not validate with /properties/category/enum: value must be one of 'a', 'b'",
        "type": "invalid_request_error",
        "failed_generation": '{"category": "made_up", "severity": 4}',
    }
}


def test_groq_schema_rejection_is_reasked_not_fatal(tmp_path: Path) -> None:
    client, calls, _ = make_client(
        tmp_path,
        [
            httpx.Response(400, json=GROQ_SCHEMA_REJECTION),
            httpx.Response(200, json=ok_body('{"category": "supply", "severity": 4}')),
        ],
    )
    risk = client.complete_json(MESSAGES, task="t", response_model=Risk)
    assert risk == Risk(category="supply", severity=4)
    assert len(calls) == 2

    reask = json.loads(calls[1].content)["messages"]
    assert reask[-2] == {"role": "assistant", "content": '{"category": "made_up", "severity": 4}'}
    assert "value must be one of" in reask[-1]["content"]

    line = log_lines(client)[-1]
    assert line["success"] is False  # not valid on the first try
    assert line["retries"] == 1
    assert line["error"].startswith("jsonschema: '/category'")
    assert line["rejected_token_estimate"] > 0


def test_rejected_tokens_count_toward_daily_budget(tmp_path: Path) -> None:
    client, _, _ = make_client(
        tmp_path,
        [
            httpx.Response(400, json=GROQ_SCHEMA_REJECTION),
            httpx.Response(200, json=ok_body('{"category": "supply", "severity": 4}')),
        ],
    )
    client.complete_json(MESSAGES, task="t", response_model=Risk)
    estimate = log_lines(client)[-1]["rejected_token_estimate"]
    assert client.log.tokens_last_24h() == 50 + 10 + estimate  # real usage + rejected estimate


def test_other_400_errors_still_fail_fast(tmp_path: Path) -> None:
    client, calls, _ = make_client(
        tmp_path, [httpx.Response(400, json={"error": {"message": "model not found"}})]
    )
    with pytest.raises(LLMError, match="HTTP 400"):
        client.complete(MESSAGES, task="t")
    assert len(calls) == 1


def test_daily_budget_stops_before_calling(tmp_path: Path) -> None:
    earlier = {
        "timestamp": datetime.now(UTC).isoformat(),
        "tokens_in": 9_000,
        "tokens_out": 900,
        "cached": False,
    }
    (tmp_path / "run_log.jsonl").write_text(json.dumps(earlier) + "\n")
    client, calls, _ = make_client(tmp_path, [], llm_daily_token_limit=10_000)
    with pytest.raises(DailyBudgetExceeded):
        client.complete(MESSAGES, task="t", max_tokens=500)
    assert calls == []


def test_json_request_asks_for_strict_schema(tmp_path: Path) -> None:
    body = ok_body('{"category": "supply", "severity": 4}')
    client, calls, _ = make_client(tmp_path, [httpx.Response(200, json=body)])
    client.complete_json(MESSAGES, task="t", response_model=Risk)
    sent = json.loads(calls[0].content)
    assert sent["response_format"]["json_schema"]["strict"] is True


def test_limiter_waits_when_minute_budget_is_full() -> None:
    clock = FakeClock()
    limiter = MinuteLimiter(8_000, 30, clock.now, clock.sleep)
    limiter.reserve(6_000)
    limiter.reserve(3_000)  # 9,000 > 8,000: must wait for the first call to leave the window
    assert clock.waits == [60.0]


def test_limiter_frees_room_when_real_usage_is_lower() -> None:
    clock = FakeClock()
    limiter = MinuteLimiter(8_000, 30, clock.now, clock.sleep)
    entry = limiter.reserve(6_000)
    limiter.settle(entry, 1_000)  # the call really used 1,000 tokens
    limiter.reserve(3_000)
    assert clock.waits == []
