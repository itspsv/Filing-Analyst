"""One small client for any OpenAI-compatible chat API (Groq by default).

Every LLM call in the project goes through LLMClient, which adds: a disk cache, a per-minute rate
limiter, a daily token budget, retries with growing waits plus jitter, Pydantic validation of JSON
answers (re-asking once or twice on bad output), and one line per call in the run log.
"""

import contextlib
import hashlib
import json
import random
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from pydantic import BaseModel, ValidationError

from filing_analyst.config import Settings

RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_HTTP_RETRIES = 5
MAX_VALIDATION_RETRIES = 2
BACKOFF_BASE_SECONDS = 1.0
TIMEOUT_SECONDS = 120.0
WINDOW_SECONDS = 60.0

Messages = list[dict[str, str]]


class LLMError(RuntimeError):
    pass


class DailyBudgetExceeded(LLMError):
    pass


def backoff_seconds(attempt: int, retry_after: str | None, jitter: float) -> float:
    """1, 2, 4, 8 ... seconds, never less than the server's retry-after, plus random jitter."""
    wait = BACKOFF_BASE_SECONDS * 2**attempt
    if retry_after:
        with contextlib.suppress(ValueError):  # ignore a malformed header
            wait = max(wait, float(retry_after))
    return wait + jitter


def describe_validation_error(err: ValidationError) -> str:
    """Name the field and the reason, e.g. 'risks.3.severity: Input should be ... (+1 more)'."""
    errors = err.errors()
    first = errors[0]
    where = ".".join(str(part) for part in first["loc"]) or "answer"
    more = f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""
    return f"{where}: {first['msg']}{more}"[:300]


class MinuteLimiter:
    """Sliding 60-second window: blocks until a call fits under both per-minute limits."""

    def __init__(
        self,
        tokens_per_minute: int,
        requests_per_minute: int,
        now: Callable[[], float],
        sleep: Callable[[float], None],
    ) -> None:
        self.tokens_per_minute = tokens_per_minute
        self.requests_per_minute = requests_per_minute
        self._now, self._sleep = now, sleep
        self._window: deque[list[float]] = deque()  # [timestamp, tokens]
        self._lock = threading.Lock()

    def reserve(self, tokens: int) -> list[float]:
        if tokens > self.tokens_per_minute:
            raise LLMError(f"one call needs ~{tokens} tokens, over the per-minute limit")
        while True:
            with self._lock:
                now = self._now()
                while self._window and now - self._window[0][0] >= WINDOW_SECONDS:
                    self._window.popleft()
                used = sum(entry[1] for entry in self._window)
                fits = used + tokens <= self.tokens_per_minute
                if fits and len(self._window) < self.requests_per_minute:
                    entry = [now, float(tokens)]
                    self._window.append(entry)
                    return entry
                wait = WINDOW_SECONDS - (now - self._window[0][0])
            self._sleep(max(wait, 0.05))

    def settle(self, entry: list[float], actual_tokens: int) -> None:
        """Replace the reserved estimate with the real token count from the response."""
        with self._lock:
            entry[1] = float(actual_tokens)


class RunLog:
    """Append-only JSONL log, one line per call. Also tells how many tokens the last 24 h used."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def append(self, record: dict) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(record) + "\n")

    def tokens_last_24h(self) -> int:
        if not self.path.exists():
            return 0
        since = datetime.now(UTC) - timedelta(hours=24)
        total = 0
        for line in self.path.read_text().splitlines():
            record = json.loads(line)
            if not record["cached"] and datetime.fromisoformat(record["timestamp"]) > since:
                total += record["tokens_in"] + record["tokens_out"]
        return total


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        if settings.llm_api_key is None:
            raise LLMError("no API key: set GROQ_API_KEY (or LLM_API_KEY) in .env")
        self.settings = settings
        self._http = httpx.Client(
            base_url=settings.llm_base_url,
            headers={"Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}"},
            timeout=TIMEOUT_SECONDS,
            transport=transport,
        )
        self._sleep = sleep
        self._slots = threading.BoundedSemaphore(settings.llm_max_concurrency)
        self.limiter = MinuteLimiter(
            settings.llm_tokens_per_minute, settings.llm_requests_per_minute, now, sleep
        )
        self.log = RunLog(settings.run_log)
        self._used_today = self.log.tokens_last_24h()
        self._budget_lock = threading.Lock()

    def complete(self, messages: Messages, task: str, max_tokens: int = 1024) -> str:
        return self._complete(messages, task, max_tokens, response_model=None)

    def complete_json[T: BaseModel](
        self, messages: Messages, task: str, response_model: type[T], max_tokens: int = 2048
    ) -> T:
        text = self._complete(messages, task, max_tokens, response_model)
        return response_model.model_validate_json(text)

    def _complete(
        self,
        messages: Messages,
        task: str,
        max_tokens: int,
        response_model: type[BaseModel] | None,
    ) -> str:
        payload = self._payload(messages, max_tokens, response_model)
        cache_path = self._cache_path(payload)
        if cache_path.exists():
            self._log(task, 0, 0, 0, success=True, retries=0, error=None, cached=True)
            return json.loads(cache_path.read_text())["text"]

        api_seconds = 0.0  # time spent waiting on the API only, not on our own rate limiter
        tokens_in = tokens_out = retries = 0
        first_error: str | None = None
        attempt_messages = messages
        for attempt in range(MAX_VALIDATION_RETRIES + 1):
            body, http_retries, seconds = self._post({**payload, "messages": attempt_messages})
            api_seconds += seconds
            retries += http_retries + (1 if attempt > 0 else 0)  # HTTP retries + re-asks
            tokens_in += body["usage"]["prompt_tokens"]
            tokens_out += body["usage"]["completion_tokens"]
            text = body["choices"][0]["message"]["content"] or ""
            error = self._validation_error(text, response_model)
            if error is None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({"text": text}))
                self._log(
                    task, tokens_in, tokens_out, api_seconds, attempt == 0, retries, first_error
                )
                return text
            first_error = first_error or error
            attempt_messages = [
                *messages,
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"That was not valid: {error}. Reply with JSON only."},
            ]
        self._log(task, tokens_in, tokens_out, api_seconds, False, retries, first_error)
        raise LLMError(f"{task}: no valid answer after {MAX_VALIDATION_RETRIES + 1} tries")

    def _payload(
        self, messages: Messages, max_tokens: int, response_model: type[BaseModel] | None
    ) -> dict:
        payload: dict = {
            "model": self.settings.llm_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if self.settings.llm_reasoning_effort:
            payload["reasoning_effort"] = self.settings.llm_reasoning_effort
        if response_model is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                },
            }
        return payload

    def _cache_path(self, payload: dict) -> Path:
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return self.settings.llm_cache_dir / f"{digest}.json"

    @staticmethod
    def _validation_error(text: str, response_model: type[BaseModel] | None) -> str | None:
        if response_model is None:
            return None if text.strip() else "empty answer"
        try:
            response_model.model_validate_json(text)
        except ValidationError as err:
            return describe_validation_error(err)
        return None

    def _post(self, payload: dict) -> tuple[dict, int, float]:
        """POST with retries. Returns the JSON body, retry count and seconds spent on the API."""
        api_seconds = 0.0
        estimate = len(json.dumps(payload["messages"])) // 4 + payload["max_tokens"]
        with self._budget_lock:
            if self._used_today + estimate > self.settings.llm_daily_token_limit:
                raise DailyBudgetExceeded(
                    f"{self._used_today} tokens used in the last 24 h; "
                    f"limit {self.settings.llm_daily_token_limit}"
                )
        for attempt in range(MAX_HTTP_RETRIES + 1):
            response: httpx.Response | None = None
            with self._slots:
                entry = self.limiter.reserve(estimate)
                sent = time.perf_counter()
                with contextlib.suppress(httpx.TransportError):  # network hiccup: retry like a 5xx
                    response = self._http.post("/chat/completions", json=payload)
                api_seconds += time.perf_counter() - sent
            if response is not None and response.status_code == 200:
                body = response.json()
                used = body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
                self.limiter.settle(entry, used)
                with self._budget_lock:
                    self._used_today += used
                return body, attempt, api_seconds
            if response is not None and response.status_code not in RETRY_STATUSES:
                raise LLMError(f"HTTP {response.status_code}: {response.text[:300]}")
            if attempt < MAX_HTTP_RETRIES:
                retry_after = response.headers.get("retry-after") if response else None
                self._sleep(backoff_seconds(attempt, retry_after, random.random()))
        raise LLMError(f"gave up after {MAX_HTTP_RETRIES} retries")

    def _log(
        self,
        task: str,
        tokens_in: int,
        tokens_out: int,
        latency_s: float,
        success: bool,
        retries: int,
        error: str | None,
        cached: bool = False,
    ) -> None:
        s = self.settings
        cost = (tokens_in * s.llm_eur_per_million_in + tokens_out * s.llm_eur_per_million_out) / 1e6
        self.log.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "model": s.llm_model,
                "task": task,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_eur": round(cost, 6),
                "latency_ms": round(latency_s * 1000),
                "success": success,
                "retries": retries,
                "error": error,
                "cached": cached,
            }
        )


class _Ping(BaseModel):
    answer: str


def main() -> None:
    """Smoke test: one tiny real JSON call. Run it twice; the second run comes from the cache."""
    client = LLMClient(Settings())
    messages = [{"role": "user", "content": 'Reply with the JSON object {"answer": "ok"}.'}]
    print(client.complete_json(messages, task="smoke_test", response_model=_Ping))
    print("last log line:", client.log.path.read_text().splitlines()[-1])


if __name__ == "__main__":
    main()
