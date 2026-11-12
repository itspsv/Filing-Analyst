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
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from filing_analyst.config import Settings

RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_HTTP_RETRIES = 5
MAX_VALIDATION_RETRIES = 2
BACKOFF_BASE_SECONDS = 1.0
TIMEOUT_SECONDS = 120.0
WINDOW_SECONDS = 60.0

Messages = list[dict[str, str]]


SCHEMA_REJECTED = "does not match the expected schema"


class LLMError(RuntimeError):
    pass


class DailyBudgetExceeded(LLMError):
    pass


class SchemaRejected(LLMError):
    """Groq strict mode checks the JSON after generating it and returns HTTP 400 if it breaks
    the schema. It does not prevent bad output, so we treat this like an invalid answer."""

    def __init__(
        self,
        reason: str,
        failed_generation: str,
        token_estimate: int,
        seconds: float,
        http_retries: int,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.failed_generation = failed_generation
        self.token_estimate = token_estimate
        self.seconds = seconds
        self.http_retries = http_retries


def schema_rejection(response: httpx.Response) -> tuple[str, str] | None:
    """(reason, rejected JSON) if Groq refused the answer for breaking the schema, else None."""
    try:
        body = response.json()
    except ValueError:
        return None
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return None
    message = error.get("message", "")
    if SCHEMA_REJECTED not in message:
        return None
    return message.split("Error: ", 1)[-1][:300], error.get("failed_generation", "")


@dataclass
class CallStats:
    """Totals for one logical call, across HTTP retries and re-asks."""

    tokens_in: int = 0
    tokens_out: int = 0
    api_seconds: float = 0.0
    retries: int = 0
    rejected_token_estimate: int = 0  # Groq gives no usage for rejected answers
    first_error: str | None = None


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

    def tokens_last_24h(self, model: str | None = None) -> int:
        """Tokens used in the last 24 h, for one model or (model=None) all of them.
        Groq's daily limits are per model, so the client counts only its own model."""
        if not self.path.exists():
            return 0
        since = datetime.now(UTC) - timedelta(hours=24)
        total = 0
        for line in self.path.read_text().splitlines():
            record = json.loads(line)
            if record["cached"] or (model is not None and record.get("model") != model):
                continue
            if datetime.fromisoformat(record["timestamp"]) > since:
                total += record["tokens_in"] + record["tokens_out"]
                total += record.get("rejected_token_estimate", 0)
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
        self._used_today = self.log.tokens_last_24h(settings.llm_model)
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
            self._log(task, CallStats(), success=True, cached=True)
            return json.loads(cache_path.read_text())["text"]

        stats = CallStats()
        attempt_messages = messages
        for attempt in range(MAX_VALIDATION_RETRIES + 1):
            if attempt > 0:
                stats.retries += 1  # a re-ask
            try:
                text, error = self._attempt({**payload, "messages": attempt_messages}, stats)
            except LLMError as err:
                spent = stats.tokens_in or stats.rejected_token_estimate
                if spent or not isinstance(err, DailyBudgetExceeded):
                    self._log(task, stats, success=False, error=str(err)[:300])
                raise
            if error is None:
                error = self._validation_error(text, response_model)
            if error is None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({"text": text}))
                self._log(task, stats, success=attempt == 0)
                return text
            stats.first_error = stats.first_error or error
            attempt_messages = [
                *messages,
                *([{"role": "assistant", "content": text}] if text else []),
                {
                    "role": "user",
                    "content": f"That was not valid: {error}. "
                    "Use only the allowed values. Reply with JSON only.",
                },
            ]
        self._log(task, stats, success=False)
        raise LLMError(f"{task}: no valid answer after {MAX_VALIDATION_RETRIES + 1} tries")

    def _attempt(self, payload: dict, stats: CallStats) -> tuple[str, str | None]:
        """One request. Returns the answer text, plus Groq's rejection reason if it refused it."""
        try:
            body, http_retries, seconds = self._post(payload)
        except SchemaRejected as rejected:
            stats.api_seconds += rejected.seconds
            stats.retries += rejected.http_retries
            stats.rejected_token_estimate += rejected.token_estimate
            return rejected.failed_generation, rejected.reason
        stats.api_seconds += seconds
        stats.retries += http_retries
        stats.tokens_in += body["usage"]["prompt_tokens"]
        stats.tokens_out += body["usage"]["completion_tokens"]
        return body["choices"][0]["message"]["content"] or "", None

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
                    # strict needs extra="forbid" on every model (additionalProperties: false)
                    "strict": self.settings.llm_strict_schema,
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
                    f"{self._used_today} tokens used in the last 24 h on "
                    f"{self.settings.llm_model}; limit {self.settings.llm_daily_token_limit}"
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
            rejection = schema_rejection(response) if response is not None else None
            if response is not None and response.status_code == 400 and rejection is not None:
                with self._budget_lock:  # Groq still spent tokens on the rejected answer
                    self._used_today += estimate
                raise SchemaRejected(*rejection, estimate, api_seconds, attempt)
            if response is not None and response.status_code not in RETRY_STATUSES:
                raise LLMError(f"HTTP {response.status_code}: {response.text[:300]}")
            if attempt < MAX_HTTP_RETRIES:
                retry_after = response.headers.get("retry-after") if response else None
                self._sleep(backoff_seconds(attempt, retry_after, random.random()))
        raise LLMError(f"gave up after {MAX_HTTP_RETRIES} retries")

    def _log(
        self,
        task: str,
        stats: CallStats,
        success: bool,
        error: str | None = None,
        cached: bool = False,
    ) -> None:
        """One line per logical call. success = valid on the first try (no re-ask needed)."""
        s = self.settings
        cost = (
            stats.tokens_in * s.llm_eur_per_million_in
            + stats.tokens_out * s.llm_eur_per_million_out
        ) / 1e6
        self.log.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "model": s.llm_model,
                "task": task,
                "tokens_in": stats.tokens_in,
                "tokens_out": stats.tokens_out,
                "rejected_token_estimate": stats.rejected_token_estimate,
                "cost_eur": round(cost, 6),
                "latency_ms": round(stats.api_seconds * 1000),
                "success": success,
                "retries": stats.retries,
                "error": error or stats.first_error,
                "cached": cached,
            }
        )


class _Ping(BaseModel):
    model_config = ConfigDict(extra="forbid")  # strict mode needs it on every model

    answer: str


def main() -> None:
    """Smoke test: one tiny real JSON call. Run it twice; the second run comes from the cache."""
    client = LLMClient(Settings())
    messages = [{"role": "user", "content": 'Reply with the JSON object {"answer": "ok"}.'}]
    print(client.complete_json(messages, task="smoke_test", response_model=_Ping))
    print("last log line:", client.log.path.read_text().splitlines()[-1])


if __name__ == "__main__":
    main()
