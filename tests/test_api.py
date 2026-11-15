from fastapi.testclient import TestClient
from pydantic import BaseModel

from filing_analyst.answer import DraftAnswer, DraftCitation
from filing_analyst.api import create_app
from filing_analyst.llm import CallStats, DailyBudgetExceeded, LLMError, Messages
from filing_analyst.parse.chunk import Chunk

WAFERS = Chunk(
    chunk_id="AVGO-2025-1A-0018",
    ticker="AVGO",
    fiscal_year=2025,
    section="1A",
    text="During fiscal year 2025, approximately 95% of the wafers were produced by TSMC.",
    token_count=14,
)
GOOD = DraftAnswer(
    answer="About 95% [AVGO-2025-1A-0018].",
    citations=[DraftCitation(chunk_id=WAFERS.chunk_id, quote="approximately 95% of the wafers")],
    refused=False,
)


class FakeClient:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def complete_json_with_stats[T: BaseModel](
        self, messages: Messages, task: str, response_model: type[T], max_tokens: int = 0
    ) -> tuple[T, CallStats]:
        if self.error:
            raise self.error
        return response_model.model_validate(GOOD.model_dump()), CallStats(tokens_in=900)

    def cost_eur(self, stats: CallStats) -> float:
        return 0.0


def client(error: Exception | None = None) -> TestClient:
    return TestClient(create_app(lambda q, t: [WAFERS], FakeClient(error)))


def test_health() -> None:
    with client() as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_ask_returns_a_cited_answer() -> None:
    with client() as c:
        response = c.post("/ask", json={"question": "Who makes the wafers?", "ticker": "AVGO"})
    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is False and body["tokens"] == 900
    assert body["citations"][0]["ref"] == "AVGO-2025-1A-0018"


def test_unknown_company_is_a_normal_refusal_not_an_error() -> None:
    with client() as c:
        body = c.post("/ask", json={"question": "Intel risks?", "ticker": "INTC"}).json()
    assert body["refused"] is True and body["refusal_reason"] == "company not covered"


def test_bad_requests_are_rejected_with_422() -> None:
    with client() as c:
        assert c.post("/ask", json={"question": "hi"}).status_code == 422  # too short
        assert c.post("/ask", json={"question": "Valid?", "admin": True}).status_code == 422
        assert c.post("/ask", json={}).status_code == 422


def test_llm_problems_map_to_clear_http_errors() -> None:
    with client(DailyBudgetExceeded("190000 used")) as c:
        response = c.post("/ask", json={"question": "Who makes the wafers?", "ticker": "AVGO"})
        assert response.status_code == 503 and "budget" in response.json()["detail"]
    with client(LLMError("HTTP 500")) as c:
        response = c.post("/ask", json={"question": "Who makes the wafers?", "ticker": "AVGO"})
        assert response.status_code == 502
