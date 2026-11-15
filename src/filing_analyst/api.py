"""HTTP API: POST /ask returns a cited Answer as JSON.

    uv run uvicorn filing_analyst.api:app --port 8000
    curl -X POST localhost:8000/ask -H "content-type: application/json" \\
         -d '{"question": "What share of Broadcom wafers come from TSMC?", "ticker": "AVGO"}'

The search models load once at startup (about 30 s), not on every request.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from filing_analyst.answer import Answer, AnswerClient, Search, answer_question, load_default_search
from filing_analyst.config import Settings
from filing_analyst.llm import DailyBudgetExceeded, LLMClient, LLMError


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=500)
    ticker: str | None = Field(default=None, max_length=10, description="e.g. NVDA")


def create_app(search: Search | None = None, client: AnswerClient | None = None) -> FastAPI:
    """Real search and LLM by default; tests pass fakes instead."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = Settings() if search is None or client is None else None
        app.state.search = search or load_default_search(settings)
        app.state.client = client or LLMClient(settings)
        yield

    app = FastAPI(title="Filing Analyst", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=Answer)
    def ask(body: AskRequest, request: Request) -> Answer:
        state = request.app.state
        try:
            return answer_question(body.question, body.ticker, state.search, state.client)
        except DailyBudgetExceeded as err:
            raise HTTPException(503, f"daily LLM budget used up, try later: {err}") from err
        except LLMError as err:
            raise HTTPException(502, f"LLM call failed: {err}") from err

    return app


app = create_app()
