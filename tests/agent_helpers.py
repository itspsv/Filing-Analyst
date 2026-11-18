"""Shared by the agent tests: a fake model playing back scripted steps, a tiny facts table."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
from pydantic import BaseModel

from filing_analyst.agent.compose import DraftAgentAnswer
from filing_analyst.llm import CallStats


def make_facts_db(path: Path) -> Path:
    with duckdb.connect(str(path)) as con:
        con.execute(
            "CREATE TABLE facts (ticker VARCHAR, label VARCHAR, fiscal_year INT, value DOUBLE)"
        )
        con.execute(
            "INSERT INTO facts VALUES ('NVDA', 'revenue', 2026, 215938000000), "
            "('NVDA', 'gross_profit', 2026, 153463000000)"
        )
    return path


def call(name: str, arguments: dict | str, call_id: str = "c1") -> dict:
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


def says(*tool_calls: dict) -> dict:
    return {
        "role": "assistant",
        "content": "" if tool_calls else "done",
        "tool_calls": list(tool_calls),
    }


class ScriptedClient:
    """Plays back agent steps in order; the final answer is built from the compose prompt."""

    def __init__(
        self,
        steps: list[dict],
        compose: Callable[[str], DraftAgentAnswer] | None = None,
        tokens_per_call: int = 100,
    ) -> None:
        self.steps, self.compose, self.tokens = steps, compose, tokens_per_call
        self.seen: list[list[dict[str, Any]]] = []
        self.composed = False

    def chat_with_tools(self, messages, tools, task, max_tokens=0) -> tuple[dict, CallStats]:
        self.seen.append(list(messages))
        step = self.steps[min(len(self.seen), len(self.steps)) - 1]
        if isinstance(step, Exception):
            raise step
        return step, CallStats(tokens_in=self.tokens)

    def complete_json_with_stats[T: BaseModel](
        self, messages, task, response_model: type[T], max_tokens=0
    ) -> tuple[T, CallStats]:
        self.composed = True
        assert self.compose is not None, "compose was not expected in this test"
        draft = self.compose(messages[1]["content"])
        return response_model.model_validate(draft.model_dump()), CallStats(tokens_in=50)

    def cost_eur(self, stats: CallStats) -> float:
        return 0.0
