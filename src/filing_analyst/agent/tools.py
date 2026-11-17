"""The agent's three tools: what the model is told about them, and the code that runs them.

Every tool returns a ToolOutcome. Errors (unsafe SQL, a bad ratio input, an unknown company)
go back to the model as a message so it can try again; they never crash the loop.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from filing_analyst.answer import EVIDENCE_K, Search
from filing_analyst.config import COMPANIES
from filing_analyst.parse.chunk import Chunk
from filing_analyst.tools.compute_ratio import RATIOS, RatioError, RatioResult, compute_ratio
from filing_analyst.tools.query_financials import (
    SCHEMA_DESCRIPTION,
    QueryFailed,
    QueryResult,
    SQLRejected,
    query_financials,
)

AGENT_MAX_ROWS = 50  # fewer rows than the API default, to keep the conversation small


def _function(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SPECS = [
    _function(
        "search_filings",
        "Search the latest 10-K text (Item 1A risk factors, Item 7 management's discussion). "
        "Returns the 5 most relevant passages with their chunk_id.",
        {
            "query": {"type": "string", "description": "what to look for, in plain words"},
            "company": {"type": "string", "description": f"ticker, one of {', '.join(COMPANIES)}"},
        },
        ["query", "company"],
    ),
    _function(
        "query_financials",
        f"Run one read-only SQL SELECT on DuckDB. Only this table exists:\n{SCHEMA_DESCRIPTION}",
        {"sql": {"type": "string"}},
        ["sql"],
    ),
    _function(
        "compute_ratio",
        "Compute a ratio in code. Never calculate yourself. gross_margin needs revenue and "
        "gross_profit (or cost_of_revenue); operating_margin needs operating_income and revenue; "
        "net_margin needs net_income and revenue; growth needs old and new.",
        {
            "name": {"type": "string", "enum": sorted(RATIOS)},
            "inputs": {"type": "object", "additionalProperties": {"type": "number"}},
        },
        ["name", "inputs"],
    ),
]


@dataclass
class ToolOutcome:
    text: str  # what the model sees (JSON)
    chunks: list[Chunk] = field(default_factory=list)
    query: QueryResult | None = None
    ratio: RatioResult | None = None
    error: str | None = None


def _error(message: str) -> ToolOutcome:
    return ToolOutcome(text=json.dumps({"error": message}), error=message)


class ToolRunner:
    def __init__(self, search: Search, db_path: Path) -> None:
        self.search, self.db_path = search, db_path

    def run(self, name: str, arguments: str) -> ToolOutcome:
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return _error(f"arguments are not valid JSON: {arguments[:200]}")
        try:
            if name == "search_filings":
                return self._search(str(args["query"]), str(args.get("company", "")).upper())
            if name == "query_financials":
                return self._sql(str(args["sql"]))
            if name == "compute_ratio":
                return self._ratio(str(args["name"]), dict(args["inputs"]))
        except KeyError as err:
            return _error(f"missing argument {err} for {name}")
        except (SQLRejected, QueryFailed, RatioError, TypeError, ValueError) as err:
            return _error(f"{type(err).__name__}: {err}")
        return _error(f"unknown tool {name!r}; use one of search_filings, query_financials, "
                      "compute_ratio")  # fmt: skip

    def _search(self, query: str, company: str) -> ToolOutcome:
        if company not in COMPANIES:
            return _error(f"company {company!r} is not covered; use one of {', '.join(COMPANIES)}")
        chunks = self.search(query, company)[:EVIDENCE_K]
        shown = [{"chunk_id": c.chunk_id, "company": c.ticker, "section": f"Item {c.section}",
                  "text": c.text} for c in chunks]  # fmt: skip
        return ToolOutcome(text=json.dumps(shown), chunks=chunks)

    def _sql(self, sql: str) -> ToolOutcome:
        result = query_financials(sql, self.db_path, max_rows=AGENT_MAX_ROWS)
        return ToolOutcome(text=result.model_dump_json(), query=result)

    def _ratio(self, name: str, inputs: dict) -> ToolOutcome:
        result = compute_ratio(name, {k: float(v) for k, v in inputs.items()})
        return ToolOutcome(text=result.model_dump_json(), ratio=result)
