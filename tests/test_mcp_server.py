from pathlib import Path
from typing import Any

import anyio
import pytest
from agent_helpers import make_facts_db
from mcp import Client

from filing_analyst.agent.tools import ToolRunner
from filing_analyst.mcp_server import build_server
from filing_analyst.parse.chunk import Chunk

EXPORT = Chunk(
    chunk_id="NVDA-2026-1A-0063",
    ticker="NVDA",
    fiscal_year=2026,
    section="1A",
    text="Export controls restrict sales of H20 products to China.",
    token_count=12,
)


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return make_facts_db(tmp_path / "facts.duckdb")


def call_tool(db: Path, name: str, arguments: dict[str, Any]) -> Any:
    """Connect a real MCP client to the server in memory and call one tool."""
    server = build_server(ToolRunner(lambda q, t: [EXPORT], db, {"NVDA": "NVIDIA CORP"}))

    async def go() -> Any:
        async with Client(server) as client:
            return await client.call_tool(name, arguments)

    return anyio.run(go)


def test_lists_three_read_only_tools_with_schemas(db: Path) -> None:
    server = build_server(ToolRunner(lambda q, t: [], db))

    async def go() -> Any:
        async with Client(server) as client:
            return (await client.list_tools()).tools

    tools = {t.name: t for t in anyio.run(go)}
    assert set(tools) == {"search_filings", "query_financials", "compute_ratio"}
    assert all(t.annotations.read_only_hint for t in tools.values())
    assert tools["search_filings"].input_schema["required"] == ["query", "company"]
    ratio_name = tools["compute_ratio"].input_schema["properties"]["name"]
    assert "gross_margin" in ratio_name["description"]


def test_sql_returns_rows_with_a_query_id(db: Path) -> None:
    result = call_tool(db, "query_financials", {"sql": "SELECT label, value FROM facts"})
    assert not result.is_error
    assert result.structured_content["query_id"].startswith("sql-")
    assert ["revenue", 215938000000.0] in result.structured_content["rows"]


def test_write_sql_is_rejected_and_the_table_is_untouched(db: Path) -> None:
    result = call_tool(db, "query_financials", {"sql": "DELETE FROM facts"})
    assert result.is_error and "SQLRejected" in result.content[0].text
    rows = call_tool(db, "query_financials", {"sql": "SELECT count(*) AS n FROM facts"})
    assert rows.structured_content["rows"] == [[2]]


def test_search_resolves_company_names_and_refuses_unknown_ones(db: Path) -> None:
    found = call_tool(db, "search_filings", {"query": "export", "company": "Nvidia"})
    assert [c["chunk_id"] for c in found.structured_content["result"]] == [EXPORT.chunk_id]
    unknown = call_tool(db, "search_filings", {"query": "export", "company": "Intel"})
    assert unknown.is_error and "not covered" in unknown.content[0].text


def test_ratio_is_computed_in_code(db: Path) -> None:
    inputs = {"gross_profit": 153463000000, "revenue": 215938000000}
    result = call_tool(db, "compute_ratio", {"name": "gross_margin", "inputs": inputs})
    assert result.structured_content["percent"] == "71.07%"
    bad = call_tool(db, "compute_ratio", {"name": "pe_ratio", "inputs": inputs})
    assert bad.is_error and "RatioError" in bad.content[0].text
