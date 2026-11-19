import re
from decimal import Decimal
from pathlib import Path

import pytest
from agent_helpers import ScriptedClient, call, make_facts_db, says

from filing_analyst.agent.compose import DraftAgentAnswer, SqlCitation
from filing_analyst.agent.loop import AgentResult
from filing_analyst.agent.tools import ToolRunner
from filing_analyst.answer import Answer, Citation
from filing_analyst.config import Settings
from filing_analyst.evaluation.agent_eval import (
    TOLERANCE,
    AgentItem,
    Figure,
    figure_found,
    load_agent_items,
    run_items,
    score,
    summarize,
    written_numbers,
)

MARGIN = Figure(what="NVDA gross margin FY2026", percent="71.07%")
REVENUE = Figure(what="NVDA revenue FY2026", value=215938000000)


def item(kind: str = "mixed", figures=(MARGIN,), tickers=("NVDA",), refuse=False) -> AgentItem:
    return AgentItem(qid="t1", kind=kind, question="Q?", figures=list(figures),
                     cite_tickers=list(tickers), must_refuse=refuse, why="test")  # fmt: skip


def cite(kind: str, ticker: str = "NVDA") -> Citation:
    return Citation(kind=kind, ref="x", ticker=ticker, section="", quote_or_row="")


def result(text: str, citations: list[Citation], refused: bool = False) -> AgentResult:
    a = Answer(answer=text, citations=citations, refused=refused, steps=3, tokens=900,
               cost_eur=0.0, refusal_reason="model refused" if refused else None)  # fmt: skip
    return AgentResult(answer=a, trace=[], stop_reason="model finished", untraced_numbers=[])


def test_written_numbers_reads_units_and_percents_and_skips_chunk_ids() -> None:
    amounts, percents = written_numbers("Revenue was $215.938 billion [NVDA-2026-7-0001]; 71.07%.")
    assert amounts == [Decimal("215938000000.000")]
    assert percents == [Decimal("71.07")]


@pytest.mark.parametrize(
    ("text", "exact", "near"),
    [
        ("revenue of $215,938,000,000", True, True),
        ("revenue of $215,938 million", True, True),
        ("revenue of $215.9 billion", False, True),  # rounded: not exact, within 1%
        ("revenue of $200 billion", False, False),
    ],
)
def test_amounts_match_exactly_or_within_one_percent(text: str, exact: bool, near: bool) -> None:
    assert figure_found(REVENUE, text) is exact
    assert figure_found(REVENUE, text, TOLERANCE) is near


def test_percent_must_match_the_ratio_result() -> None:
    assert figure_found(MARGIN, "a gross margin of 71.07%")
    assert not figure_found(MARGIN, "a gross margin of 71.1%")
    assert figure_found(MARGIN, "a gross margin of 71.1%", TOLERANCE)
    assert not figure_found(MARGIN, "71.07 billion")  # a number without % is not a percent


def test_full_answer_passes() -> None:
    r = result("Gross margin was 71.07% [NVDA-2026-1A-0063].", [cite("ratio"), cite("chunk")])
    s = score(item(), r, 1.0)
    assert s.passed and s.problems == [] and s.figures_exact == 1
    assert s.citations == {"chunk": 1, "sql": 0, "ratio": 1}


@pytest.mark.parametrize(
    ("text", "citations", "problem"),
    [
        ("Gross margin was 71.1%.", [cite("ratio"), cite("chunk")], "figure missing"),
        ("Gross margin was 71.07%.", [cite("chunk")], "without a SQL or ratio citation"),
        ("Gross margin was 71.07%.", [cite("ratio"), cite("chunk", "AMD")], "no passage cited"),
    ],
)
def test_each_missing_part_is_named(text: str, citations: list[Citation], problem: str) -> None:
    s = score(item(), result(text, citations), 1.0)
    assert not s.passed
    assert any(problem in p for p in s.problems), s.problems


def test_refusals_are_scored_both_ways() -> None:
    must = item(kind="refuse", figures=(), tickers=(), refuse=True)
    assert score(must, result("x", [], refused=True), 1.0).passed
    assert score(must, result("It was 40%.", [cite("ratio")]), 1.0).problems == ["should refuse"]
    wrongly = score(item(), result("x", [], refused=True), 1.0)
    assert wrongly.problems == ["refused (model refused)"]


def test_end_to_end_with_a_scripted_model(tmp_path: Path) -> None:
    """The real graph, tools and checks; only the model is scripted."""
    db = make_facts_db(tmp_path / "facts.duckdb")
    sql = "SELECT label, value FROM facts WHERE ticker = 'NVDA' ORDER BY label"
    steps = [
        says(call("query_financials", {"sql": sql})),
        says(
            call(
                "compute_ratio",
                {
                    "name": "gross_margin",
                    "inputs": {"gross_profit": 153463000000, "revenue": 215938000000},
                },
            )
        ),  # fmt: skip
        says(),
    ]

    def compose(prompt: str) -> DraftAgentAnswer:
        qid = re.search(r'<sql id="(sql-[0-9a-f]+)">', prompt).group(1)
        rows = [SqlCitation(query_id=qid, row=0)]
        return DraftAgentAnswer(
            answer="NVIDIA's FY2026 gross margin was 71.07%.",
            refused=False,
            chunk_citations=[],
            sql_citations=rows,
            ratio_citations=[0],
        )

    number = item(kind="number", tickers=())
    scores = run_items([number], ScriptedClient(steps, compose), ToolRunner(lambda q, t: [], db))
    assert scores[0].passed, scores[0].problems
    assert scores[0].tools == ["query_financials", "compute_ratio"]
    summary = summarize(scores)
    assert summary["passed"] == 1 and summary["figures_exact"] == "1/1"


def test_question_file_is_valid_and_balanced() -> None:
    items = load_agent_items(Settings().evals_dir / "agent_questions.jsonl")
    assert len(items) == 10 and len({i.qid for i in items}) == 10
    assert sum(i.must_refuse for i in items) == 1
    for i in items:
        assert i.must_refuse == (i.kind == "refuse")
        assert all((f.value is None) != (f.percent is None) for f in i.figures)
        assert bool(i.figures) == (i.kind in ("number", "mixed"))
        assert bool(i.cite_tickers) == (i.kind in ("text", "mixed"))
