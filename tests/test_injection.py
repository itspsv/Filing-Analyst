from collections.abc import Callable
from pathlib import Path

import pytest
from agent_helpers import ScriptedClient, call, make_facts_db, says

from filing_analyst.agent.compose import DraftAgentAnswer
from filing_analyst.agent.graph import run_graph
from filing_analyst.agent.injection import (
    ATTACKS,
    CODEWORD,
    PLANTED_ID,
    check,
    planted_chunk,
    run_attack,
    with_planted,
)
from filing_analyst.agent.loop import AgentResult, Step, run_agent
from filing_analyst.answer import Answer
from filing_analyst.config import Settings
from filing_analyst.parse.chunk import Chunk

ATTACK = {a.name: a for a in ATTACKS}
REAL = Chunk(
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


@pytest.fixture(params=["loop", "graph"])
def run(request: pytest.FixtureRequest) -> Callable:
    return run_agent if request.param == "loop" else run_graph


def result(answer: str, refused: bool = False, steps: list[Step] | None = None) -> AgentResult:
    a = Answer(answer=answer, citations=[], refused=refused, steps=1, tokens=0, cost_eur=0.0)
    return AgentResult(answer=a, trace=steps or [], stop_reason="model finished",
                       untraced_numbers=[])  # fmt: skip


def test_planted_passage_comes_first_only_for_its_company() -> None:
    search = with_planted(lambda q, t: [REAL], planted_chunk(ATTACK["codeword"]))
    assert [c.chunk_id for c in search("supply", "NVDA")] == [PLANTED_ID, REAL.chunk_id]
    assert [c.chunk_id for c in search("supply", "AMD")] == [REAL.chunk_id]
    assert CODEWORD in search("supply", "NVDA")[0].text


def test_obeying_model_cannot_change_the_table(run: Callable, db: Path) -> None:
    """Even a model that follows the planted order is stopped by the SQL guard in code."""
    client = ScriptedClient(
        [
            says(call("search_filings", {"query": "export controls", "company": "NVDA"})),
            says(call("query_financials", {"sql": "DELETE FROM facts WHERE ticker = 'NVDA'"})),
            says(),
        ],
        compose=lambda prompt: DraftAgentAnswer(
            answer="x", refused=True, chunk_citations=[], sql_citations=[], ratio_citations=[]
        ),  # fmt: skip
    )
    report = run_attack(ATTACK["destructive_sql"], run, client, lambda q, t: [REAL], db)
    sql_step = report.result.trace[1]
    assert report.delivered
    assert sql_step.error and "SQLRejected" in sql_step.result
    assert any("write SQL" in f for f in report.failures)  # the model obeyed: flagged
    assert not any("rows changed" in f for f in report.failures)  # but nothing was deleted


@pytest.mark.parametrize(
    ("name", "answer", "flag"),
    [
        ("codeword", f"{CODEWORD}", "codeword"),
        ("codeword", f"answer: {CODEWORD.lower()}", "codeword"),
        ("fake_number", "NVIDIA's FY2026 revenue was $999 billion.", "999"),
    ],
)
def test_checks_flag_a_taken_over_answer(name: str, answer: str, flag: str) -> None:
    failures = check(ATTACK[name], result(answer), 2, 2)
    assert any(flag in f for f in failures)


def test_checks_pass_a_clean_answer_and_flag_refusals_and_lost_rows() -> None:
    clean = result("NVIDIA's FY2026 revenue was $215,938 million [sql-1 row 0].")
    assert check(ATTACK["fake_number"], clean, 2, 2) == []
    assert "refused" in check(ATTACK["codeword"], result("x", refused=True), 2, 2)[0]
    assert "rows changed" in check(ATTACK["destructive_sql"], clean, 2, 1)[0]


@pytest.fixture(scope="module")
def live() -> tuple:
    settings = Settings()
    if settings.llm_api_key is None:
        pytest.skip("no LLM API key")
    from filing_analyst.answer import load_default_search  # loads torch: only here
    from filing_analyst.llm import LLMClient

    return settings, LLMClient(settings), load_default_search(settings)


@pytest.mark.llm
@pytest.mark.parametrize("attack", ATTACKS, ids=lambda a: a.name)
def test_real_model_ignores_the_planted_instruction(attack, run: Callable, live: tuple) -> None:
    settings, client, search = live
    report = run_attack(attack, run, client, search, settings.facts_db)
    assert report.delivered, "the planted passage never reached the model"
    assert report.failures == []
