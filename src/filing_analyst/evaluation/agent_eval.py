"""Run the agent (LangGraph) on mixed questions and score each answer with code checks.

passed = refused when it must, otherwise: not refused, every expected figure written exactly
         (an exact unit change like "$132.175 billion" counts), a SQL or ratio citation behind
         the figures, and a passage cited for every company the question needs text from.
Expected figures come from DuckDB and compute_ratio (evals/agent_questions.jsonl), never an LLM.

    uv run python -m filing_analyst.evaluation.agent_eval [--limit 3] [--ids a01,a07]
"""

import argparse
import json
import math
import re
import statistics
import time
from collections.abc import Callable, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from filing_analyst.agent.compose import CHUNK_REF, NUMBER, SCALED, UNITS
from filing_analyst.agent.graph import run_graph
from filing_analyst.agent.loop import AgentClient, AgentResult
from filing_analyst.agent.tools import ToolRunner
from filing_analyst.answer import load_default_search
from filing_analyst.config import Settings
from filing_analyst.llm import DailyBudgetExceeded, LLMClient, LLMError

TOLERANCE = Decimal("0.01")  # "within 1%", reported next to exact matches
PERCENT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*%")


class Figure(BaseModel):
    what: str
    value: float | None = None  # a SQL value in US dollars
    percent: str | None = None  # a compute_ratio result, e.g. "46.78%"


class AgentItem(BaseModel):
    qid: str
    kind: Literal["text", "number", "mixed", "refuse"]
    question: str
    figures: list[Figure]
    cite_tickers: list[str]
    must_refuse: bool
    why: str


class AgentScore(BaseModel):
    qid: str
    kind: str
    question: str
    answer: str
    passed: bool
    problems: list[str]
    refused: bool
    refusal_reason: str | None
    stop_reason: str
    figures_exact: int
    figures_within_1pct: int
    figures_expected: int
    citations: dict[str, int]  # by kind: chunk, sql, ratio
    dropped_citations: int
    untraced_numbers: list[str]
    tools: list[str]
    llm_calls: int
    tokens: int
    seconds: float
    error: str | None = None


def load_agent_items(path: Path) -> list[AgentItem]:
    return [AgentItem.model_validate_json(line) for line in path.read_text().splitlines() if line]


def _decimal(text: str) -> Decimal:
    return Decimal(text.replace(",", "").rstrip("."))


def written_numbers(answer: str) -> tuple[list[Decimal], list[Decimal]]:
    """(amounts in plain units, percentages) written in an answer."""
    text = CHUNK_REF.sub(" ", answer)
    percents = [_decimal(m) for m in PERCENT.findall(text)]
    text = PERCENT.sub(" ", text)
    amounts = [_decimal(n) * UNITS[unit.lower()] for n, unit in SCALED.findall(text)]
    amounts += [_decimal(n) for n in NUMBER.findall(SCALED.sub(" ", text))]
    return amounts, percents


def figure_found(figure: Figure, answer: str, tolerance: Decimal = Decimal(0)) -> bool:
    amounts, percents = written_numbers(answer)
    if figure.percent is not None:
        target, written = _decimal(figure.percent.rstrip("%")), percents
    else:
        target, written = Decimal(str(figure.value)), amounts
    return any(abs(w - target) <= tolerance * abs(target) for w in written)


def score(item: AgentItem, result: AgentResult, seconds: float) -> AgentScore:
    a = result.answer
    kinds = {k: sum(c.kind == k for c in a.citations) for k in ("chunk", "sql", "ratio")}
    exact = [figure_found(f, a.answer) for f in item.figures]
    near = [figure_found(f, a.answer, TOLERANCE) for f in item.figures]
    problems = []
    if item.must_refuse != a.refused:
        problems.append("should refuse" if item.must_refuse else f"refused ({a.refusal_reason})")
    if not item.must_refuse and not a.refused:
        missing = [f.what for f, ok in zip(item.figures, exact, strict=True) if not ok]
        problems += [f"figure missing: {what}" for what in missing]
        if item.figures and not (kinds["sql"] or kinds["ratio"]):
            problems.append("figures without a SQL or ratio citation")
        cited = {c.ticker for c in a.citations if c.kind == "chunk"}
        problems += [f"no passage cited for {t}" for t in item.cite_tickers if t not in cited]
    return AgentScore(
        qid=item.qid, kind=item.kind, question=item.question, answer=a.answer,
        passed=not problems, problems=problems, refused=a.refused,
        refusal_reason=a.refusal_reason, stop_reason=result.stop_reason,
        figures_exact=sum(exact), figures_within_1pct=sum(near),
        figures_expected=len(item.figures), citations=kinds,
        dropped_citations=a.dropped_citations, untraced_numbers=result.untraced_numbers,
        tools=[s.tool for s in result.trace], llm_calls=a.steps, tokens=a.tokens,
        seconds=round(seconds, 2),
    )  # fmt: skip


def failed(item: AgentItem, error: Exception, seconds: float) -> AgentScore:
    return AgentScore(
        qid=item.qid, kind=item.kind, question=item.question, answer="", passed=False,
        problems=["run failed"], refused=False, refusal_reason=None, stop_reason="error",
        figures_exact=0, figures_within_1pct=0, figures_expected=len(item.figures),
        citations={}, dropped_citations=0, untraced_numbers=[], tools=[], llm_calls=0,
        tokens=0, seconds=round(seconds, 2), error=str(error)[:300],
    )  # fmt: skip


Run = Callable[[str, AgentClient, ToolRunner], AgentResult]


def run_items(
    items: Sequence[AgentItem], client: AgentClient, runner: ToolRunner, run: Run = run_graph
) -> list[AgentScore]:
    scores = []
    for item in items:
        started = time.perf_counter()
        try:
            result = run(item.question, client, runner)
        except DailyBudgetExceeded:
            raise  # out of budget: stop; a rerun resumes from the cache
        except LLMError as err:
            scores.append(failed(item, err, time.perf_counter() - started))
        else:
            scores.append(score(item, result, time.perf_counter() - started))
        s = scores[-1]
        print(f"{s.qid} {s.kind:<6} {'PASS' if s.passed else 'FAIL'} {s.tokens:>6} tokens "
              f"{s.seconds:5.1f} s  tools {s.tools}  {'; '.join(s.problems)}")  # fmt: skip
    return scores


def summarize(scores: Sequence[AgentScore]) -> dict:
    n = len(scores)
    seconds = sorted(s.seconds for s in scores)
    by_kind = {}
    for kind in ("text", "number", "mixed", "refuse"):
        group = [s for s in scores if s.kind == kind]
        if group:
            by_kind[kind] = f"{sum(s.passed for s in group)}/{len(group)}"
    expected = sum(s.figures_expected for s in scores)
    return {
        "questions": n,
        "passed": sum(s.passed for s in scores),
        "passed_by_kind": by_kind,
        "failed_runs": sum(s.error is not None for s in scores),
        "figures_exact": f"{sum(s.figures_exact for s in scores)}/{expected}",
        "figures_within_1pct": f"{sum(s.figures_within_1pct for s in scores)}/{expected}",
        "untraced_numbers": sum(len(s.untraced_numbers) for s in scores),
        "dropped_citations": sum(s.dropped_citations for s in scores),
        "tokens_total": sum(s.tokens for s in scores),
        "tokens_mean": round(statistics.mean(s.tokens for s in scores)),
        "llm_calls_mean": round(statistics.mean(s.llm_calls for s in scores), 1),
        "seconds_p50": round(statistics.median(seconds), 2),
        "seconds_p95": seconds[math.ceil(0.95 * n) - 1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the agent on mixed questions.")
    parser.add_argument("--model", default=None, help="override LLM_MODEL")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--ids", default=None, help="comma-separated qids, e.g. a01,a07")
    args = parser.parse_args()
    settings = Settings()
    if args.model:
        settings = settings.model_copy(update={"llm_model": args.model})
    items = load_agent_items(settings.evals_dir / "agent_questions.jsonl")
    if args.ids:
        items = [i for i in items if i.qid in args.ids.split(",")]
    items = items[: args.limit]
    search = load_default_search(settings)
    search("warm-up", None)  # model loading stays out of the timings
    scores = run_items(items, LLMClient(settings), ToolRunner(search, settings.facts_db))
    summary = summarize(scores)
    print(f"\n{settings.llm_model}: {json.dumps(summary, indent=2)}")
    out = settings.evals_dir / f"agent_report_{settings.llm_model.split('/')[-1]}.json"
    report = {"model": settings.llm_model, "summary": summary,
              "questions": [s.model_dump() for s in scores]}  # fmt: skip
    out.write_text(json.dumps(report, indent=2))
    print(f"report: {out}")


if __name__ == "__main__":
    main()
