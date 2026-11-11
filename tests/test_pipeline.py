from pathlib import Path

import pytest

from filing_analyst.config import Settings
from filing_analyst.extract import run_extraction
from filing_analyst.llm import DailyBudgetExceeded
from filing_analyst.pipeline import PipelineError, Step, build_steps, run_pipeline

SETTINGS = Settings(_env_file=None, sec_user_agent="Test test@example.com")


def recording_steps(names: list[str], fail_at: str | None = None) -> tuple[list[Step], list[str]]:
    ran: list[str] = []

    def make(name: str) -> Step:
        def step(_: Settings) -> str:
            ran.append(name)
            if name == fail_at:
                raise ValueError("boom")
            return "ok"

        return name, step

    return [make(n) for n in names], ran


def test_steps_run_in_order() -> None:
    steps, ran = recording_steps(["tickers", "filings", "sections", "facts", "risks"])
    run_pipeline(SETTINGS, steps)
    assert ran == ["tickers", "filings", "sections", "facts", "risks"]


def test_failing_step_stops_the_pipeline() -> None:
    steps, ran = recording_steps(["tickers", "filings", "sections"], fail_at="filings")
    with pytest.raises(PipelineError, match=r"step 2 \(filings\) failed: boom"):
        run_pipeline(SETTINGS, steps)
    assert ran == ["tickers", "filings"]  # sections never ran


def test_no_llm_skips_the_risk_step() -> None:
    assert [name for name, _ in build_steps(no_llm=True)] == [
        "tickers",
        "filings",
        "sections",
        "facts",
    ]
    assert [name for name, _ in build_steps(no_llm=False)][-1] == "risks"


class BudgetGoneClient:
    def complete_json(self, *args: object, **kwargs: object) -> None:
        raise DailyBudgetExceeded("190000 tokens used")


def test_extraction_stops_cleanly_at_daily_budget(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, sec_user_agent="Test test@example.com", data_dir=tmp_path)
    settings.sections_dir.mkdir(parents=True)
    for ticker in ("AAA", "BBB"):
        (settings.sections_dir / f"{ticker}_1A.txt").write_text("Some risk text.")
    run = run_extraction(settings, ["AAA", "BBB"], BudgetGoneClient())
    assert run.stopped == "190000 tokens used"
    assert run.done == []
    assert run.failed == {}  # a budget stop is not a failure; rerun later resumes
