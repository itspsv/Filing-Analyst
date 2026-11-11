"""The whole Week 1 data pipeline in one command.

    uv run python -m filing_analyst.pipeline            # all 5 steps
    uv run python -m filing_analyst.pipeline --no-llm   # data steps only, no tokens spent

Every step reuses what is already on disk, so a rerun is fast and safe.
"""

import argparse
import sys
import time
from collections.abc import Callable

from filing_analyst.config import COMPANIES, Settings
from filing_analyst.db import write_facts
from filing_analyst.extract import log_length, print_run_summary, run_extraction
from filing_analyst.llm import LLMClient
from filing_analyst.parse.sections import extract_all_sections, missing_sections
from filing_analyst.sec.filings import download_all, write_manifest
from filing_analyst.sec.tickers import load_ticker_table, lookup
from filing_analyst.sec.xbrl import coverage_gaps, load_all_facts

Step = tuple[str, Callable[[Settings], str]]


class PipelineError(RuntimeError):
    pass


def step_tickers(settings: Settings) -> str:
    companies = lookup(COMPANIES, load_ticker_table(settings))
    return f"{len(companies)}/{len(COMPANIES)} companies mapped to CIKs"


def step_filings(settings: Settings) -> str:
    results = download_all(settings)
    write_manifest([f for f, _ in results], settings.raw_dir / "10k" / "manifest.json")
    downloaded = sum(not cached for _, cached in results)
    return f"{len(results)} 10-Ks ({downloaded} downloaded, {len(results) - downloaded} cached)"


def step_sections(settings: Settings) -> str:
    results = extract_all_sections(settings)
    missing = missing_sections(results)
    total = sum(len(sizes) for sizes in results.values())
    summary = f"{total - len(missing)}/{total} sections found"
    return f"{summary}; missing: {', '.join(missing)}" if missing else summary


def step_facts(settings: Settings) -> str:
    facts = load_all_facts(settings)
    rows = write_facts(facts, settings.facts_db)
    return f"{rows:,} rows in DuckDB, {len(coverage_gaps(facts))} known gaps in latest year"


def step_risks(settings: Settings) -> str:
    client = LLMClient(settings)
    log_start = log_length(client)
    run = run_extraction(settings, list(COMPANIES), client)
    print_run_summary(client, log_start)
    summary = f"{len(run.done)}/{len(COMPANIES)} companies done, {len(run.failed)} failed"
    return f"{summary}; stopped early at the daily budget, rerun later" if run.stopped else summary


def build_steps(no_llm: bool) -> list[Step]:
    steps: list[Step] = [
        ("tickers", step_tickers),
        ("filings", step_filings),
        ("sections", step_sections),
        ("facts", step_facts),
    ]
    return steps if no_llm else [*steps, ("risks", step_risks)]


def run_pipeline(settings: Settings, steps: list[Step]) -> None:
    """Run steps in order; stop at the first one that raises."""
    for number, (name, step) in enumerate(steps, start=1):
        print(f"{number}. {name} ...", flush=True)
        started = time.perf_counter()
        try:
            summary = step(settings)
        except Exception as err:
            raise PipelineError(f"step {number} ({name}) failed: {err}") from err
        print(f"{number}. {name}: {summary} ({time.perf_counter() - started:.1f} s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Filing Analyst data pipeline.")
    parser.add_argument("--no-llm", action="store_true", help="skip the risk extraction step")
    args = parser.parse_args()
    try:
        run_pipeline(Settings(), build_steps(args.no_llm))
    except PipelineError as err:
        print(f"\nPIPELINE STOPPED: {err}")
        sys.exit(1)
    print("\npipeline finished")


if __name__ == "__main__":
    main()
