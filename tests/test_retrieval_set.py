import csv
from pathlib import Path

import pytest
from pydantic import BaseModel

from filing_analyst.config import COMPANIES
from filing_analyst.evaluation.retrieval_set import (
    DraftQuestion,
    contains_quote,
    draft_items,
    quote_problem,
    read_reviewed,
    sample_chunks,
    write_review_csv,
)
from filing_analyst.llm import DailyBudgetExceeded, LLMError, Messages
from filing_analyst.parse.chunk import Chunk

PROSE = (
    "During fiscal year 2025, approximately 95% of the wafers manufactured by our contract "
    "manufacturers were produced by TSMC, and we depend on a limited number of suppliers. "
) * 8


def chunk(ticker: str, section: str, i: int, text: str = PROSE, tokens: int = 300) -> Chunk:
    return Chunk(
        chunk_id=f"{ticker}-2025-{section}-{i:04d}",
        ticker=ticker,
        fiscal_year=2025,
        section=section,
        text=text,
        token_count=tokens,
    )


def test_quote_matching_ignores_curly_quotes_case_and_spacing() -> None:
    source = "Our  “key” supplier’s capacity\nis limited."
    assert contains_quote(source, 'our "key" supplier\'s capacity is limited')
    assert not contains_quote(source, "our key supplier has unlimited capacity")


def test_quote_problems_are_named() -> None:
    assert quote_problem("produced by TSMC, and we depend on", PROSE) is None
    assert "not found" in quote_problem("produced by Samsung in Korea last year", PROSE)
    assert "words" in quote_problem("TSMC", PROSE)  # too short to be a specific label


def test_sampling_is_repeatable_and_skips_tables_and_tails() -> None:
    chunks = []
    for t in COMPANIES:
        chunks += [chunk(t, "1A", i) for i in range(5)] + [chunk(t, "7", i) for i in range(3)]
        chunks.append(chunk(t, "1A", 90, tokens=55))  # short tail chunk
        chunks.append(chunk(t, "7", 91, text="12 345 678 " * 50))  # table numbers
    first = sample_chunks(chunks)
    assert first == sample_chunks(chunks)  # same seed, same pick
    assert len(first) == 3 * len(COMPANIES)
    assert not any(c.chunk_id.endswith(("0090", "0091")) for c in first)
    nvda = [c.section for c in first if c.ticker == "NVDA"]
    assert sorted(nvda) == ["1A", "1A", "7"]


class FakeDrafter:
    """Returns prepared drafts in order and remembers the styles it was asked for."""

    def __init__(self, drafts: list[DraftQuestion | Exception]) -> None:
        self.drafts = drafts
        self.styles: list[str] = []

    def complete_json[T: BaseModel](
        self, messages: Messages, task: str, response_model: type[T], max_tokens: int = 0
    ) -> T:
        self.styles.append(messages[1]["content"].split("Style: ")[1].split(".")[0])
        draft = self.drafts[len(self.styles) - 1]
        if isinstance(draft, Exception):
            raise draft
        return draft


def test_drafts_with_bad_quotes_or_unusable_passages_are_dropped() -> None:
    chunks = [chunk("AVGO", "1A", i) for i in range(3)]
    client = FakeDrafter(
        [
            DraftQuestion(usable=True, question="How much does Broadcom rely on TSMC?",
                          quote="approximately 95% of the wafers manufactured"),
            DraftQuestion(usable=True, question="Q?", quote="a quote the model made up here"),
            DraftQuestion(usable=False, question="", quote=""),
        ]
    )  # fmt: skip
    kept, dropped = draft_items(chunks, client, workers=1)
    assert client.styles == ["literal", "paraphrase", "literal"]  # styles alternate
    assert [k.question for k in kept] == ["How much does Broadcom rely on TSMC?"]
    assert kept[0].qid == "q001" and kept[0].style == "literal"
    assert len(dropped) == 2 and "not found" in dropped[0] and "unusable" in dropped[1]


GOOD = DraftQuestion(usable=True, question="How much does Broadcom rely on TSMC?",
                     quote="approximately 95% of the wafers manufactured")  # fmt: skip


def test_one_failing_passage_is_dropped_not_fatal() -> None:
    """Real case, Nov 12: the model kept writing a 677-character quote; the run must go on."""
    chunks = [chunk("AVGO", "1A", i) for i in range(2)]
    client = FakeDrafter([LLMError("no valid answer after 3 tries"), GOOD])
    kept, dropped = draft_items(chunks, client, workers=1)
    assert len(kept) == 1
    assert dropped == ["AVGO-2025-1A-0000: no valid draft (no valid answer after 3 tries)"]


def test_daily_budget_still_stops_drafting() -> None:
    client = FakeDrafter([DailyBudgetExceeded("190000 tokens used"), GOOD])
    with pytest.raises(DailyBudgetExceeded):
        draft_items([chunk("AVGO", "1A", i) for i in range(2)], client, workers=1)


def mark_keep(path: Path, keep: dict[str, str], new_quote: dict[str, str] | None = None) -> None:
    rows = list(csv.DictReader(path.open(newline="")))
    for row in rows:
        row["keep"] = keep.get(row["qid"], "")
        row["quote"] = (new_quote or {}).get(row["qid"], row["quote"])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_review_sheet(tmp_path: Path) -> Path:
    chunks = [chunk("AVGO", "1A", 0), chunk("AVGO", "1A", 1)]
    client = FakeDrafter(
        [
            DraftQuestion(usable=True, question="Who makes Broadcom's wafers?",
                          quote="were produced by TSMC, and we depend on"),
            DraftQuestion(usable=True, question="Does Broadcom have few suppliers?",
                          quote="we depend on a limited number of suppliers"),
        ]
    )  # fmt: skip
    kept, _ = draft_items(chunks, client, workers=1)
    path = tmp_path / "review.csv"
    write_review_csv(kept, chunks, path)
    return path


SECTIONS = {("AVGO", "1A"): PROSE}


def test_review_round_trip_keeps_only_marked_rows(tmp_path: Path) -> None:
    path = make_review_sheet(tmp_path)
    mark_keep(path, {"q001": "y", "q002": "n"})
    items = read_reviewed(path, SECTIONS)
    assert [i.qid for i in items] == ["q001"]
    assert items[0].quotes == ["were produced by TSMC, and we depend on"]


def test_edited_quote_must_still_exist_in_filing(tmp_path: Path) -> None:
    path = make_review_sheet(tmp_path)
    mark_keep(path, {"q001": "y"}, new_quote={"q001": "produced by Samsung in Korea last year"})
    with pytest.raises(ValueError, match="q001: quote not found"):
        read_reviewed(path, SECTIONS)


def test_several_quotes_separated_by_pipe(tmp_path: Path) -> None:
    path = make_review_sheet(tmp_path)
    two = "were produced by TSMC, and we depend on | we depend on a limited number of suppliers"
    mark_keep(path, {"q001": "yes"}, new_quote={"q001": two})
    assert len(read_reviewed(path, SECTIONS)[0].quotes) == 2
