import pytest

from filing_analyst.evaluation.retrieval_eval import (
    evaluate,
    groups,
    measure_latency,
    score,
    summarize,
    unreachable,
)
from filing_analyst.evaluation.retrieval_set import RetrievalItem
from filing_analyst.parse.chunk import Chunk


def chunk(ticker: str, i: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{ticker}-2025-1A-{i:04d}",
        ticker=ticker,
        fiscal_year=2025,
        section="1A",
        text=text,
        token_count=len(text.split()),
    )


def item(qid: str, quotes: list[str], style: str = "literal", ticker: str = "AVGO",
         question: str = "Q?") -> RetrievalItem:  # fmt: skip
    return RetrievalItem(
        qid=qid,
        ticker=ticker,
        section="1A",
        style=style,
        question=question,
        quotes=quotes,
        source_chunk_id="x",
    )


ANSWER = chunk("AVGO", 1, "about 95% of the wafers were produced by TSMC last year")
FILLER = [chunk("AVGO", i, f"unrelated text number {i}") for i in range(10, 20)]
TSMC = "95% of the wafers were produced by TSMC"


def test_hit_at_rank_3_gives_recall_1_and_rr_one_third() -> None:
    result = score(item("q1", [TSMC]), [FILLER[0], FILLER[1], ANSWER, FILLER[2]], 12.0)
    assert result.recall == 1.0
    assert result.reciprocal_rank == pytest.approx(1 / 3)
    assert result.first_hit_rank == 3


def test_hit_at_rank_7_counts_for_mrr_but_not_recall_at_5() -> None:
    ranked = [*FILLER[:6], ANSWER]
    result = score(item("q1", [TSMC]), ranked, 1.0)
    assert result.recall == 0.0
    assert result.reciprocal_rank == pytest.approx(1 / 7)


def test_same_text_from_another_company_is_not_relevant() -> None:
    copycat = chunk("AMD", 2, ANSWER.text)
    assert score(item("q1", [TSMC]), [copycat], 1.0).recall == 0.0


def test_two_quotes_one_found_gives_recall_half() -> None:
    result = score(item("q1", [TSMC, "a sentence that is nowhere"]), [ANSWER], 1.0)
    assert result.recall == 0.5


def test_summary_averages_and_groups_by_style_and_item_mention() -> None:
    items = [
        item("q1", [TSMC], style="literal"),
        item("q2", [TSMC], style="paraphrase", question="Per Item 1A, who makes the wafers?"),
    ]
    answers = {"Q?": [ANSWER], "Per Item 1A, who makes the wafers?": [FILLER[0], ANSWER]}
    results = evaluate(items, lambda q, ticker: answers[q])
    by_group = {name: summarize(rs) for name, rs in groups(results).items()}
    assert by_group["all"]["recall@5"] == 1.0
    assert by_group["all"]["mrr@10"] == pytest.approx((1 + 1 / 2) / 2, abs=1e-3)
    assert by_group["literal"]["questions"] == 1
    assert by_group["names an Item"]["mrr@10"] == 0.5


def test_company_filter_passes_the_question_ticker_to_search() -> None:
    asked: list[str | None] = []

    def search(question: str, ticker: str | None) -> list:
        asked.append(ticker)
        return [ANSWER]

    evaluate([item("q1", [TSMC])], search, company_filter=False)
    evaluate([item("q1", [TSMC])], search, company_filter=True)
    assert asked == [None, "AVGO"]


def test_quote_cut_by_a_chunk_boundary_is_reported_as_unreachable() -> None:
    halves = [
        chunk("AVGO", 1, "about 95% of the wafers"),
        chunk("AVGO", 2, "were produced by TSMC"),
    ]
    assert unreachable([item("q1", [TSMC])], halves) == ["q1"]
    assert unreachable([item("q1", [TSMC])], [ANSWER]) == []


def test_latency_is_measured_for_every_question_and_pass() -> None:
    calls: list[str | None] = []
    items = [item("q1", [TSMC]), item("q2", [TSMC])]
    lat = measure_latency(items, lambda q, t: calls.append(t) or [], repeats=3)
    assert lat["searches"] == 6 and len(calls) == 6
    assert set(calls) == {"AVGO"}  # timed with the company filter, as the agent searches
    assert 0 <= lat["p50_ms"] <= lat["p95_ms"]
