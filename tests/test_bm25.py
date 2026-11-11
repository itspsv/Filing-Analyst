import math

import pytest

from filing_analyst.parse.chunk import Chunk
from filing_analyst.retrieval.bm25 import BM25Index, tokenize


def chunks(*texts: str, ticker: str = "ACME") -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{ticker}-2025-1A-{i:04d}",
            ticker=ticker,
            fiscal_year=2025,
            section="1A",
            text=text,
            token_count=len(text.split()),
        )
        for i, text in enumerate(texts)
    ]


def test_tokenize_lowercases_and_keeps_codes() -> None:
    assert tokenize("NVIDIA's H200, built at 7 nm!") == [
        "nvidia",
        "s",
        "h200",
        "built",
        "at",
        "7",
        "nm",
    ]


def test_score_matches_bm25_formula_worked_by_hand() -> None:
    # 3 chunks of 3, 6 and 2 words: average length 11/3. "tsmc" is in 2 of the 3 chunks.
    index = BM25Index(
        chunks("tsmc makes wafers", "apple buys wafers from tsmc tsmc", "weather risk")
    )
    idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))  # = ln(1.6)
    length_norm = 1 - 0.75 + 0.75 * 3 / (11 / 3)  # chunk 0 has 3 words
    expected = idf * 1 * (1.5 + 1) / (1 + 1.5 * length_norm)  # tf = 1
    assert index.scores("tsmc")[0] == pytest.approx(expected, rel=1e-6)
    assert index.scores("tsmc")[2] == 0  # no shared word


def test_repeating_a_word_has_diminishing_returns() -> None:
    once = "tsmc " + "x " * 9
    ten_times = "tsmc " * 10
    filler = "other words " * 20  # so "tsmc" is a rare word with a positive weight
    index = BM25Index(chunks(once, ten_times, filler))
    s_once, s_ten = index.scores("tsmc")[:2]
    assert s_ten > s_once
    assert s_ten < 2.5 * s_once  # 10x the mentions, far less than 10x the score


def test_shorter_chunk_wins_for_same_count() -> None:
    index = BM25Index(chunks("tsmc wafers", "tsmc wafers " + "and more words " * 10, "other"))
    results = index.search("tsmc", k=2)
    assert results[0][0].chunk_id == "ACME-2025-1A-0000"


def test_unknown_word_gives_no_results() -> None:
    index = BM25Index(chunks("tsmc makes wafers", "weather risk"))
    assert index.search("blockchain") == []


def test_ticker_filter() -> None:
    index = BM25Index(
        chunks("export controls", ticker="NVDA") + chunks("export rules", ticker="AAPL")
    )
    results = index.search("export", k=5, ticker="aapl")
    assert [c.ticker for c, _ in results] == ["AAPL"]
