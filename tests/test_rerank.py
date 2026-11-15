import math

import numpy as np
import pytest

from filing_analyst.parse.chunk import Chunk
from filing_analyst.retrieval.rerank import RerankedSearcher, rerank


def chunk(i: int, text: str, ticker: str = "AVGO") -> Chunk:
    return Chunk(
        chunk_id=f"{ticker}-2025-1A-{i:04d}",
        ticker=ticker,
        fiscal_year=2025,
        section="1A",
        text=text,
        token_count=len(text.split()),
    )


class OverlapScorer:
    """Fake cross-encoder: score = number of question words found in the passage."""

    def __init__(self) -> None:
        self.pairs_seen = 0

    def predict(self, sentences: list[tuple[str, str]], **kwargs: object) -> np.ndarray:
        self.pairs_seen += len(sentences)
        return np.array(
            [sum(w in p.lower().split() for w in q.lower().split()) for q, p in sentences],
            dtype=float,
        )


CHUNKS = [
    chunk(0, "weather risk and offices"),
    chunk(1, "wafers produced by tsmc"),
    chunk(2, "tsmc supplies wafers and tsmc capacity matters"),
]


def test_rerank_puts_the_best_pair_first_and_caps_k() -> None:
    results = rerank("tsmc wafers capacity", CHUNKS, OverlapScorer(), k=2)
    assert [c.chunk_id for c, _ in results] == ["AVGO-2025-1A-0002", "AVGO-2025-1A-0001"]


def test_rerank_of_nothing_is_nothing() -> None:
    assert rerank("anything", [], OverlapScorer(), k=5) == []


class NanScorer:
    def predict(self, sentences: list[tuple[str, str]], **kwargs: object) -> np.ndarray:
        return np.full(len(sentences), math.nan)


def test_nan_scores_are_refused_not_silently_ranked() -> None:
    """Real case, Nov 15: the model read from the memory-mapped file returned NaN."""
    with pytest.raises(ValueError, match="NaN"):
        rerank("tsmc", CHUNKS, NanScorer(), k=2)


class FakeHybrid:
    def __init__(self) -> None:
        self.asked: list[tuple[int, str | None]] = []

    def search(self, query: str, k: int = 5, ticker: str | None = None) -> list:
        self.asked.append((k, ticker))

        class Result:
            def __init__(self, c: Chunk) -> None:
                self.chunk = c

        return [Result(c) for c in CHUNKS]


def test_reranked_searcher_asks_hybrid_for_a_pool_and_keeps_the_filter() -> None:
    hybrid, scorer = FakeHybrid(), OverlapScorer()
    results = RerankedSearcher(hybrid, scorer, candidates=20).search("tsmc", k=1, ticker="avgo")
    assert hybrid.asked == [(20, "avgo")]
    assert scorer.pairs_seen == len(CHUNKS)  # every candidate is re-scored
    assert len(results) == 1


@pytest.mark.network
def test_real_reranker_scores_are_finite_and_sensible() -> None:
    from filing_analyst.retrieval.rerank import load_reranker

    try:
        scorer = load_reranker()
    except Exception:
        pytest.skip("could not download the cross-encoder")
    scores = scorer.predict(
        [
            ("Which companies depend on TSMC?", "About 95% of our wafers were produced by TSMC."),
            ("Which companies depend on TSMC?", "Severe weather could disrupt our offices."),
        ]
    )
    assert np.isfinite(scores).all()
    assert scores[0] > scores[1]
