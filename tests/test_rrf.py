import numpy as np
import pytest

from filing_analyst.parse.chunk import Chunk
from filing_analyst.retrieval.bm25 import BM25Index
from filing_analyst.retrieval.dense import DenseIndex, normalize
from filing_analyst.retrieval.rrf import HybridSearcher, rrf


def test_fused_scores_match_hand_calculation() -> None:
    fused = dict(rrf([["a", "b", "c"], ["b", "c", "d"]]))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)  # rank 2, then rank 1
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 62)
    assert fused["a"] == pytest.approx(1 / 61)  # only in the first list
    assert fused["d"] == pytest.approx(1 / 63)  # only in the second list


def test_items_liked_by_both_lists_win() -> None:
    order = [item for item, _ in rrf([["a", "b", "c"], ["b", "c", "d"]])]
    assert order == ["b", "c", "a", "d"]  # a was #1 once, but b and c were found twice


def test_empty_input_gives_empty_output() -> None:
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_ties_keep_a_stable_order() -> None:
    first = rrf([["a", "b"], ["b", "a"]])  # a and b get exactly the same points
    assert [item for item, _ in first] == ["a", "b"]
    assert rrf([["a", "b"], ["b", "a"]]) == first


def test_duplicate_inside_one_list_counts_once() -> None:
    fused = dict(rrf([["a", "a", "b"]]))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)  # b is the 2nd distinct item, not the 3rd


WORDS = ["export", "tsmc", "weather"]


class KeywordEmbedder:
    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        return np.array([[s.lower().count(w) + 0.01 for w in WORDS] for s in sentences])


def make_chunks() -> list[Chunk]:
    texts = [
        ("NVDA", "export controls on chips"),
        ("AMD", "wafers made by tsmc"),
        ("AMD", "tsmc tsmc capacity and export rules"),
        ("TXN", "weather risk"),
    ]
    return [
        Chunk(
            chunk_id=f"{t}-2025-1A-{i:04d}",
            ticker=t,
            fiscal_year=2025,
            section="1A",
            text=text,
            token_count=len(text.split()),
        )
        for i, (t, text) in enumerate(texts)
    ]


def make_searcher() -> HybridSearcher:
    chunks, embedder = make_chunks(), KeywordEmbedder()
    dense = DenseIndex(chunks, normalize(embedder.encode([c.text for c in chunks])))
    return HybridSearcher(dense, BM25Index(chunks), embedder)


def test_hybrid_reports_rank_in_each_method() -> None:
    results = make_searcher().search("tsmc", k=2)
    assert {r.chunk.ticker for r in results} == {"AMD"}
    for r in results:
        assert set(r.ranks) == {"dense", "bm25"}
        assert r.ranks["bm25"] in (1, 2)


def test_hybrid_ticker_filter() -> None:
    results = make_searcher().search("export", k=5, ticker="nvda")
    assert results
    assert {r.chunk.ticker for r in results} == {"NVDA"}


def test_refuses_indexes_built_from_different_chunks() -> None:
    chunks, embedder = make_chunks(), KeywordEmbedder()
    dense = DenseIndex(chunks, normalize(embedder.encode([c.text for c in chunks])))
    with pytest.raises(ValueError, match="different chunks"):
        HybridSearcher(dense, BM25Index(chunks[:2]), embedder)
