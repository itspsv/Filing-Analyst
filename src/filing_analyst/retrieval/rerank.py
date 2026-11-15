"""Cross-encoder re-ranking: re-score the top hybrid results with a slower, sharper model.

Dense search embeds question and passage separately and compares the vectors. A cross-encoder
reads the question and one passage together, so it can judge whether that passage really
answers the question. Too slow for 2,000+ chunks, so it only re-orders hybrid's top 20.

    uv run python -m filing_analyst.retrieval.rerank "Which companies depend on TSMC?"
"""

import argparse
import time
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch

from filing_analyst.config import Settings
from filing_analyst.parse.chunk import Chunk
from filing_analyst.retrieval.bm25 import BM25Index
from filing_analyst.retrieval.dense import load_embedder, load_or_build_index
from filing_analyst.retrieval.rrf import HybridSearcher

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"  # renamed from ...MiniLM-L-6-v2
RERANK_CANDIDATES = 20


class PairScorer(Protocol):
    def predict(self, sentences: list[tuple[str, str]], **kwargs: object) -> np.ndarray: ...


def load_reranker() -> PairScorer:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(RERANK_MODEL, device="cpu")
    # On this Mac (torch 2.14.1, transformers 5.22.2) the weights read straight from the
    # memory-mapped file give NaN scores or a bus error for this model. A one-time copy into
    # normal memory (~90 MB) fixes it; test_real_reranker_scores_are_finite guards it.
    with torch.no_grad():
        for param in model.model.parameters():
            param.data = param.data.clone()
    return model


def rerank(
    question: str, chunks: Sequence[Chunk], scorer: PairScorer, k: int
) -> list[tuple[Chunk, float]]:
    """The k chunks the cross-encoder scores highest for this question, best first."""
    if not chunks:
        return []
    scores = np.asarray(scorer.predict([(question, c.text) for c in chunks]), dtype=np.float32)
    if not np.isfinite(scores).all():
        raise ValueError("re-ranker returned NaN or infinite scores")
    order = np.argsort(-scores, kind="stable")[:k]
    return [(chunks[i], float(scores[i])) for i in order]


class RerankedSearcher:
    """Hybrid search for a candidate pool, then cross-encoder re-ranking of that pool."""

    def __init__(
        self, hybrid: HybridSearcher, scorer: PairScorer, candidates: int = RERANK_CANDIDATES
    ) -> None:
        self.hybrid, self.scorer, self.candidates = hybrid, scorer, candidates

    def search(
        self, query: str, k: int = 5, ticker: str | None = None
    ) -> list[tuple[Chunk, float]]:
        pool = [r.chunk for r in self.hybrid.search(query, k=self.candidates, ticker=ticker)]
        return rerank(query, pool, self.scorer, k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid search + cross-encoder re-ranking.")
    parser.add_argument("query")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    embedder = load_embedder()
    dense = load_or_build_index(Settings(), make_embedder=lambda: embedder)
    searcher = RerankedSearcher(HybridSearcher(dense, BM25Index(dense.chunks), embedder),
                                load_reranker())  # fmt: skip
    started = time.perf_counter()
    results = searcher.search(args.query, k=args.k, ticker=args.ticker)
    print(f"search + re-rank took {(time.perf_counter() - started) * 1000:.0f} ms\n")
    for chunk, score in results:
        snippet = " ".join(chunk.text.split())[:150]
        print(f"{score:7.3f}  {chunk.chunk_id}\n         {snippet}...")


if __name__ == "__main__":
    main()
