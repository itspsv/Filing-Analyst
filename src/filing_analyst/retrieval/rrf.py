"""Reciprocal Rank Fusion (RRF): merge ranked lists using ranks, not scores.

BM25 scores (~20) and cosine scores (~0.8) live on different scales and cannot be added. RRF
gives every chunk 1 / (60 + rank) points per list and adds them up, so chunks that both
methods rank highly rise to the top, and a chunk found by only one method can still make it.

    uv run python -m filing_analyst.retrieval.rrf "Which companies depend on TSMC?"
"""

import argparse
import time
from dataclasses import dataclass

from filing_analyst.config import Settings
from filing_analyst.parse.chunk import Chunk
from filing_analyst.retrieval.bm25 import BM25Index
from filing_analyst.retrieval.dense import (
    DenseIndex,
    Embedder,
    embed_query,
    load_embedder,
    load_or_build_index,
)

RRF_K = 60  # standard constant from Cormack, Clarke and Buettcher (2009)
CANDIDATES = 50  # how deep each method's list goes before fusing


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse ranked lists of ids (best first). Returns (id, score), best first.
    An id repeated inside one list counts once, at its best rank. Ties keep first-seen order."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        unique = list(dict.fromkeys(ranking))
        for rank, item in enumerate(unique, start=1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda pair: -pair[1])  # sorted() is stable


@dataclass
class HybridResult:
    chunk: Chunk
    score: float
    ranks: dict[str, int | None]  # method -> 1-based rank in its list, None if not in it


class HybridSearcher:
    def __init__(self, dense: DenseIndex, bm25: BM25Index, embedder: Embedder) -> None:
        dense_ids = [c.chunk_id for c in dense.chunks]
        if dense_ids != [c.chunk_id for c in bm25.chunks]:
            raise ValueError("dense and BM25 indexes were built from different chunks")
        self.dense, self.bm25, self.embedder = dense, bm25, embedder

    def search(
        self, query: str, k: int = 5, ticker: str | None = None, candidates: int = CANDIDATES
    ) -> list[HybridResult]:
        query_vector = embed_query(query, self.embedder)
        hits = {
            "dense": self.dense.search(query_vector, k=candidates, ticker=ticker),
            "bm25": self.bm25.search(query, k=candidates, ticker=ticker),
        }
        ranked_ids = {name: [c.chunk_id for c, _ in found] for name, found in hits.items()}
        chunks_by_id = {c.chunk_id: c for found in hits.values() for c, _ in found}
        results = []
        for chunk_id, score in rrf(list(ranked_ids.values()))[:k]:
            ranks = {
                name: ids.index(chunk_id) + 1 if chunk_id in ids else None
                for name, ids in ranked_ids.items()
            }
            results.append(HybridResult(chunks_by_id[chunk_id], score, ranks))
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid search (dense + BM25, fused by RRF).")
    parser.add_argument("query")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    embedder = load_embedder()
    dense = load_or_build_index(Settings(), make_embedder=lambda: embedder)
    searcher = HybridSearcher(dense, BM25Index(dense.chunks), embedder)
    started = time.perf_counter()
    results = searcher.search(args.query, k=args.k, ticker=args.ticker)
    print(f"search took {(time.perf_counter() - started) * 1000:.0f} ms\n")
    for r in results:
        ranks = "  ".join(
            f"{name} #{rank}" if rank else f"{name} -" for name, rank in r.ranks.items()
        )
        snippet = " ".join(r.chunk.text.split())[:150]
        print(f"{r.score:.4f}  {r.chunk.chunk_id:<20} ({ranks})\n        {snippet}...")


if __name__ == "__main__":
    main()
