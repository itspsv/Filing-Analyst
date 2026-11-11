"""BM25 keyword search, from scratch.

A chunk scores high when it contains the question's words: often (with diminishing returns),
rare words count more than common ones, and short chunks get a small boost.

    uv run python -m filing_analyst.retrieval.bm25 "TSMC foundry"
"""

import argparse
import math
import re
import time
from collections import Counter

import numpy as np

from filing_analyst.config import Settings
from filing_analyst.parse.chunk import Chunk, load_chunks
from filing_analyst.retrieval.dense import best_k

K1 = 1.5  # how fast repeated words stop adding score (common default)
B = 0.75  # how much long chunks are penalised (0 = not at all, 1 = fully; common default)
WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase letters and digits only: "NVIDIA's H200" -> ["nvidia", "s", "h200"]."""
    return WORD.findall(text.lower())


def idf(n_docs: int, doc_freq: int) -> float:
    """Inverse document frequency: rare words weigh more. The 1 + keeps it positive."""
    return math.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = K1, b: float = B) -> None:
        self.chunks, self.k1, self.b = chunks, k1, b
        counts = [Counter(tokenize(c.text)) for c in chunks]
        self.doc_len = np.array([sum(c.values()) for c in counts], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if chunks else 0.0
        self.tickers = np.array([c.ticker for c in chunks])

        # Inverted index: word -> (which chunks contain it, how often in each).
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for i, doc in enumerate(counts):
            for word, tf in doc.items():
                ids, tfs = postings.setdefault(word, ([], []))
                ids.append(i)
                tfs.append(tf)
        self.postings = {
            word: (np.array(ids), np.array(tfs, dtype=np.float32))
            for word, (ids, tfs) in postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every chunk for the query (0 for chunks sharing no word with it)."""
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for word in set(tokenize(query)):  # a word repeated in the question counts once
            if word not in self.postings:
                continue
            ids, tf = self.postings[word]
            length_norm = 1 - self.b + self.b * self.doc_len[ids] / self.avg_len
            weight = idf(len(self.chunks), len(ids))
            scores[ids] += weight * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
        return scores

    def search(
        self, query: str, k: int = 5, ticker: str | None = None
    ) -> list[tuple[Chunk, float]]:
        scores = self.scores(query)
        mask = scores > 0  # no shared word = not a match at all
        if ticker is not None:
            mask &= self.tickers == ticker.upper()
        return [(self.chunks[i], s) for i, s in best_k(scores, k, mask)]


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 keyword search over 10-K chunks.")
    parser.add_argument("query")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    chunks = load_chunks(Settings().chunks_dir / "fixed.jsonl")
    started = time.perf_counter()
    index = BM25Index(chunks)
    built = time.perf_counter() - started
    print(f"index: {len(chunks):,} chunks, {len(index.postings):,} distinct words, {built:.2f} s")
    started = time.perf_counter()
    results = index.search(args.query, k=args.k, ticker=args.ticker)
    print(f"search took {(time.perf_counter() - started) * 1000:.1f} ms\n")
    for chunk, score in results:
        snippet = " ".join(chunk.text.split())[:160]
        print(f"{score:6.2f}  {chunk.chunk_id}\n        {snippet}...")


if __name__ == "__main__":
    main()
