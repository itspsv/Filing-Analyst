"""Dense (meaning-based) search, from scratch in NumPy.

Each chunk becomes a 384-number vector (bge-small). Vectors are normalised to length 1, so the
cosine similarity between a question and every chunk is one matrix-vector product.

    uv run python -m filing_analyst.retrieval.dense "export controls" --ticker NVDA
"""

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from filing_analyst.config import Settings
from filing_analyst.parse.chunk import EMBEDDING_MODEL, Chunk, load_chunks

# From the bge model card: add this to questions (not to passages) for retrieval.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 32


class Embedder(Protocol):
    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray: ...


def load_embedder() -> Embedder:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale every row to length 1, so a dot product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.clip(norms, 1e-12, None)).astype(np.float32)


def embed_texts(texts: list[str], embedder: Embedder) -> np.ndarray:
    vectors = embedder.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False)
    return normalize(np.asarray(vectors, dtype=np.float32))


def embed_query(query: str, embedder: Embedder) -> np.ndarray:
    return embed_texts([QUERY_PREFIX + query], embedder)[0]


def best_k(scores: np.ndarray, k: int, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
    """Indices and scores of the k highest scores, best first. Rows where mask is False skip.
    Shared by dense and BM25 search, so both pick winners the same way."""
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    k = min(k, int(np.isfinite(scores).sum()))
    if k == 0:
        return []
    best = np.argpartition(-scores, k - 1)[:k]  # the k best, in any order (fast)
    best = best[np.argsort(-scores[best], kind="stable")]  # then sort just those k
    return [(int(i), float(scores[i])) for i in best]


def top_k(
    query_vector: np.ndarray, vectors: np.ndarray, k: int, mask: np.ndarray | None = None
) -> list[tuple[int, float]]:
    """Indices and cosine scores of the k rows most similar to the query, best first."""
    return best_k(vectors @ query_vector, k, mask)


@dataclass
class DenseIndex:
    chunks: list[Chunk]
    vectors: np.ndarray  # shape (len(chunks), 384), rows of length 1

    def search(
        self, query_vector: np.ndarray, k: int = 5, ticker: str | None = None
    ) -> list[tuple[Chunk, float]]:
        mask = None
        if ticker is not None:
            mask = np.array([c.ticker == ticker.upper() for c in self.chunks])
        return [(self.chunks[i], s) for i, s in top_k(query_vector, self.vectors, k, mask)]


def fingerprint(data: bytes) -> str:
    return hashlib.sha256(data + EMBEDDING_MODEL.encode()).hexdigest()[:16]


def load_or_build_index(
    settings: Settings,
    name: str = "fixed",
    make_embedder: Callable[[], Embedder] = load_embedder,
) -> DenseIndex:
    """Embed the chunks once; reuse the saved vectors while the chunk file is unchanged."""
    chunks_path = settings.chunks_dir / f"{name}.jsonl"
    chunks = load_chunks(chunks_path)
    vectors_path = settings.embeddings_dir / f"{name}.npy"
    meta_path = settings.embeddings_dir / f"{name}.json"
    expected = {"fingerprint": fingerprint(chunks_path.read_bytes()), "count": len(chunks)}

    if meta_path.exists() and json.loads(meta_path.read_text()) == expected:
        return DenseIndex(chunks, np.load(vectors_path))

    started = time.perf_counter()
    vectors = embed_texts([c.text for c in chunks], make_embedder())
    print(f"embedded {len(chunks):,} chunks in {time.perf_counter() - started:.1f} s")
    settings.embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, vectors)
    meta_path.write_text(json.dumps(expected))
    return DenseIndex(chunks, vectors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense search over 10-K chunks.")
    parser.add_argument("query")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    settings = Settings()
    embedder = load_embedder()  # loaded once: used for the chunks (if needed) and the query
    index = load_or_build_index(settings, make_embedder=lambda: embedder)
    started = time.perf_counter()
    results = index.search(embed_query(args.query, embedder), k=args.k, ticker=args.ticker)
    print(f"search took {(time.perf_counter() - started) * 1000:.0f} ms (embed query + scan)\n")
    for chunk, score in results:
        snippet = " ".join(chunk.text.split())[:160]
        print(f"{score:.3f}  {chunk.chunk_id}\n       {snippet}...")


if __name__ == "__main__":
    main()
