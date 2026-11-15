"""Dense search through Qdrant in local mode (in-process: no server, no Docker).

Same vectors as the NumPy version, but stored in a vector database with metadata filters
(company, fiscal year, section) applied by Qdrant itself. Local mode searches exactly
(brute force), so results must match NumPy; `--check` proves it on the eval questions.

    uv run python -m filing_analyst.retrieval.store --check
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from filing_analyst.config import Settings
from filing_analyst.evaluation.retrieval_eval import load_items
from filing_analyst.parse.chunk import Chunk
from filing_analyst.retrieval.dense import DenseIndex, embed_query, load_embedder
from filing_analyst.retrieval.dense import load_or_build_index as load_numpy_index

UPSERT_BATCH = 256


class QdrantIndex:
    """Drop-in for DenseIndex: same `chunks` and `search`, plus year and section filters."""

    def __init__(self, chunks: list[Chunk], client: QdrantClient, collection: str) -> None:
        self.chunks, self.client, self.collection = chunks, client, collection
        self.rebuilt = False  # set by load_or_build_qdrant; useful in tests and logs

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        ticker: str | None = None,
        fiscal_year: int | None = None,
        section: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        wanted = {"ticker": ticker.upper() if ticker else None, "fiscal_year": fiscal_year,
                  "section": section}  # fmt: skip
        conditions = [
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in wanted.items()
            if value is not None
        ]
        response = self.client.query_points(
            self.collection,
            query=query_vector.tolist(),
            query_filter=models.Filter(must=conditions) if conditions else None,
            limit=k,
            with_payload=False,
        )
        return [(self.chunks[int(p.id)], p.score) for p in response.points]


def build_collection(
    client: QdrantClient, collection: str, chunks: list[Chunk], vectors: np.ndarray
) -> None:
    """(Re)create the collection: point id = chunk position, payload = filterable metadata."""
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE),
    )
    for start in range(0, len(chunks), UPSERT_BATCH):
        batch = range(start, min(start + UPSERT_BATCH, len(chunks)))
        client.upsert(
            collection,
            points=[
                models.PointStruct(
                    id=i,
                    vector=vectors[i].tolist(),
                    payload={
                        "chunk_id": chunks[i].chunk_id,
                        "ticker": chunks[i].ticker,
                        "fiscal_year": chunks[i].fiscal_year,
                        "section": chunks[i].section,
                    },
                )
                for i in batch
            ],
        )


def vectors_fingerprint(vectors: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(vectors).tobytes()).hexdigest()[:16]


def load_or_build_qdrant(
    client: QdrantClient, meta_path: Path, dense: DenseIndex, name: str = "fixed"
) -> QdrantIndex:
    """Load the vectors into Qdrant once; reuse the collection while the vectors are unchanged."""
    collection = f"chunks_{name}"
    expected = {"fingerprint": vectors_fingerprint(dense.vectors), "count": len(dense.chunks)}
    index = QdrantIndex(dense.chunks, client, collection)
    fresh = meta_path.exists() and json.loads(meta_path.read_text()) == expected
    if not (fresh and client.collection_exists(collection)):
        build_collection(client, collection, dense.chunks, dense.vectors)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(expected))
        index.rebuilt = True
    return index


def open_qdrant(settings: Settings, dense: DenseIndex, name: str = "fixed") -> QdrantIndex:
    client = QdrantClient(path=str(settings.data_dir / "qdrant"))
    return load_or_build_qdrant(client, settings.data_dir / "qdrant" / f"{name}.json", dense, name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qdrant local index for 10-K chunks.")
    parser.add_argument("--check", action="store_true", help="compare with NumPy on eval set")
    args = parser.parse_args()

    settings = Settings()
    embedder = load_embedder()
    numpy_index = load_numpy_index(settings, make_embedder=lambda: embedder)
    qdrant = open_qdrant(settings, numpy_index)
    print(f"qdrant collection {qdrant.collection}: {len(qdrant.chunks):,} points, "
          f"{'rebuilt' if qdrant.rebuilt else 'reused'}")  # fmt: skip
    if not args.check:
        return

    items = load_items(settings)
    for label, use_filter in (("no filter", False), ("company filter", True)):
        same = 0
        for item in items:
            vector = embed_query(item.question, embedder)
            ticker = item.ticker if use_filter else None
            a = [c.chunk_id for c, _ in numpy_index.search(vector, k=10, ticker=ticker)]
            b = [c.chunk_id for c, _ in qdrant.search(vector, k=10, ticker=ticker)]
            same += a == b
            if a != b:
                print(f"  differs: {item.qid} numpy {a[:3]} vs qdrant {b[:3]}")
        print(f"{label}: identical top 10 for {same}/{len(items)} questions")


if __name__ == "__main__":
    main()
