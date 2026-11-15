"""Cut section text into small, searchable chunks, with one of three strategies.

All three cap a chunk at 300 tokens, counted with the embedding model's own tokenizer
(bge-small reads at most 512 tokens per input), so the comparison is fair:
- fixed:     300-token windows with 50 tokens of overlap; cuts anywhere (but not mid-word).
- recursive: cut at paragraph breaks first, then sentences, then words, only as far as needed.
- section:   keep each heading with its paragraphs; a long block is split recursively and
             every piece starts with its heading, so it still says what the text is about.

    uv run python -m filing_analyst.parse.chunk --strategy recursive
"""

import argparse
import re
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol, get_args

from pydantic import BaseModel

from filing_analyst.config import COMPANIES, Settings
from filing_analyst.parse.sections import SECTIONS, load_manifest

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_TOKENS = 300
OVERLAP_TOKENS = 50
SEPARATORS = ("\n", ". ", " ")  # paragraph, sentence, word: coarsest first
BULLETS = ("•", "●", "◦", "▪", "-")
HEADING_CHARS = (20, 300)  # a heading is a short line followed by a long paragraph
MIN_PIECE_TOKENS = 50  # smaller section pieces are joined to the next one

Strategy = Literal["fixed", "recursive", "section"]

# Page furniture left over from the HTML: measured 482 "Table of Contents" lines, 551 bare page
# numbers and ~50 "Form 10-K" headers/footers across the 40 section files.
NOISE_LINES = [
    re.compile(r"^table of contents$", re.IGNORECASE),
    re.compile(r"^\d{1,3}$"),  # page numbers (also drops a few tiny table cells; fine for text)
    re.compile(r"\|.*form 10-k|form 10-k.*\|", re.IGNORECASE),  # "Apple Inc. | 2026 Form 10-K | 19"
    re.compile(r"^(fy\s*)?\d{4}\s+form 10-k$", re.IGNORECASE),  # "FY 2025 FORM 10-K"
]


class Chunk(BaseModel):
    chunk_id: str  # e.g. "AAPL-2026-1A-0007"; the retrieval test set points at these ids
    ticker: str
    fiscal_year: int
    section: Literal["1A", "7"]
    text: str
    token_count: int


class Encoding(Protocol):
    offsets: Sequence[tuple[int, int]]
    word_ids: Sequence[int | None]


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> Encoding: ...


def clean_text(text: str) -> str:
    """Drop page headers, footers and page numbers; keep everything else."""
    kept = [line for line in text.splitlines() if not any(p.search(line) for p in NOISE_LINES)]
    return "\n".join(kept)


def _word_start(word_ids: Sequence[int | None], i: int) -> int:
    """Move i left until it is the first token of a word, so no word is cut in half."""
    while 0 < i < len(word_ids) and word_ids[i] is not None and word_ids[i] == word_ids[i - 1]:
        i -= 1
    return i


def token_windows(word_ids: Sequence[int | None], size: int, overlap: int) -> list[tuple[int, int]]:
    """(start, end) token ranges of at most `size` tokens, each overlapping the last by ~overlap."""
    windows: list[tuple[int, int]] = []
    start, n = 0, len(word_ids)
    while start < n:
        end = min(start + size, n)
        if end < n:
            boundary = _word_start(word_ids, end)
            end = boundary if boundary > start else end  # one giant "word": cut it anyway
        windows.append((start, end))
        if end == n:
            break
        next_start = _word_start(word_ids, end - overlap)
        start = next_start if next_start > start else end
    return windows


def chunk_text(
    text: str,
    ticker: str,
    fiscal_year: int,
    section: Literal["1A", "7"],
    tokenizer: Tokenizer,
    size: int = CHUNK_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> list[Chunk]:
    encoding = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i, (start, end) in enumerate(token_windows(encoding.word_ids, size, overlap)):
        piece = text[encoding.offsets[start][0] : encoding.offsets[end - 1][1]]
        chunks.append(
            Chunk(
                chunk_id=f"{ticker}-{fiscal_year}-{section}-{i:04d}",
                ticker=ticker,
                fiscal_year=fiscal_year,
                section=section,
                text=piece,
                token_count=end - start,
            )
        )
    return chunks


def count_tokens(text: str, tokenizer: Tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).offsets)


def split_recursive(
    text: str, tokenizer: Tokenizer, max_tokens: int, separators: Sequence[str] = SEPARATORS
) -> list[str]:
    """Pieces of at most max_tokens, cut at the coarsest separator that works.
    The separator stays attached, so "".join(pieces) == text (no text lost or changed)."""
    if count_tokens(text, tokenizer) <= max_tokens:
        return [text]
    if not separators:  # one "word" longer than the limit: cut it in half (practically never)
        half = len(text) // 2
        return [*split_recursive(text[:half], tokenizer, max_tokens, ()),
                *split_recursive(text[half:], tokenizer, max_tokens, ())]  # fmt: skip
    sep, finer = separators[0], separators[1:]
    parts = text.split(sep)
    if len(parts) == 1:
        return split_recursive(text, tokenizer, max_tokens, finer)
    parts = [p + sep for p in parts[:-1]] + [parts[-1]]
    pieces: list[str] = []
    for part in parts:
        pieces += split_recursive(part, tokenizer, max_tokens, finer)
    return merge_pieces(pieces, tokenizer, max_tokens)


def merge_pieces(pieces: Sequence[str], tokenizer: Tokenizer, max_tokens: int) -> list[str]:
    """Join neighbouring pieces as long as the result still fits."""
    merged: list[str] = []
    current = ""
    for piece in pieces:
        if current and count_tokens(current + piece, tokenizer) > max_tokens:
            merged.append(current)
            current = piece
        else:
            current += piece
    if current:
        merged.append(current)
    return merged


def is_heading(line: str, next_line: str) -> bool:
    short = HEADING_CHARS[0] <= len(line) <= HEADING_CHARS[1]
    return short and not line.lstrip().startswith(BULLETS) and len(next_line) > HEADING_CHARS[1]


def heading_groups(text: str) -> list[tuple[str, str]]:
    """(heading, body) blocks. Text before the first heading gets an empty heading."""
    lines = text.splitlines()
    groups: list[tuple[str, str]] = []
    heading, body = "", []
    for i, line in enumerate(lines):
        if is_heading(line, lines[i + 1] if i + 1 < len(lines) else ""):
            if heading or body:
                groups.append((heading, "\n".join(body)))
            heading, body = line, []
        else:
            body.append(line)
    if heading or body:
        groups.append((heading, "\n".join(body)))
    return groups


def section_pieces(text: str, tokenizer: Tokenizer, max_tokens: int) -> list[str]:
    pieces: list[str] = []
    for heading, body in heading_groups(text):
        block = f"{heading}\n{body}" if heading else body
        if count_tokens(block, tokenizer) <= max_tokens:
            pieces.append(block)
            continue
        budget = max_tokens - count_tokens(heading, tokenizer) - 1  # room for heading + newline
        for part in split_recursive(body, tokenizer, budget):
            pieces.append(f"{heading}\n{part.strip()}" if heading else part)
    return join_small(pieces, tokenizer, max_tokens)


def join_small(pieces: Sequence[str], tokenizer: Tokenizer, max_tokens: int) -> list[str]:
    """A tiny piece (a lone heading, a one-line note) is joined to the next one if it fits."""
    joined: list[str] = []
    carry = ""
    for piece in pieces:
        candidate = f"{carry}\n{piece}" if carry else piece
        if carry and count_tokens(candidate, tokenizer) > max_tokens:
            joined.append(carry)
            candidate = piece
        if count_tokens(candidate, tokenizer) < MIN_PIECE_TOKENS:
            carry = candidate
        else:
            joined.append(candidate)
            carry = ""
    if carry:
        joined.append(carry)
    return joined


def chunk_section(
    text: str,
    ticker: str,
    fiscal_year: int,
    section: Literal["1A", "7"],
    tokenizer: Tokenizer,
    strategy: Strategy,
) -> list[Chunk]:
    if strategy == "fixed":
        return chunk_text(text, ticker, fiscal_year, section, tokenizer)
    if strategy == "recursive":
        pieces = split_recursive(text, tokenizer, CHUNK_TOKENS)
    else:
        pieces = section_pieces(text, tokenizer, CHUNK_TOKENS)
    texts = [p.strip() for p in pieces if p.strip()]
    return [
        Chunk(
            chunk_id=f"{ticker}-{fiscal_year}-{section}-{i:04d}",
            ticker=ticker,
            fiscal_year=fiscal_year,
            section=section,
            text=t,
            token_count=count_tokens(t, tokenizer),
        )
        for i, t in enumerate(texts)
    ]


def load_tokenizer() -> Tokenizer:
    from tokenizers import Tokenizer as HFTokenizer

    tokenizer = HFTokenizer.from_pretrained(EMBEDDING_MODEL)
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def build_chunks(
    settings: Settings, tokenizer: Tokenizer, strategy: Strategy = "fixed"
) -> list[Chunk]:
    fiscal_years = {f.ticker: f.report_date.year for f in load_manifest(settings)}
    chunks: list[Chunk] = []
    for ticker in COMPANIES:
        for section in SECTIONS:
            text = clean_text((settings.sections_dir / f"{ticker}_{section}.txt").read_text())
            chunks += chunk_section(
                text, ticker, fiscal_years[ticker], section, tokenizer, strategy
            )
    return chunks


def save_chunks(chunks: list[Chunk], settings: Settings, name: str = "fixed") -> None:
    settings.chunks_dir.mkdir(parents=True, exist_ok=True)
    lines = (c.model_dump_json() for c in chunks)
    (settings.chunks_dir / f"{name}.jsonl").write_text("\n".join(lines) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk.model_validate_json(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk the 10-K sections for search.")
    parser.add_argument("--strategy", choices=get_args(Strategy), default="fixed")
    args = parser.parse_args()
    settings = Settings()
    chunks = build_chunks(settings, load_tokenizer(), args.strategy)
    save_chunks(chunks, settings, name=args.strategy)
    for ticker in COMPANIES:
        counts = {s: sum(c.ticker == ticker and c.section == s for c in chunks) for s in SECTIONS}
        print(f"{ticker:<6} 1A={counts['1A']:>4}  7={counts['7']:>4}")
    sizes = [c.token_count for c in chunks]
    median = statistics.median(sizes)
    print(f"\n{args.strategy}: {len(chunks):,} chunks; tokens min {min(sizes)}, ", end="")
    print(f"median {median}, max {max(sizes)}")
    print(f"written to {settings.chunks_dir / f'{args.strategy}.jsonl'}")


if __name__ == "__main__":
    main()
