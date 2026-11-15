"""Does a quote really appear in a text? Used for test-set labels and for answer citations."""

import re

FANCY_TO_PLAIN = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"}


def normalize_for_match(text: str) -> str:
    """For matching only: lowercase, straight quotes and dashes, single spaces."""
    for fancy, plain in FANCY_TO_PLAIN.items():
        text = text.replace(fancy, plain)
    return re.sub(r"\s+", " ", text).strip().lower()


def contains_quote(text: str, quote: str) -> bool:
    return normalize_for_match(quote) in normalize_for_match(text)
