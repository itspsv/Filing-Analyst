"""Polite SEC downloads: User-Agent header, a pause between requests, and a disk cache."""

import time
from pathlib import Path

import httpx

PAUSE_SECONDS = 0.2  # SEC limit is 10 requests/second; 0.2 s keeps us well under it
TIMEOUT_SECONDS = 30.0


def fetch_cached(url: str, dest: Path, user_agent: str) -> Path:
    """Return dest, downloading url into it first only if it is not already on disk."""
    if dest.exists():
        return dest
    response = httpx.get(url, headers={"User-Agent": user_agent}, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    time.sleep(PAUSE_SECONDS)
    return dest
