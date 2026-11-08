"""Cut Item 1A (Risk Factors) and Item 7 (MD&A) out of 10-K text.

A real heading starts its own line ("ITEM 1A. RISK FACTORS"); cross-references sit mid-sentence
("see Item 1A"). The table of contents also has headings, but its "section" is tiny, so we keep
the longest span from a start heading to the next end heading.
"""

import json
import re

from filing_analyst.config import Settings
from filing_analyst.parse.html_text import html_to_text
from filing_analyst.sec.filings import Filing

# section name -> (item that starts it, items that can end it)
SECTIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "1A": ("1A", ("1B", "1C", "2")),
    "7": ("7", ("7A", "8")),
}
MIN_SECTION_CHARS = 5_000


def heading_positions(text: str, item: str) -> list[int]:
    """Positions where a line starts with "Item <item>" ("Item 7" does not match "Item 7A")."""
    pattern = rf"^item\s*{item}\b"
    return [m.start() for m in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)]


def find_section(text: str, start_item: str, end_items: tuple[str, ...]) -> str | None:
    """Longest span from a start heading to the first end heading after it."""
    ends = sorted(pos for item in end_items for pos in heading_positions(text, item))
    spans = []
    for start in heading_positions(text, start_item):
        end = next((pos for pos in ends if pos > start), None)
        if end is not None:
            spans.append((start, end))
    if not spans:
        return None
    start, end = max(spans, key=lambda span: span[1] - span[0])
    return text[start:end].strip()


def extract_section(text: str, name: str, min_chars: int = MIN_SECTION_CHARS) -> str | None:
    """Return the section, or None if it is missing or too short to be the real one."""
    start_item, end_items = SECTIONS[name]
    section = find_section(text, start_item, end_items)
    if section is None or len(section) < min_chars:
        return None
    return section


def load_manifest(settings: Settings) -> list[Filing]:
    raw = json.loads((settings.raw_dir / "10k" / "manifest.json").read_text())
    return [Filing.model_validate(row) for row in raw]


def main() -> None:
    settings = Settings()
    settings.sections_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    for filing in load_manifest(settings):
        text = html_to_text((settings.raw_dir / "10k" / filing.local_name).read_bytes())
        sizes = []
        for name in SECTIONS:
            section = extract_section(text, name)
            if section is None:
                found = find_section(text, *SECTIONS[name])
                size = len(found) if found else 0
                missing.append(f"{filing.ticker} Item {name}: best match only {size} chars")
                sizes.append(f"{name}=MISSING")
                continue
            (settings.sections_dir / f"{filing.ticker}_{name}.txt").write_text(section)
            sizes.append(f"{name}={len(section):>7,}")
        print(f"{filing.ticker:<6} " + "  ".join(sizes))
    print(f"\nnot found ({len(missing)}):" if missing else "\nall sections found")
    for line in missing:
        print(f"  {line}")


if __name__ == "__main__":
    main()
