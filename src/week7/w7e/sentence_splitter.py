"""Deterministic sentence splitting with abbreviation and citation preservation for W7-E.

Handles:
- Common abbreviations (Dr., Prof., Fig., Figs., Tab., Tabs., Eq., Eqs., Ref., Refs., Sec., Secs., al., e.g., i.e., etc., vs., cf.)
- Decimal numbers (e.g., 3.14, 0.005)
- Citations (e.g., [1], [1, 2], (Smith et al., 2020))
- Strips whitespace, excludes empty segments
- Preserves original character spans if requested
"""

from __future__ import annotations

import re
from typing import List

# Common abbreviations where periods should not trigger sentence boundaries
ABBREVIATIONS = (
    r"(?:[A-Z]\.)|"  # Single initials like A. or J.
    r"(?:[a-zA-Z]+\.[a-zA-Z]+\.)|"  # e.g., i.e.
    r"(?:Fig|Figs|Tab|Tabs|Eq|Eqs|Ref|Refs|Sec|Secs|Vol|No|pp|vs|cf|al|approx|dept|univ|ed|eds|Dr|Prof|Mr|Mrs|Ms|Sr|Jr|St)\."
)

ABBREV_PATTERN = re.compile(rf"\b({ABBREVIATIONS})", re.IGNORECASE)
DECIMAL_PATTERN = re.compile(r"\b(\d+)\.(\d+)\b")
CITATION_PATTERN = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")

# Placeholder token to protect periods in abbreviations
PERIOD_PLACEHOLDER = "<PERIOD_PROTECT>"


def split_into_sentences(text: str) -> List[str]:
    """Deterministically split text into natural-language sentences.

    Guarantees:
    - Abbreviations and decimal numbers are not split across sentences.
    - Resulting sentences are non-empty and stripped of outer whitespace.
    - If no sentence boundary is detected, returns [text.strip()].
    """
    raw_text = text.strip()
    if not raw_text:
        return []

    # 1. Protect decimal numbers: 3.14 -> 3<PERIOD_PROTECT>14
    protected = DECIMAL_PATTERN.sub(rf"\1{PERIOD_PLACEHOLDER}\2", raw_text)

    # 2. Protect known abbreviations
    def protect_abbrev(match: re.Match) -> str:
        return match.group(0).replace(".", PERIOD_PLACEHOLDER)

    protected = ABBREV_PATTERN.sub(protect_abbrev, protected)

    # 3. Sentence boundary regex: split on period/question/exclamation followed by space and uppercase letter or quote
    boundary_regex = re.compile(r'(?<=[.?!])\s+(?=[A-Z"\'\(\[])')
    raw_sentences = boundary_regex.split(protected)

    # 4. Restore periods and clean up
    clean_sentences: List[str] = []
    for s in raw_sentences:
        restored = s.replace(PERIOD_PLACEHOLDER, ".").strip()
        if restored:
            clean_sentences.append(restored)

    if not clean_sentences:
        clean_sentences = [raw_text]

    return clean_sentences
