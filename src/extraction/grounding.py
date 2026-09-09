# src/extraction/grounding.py

import re
from typing import Optional, Tuple

from rapidfuzz import fuzz


def ground_quote(
    quote: str,
    source_text: str,
    min_score: float = 80.0,
    expand_to_sentence: bool = True,
) -> Tuple[str, float]:

    quote = (quote or "").strip()
    source_text = source_text or ""

    if not quote or not source_text:
        return "", 0.0

    # Exact source match. An exact substring is already the strongest,
    # most concise evidence — return the verbatim span itself rather than
    # expanding to the whole sentence.
    index = source_text.find(quote)

    if index != -1:
        return _build_span(
            source_text, index, index + len(quote),
            base_score=100.0, expand=False,
        )

    # Case-insensitive source match.
    lower_source = source_text.lower()
    lower_quote = quote.lower()

    index = lower_source.find(lower_quote)

    if index != -1:
        return _build_span(
            source_text, index, index + len(quote),
            base_score=98.0, expand=False,
        )

    # Normalize whitespace while retaining the original source.
    normalized_quote = re.sub(
        r"\s+",
        " ",
        quote,
    ).strip()

    normalized_source = re.sub(
        r"\s+",
        " ",
        source_text,
    ).strip()

    if normalized_quote:
        index = normalized_source.lower().find(
            normalized_quote.lower()
        )

        if index != -1:
            return _recover_source_span(
                source_text,
                quote,
            )

    # Fuzzy alignment. Only this path expands to the containing sentence,
    # because a fuzzy match may land mid-sentence and needs context to
    # establish the claim.
    alignment = _fuzzy_align(quote, source_text)
    if alignment is not None:
        snippet, score, dest_start = alignment
        if snippet and score >= min_score:
            if expand_to_sentence:
                snippet = _expand_to_sentence(
                    snippet,
                    source_text,
                    start=dest_start,
                )
            return snippet, score

    # Fall back to grounding on the numeric value alone. The model often
    # paraphrases the surrounding text of a bare table cell (e.g. a "total"
    # row), but the number itself is always verbatim. Extracting and
    # locating the numeric token is a robust, document-agnostic anchor.
    number_token = _extract_number_literal(quote)
    if number_token:
        index = source_text.find(number_token)
        if index != -1:
            snippet = source_text[
                index:index + len(number_token)
            ].strip()
            if snippet:
                if expand_to_sentence:
                    snippet = _expand_to_sentence(
                        snippet,
                        source_text,
                        start=index,
                    )
                return snippet, 85.0

    # Never return the model's quote as evidence.
    return "", 0.0

def _build_span(
    source_text: str,
    start: int,
    end: int,
    base_score: float,
    expand: bool = True,
) -> Tuple[str, float]:
    """Build a grounded source span from character offsets.

    Optionally expands the matched quote to the containing sentence while
    preserving the original source text.
    """
    if not source_text or start < 0 or end <= start:
        return "", 0.0

    start = max(0, start)
    end = min(len(source_text), end)

    snippet = source_text[start:end].strip()
    if not snippet:
        return "", 0.0

    if expand:
        snippet = _expand_to_sentence(
            snippet,
            source_text,
            start=start,
        )

    return snippet, base_score

def _expand_to_sentence(
    snippet: str,
    source_text: str,
    start: Optional[int] = None,
) -> str:
    """Expand a matched snippet to the boundaries of its containing
    sentence in `source_text`, preserving the original source wording."""
    if start is None:
        start = source_text.find(snippet)
        if start == -1:
            return snippet

    end = start + len(snippet)

    # Walk backwards to the nearest sentence-ending punctuation.
    sentence_start = start
    sentence_end = end

    previous_boundary = max(
        source_text.rfind(".", 0, start),
        source_text.rfind("!", 0, start),
        source_text.rfind("?", 0, start),
    )

    if previous_boundary != -1:
        sentence_start = previous_boundary + 1

    # Walk forwards to the next sentence-ending punctuation.
    boundaries = [
        pos for pos in (
            source_text.find(".", end),
            source_text.find("!", end),
            source_text.find("?", end),
        )
        if pos != -1
    ]

    if boundaries:
        sentence_end = min(boundaries) + 1
    else:
        sentence_end = len(source_text)

    expanded = source_text[sentence_start:sentence_end].strip()

    return expanded if expanded else snippet

def _fuzzy_align(
    quote: str,
    source_text: str,
):
    """Find the best fuzzy alignment of `quote` within `source_text`.

    Returns ``(snippet, score, dest_start)`` or None. Handles both the
    rapidfuzz ``ScoreAlignment`` object and older plain-tuple returns.
    """
    try:
        alignment = fuzz.partial_ratio_alignment(
            quote,
            source_text,
        )
    except AttributeError:
        return None

    if hasattr(alignment, "dest_start"):
        return (
            source_text[
                alignment.dest_start:alignment.dest_end
            ].strip(),
            float(alignment.score),
            alignment.dest_start,
        )

    try:
        parts = tuple(alignment)
    except TypeError:
        return None

    if len(parts) >= 3:
        score = float(parts[0])
        dest_start, dest_end = parts[-2], parts[-1]
        return (
            source_text[dest_start:dest_end].strip(),
            score,
            dest_start,
        )

    return None


def _extract_number_literal(text: str):
    """Return the longest verbatim numeric token (incl. currency/thousands)
    found in `text`, or None. Aggnostic to units and domain."""
    matches = re.findall(
        r"(?:₹|Rs\.?|USD|US\$|\$|€|£)?"
        r"\d{1,3}(?:,\d{3})*"
        r"(?:\.\d+)?",
        text,
    )
    if not matches:
        return None
    return max(matches, key=len)


def _recover_source_span(
    source_text: str,
    quote: str,
) -> Tuple[str, float]:

    quote_tokens = re.findall(
        r"\S+",
        quote,
    )

    if not quote_tokens:
        return "", 0.0

    pattern = r"\s+".join(
        re.escape(token)
        for token in quote_tokens
    )

    match = re.search(
        pattern,
        source_text,
        flags=re.IGNORECASE,
    )

    if not match:
        return "", 0.0

    return (
        source_text[
            match.start():match.end()
        ],
        96.0,
    )