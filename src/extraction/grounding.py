# src/extraction/grounding.py

import re
from typing import Tuple

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

    # Exact source match.
    index = source_text.find(quote)

    if index != -1:
        return _build_span(
            source_text, index, index + len(quote),
            base_score=100.0, expand=expand_to_sentence,
        )

    # Case-insensitive source match.
    lower_source = source_text.lower()
    lower_quote = quote.lower()

    index = lower_source.find(lower_quote)

    if index != -1:
        return _build_span(
            source_text, index, index + len(quote),
            base_score=98.0, expand=expand_to_sentence,
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

    # Fuzzy alignment.
    alignment = _fuzzy_align(quote, source_text)
    if alignment is not None:
        snippet, score = alignment
        if snippet and score >= min_score:
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

    if not expand:
        return source_text[start:end].strip(), base_score

    # Find the sentence boundaries around the matched span.
    sentence_start = start
    sentence_end = end

    # Walk backwards to the nearest sentence-ending punctuation.
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

    snippet = source_text[sentence_start:sentence_end].strip()

    if not snippet:
        return "", 0.0

    return snippet, base_score

def _fuzzy_align(
    quote: str,
    source_text: str,
):
    try:
        alignment = fuzz.partial_ratio_alignment(
            quote,
            source_text,
        )
    except AttributeError:
        return None

    snippet = source_text[
        alignment.dest_start:alignment.dest_end
    ].strip()
    return snippet, float(alignment.score)


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