"""
Grounds an LLM-reported "verbatim quote" back into the real source text.

Small local models paraphrase even when told to quote exactly, so we
never trust the model's own claim of what the source said — we verify
it against the actual page text pulled by pdfplumber and report a
match score. Low-scoring quotes are treated as unreliable by the caller
(pdf_extractor.py drops them rather than showing invented "evidence").
"""

from typing import Tuple

from rapidfuzz import fuzz


def ground_quote(quote: str, source_text: str, min_score: float = 50.0) -> Tuple[str, float]:
    """
    Attempt to locate `quote` inside `source_text`.

    Returns:
        (grounded_snippet, score) where score is 0-100. When an exact
        substring match exists, score is 100 and the snippet is the
        exact quote. Otherwise we fuzzy-align and return the best
        matching span from the real text (not the model's paraphrase),
        along with how confident that alignment is.
    """
    quote = (quote or "").strip()
    source_text = source_text or ""

    if not quote or not source_text:
        return quote, 0.0

    # Exact match — best case, and the common case for short numeric quotes.
    idx = source_text.find(quote)
    if idx != -1:
        return quote, 100.0

    # Case-insensitive exact match.
    idx = source_text.lower().find(quote.lower())
    if idx != -1:
        return source_text[idx : idx + len(quote)], 98.0

    # Fuzzy alignment: find the best-matching span of source_text for quote.
    try:
        alignment = fuzz.partial_ratio_alignment(quote, source_text)
        snippet = source_text[alignment.dest_start : alignment.dest_end].strip()
        if snippet:
            return snippet, alignment.score
    except AttributeError:
        # Older rapidfuzz without alignment support — fall back to a
        # whole-string score and keep the model's own quote text, but
        # let the low score signal low trust to the caller.
        pass

    score = fuzz.partial_ratio(quote, source_text)
    return quote, score
