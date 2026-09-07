import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

import pdfplumber

from src.models.fact import Fact, FactType
from src.extraction.llm_client import LLMClient
from src.extraction.grounding import ground_quote


EXTRACTION_SYSTEM_PROMPT = """You are a careful fact-extraction engine. You read a chunk of \
text from any document and extract meaningful, self-contained facts stated or implied in it.

A "fact" is any concrete, checkable claim: a quantity, count, metric, duration, date, \
named relationship, specification, policy rule, status, or obligation. Skip filler text, \
page headers, and sentences with no checkable content.

Do NOT assume any particular document type. Let the text itself tell you what facts matter.

For every fact you find, output an object with these fields:
- "claim": one sentence stating the fact plainly, in your own words.
- "fact_type": one of "numeric", "semantic", "date", "boolean", "entity".
- "value": the core value of the fact (a number, a short string, or a date string).
- "unit": the unit if applicable (e.g. "USD", "percent", "minutes", "items"), else null.
- "entity": the main subject the fact applies to, else null.
- "time_period": the period or date the fact applies to, else null.
- "scope": any qualifying condition or scope, else null.
- "verbatim_quote": the EXACT character-for-character substring from the text supporting this fact.
- "confidence": your confidence between 0.0 and 1.0.

Return ONLY a JSON object in this exact schema:
{
  "facts": [...]
}
If the text contains no extractable facts, return {"facts": []}."""


class PDFExtractor:
    """
    LLM-backed, document-agnostic PDF fact extractor.

    pdfplumber handles PDF -> text (with page numbers preserved). A local
    LLM (via LLMClient) reads each chunk of that text and proposes facts
    as structured JSON. Every proposed fact is then grounded: its
    "verbatim_quote" is fuzzy-matched back into the real source text
    (grounding.ground_quote) so we never show the user evidence the
    model invented. Facts that can't be grounded above a score threshold
    are dropped rather than surfaced with fake evidence.
    """

    # Small local models have limited context; chunk long pages rather
    # than sending an entire page in one call.
    MAX_CHUNK_CHARS = 600
    CHUNK_OVERLAP = 200

    # Below this fuzzy-match score, we don't trust the model's quote
    # enough to call it "evidence" — the fact is dropped.
    MIN_GROUNDING_SCORE = 50.0

    _KNOWN_RAW_KEYS = {
        "claim", "text", "fact_type", "value", "unit", "verbatim_quote",
        "evidence", "confidence", "entity", "time_period", "scope",
    }

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    # ------------------------------------------------------------------
    # PDF -> text (unchanged from the original extractor)
    # ------------------------------------------------------------------

    def extract_text(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text while preserving page boundaries.

        Returns: [{"page": 1, "text": "..."}, ...]
        """
        pages = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""

                    if not text.strip():
                        try:
                            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                        except Exception:
                            text = ""

                    pages.append({"page": page_number, "text": text.strip()})

        except Exception as exc:
            raise RuntimeError(f"Unable to extract PDF '{pdf_path}': {exc}") from exc

        return pages

    # ------------------------------------------------------------------
    # Main extraction pipeline
    # ------------------------------------------------------------------

    def extract_facts(self, pdf_path: str, source_document: Optional[str] = None) -> List[Fact]:
        """
        Extract document-agnostic facts from a PDF using a local LLM.

        No document type is detected. No filename-based rules are used.
        No domain-specific extraction path is used — the prompt and the
        grounding step are the only logic; everything else is left to
        the model, which is what lets this generalize to unseen PDFs.
        """
        source_document = source_document or pdf_path
        pages = self.extract_text(pdf_path)

        facts: List[Fact] = []

        for page_data in pages:
            page_number = page_data["page"]
            page_text = page_data["text"]

            if not page_text:
                continue

            for chunk in self._chunk_page(page_text):
                raw_facts = self._extract_facts_from_chunk(chunk)

                for raw in raw_facts:
                    fact = self._build_fact(raw, chunk, page_number, source_document)
                    if fact:
                        facts.append(fact)

        return self._deduplicate_facts(facts)

    def _chunk_page(self, text: str) -> List[str]:
        if len(text) <= self.MAX_CHUNK_CHARS:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.MAX_CHUNK_CHARS
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - self.CHUNK_OVERLAP
        return chunks

    def _extract_facts_from_chunk(self, chunk: str) -> List[Dict[str, Any]]:
        try:
            result = self.llm.generate_json(
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                user_prompt=f'TEXT:\n"""\n{chunk}\n"""',
            )
        except (ValueError, RuntimeError) as exc:
            # A chunk the model couldn't handle (bad JSON, timeout, etc).
            # This is a real extraction-failure mode worth showing in the
            # README's "failure case" — we log and skip rather than crash
            # the whole document.
            print(f"[extractor] skipping chunk ({len(chunk)} chars) due to error: {exc}")
            return []

        # The model may wrap the array under a key instead of returning
        # it bare, despite instructions. Handle both.
        if isinstance(result, dict):
            for key in ("facts", "items", "results"):
                if isinstance(result.get(key), list):
                    return result[key]
            return []
        if isinstance(result, list):
            return result
        return []

    def _build_fact(
        self,
        raw: Any,
        chunk: str,
        page_number: int,
        source_document: str,
    ) -> Optional[Fact]:
        if not isinstance(raw, dict):
            return None

        claim = str(raw.get("claim") or raw.get("text") or "").strip()
        quote = str(raw.get("verbatim_quote") or raw.get("evidence") or "").strip()

        if not claim or not quote:
            return None

        grounded_snippet, score = ground_quote(quote, chunk, self.MIN_GROUNDING_SCORE)

        if score < self.MIN_GROUNDING_SCORE:
            # Can't verify this quote appears in the source at all —
            # treat as a likely hallucination and drop it rather than
            # present ungrounded "evidence".
            return None

        try:
            llm_confidence = float(raw.get("confidence"))
        except (TypeError, ValueError):
            llm_confidence = 0.6
        llm_confidence = max(0.0, min(1.0, llm_confidence))

        # Blend the model's self-reported confidence with how well we
        # could actually verify its quote against real text.
        confidence = round(min(llm_confidence, score / 100.0), 2)

        context: Dict[str, Any] = {}
        for field in ("entity", "time_period", "scope"):
            if raw.get(field):
                context[field] = raw[field]

        # Anything else the model returned flows straight into context.
        # This is what lets the schema evolve per-document without code
        # changes when a new PDF surfaces a field we didn't anticipate.
        for key, value in raw.items():
            if key not in self._KNOWN_RAW_KEYS and value not in (None, "", []):
                context[key] = value

        return Fact(
            id=str(uuid.uuid4()),
            text=claim,
            fact_type=self._normalize_fact_type(raw.get("fact_type")),
            value=raw.get("value"),
            unit=raw.get("unit"),
            context=context,
            source_document=source_document,
            source_page=page_number,
            source_snippet=grounded_snippet,
            confidence=confidence,
            extraction_method="llm",
            created_at=datetime.now(),
        )

    def _normalize_fact_type(self, value: Any) -> FactType:
        if isinstance(value, str):
            normalized = value.strip().lower()
            for member in FactType:
                if member.value == normalized:
                    return member
        return FactType.SEMANTIC

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate_facts(self, facts: List[Fact]) -> List[Fact]:
        unique = []
        seen = set()

        for fact in facts:
            key = (
                fact.source_document,
                fact.source_page,
                fact.fact_type.value,
                self._normalize_text(str(fact.value)),
                fact.unit,
                self._normalize_text(fact.source_snippet),
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(fact)

        return unique

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()