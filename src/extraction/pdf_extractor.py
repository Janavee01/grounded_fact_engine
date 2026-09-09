# src/extraction/pdf_extractor.py

import hashlib
import json
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import pdfplumber
from rapidfuzz import fuzz

from src.models.fact import Fact, FactType
from src.extraction.llm_client import LLMClient
from src.extraction.grounding import ground_quote

EXTRACTION_SYSTEM_PROMPT = """You are a fact extractor. Read the text carefully and extract the concrete, meaningful facts it states.

Each fact must have these fields:
- claim: a clear, self-contained sentence summarizing the fact
- fact_type: one of "numeric", "semantic", "date", "time", "boolean", or "entity"
- value: the number or value if the fact contains one, otherwise null
- unit: the unit of measurement if present, otherwise null
- verbatim_quote: an exact quote from the text that supports the claim
- confidence: a number 0.0 to 1.0 reflecting how directly the text states the fact (1.0 = explicitly stated; lower = inferred/uncertain). Do not set 1.0 unless the claim is stated almost verbatim.

CLASSIFICATION RULES (determine fact_type by semantic meaning, not by how the value looks):
- "numeric": a fact whose core content is a meaningful quantity (e.g. revenue, population, percentage, count). A number that merely appears inside an address, date, identifier, serial code, or other structured value should NOT be classified as numeric — classify the whole structured value by its semantic role instead.
- "date": a fact whose core content is a calendar date (e.g. founding date, reporting period end date, effective date).
- "time": a fact whose core content is a time of day, clock time, or time range (e.g. "10:00 AM to 6:00 PM", "the meeting started at 3:30 PM"). Even if expressed as text, a time reference must be classified as "time", not "semantic".
- "boolean": a fact whose core content is an explicit true/false state, presence/absence, or yes/no condition.
- "entity": a fact whose core content is a meaningful named entity (person, organization, location, product) where the entity itself is the fact.
- "semantic": a meaningful textual fact that does not fit any of the above categories (e.g. a policy description, a qualitative statement, a relationship).

MEANINGFULNESS RULES (only extract facts that carry substantive information):
- Extract only facts that are directly and explicitly stated in the text.
- Skip isolated labels, headers, section titles, column names, and fragments that do not state a complete proposition.
- Skip standalone identifiers, registration numbers, reference codes, serial numbers, and similar alphanumeric tokens when they are not part of a meaningful proposition.
- Skip boilerplate text, disclaimers, legal boilerplate, and template placeholders (e.g. "[INSERT NAME]", "XXX", "***", "N/A", "TBD").
- Skip masked or redacted values (e.g. "XXXX-XXXX-XXXX", "***-**-****", "[REDACTED]").
- Skip credentials, tokens, keys, passwords, or authentication-related values.
- Skip page numbers, footers, headers, watermarks, and document metadata that convey no factual content.

EVIDENCE RULES (every fact must be grounded in the source text):
- The verbatim_quote must be an exact substring of the supplied text — copy it character-for-character.
- Do not paraphrase the quote — use the original words.
- Prefer the shortest exact substring that directly supports the claim. Do not use the entire text block when a shorter phrase suffices.
- You may extract multiple facts from a single text block.
- Preserve qualifiers like "approximately", "as of", "net of", etc.

Return your answer as JSON in this exact format:
{"facts": [{"claim": "...", "fact_type": "...", "value": null, "unit": null, "verbatim_quote": "...", "confidence": 0.9}]}

Example — given this text:
"Alpha Corp was founded on 14 March 2012. Its annual output was 85,000 units in fiscal year 2023. The office is open from 9:00 AM to 5:00 PM."

You would return:
{"facts": [
  {"claim": "Alpha Corp was founded on 14 March 2012.", "fact_type": "date", "value": "2012-03-14", "unit": null, "verbatim_quote": "Alpha Corp was founded on 14 March 2012.", "confidence": 1.0},
  {"claim": "Alpha Corp's annual output was 85,000 units in fiscal year 2023.", "fact_type": "numeric", "value": 85000, "unit": "units", "verbatim_quote": "Its annual output was 85,000 units in fiscal year 2023.", "confidence": 1.0},
  {"claim": "The office is open from 9:00 AM to 5:00 PM.", "fact_type": "time", "value": "09:00-17:00", "unit": null, "verbatim_quote": "The office is open from 9:00 AM to 5:00 PM.", "confidence": 1.0}
]}"""


class PDFExtractor:
    """
    Document-agnostic PDF fact extractor.

    No filename-based rules, domain-specific rules, hard-coded metrics,
    hard-coded entities, or document-specific extraction paths are used.
    """

    MAX_CHUNK_CHARS = 14000
    CHUNK_OVERLAP = 50
    MIN_GROUNDING_SCORE = 80.0
    MIN_QUOTE_CHARS = 12

    _KNOWN_RAW_KEYS = {
        "claim",
        "text",
        "fact_type",
        "value",
        "unit",
        "verbatim_quote",
        "evidence",
        "confidence",
        "entity",
        "time_period",
        "scope",
    }

    @staticmethod
    def _log(message: str) -> None:
        # Upload requests can take several model calls. Show progress by
        # default so a synchronous UI does not look stalled; users can still
        # silence this with PDF_EXTRACTION_VERBOSE=0.
        verbose = os.getenv("PDF_EXTRACTION_VERBOSE", "1") == "1"
        if verbose:
            print(f"[extractor] {message}", flush=True)

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_workers: Optional[int] = None,
    ):
        self.llm = llm_client or LLMClient()
        # Parallelism changes throughput only: every chunk still receives the
        # same prompt and goes through the same grounding and deduplication.
        # Local Ollama queues parallel requests internally, so a single worker
        # is right there. Hosted OpenRouter (and other HTTP endpoints) handle
        # concurrency natively, so we try to keep the CPU warm by default.
        configured_workers = (
            max_workers
            if max_workers is not None
            else os.getenv("PDF_EXTRACTION_WORKERS")
        )
        if configured_workers:
            try:
                self.max_workers = max(1, int(configured_workers))
            except (TypeError, ValueError):
                self.max_workers = 1
        elif getattr(self.llm, "provider", "ollama") == "openrouter":
            self.max_workers = max(
                1,
                min(3, (os.cpu_count() or 4)),
            )
        elif os.getenv("OLLAMA_NUM_PARALLEL", "1") != "1":
            self.max_workers = max(
                1,
                min(int(os.getenv("OLLAMA_NUM_PARALLEL", "1")), (os.cpu_count() or 4)),
            )
        else:
            self.max_workers = 1

    # ------------------------------------------------------------------
    #  OCR fallback for scanned / image-only pages
    # ------------------------------------------------------------------

    OCR_DPI = 300

    def _ocr_page(self, page: Any) -> str:
        """
        Second-chance extraction for pages with no selectable text
        (scanned images, heavily graphical layouts).  Renders the page
        to a bitmap at OCR_DPI and runs Tesseract over it.

        pytesseract/Pillow are imported lazily so extraction still works
        on machines without OCR installed — those pages stay empty
        (logged honestly) instead of crashing the run.
        """
        import pytesseract

        image = page.to_image(resolution=self.OCR_DPI).original
        text = pytesseract.image_to_string(image)
        return (text or "").strip()

    def extract_text(self, pdf_path: str) -> List[Dict[str, Any]]:
        pages = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    text_source = "pdfplumber"

                    try:
                        text = self._page_text(page)
                    except Exception:
                        text = page.extract_text() or ""

                    if not text.strip():
                        try:
                            text = (
                                page.extract_text(
                                    x_tolerance=2,
                                    y_tolerance=2,
                                )
                                or ""
                            )
                        except Exception:
                            text = ""

                    if not text.strip():
                        try:
                            text = self._ocr_page(page)
                        except Exception as exc:
                            text = ""
                            self._log(
                                f"page {page_number}: OCR unavailable "
                                f"({exc}); page left empty"
                            )

                        if text:
                            text_source = "ocr"
                            self._log(
                                f"page {page_number}: OCR recovered "
                                f"{len(text):,} chars"
                            )

                    pages.append(
                        {
                            "page": page_number,
                            "text": text.strip(),
                            "source": text_source,
                        }
                    )

        except Exception as exc:
            raise RuntimeError(
                f"Unable to extract PDF '{pdf_path}': {exc}"
            ) from exc

        return pages

    # ------------------------------------------------------------------
    #  Table-aware page rectification
    # ------------------------------------------------------------------

    _TABLE_FILTER_PAD = 2.0
    _COL_GAP_TOLERANCE = 18.0
    _COL_MIN_GROUP_SIZE = 4

    def _page_text(self, page: Any) -> str:
        """Best-effort text extraction that separates prose from tables."""

        kept_tables = self._kept_tables(page)
        table_bboxes = [t.bbox for t in kept_tables]

        all_words = page.extract_words(
            x_tolerance=2, y_tolerance=2
        ) or []

        outside_words = [
            w for w in all_words
            if not self._inside_any_rect(w, table_bboxes)
        ]

        # --- prose layer (chars outside table bounding boxes) ------------

        if table_bboxes:

            def _keep(obj: dict) -> bool:
                if obj.get("object_type") != "char":
                    return True
                cx = (obj["x0"] + obj["x1"]) / 2.0
                cy = (obj["top"] + obj["bottom"]) / 2.0
                return not self._inside_rect(cx, cy, table_bboxes)

            clipped = page.filter(_keep)
            prose = clipped.extract_text(
                x_tolerance=2, y_tolerance=2
            ) or ""

        else:
            prose = page.extract_text(
                x_tolerance=2, y_tolerance=2
            ) or ""

        # --- column-group fallback for line-less sub-tables ---------------

        col_groups = self._detect_column_groups(outside_words)

        if col_groups is not None:
            prose = self._render_words_with_columns(
                all_words, col_groups
            )

        # --- serialize detected (line-ruled) tables -----------------------

        parts = [prose.strip()]

        for table in kept_tables:
            block = self._serialize_table(table)
            if block:
                parts.append(block)

        combined = "\n\n".join(parts)

        # --- safety: any word swallowed by neither prose nor tables? -------

        combined_lower = combined.lower()
        lost = [
            w for w in outside_words
            if w["text"].lower() not in combined_lower
            and w["text"].strip()
        ]

        if lost:
            orphan_lines: Dict[int, List[str]] = {}
            for w in sorted(lost, key=lambda w: w["top"]):
                orphan_lines.setdefault(
                    round(w["top"], 1), []
                ).append(w["text"])
            orphan_text = "\n".join(
                " ".join(parts) for parts in orphan_lines.values()
            )
            combined += "\n\n" + orphan_text

        return combined.strip()

    def _kept_tables(self, page: Any) -> list:
        """Return tables after removing nested/contained bounding boxes."""

        tables = page.find_tables()
        if not tables:
            return []

        bboxes = [t.bbox for t in tables]
        kept = []

        for i, table in enumerate(tables):
            nested = any(
                self._rect_contains(bboxes[j], bboxes[i])
                for j in range(len(bboxes))
                if j != i
            )
            if not nested:
                kept.append(table)

        return kept

    @staticmethod
    def _rect_contains(outer: tuple, inner: tuple) -> bool:
        return (
            outer[0] - 1.0 <= inner[0]
            and outer[1] - 1.0 <= inner[1]
            and inner[2] <= outer[2] + 1.0
            and inner[3] <= outer[3] + 1.0
        )

    def _inside_rect(self, x: float, y: float, rects: list) -> bool:
        pad = self._TABLE_FILTER_PAD
        for x0, top, x1, bottom in rects:
            if x0 - pad <= x <= x1 + pad and top - pad <= y <= bottom + pad:
                return True
        return False

    def _inside_any_rect(self, word: dict, rects: list) -> bool:
        cx = (word["x0"] + word["x1"]) / 2.0
        cy = (word["top"] + word["bottom"]) / 2.0
        return self._inside_rect(cx, cy, rects)

    def _serialize_table(self, table: Any) -> str:
        rows = table.extract()
        lines = []
        for row in rows:
            cells = [
                re.sub(r"\s+", " ", (c or "")).strip()
                for c in row
            ]
            cells = [c for c in cells if c]
            if cells:
                lines.append(" | ".join(cells))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    #  Line-less column detection (Dutch-method lite)
    # ------------------------------------------------------------------

    def _detect_column_groups(
        self, words: list
    ) -> Optional[dict]:
        """
        For line-less tables (e.g. director lists), detect lines that have
        ≥2 column-aligned blocks and return them for pipe-delimited
        rendering.  Returns None if no multi-column group is found.
        """

        if len(words) < 8:
            return None

        line_map: Dict[int, list] = {}
        for w in words:
            line_map.setdefault(round(w["top"], 1), []).append(w)

        for lw in line_map.values():
            lw.sort(key=lambda w: w["x0"])

        # --- compute an adaptive inter-word gap threshold ---------------

        all_gaps = []
        for lw in line_map.values():
            for prev, nxt in zip(lw, lw[1:]):
                gap = nxt["x0"] - prev["x1"]
                if gap > 0:
                    all_gaps.append(gap)

        if not all_gaps:
            return None

        all_gaps.sort()
        median_gap = all_gaps[len(all_gaps) // 2]
        threshold = max(self._COL_GAP_TOLERANCE, 2.2 * median_gap)

        # --- split each line into column blocks -------------------------

        groups: Dict[tuple, list] = {}
        for top in sorted(line_map):
            lw = line_map[top]
            blocks = []
            current = [lw[0]]
            for prev, nxt in zip(lw, lw[1:]):
                gap = nxt["x0"] - prev["x1"]
                if gap > threshold:
                    blocks.append(current)
                    current = [nxt]
                else:
                    current.append(nxt)
            blocks.append(current)

            if len(blocks) < 2:
                continue

            starts = tuple(
                round(b[0]["x0"], -1) for b in blocks
            )
            signature = (len(blocks), starts)
            groups.setdefault(signature, []).append(
                (top, lw, blocks)
            )

        # --- keep only signatures shared across ≥ N lines ---------------

        selected: Dict[int, list] = {}
        for signature, entries in groups.items():
            if len(entries) >= self._COL_MIN_GROUP_SIZE:
                for top, _lw, blocks in entries:
                    selected[top] = blocks

        return selected if selected else None

    def _render_words_with_columns(
        self,
        all_words: list,
        column_groups: dict,
    ) -> str:
        line_map: Dict[int, list] = {}
        for w in all_words:
            line_map.setdefault(round(w["top"], 1), []).append(w)

        parts = []
        for top in sorted(line_map):
            lw = sorted(line_map[top], key=lambda w: w["x0"])
            if top in column_groups:
                col_texts = [
                    " ".join(w["text"] for w in block)
                    for block in column_groups[top]
                ]
                parts.append(" | ".join(col_texts))
            else:
                parts.append(" ".join(w["text"] for w in lw))

        return "\n".join(parts)

    # ------------------------------------------------------------------
    #  Chunking
    # ------------------------------------------------------------------

    def extract_facts(
        self,
        pdf_path: str,
        source_document: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[Fact]:

        source_document = source_document or pdf_path

        if use_cache:
            cached = self._load_fact_cache(
                pdf_path,
                source_document,
                cache_dir,
            )
            if cached is not None:
                return cached

        facts = self._extract_facts_uncached(pdf_path, source_document)

        if use_cache:
            self._save_fact_cache(
                pdf_path,
                source_document,
                facts,
                cache_dir,
            )

        return facts

    def _extract_facts_uncached(
        self,
        pdf_path: str,
        source_document: str,
    ) -> List[Fact]:

        pages = self.extract_text(pdf_path)

        # Extract PDF text first so that page/table reconstruction remains
        # deterministic. LLM calls are independent after this point.
        chunks = []
        for page_data in pages:
            page_number = page_data["page"]
            page_text = page_data["text"]
            text_source = page_data.get("source", "pdfplumber")

            if not page_text:
                continue

            chunks.extend(
                (page_number, chunk, text_source)
                for chunk in self._chunk_page(page_text)
            )

        total_chunks = len(chunks)
        self._log(f"extracting {total_chunks} chunks "
                  f"with {self.max_workers} worker(s)")

        if self.max_workers == 1:
            raw_results = []
            for idx, (_page_number, chunk, _source) in enumerate(chunks, start=1):
                self._log(f"  chunk {idx}/{total_chunks} ({len(chunk)} chars)")
                raw_results.append(
                    self._extract_facts_from_chunk(chunk)
                )
        else:
            # executor.map preserves the input order, so fact ordering and
            # subsequent deduplication stay stable even when calls finish in
            # a different order.
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                done = 0
                raw_results = []
                for result in pool.map(
                    self._extract_facts_from_chunk,
                    (chunk for _page_number, chunk, _source in chunks),
                ):
                    done += 1
                    if done % 5 == 0 or done == total_chunks:
                        self._log(f"  chunk {done}/{total_chunks}")
                    raw_results.append(result)

        facts: List[Fact] = []
        for (page_number, chunk, text_source), raw_facts in zip(
            chunks,
            raw_results,
        ):
            for raw in raw_facts:
                fact = self._build_fact(
                    raw,
                    chunk,
                    page_number,
                    source_document,
                    text_source=text_source,
                )

                if fact:
                    facts.append(fact)

        return self._deduplicate_facts(facts)

    # ------------------------------------------------------------------
    #  Disk cache for extracted facts
    # ------------------------------------------------------------------

    # Cached facts are derived solely from the PDF and extraction settings.
    # The caller's source label is metadata and must not cause a second
    # extraction of the same file (for example, ``rbi.pdf`` vs its original
    # filename in separate comparison tests).
    CACHE_VERSION = 3

    @staticmethod
    def _default_cache_dir() -> Path:
        return (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "fact_cache"
        )

    def _cache_signature(self) -> str:
        payload = {
            "version": self.CACHE_VERSION,
            "model": getattr(self.llm, "model", "unknown"),
            "max_chunk_chars": self.MAX_CHUNK_CHARS,
            "chunk_overlap": self.CHUNK_OVERLAP,
            "min_grounding_score": self.MIN_GROUNDING_SCORE,
            "min_quote_chars": self.MIN_QUOTE_CHARS,
            "prompt": EXTRACTION_SYSTEM_PROMPT,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

    def _cache_path(
        self,
        pdf_path: str,
        cache_dir: Optional[str],
    ) -> Path:
        directory = (
            Path(cache_dir)
            if cache_dir
            else self._default_cache_dir()
        )
        pdf_hash = self._pdf_sha256(pdf_path)
        key = pdf_hash + "_" + self._cache_signature()
        return directory / f"{key}.json"

    @staticmethod
    def _pdf_sha256(pdf_path: str) -> str:
        with open(pdf_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]

    def _load_fact_cache(
        self,
        pdf_path: str,
        source_document: str,
        cache_dir: Optional[str],
    ) -> Optional[List[Fact]]:
        path = self._cache_path(pdf_path, cache_dir)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if data.get("pdf_sha256") != self._pdf_sha256(pdf_path):
            return None
        if data.get("signature") != self._cache_signature():
            return None
        try:
            # Preserve the source label requested by this invocation.  The
            # cached extraction may have been created by another caller that
            # used a different display name for the identical PDF.
            return [
                Fact.model_validate_json(entry).model_copy(
                    update={"source_document": source_document}
                )
                for entry in data.get("facts", [])
            ]
        except Exception:
            return None

    def _save_fact_cache(
        self,
        pdf_path: str,
        source_document: str,
        facts: List[Fact],
        cache_dir: Optional[str],
    ) -> None:
        path = self._cache_path(pdf_path, cache_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pdf_sha256": self._pdf_sha256(pdf_path),
            "signature": self._cache_signature(),
            "facts": [
                fact.model_dump_json()
                for fact in facts
            ],
        }
        path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    def _chunk_page(self, text: str) -> List[str]:
        """
        Generic boundary-aware chunking.

        No knowledge of document content is used.
        """

        text = text.strip()

        if not text:
            return []

        if len(text) <= self.MAX_CHUNK_CHARS:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            target_end = min(
                start + self.MAX_CHUNK_CHARS,
                len(text),
            )

            if target_end >= len(text):
                chunk = text[start:].strip()

                if chunk:
                    chunks.append(chunk)

                break

            end = self._find_chunk_boundary(
                text,
                start,
                target_end,
            )

            if end <= start:
                end = target_end

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            next_start = max(
                start + 1,
                end - self.CHUNK_OVERLAP,
            )

            start = next_start

        return chunks

    def _find_chunk_boundary(
        self,
        text: str,
        start: int,
        target_end: int,
    ) -> int:

        candidate_text = text[start:target_end]

        # Prefer paragraph/line boundaries.
        newline_positions = [
            match.end()
            for match in re.finditer(
                r"\n+",
                candidate_text,
            )
        ]

        if newline_positions:
            return start + newline_positions[-1]

        # Then sentence boundaries.
        sentence_positions = [
            match.end()
            for match in re.finditer(
                r"[.!?](?=\s|$)",
                candidate_text,
            )
        ]

        if sentence_positions:
            return start + sentence_positions[-1]

        # Finally use a whitespace boundary.
        whitespace_positions = [
            match.end()
            for match in re.finditer(
                r"\s+",
                candidate_text,
            )
        ]

        if whitespace_positions:
            return start + whitespace_positions[-1]

        return target_end

    def _extract_facts_from_chunk(
        self,
        chunk: str,
    ) -> List[Dict[str, Any]]:

        try:
            result = self.llm.generate_json(
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                user_prompt=(
                    'TEXT:\n"""\n'
                    f"{chunk}"
                    '\n"""'
                ),
            )

        except (ValueError, RuntimeError) as exc:
            print(
                f"[extractor] ERROR chunk "
                f"({len(chunk)} chars): {exc}"
            )
            return []

        if isinstance(result, dict):
            facts = result.get("facts")

            if isinstance(facts, list):
                return [
                    item
                    for item in facts
                    if isinstance(item, dict)
                ]

            return []

        if isinstance(result, list):
            return [
                item
                for item in result
                if isinstance(item, dict)
            ]

        return []

    def _claim_supported_by_quote(
        self,
        claim: str,
        quote: str,
    ) -> bool:

        claim_tokens = {
            token.lower()
            for token in re.findall(r"\b\w+\b", claim)
            if len(token) > 2
        }

        quote_tokens = {
            token.lower()
            for token in re.findall(r"\b\w+\b", quote)
        }

        if not claim_tokens:
            return False

        overlap = (
            len(claim_tokens & quote_tokens)
            / len(claim_tokens)
        )

        return overlap >= 0.5

    @staticmethod
    def _extract_number_literal(text: str) -> Optional[str]:
        match = re.search(
            r"(?:₹|Rs\.?|USD|US\$|\$|€|£)?"
            r"\d{1,3}(?:,\d{3})*(?:\.\d+)?",
            text,
        )
        return match.group(0) if match else None

    @staticmethod
    def _numbers_equal(a: str, b: str) -> bool:
        def norm(s: str):
            m = re.search(
                r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?",
                s,
            )
            if not m:
                return None
            return float(m.group(0).replace(",", ""))
        na, nb = norm(a), norm(b)
        return na is not None and nb is not None and na == nb

    _PLACEHOLDER_RE = re.compile(
        r"\[.+\]|{.+}|<.+>|XXX+|\*{2,}|"
        r"\b(?:TBD|TODO|PLACEHOLDER|REDACTED|MASKED|N\.?A\.?)\b|"
        r"XXXX[\-\/]?XXXX[\-\/]?XXXX|"
        r"\*{2,}[\-\/]?\*{2,}[\-\/]?\*{2,}|"
        r"\d{4}[\-\/]\*{4}[\-\/]\*{4}",
        re.IGNORECASE,
    )

    _CREDENTIAL_RE = re.compile(
        r"(?:password|passwd|pwd|secret|token|api[_\s]?key|"
        r"access[_\s]?key|auth[_\s]?key|private[_\s]?key|"
        r"bearer|credential|login|authorization)\s*[:=]\s*\S+"
        r"|\bBearer\s+[A-Za-z0-9._~+/=-]+"
        r"|\b(?:jwt|oauth|csrf)[\s_-]*token\b",
        re.IGNORECASE,
    )

    _DOCUMENT_META_RE = re.compile(
        r"^(?:page|folio|sheet)\s*\d+\s*(?:of\s*\d+)?\s*$|"
        r"^(?:footer|header|watermark)\s*[:\-]?\s*\S*$|"
        r"^(?:draft|final|version|rev\.?|v\d+)\s*$",
        re.IGNORECASE,
    )

    _LABEL_TITLE_RE = re.compile(
        r"^(?:section|chapter|part|item|clause|page|article|appendix|"
        r"annexure|annex|figure|table|column|row|field|title|heading|"
        r"schedule|exhibit|entry|line)\s+[0-9IVX]+$",
        re.IGNORECASE,
    )

    _REFERENCE_RE = re.compile(
        r"^(?:(?:registration|reg(?:istered)?|ref(?:erence)?|serial"
        r"|account|invoice|policy|voucher|bill|file|case|permit|license|"
        r"licence|passport|roll|seat|token|tracking|order|transaction|"
        r"certificate|document|folio|employee|investor|report)\s+)?"
        r"(?:#|no\.?|number|id|code|identifier)\s*[:#]?\s*"
        r"[A-Za-z0-9][A-Za-z0-9\-/._]{2,}$",
        re.IGNORECASE,
    )

    _STOP_WORDS = frozenset({
        "the", "a", "an", "of", "and", "or", "for", "to",
        "in", "on", "by", "at", "is", "are", "was", "were",
        "be", "been", "being", "has", "had", "have", "do",
        "does", "did", "will", "would", "could", "should",
        "may", "might", "shall", "can", "that", "this",
        "with", "from", "as", "its", "it", "not", "no",
    })

    def _is_meaningful_claim(self, claim: str) -> bool:
        """
        Cheap, document-agnostic importance gate. Rejects standalone
        identifiers, boilerplate, placeholders, credentials, masked
        values, and single-token labels that carry no proposition,
        while keeping real facts. Heuristics apply to all documents
        equally.
        """
        if not claim:
            return False

        claim_stripped = claim.strip(" \t\n·•-–—")
        if len(claim_stripped) < 6:
            return False

        # Reject placeholder / template text.
        if self._PLACEHOLDER_RE.search(claim_stripped):
            return False

        # Reject credential-like values.
        if self._CREDENTIAL_RE.search(claim_stripped):
            return False

        # Reject document metadata (page numbers, headers, etc.).
        if self._DOCUMENT_META_RE.match(claim_stripped):
            return False

        # Reject isolated section/column-style labels ("Section 4",
        # "Table 2", "Part III").
        if self._LABEL_TITLE_RE.match(claim_stripped):
            return False

        # A claim must state a proposition: require content words beyond
        # stop words.
        content_words = [
            w for w in re.findall(r"\b[A-Za-z]{2,}\b", claim_stripped)
            if w.lower() not in self._STOP_WORDS
        ]

        if not content_words:
            return False

        # Bare identifiers (CINs, codes, serials): dominated by non-word
        # characters and digits, or a single long alphanumeric token.
        words = re.findall(r"\S+", claim_stripped)
        if len(words) <= 2:
            alpha_tokens = re.findall(r"[A-Za-z]{2,}", " ".join(words))
            if len(alpha_tokens) <= 1 and not any(
                len(t) > 4 for t in words if t.isalpha()
            ):
                return False

        # Registration/reference-style values: a reference label followed
        # by a code, with no predicate ("Ref No: 1234", "Invoice #A-99").
        if self._REFERENCE_RE.match(claim_stripped):
            return False

        if re.fullmatch(
            r"[A-Za-z0-9/\-]{5,}", claim_stripped
        ):
            return False

        # Pure numeric or alphanumeric codes with no semantic content.
        if re.fullmatch(
            r"[\d\s,\.\-/()+:]+", claim_stripped
        ):
            return False

        return True

    def _build_fact(
        self,
        raw: Any,
        chunk: str,
        page_number: int,
        source_document: str,
        text_source: str = "pdfplumber",
    ) -> Optional[Fact]:

        # A single malformed fact must never take down extraction of the
        # rest of the chunk/document. Reject it individually and continue.
        try:
            return self._build_fact_or_none(
                raw,
                chunk,
                page_number,
                source_document,
                text_source,
            )
        except Exception as exc:
            print(
                f"[extractor] Rejected invalid fact "
                f"({exc}); continuing."
            )
            return None

    def _build_fact_or_none(
        self,
        raw: Any,
        chunk: str,
        page_number: int,
        source_document: str,
        text_source: str = "pdfplumber",
    ) -> Optional[Fact]:

        if not isinstance(raw, dict):
            return None

        claim = str(
            raw.get("claim")
            or raw.get("text")
            or ""
        ).strip()

        quote = str(
            raw.get("verbatim_quote")
            or raw.get("evidence")
            or ""
        ).strip()

        if not claim or not quote:
            return None

        if len(quote) < self.MIN_QUOTE_CHARS:
            return None

        grounded_snippet, score = ground_quote(
            quote,
            chunk,
            self.MIN_GROUNDING_SCORE,
        )

        if score < self.MIN_GROUNDING_SCORE:
            return None

        if not grounded_snippet:
            return None

        if not self._claim_supported_by_quote(
            claim,
            grounded_snippet,
        ):
            # A bare numeric token (fallback grounding) has no surrounding
            # words to match the claim's prose, yet it is a valid anchor.
            # Accept only if the claim actually contains that same number.
            number_in_claim = self._extract_number_literal(claim)
            number_in_snippet = self._extract_number_literal(
                grounded_snippet
            )
            if not (
                number_in_claim
                and number_in_snippet
                and self._numbers_equal(
                    number_in_claim, number_in_snippet
                )
            ):
                return None

        # --- importance / trivia filter --------------------------------
        # Document-agnostic: reject standalone identifiers, boilerplate,
        # placeholders, masked values, and fragments that do not form a
        # complete, meaningful proposition.
        if not self._is_meaningful_claim(claim):
            return None

        fact_type = self._normalize_fact_type(
            raw.get("fact_type")
        )

        value = raw.get("value")

        normalized_value = self._normalize_value(
            value,
            fact_type,
            grounded_snippet,
        )

        # --- value/fact-type compatibility -----------------------------
        # An accepted fact must carry a value that is consistent with its
        # declared fact type (numbers for numeric, ISO dates for date,
        # HH:MM for time, booleans for boolean, text for semantic/entity).
        if not self._value_compatible_with_type(
            normalized_value,
            fact_type,
            grounded_snippet,
        ):
            return None

        try:
            llm_confidence = float(
                raw.get("confidence", 0.6)
            )
        except (TypeError, ValueError):
            llm_confidence = 0.6

        llm_confidence = max(
            0.0,
            min(1.0, llm_confidence),
        )

        # Blend the model's stated confidence with the objective grounding
        # score so confidence reflects real evidence, not a constant.
        grounding_confidence = score / 100.0
        confidence = round(
            min(llm_confidence, grounding_confidence),
            2,
        )

        context: Dict[str, Any] = {}

        if text_source != "pdfplumber":
            context["text_source"] = text_source

        if value not in (None, ""):
            context["raw_value"] = value

        for field in (
            "entity",
            "time_period",
            "scope",
        ):
            field_value = raw.get(field)

            if field_value not in (
                None,
                "",
                [],
                {},
            ):
                context[field] = field_value

        # Preserve future fields without requiring schema changes.
        for key, extra_value in raw.items():
            if (
                key not in self._KNOWN_RAW_KEYS
                and extra_value not in (
                    None,
                    "",
                    [],
                    {},
                )
            ):
                context[key] = extra_value

        return Fact(
            id=str(uuid.uuid4()),
            text=claim,
            fact_type=fact_type,
            value=normalized_value,
            unit=self._clean_string(
                raw.get("unit")
            ),
            context=context,
            source_document=source_document,
            source_page=page_number,
            source_snippet=grounded_snippet,
            confidence=confidence,
            extraction_method="llm",
            created_at=datetime.now(),
        )

    def _value_compatible_with_type(
        self,
        value: Any,
        fact_type: FactType,
        snippet: Optional[str] = None,
    ) -> bool:
        """
        Generic, document-agnostic check that a normalized value is
        consistent with its declared fact type. A missing value is allowed
        only after the fact has already passed grounding + meaningfulness.
        """
        if fact_type == FactType.NUMERIC:
            if isinstance(value, (int, float)):
                return True
            if value is None:
                # _normalize_value re-derives numeric facts from the
                # grounded snippet; if neither the snippet nor the model
                # supplied a number, the value is incompatible.
                return bool(
                    snippet
                    and self._extract_numeric_scalar(snippet) is not None
                )
            return (
                isinstance(value, str)
                and self._extract_numeric_scalar(value) is not None
            )

        if fact_type == FactType.DATE:
            return isinstance(value, str) and bool(
                re.fullmatch(
                    r"\d{4}-\d{2}-\d{2}", value
                )
            ) and self._extract_date_scalar(value) == value

        if fact_type == FactType.TIME:
            return isinstance(value, str) and bool(
                re.fullmatch(
                    r"\d{2}:\d{2}(?:-\d{2}:\d{2})?",
                    value,
                )
            )

        if fact_type == FactType.BOOLEAN:
            return isinstance(value, bool)

        # Semantic / entity facts are textual by nature.
        if fact_type in (FactType.SEMANTIC, FactType.ENTITY):
            return value is None or (
                isinstance(value, str) and bool(value.strip())
            )

        return True

    def _normalize_fact_type(
        self,
        value: Any,
    ) -> FactType:

        if isinstance(value, str):
            normalized = value.strip().lower()

            for member in FactType:
                if member.value == normalized:
                    return member

        return FactType.SEMANTIC

    def _clean_string(
        self,
        value: Any,
    ) -> Optional[str]:

        if value is None:
            return None

        value = str(value).strip()

        return value if value else None

    def _normalize_value(
        self,
        value: Any,
        fact_type: FactType,
        snippet: Optional[str] = None,
    ) -> Any:
        """
        Collapse an LLM's verbose value string into a clean scalar where
        the fact type clearly implies one.  For numeric facts the value is
        re-derived from the grounded (verbatim) snippet rather than trusted
        from the model, because models often convert or scale the number
        (e.g. multiply by "million") and return a wrong magnitude.  The
        original model value is preserved in context['raw_value'].  Fully
        document- and domain-agnostic.
        """
        if fact_type == FactType.NUMERIC:
            if snippet:
                from_snippet = self._extract_numeric_scalar(snippet)
                if from_snippet is not None:
                    return from_snippet
            if value is not None:
                from_value = self._extract_numeric_scalar(value)
                if from_value is not None:
                    return from_value
            return value

        if fact_type == FactType.DATE:
            for candidate in (value, snippet):
                if candidate in (None, ""):
                    continue
                parsed = self._extract_date_scalar(candidate)
                if (
                    isinstance(parsed, str)
                    and re.fullmatch(r"\d{4}-\d{2}-\d{2}", parsed)
                ):
                    return parsed
            return value

        if fact_type == FactType.TIME:
            for candidate in (value, snippet):
                if candidate in (None, ""):
                    continue
                parsed = self._extract_time_scalar(candidate)
                if (
                    isinstance(parsed, str)
                    and re.fullmatch(
                        r"\d{2}:\d{2}(?:-\d{2}:\d{2})?", parsed
                    )
                ):
                    return parsed
            return value

        if fact_type == FactType.BOOLEAN:
            return self._extract_boolean_scalar(value)

        return value

    @staticmethod
    def _extract_boolean_scalar(value: Any) -> Any:
        if isinstance(value, bool):
            return value
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in ("true", "yes", "1", "positive", "present", "active"):
            return True
        if text in ("false", "no", "0", "negative", "absent", "inactive"):
            return False
        return value

    @staticmethod
    def _extract_time_scalar(value: Any) -> Optional[str]:
        """
        Normalize a time value or time range to a machine-readable string.
        Single times become HH:MM (24h). Ranges become HH:MM-HH:MM.
        """
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None

        def _to_24h(hour_str: str, minute_str: str, ampm: str) -> Optional[str]:
            try:
                hour = int(hour_str)
            except (TypeError, ValueError):
                return None
            minute = int(minute_str) if minute_str else 0
            ampm = (ampm or "").upper()
            if ampm == "AM":
                if hour == 12:
                    hour = 0
            elif ampm == "PM":
                if hour != 12:
                    hour += 12
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return f"{hour:02d}:{minute:02d}"
            return None

        _TIME_PATTERN = (
            r"(\d{1,2})(?::(\d{2}))?"
            r"\s*(AM|PM|am|pm)?"
        )
        _SEPARATOR = (
            r"(?:\s*[-–—~]\s*"
            r"|\s+(?:to|until|through|from)\s+)"
        )

        # ISO-like 24h times ("14:30", "09:15:00").
        iso_match = re.fullmatch(
            r"(\d{1,2}):(\d{2})(?::(\d{2}))?",
            text,
        )
        if iso_match:
            hour, minute = int(iso_match.group(1)), int(iso_match.group(2))
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return f"{hour:02d}:{minute:02d}"

        range_match = re.search(
            _TIME_PATTERN + _SEPARATOR + _TIME_PATTERN,
            text,
            flags=re.IGNORECASE,
        )
        if range_match:
            t1 = _to_24h(
                range_match.group(1),
                range_match.group(2),
                range_match.group(3),
            )
            t2 = _to_24h(
                range_match.group(4),
                range_match.group(5),
                range_match.group(6),
            )
            if t1 and t2:
                return f"{t1}-{t2}"

        single_match = re.search(_TIME_PATTERN, text)
        if single_match:
            t = _to_24h(
                single_match.group(1),
                single_match.group(2),
                single_match.group(3),
            )
            if t:
                return t

        return text

    @staticmethod
    def _extract_numeric_scalar(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str):
            return None

        text = value.strip()
        if not text:
            return None

        if text.startswith("("):
            negative = True
            text = text.strip("()")
        elif text.startswith("-"):
            negative = True
            text = text[1:]
        else:
            negative = False

        match = re.search(
            r"\d{1,3}(?:,\d{3})*(?:\.\d+)?",
            text,
        )
        if not match:
            return None

        cleaned = match.group(0).replace(",", "")
        try:
            number = float(cleaned)
        except ValueError:
            return None

        # Apply scale qualifiers ("crore", "lakh", "million", "billion")
        # found anywhere in the string, so that "52.35 million" becomes
        # 52,350,000 rather than 52.35. Mapping:
        #   crore -> 10^7, lakh -> 10^5, million -> 10^6, billion -> 10^9
        scale = 1.0
        lower = text.lower()
        for word, multiplier in (
            ("billion", 1_000_000_000.0),
            ("million", 1_000_000.0),
            ("crore", 10_000_000.0),
            ("lakh", 100_000.0),
        ):
            if re.search(r"\b" + word + r"\b", lower):
                scale = multiplier
                break

        number *= scale

        return -number if negative else number

    @staticmethod
    def _extract_date_scalar(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return value if value is not None else None

        text = value.strip()
        if not text:
            return None

        # YYYY-MM-DD (already normalized).
        iso_match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
        if iso_match:
            y, mo, d = int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))
            if 1 <= mo <= 12 and 1 <= d <= 31:
                return f"{y:04d}-{mo:02d}-{d:02d}"

        # YYYY/MM/DD, YYYY.MM.DD, DD/MM/YYYY, DD.MM.YYYY, MM/DD/YYYY.
        m = re.search(
            r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})"
            r"|(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})",
            text,
        )
        if m:
            if m.group(1):
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            else:
                d, mo, y = int(m.group(4)), int(m.group(5)), int(m.group(6))
            if 1 <= mo <= 12 and 1 <= d <= 31:
                return f"{y:04d}-{mo:02d}-{d:02d}"

        # "14 March 2012", "March 14, 2012", "Mar 14 2012", etc.
        months = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ]
        month_map = {}
        for index, name in enumerate(months):
            month_map[name] = index + 1
            month_map[name[:3]] = index + 1

        day_match = re.search(
            r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
            r"([A-Za-z]+)\s+(\d{4})\b",
            text,
        )
        if day_match:
            d = int(day_match.group(1))
            mo = month_map.get(day_match.group(2).lower())
            y = int(day_match.group(3))
            if mo and 1 <= d <= 31:
                return f"{y:04d}-{mo:02d}-{d:02d}"

        mon_day = re.search(
            r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
            text,
        )
        if mon_day:
            mo = month_map.get(mon_day.group(1).lower())
            d = int(mon_day.group(2))
            y = int(mon_day.group(3))
            if mo and 1 <= d <= 31:
                return f"{y:04d}-{mo:02d}-{d:02d}"

        # "FY2024", "FY 2024", "FY24" — fiscal years are not dates;
        # return as-is so the raw value is preserved.
        return text

    def _deduplicate_facts(
        self,
        facts: List[Fact],
    ) -> List[Fact]:

        unique = []
        seen = set()

        for fact in facts:
            doc = fact.source_document
            page = fact.source_page
            ftype = fact.fact_type.value
            normalized_value = self._normalize_text(
                str(fact.value)
            )
            normalized_snippet = self._normalize_text(
                fact.source_snippet
            )

            # Exact composite key first (fast path).
            exact_key = (
                doc, page, ftype,
                normalized_value,
                normalized_snippet,
            )

            if exact_key in seen:
                continue

            # Fuzzy pass: collapse near-duplicate snippets (same doc/page/type
            # and a value token in common) that differ only by whitespace or
            # minor table-cell wrapping.
            duplicate = False
            for other in unique:
                if (
                    other.source_document != doc
                    or other.source_page != page
                    or other.fact_type.value != ftype
                ):
                    continue
                other_value = self._normalize_text(str(other.value))
                other_snippet = self._normalize_text(
                    other.source_snippet
                )
                if (
                    normalized_value
                    and normalized_value == other_value
                ):
                    similarity = fuzz.ratio(
                        normalized_snippet,
                        other_snippet,
                    )
                    if similarity >= 95.0:
                        duplicate = True
                        break

            if duplicate:
                continue

            seen.add(exact_key)
            unique.append(fact)

        return unique

    def _normalize_text(
        self,
        text: str,
    ) -> str:

        return re.sub(
            r"\s+",
            " ",
            str(text).lower(),
        ).strip()
