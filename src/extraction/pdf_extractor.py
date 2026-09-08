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
- fact_type: one of "numeric", "semantic", "date", "boolean", or "entity"
- value: the number or value if the fact contains one, otherwise null
- unit: the unit of measurement if present, otherwise null
- verbatim_quote: an exact quote from the text that supports the claim
- confidence: a number 0.0 to 1.0 reflecting how directly the text states the fact (1.0 = explicitly stated; lower = inferred/uncertain). Do not set 1.0 unless the claim is stated almost verbatim.

Rules:
- Extract only facts that are directly and explicitly stated in the text.
- Extract facts that convey meaningful information. Skip isolated labels, standalone identifiers (e.g. registration numbers, reference codes, CINs), boilerplate, and fragments that do not state a complete proposition.
- The verbatim_quote must be an exact substring of the supplied text — copy it character-for-character.
- Do not paraphrase the quote — use the original words.
- You may extract multiple facts from a single text block.
- Preserve qualifiers like "approximately", "as of", "net of", etc.

Return your answer as JSON in this exact format:
{"facts": [{"claim": "...", "fact_type": "...", "value": null, "unit": null, "verbatim_quote": "...", "confidence": 0.9}]}

Example — given this text:
"Alpha Corp was founded on 14 March 2012. Its annual output was 85,000 units in fiscal year 2023."

You would return:
{"facts": [
  {"claim": "Alpha Corp was founded on 14 March 2012.", "fact_type": "date", "value": "2012-03-14", "unit": null, "verbatim_quote": "Alpha Corp was founded on 14 March 2012.", "confidence": 1.0},
  {"claim": "Alpha Corp's annual output was 85,000 units in fiscal year 2023.", "fact_type": "numeric", "value": 85000, "unit": "units", "verbatim_quote": "Its annual output was 85,000 units in fiscal year 2023.", "confidence": 1.0}
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
        verbose = os.getenv("PDF_EXTRACTION_VERBOSE", "0") == "1"
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

    def extract_text(self, pdf_path: str) -> List[Dict[str, Any]]:
        pages = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
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

                    pages.append(
                        {
                            "page": page_number,
                            "text": text.strip(),
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

            if not page_text:
                continue

            chunks.extend(
                (page_number, chunk)
                for chunk in self._chunk_page(page_text)
            )

        total_chunks = len(chunks)
        self._log(f"extracting {total_chunks} chunks "
                  f"with {self.max_workers} worker(s)")

        if self.max_workers == 1:
            raw_results = []
            for idx, (_page_number, chunk) in enumerate(chunks, start=1):
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
                    (chunk for _page_number, chunk in chunks),
                ):
                    done += 1
                    if done % 5 == 0 or done == total_chunks:
                        self._log(f"  chunk {done}/{total_chunks}")
                    raw_results.append(result)

        facts: List[Fact] = []
        for (page_number, chunk), raw_facts in zip(chunks, raw_results):
            for raw in raw_facts:
                fact = self._build_fact(
                    raw,
                    chunk,
                    page_number,
                    source_document,
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
    CACHE_VERSION = 2

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

    def _is_meaningful_claim(self, claim: str) -> bool:
        """
        Cheap, document-agnostic importance gate. Rejects standalone
        identifiers / single-token labels that carry no proposition, while
        keeping real facts. Heuristics apply to all documents equally.
        """
        if not claim:
            return False

        claim_stripped = claim.strip(" \t\n·•-–")
        if len(claim_stripped) < 6:
            return False

        # A claim must state a proposition: require a verb-like predicate.
        # Standalone identifiers / labels have almost no content words.
        content_words = [
            w for w in re.findall(r"\b[A-Za-z]{2,}\b", claim_stripped)
            if w.lower()
            not in {
                "the", "a", "an", "of", "and", "or", "for", "to",
                "in", "on", "by", "at", "is", "are", "was", "were",
            }
        ]

        if not content_words:
            return False

        # Bare identifiers (CINs, codes, serials): dominated by non-word
        # characters and digits, or a single long alphanumeric token.
        words = re.findall(r"\S+", claim_stripped)
        if len(words) <= 2:
            alnum_token = re.findall(r"[A-Za-z]{2,}", " ".join(words))
            if len(alnum_token) <= 1 and not any(
                len(t) > 4 for t in words if t.isalpha()
            ):
                return False

        # Registration/reference-style values: mostly digits/codes with no
        # predicate.
        if re.fullmatch(
            r"^(?:registration\s+)?number\s*[:#]?\s*"
            r"[A-Za-z0-9\-/]{3,}$",
            claim_stripped,
            flags=re.IGNORECASE,
        ):
            return False

        if re.fullmatch(
            r"[A-Za-z0-9/\-]{5,}", claim_stripped
        ):
            return False

        return True

    def _build_fact(
        self,
        raw: Any,
        chunk: str,
        page_number: int,
        source_document: str,
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
        # Document-agnostic: reject standalone identifiers and single-token
        # labels that do not form a complete, meaningful proposition.
        if not self._is_meaningful_claim(claim):
            return None

        value = raw.get("value")

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
            fact_type=self._normalize_fact_type(
                raw.get("fact_type")
            ),
            value=self._normalize_value(
                value,
                self._normalize_fact_type(
                    raw.get("fact_type")
                ),
                grounded_snippet,
            ),
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
            date_value = self._extract_date_scalar(value)
            if date_value is not None:
                return date_value

        return value

    @staticmethod
    def _extract_numeric_scalar(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str):
            return None

        text = value.strip()
        if not text:
            return None

        if text.startswith(("(", "-")):
            text = text.strip("()")
            if text.startswith("-"):
                negative = True
                text = text[1:]
            else:
                negative = True
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
            return value

        m = re.search(
            r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})"
            r"|(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})",
            value,
        )
        if m:
            if m.group(1):
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            else:
                d, mo, y = int(m.group(4)), int(m.group(5)), int(m.group(6))
            return f"{y:04d}-{mo:02d}-{d:02d}"

        months = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ]
        month_map = {name: i + 1 for i, name in enumerate(months)}
        day_match = re.search(
            r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
            r"([A-Za-z]+)\s+(\d{4})\b",
            value,
        )
        if day_match:
            d = int(day_match.group(1))
            mo = month_map.get(day_match.group(2).lower())
            y = int(day_match.group(3))
            if mo:
                return f"{y:04d}-{mo:02d}-{d:02d}"

        return value

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
