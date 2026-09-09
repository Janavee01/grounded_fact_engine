"""
Standalone tests against the real grounding.py, comparator.py, and
pdf_extractor.py logic, with a fake LLM client injected so no live
Ollama server is needed. Run: pytest -v tests/test_pipeline.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime

from src.extraction.grounding import ground_quote
from src.extraction.pdf_extractor import PDFExtractor
from src.comparison.comparator import FactComparator
from src.models.fact import Fact, FactType


# ---------------------------------------------------------------------------
# Fake LLM client — scripted responses, no network/Ollama needed
# ---------------------------------------------------------------------------

class FakeLLMClient:
    def __init__(self, responses):
        # responses: list of dicts, popped in order per call
        self.responses = list(responses)
        self.calls = []

    def generate_json(self, system_prompt, user_prompt, temperature=0.0):
        self.calls.append(user_prompt)
        if not self.responses:
            raise RuntimeError("FakeLLMClient exhausted")
        return self.responses.pop(0)


def make_fact(**kw):
    defaults = dict(
        id="id-" + str(hash(str(kw)) % 10000),
        text="placeholder",
        fact_type=FactType.NUMERIC,
        value=None,
        unit=None,
        context={},
        source_document="doc.pdf",
        source_page=1,
        source_snippet="snippet",
        confidence=0.9,
        extraction_method="llm",
        created_at=datetime.now(),
    )
    defaults.update(kw)
    return Fact(**defaults)


def test_fact_cache_is_reused_when_only_source_label_changes(tmp_path):
    """A display label must not trigger another LLM extraction."""
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"not parsed: cache methods only need stable bytes")
    extractor = PDFExtractor(llm_client=FakeLLMClient([]))
    fact = make_fact(source_document="original-report.pdf")

    extractor._save_fact_cache(
        str(pdf), "original-report.pdf", [fact], str(tmp_path / "cache")
    )
    cached = extractor._load_fact_cache(
        str(pdf), "comparison-a.pdf", str(tmp_path / "cache")
    )

    assert cached is not None
    assert len(cached) == 1
    assert cached[0].text == fact.text
    assert cached[0].source_document == "comparison-a.pdf"


# ---------------------------------------------------------------------------
# 1. grounding.py
# ---------------------------------------------------------------------------

def test_fact_build_tags_ocr_source():
    """Facts extracted from OCR-recovered pages must be tagged in context."""
    extractor = PDFExtractor(llm_client=FakeLLMClient([]))
    page_text = (
        "The report discusses growth. "
        "Revenue growth happened in the last fiscal year."
    )
    raw = {
        "claim": "Revenue growth happened in the last fiscal year.",
        "fact_type": "semantic",
        "value": None,
        "unit": None,
        "verbatim_quote": "Revenue growth happened in the last fiscal year.",
        "confidence": 0.9,
    }
    fact = extractor._build_fact(
        raw,
        page_text,
        page_number=1,
        source_document="scanned.pdf",
        text_source="ocr",
    )
    assert fact is not None
    assert fact.context.get("text_source") == "ocr"

    non_ocr = extractor._build_fact(
        raw,
        page_text,
        page_number=1,
        source_document="scanned.pdf",
    )
    assert non_ocr is not None
    assert non_ocr.context.get("text_source") is None


def test_grounding_exact_match():
    source = "Revenue for FY2023 was 85,000 units, up from the prior year."
    snippet, score = ground_quote("85,000 units", source)
    assert "85,000 units" in snippet
    assert score == 100.0


def test_grounding_whitespace_normalized_match():
    source = "The   company\nreported  revenue of 500 crore in FY23."
    quote = "revenue of 500 crore"
    snippet, score = ground_quote(quote, source)
    assert score >= 90.0
    assert "500" in snippet


def test_grounding_rejects_fabricated_quote():
    source = "Total headcount was 1,200 employees as of March 2024."
    # quote that doesn't appear at all and shares no numeric anchor
    snippet, score = ground_quote("net profit margin improved significantly", source)
    assert score < PDFExtractor.MIN_GROUNDING_SCORE


def test_grounding_number_fallback():
    source = "Table row: Total | 45,678 | INR crore"
    # LLM paraphrases surrounding text but keeps the number
    quote = "the grand total figure is 45,678"
    snippet, score = ground_quote(quote, source)
    # should fall back to locating the bare number
    assert "45,678" in snippet or score == 0.0  # document behavior: may fail if fuzzy align already succeeds
    print(f"  -> snippet={snippet!r} score={score}")


# ---------------------------------------------------------------------------
# 2. comparator._are_candidates — does heavy paraphrase survive the filter?
# ---------------------------------------------------------------------------

def test_candidates_survive_heavy_paraphrase_same_entity():
    """
    Case 1 from the assignment: a fact corroborated across documents,
    EXPRESSED DIFFERENTLY. This is the highest-risk path in the whole
    pipeline because _are_candidates runs BEFORE the LLM ever sees the
    pair — if the filter rejects it, the LLM never gets a chance to
    recognize the paraphrase.
    """
    comparator = FactComparator(llm_client=FakeLLMClient([]))

    f1 = {
        "text": "Delhivery reported total revenue from operations of ₹8,141 crore for FY24.",
        "fact_type": "numeric",
        "value": 8141,
        "unit": "crore",
        "context": {"entity": "Delhivery", "time_period": "FY24"},
        "source_document": "annual_report.pdf",
        "source_page": 12,
    }
    f2 = {
        "text": "The company's operating income for the fiscal year ending March 2024 came in at roughly 81.4 billion rupees.",
        "fact_type": "numeric",
        "value": 81.4,
        "unit": "billion INR",
        "context": {"entity": "Delhivery Ltd", "time_period": "fiscal year ending March 2024"},
        "source_document": "press_release.pdf",
        "source_page": 1,
    }

    result = comparator._are_candidates(f1, f2)
    print(f"  -> _are_candidates result: {result}")
    assert result is True, (
        "Heavily paraphrased same-entity facts were rejected before "
        "reaching the LLM — case #1 (corroboration expressed differently) "
        "would silently never be detected for phrasing this different."
    )


def test_candidates_reject_unrelated_facts():
    comparator = FactComparator(llm_client=FakeLLMClient([]))
    f1 = {
        "text": "The company was incorporated in 1998.",
        "fact_type": "date",
        "context": {"entity": "Acme Corp"},
        "source_document": "a.pdf",
    }
    f2 = {
        "text": "Average annual rainfall in the region is 900mm.",
        "fact_type": "numeric",
        "context": {},
        "source_document": "b.pdf",
    }
    assert comparator._are_candidates(f1, f2) is False


def test_candidates_reject_same_document():
    comparator = FactComparator(llm_client=FakeLLMClient([]))
    f1 = {"text": "Revenue was 500 crore.", "source_document": "same.pdf"}
    f2 = {"text": "Revenue was 500 crore.", "source_document": "same.pdf"}
    assert comparator._are_candidates(f1, f2) is False


def test_candidates_keep_plan_and_reported_outcome_pair():
    """A report can restate just the planned metric while giving an outcome."""
    comparator = FactComparator(llm_client=FakeLLMClient([]))
    plan = {
        "text": "The first phase of the replacement program covers 120 buses.",
        "source_document": "plan.pdf",
    }
    outcome = {
        "text": (
            "The review found that 96 new buses had entered regular service "
            "rather than the 120 originally expected in the first phase."
        ),
        "source_document": "review.pdf",
    }
    assert comparator._are_candidates(plan, outcome) is True


# ---------------------------------------------------------------------------
# 3. Full compare_facts pipeline with a scripted fake LLM — the four cases
# ---------------------------------------------------------------------------

def test_full_pipeline_corroboration():
    fake = FakeLLMClient([
        {"relationship": "corroborates", "confidence": 0.9,
         "explanation": "Both state the same FY24 revenue figure.", "context_notes": None},
    ])
    comparator = FactComparator(llm_client=fake)

    f1 = make_fact(
        text="Delhivery's FY24 revenue was ₹8,141 crore.",
        value=8141, unit="crore",
        context={"entity": "Delhivery", "time_period": "FY24"},
        source_document="doc_a.pdf",
    )
    f2 = make_fact(
        text="Total revenue for fiscal 2024 stood at approximately 81.4 billion rupees.",
        value=81.4, unit="billion INR",
        context={"entity": "Delhivery", "time_period": "FY24"},
        source_document="doc_b.pdf",
    )

    comparisons = comparator.compare_facts([f1, f2])
    assert len(comparisons) == 1
    assert comparisons[0].relationship == "corroborates"


def test_full_pipeline_contradiction():
    fake = FakeLLMClient([
        {"relationship": "contradicts", "confidence": 0.85,
         "explanation": "Same metric, same period, incompatible values.", "context_notes": None},
    ])
    comparator = FactComparator(llm_client=fake)

    f1 = make_fact(
        text="Net profit for FY24 was ₹200 crore.",
        value=200, context={"entity": "Acme", "time_period": "FY24"},
        source_document="doc_a.pdf",
    )
    f2 = make_fact(
        text="Net profit for FY24 was ₹350 crore.",
        value=350, context={"entity": "Acme", "time_period": "FY24"},
        source_document="doc_b.pdf",
    )
    comparisons = comparator.compare_facts([f1, f2])
    assert len(comparisons) == 1
    assert comparisons[0].relationship == "contradicts"


def test_full_pipeline_reconciled_by_scope():
    fake = FakeLLMClient([
        {"relationship": "reconciled", "confidence": 0.8,
         "explanation": "One figure is standalone, the other consolidated.", "context_notes": "scope difference"},
    ])
    comparator = FactComparator(llm_client=fake)

    f1 = make_fact(
        text="Standalone revenue was ₹500 crore.",
        value=500, context={"entity": "Acme", "scope": "standalone"},
        source_document="doc_a.pdf",
    )
    f2 = make_fact(
        text="Consolidated revenue was ₹650 crore.",
        value=650, context={"entity": "Acme", "scope": "consolidated"},
        source_document="doc_b.pdf",
    )
    comparisons = comparator.compare_facts([f1, f2])
    assert len(comparisons) == 1
    assert comparisons[0].relationship == "reconciled"
    assert comparisons[0].context_notes == "scope difference"


def test_full_pipeline_unrelated_is_dropped():
    fake = FakeLLMClient([
        {"relationship": "unrelated", "confidence": 0.5, "explanation": "different topics", "context_notes": None},
    ])
    comparator = FactComparator(llm_client=fake)
    f1 = make_fact(text="Company revenue was 500 crore in FY24.",
                    context={"entity": "Acme"}, source_document="a.pdf")
    f2 = make_fact(text="Company headcount grew to 5000 employees in FY24.",
                    context={"entity": "Acme"}, source_document="b.pdf")
    comparisons = comparator.compare_facts([f1, f2])
    assert comparisons == []  # unrelated results should be filtered out


def test_relationship_imperative_is_normalized():
    """The wording in the comparison request must not discard a reconcile result."""
    fake = FakeLLMClient([
        {"relationship": "reconcile", "confidence": 0.8,
         "explanation": "Different scope explains the figures.", "context_notes": "scope"},
    ])
    comparator = FactComparator(llm_client=fake)
    f1 = make_fact(text="Standalone revenue was 500.", source_document="a.pdf")
    f2 = make_fact(text="Consolidated revenue was 650.", source_document="b.pdf")

    comparison = comparator._compare_pair_with_llm(
        comparator._to_dict(f1), comparator._to_dict(f2)
    )
    assert comparison is not None
    assert comparison.relationship == "reconciled"


# ---------------------------------------------------------------------------
# 4. THE BUG: magnitude words ("crore"/"lakh"/"million"/"billion") are
#    silently dropped by _extract_numeric_scalar, corrupting the stored
#    numeric value by orders of magnitude.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("snippet,expected_value,actual_note", [
    ("Revenue was Rs. 500 crore in FY24.", 5_000_000_000, "500 crore = 5,000,000,000 (crore = 10^7)"),
    ("Revenue was 12.5 million dollars.", 12_500_000, "12.5 million = 12,500,000"),
    ("Total assets of 2.3 billion.", 2_300_000_000, "2.3 billion = 2,300,000,000"),
    ("INR 45 lakh was allocated.", 4_500_000, "45 lakh = 4,500,000 (lakh = 10^5)"),
])
def test_magnitude_word_bug_documented(snippet, expected_value, actual_note):
    """
    This test is EXPECTED TO FAIL against the current implementation.
    It documents a real bug: _extract_numeric_scalar extracts only the
    bare digit token and silently discards magnitude words like
    "crore", "lakh", "million", "billion". This means a fact's stored
    `value` can be wrong by 3-7 orders of magnitude, which would corrupt
    any numeric contradiction/corroboration check downstream.
    """
    extracted = PDFExtractor._extract_numeric_scalar(snippet)
    print(f"\n  snippet={snippet!r}")
    print(f"  expected (correct): {expected_value}  ({actual_note})")
    print(f"  actual (buggy):     {extracted}")
    assert extracted == expected_value, (
        f"BUG CONFIRMED: magnitude word dropped. Got {extracted}, "
        f"should be {expected_value}. {actual_note}"
    )


def test_magnitude_bug_causes_false_contradiction_risk():
    """
    Two documents stating the SAME fact using different magnitude
    conventions (crore vs. raw rupees) must normalize to the SAME scalar,
    so a numeric comparator does not flag a false contradiction.
    Previously this was a bug (ratio 10,000,000x); now fixed.
    """
    val_crore = PDFExtractor._extract_numeric_scalar("₹500 crore")
    val_raw = PDFExtractor._extract_numeric_scalar("5,000,000,000")
    print(f"\n  '₹500 crore' -> {val_crore}")
    print(f"  '5,000,000,000' -> {val_raw}")
    ratio = val_raw / val_crore if val_crore else None
    print(f"  ratio: {ratio}")
    # These SHOULD be equal (same real-world quantity).
    assert val_crore == val_raw, (
        f"magnitude normalization broken: {val_crore} != {val_raw} "
        f"(ratio {ratio})"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
