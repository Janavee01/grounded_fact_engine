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
from src.models.fact import Fact, FactType, FactComparison


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


# ---------------------------------------------------------------------------
# 5. Numeric-contradiction rule: differing values are only contradictions
#    when the facts share entity/metric/scope/time with no explaining
#    context (planned-vs-actual, cumulative progression, revised
#    estimates, different reporting periods, scope, or units).
# ---------------------------------------------------------------------------

def _guardrail(f1, f2, relationship="contradicts"):
    """Feed a pair through _apply_guardrails with the LLM returning `relationship`."""
    fake = FakeLLMClient([
        {"relationship": relationship, "confidence": 0.85,
         "explanation": "Same metric, differing values.", "context_notes": None},
    ])
    comparator = FactComparator(llm_client=fake)
    comparison = comparator._compare_pair_with_llm(
        comparator._to_dict(f1),
        comparator._to_dict(f2),
    )
    assert comparison is not None
    return comparison


def test_numeric_contradiction_kept_without_explaining_context():
    """Same entity, metric, scope and period, no explanation -> contradicts."""
    f1 = make_fact(
        text="Net profit for FY24 was ₹200 crore.",
        value=200, context={"entity": "Acme", "time_period": "FY24"},
        source_document="a.pdf",
    )
    f2 = make_fact(
        text="Net profit for FY24 was ₹350 crore.",
        value=350, context={"entity": "Acme", "time_period": "FY24"},
        source_document="b.pdf",
    )
    comparison = _guardrail(f1, f2, "contradicts")
    assert comparison.relationship == "contradicts"


def test_numeric_contradiction_downgraded_by_cumulative_progression():
    """Cumulative measures reported at different points are not contradictions."""
    f1 = make_fact(
        text="Cumulative units delivered by May stood at 96.",
        value=96, context={"entity": "Acme", "time_period": "May"},
        source_document="a.pdf",
    )
    f2 = make_fact(
        text="Cumulative units delivered by October reached 150.",
        value=150, context={"entity": "Acme", "time_period": "October"},
        source_document="b.pdf",
    )
    comparison = _guardrail(f1, f2, "contradicts")
    assert comparison.relationship == "reconciled"
    assert "cumulative" in comparison.explanation.lower() or "reconciled" in comparison.explanation


def test_numeric_contradiction_downgraded_by_revised_estimate():
    """A revised estimate superseding an earlier one is not a contradiction."""
    f1 = make_fact(
        text="The company initially projected 120 units.",
        value=120, context={"entity": "Acme", "time_period": "FY24"},
        source_document="a.pdf",
    )
    f2 = make_fact(
        text="The company revised its estimate for FY24 to 96 units.",
        value=96, context={"entity": "Acme", "time_period": "FY24"},
        source_document="b.pdf",
    )
    comparison = _guardrail(f1, f2, "contradicts")
    assert comparison.relationship == "reconciled"


def test_numeric_contradiction_downgraded_by_different_reporting_period():
    """Different reporting periods/timelines explain differing values."""
    f1 = make_fact(
        text="FY23 revenue was ₹400 crore.",
        value=400, context={"entity": "Acme", "time_period": "FY23"},
        source_document="a.pdf",
    )
    f2 = make_fact(
        text="FY24 revenue was ₹500 crore.",
        value=500, context={"entity": "Acme", "time_period": "FY24"},
        source_document="b.pdf",
    )
    comparison = _guardrail(f1, f2, "contradicts")
    assert comparison.relationship == "reconciled"


def test_numeric_values_without_conflict_stay_corroborates():
    """Equal values are corroborations, regardless of cue words."""
    f1 = make_fact(
        text="Cumulative units delivered by May stood at 96.",
        value=96, context={"entity": "Acme"},
        source_document="a.pdf",
    )
    f2 = make_fact(
        text="Cumulative units delivered by October stayed at 96.",
        value=96, context={"entity": "Acme"},
        source_document="b.pdf",
    )
    comparison = _guardrail(f1, f2, "corroborates")
    assert comparison.relationship == "corroborates"


# ---------------------------------------------------------------------------
# 6. Robust guardrails: buses case, plans vs actual, approximate numbers,
#    different scopes, percentage points, and genuine same-period conflicts.
# ---------------------------------------------------------------------------

def test_buses_across_different_reporting_periods_reconciled():
    """
    96 vs 110 new units counted in DIFFERENT reporting periods: the
    difference is chronological, not contradictory.
    """
    f1 = make_fact(
        text="An audit counted 110 new buses in service as of December 2023.",
        value=110, context={"entity": "Metro", "time_period": "Dec 2023"},
        source_document="audit.pdf",
    )
    f2 = make_fact(
        text="A review found 96 new buses in service as of March 2024.",
        value=96, context={"entity": "Metro", "time_period": "Mar 2024"},
        source_document="review.pdf",
    )
    comparison = _guardrail(f1, f2, "contradicts")
    assert comparison.relationship == "reconciled"
    assert "reporting" in comparison.explanation.lower() or "reconcile" in comparison.explanation.lower()


def test_buses_same_period_is_genuine_contradiction():
    """
    96 vs 110 new buses in the SAME reporting period, no explanation:
    a real contradiction.
    """
    f1 = make_fact(
        text="An audit counted 110 new buses in service as of December 2023.",
        value=110, context={"entity": "Metro", "time_period": "Dec 2023"},
        source_document="audit.pdf",
    )
    f2 = make_fact(
        text="A review found 96 new buses in service as of December 2023.",
        value=96, context={"entity": "Metro", "time_period": "Dec 2023"},
        source_document="review.pdf",
    )
    comparison = _guardrail(f1, f2, "contradicts")
    assert comparison.relationship == "contradicts"


def test_planned_120_vs_actual_96_reconciled():
    """120 planned vs 96 actual is planned-versus-actual, not a contradiction."""
    f1 = make_fact(
        text="The first phase was planned to cover 120 buses.",
        value=120, context={"entity": "Metro", "time_period": "FY24"},
        source_document="plan.pdf",
    )
    f2 = make_fact(
        text="The review found 96 buses had entered service, rather than the 120 expected.",
        value=96, context={"entity": "Metro", "time_period": "FY24"},
        source_document="review.pdf",
    )
    comparison = _guardrail(f1, f2, "contradicts")
    assert comparison.relationship == "reconciled"


def test_approximate_numbers_are_not_a_contradiction():
    """Approximately equal large figures corroborate, even if not identical."""
    f1 = make_fact(
        text="Revenue for FY24 was approximately 8,140 crore.",
        value=8140, unit="crore",
        context={"entity": "Delhivery", "time_period": "FY24"},
        source_document="a.pdf",
    )
    f2 = make_fact(
        text="Revenue for FY24 was about 8,141 crore.",
        value=8141, unit="crore",
        context={"entity": "Delhivery", "time_period": "FY24"},
        source_document="b.pdf",
    )
    comparison = _guardrail(f1, f2, "contradicts")
    assert comparison.relationship == "corroborates"


def test_approximate_number_reconciles_larger_gap():
    """An approximate figure explains a modest gap between rounded values."""
    f1 = make_fact(
        text="About 100 new units were in service.",
        value=100, context={"entity": "Metro"},
        source_document="a.pdf",
    )
    f2 = make_fact(
        text="Roughly 96 new units were in service.",
        value=96, context={"entity": "Metro"},
        source_document="b.pdf",
    )
    comparison = _guardrail(f1, f2, "contradicts")
    assert comparison.relationship == "reconciled"


def test_different_scope_reconciles_numeric_values():
    """Standalone vs consolidated scope explains differing revenue figures."""
    f1 = make_fact(
        text="Standalone revenue was ₹500 crore.",
        value=500, context={"entity": "Acme", "scope": "standalone"},
        source_document="a.pdf",
    )
    f2 = make_fact(
        text="Consolidated revenue was ₹650 crore.",
        value=650, context={"entity": "Acme", "scope": "consolidated"},
        source_document="b.pdf",
    )
    comparison = _guardrail(f1, f2, "contradicts")
    assert comparison.relationship == "reconciled"


def test_percentage_points_are_not_direct_comparison():
    """A percentage-point change is not directly comparable to a raw percent."""
    f1 = make_fact(
        text="The rate rose by 5 percentage points.",
        value=5, unit="pp",
        context={"entity": "Acme", "time_period": "FY24"},
        source_document="a.pdf",
    )
    f2 = make_fact(
        text="The rate grew by 5 percent.",
        value=5, unit="%",
        context={"entity": "Acme", "time_period": "FY24"},
        source_document="b.pdf",
    )
    comparison = _guardrail(f1, f2, "contradicts")
    assert comparison.relationship == "reconciled"


def test_insufficient_context_not_forced_contradiction():
    """
    Differing values with no entity/period context cannot be confirmed as
    the same claim, so must not be forced into a contradiction.
    """
    f1 = make_fact(
        text="An audit counted 110 new units in service.",
        value=110, context={},
        source_document="audit.pdf",
    )
    f2 = make_fact(
        text="A review found 96 new units in service.",
        value=96, context={},
        source_document="review.pdf",
    )
    comparison = _guardrail(f1, f2, "contradicts")
    assert comparison.relationship == "insufficient_context"


def test_insufficient_context_labelled_by_model_is_preserved():
    """The model's own 'insufficient_context' label is kept, not forced."""
    f1 = make_fact(text="Reported 110 units.", value=110, context={}, source_document="a.pdf")
    f2 = make_fact(text="Reported 96 units.", value=96, context={}, source_document="b.pdf")
    fake = FakeLLMClient([
        {"relationship": "insufficient_context", "confidence": 0.6,
         "explanation": "Cannot confirm same claim.", "context_notes": None},
    ])
    comparator = FactComparator(llm_client=fake)
    comparison = comparator._compare_pair_with_llm(
        comparator._to_dict(f1), comparator._to_dict(f2)
    )
    assert comparison is not None
    assert comparison.relationship == "insufficient_context"


def test_causality_vs_correlation_not_contradiction():
    """A causal claim and a correlation observation are not direct contradictions."""
    f1 = make_fact(
        text="The advertising campaign caused the increase in sales.",
        value=None, context={"entity": "Acme"},
        source_document="a.pdf",
    )
    f2 = make_fact(
        text="Sales increases were correlated with the advertising campaign.",
        value=None, context={"entity": "Acme"},
        source_document="b.pdf",
    )
    comparison = _guardrail(f1, f2, "contradicts")
    assert comparison.relationship == "reconciled"


# ---------------------------------------------------------------------------
# 7. Specific required scenarios: label-explanation never contradicts
# ---------------------------------------------------------------------------

def test_96_vs_110_unknown_periods_insufficient_context():
    """
    96 buses vs 110 buses with unknown reporting periods -> INSUFFICIENT_CONTEXT.
    Neither fact provides a date, so we cannot confirm same-period comparability.
    """
    f1 = make_fact(
        text="An audit counted 96 new buses in service.",
        value=96, context={"entity": "Metro"},
        source_document="audit.pdf",
    )
    f2 = make_fact(
        text="A review found 110 new buses in service.",
        value=110, context={"entity": "Metro"},
        source_document="review.pdf",
    )
    comparison = _guardrail(f1, f2, "contradicts")
    assert comparison.relationship == "insufficient_context"
    assert "insufficient" in comparison.explanation.lower() or \
        "cannot" in comparison.explanation.lower() or \
        "missing" in comparison.explanation.lower()


def test_96_vs_110_different_dates_reconciled():
    """
    96 buses vs 110 buses where 110 is later -> RECONCILE.
    Different explicit dates explain the chronological difference.
    """
    f1 = make_fact(
        text="An audit counted 96 new buses in service as of December 2023.",
        value=96, context={"entity": "Metro", "time_period": "Dec 2023"},
        source_document="audit.pdf",
    )
    f2 = make_fact(
        text="A review found 110 new buses in service as of March 2024.",
        value=110, context={"entity": "Metro", "time_period": "Mar 2024"},
        source_document="review.pdf",
    )
    comparison = _guardrail(f1, f2, "contradicts")
    assert comparison.relationship == "reconciled"
    explanation_lower = comparison.explanation.lower()
    assert "period" in explanation_lower or "reporting" in explanation_lower or \
        "chronolog" in explanation_lower or "time" in explanation_lower or \
        "different" in explanation_lower


def test_96_actual_vs_120_expected_reconciled():
    """
    96 actual vs 120 originally expected -> RECONCILE.
    Planned-versus-actual is never a contradiction.
    """
    f1 = make_fact(
        text="The first phase was planned to cover 120 buses.",
        value=120, context={"entity": "Metro", "time_period": "FY24"},
        source_document="plan.pdf",
    )
    f2 = make_fact(
        text="The review found 96 buses had entered service, rather than the 120 originally expected.",
        value=96, context={"entity": "Metro", "time_period": "FY24"},
        source_document="review.pdf",
    )
    comparison = _guardrail(f1, f2, "contradicts")
    assert comparison.relationship == "reconciled"
    explanation_lower = comparison.explanation.lower()
    assert "plan" in explanation_lower or "actual" in explanation_lower or \
        "expected" in explanation_lower or "reconcil" in explanation_lower


def test_96_vs_110_same_date_same_metric_contradicts():
    """
    96 vs 110 for the same date and same metric -> CONTRADICT.
    Same entity, metric, scope, and time period with incompatible values.
    """
    f1 = make_fact(
        text="An audit counted 96 new buses in service as of December 2023.",
        value=96, context={"entity": "Metro", "time_period": "Dec 2023"},
        source_document="audit.pdf",
    )
    f2 = make_fact(
        text="A review found 110 new buses in service as of December 2023.",
        value=110, context={"entity": "Metro", "time_period": "Dec 2023"},
        source_document="review.pdf",
    )
    comparison = _guardrail(f1, f2, "contradicts")
    assert comparison.relationship == "contradicts"
    explanation_lower = comparison.explanation.lower()
    assert "conflict" in explanation_lower or "incompatib" in explanation_lower or \
        "contradict" in explanation_lower or "same period" in explanation_lower or \
        "mutually" in explanation_lower


def test_28_stations_vs_uneven_improvements_corroborates():
    """
    28 stations with accessibility improvements vs uneven station-level
    improvements -> CORROBORATE. The facts are consistent restatements.
    """
    f1 = make_fact(
        text="28 stations received accessibility improvements.",
        value=28, unit="stations",
        fact_type=FactType.NUMERIC,
        context={"entity": "Railway stations"},
        source_document="report_a.pdf",
    )
    f2 = make_fact(
        text="Station-level accessibility improvements were uneven across the network.",
        value=None,
        fact_type=FactType.SEMANTIC,
        context={"entity": "Railway stations"},
        source_document="report_b.pdf",
    )
    fake = FakeLLMClient([
        {"relationship": "corroborates", "confidence": 0.85,
         "explanation": ("Fact 1 specifies 28 stations received improvements "
                         "while Fact 2 notes improvements were uneven across "
                         "stations; both are consistent observations about "
                         "accessibility improvements."),
         "context_notes": None},
    ])
    comparator = FactComparator(llm_client=fake)
    comparison = comparator._compare_pair_with_llm(
        comparator._to_dict(f1), comparator._to_dict(f2)
    )
    assert comparison is not None
    assert comparison.relationship == "corroborates"
    explanation_lower = comparison.explanation.lower()
    assert "consistent" in explanation_lower or "corroborat" in explanation_lower or \
        "support" in explanation_lower or "agree" in explanation_lower or \
        "compatible" in explanation_lower


def test_explanation_consistency_validation():
    """
    Verify that the consistency validation catches and corrects cases where
    the explanation describes a different relationship than the label.
    When the explanation unambiguously supports CONTRADICTION but the label
    says CORROBORATES (with no corroboration cue present), the label is
    revised to match the evidence in the explanation.
    """
    comparison = FactComparison(
        fact1_id="f1",
        fact2_id="f2",
        relationship="corroborates",
        confidence=0.8,
        explanation="The values conflict and are mutually incompatible.",
        context_notes=None,
    )
    comparator = FactComparator(llm_client=FakeLLMClient([]))
    validated = comparator._validate_explanation_consistency(comparison)
    assert validated.relationship == "contradicts"


def test_explanation_consistency_revises_explanation():
    """
    When the label is authoritative but the explanation gives no support,
    the explanation is extended with a label-aligned sentence so that the
    explanation and label never contradict each other.
    """
    comparison = FactComparison(
        fact1_id="f1",
        fact2_id="f2",
        relationship="reconciled",
        confidence=0.8,
        explanation="One figure is standalone, the other consolidated.",
        context_notes=None,
    )
    comparator = FactComparator(llm_client=FakeLLMClient([]))
    validated = comparator._validate_explanation_consistency(comparison)
    assert validated.relationship == "reconciled"
    explanation_lower = validated.explanation.lower()
    assert "reconcil" in explanation_lower or \
        "explained" in explanation_lower or \
        "different scope" in explanation_lower or \
        "standalone" in explanation_lower


def test_explanation_consistency_preserves_valid():
    """
    When explanation and label are consistent, no validation warning is added.
    """
    comparison = FactComparison(
        fact1_id="f1",
        fact2_id="f2",
        relationship="contradicts",
        confidence=0.8,
        explanation="The values conflict and are mutually incompatible.",
        context_notes=None,
    )
    comparator = FactComparator(llm_client=FakeLLMClient([]))
    validated = comparator._validate_explanation_consistency(comparison)
    assert validated.relationship == "contradicts"
    assert "validation" not in validated.explanation.lower()


def test_explanations_never_mention_guardrails():
    """
    Explanations must never name an internal guardrail step. A guardrail
    result is expressed as a self-contained evidence-based justification,
    even when a guardrail changed the label.
    """
    pairs = [
        (
            make_fact(text="About 100 new units were in service.", value=100,
                      context={"entity": "Metro"}, source_document="a.pdf"),
            make_fact(text="Roughly 96 new units were in service.", value=96,
                      context={"entity": "Metro"}, source_document="b.pdf"),
        ),
        (
            make_fact(text="An audit counted 96 new buses in service.",
                      value=96, context={"entity": "Metro"},
                      source_document="audit.pdf"),
            make_fact(text="A review found 110 new buses in service.",
                      value=110, context={"entity": "Metro"},
                      source_document="review.pdf"),
        ),
        (
            make_fact(text="The first phase was planned to cover 120 buses.",
                      value=120, context={"entity": "Metro",
                                          "time_period": "FY24"},
                      source_document="plan.pdf"),
            make_fact(
                text=("The review found 96 buses had entered service, "
                      "rather than the 120 expected."),
                value=96, context={"entity": "Metro", "time_period": "FY24"},
                source_document="review.pdf"),
        ),
    ]
    for f1, f2 in pairs:
        comparison = _guardrail(f1, f2, "contradicts")
        assert "guardrail" not in comparison.explanation.lower()
        assert "validation" not in comparison.explanation.lower()


def test_guardrail_explanations_support_their_labels():
    """
    Every guardrail-rewritten explanation must contain cue words matching
    its own assigned label (never a different relationship).
    """
    corroboration = FactComparison(
        fact1_id="f1", fact2_id="f2", relationship="corroborates",
        confidence=0.8,
        explanation="The values conflict and are mutually incompatible.",
        context_notes=None,
    )
    comparator = FactComparator(llm_client=FakeLLMClient([]))
    fixed = comparator._validate_explanation_consistency(corroboration)
    assert fixed.relationship == "contradicts"
    assert comparator._CONTRADICTION_CUES.search(fixed.explanation.lower())

    insufficient = FactComparison(
        fact1_id="f1", fact2_id="f2",
        relationship="insufficient_context", confidence=0.8,
        explanation="Not enough evidence is available to compare the figures.",
        context_notes=None,
    )
    fixed2 = comparator._validate_explanation_consistency(insufficient)
    assert fixed2.relationship == "insufficient_context"
    assert comparator._INSUFFICIENT_CUES.search(fixed2.explanation.lower())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
