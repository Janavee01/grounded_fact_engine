"""
Generic, document-agnostic tests for fact extraction quality and grounding.

These tests use synthetic/general-purpose examples only — no real user
document, no hardcoded organization/field/ID names. A fake LLM client is
injected so no Ollama server is needed.

Run: pytest -q tests/test_generic_extraction.py
"""
import sys
import os
from datetime import datetime

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extraction.pdf_extractor import PDFExtractor
from src.extraction.grounding import ground_quote
from src.models.fact import Fact, FactType


class FakeLLMClient:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def generate_json(self, system_prompt, user_prompt, temperature=0.0):
        self.calls.append(user_prompt)
        if not self.responses:
            raise RuntimeError("FakeLLMClient exhausted")
        return self.responses.pop(0)


def extractor():
    return PDFExtractor(llm_client=FakeLLMClient([]))


def make_fact(**kw):
    defaults = dict(
        id="id-" + str(hash(str(kw)) % 100000),
        text="placeholder",
        fact_type=FactType.SEMANTIC,
        value=None,
        unit=None,
        context={},
        source_document="synthetic.pdf",
        source_page=1,
        source_snippet="snippet",
        confidence=0.9,
        extraction_method="llm",
        created_at=datetime.now(),
    )
    defaults.update(kw)
    return Fact(**defaults)


# ---------------------------------------------------------------------------
# 1. Fact types
# ---------------------------------------------------------------------------

def test_fact_type_enum_supports_all_required_types():
    assert [t.value for t in FactType] == [
        "numeric",
        "semantic",
        "date",
        "time",
        "boolean",
        "entity",
    ]


# ---------------------------------------------------------------------------
# 2. Fact-type classification / value normalization per type
# ---------------------------------------------------------------------------

def test_numeric_fact_is_normalized_to_number():
    chunk = "The project budget was approved at 1.5 million dollars in March."
    raw = {
        "claim": "The project budget was approved at 1.5 million dollars.",
        "fact_type": "numeric",
        "value": "1.5 million",
        "unit": "dollars",
        "verbatim_quote": "The project budget was approved at 1.5 million dollars.",
        "confidence": 0.95,
    }
    fact = extractor()._build_fact(raw, chunk, 1, "synthetic.pdf")
    assert fact is not None
    assert fact.fact_type == FactType.NUMERIC
    assert fact.value == 1_500_000.0
    assert fact.unit == "dollars"


def test_semantic_fact_is_kept_textual():
    chunk = "Company policy allows employees to work remotely on Fridays."
    raw = {
        "claim": "Company policy allows employees to work remotely on Fridays.",
        "fact_type": "semantic",
        "value": None,
        "unit": None,
        "verbatim_quote": "Company policy allows employees to work remotely on Fridays.",
        "confidence": 0.9,
    }
    fact = extractor()._build_fact(raw, chunk, 1, "synthetic.pdf")
    assert fact is not None
    assert fact.fact_type == FactType.SEMANTIC
    assert fact.value is None


def test_date_fact_is_normalized_to_iso():
    chunk = "The company was incorporated on 14 March 2012 in Bengaluru."
    raw = {
        "claim": "The company was incorporated on 14 March 2012.",
        "fact_type": "date",
        "value": "14 March 2012",
        "unit": None,
        "verbatim_quote": "The company was incorporated on 14 March 2012.",
        "confidence": 0.95,
    }
    fact = extractor()._build_fact(raw, chunk, 1, "synthetic.pdf")
    assert fact is not None
    assert fact.fact_type == FactType.DATE
    assert fact.value == "2012-03-14"


def test_single_time_fact_is_normalized():
    chunk = "A server backup is scheduled at 2:30 AM every night."
    raw = {
        "claim": "A server backup is scheduled at 2:30 AM every night.",
        "fact_type": "time",
        "value": "2:30 AM",
        "unit": None,
        "verbatim_quote": "A server backup is scheduled at 2:30 AM every night.",
        "confidence": 0.9,
    }
    fact = extractor()._build_fact(raw, chunk, 1, "synthetic.pdf")
    assert fact is not None
    assert fact.fact_type == FactType.TIME
    assert fact.value == "02:30"


def test_time_range_fact_is_normalized():
    chunk = "The help desk is open from 9:00 AM to 5:00 PM on weekdays."
    raw = {
        "claim": "The help desk is open from 9:00 AM to 5:00 PM on weekdays.",
        "fact_type": "time",
        "value": "9:00 AM to 5:00 PM",
        "unit": None,
        "verbatim_quote": "The help desk is open from 9:00 AM to 5:00 PM on weekdays.",
        "confidence": 0.95,
    }
    fact = extractor()._build_fact(raw, chunk, 1, "synthetic.pdf")
    assert fact is not None
    assert fact.fact_type == FactType.TIME
    assert fact.value == "09:00-17:00"


def test_boolean_fact_stays_boolean():
    chunk = "The device is waterproof."
    raw = {
        "claim": "The device is waterproof.",
        "fact_type": "boolean",
        "value": True,
        "unit": None,
        "verbatim_quote": "The device is waterproof.",
        "confidence": 0.9,
    }
    fact = extractor()._build_fact(raw, chunk, 1, "synthetic.pdf")
    assert fact is not None
    assert fact.fact_type == FactType.BOOLEAN
    assert fact.value is True


def test_entity_fact_is_kept_textual():
    chunk = "Acme Aeronautics is headquartered in Springfield."
    raw = {
        "claim": "Acme Aeronautics is headquartered in Springfield.",
        "fact_type": "entity",
        "value": "Acme Aeronautics",
        "unit": None,
        "verbatim_quote": "Acme Aeronautics is headquartered in Springfield.",
        "confidence": 0.9,
    }
    fact = extractor()._build_fact(raw, chunk, 1, "synthetic.pdf")
    assert fact is not None
    assert fact.fact_type == FactType.ENTITY
    assert fact.value == "Acme Aeronautics"


# ---------------------------------------------------------------------------
# 3. Meaningful-fact filtering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("claim", [
    "U51100KA2012PTC063107",
    "Reg No: 12-3456789",
    "The card number is XXXX-XXXX-XXXX-1234.",
    "password=s3cret_Pa55",
    "authorization: Bearer abc123def456",
    "Reported value: [INSERT AMOUNT]",
    "Total: ***",
    "123-45-6789",
])
def test_meaningless_facts_are_rejected(claim):
    chunk = claim
    raw = {
        "claim": claim,
        "fact_type": "semantic",
        "value": None,
        "unit": None,
        "verbatim_quote": claim,
        "confidence": 0.9,
    }
    result = extractor()._build_fact(raw, chunk, 1, "synthetic.pdf")
    assert result is None, f"expected rejection for meaningless claim: {claim!r}"


def test_is_meaningful_claim_rejects_boilerplate_and_fragments():
    extract = extractor()._is_meaningful_claim
    assert extract("Registration number 5A8F3") is False
    assert extract("Section 4") is False
    assert extract("Page 1 of 4") is False


# ---------------------------------------------------------------------------
# 4. Evidence grounding
# ---------------------------------------------------------------------------

def test_quote_not_present_in_source_is_rejected():
    chunk = "The company opened 12 new branches this year."
    raw = {
        "claim": "The company opened 12 new branches this year.",
        "fact_type": "numeric",
        "value": 12,
        "unit": "branches",
        "verbatim_quote": "sales increased by 30 percent this quarter",
        "confidence": 0.9,
    }
    result = extractor()._build_fact(raw, chunk, 1, "synthetic.pdf")
    assert result is None


def test_unsupported_claim_is_rejected():
    chunk = "Revenue grew by 5% during the fourth quarter."
    raw = {
        "claim": "The company's net profit trebled in the year.",
        "fact_type": "numeric",
        "value": 5,
        "unit": "percent",
        "verbatim_quote": "Revenue grew by 5% during the fourth quarter.",
        "confidence": 0.9,
    }
    result = extractor()._build_fact(raw, chunk, 1, "synthetic.pdf")
    assert result is None


def test_evidence_snippet_is_concise_not_whole_chunk():
    chunk = (
        "The annual report summarizes the performance of all operating "
        "segments. Net profit rose to 42 million during the reporting period. "
        "The outlook section describes expectations for the coming year."
    )
    raw = {
        "claim": "Net profit rose to 42 million during the reporting period.",
        "fact_type": "numeric",
        "value": "42 million",
        "unit": None,
        "verbatim_quote": "Net profit rose to 42 million during the reporting period.",
        "confidence": 0.95,
    }
    fact = extractor()._build_fact(raw, chunk, 1, "synthetic.pdf")
    assert fact is not None
    assert fact.source_snippet == (
        "Net profit rose to 42 million during the reporting period."
    )
    assert fact.source_snippet in chunk
    assert len(fact.source_snippet) < len(chunk)


def test_grounding_exact_match_returns_concise_verbatim_span():
    source = (
        "A long preamble discusses the general market outlook in detail. "
        "Inventory turnover improved to 6.2 times annually. "
        "Further commentary follows at length after this fact."
    )
    snippet, score = ground_quote(
        "Inventory turnover improved to 6.2 times annually.", source
    )
    assert snippet == "Inventory turnover improved to 6.2 times annually."
    assert snippet in source
    assert score == 100.0
    assert len(snippet) < len(source)


# ---------------------------------------------------------------------------
# 5. Value / fact-type compatibility
# ---------------------------------------------------------------------------

def test_numeric_with_unparseable_value_is_rejected():
    chunk = "Treasury yields moved higher last year."
    raw = {
        "claim": "Treasury yields moved higher last year.",
        "fact_type": "numeric",
        "value": "abc",
        "unit": None,
        "verbatim_quote": "Treasury yields moved higher last year.",
        "confidence": 0.9,
    }
    result = extractor()._build_fact(raw, chunk, 1, "synthetic.pdf")
    assert result is None


def test_date_with_non_date_value_is_rejected():
    chunk = "The event was held in fiscal year 2024."
    raw = {
        "claim": "The event was held in fiscal year 2024.",
        "fact_type": "date",
        "value": "FY2024",
        "unit": None,
        "verbatim_quote": "The event was held in fiscal year 2024.",
        "confidence": 0.9,
    }
    result = extractor()._build_fact(raw, chunk, 1, "synthetic.pdf")
    assert result is None


def test_time_with_unparseable_value_is_rejected():
    chunk = "The meeting was postponed to next week."
    raw = {
        "claim": "The meeting was postponed to next week.",
        "fact_type": "time",
        "value": "next week",
        "unit": None,
        "verbatim_quote": "The meeting was postponed to next week.",
        "confidence": 0.9,
    }
    result = extractor()._build_fact(raw, chunk, 1, "synthetic.pdf")
    assert result is None


def test_boolean_with_non_boolean_value_is_rejected():
    chunk = "Membership eligibility varies by region."
    raw = {
        "claim": "Membership eligibility varies by region.",
        "fact_type": "boolean",
        "value": "maybe",
        "unit": None,
        "verbatim_quote": "Membership eligibility varies by region.",
        "confidence": 0.9,
    }
    result = extractor()._build_fact(raw, chunk, 1, "synthetic.pdf")
    assert result is None


# ---------------------------------------------------------------------------
# 6. Generic value normalization helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("given,expected", [
    ("9:00 AM", "09:00"),
    ("3:30 PM", "15:30"),
    ("12:00 AM", "00:00"),
    ("12:00 PM", "12:00"),
    ("14:30", "14:30"),
    ("09:15:00", "09:15"),
    ("9:00 AM to 5:00 PM", "09:00-17:00"),
    ("10:00 AM - 6:00 PM", "10:00-18:00"),
])
def test_time_scalar_normalization(given, expected):
    assert PDFExtractor._extract_time_scalar(given) == expected


@pytest.mark.parametrize("given,expected", [
    ("2012-03-14", "2012-03-14"),
    ("14 March 2012", "2012-03-14"),
    ("March 14, 2012", "2012-03-14"),
    ("Mar 14 2012", "2012-03-14"),
    ("14/03/2012", "2012-03-14"),
    ("2024.05.31", "2024-05-31"),
])
def test_date_scalar_normalization(given, expected):
    assert PDFExtractor._extract_date_scalar(given) == expected


def test_boolean_scalar_normalization():
    assert PDFExtractor._extract_boolean_scalar("yes") is True
    assert PDFExtractor._extract_boolean_scalar("false") is False
    assert PDFExtractor._extract_boolean_scalar("1") is True
    assert PDFExtractor._extract_boolean_scalar("present") is True
    assert PDFExtractor._extract_boolean_scalar("absent") is False


# ---------------------------------------------------------------------------
# 7. Error isolation: one bad fact must not stop valid facts
# ---------------------------------------------------------------------------

def test_invalid_fact_is_dropped_while_valid_fact_survives():
    extract = extractor()
    chunk = "Treasury yields moved higher last year. Revenue increased by 7%."
    raw_bad = {
        "claim": "Treasury yields moved higher last year.",
        "fact_type": "numeric",
        "value": "abc",
        "unit": None,
        "verbatim_quote": "Treasury yields moved higher last year.",
        "confidence": 0.9,
    }
    raw_good = {
        "claim": "Revenue increased by 7%.",
        "fact_type": "numeric",
        "value": "7%",
        "unit": "percent",
        "verbatim_quote": "Revenue increased by 7%.",
        "confidence": 0.9,
    }
    bad = extract._build_fact(raw_bad, chunk, 1, "synthetic.pdf")
    good = extract._build_fact(raw_good, chunk, 1, "synthetic.pdf")
    assert bad is None
    assert good is not None
    assert good.value == 7.0


# ---------------------------------------------------------------------------
# 8. Database persistence of JSON-compatible value types
# ---------------------------------------------------------------------------

def test_database_persists_all_value_types(tmp_path):
    from src.storage.database import Database

    db = Database(str(tmp_path / "facts.db"))

    facts = [
        make_fact(
            id="f-numeric",
            text="Budget approved at 1.5 million dollars.",
            fact_type=FactType.NUMERIC,
            value=1_500_000.0,
            unit="dollars",
        ),
        make_fact(
            id="f-date",
            text="Incorporated on 14 March 2012.",
            fact_type=FactType.DATE,
            value="2012-03-14",
        ),
        make_fact(
            id="f-time",
            text="Help desk open 9:00 AM to 5:00 PM.",
            fact_type=FactType.TIME,
            value="09:00-17:00",
        ),
        make_fact(
            id="f-bool",
            text="The device is waterproof.",
            fact_type=FactType.BOOLEAN,
            value=True,
        ),
        make_fact(
            id="f-null",
            text="Policy permits remote work on Fridays.",
            fact_type=FactType.SEMANTIC,
            value=None,
        ),
        make_fact(
            id="f-entity",
            text="Acme Aeronautics is headquartered in Springfield.",
            fact_type=FactType.ENTITY,
            value="Acme Aeronautics",
        ),
    ]

    for fact in facts:
        db.save_fact(fact)

    rows = db.get_all_facts()
    assert len(rows) == len(facts)

    by_id = {r["id"]: r for r in rows}

    assert by_id["f-numeric"]["value"] == 1_500_000.0
    assert by_id["f-date"]["value"] == "2012-03-14"
    assert by_id["f-time"]["value"] == "09:00-17:00"
    assert by_id["f-bool"]["value"] is True
    assert by_id["f-null"]["value"] is None
    assert by_id["f-entity"]["value"] == "Acme Aeronautics"

    # Source traceability must survive the round trip.
    for fact in facts:
        row = by_id[fact.id]
        assert row["source_document"] == fact.source_document
        assert row["source_page"] == fact.source_page
        assert row["source_snippet"] == fact.source_snippet
        assert row["confidence"] == fact.confidence
        assert row["extraction_method"] == "llm"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))