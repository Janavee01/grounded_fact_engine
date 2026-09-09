"""
End-to-end tests: real PDFs → real Ollama → extraction → comparison.

Run:
    python -m pytest -v -s tests/test_real.py

Requires Ollama running with qwen3:8b pulled (already installed locally).
Set LLM_PROVIDER=openrouter + OPENROUTER_API_KEY to use a hosted model.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path

from src.extraction.pdf_extractor import PDFExtractor
from src.comparison.comparator import FactComparator
from src.extraction.llm_client import LLMClient

DATA = Path(__file__).resolve().parent.parent / "data"

PDFS = {
    "india_es":  DATA / "india-macroeconomy" / "01-india-economic-survey-2024-25-excerpt.pdf",
    "india_rbi": DATA / "india-macroeconomy" / "02-rbi-annual-report-2024-25-excerpt.pdf",
    "india_imf": DATA / "india-macroeconomy" / "03-imf-india-2025-article-iv-excerpt.pdf",
    "delh_ar":   DATA / "delhivery" / "02-delhivery-annual-report-fy24-excerpt.pdf",
    "delh_q4":   DATA / "delhivery" / "03-delhivery-q4-fy24-earnings-presentation.pdf",
}


@pytest.fixture(scope="module")
def llm():
    return LLMClient(provider=os.getenv("LLM_PROVIDER", "ollama"))


@pytest.fixture(scope="module")
def extractor(llm):
    return PDFExtractor(llm_client=llm)


@pytest.fixture(scope="module")
def comparator(llm):
    return FactComparator(llm_client=llm)


# ------------------------------------------------------------------
# Part 1 — individual extraction from each PDF
# ------------------------------------------------------------------

class TestExtraction:
    """Extract facts from every PDF and verify output shape."""

    @pytest.mark.parametrize("key,path", PDFS.items(), ids=lambda k: k)
    def test_extract_returns_facts(self, extractor, key, path):
        facts = extractor.extract_facts(str(path), source_document=path.name)
        print(f"\n--- {key}: {len(facts)} facts ---")
        for f in facts[:8]:
            print(f"  [{f.fact_type.value}] {f.text[:100]}  (val={f.value}, conf={f.confidence})")
        if len(facts) > 8:
            print(f"  ... and {len(facts) - 8} more")
        assert len(facts) > 0, f"No facts extracted from {path.name}"


# ------------------------------------------------------------------
# Part 2 — cross-document comparison (the 4 special cases)
# ------------------------------------------------------------------

class TestFourCases:
    """
    Pair facts from different PDFs and let the real LLM classify
    each pair.  We don't tell the model what to expect.
    """

    def test_corroboration(self, extractor, comparator):
        """
        Both the RBI report and the IMF report discuss India's macro
        outlook.  Extract from each, then compare all cross-doc pairs.
        At least one corroboration should appear (same metric, same
        entity, consistent values).
        """
        f_rbi  = extractor.extract_facts(str(PDFS["india_rbi"]), source_document=PDFS["india_rbi"].name)
        f_imf  = extractor.extract_facts(str(PDFS["india_imf"]), source_document=PDFS["india_imf"].name)

        print(f"\nRBI facts: {len(f_rbi)}, IMF facts: {len(f_imf)}")
        comparisons = comparator.compare_facts(f_rbi + f_imf)

        print("\n--- RBI vs IMF comparisons ---")
        for c in comparisons:
            print(f"  {c.relationship}: {c.explanation[:120]}")

        rels = [c.relationship for c in comparisons]
        assert "corroborates" in rels, (
            f"Expected at least one corroboration between RBI and IMF docs. "
            f"Got: {rels}"
        )

    def test_contradiction(self, extractor, comparator):
        """
        The Economic Survey gives India's FY25 GDP as 6.4% and the
        IMF says 6.5%.  Same metric, same country, slight mismatch.
        The LLM should flag this.
        """
        f_es   = extractor.extract_facts(str(PDFS["india_es"]), source_document=PDFS["india_es"].name)
        f_imf  = extractor.extract_facts(str(PDFS["india_imf"]), source_document=PDFS["india_imf"].name)

        print(f"\nES facts: {len(f_es)}, IMF facts: {len(f_imf)}")
        comparisons = comparator.compare_facts(f_es + f_imf)

        print("\n--- ES vs IMF comparisons ---")
        for c in comparisons:
            print(f"  {c.relationship}: {c.explanation[:120]}")

        rels = [c.relationship for c in comparisons]
        hit = "contradicts" in rels or "reconciled" in rels
        assert hit, (
            f"Expected a contradiction or reconciliation between ES and IMF "
            f"GDP figures. Got: {rels}"
        )

    def test_reconciled(self, extractor, comparator):
        """
        Delhivery's annual report and Q4 earnings deck both state FY24
        revenue — one uses ₹81,415 million, the other ₹8,142 crore.
        Same real figure, different units.  The comparator should
        recognise this as reconciled (or corroborates, both acceptable).
        """
        f_ar  = extractor.extract_facts(str(PDFS["delh_ar"]), source_document=PDFS["delh_ar"].name)
        f_q4  = extractor.extract_facts(str(PDFS["delh_q4"]), source_document=PDFS["delh_q4"].name)

        print(f"\nAR facts: {len(f_ar)}, Q4 facts: {len(f_q4)}")
        comparisons = comparator.compare_facts(f_ar + f_q4)

        print("\n--- AR vs Q4 comparisons ---")
        for c in comparisons:
            print(f"  {c.relationship}: {c.explanation[:120]}")

        rels = [c.relationship for c in comparisons]
        hit = any(r in rels for r in ("corroborates", "reconciled"))
        assert hit, (
            f"Expected corroboration or reconciliation between Delhivery "
            f"AR and Q4 revenue figures. Got: {rels}"
        )

    def test_unrelated(self, extractor, comparator):
        """
        Delhivery's revenue has nothing to do with India's CPI
        inflation.  The comparator must drop the pair as unrelated.
        """
        f_ar  = extractor.extract_facts(str(PDFS["delh_ar"]), source_document=PDFS["delh_ar"].name)
        f_imf = extractor.extract_facts(str(PDFS["india_imf"]), source_document=PDFS["india_imf"].name)

        print(f"\nAR facts: {len(f_ar)}, IMF facts: {len(f_imf)}")
        comparisons = comparator.compare_facts(f_ar + f_imf)

        print("\n--- AR vs IMF comparisons ---")
        for c in comparisons:
            print(f"  {c.relationship}: {c.explanation[:120]}")

        rels = [c.relationship for c in comparisons]
        assert "unrelated" not in rels, (
            "Unrelated pairs should be filtered from results"
        )
