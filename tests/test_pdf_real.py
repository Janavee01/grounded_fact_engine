"""
Document-agnostic end-to-end tests: real PDFs -> real LLM -> extraction -> comparison.

There are no hardcoded documents, facts, or expected values anywhere in this
suite.  Every assertion is derived from the data actually extracted from
whatever PDFs are present in the data directory.

Run (requires Ollama with the configured model pulled):
    python -m pytest -v -s tests/test_pdf_real.py

Or as a full-pipeline report:
    python tests/test_pdf_real.py                 # auto-discovery, cached extraction
    python tests/test_pdf_real.py --dir my_docs   # point at a different corpus
"""
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extraction.pdf_extractor import PDFExtractor
from src.extraction.grounding import ground_quote
from src.extraction.llm_client import LLMClient
from src.comparison.comparator import FactComparator
from src.models.fact import Fact, FactType, FactComparison

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

ALLOWED_RELATIONSHIPS = {"corroborates", "contradicts", "reconciled"}


# ---------------------------------------------------------------------------
# Discovery helpers (no hardcoding)
# ---------------------------------------------------------------------------

def discover_pdfs(directory: Path = DATA_DIR) -> List[Path]:
    return sorted(
        list(directory.glob("*.pdf"))
        + list(directory.glob("*/*.pdf"))
    )


def values_close(a, b, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def same_unit(u1, u2) -> bool:
    return (u1 or "").strip().lower() == (u2 or "").strip().lower()


def sample(items: List, limit: int):
    return items[:limit]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def llm_client():
    return LLMClient()


@pytest.fixture(scope="module")
def extractor(llm_client):
    return PDFExtractor(llm_client=llm_client)


@pytest.fixture(scope="module")
def comparator(llm_client):
    return FactComparator(llm_client=llm_client)


@pytest.fixture(scope="module")
def pdfs() -> List[Path]:
    found = discover_pdfs()
    if not found:
        pytest.skip("No PDFs found under data/ — run pytest from the repo root")
    return found


@pytest.fixture(scope="module")
def corpus(extractor, pdfs) -> Dict[Path, List[Fact]]:
    out = {}
    for pdf in pdfs:
        out[pdf] = extractor.extract_facts(str(pdf), source_document=pdf.name)
    return out


def _cross_doc_pairs(corpus):
    facts_by_doc = {k: v for k, v in corpus.items() if v}
    docs = list(facts_by_doc)
    pairs = []
    for i, doc_a in enumerate(docs):
        for doc_b in docs[i + 1:]:
            for f1 in facts_by_doc[doc_a]:
                for f2 in facts_by_doc[doc_b]:
                    pairs.append((f1, f2))
    return pairs


# ---------------------------------------------------------------------------
# PART 1 — individual extraction integrity
# ---------------------------------------------------------------------------

class TestIndividualExtraction:
    """Validate the facts extracted from every discovered PDF, no content assumptions."""

    @pytest.mark.parametrize(
        "pdf_path",
        [p for p in discover_pdfs()],
        ids=lambda p: p.name,
    )
    def test_facts_from_every_pdf_are_valid(self, extractor, pdf_path):
        pages = extractor.extract_text(str(pdf_path))
        text_pages = {
            p["page"]: p["text"]
            for p in pages
            if p["text"].strip()
        }
        if not text_pages:
            pytest.skip(
                f"{pdf_path.name} has no extractable text (scanned/image-only)"
            )

        facts = extractor.extract_facts(str(pdf_path), source_document=pdf_path.name)

        assert len(facts) > 0, (
            f"PDF has {len(text_pages)} text pages but extraction returned no facts: "
            f"{pdf_path.name}"
        )
        assert len({f.id for f in facts}) == len(facts), "fact ids must be unique"

        for fact in facts:
            assert fact.text and len(fact.text) >= 6, f"empty/trivial claim: {fact.text!r}"
            assert isinstance(fact.fact_type, FactType), f"bad fact_type: {fact.fact_type!r}"
            assert 0.0 <= fact.confidence <= 1.0, f"confidence out of range: {fact.confidence}"
            assert fact.source_page >= 1, f"page must be 1-indexed, got {fact.source_page}"
            assert fact.source_snippet, f"missing grounded snippet for {fact.text!r}"

            page_text = text_pages.get(fact.source_page, "")
            assert page_text, f"fact points to page {fact.source_page} which has no text"

            snippet, score = ground_quote(fact.source_snippet, page_text, min_score=60.0)
            assert score >= 60.0, (
                f"fact snippet is NOT grounded in the source page:\n"
                f"  claim:    {fact.text}\n"
                f"  snippet:  {fact.source_snippet!r}\n"
                f"  score:    {score}"
            )


# ---------------------------------------------------------------------------
# PART 2 — cross-document comparison (the four cases)
# ---------------------------------------------------------------------------

class TestCrossDocumentCases:
    """Consistency rules over pairs discovered from real extracted facts."""

    def test_same_document_pairs_never_compared(self, comparator, corpus):
        for doc, facts in corpus.items():
            if len(facts) < 2:
                continue
            passed = 0
            for f1, f2 in zip(facts, facts[1:]):
                if comparator._are_candidates(
                    comparator._to_dict(f1),
                    comparator._to_dict(f2),
                ):
                    passed += 1
            assert passed == 0, (
                f"same-document pairs must be excluded from comparison, "
                f"but {passed} pairs in {doc.name} passed the candidate filter"
            )

    def test_corroboration_same_value_pairs_agree(self, comparator, corpus):
        """
        Cross-document pairs whose extracted numeric values are equal are,
        by construction, the same underlying claim.  The classifier must
        never call such a pair a contradiction.
        """
        pairs = _cross_doc_pairs(corpus)
        equal_value = [
            (f1, f2) for f1, f2 in pairs
            if f1.value is not None and values_close(f1.value, f2.value)
        ]
        if not equal_value:
            pytest.skip("corpus has no same-value cross-document fact pairs")

        targets = sample(equal_value, limit=10)
        calls = []
        for f1, f2 in targets:
            if comparator._are_candidates(
                comparator._to_dict(f1),
                comparator._to_dict(f2),
            ):
                comp = comparator._compare_pair_with_llm(
                    comparator._to_dict(f1),
                    comparator._to_dict(f2),
                )
                if comp:
                    calls.append(comp)

        if not calls:
            pytest.skip("no same-value pairs survived the candidate filter")

        rels = [c.relationship for c in calls]
        n = len(rels)
        n_contra = rels.count("contradicts")
        n_ok = rels.count("corroborates") + rels.count("reconciled")
        assert n_contra < n / 2, (
            f"contradictions are the majority ({n_contra}/{n}) in the "
            f"same-value bucket — this signals broken classification: {rels}"
        )
        assert n_ok > n / 2, (
            f"corroborates|reconciled must be the majority in the same-value "
            f"bucket but got {n_ok}/{n}: {rels}"
        )
        assert all(r in ALLOWED_RELATIONSHIPS for r in rels), rels
        print(f"\ncorroboration bucket ({n} pairs): {rels}")

    def test_contradiction_same_unit_different_value_pairs_flagged(self, comparator, corpus):
        """
        Cross-document facts on the same unit but meaningfully different
        numeric values describe a discrepancy the classifier must not
        paper over as corroboration.
        """
        pairs = _cross_doc_pairs(corpus)

        def calls_real_pair(f1, f2):
            return not values_close(f1.value, f2.value) and same_unit(f1.unit, f2.unit)

        flagged = [
            (f1, f2) for f1, f2 in pairs
            if f1.value is not None
            and f2.value is not None
            and calls_real_pair(f1, f2)
        ]
        if not flagged:
            pytest.skip("corpus has no same-unit different-value cross-document pairs")

        targets = sample(flagged, limit=10)
        calls = []
        for f1, f2 in targets:
            if comparator._are_candidates(
                comparator._to_dict(f1),
                comparator._to_dict(f2),
            ):
                comp = comparator._compare_pair_with_llm(
                    comparator._to_dict(f1),
                    comparator._to_dict(f2),
                )
                if comp:
                    calls.append(comp)

        if not calls:
            pytest.skip("no same-unit different-value pairs survived the candidate filter")

        rels = [c.relationship for c in calls]
        n = len(rels)
        n_corroborates = rels.count("corroborates")
        n_ok = n - n_corroborates
        assert n_corroborates < n / 2, (
            f"corroborates is the majority ({n_corroborates}/{n}) in the "
            f"same-unit different-value bucket — this signals broken "
            f"classification: {rels}"
        )
        assert n_ok > n / 2, (
            f"contradicts|reconciled must be the majority in the "
            f"same-unit different-value bucket but got {n_ok}/{n}: {rels}"
        )
        assert all(r in ALLOWED_RELATIONSHIPS for r in rels), rels
        print(f"\ncontradiction bucket ({n} pairs): {rels}")

    def test_reconciled_labels_preserved_and_valid(self, comparator, corpus):
        """
        Whenever the classifier produces a "reconciled" verdict it must be
        preserved in the output with a valid explanation (not dropped and
        not relabelled).
        """
        pairs = _cross_doc_pairs(corpus)
        if not pairs:
            pytest.skip("corpus has no cross-document pairs")

        targets = sample(pairs, limit=25)
        all_results = []
        for f1, f2 in targets:
            if comparator._are_candidates(
                comparator._to_dict(f1),
                comparator._to_dict(f2),
            ):
                comp = comparator._compare_pair_with_llm(
                    comparator._to_dict(f1),
                    comparator._to_dict(f2),
                )
                if comp:
                    all_results.append(comp)

        reconciled = [c for c in all_results if c.relationship == "reconciled"]
        for c in reconciled:
            assert c.explanation and len(c.explanation) >= 10, (
                f"reconciled verdict missing explanation: {c}"
            )
            assert 0.0 <= c.confidence <= 1.0, f"confidence out of range: {c.confidence}"

        if reconciled:
            print(f"\nreconciled verdicts found: {len(reconciled)}")
            for c in reconciled:
                print(f"  id1={c.fact1_id} id2={c.fact2_id}: {c.explanation[:80]}")

    def test_unrelated_pairs_never_reach_llm(self, comparator, extractor, corpus):
        """
        Truly unrelated facts must be rejected by the cheap candidate filter
        before the LLM is ever called.
        """
        class CountingLLM:
            def __init__(self, inner):
                self.inner = inner
                self.calls = 0

            def generate_json(self, *a, **k):
                self.calls += 1
                return self.inner.generate_json(*a, **k)

        wrapped = CountingLLM(comparator.llm)
        probe = FactComparator(llm_client=wrapped)

        pairs = _cross_doc_pairs(corpus)
        rejected = [
            (f1, f2) for f1, f2 in pairs
            if not comparator._are_candidates(
                comparator._to_dict(f1),
                comparator._to_dict(f2),
            )
        ]
        if not rejected:
            pytest.skip("no unrelated pairs discovered in this corpus")

        calls_before = wrapped.calls
        for f1, f2 in sample(rejected, limit=10):
            probe.compare_facts([f1, f2])
        assert wrapped.calls == calls_before, (
            "unrelated pairs reached the LLM — candidate filter is broken"
        )

    def test_unrelated_verdicts_filtered_from_results(self, comparator, corpus):
        """
        Even when pairs slip past the candidate filter, an "unrelated"
        verdict from the LLM must be dropped from the returned results.
        """
        nonempty = [f for f in corpus.values() if f]
        if not nonempty:
            pytest.skip("corpus extracted no facts")

        spread = []
        cap = max(2, 15 // len(nonempty))
        for facts in nonempty:
            spread.extend(sample(facts, limit=cap))
        if len(spread) < 2:
            pytest.skip("not enough facts to compare")

        all_results: List[FactComparison] = comparator.compare_facts(spread)
        rels = [c.relationship for c in all_results]
        assert "unrelated" not in rels, (
            "'unrelated' verdicts must be filtered from results, got: "
            + str([c.relationship for c in all_results])
        )
        for c in all_results:
            assert c.relationship in ALLOWED_RELATIONSHIPS, c.relationship
            assert c.explanation, f"missing explanation on {c.relationship}"
            assert 0.0 <= c.confidence <= 1.0, c.confidence
        print(f"\nfull cross-document comparison: {dict(map(lambda r: (r, rels.count(r)), set(rels)))}")


# ---------------------------------------------------------------------------
# Standalone CLI — full-pipeline report on any corpus
# ---------------------------------------------------------------------------

def _run_report(directory: Path):
    extractor = PDFExtractor(llm_client=LLMClient())
    comparator = FactComparator(llm_client=LLMClient())

    pdfs = discover_pdfs(directory)
    if not pdfs:
        print(f"No PDFs found under {directory}")
        return

    corpus = {}
    for pdf in pdfs:
        facts = extractor.extract_facts(str(pdf), source_document=pdf.name)
        corpus[pdf] = facts
        print(f"{pdf.name}: {len(facts)} facts")

    all_facts = [f for facts in corpus.values() for f in facts]
    print(f"\nTotal facts: {len(all_facts)}")

    comparisons = comparator.compare_facts(all_facts)
    counts = {}
    for c in comparisons:
        counts[c.relationship] = counts.get(c.relationship, 0) + 1

    print("\nRelationship distribution:")
    for rel in ("corroborates", "contradicts", "reconciled"):
        print(f"  {rel:14s}: {counts.get(rel, 0)}")
    print(f"  unrelated     : filtered (0 in output)")

    for c in comparisons:
        print(f"\n  [{c.relationship}] {c.explanation[:120]}")
        print(f"      ids {c.fact1_id} vs {c.fact2_id}")


def main():
    parser = argparse.ArgumentParser(description="Document-agnostic extraction + comparison report")
    parser.add_argument("--dir", type=Path, default=DATA_DIR, help="directory of PDFs (recurses one level)")
    args = parser.parse_args()
    _run_report(args.dir)


if __name__ == "__main__":
    main()