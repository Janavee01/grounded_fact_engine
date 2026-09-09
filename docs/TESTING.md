# Testing — Strategy & How to Run

How the project is tested: a layered strategy of fast offline unit tests plus
real end-to-end tests, covering extraction, grounding, comparison, and the four
required demo cases.

## 1. Test Pyramid

```
                ┌───────────────────┐
                │  END-TO-END       │   fewest, slowest, real LLM
                │  test_real        │   verifies the 4 required cases
                ├───────────────────┤
                │  REAL PDF smoke   │   real PDFs, no fake data
                │  test_pdf_real    │
                ├───────────────────┤
                │  UNIT (offline)   │   most, fastest, NO network
                │  test_pipeline    │
                │  test_generic_extraction │
                └───────────────────┘
```

| Layer | File | Runtime | Covers |
|-------|------|---------|--------|
| Unit (offline) | `test_pipeline.py` (42 tests) | seconds, no network | grounding, candidate selection, guardrails, magnitude handling, full pipeline |
| Unit (offline) | `test_generic_extraction.py` (23 tests) | seconds, no network | fact-type classes, meaningfulness filter, evidence grounding, value normalization, DB round-trip |
| Real PDF smoke | `test_pdf_real.py` | minutes, real PDFs | real PDF extraction smoke tests |
| End-to-end | `test_real.py` | slow, real LLM | all four required cases |

## 2. Unit Tests — `test_pipeline.py`

42 tests with **no network access**, so they run fast and offline. They exercise
the logic that doesn't need a live model:

- **Grounding** — quote matching, threshold behavior, failure returns empty
  evidence.
- **Candidate selection** — which fact pairs pass the pre-filter.
- **Numeric guardrails** — plan-vs-actual, cumulative/revised, period and scope
  differences, percentage-points, approximation, causality.
- **Magnitude words** — crore/lakh/million/billion scale handling.
- **Full pipeline** — extraction → grounding → fact building without a network.

## 3. Unit Tests — `test_generic_extraction.py`

23 offline tests focused on extraction validity:

- **Fact-type classification** — each `FactType` (numeric, semantic, date,
  time, boolean, entity) is classified correctly.
- **Meaningfulness filter** — boilerplate, IDs, credentials, and placeholders
  are rejected.
- **Evidence grounding** — ungrounded quotes and unsupported claims rejected;
  snippets stay concise.
- **Value normalization** — time/date/boolean scalars re-derived from the
  grounded snippet.
- **Error isolation** — one bad fact never kills the run.
- **DB round-trip** — all value types persist through SQLite.

## 4. Real PDF Smoke Tests — `test_pdf_real.py`

Run the actual extractor against real PDFs to confirm the pipeline produces
output end-to-end. Used to catch regressions in text extraction (prose, tables)
and OCR.

## 5. End-to-End Tests — `test_real.py`

Full pipeline with **real PDFs and a real LLM**, verifying the system produces
all four required outcomes:

```
test_real.py
   ├─ corroboration   ── India FY25 GDP growth (ES 6.4% vs IMF ~6.5%)
   ├─ contradiction   ── same metric, incompatible values
   ├─ reconciliation  ── ₹81,415M vs ₹8,142 crore explained via units
   └─ unrelated       ── unrelated pairs are filtered out
```

## 6. How to Run

```bash
# Unit tests only — fast, offline, no LLM required
pytest tests/test_pipeline.py tests/test_generic_extraction.py

# Real PDF smoke tests — real PDFs, no live LLM needed unless extraction asks for one
pytest tests/test_pdf_real.py -v -s

# Real end-to-end — requires LLM backend (Ollama or OpenRouter)
pytest tests/test_real.py -v -s
```

### Prerequisites for the real tests
- A running LLM backend (see [API.md](API.md) for setup).
- The starter PDFs present in `data/` (see
  [ARCHITECTURE.md](ARCHITECTURE.md)).

## 7. Test Coverage Map

| Commit | Test contribution |
|--------|-------------------|
| Initial implementation | Early `test_extraction.py` / `test_extraction_manual.py` |
| Test suite (`5d58f06`) | `test_pipeline.py` (17 unit), `test_real.py` (4 cases), `test_pdf_real.py` |
| Later hardening | `test_pipeline.py` grew to 42 tests; `test_generic_extraction.py` (23 tests) added |
| OCR fallback | Unit test verifying OCR source tagging on built facts |

## 8. Conventions

- **Unit tests never touch the network** — deterministic and CI-friendly.
- **Real tests are opt-in and explicit** (`-s` for streaming output), since they
  take minutes and require a model.
- Tests exercise the **real extractor API** rather than mocking around it, so
  they also catch schema/interface changes.
