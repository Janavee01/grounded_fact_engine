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
                └───────────────────┘
```

| Layer | File | Runtime | Covers |
|-------|------|---------|--------|
| Unit (offline) | `test_pipeline.py` | seconds, no network | grounding, candidate selection, magnitude words, full pipeline |
| Real PDF smoke | `test_pdf_real.py` | minutes | real PDF extraction smoke tests |
| End-to-end | `test_real.py` | slow, real LLM | all four required cases |

## 2. Unit Tests — `test_pipeline.py`

17 tests with **no network access**, so they run fast and offline. They exercise
the logic that doesn't need a live model:

- **Grounding** — quote matching, threshold behavior, failure returns empty
  evidence.
- **Candidate selection** — which fact pairs pass the pre-filter.
- **Magnitude words** — crore/lakh/million/billion scale handling.
- **Full pipeline** — extraction → grounding → fact building without a network.

## 3. Real PDF Smoke Tests — `test_pdf_real.py`

Run the actual extractor against real PDFs to confirm the pipeline produces
output end-to-end. Used to catch regressions in text extraction (prose, tables)
and OCR.

## 4. End-to-End Tests — `test_real.py`

Full pipeline with **real PDFs and a real LLM**, verifying the system produces
all four required outcomes:

```
test_real.py
   ├─ corroboration   ── India FY25 GDP growth (ES 6.4% vs IMF ~6.5%)
   ├─ contradiction   ── same metric, incompatible values
   ├─ reconciliation  ── ₹81,415M vs ₹8,142 crore explained via units
   └─ unrelated       ── unrelated pairs are filtered out
```

## 5. How to Run

```bash
# Unit tests only — fast, offline, no LLM required
pytest tests/test_pipeline.py

# Real end-to-end — requires LLM backend (Ollama or OpenRouter)
pytest tests/test_real.py -v -s
```

### Prerequisites for the real tests
- A running LLM backend (see [API.md](API.md) for setup).
- The starter PDFs present in `data/` (see
  [ARCHITECTURE.md](ARCHITECTURE.md)).

## 6. Test Coverage Map

| Commit | Test contribution |
|--------|-------------------|
| Initial implementation | Early `test_extraction.py` / `test_extraction_manual.py` |
| Test suite (`5d58f06`) | `test_pipeline.py` (17 unit), `test_real.py` (4 cases), `test_pdf_real.py` |
| OCR fallback | Unit test verifying OCR source tagging on built facts |

## 7. Conventions

- **Unit tests never touch the network** — deterministic and CI-friendly.
- **Real tests are opt-in and explicit** (`-s` for streaming output), since they
  take minutes and require a model.
- Tests exercise the **real extractor API** rather than mocking around it, so
  they also catch schema/interface changes.
