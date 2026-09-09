# Architecture — Fact Knowledge Layer

## 1. Overview

The Fact Knowledge Layer extracts **grounded facts** from PDFs, links every fact
to verbatim source evidence, and automatically detects when facts across
documents **corroborate**, **contradict**, or **reconcile** — with an LLM-written
explanation for every relationship.

It is fully **document-agnostic**. It works with any PDF via a FastAPI + Streamlit UI.

## 2. System Diagram

```
                    ┌─────────────────────────────────────────────────────┐
                    │                   FACT KNOWLEDGE LAYER              │
                    │                                                     │
  ┌────────┐        │  ┌─────────────┐   ┌───────────┐   ┌─────────────┐  │
  │  PDF   │───────►│  │ EXTRACTION  │──►│ GROUNDING │──►│  STORAGE    │  │
  │(any doc)│       │  │  pipeline   │   │   gate    │   │  (SQLite)   │  │
  └────────┘        │  └─────────────┘   └───────────┘   └──────┬──────┘  │
  ┌────────┐        │        │                 ▲                 │        │
  │ Streamlit│      │        ▼                 │                 ▼        │
  │   UI   │◄───────┤  ┌─────────────┐        │          ┌─────────────┐  │
  └────────┘        │  │ COMPARISON  │◄───────┼──────────│   FACTS     │  │
  ┌────────┐        │  │ comparator  │        └──────────│(grounded +  │  │
  │FastAPI │◄───────┤  └─────────────┘                   │  evidence)  │  │
  └────────┘        │        ▲                           └─────────────┘  │
                    └────────┼────────────────────────────────────────────┘
                             │
                        ┌────┴────┐   ┌──────────┐
                        │ OLLAMA  │   │OpenRouter│  (LLM backends)
                        └─────────┘   └──────────┘
```

### Legend
| Element | Role |
|---------|------|
| **PDF** | Any input document (plain text, table-heavy, or scanned) |
| **Extraction** | PDF → page text → chunks → raw LLM facts |
| **Grounding** | Anti-hallucination gate; matches quote against source text |
| **Storage** | SQLite persistence + JSON disk cache |
| **Comparison** | Pairs facts across docs and classifies relationships |
| **FastAPI / Streamlit** | API and UI entry points |
| **Ollama / OpenRouter** | Local / hosted LLM backends |

## 3. Module Layout

```
colab/                 Colab (T4) one-shot notebook
notebooks/
  experiments/         scratch experiments (not part of the pipeline)
data/                  Starter PDFs + fact cache (generated)
src/
  extraction/
    pdf_extractor.py   chunking, LLM prompt, fact building, dedup, cache
    llm_client.py      Ollama / OpenRouter wrapper, retries, JSON repair
    grounding.py       RapidFuzz anti-hallucination gate
  comparison/
    comparator.py      fact pairing + relationship classification
  storage/
    database.py        SQLite persistence
  models/
    fact.py            pydantic schemas (Fact, FactComparison)
  api/
    main.py            FastAPI
  ingest.py            CLI batch-extraction utility
app_ui.py              Streamlit UI (Facts / Compare / Stats / Demo Cases)
debug_pdf.py           PDF text extraction debugger
demo_cases.py          Demo evaluation CLI (queries API endpoints)
smoke_test.py          Quick extraction smoke test on PDFs
run.py                 FastAPI launcher
tests/
  test_pipeline.py         unit tests (no network, 42 tests)
  test_generic_extraction.py  unit tests for extraction validity
  test_pdf_real.py         real PDF smoke tests
  test_real.py             end-to-end tests (real LLM required)
  test_extraction.py       CLI extraction script (not a pytest suite)
  test_extraction_manual.py  CLI manual extraction detail viewer
```

## 4. End-to-End Data Flow

```
PDF ──► pdfplumber (plain text + tables, page-by-page)
         │  pages with no selectable text (scanned images/graphics)
         │  ──► render to bitmap ─► Tesseract OCR (pytesseract)
         │        facts from OCR pages tagged context["text_source"]="ocr"
         ▼
   1 LLM call per page (max ~14k chars; local qwen3:8b or OpenRouter model)
         │   system prompt = schema + grounding rules
         ▼
   raw facts {claim, fact_type, value, unit, verbatim_quote, confidence}
         │
         ▼
   ground_quote(): every fact's quote fuzzy-matched against raw page text,
   score < 80 → fact rejected.  [the anti-hallucination gate]
         │
         ▼
   Fact (schema-validated; numeric value re-extracted from the grounded
   snippet, not trusted from the model) ──► SQLite + JSON disk cache
         │
         ▼
   FactComparator: pair candidates across *different* documents
   (token-set similarity gate) ──► LLM classifies: corroborates /
   contradicts / reconciled / unrelated ──► explanation + confidence
```

## 5. Core Design Decisions

- **Per-page LLM calls** (not one call per PDF). The grounding gate needs the
  exact `verbatim_quote` inside the prompt's context window. Feeding a 200k-char
  report in one call makes the model paraphrase, the grounding score collapses,
  and facts are silently dropped. Per-page chunks (~14k chars) keep evidence
  in-context. Full rationale in [EXTRACTION.md](EXTRACTION.md).

- **Facts are only as good as their evidence.** A fact is discarded unless its
  `verbatim_quote` fuzzy-matches the actual PDF text (`MIN_GROUNDING_SCORE=80`).

- **Values are re-derived from grounded text, not trusted from the LLM.**
  Models routinely re-scale numbers, so numeric facts are re-parsed from the
  quoted snippet with deterministic scale-word handling
  (crore=10⁷, lakh=10⁵, million=10⁶, billion=10⁹).

- **Disk cache** (`data/fact_cache/`) keyed by PDF hash + prompt version, so
  re-processing a document costs zero LLM calls.

- **Progress over perfection.** Every chunk is tried independently; a bad chunk
  is skipped and logged, never fatal. Unparseable model output is repaired
  (thinking-tags stripped, JSON rescued from text) before being rejected.

## 6. LLM Backends

```
                 ┌─────────────┐
   LLM_PROVIDER  │ Ollama      │  local, on-device, no API keys (default)
      ─────────► │ OpenRouter  │  hosted, needs OPENROUTER_API_KEY
                 └─────────────┘

   failure handling (llm_client.py):
     "think" tag strip  →  JSON rescue from text  →  retry w/ exponential
     backoff (respects Retry-After header)  →  clear error on final failure
```

## 7. Related Documents

| Document | Content |
|----------|---------|
| [EXTRACTION.md](EXTRACTION.md) | Extraction pipeline, grounding gate, OCR |
| [COMPARISON.md](COMPARISON.md) | Cross-document comparison engine |
| [API.md](API.md) | FastAPI endpoints + UI integration |
| [TESTING.md](TESTING.md) | Test strategy and how to run tests |