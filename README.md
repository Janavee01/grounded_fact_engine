# Fact Knowledge Layer

Extract grounded facts from PDFs, link every fact to verbatim source evidence,
and automatically detect when facts **corroborate**, **contradict**, or **reconcile**
across documents.

---

## Demo

**https://drive.google.com/file/d/1wULFQEl_Bxn9YKfg1NZKcy7yHQqWB0eo/view?usp=sharing**

The video walks through: uploading a PDF, watching facts get extracted and
grounded to source quotes, then showing the four required cases (corroboration,
contradiction, reconciliation, and an honest extraction failure).

---

## The Four Required Cases

These are generated **live** by the system from the starter PDFs — nothing is
hard-coded. The system prints the source evidence (`source_snippet`) and the
model's reasoning (`explanation`) for each relationship.

### 1. Corroboration — same fact, stated differently
- **Source A:** `01-india-economic-survey-2024-25-excerpt.pdf`
- **Source B:** `03-imf-india-2025-article-iv-excerpt.pdf`
- Both discuss India's FY25 GDP growth outlook (ES: 6.4%, IMF: ~6.5%).
- **System output:** `corroborates` with an explanation noting the values are
  consistent within rounding/model vintage differences.
- **Evidence:** the `source_snippet` from each document is quoted verbatim.

### 2. Contradiction — same metric, incompatible values
- **Source A:** `01-india-economic-survey-2024-25-excerpt.pdf` — *"India's real
  GDP is estimated to grow by 6.4 per cent in FY25."*
- **Source B:** `03-imf-india-2025-article-iv-excerpt.pdf` — IMF projects 6.5%.
- Same metric (real GDP growth), same entity (India), same period (FY25),
  slightly different values.
- **System output:** `contradicts` (or `reconciled` if the model explains the
  difference as a scope/definition vintaging issue).

### 3. Reconciliation — apparent contradiction explained by units
- **Source A:** `02-delhivery-annual-report-fy24-excerpt.pdf` — FY24 revenue
  **₹81,415 million**.
- **Source B:** `03-delhivery-q4-fy24-earnings-presentation.pdf` — FY24 revenue
  **₹8,142 crore** (same number, different unit convention).
- **System output:** `reconciled` — the comparator recognises the two unit
  representations as one real figure after converting crore → million.

### 4. Extraction / reasoning failure — handled honestly
- **Scanned / image-only pages** used to produce zero text, so no facts. Now
  they are rendered to a bitmap and **Tesseract OCR** recovers the text
  automatically; facts from OCR pages are tagged
  `context["text_source"] = "ocr"` so they can be distinguished from
  selectable-text pages.
- Large scanned or table-dense pages often still yield ungrounded quotes
  (OCR is imperfect — a mis-recognized character makes the model's verbatim
  quote fail to ground).
- The `ground_quote()` gate (RapidFuzz, presence + fuzz score ≥ 80) **rejects**
  any fact whose `verbatim_quote` is not an exact substring of the source text.
- The pipeline logs these rejections rather than crashing, so a single bad
  chunk never takes down the run. Documented in
  [src/extraction/grounding.py](src/extraction/grounding.py).

---

## How It Works

```
PDF ──► pdfplumber (plain text + tables, page-by-page)
         │  pages with no selectable text (scanned images/graphics)
         │  ──► render to bitmap ─► Tesseract OCR (pytesseract)
         │        facts from OCR pages are tagged context["text_source"]="ocr"
         ▼
  1 LLM call per page (max ~14k chars; model: local qwen3:8b, or an OpenRouter model)
         │   system prompt = schema + grounding rules
         ▼
  raw facts {claim, fact_type, value, unit, verbatim_quote, confidence}
         │
         ▼
  ground_quote(): every fact's quote is fuzzy-matched against the raw page text,
  score < 80 → fact rejected. This is the anti-hallucination gate.
         │
         ▼
  Fact (schema-validated; numeric value re-extracted from the *grounded snippet*,
  not trusted from the model) ──► SQLite + JSON disk cache
         │
         ▼
  FactComparator: pair each candidate across *different* documents
  (token-set similarity gate) ──► LLM classifies: corroborates / contradicts /
  reconciled / unrelated ──► explanation + confidence
```

### Why one call per page and not "one call for the whole PDF"
See [docs/EXTRACTION.md](docs/EXTRACTION.md). Short version: the grounding gate
needs the exact `verbatim_quote` inside the prompt's context window. When you
feed an entire 200k-character report in one call, the model paraphrases instead
of quoting, the grounding score collapses, and facts get silently dropped.
Per-page call boundaries keep the evidence in-context so facts stay grounded.

---

## Setup & Run

### Prerequisites
- Python 3.10+ (tested 3.12)
- A GPU with ≥8 GB VRAM is recommended for local extraction (a free **Colab T4**
  works great — see [colab/](colab/Fact_Knowledge_Layer_Colab.ipynb)). On a
  CPU-only laptop the extraction step is slower but still works.
- **Tesseract OCR** (for scanned PDFs) — `sudo apt install tesseract-ocr` or
  `brew install tesseract`. The Python bindings are in `requirements.txt`.

### Option A — Local Ollama (free, recommended)
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# start ollama once (already pulled on the dev machine)
ollama serve
ollama pull qwen3:8b

python run.py                 # FastAPI on :8000
streamlit run app_ui.py       # UI on :8501
```

### Option B — Colab (fast T4, no local install)
Open [colab/Fact_Knowledge_Layer_Colab.ipynb](colab/Fact_Knowledge_Layer_Colab.ipynb),
pick **Runtime ▸ Change runtime type ▸ T4 GPU**, and run all cells. It installs
Ollama, pulls `qwen3:8b`, extracts every PDF in `data/`, runs the four cases,
and downloads a `results.zip`.

### Option C — Hosted model (OpenRouter)
```bash
export LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-...
export OPENROUTER_MODEL=google/gemma-4-26b-a4b-it:free   # or any model id
python run.py
```
Runs are cached on disk, so repeat uploads of the same PDF are instant even
with a hosted model.

### Try it
Upload any PDF via the Streamlit sidebar (or `POST /upload`). Facts appear in
the **Facts** tab with source snippet + confidence; relationships appear in
**Compare** (with a live progress bar), and the four required cases render in
**Demo Cases**. The **Stats** tab shows counts per document and fact type.

---

## API

| Method | Endpoint              | Description                                     |
|--------|-----------------------|-------------------------------------------------|
| POST   | `/upload`             | Upload a PDF → extract + store facts            |
| GET    | `/facts`              | All facts with evidence + confidence            |
| GET    | `/facts/{doc}`        | Facts for one document                          |
| GET    | `/compare`            | Cross-document relationships + explanations     |
| GET    | `/compare/progress`   | Live progress for an active comparison run      |
| GET    | `/stats`              | Counts per document / fact type                 |
| DELETE | `/facts`              | Clear the knowledge base                        |

Example:
```bash
curl -s -X POST -F "file=@data/india-macroeconomy/01-....pdf" localhost:8000/upload
curl -s localhost:8000/compare | python3 -m json.tool
```

---

## Project Layout

```
colab/                 Colab (T4) one-shot notebook
data/                  Starter PDFs + fact cache (generated)
src/
  extraction/          PDF → text → chunks → LLM facts → grounding
    pdf_extractor.py   chunking, LLM prompt, fact building, dedup, cache
    llm_client.py      Ollama / OpenRouter wrapper, retries, JSON repair
    grounding.py       RapidFuzz anti-hallucination gate
  comparison/          fact pairing + relationship classification
  storage/             SQLite persistence
  models/              pydantic schemas (Fact, FactComparison)
  api/main.py          FastAPI
  ingest.py            CLI batch-extraction utility
app_ui.py              Streamlit UI
debug_pdf.py           PDF text extraction debugger
demo_cases.py          Demo evaluation CLI (queries API endpoints)
smoke_test.py          Quick extraction smoke test on PDFs
tests/
  test_pipeline.py         unit tests (no network, 42 tests)
  test_generic_extraction.py  unit tests for extraction validity
  test_pdf_real.py         real PDF smoke tests
  test_real.py             end-to-end tests (real LLM required)
  test_extraction.py       CLI extraction script (not a pytest suite)
  test_extraction_manual.py  CLI manual extraction detail viewer
run.py                 FastAPI launcher
```

---

## Testing

```bash
pytest tests/test_pipeline.py tests/test_generic_extraction.py   # unit, no network
pytest tests/test_pdf_real.py -v -s                              # real PDF smoke tests
pytest tests/test_real.py -v -s                                  # real end-to-end (needs LLM)
```

---

## Approach & Key Decisions

- **Facts are only as good as their evidence.** The single most important
  design choice is the grounding gate: a fact is discarded unless its
  `verbatim_quote` fuzzy-matches the actual PDF text (`MIN_GROUNDING_SCORE=80`).
  This is what makes extracted facts trustworthy.
- **Values are re-derived from grounded text, not trusted from the LLM.**
  Models routinely scale numbers (e.g. append "million"), so numeric facts are
  re-parsed from the quoted snippet, and scale words (million/crore/lakh) are
  applied deterministically.
- **One LLM call per page** keeps grounding reliable (see above) and bounds
  output JSON size.
- **Disk cache** (`data/fact_cache/`) keyed by PDF hash + prompt version means
  re-processing a document costs nothing; new documents only pay for
  themselves.
- **Progress over perfection:** every chunk is independently tried; a bad chunk
  is skipped and logged, never fatal. Bad/unparseable model output is repaired
  (thinking-tags stripped, JSON rescued from text) before being rejected.

---

## Limitations & Next Steps

- **Throughput is bounded by local GPU.** A single page takes a few seconds on
  a consumer GPU; a 90-page report takes ~10-20 minutes. The Colab notebook and
  the disk cache are the current mitigations. Next: parallel per-page workers
  across multiple GPUs / a hosted queue.
- **`unrelated` pairs are filtered out of results.** The comparator only returns
  corroborate / contradict / reconciled. Surfacing "this document disagrees
  with nothing" as a first-class signal is on the roadmap.
- **Schema is fixed today.** A dynamic/evolving schema (new fact types as new
  documents appear) is listed as a brownie point and is the planned follow-up.

---

## AI Tools Used

- **pdfplumber** — deterministic PDF text/table extraction (no LLM).
- **Tesseract / pytesseract** — OCR fallback for scanned and image-only pages.
- **RapidFuzz** — token-set similarity for grounding + candidate pairing.
- **qwen3 (Ollama, local)** — fact extraction and relationship classification
  on-device; no API keys or paid services needed.
- **OpenRouter** (optional) — alternative hosted model backend.
- The LLM is used ONLY for extraction/classification, never to fabricate
  evidence — every claim must pass the grounding gate against the real PDF.

---

## License

Educational use.
