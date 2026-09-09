# API Reference — FastAPI Endpoints & UI Integration

Reference for the HTTP API surface (FastAPI) and how the Streamlit UI consumes
it. 

## 1. Service Topology

```
  ┌────────────┐   HTTP   ┌─────────────┐   ┌───────────────┐
  │ Streamlit  │ ───────► │   FastAPI   │──►│ Extraction    │
  │    UI      │ ◄─────── │  (:8000)    │   │ + Grounding   │
  └────────────┘          └──────┬──────┘   └───────┬───────┘
                                 │                  │
                                 ▼                  ▼
                          ┌─────────────┐   ┌───────────────┐
                          │ SQLite      │   │ JSON disk     │
                          │ (facts)     │   │ cache         │
                          └─────────────┘   └───────────────┘
```

- **FastAPI** serves the API on port `:8000` (launched via `run.py`).
- **Streamlit** (`app_ui.py`) serves the UI on `:8501` and calls the API.

## 2. Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload a PDF → extract + ground + store facts |
| GET | `/facts` | All facts with evidence + confidence |
| GET | `/facts/{doc}` | Facts for one document |
| GET | `/compare` | Cross-document relationships + explanations |
| GET | `/compare/progress` | Live progress for an active comparison run |
| GET | `/stats` | Counts per document / fact type |
| DELETE | `/facts` | Clear the knowledge base |

### POST `/upload`

Upload a PDF file; the server runs extraction + grounding and persists the
facts. Repeated uploads of the same PDF return instantly via the disk cache.

```bash
curl -s -X POST -F "file=@data/india-macroeconomy/01-....pdf" localhost:8000/upload
```

### GET `/facts`

```bash
curl -s localhost:8000/facts | python3 -m json.tool
```

Returns every stored fact with its verbatim evidence snippet and confidence.

### GET `/facts/{doc}`

Facts filtered to a single document name.

### GET `/compare`

```bash
curl -s localhost:8000/compare | python3 -m json.tool
```

Returns the cross-document relationships:
`corroborates` / `contradicts` / `reconciled`, each with an explanation,
confidence, and source snippets. `unrelated` pairs are filtered out.

### GET `/compare/progress`

```bash
curl -s localhost:8000/compare/progress
```

Returns the live progress of an in-flight comparison run:

```json
{ "running": true, "completed": 12, "total": 40, "stage": "Comparing pair 12/40", "error": null }
```

The Streamlit Compare tab polls this endpoint to render its progress bar.

### GET `/stats`

Counts grouped by document. Fact types currently counts two buckets
(`numeric` and `semantic`); the other enum types (`date`, `time`, `boolean`,
`entity`) are not yet broken out in this endpoint.

### DELETE `/facts`

Clears the entire knowledge base (drops stored facts).

```bash
curl -s -X DELETE localhost:8000/facts
```

## 3. Streamlit UI Integration

```
   Streamlit UI (app_ui.py)
   ├─ Sidebar: upload PDF ────► POST /upload ──► extraction
   ├─ Facts tab        ───────► GET /facts      (snippet + confidence)
   ├─ Compare tab      ───────► GET /compare + polls GET /compare/progress
   ├─ Stats tab        ───────► GET /stats      (counts)
   └─ Demo Cases tab   ───────► GET /compare    (live four required cases)
                                    │
        corroboration  contradiction  reconciliation  failure(summary)
```

### Demo Cases tab (live, not hardcoded)
The Demo Cases tab renders **real** results fetched from `GET /compare`.

- **Case 1 — Corroboration:** India FY25 GDP growth (ES 6.4% vs IMF ~6.5%).
- **Case 2 — Contradiction:** same metric/entity/period with differing values.
- **Case 3 — Reconciliation:** ₹81,415M vs ₹8,142 crore explained via units.
- **Case 4 — Extraction failure:** shows the grounding threshold and comparison
  count when pages fail grounding (e.g. imperfect OCR).

## 4. Setup

### Local Ollama (default)
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
ollama serve
ollama pull qwen3:8b
python run.py                 # FastAPI on :8000
streamlit run app_ui.py       # UI on :8501
```

### Hosted model (OpenRouter)
```bash
export LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-...
export OPENROUTER_MODEL=google/gemma-4-26b-a4b-it:free
python run.py
```

Runs are cached on disk, so repeat uploads of the same PDF are instant even
with a hosted model.