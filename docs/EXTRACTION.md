# PDF Extraction — Extraction, Grounding & OCR

Deep-dive on how a PDF becomes a set of grounded facts: deterministic text
extraction, LLM fact generation, the anti-hallucination grounding gate, numeric
re-derivation, deduplication, caching, and the OCR fallback for scanned pages.

## 1. Pipeline Overview

```
            ┌────────────────────────────────────────────────────────┐
            │                    pdf_extractor.py                    │
            │                                                        │
 PDF ──────►│ 1. page-level text (pdfplumber)                        │
            │      ├─ prose                                          │
            │      └─ tables (bounding-box separated)                │
            │          └─ unruled sub-tables (line-less columns)     │
            │      └─ no text?  ──► OCR fallback (see §6)            │
            │                      │                                 │
            │ 2. chunk ~14k chars (≈ one page)                       │
            │      ──► 1 LLM call per page (parallel: ThreadPool)    │
            │                      │                                 │
            │ 3. raw facts {claim, type, value, unit, quote, conf}   │
            │                      │                                 │
            │ 4. GROUNDING GATE (see §4)                             │
            │      quote vs source text, score ≥ 80 else rejected    │
            │                      │                                 │
            │ 5. numeric value RE-DERIVED from grounded snippet      │
            │      scale words applied deterministically             │
            │                      │                                 │
            │ 6. dedup: exact + fuzzy (RapidFuzz ≥ 95%)              │
            │ 7. importance gate: reject standalone identifiers      │
            │ 8. support check: quote must support the claim         │
            │                      │                                 │
            │                      ▼                                 │
            │  Fact (schema-validated) ──► storage + DISK CACHE      │
            └────────────────────────────────────────────────────────┘
```

## 2. Why Per-Page Chunking

The grounding gate requires the model to emit a `verbatim_quote` that is a
character-exact (or near-exact) substring of the source it was shown. This only
works when the source text is in the prompt's context window.

- **One call per whole PDF** (e.g. a 200k-char report) → the model paraphrases
  instead of quoting exactly → grounding score collapses → facts silently
  dropped.
- **Tiny chunks (750 chars)** → extremely grounded but ~943 calls per report —
  hours on a free-tier model.
- **Sweet spot: ~14k chars ≈ one page.** Evidence stays in-context and call
  count drops ~10x vs 750-char chunks.

## 3. Table-Aware Extraction

Real reports are table-heavy. Tables are handled separately from prose:

- **Table regions** are separated from prose by bounding-box.
- **Unruled tables** (no visible borders) get **line-less column detection**
  so columns can still be parsed into facts.

Numeric facts from tables get their values re-derived deterministically (see
[§5](#5-numeric-value-handling)) — this keeps the Delhivery
`₹81,415M` vs `₹8,142 crore` reconciliation exact rather than eyeballed.

## 4. The Grounding Gate (Anti-Hallucination)

The most important design choice. A fact is discarded unless its
`verbatim_quote` fuzzy-matches the actual PDF text at score ≥ `MIN_GROUNDING_SCORE`
(80).

```
                    model emits verbatim_quote
                              │
                              ▼
              ┌───────────────────────────────┐
              │ 1. sentence-boundary expansion│  enrich evidence around match
              │    (match → full sentence)    │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ 2. whitespace-normalized match│  handle extra spaces/newlines
              │    (before fuzzy fallback)    │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ 3. numeric-only fallback      │  number verbatim but text
              │    (locate bare token)        │  paraphrased
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ 4. reject if nothing matched  │  NEVER return the model's own
              │    (return empty evidence)    │  quote as evidence
              └───────────────────────────────┘
```

### Behaviors
- Quotes are expanded to **full sentence boundaries** for richer evidence.
- **Whitespace-normalized matching** runs before the fuzzy fallback so extra
  spaces/newlines don't cause false rejections.
- **Numeric fallback** grounds on the bare, verbatim number when the model
  paraphrased the surrounding text.
- On total failure the gate **returns empty evidence** — it never passes the
  model's own output back as if it were real source text.
- `min_score` was raised from 50.0 to **80.0** for stricter grounding.

## 5. Numeric Value Handling

The LLM's `value` field is treated as **untrusted**. Models routinely re-scale
numbers (e.g. append "million" or round).

For `numeric` facts, the value is **re-parsed from the grounded snippet**, and
scale words are applied deterministically:

| Word | Factor |
|------|--------|
| crore | 10⁷ |
| lakh | 10⁵ |
| million | 10⁶ |
| billion | 10⁹ |

## 6. OCR Fallback (Scanned / Image-Only Pages)

Scanned or graphical pages have no selectable text. Without OCR they produce
zero facts — the original "honest failure" case.

### Flow

```
   pdfplumber page extraction
             │
             ▼
   ┌───────────────────────────────┐
   │ text found?  ──yes──► normal text path (§3–§5)
   └───────────────────────────────┘
             │ no (scanned / graphical)
             ▼
   render page to bitmap (300 DPI via pdfplumber .to_image)
             │
             ▼
   Tesseract OCR (pytesseract, lazy import)
             │
             ▼
   recovered text tagged source='ocr'
   facts carry context['text_source']='ocr'
```

### Details
- OCR runs only when pdfplumber yields **no text** on a page.
- Pages are rendered at **300 DPI** and processed by **Tesseract** via
  pytesseract.
- Recovered facts are tagged `context['text_source']='ocr'` so they can be
  distinguished from selectable-text pages in the UI.
- **Optional dependency:** pytesseract is imported lazily, so extraction still
  works (logging an honest skip) on machines without Tesseract.
- `CACHE_VERSION` was bumped to 3 because OCR changes what text is extracted.

**Verified:** a 5-page scanned PDF (0 selectable chars) yields ~2.3–2.8k
chars/page and 18 grounded facts through the full pipeline.

## 7. Deduplication & Validations

After grounding and value re-derivation, each fact passes:

1. **Fact dedup** — exact + fuzzy pass with RapidFuzz (≥ 95% similarity) to
   collapse near-duplicate facts.
2. **Importance gate** — rejects standalone identifiers (e.g. a bare number or
   label) that aren't meaningful facts.
3. **Claim–quote support check** — rejects facts where the quote doesn't
   actually support the claim.

## 8. Caching

- Disk cache keyed by `sha256(pdf_bytes)[:16]` + a signature of (model, chunk
  budget, grounding thresholds, prompt version).
- Re-processing the same PDF costs **zero** LLM calls.
- Changing the prompt and bumping the version invalidates the cache cleanly.

## 9. Parallelism

Chunk extraction runs through a `ThreadPoolExecutor`. Concurrency is deliberately
kept low (2–3) because the dominant cost is **token generation speed**, fixed by
the GPU + model size — not parallelism. On small GPUs extra threads just queue
onto the same device (and on free hosted tiers trigger `429`s).
